#include <MolHandler/chem_utils.h>
#include <MolHandler/data_utils.h>

namespace MolHandler {
    molPreprocess::molPreprocess(const int64_t maxDeg, const int64_t maxK, const int64_t maxPath): maxDeg(maxDeg), maxK(maxK), maxPath(maxPath){}

    molPreprocess::molPreprocess(const str &vdir, const int64_t maxDeg, const int64_t maxK, const int64_t maxPath): maxDeg(maxDeg), maxK(maxK), maxPath(maxPath){this->vocab = this->readVocabulary(vdir);}

    //Read vocabulary from vocabulary.txt
    std::map<str, int64_t> molPreprocess::readVocabulary(
        const str &vdir,
        const std::initializer_list<str> &extraToken
    ){
        std::map<str, int64_t> vocab;
        std::ifstream fin(vdir);
        str line, token;
        int64_t count = 0;
        while (getline(fin, line)){
            token = strtok(line.data(), "\t");
            vocab.insert(std::make_pair(token, count));
            count++;
        }
        fin.close();
        for (auto token : extraToken){vocab.insert(std::make_pair(token, vocab.size()));}
        return vocab;
    }

    //Generate canonicalize SMILES
    inline std::tuple<str, bool> molPreprocess::canonicalizeSmiles(const str &smi, const bool heavyAtomCheck){
        RDKit::RWMol *mol1 = RDKit::SmilesToMol(smi, 0, false);
        str cano_smi = "";
        bool is_valid = true;
        try {
            if (mol1 == nullptr){throw 1;}
            else if (heavyAtomCheck && mol1->getNumHeavyAtoms() < 2){throw 2;}
            else {
                for (auto atom : mol1->atoms()){atom->clearProp("molAtomMapNumber");}
                cano_smi = RDKit::MolToSmiles(*mol1);
            }
        } catch (const int &warnId){
            switch (warnId){
                case 1: {
                    std::cout << "Input molecule \"" + smi + "\" is Invalid !" << std::endl;
                    cano_smi = smi;
                    is_valid = false;
                    break;
                }
                case 2: {
                    std::cout << "Input molecule \"" + smi + "\" is too Small (HeavyAtoms < 2) !" << std::endl;
                    cano_smi = "CC";
                    break;
                }
            }
        }
        return std::make_tuple(cano_smi, is_valid);
    }

    //Generate sequence with vocabulary
    inline MatRX<int64_t> molPreprocess::generateSeq(const str &smi, std::map<str, int64_t> *vocab){
        if (vocab == nullptr){vocab = &(this->vocab);}
        std::sregex_iterator tokenIter(smi.begin(), smi.end(), SMIREGEX);
        auto iterEnd = std::sregex_iterator();
        MatRX<int64_t> tokens(1, std::distance(tokenIter, iterEnd));

        int64_t unkId = (*vocab).find("<UNK>")->second;
        for (int i=0; tokenIter != iterEnd; tokenIter++, i++){
            tokens(0, i) = mapGet((*vocab), tokenIter->str(), unkId);
        }
        return tokens;
    }

    inline void molPreprocess::generateGraph(
        const str &smi, inputData &inData,
        int64_t &preCumLength, int64_t &preLength, int64_t &batchId,
        const int lTask, const int lClass, const int lRoot, const int lShuffle
    ){
        auto mol = *(RDKit::SmilesToMol(smi));
        auto ringInfo = *(mol.getRingInfo());
        int64_t atomNum = mol.getNumAtoms();
        int64_t bondNum = mol.getNumBonds();
        MatRX<int> atomLabel = MatRX<int>::Constant(atomNum, ATOMFEATNUM, -1);
        MatRX<int> bondLabel = MatRX<int>::Constant(bondNum * 2, BONDFEATNUM, -1);
        MatRX<float> atomFeat = MatRX<float>::Zero(atomNum, std::accumulate(ATOMFEATDIM.begin(), ATOMFEATDIM.end(), 0));
        MatRX<float> bondFeat = MatRX<float>::Zero(bondNum * 2, std::accumulate(BONDFEATDIM.begin(), BONDFEATDIM.end(), 0));
        MatRX<int64_t> bondIdx(2, bondNum * 2);

        int64_t start, end;
        int rbI = 0;
        for (auto atom : mol.atoms()){
            getAtomFeat(*atom, atomLabel.row(rbI).data(), lTask, lClass, lRoot, lShuffle);
            onehotConvert(atomLabel.row(rbI).data(), atomFeat.row(rbI).data(), ATOMFEATDIM);
            rbI++;
        }
        rbI = 0;
        for (auto bond : mol.bonds()){
            start = bond->getBeginAtomIdx();
            end = bond->getEndAtomIdx();
            bondIdx(0, rbI) = start;
            bondIdx(1, rbI) = end;
            bondIdx(0, rbI + 1) = end;
            bondIdx(1, rbI + 1) = start;
            getBondFeat(*bond, ringInfo, bondLabel.row(rbI).data());
            // std::copy(bondLabel.row(rbI).data(), bondLabel.row(rbI).data() + bondLabel.cols(), bondLabel.row(rbI + 1).data());
            onehotConvert(bondLabel.row(rbI).data(), bondFeat.row(rbI).data(), BONDFEATDIM);
            std::copy(bondFeat.row(rbI).data(), bondFeat.row(rbI).data() + bondFeat.cols(), bondFeat.row(rbI + 1).data());
            rbI += 2;
        }
        //sort
        std::vector<int64_t> sortScore(bondIdx.cols());
        for (int i=0; i <sortScore.size(); i++){sortScore[i] = bondIdx(0, i) * atomNum + bondIdx(1, i);}

        for (int i=0, srcCount=bondIdx.cols() - 1; i < srcCount; i++){
            for (int j=0; j < srcCount - i; j++){
                if (sortScore[j] > sortScore[j+1]){
                    bondIdx.col(j).swap(bondIdx.col(j+1));
                    bondFeat.row(j).swap(bondFeat.row(j+1));

                    std::swap(sortScore[j], sortScore[j+1]);
                }
            }
        }
        // std::cout << atomLabel << std::endl;
        // std::cout << "atomLen:" << atomLabel.rows() << std::endl;
        // std::cout << bondFeat << std::endl;
        // std::cout << "bondLen" << bondFeat.rows() << std::endl;
        auto [degMatrix, distMatrix] = getAttentionBias(mol);
        auto [kBondIdx, kBondFeat] = getKhopFeat(mol);
        kBondIdx.insert(kBondIdx.begin(), bondIdx);

        //generate a batch
        inData.atomFeat = inData.atomFeat.size() > 0 ? concat(inData.atomFeat, atomFeat) : atomFeat;
        inData.deg = inData.deg.size() > 0 ? concat(inData.deg, degMatrix, -1) : degMatrix;
        inData.dist = inData.dist.size() > 0 ? concat(inData.dist, distMatrix, -1) : distMatrix;
        this->getQKVIndex(inData.queryIdx, inData.keyIdx, preLength, atomNum);
        this->getBatchBond(inData.bondFeat, bondFeat, inData.kBondFeat, kBondFeat, inData.bondIdx, kBondIdx, inData.attnBondIdx, preCumLength, preLength, atomNum);
        inData.graphLength(0, batchId) = atomNum;
        preCumLength += (atomNum * atomNum);
        preLength += atomNum;
    }

    //generate Q, K, V Index in global attention
    inline void molPreprocess::getQKVIndex(
        MatRX<int64_t> &qIdx, MatRX<int64_t> &kvIdx,
        const int64_t &preLength, const int64_t &curLength
    ){
        MatRX<int64_t> temp(1, curLength);
        for (int64_t i=0; i < curLength; i++){temp(0, i) = preLength + i;}
        auto tempKV = temp.replicate(1, curLength);
        MatRX<int64_t> tempQ = temp.replicate(curLength, 1).reshaped(1, curLength * curLength); // reshaped is column major
        qIdx = qIdx.size() > 0 ? concat<int64_t>(qIdx, tempQ, -1) : tempQ;
        kvIdx = kvIdx.size() > 0 ? concat<int64_t>(kvIdx, tempKV, -1) : tempKV;
    }

    //concat bond for generating batch
    inline void molPreprocess::getBatchBond(
        MatRX<float> &bondFeat, MatRX<float> &curBondFeat,
        std::vector<MatRX<int64_t>> &kBondFeat, std::vector<MatRX<int64_t>> &curkBondFeat,
        std::vector<MatRX<int64_t>> &bondIdx, std::vector<MatRX<int64_t>> &curBondIdx,
        std::vector<MatRX<int64_t>> &attnBondIdx,
        const int64_t &preCumLength, const int64_t &preLength, const int64_t &curLength
    ){
        bondFeat = bondFeat.size() > 0 ? concat(bondFeat, curBondFeat) : curBondFeat;
        for (int i=0; i < this->maxK; i++){
            auto &idx = curBondIdx[i];
            if (idx.size() > 0){
                idx.row(0) = idx.row(0) + preLength;
                idx.row(1) = idx.row(1) + preLength;
                auto attnIdx = (idx.row(0) - preLength) * curLength + (idx.row(1) - preLength) + preCumLength;
                bondIdx[i] = bondIdx[i].size() > 0 ? concat(bondIdx[i], idx, -1) : idx;
                attnBondIdx[i] = attnBondIdx[i].size() > 0 ? concat(attnBondIdx[i], attnIdx, -1) : attnIdx;
                if (i < this->maxK - 1){
                    auto &feat = curkBondFeat[i];
                    kBondFeat[i] = kBondFeat[i].size() > 0 ? concat(kBondFeat[i], feat) : feat;
                }
                
            }
        }
    }

    //padding sequence
    void molPreprocess::getPaddingSeq(
        std::vector<MatRX<int64_t>> &unpadSeq, MatRX<int64_t> &seqFeat,
        MatRX<int64_t> &seqLength, int64_t maxSeqLength,
        bool needBos, std::map<str, int64_t> *vocab
    ){
        if (vocab == nullptr){vocab = &(this->vocab);}
        auto eosId = (*vocab).find("<EOS>")->second;
        auto padId = (*vocab).find("<PAD>")->second;

        int64_t bosLength = needBos ? 1 : 0;
        auto bosId = needBos ? (*vocab).find("<BOS>")->second : 0;

        maxSeqLength += (1 + bosLength); //BOS and EOS
        seqFeat = MatRX<int64_t>(unpadSeq.size(), maxSeqLength);
        seqFeat.fill(padId);
        for (int64_t batchId=0; batchId < unpadSeq.size(); batchId++){
            int curLength = seqLength(0, batchId);
            seqFeat(batchId, 0) = bosId;
            seqFeat.block(batchId, bosLength, 1, curLength) = unpadSeq[batchId];
            seqFeat(batchId, bosLength + curLength) = eosId;
            seqLength(0, batchId) += (1 + bosLength);
        }
    }

    inputData molPreprocess::generateBatch(
        const std::vector<str> &smis, const std::vector<int64_t> &lTasks,
        const bool needSeq
    ){
        int64_t preCumLength = 0;
        int64_t preLength = 0;
        int64_t maxSeqLength = 0;
        inputData inData(smis.size());
        std::vector<MatRX<int64_t>> unpadSeq(smis.size());
        for (int64_t batchId=0; batchId < smis.size(); batchId++){
            auto [canoSmi, isValid] = this->canonicalizeSmiles(smis[batchId]);
            canoSmi = isValid ? canoSmi : "CC";
            this->generateGraph(canoSmi, inData, preCumLength, preLength, batchId, lTasks[batchId], -1);
            if (needSeq){
                unpadSeq[batchId] = this->generateSeq(canoSmi);
                maxSeqLength = maxSeqLength > unpadSeq[batchId].cols() ? maxSeqLength : unpadSeq[batchId].cols();
                inData.seqLength(0, batchId) = unpadSeq[batchId].cols();
            }
        }
        inData.bondSplit(0, 0) = inData.bondFeat.rows();
        for (int i=1; i < this->maxK; i++){inData.bondSplit(0, i) = inData.kBondFeat[i-1].rows();}
        if (needSeq){
            this->getPaddingSeq(unpadSeq, inData.seqFeat, inData.seqLength, maxSeqLength, true);
        }
        // inData.lTask = Eigen::Map<MatRX<int64_t>>(lTasks.data(), 1, smis.size());
        for (int i=0; i < inData.lTask.cols(); i++){inData.lTask(0, i) = lTasks[i];}
        inData.lClass = MatRX<int64_t>::Zero(1, smis.size());
        return inData;
    }

    inline std::tuple<MatRX<int64_t>, MatRX<int64_t>> molPreprocess::getAttentionBias(const RDKit::ROMol &mol){
        int atomNum = mol.getNumAtoms();
        auto adjMatrix = RDKit::MolOps::getAdjacencyMatrix(mol);
        auto spdMatrix = RDKit::MolOps::getDistanceMat(mol);

        MatRX<int64_t> degMatrix(1, atomNum);
        degMatrix.fill(0);
        MatRX<int64_t> distMatrix(1, atomNum * atomNum);
        distMatrix.fill(-1);

        for (int i=0; i < atomNum; i++){
            for (int j=0; j < atomNum; j++){
                degMatrix(0, i) += *(adjMatrix + i * atomNum + j);
                distMatrix(0, i * atomNum + j) = getDist(*(spdMatrix + i * atomNum + j));

            }
            degMatrix(0, i) = degMatrix(0, i) > this->maxDeg ? maxDeg : degMatrix(0, i);
        }
        return std::make_tuple(degMatrix, distMatrix);
    }

    inline std::tuple<std::vector<MatRX<int64_t>>, std::vector<MatRX<int64_t>>> molPreprocess::getKhopFeat(const RDKit::ROMol &mol){
        int atomNum = mol.getNumAtoms();
        std::vector<MatRX<double>> kAdjMatrix;
        std::vector<MatRX<int64_t>> kBondIdx;
        std::vector<MatRX<int64_t>> kBondFeat;

        Eigen::Map<MatRX<double>> adjMatrix(RDKit::MolOps::getAdjacencyMatrix(mol), atomNum, atomNum);

        kAdjMatrix.push_back(adjMatrix);
        for (int i=0; i < this->maxK - 1; i++){kAdjMatrix.push_back(kAdjMatrix.back() * adjMatrix);}
        for (auto &matrix : kAdjMatrix){
            for (int i=0; i < atomNum; i++){matrix(i, i) = 0;}
        }
        
        auto lastPath = kAdjMatrix[0];
        for (int i=1; i < this->maxK; i++){
            matrixFilter(kAdjMatrix[i].data(), lastPath.data(), kAdjMatrix[i].size(), 0, [](const int &val){return val > 0;});
            lastPath += kAdjMatrix[i];
        }

        for (int i=1; i < this->maxK; i++){
            auto &iAdj = kAdjMatrix[i];
            MatRX<int64_t> iBondFeat(atomNum * atomNum, 1);
            MatRX<int64_t> iBondIdx(2, atomNum * atomNum);
            int count=0;
            for (int64_t row=0; row < atomNum; row++){
                for (int64_t col=0; col < atomNum; col++){
                    if (iAdj(row, col) > 0){
                        iBondIdx(0, count) = row;
                        iBondIdx(1, count) = col;
                        iBondFeat(count, 0) = iAdj(row, col) > this->maxPath ? this->maxPath : iAdj(row, col);
                        count++;
                    }
                }
            }
            kBondIdx.push_back(iBondIdx.leftCols(count));
            kBondFeat.push_back(iBondFeat.topRows(count));
        }
        return std::make_tuple(kBondIdx, kBondFeat);
    }
}

// int main(){
//     std::vector<str> smis = {"C=CCc1cccc([N+](=O)[O-])c1OCC#CC", "CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1"};
//     std::vector<int> ltask = {0, 0};
//     molPreprocess Solver("/Users/sophie/Code/BiRetroSys/Preprocess/vocabulary(uspto_full).txt");
//     auto res = Solver.generateBatch(smis, ltask, true);
//     int i = 0;
// }