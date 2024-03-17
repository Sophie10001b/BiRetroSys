#pragma once
#include <MolHandler/include_head.h>

#define MATRXOPERATOR(NAME)\
    inline MatRX<NAME> operator +(const MatRX<NAME> &mat, const NAME &scalar){\
        MatRX<NAME> res(mat.rows(), mat.cols());\
        auto matPtr = mat.data();\
        auto resPtr = res.data();\
        for (int i=0; i < mat.size(); i++){*(resPtr + i) = *(matPtr + i) + scalar;}\
        return res;\
    }\
    \
    inline void operator +=(MatRX<NAME> &mat, const NAME &scalar){\
        auto matPtr = mat.data();\
        for (int i=0; i < mat.size(); i++){*(matPtr + i) += scalar;}\
    }\
    \
    inline MatRX<NAME> operator -(const MatRX<NAME> &mat, const NAME &scalar){\
        MatRX<NAME> res(mat.rows(), mat.cols());\
        auto matPtr = mat.data();\
        auto resPtr = res.data();\
        for (int i=0; i < mat.size(); i++){*(resPtr + i) = *(matPtr + i) - scalar;}\
        return res;\
    }\
    \
    inline void operator -=(MatRX<NAME> &mat, const NAME &scalar){\
        auto matPtr = mat.data();\
        for (int i=0; i < mat.size(); i++){*(matPtr + i) -= scalar;}\
    }\
    \

namespace MolHandler {
    constexpr int MAXK = 4;
    constexpr int MAXDEG = 9;
    constexpr int MAXPATH = 15;
    const std::vector<std::vector<int64_t>> DISTBLOCK = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8, 15}, {15, 2048}};

    const std::regex SMIREGEX(R"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])");

    struct inputData{
        MatRX<float> atomFeat;
        MatRX<float> bondFeat;
        std::vector<MatRX<int64_t>> kBondFeat;
        std::vector<MatRX<int64_t>> bondIdx;
        std::vector<MatRX<int64_t>> attnBondIdx;
        MatRX<int64_t> queryIdx;
        MatRX<int64_t> keyIdx;
        MatRX<int64_t> deg;
        MatRX<int64_t> dist;
        MatRX<int64_t> seqFeat;
        MatRX<int64_t> lTask;
        MatRX<int64_t> lClass;
        MatRX<int64_t> graphLength;
        MatRX<int64_t> seqLength;
        MatRX<int64_t> bondSplit;

        inputData(const int &num=0, const int &k=MAXK): lTask(MatRX<int64_t>(1, num)), lClass(MatRX<int64_t>(1, num)), graphLength(MatRX<int64_t>(1, num)), seqLength(MatRX<int64_t>(1, num)), bondSplit(MatRX<int64_t>(1, k)){
            kBondFeat = std::vector<MatRX<int64_t>>(k - 1);
            bondIdx = std::vector<MatRX<int64_t>>(k);
            attnBondIdx = std::vector<MatRX<int64_t>>(k);
        }
    };

    inline int64_t getDist(const int64_t dist){
        int64_t start, end, res;
        int64_t id = 0;
        int64_t maxDist = DISTBLOCK.size();
        for (auto blk : DISTBLOCK){
            start = blk[0];
            end = blk.size() > 1 ? blk[1] : blk[0];
            if (dist < start && id == 0){res = maxDist; break;}
            else if ((dist == start && dist == end) || (dist >= start && dist < end)){res = id; break;}
            else if (dist >= end && id == maxDist - 1){res = maxDist; break;}
            id++;
        }
        return res;
    }

    template <typename T1, typename T2>
    inline void onehotConvert(T1 *container, T2 *resContainer, std::vector<int> dimList){
        T1 val;
        for (int ptr=0, resPtr=0; ptr < dimList.size(); resPtr += dimList[ptr], ptr++){
            val = *(container + ptr);
            if (val >= 0){*(resContainer + resPtr + val) = 1;}
        }
    }

    MATRXOPERATOR(int)
    MATRXOPERATOR(float)
    MATRXOPERATOR(int64_t)

    // only for 2D Eigen::Matrix
    template <typename T>
    inline MatRX<T> concat(const MatRX<T> &mat1, const MatRX<T> &mat2, const int dim=0){
        if (dim == 0){
            MatRX<T> temp(mat1.rows() + mat2.rows(), mat1.cols());
            temp << mat1,
                    mat2;
            return temp;
        }
        else {
            MatRX<T> temp(mat1.rows(), mat1.cols() + mat2.cols());
            temp << mat1, mat2;
            return temp;
        }
    }

    //element-wise apply condition in matrix2, then replace the the element with val in matrix1
    template <typename Tmatrix, typename T>
    inline void matrixFilter(Tmatrix *matrix1, Tmatrix *matrix2, std::initializer_list<int> sizes, T val, std::function<bool(int)> condition){
        int num = 1;
        for (auto i : sizes){num *= i;}
        for (int i=0; i < num; i++){
            *(matrix1 + i) = condition(*(matrix2 + i)) ? val : *(matrix1 + i);
        }
    }
    //element-wise apply condition in matrix2, then replace the the element with val in matrix1
    template <typename Tmatrix, typename T>
    inline void matrixFilter(Tmatrix *matrix1, const Tmatrix *matrix2, int sizes, T val, const std::function<bool(int)> &condition){
        for (int i=0; i < sizes; i++){
            *(matrix1 + i) = condition(*(matrix2 + i)) ? val : *(matrix1 + i);
        }
    }
    //element-wise apply condition in matrix1, then replace the the element with val
    template <typename Tmatrix, typename T>
    inline void matrixFilter(Tmatrix *matrix1, std::initializer_list<int> sizes, T val, std::function<bool(int)> condition){
        int num = 1;
        for (auto i : sizes){num *= i;}
        for (int i=0; i < num; i++){
            *(matrix1 + i) = condition(*(matrix1 + i)) ? val : *(matrix1 + i);
        }
    }
    //element-wise apply condition in matrix1, then replace the the element with val
    template <typename Tmatrix, typename T>
    inline void matrixFilter(Tmatrix *matrix1, int sizes, T val, std::function<bool(int)> condition){
        for (int i=0; i < sizes; i++){
            *(matrix1 + i) = condition(*(matrix1 + i)) ? val : *(matrix1 + i);
        }
    }

    class molPreprocess{
        public:
        const int64_t maxDeg;
        const int64_t maxK;
        const int64_t maxPath;

        std::map<str, int64_t> vocab;
        molPreprocess(const int64_t maxDeg=MAXDEG, const int64_t maxK=MAXK, const int64_t maxPath=MAXPATH);

        molPreprocess(const str &vdir, const int64_t maxDeg=MAXDEG, const int64_t maxK=MAXK, const int64_t maxPath=MAXPATH);
        
        //Read vocabulary from vocabulary.txt
        std::map<str, int64_t> readVocabulary(
            const str &vdir,
            const std::initializer_list<str> &extraToken = {"<BOS>", "<EOS>", "<PAD>", "<UNK>"}
        );

        //Generate canonicalize SMILES
        std::tuple<str, bool> canonicalizeSmiles(const str &smi, const bool heavyAtomCheck=false);

        //Generate sequence with vocabulary
        MatRX<int64_t> generateSeq(const str &smi, std::map<str, int64_t> *vocab=nullptr);

        //Generate graph in batches
        void generateGraph(
            const str &smi, inputData &inData,
            int64_t &preCumLength, int64_t &preLength, int64_t &batchId,
            const int lTask=0, const int lClass=0, const int lRoot=0, const int lShuffle=0
        );

        //generate Q, K, V Index in global attention
        void getQKVIndex(
            MatRX<int64_t> &qIdx, MatRX<int64_t> &kvIdx,
            const int64_t &preLength, const int64_t &curLength
        );

        //concat bond for generating batch
        void getBatchBond(
            MatRX<float> &bondFeat, MatRX<float> &curBondFeat,
            std::vector<MatRX<int64_t>> &kBondFeat, std::vector<MatRX<int64_t>> &curkBondFeat,
            std::vector<MatRX<int64_t>> &bondIdx, std::vector<MatRX<int64_t>> &curBondIdx,
            std::vector<MatRX<int64_t>> &attnBondIdx,
            const int64_t &preCumLength, const int64_t &preLength, const int64_t &curLength
        );

        //padding sequence
        void getPaddingSeq(
            std::vector<MatRX<int64_t>> &unpadSeq, MatRX<int64_t> &seqFeat,
            MatRX<int64_t> &seqLength, int64_t maxSeqLength,
            bool needBos=false, std::map<str, int64_t> *vocab=nullptr
        );

        //Generate a complete batch
        inputData generateBatch(
            const std::vector<str> &smis, const std::vector<int64_t> &lTasks,
            const bool needSeq=false
        );

        //get graph attention bias
        std::tuple<MatRX<int64_t>, MatRX<int64_t>> getAttentionBias(const RDKit::ROMol &mol);

        //get K-hop bond features
        std::tuple<std::vector<MatRX<int64_t>>, std::vector<MatRX<int64_t>>> getKhopFeat(const RDKit::ROMol &mol);
    };
}