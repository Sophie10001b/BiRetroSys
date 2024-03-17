#include <Inference/model_utils.h>
#include <Inference/tensor_utils.h>

namespace Inference {
    SeqAGraphInfer::SeqAGraphInfer(
        const modelClass &modelSelect, const str &device,
        const initlist<str> &extraToken
    ){
        str curPath = std::filesystem::current_path().parent_path();
        curPath = curPath + "/Models/" + (modelSelect == uspto50k ? "50k/" : "full/");

        const str vocabDir = curPath + "vocabulary" + (modelSelect == uspto50k ? "(uspto_50k).txt" : "(uspto_full).txt");
        const std::vector<str> modelDir = {curPath+"encoder.onnx", curPath+"extra_embedding.onnx", curPath+"decoder.onnx"};

        //load vocab
        std::ifstream fin(vocabDir);
        str line, token;
        int64_t count = 0;
        while (getline(fin, line)){
            token = strtok(line.data(), "\t");
            this->vocab.insert(std::make_pair(token, count));
            this->rvocab.insert(std::make_pair(count, token));
            count++;
        }
        fin.close();
        for (auto t : extraToken){
            this->vocab.insert(std::make_pair(t, this->vocab.size()));
            this->rvocab.insert(std::make_pair(this->rvocab.size(), t));
        }

        //load preprocessor
        this->molHandler.vocab = this->vocab;

        //load model
        if (device == "cuda"){
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            this->sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        Encoder = new Ort::Session(this->env, modelDir[0].c_str(), this->sessionOption);
        ExtraEmbedding = new Ort::Session(this->env, modelDir[1].c_str(), this->sessionOption);
        Decoder = new Ort::Session(this->env, modelDir[2].c_str(), this->sessionOption);
    }

    SeqAGraphInfer::~SeqAGraphInfer(){
        delete Encoder;
        delete ExtraEmbedding;
        delete Decoder;
    }

    std::vector<Ort::Value> SeqAGraphInfer::encoderRun(MolHandler::inputData &mol){
        std::vector<Ort::Value> inputs;
        const std::vector<const char*> inputsName = {"atomFeat", "queryIdx", "keyIdx", "deg", "dist", "bondFeat0", "bondFeat1", "bondFeat2", "bondFeat3", "bondIdx0", "bondIdx1", "bondIdx2", "bondIdx3", "attnBondIdx0", "attnBondIdx1", "attnBondIdx2", "attnBondIdx3", "bondSplit"};
        const std::vector<const char*> outputsName = {"graphOutput"};

        inputs.push_back(std::move(convertTensor<float, float>(mol.atomFeat, this->memInfo, {mol.atomFeat.rows(), mol.atomFeat.cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.queryIdx, this->memInfo)));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.keyIdx, this->memInfo)));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.deg, this->memInfo)));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.dist, this->memInfo)));
        inputs.push_back(std::move(convertTensor<float, float>(mol.bondFeat, this->memInfo, {mol.bondFeat.rows(), mol.bondFeat.cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.kBondFeat[0], this->memInfo, {mol.kBondFeat[0].rows()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.kBondFeat[1], this->memInfo, {mol.kBondFeat[1].rows()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.kBondFeat[2], this->memInfo, {mol.kBondFeat[2].rows()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.bondIdx[0], this->memInfo, {2, mol.bondIdx[0].cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.bondIdx[1], this->memInfo, {2, mol.bondIdx[1].cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.bondIdx[2], this->memInfo, {2, mol.bondIdx[2].cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.bondIdx[3], this->memInfo, {2, mol.bondIdx[3].cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.attnBondIdx[0], this->memInfo, {mol.attnBondIdx[0].cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.attnBondIdx[1], this->memInfo, {mol.attnBondIdx[1].cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.attnBondIdx[2], this->memInfo, {mol.attnBondIdx[2].cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.attnBondIdx[3], this->memInfo, {mol.attnBondIdx[3].cols()})));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.bondSplit, this->memInfo)));
        
        // for (int i=0; i < inputs.size(); i++){
        //     const auto data = inputs[i].GetTensorData<int64_t>();
        //     auto count = inputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
        //     for (int i=0; i < count; i++){
        //         std::cout << *(data + i) << "\t";
        //     }
        //     std::cout << "\n" << "--------------" << std::endl;
        // }

        return this->Encoder->Run(
            Ort::RunOptions{nullptr},
            inputsName.data(), 
            inputs.data(), inputs.size(),
            outputsName.data(), outputsName.size()
        );
    }

    std::vector<Ort::Value> SeqAGraphInfer::embeddingRun(MolHandler::inputData &mol){
        std::vector<Ort::Value> inputs;
        const std::vector<const char*> inputsName = {"Task", "Class"};
        const std::vector<const char*> outputsName = {"extraToken"};

        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.lTask, this->memInfo)));
        inputs.push_back(std::move(convertTensor<int64_t, int64_t>(mol.lClass, this->memInfo)));
        
        return this->ExtraEmbedding->Run(
            Ort::RunOptions{nullptr},
            inputsName.data(), 
            inputs.data(), inputs.size(),
            outputsName.data(), outputsName.size()
        );
    }

    std::tuple<std::vector<str>, std::vector<float>> SeqAGraphInfer::decoderRun(
        const Ort::Value &encRes, const Ort::Value &embRes, const MolHandler::inputData &mol,
        SearchMethods &mSearch
    ){
        const std::vector<const char*> inputsName = {"tokens", "extraTokenEmb", "msaCache", "mcaCache", "contextMask", "extraQ", "extraK", "numList", "step"};
        const std::vector<const char*> outputsName = {"tokenProb", "updatedMSACache"};

        auto graphEmb = graphPadding(
            encRes.GetTensorData<float>(),
            encRes.GetTensorTypeAndShapeInfo().GetShape(),
            mol.graphLength
        );
        auto mask = getMask(3, graphEmb[0].rows(), mol.graphLength);
        int64_t batchSize = graphEmb.size();
        int64_t dModel = graphEmb[0].cols();

        std::vector<int64_t> msaShape = {8, batchSize * mSearch.beamSize, 0, dModel};
        std::vector<int64_t> mcaShape = {batchSize, graphEmb[0].rows(), dModel};
        std::vector<int64_t> extraEmbShape = embRes.GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> maskShape = {batchSize, 1, mask[0].rows(), mask[0].cols()};
        std::vector<int64_t> taskCountShape = {batchSize};
        std::vector<int64_t> numListShape = {2};

        float *extraTokenEmb = nullptr;
        float *msaCache = nullptr;
        float *mcaCache = nullptr;
        bool *contextMask = nullptr;
        int64_t *taskCount = nullptr;
        std::vector<int64_t> numList;

        std::vector<int64_t> extraQShape = {1};
        std::vector<int64_t> extraKShape = {1};
        std::vector<int64_t> stepShape = {2};
        std::vector<int64_t> extraQ = {2};
        std::vector<int64_t> extraK = {2};
        std::vector<int64_t> step = {0, 0};
        std::vector<int64_t> inputTokenShape = {0, 1};

        for (int i=0, maxStep=mSearch.maxLength; i < maxStep; i++){
            std::vector<Ort::Value> inputs;
            auto inputToken = mSearch.currentToken();
            inputTokenShape[0] = inputToken.size();
            step[0] = i;

            if (i == 0){
                //beam repeat
                extraTokenEmb = batchRepeatInterleave(embRes.GetTensorData<float>(), extraEmbShape, mSearch.beamSize, false);
                mcaCache = batchRepeatInterleave<MatRX<float>, float>(graphEmb, mcaShape, mSearch.beamSize);
                contextMask = batchRepeatInterleave<MatRX<bool>, bool>(mask, maskShape, mSearch.beamSize);
                taskCount = batchRepeatInterleave(mol.lTask.data(), taskCountShape, mSearch.beamSize, false);
                numList = constBinCount(taskCount, taskCountShape[0], {0, 1});
                numListShape[0] = numList.size();
            }
            inputs.push_back(std::move(
                Ort::Value::CreateTensor<int64_t>(this->memInfo, inputToken.data(), inputToken.size(), inputTokenShape.data(), inputTokenShape.size())
            ));
            inputs.push_back(std::move(convertTensor<float, float>(extraTokenEmb, extraEmbShape, this->memInfo)));
            inputs.push_back(std::move(convertTensor<float, float>(msaCache, msaShape, this->memInfo)));
            inputs.push_back(std::move(convertTensor<float, float>(mcaCache, mcaShape, this->memInfo)));
            inputs.push_back(std::move(convertTensor<bool, bool>(contextMask, maskShape, this->memInfo)));
            inputs.push_back(std::move(convertTensor<int64_t, int64_t>(extraQ.data(), extraQShape, this->memInfo)));
            inputs.push_back(std::move(convertTensor<int64_t, int64_t>(extraK.data(), extraKShape, this->memInfo)));
            inputs.push_back(std::move(convertTensor<int64_t, int64_t>(numList.data(), numListShape, this->memInfo)));
            inputs.push_back(std::move(convertTensor<int64_t, int64_t>(step.data(), stepShape, this->memInfo)));

            auto outputs = this->Decoder->Run(
                Ort::RunOptions{nullptr},
                inputsName.data(), 
                inputs.data(), inputs.size(),
                outputsName.data(), outputsName.size()
            );

            if (i == 0){
                auto tokenProbShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                auto tokenProb = indexSelect(outputs[0].GetTensorData<float>(), tokenProbShape, {2}, 1, false);

                mSearch.generate(convertTensor<float, float>(tokenProb, tokenProbShape, this->memInfo));
                
                delete []tokenProb;
                delete []extraTokenEmb;
                extraTokenEmb = nullptr;
                contextMask = indexSelect(contextMask, maskShape, {0}, 2);
                extraEmbShape[1] = 0;
                extraQ[0] = 0;
            }
            else {mSearch.generate(outputs[0]);}
            if (mSearch.isDone()){break;}
            
            delete []msaCache;
            auto unfinishIdx = mSearch.unfinishIndex();
            msaShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
            msaCache = indexSelect(outputs[1].GetTensorData<float>(), msaShape, unfinishIdx, 1, false);
            mcaCache = indexSelect(mcaCache, mcaShape, unfinishIdx, 0);
            taskCount = indexSelect(taskCount, taskCountShape, unfinishIdx, 0);
            numList = constBinCount(taskCount, taskCountShape[0], {0, 1});
            contextMask = indexSelect(contextMask, maskShape, unfinishIdx, 0);
        }

        delete []msaCache;
        delete []mcaCache;
        delete []taskCount;
        delete []contextMask;
        return mSearch.finalize(this->rvocab);
    }

    std::tuple<std::vector<str>, std::vector<float>> SeqAGraphInfer::inferRun(
        const std::vector<str> &smis, std::vector<int64_t> &lTask,
        const int64_t beamSize, const int64_t batchSize,
        const float lengthPenalty, const int64_t minLength, const int64_t maxLength,
        const int64_t beamGroup, const float T, const int64_t returnNum, const str device
    ){
        auto batch = this->molHandler.generateBatch(smis, lTask);

        auto mSearch = Inference::SearchMethods(
            beamSize, batchSize, this->vocab["<BOS>"], this->vocab["<PAD>"], this->vocab["<EOS>"],
            lengthPenalty, minLength, maxLength, beamGroup, T, returnNum, device
        );
        return this->decoderRun(this->encoderRun(batch)[0], this->embeddingRun(batch)[0], batch, mSearch);
    }
}



// int main(){
//     // auto curDir = std::filesystem::current_path();
//     // str modelDir = "../Models/50k/encoder.onnx";
//     // Ort::Env env;
//     // Ort::SessionOptions sessionOption;
//     // auto Encoder = Ort::Session(env, modelDir.c_str(), sessionOption);
//     std::vector<str> smis = {"C=CCc1cccc([N+](=O)[O-])c1OCC#CC", "CC(=O)c1ccc2c(ccn2C(=O)OC(C)(C)C)c1"};
//     // std::vector<str> smis = {"C=CCc1cccc([N+](=O)[O-])c1OCC#CC", "CCCCC"};
//     // std::vector<str>smis = {"CCCCCC", "CCCCCCCCCCCC"};
//     std::vector<int64_t> lTask = {0, 0};
//     Inference::SeqAGraphInfer solver = Inference::SeqAGraphInfer(Inference::usptofull);
//     auto batch = solver.molHandler.generateBatch(smis, lTask);
//     auto encRes = solver.encoderRun(batch);
//     // // auto dataInfo = encRes[0].GetTensorTypeAndShapeInfo();
//     // // auto count = dataInfo.GetElementCount();
//     // // auto sp = dataInfo.GetShape();
//     auto embRes = solver.embeddingRun(batch);
//     auto mSearch = Inference::SearchMethods(
//         20, smis.size(), solver.vocab["<BOS>"], solver.vocab["<PAD>"], solver.vocab["<EOS>"], 1.0, 1, 300, 1, 1.0, 10, "cpu"
//     );
//     auto decRes = solver.decoderRun(encRes[0], embRes[0], batch, mSearch);
//     // auto padMats = Inference::graphPadding(encRes[0].GetTensorData<float>(), sp, batch.graphLength);
//     // std::cout << padMats[1].col(255) << std::endl;
//     // auto mask = Inference::getMask(3, batch.graphLength.maxCoeff(), batch.graphLength);
//     // std::cout << mask[0].row(2) << std::endl;
//     // MatRX<int64_t> a = MatRX<int64_t>::Zero(2, 2);
//     // MatRX<int64_t> b = MatRX<int64_t>::Zero(2, 2);
//     // std::vector<int64_t> a = {1, 2, 3, 4, 5, 6};
//     // std::vector<int64_t> shape1 = {2, 3};
//     // std::vector<int64_t> shape2 = {2, 3};
//     // auto b = Inference::indexSelect(a.data(), shape1, {1}, 0);
//     // for (int i=0; i < Inference::numel(shape1); i++){
//     //     std::cout << *(b + i) << "\t";
//     // }
//     // delete []b;
//     // b = Inference::indexSelect(a.data(), shape2, {0, 2}, 1);
//     // std::cout << "\n-----------" << std::endl;
//     // for (int i=0; i < Inference::numel(shape2); i++){
//     //     std::cout << *(b + i) << "\t";
//     // }
//     // auto index1 = Inference::indexGenerate(std::move(boolIndex), true);
//     // auto index2 = Inference::indexGenerate(std::vector<bool>{false, false, true}, false);

//     // a << 1, 2,
//     //      3, 4;
//     // b << -1, -2,
//     //      -3, -4;
//     // std::vector<MatRX<int64_t>> an;
//     // an.push_back(a);
//     // an.push_back(b);
//     // std::vector<int64_t> shape = {int64_t(an.size()), a.rows(), a.cols()};
//     // auto ra = Inference::batchRepeatInterleave<MatRX<int64_t>, int64_t>(an, shape, 3);
//     // // auto ra = Inference::binCount(a.data(), a.size(), 5);
    
//     // // int64_t count = 1;
//     // // for (auto i : shape){count *= i;}
//     // for (int i=0; i < Inference::numel(shape); i++){
//     //     std::cout << ra[i] << "\t";
//     // }

//     int k = 0;
// }