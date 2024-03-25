#include <Search/value_fun.h>

namespace Search {
    valueModel::valueModel(const str &device){
        if (device == "cuda"){
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            this->sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        str curPath = std::filesystem::current_path().parent_path();
        const str valueModelDir = curPath + "/Models/valueMLP.onnx";
        this->vModel = new Ort::Session(this->env, valueModelDir.c_str(), this->sessionOption);
    }

    std::vector<float> valueModel::valueRun(const std::vector<str> &smis){
        const std::vector<const char*> inputsName = {"molFP"};
        const std::vector<const char*> outputsName = {"molValue"};
        const int bsz = smis.size();

        std::vector<float> inputs;
        for (const str &s : smis){
            auto res = RDKit::MorganFingerprints::getFingerprintAsBitVect(*RDKit::SmilesToMol(s), 2, this->dFP);
            std::vector<float> bitsVec(this->dFP, 0);
            std::vector<int> bits(res->getNumBits());
            res->getOnBits(bits);
            for (const int &idx : bits) bitsVec[idx] = 1;
            inputs.insert(inputs.end(), bitsVec.begin(), bitsVec.end());
        }
        std::vector<int64_t> inputSize = {bsz, this->dFP};
        Ort::Value inputOrt = Inference::convertTensor<float, float>(inputs.data(), inputSize, this->memInfo);

        auto outputs = this->vModel->Run(
            Ort::RunOptions{nullptr},
            inputsName.data(), &inputOrt, inputsName.size(),
            outputsName.data(), outputsName.size()
        );

        auto res = outputs[0].GetTensorData<float>();
        return std::vector<float>(res, res + bsz);
    }

    valueModel::~valueModel(){
        delete this->vModel;
    }
}