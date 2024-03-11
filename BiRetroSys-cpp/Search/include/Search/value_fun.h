#pragma once
#include <Search/include_head.h>

namespace Search {
    class valueModel {
        public:
        const int dFP = 2048;

        valueModel(const str &device="cpu");

        std::vector<float> valueRun(const std::vector<str> &smis);

        ~valueModel();

        private:
        Ort::Env env;
        Ort::Session *vModel = nullptr;
        Ort::SessionOptions sessionOption;
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    };
}
