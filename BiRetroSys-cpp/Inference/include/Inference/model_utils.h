#pragma once
#include <Inference/include_head.h>

namespace Inference {
    enum modelClass {uspto50k, usptofull};

//----------------------------------------------------------------------------
    class SearchHypotheses {
        public:
        const int64_t beamSize;
        const float lengthPenalty;
        const bool doEarlyStop;

        std::vector<std::tuple<float, std::vector<int64_t>>> beams;

        SearchHypotheses(const int64_t beamSize, const float lengthPenalty, const bool doEarlyStop);

        void push(const std::vector<int64_t> &hyp, float sumLogProbs);
        bool isDone(float bestProbs, int64_t curLength);

        private:
        float worstScore = 1e9;
        std::function<bool(const std::tuple<float, std::vector<int64_t>>, const std::tuple<float, std::vector<int64_t>>)> beamsCompare = [](const std::tuple<float, std::vector<int64_t>> &a, const std::tuple<float, std::vector<int64_t>> &b)-> bool {return std::get<0>(a) > std::get<0>(b);};
    };

    class SearchScorer {
        public:
        const int64_t batchSize, beamSize, beamGroup, padIds, eosIds;
        const float lengthPenalty;
        const bool doEarlyStop;

        const int64_t groupSize;
        std::vector<bool> done;

        SearchScorer(
            const int64_t batchSize, const int64_t beamSize, const int64_t beamGroup, const int64_t padIds, const int64_t eosIds,
            const float lengthPenalty, const bool doEarlyStop
        );

        bool isDone();
        std::tuple<std::vector<float>, std::vector<int64_t>, std::vector<int64_t>> process(
            std::vector<std::vector<int64_t>> &curToken, std::vector<float> &nextScore,
            std::vector<int64_t> &nextToken, std::vector<int64_t> &nextIdx
        );
        std::tuple<std::vector<std::vector<int64_t>>, std::vector<float>> finalize(
            std::vector<std::vector<int64_t>> &curToken, std::vector<float> &finalScore,
            const int64_t maxLength, const int64_t returnNum
        );

        private:
        bool isInit = false;
        std::vector<SearchHypotheses> beamHyps;
    };

    class SearchMethods {
        public:
        const int64_t beamSize, batchSize, bosIds, padIds, eosIds, minLength, maxLength, beamGroup, returnNum;
        const float lengthPenalty, T;
        const str device;

        const int64_t groupSize;

        SearchMethods(
            const int64_t beamSize=20, const int64_t batchSize=1,
            const int64_t bosIds=-1, const int64_t padIds=-1, const int64_t eosIds=-1,
            const float lengthPenalty=1.0, const int64_t minLength=1, const int64_t maxLength=150,
            const int64_t beamGroup=1, const float T=1.0, const int64_t returnNum=10, const str device="cpu"
        );

        std::vector<int64_t> currentToken();
        std::vector<int64_t> unfinishIndex();

        bool isDone();
        void generate(const Ort::Value &decOutput);
        std::tuple<std::vector<std::vector<int64_t>>, std::vector<float>> finalize();
        std::tuple<std::vector<str>, std::vector<float>> finalize(const std::map<int64_t, str> &rvocab);
        
        ~SearchMethods();

        private:
        std::vector<int64_t> curToken;
        std::vector<int64_t> memIdx;
        std::vector<int64_t> beamIdx;
        std::vector<int64_t> aliveIdx;
        std::vector<int64_t> aliveBatch;
        std::vector<int64_t> unfinishIdx;
        std::vector<int64_t> groupIdx;
        std::vector<float> beamScore;
        std::vector<std::vector<int64_t>> allToken;
        SearchScorer *searchScorer;

        float *finishBatchPad(const Ort::Value &decOutput, const std::vector<int64_t> &decShape);
    };


//----------------------------------------------------------------------------
    class SeqAGraphInfer {
        public:
        std::map<str, int64_t> vocab;
        std::map<int64_t, str> rvocab;
        MolHandler::molPreprocess molHandler = MolHandler::molPreprocess();

        SeqAGraphInfer(
            const modelClass &modelSelect=usptofull, const str &device="cpu",
            const initlist<str> &extraToken={"<BOS>", "<EOS>", "<PAD>", "<UNK>"}
        );

        std::vector<Ort::Value> encoderRun(MolHandler::inputData &mol);

        std::vector<Ort::Value> embeddingRun(MolHandler::inputData &mol);

        std::tuple<std::vector<str>, std::vector<float>> decoderRun(const Ort::Value &encRes, const Ort::Value &embRes, const MolHandler::inputData &mol, SearchMethods &mSearch);

        std::tuple<std::vector<str>, std::vector<float>> inferRun(
            const std::vector<str> &smis, std::vector<int64_t> &lTask,
            const int64_t beamSize=20, const int64_t batchSize=1,
            const float lengthPenalty=1.0, const int64_t minLength=1, const int64_t maxLength=150,
            const int64_t beamGroup=1, const float T=1.0, const int64_t returnNum=10, const str device="cpu"
        );

        ~SeqAGraphInfer();

        private:
        Ort::Env env;
        Ort::Session *Encoder = nullptr;
        Ort::Session *ExtraEmbedding = nullptr;
        Ort::Session *Decoder = nullptr;
        Ort::SessionOptions sessionOption;
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    };
}