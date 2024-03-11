#include <Inference/model_utils.h>
#include <Inference/tensor_utils.h>

namespace Inference{
    SearchHypotheses::SearchHypotheses(const int64_t beamSize, const float lengthPenalty, const bool doEarlyStop)
    :beamSize(beamSize), lengthPenalty(lengthPenalty), doEarlyStop(doEarlyStop){}

    void SearchHypotheses::push(const std::vector<int64_t> &hyp, float sumLogProbs){
        auto curScore = sumLogProbs / (pow(hyp.size(), this->lengthPenalty));
        if (this->beams.size() < this->beamSize || curScore > this->worstScore){
            this->beams.push_back(std::make_tuple(curScore, hyp));
            std::sort(this->beams.begin(), this->beams.end(), this->beamsCompare);
            if (this->beams.size() > this->beamSize){
                this->beams.pop_back();
                this->worstScore = std::get<0>(this->beams.back());
            }
            else {this->worstScore = curScore < this->worstScore ? curScore : this->worstScore;}
        }
    };

    bool SearchHypotheses::isDone(float bestProbs, int64_t curLength){
        if (this->beams.size() < this->beamSize){return false;}
        else if (this->doEarlyStop){return true;}
        else {
            auto curScore = bestProbs / (pow(curLength, this->lengthPenalty));
            return this->worstScore >= curScore;
        }
    };

    SearchScorer::SearchScorer(const int64_t batchSize, const int64_t beamSize, const int64_t beamGroup, const int64_t padIds, const int64_t eosIds, const float lengthPenalty, const bool doEarlyStop): batchSize(batchSize), beamSize(beamSize), padIds(padIds), eosIds(eosIds), beamGroup(beamGroup), lengthPenalty(lengthPenalty), doEarlyStop(doEarlyStop), groupSize(beamSize / beamGroup){
        this->done = std::vector<bool>(batchSize, false);
        this->beamHyps = std::vector<SearchHypotheses>(batchSize, SearchHypotheses(
            beamSize, lengthPenalty, doEarlyStop
        ));
        assert(this->beamGroup <= this->beamSize);
        assert(this->beamSize % this->beamGroup == 0);
    }
    
    bool SearchScorer::isDone(){return std::all_of(this->done.begin(), this->done.end(), [](const bool &a){return a;});}

    std::tuple<std::vector<float>, std::vector<int64_t>, std::vector<int64_t>> SearchScorer::process(
        std::vector<std::vector<int64_t>> &curToken, std::vector<float> &nextScore,
        std::vector<int64_t> &nextToken, std::vector<int64_t> &nextIdx
    ){
        int64_t curLength = curToken[0].size();
        int64_t candidateSize = nextScore.size() / this->batchSize;
        std::vector<float> nextBeamScore = std::vector<float>(this->batchSize * this->groupSize, 0);
        std::vector<int64_t> nextBeamToken = std::vector<int64_t>(this->batchSize * this->groupSize, 0);
        std::vector<int64_t> nextBeamIdx = std::vector<int64_t>(this->batchSize * this->groupSize, 0);

        int64_t beamIdx = 0;
        int64_t batchIdx = 0;
        for (auto &hyp : this->beamHyps){
            if (this->done[batchIdx]){
                std::fill(nextBeamScore.begin() + batchIdx * this->groupSize, nextBeamScore.begin() + (batchIdx + 1) * this->groupSize, 0);
                std::fill(nextBeamToken.begin() + batchIdx * this->groupSize, nextBeamToken.begin() + (batchIdx + 1) * this->groupSize, this->padIds);
                std::fill(nextBeamIdx.begin() + batchIdx * this->groupSize, nextBeamIdx.begin() + (batchIdx + 1) * this->groupSize, 0);
                batchIdx++;
                continue;
            }

            beamIdx = 0;
            for (int64_t tokenRank=0; tokenRank < candidateSize; tokenRank++){
                auto batchBeamIdx = batchIdx * this->groupSize + nextIdx[batchIdx * candidateSize + tokenRank];
                if (nextToken[batchIdx * candidateSize + tokenRank] == this->eosIds){
                    if (tokenRank >= this->groupSize) continue;
                    hyp.push(curToken[batchBeamIdx], nextScore[batchIdx * candidateSize + tokenRank]);
                }
                else {
                    nextBeamScore[batchIdx * this->groupSize + beamIdx] = nextScore[batchIdx * candidateSize + tokenRank];
                    nextBeamToken[batchIdx * this->groupSize + beamIdx] = nextToken[batchIdx * candidateSize + tokenRank];
                    nextBeamIdx[batchIdx * this->groupSize + beamIdx] = batchBeamIdx;
                    beamIdx++;
                }
                if (beamIdx == this->groupSize) break;
            }

            assert(beamIdx >= this->groupSize);
            this->done[batchIdx] = (this->done[batchIdx] || hyp.isDone(*(std::max_element(nextScore.begin() + batchIdx * candidateSize, nextScore.begin() + (batchIdx + 1) * candidateSize)), curLength));

            batchIdx++;
        }
        return std::make_tuple(nextBeamScore, nextBeamToken, nextBeamIdx);
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<float>> SearchScorer::finalize(
        std::vector<std::vector<int64_t>> &curToken, std::vector<float> &finalScore,
        const int64_t maxLength, const int64_t returnNum
    ){
        int64_t batchIdx = 0;
        for (auto &hyp : this->beamHyps){
            if (this->done[batchIdx]){
                batchIdx++;
                continue;
            }
            for (int64_t beamIdx=0; beamIdx < this->beamSize; beamIdx++){
                auto batchBeamIdx = batchIdx * this->beamSize + beamIdx;
                hyp.push(curToken[batchBeamIdx], finalScore[batchBeamIdx]);
            }
            batchIdx++;
        }

        std::vector<int64_t> resLength(this->batchSize * returnNum);
        std::vector<std::vector<int64_t>> bestHyp;
        std::vector<float> bestHypScore(this->batchSize * returnNum, 0);

        int64_t hypIdx = 0;
        for (auto &hyp : this->beamHyps){
            for (int64_t beamIdx=0; beamIdx < returnNum; beamIdx++){
                auto [hypScore, hypRes] = hyp.beams[beamIdx];
                bestHypScore[returnNum * hypIdx + beamIdx] = hypScore;
                //remove ["<BOS>"]
                resLength[returnNum * hypIdx + beamIdx] = hypRes.size() - 1;
                bestHyp.push_back(std::vector<int64_t>(hypRes.begin() + 1, hypRes.end()));
            }
            hypIdx++;
        }
        return std::make_tuple(bestHyp, bestHypScore);

        //no need to padding
        // auto resMaxLength = *(std::max(resLength.begin(), resLength.end())) + 1;
        // if (resMaxLength > maxLength) resMaxLength = maxLength;
        // std::vector<std::vector<int64_t>> res(this->batchSize * this->beamSize, std::vector<int64_t>(resMaxLength, this->padIds));

        // for (int64_t i=0; i < bestHyp.size(); i++){
        //     auto copyLength = resLength[i] < res[i].size() ? resLength[i] : res[i].size();
        //     std::copy(bestHyp[i].begin(), bestHyp[i].begin() + copyLength, res[i].begin());
        //     if (copyLength == resLength[i]){res[i][copyLength] = this->eosIds;}
        // }
        // return std::make_tuple(res, bestHypScore);
    }

    SearchMethods::SearchMethods(
        const int64_t beamSize, const int64_t batchSize,
        const int64_t bosIds, const int64_t padIds, const int64_t eosIds,
        const float lengthPenalty, const int64_t minLength,
        const int64_t maxLength, const int64_t beamGroup, const float T, const int64_t returnNum, const str device
    ): beamSize(beamSize), batchSize(batchSize), bosIds(bosIds), padIds(padIds), eosIds(eosIds), lengthPenalty(lengthPenalty), minLength(minLength), maxLength(maxLength), beamGroup(beamGroup), T(T), returnNum(returnNum), device(device), groupSize(beamSize / beamGroup){
        assert(this->returnNum <= this->beamSize);
        this->curToken = std::vector<int64_t>(batchSize * beamSize, bosIds);
        this->allToken = std::vector<std::vector<int64_t>>(batchSize * beamSize, std::vector<int64_t>(1, bosIds));
        this->searchScorer = new SearchScorer(batchSize, beamSize, beamGroup, padIds, eosIds, lengthPenalty, false);

        this->beamScore = std::vector<float>(batchSize * beamSize, -std::numeric_limits<float>::infinity());
        for (int i=0; i < batchSize * beamGroup; i++){this->beamScore[i * this->groupSize] = 0;}

        this->beamIdx = std::vector<int64_t>(batchSize * beamSize);
        std::iota(this->beamIdx.begin(), this->beamIdx.end(), 0);

        this->groupIdx = std::vector<int64_t>(batchSize * beamSize);
        for (int i=0; i < batchSize; i++){
            for (int j=0; j < groupSize; j++){this->groupIdx[i * beamSize + j] = j / groupSize;}
        }

        this->aliveBatch = std::vector<int64_t>(batchSize);
        std::iota(this->aliveBatch.begin(), this->aliveBatch.end(), 0);
        this->aliveIdx = std::vector<int64_t>(batchSize * beamSize);
        std::copy(this->beamIdx.begin(), this->beamIdx.end(), this->aliveIdx.begin());
        this->unfinishIdx = indexGenerate(this->searchScorer->done, false);
    }

    SearchMethods::~SearchMethods(){delete searchScorer;}

    bool SearchMethods::isDone(){return (this->searchScorer->isDone() || this->allToken[0].size() >= this->maxLength);}

    std::vector<int64_t> SearchMethods::currentToken(){
        if (this->unfinishIdx.size() < this->batchSize){
            return std::get<0>(indexSelect(this->curToken, {this->batchSize, this->beamSize}, this->unfinishIdx, 0));
        }
        else {return this->curToken;}
    }

    std::vector<int64_t> SearchMethods::unfinishIndex(){
        if (this->unfinishIdx.size() < this->batchSize){
            auto unfinish = std::get<0>(indexSelect(this->beamIdx, {this->batchSize, this->beamSize}, this->unfinishIdx, 0));
            if (this->aliveBatch.size() < this->batchSize){
                std::vector<int64_t> aliveArange(this->aliveBatch.size() * this->beamSize);
                std::iota(aliveArange.begin(), aliveArange.end(), 0);
                indexCopy(this->aliveIdx.data(), aliveArange.data(), {this->batchSize, this->beamSize}, this->aliveBatch);
                unfinish = std::get<0>(indexSelect(this->aliveIdx, {this->batchSize * this->beamSize}, unfinish, 0));
            }
            this->aliveBatch = this->unfinishIdx;
            return unfinish;
        }
        else {return this->beamIdx;}
    }

    float *SearchMethods::finishBatchPad(const Ort::Value &decOutput, const std::vector<int64_t> &decShape){
        float *decVecPtr = new float[this->batchSize * this->beamSize * decShape.back()]();
        if (decShape[0] < this->batchSize * this->beamSize){
            indexCopy(decVecPtr, decOutput.GetTensorData<float>(), {this->batchSize, this->beamSize * decShape.back()}, this->unfinishIdx);
        }
        else {std::copy(decOutput.GetTensorData<float>(), decOutput.GetTensorData<float>() + numel(decShape), decVecPtr);}
        return decVecPtr;
    };

    void SearchMethods::generate(const Ort::Value &decOutput){
        std::vector<int64_t> decOutShape = decOutput.GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocabSize = decOutShape.back();
        auto padDecOut = this->finishBatchPad(decOutput, decOutShape);
        std::vector<int64_t> padDecShape = {this->batchSize * this->beamSize, vocabSize};
        lastSoftmax(padDecOut, padDecShape, this->T, true);
        
        for (int i=0; i < padDecShape[0]; i++){
            auto beamScore = this->beamScore[i];
            std::for_each(padDecOut + i * vocabSize, padDecOut + (i + 1) * vocabSize, [&beamScore](float &a){a += beamScore;});
        }
        padDecShape = {this->batchSize, this->beamGroup, this->groupSize * vocabSize};

        if (this->beamGroup > 1){
            for (int groupId=0; groupId < this->beamGroup; groupId++){
                std::vector<int64_t> groupIdx = {};
                for (int i=0; i < this->groupIdx.size(); i++){
                    if (groupId == this->groupIdx[i]){groupIdx.push_back(i);}
                }
                auto groupAllToken = firstIndexSelect(this->allToken, groupIdx);
                auto [groupLogit, groupShape] = indexSelect(padDecOut, std::move(padDecShape), {groupId}, 1, false);
                groupShape.erase(groupShape.begin() + 1);
                auto [nextTokenScore, nextToken] = lastTopK(groupLogit, groupShape, this->groupSize * 2, true, false);
                
                std::vector<int64_t> nextBeamIdx(nextToken.size());
                std::transform(nextToken.begin(), nextToken.end(), nextBeamIdx.begin(), [&vocabSize](const int64_t &a){return a / vocabSize;});
                std::for_each(nextToken.begin(), nextToken.end(), [&vocabSize](int64_t &a){a = a % vocabSize;});
                auto beamRes = this->searchScorer->process(groupAllToken, nextTokenScore, nextToken, nextBeamIdx);
                nextBeamIdx = std::get<2>(beamRes);

                indexCopy(this->beamScore.data(), std::get<0>(beamRes).data(), {this->batchSize * this->beamSize}, groupIdx);
                indexCopy(this->allToken, firstIndexSelect(groupAllToken, nextBeamIdx), groupIdx);
                indexCopy(this->curToken.data(), std::get<1>(beamRes).data(), {this->batchSize * this->beamSize}, groupIdx);

                std::vector<int64_t> updateBeamIdx(this->batchSize * this->groupIdx.size());
                std::for_each(nextBeamIdx.begin(), nextBeamIdx.end(), [this, &groupId](int64_t &a){a = (a / this->groupSize) * this->groupSize + groupId * this->beamSize + a % this->groupSize;});
                indexCopy(this->beamIdx.data(), nextBeamIdx.data(), {this->batchSize * this->beamSize}, groupIdx);
            }
        }
        else {
            auto [nextTokenScore, nextToken] = lastTopK(padDecOut, padDecShape, this->beamSize * 2, true, false);
            std::vector<int64_t> nextBeamIdx(nextToken.size());
            std::transform(nextToken.begin(), nextToken.end(), nextBeamIdx.begin(), [&vocabSize](const int64_t &a){return a / vocabSize;});
            std::for_each(nextToken.begin(), nextToken.end(), [&vocabSize](int64_t &a){a = a % vocabSize;});
            auto beamRes = this->searchScorer->process(this->allToken, nextTokenScore, nextToken, nextBeamIdx);

            this->allToken = firstIndexSelect(this->allToken, std::get<2>(beamRes));
            this->beamScore = std::get<0>(beamRes);
            this->curToken = std::get<1>(beamRes);
            this->beamIdx = std::get<2>(beamRes);
        }
        delete []padDecOut;
        for (int i=0; i < this->allToken.size(); i++){this->allToken[i].push_back(this->curToken[i]);}
        if (std::any_of(this->searchScorer->done.begin(), this->searchScorer->done.end(), [](const bool &a){return a;})){
            this->unfinishIdx = indexGenerate(this->searchScorer->done, false);
        }
    }

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<float>> SearchMethods::finalize(){
        return this->searchScorer->finalize(this->allToken, this->beamScore, this->maxLength, this->returnNum);
    }

    std::tuple<std::vector<str>, std::vector<float>> SearchMethods::finalize(const std::map<int64_t, str> &rvocab){
        auto [beamRes, beamScore] = this->searchScorer->finalize(this->allToken, this->beamScore, this->maxLength, this->returnNum);
        std::vector<str> beamStrRes;

        for (int i=0; i < beamRes.size(); i++){
            str tempRes = "";
            for (auto idx : beamRes[i]){
                auto findRes = rvocab.find(idx);
                if (findRes != rvocab.end()){tempRes += findRes->second;}
            }
            beamStrRes.push_back(tempRes);
        }
        return std::make_tuple(beamStrRes, beamScore);
    }
}