#include <Test/include_head.h>

int main(){
    const str datasetDir = "/Users/sophie/Code/BiRetroSys/BiRetroSys/Models/50k/token(test).txt";
    std::ifstream fin(datasetDir);
    str line;
    std::vector<str> prods, reacs;
    int64_t count = 0;
    int64_t testCount = 20;
    while (count < testCount && getline(fin, line)){
        prods.push_back(strtok(line.data(), "\t"));
        reacs.push_back(strtok(NULL, "\t"));
        count++;
    }

    const int64_t datasetSize = prods.size();
    const int64_t assumBatchSize = 1;
    const int64_t beamSize = 20;
    const int64_t returnNum = 10;
    const float T = 1.0;
    const str device = "cpu";

    int64_t batchSize = assumBatchSize;

    std::vector<int64_t> topnCount(returnNum, 0);
    std::vector<float> topnAcc(returnNum, 0);
    std::vector<int64_t> lTask(batchSize, 0);
    std::vector<str> smis(batchSize);
    std::vector<str> tgtSmis(batchSize);
    auto solver = Inference::SeqAGraphInfer(Inference::usptofull);
    int64_t processCount = prods.size() % assumBatchSize == 0 ? prods.size() / assumBatchSize : (prods.size() / assumBatchSize) + 1;
    // int64_t processCount = 5;

    float timeCount = 0;

    auto timeBegin = std::chrono::high_resolution_clock::now();
    for (int64_t bcnt=0; bcnt < processCount; bcnt++){
        auto batchBegin = std::chrono::high_resolution_clock::now();
        int64_t finishCount = bcnt * assumBatchSize;
        int64_t lefts = prods.size() - finishCount;
        batchSize = lefts >= assumBatchSize ? assumBatchSize : lefts;

        std::vector<int64_t> lTask(batchSize, 0);
        std::vector<str> smis(batchSize);
        std::vector<str> tgtSmis(batchSize);
        std::transform(prods.begin() + finishCount, prods.begin() + finishCount + batchSize, smis.begin(), [](str a){return a;});
        std::transform(reacs.begin() + finishCount, reacs.begin() + finishCount + batchSize, tgtSmis.begin(), [](str a){return a;});
        // std::copy(prods.begin() + finishCount, prods.begin() + finishCount + batchSize, smis.begin());
        // std::copy(reacs.begin() + finishCount, reacs.begin() + finishCount + batchSize, tgtSmis.begin());

        auto inferRes = solver.inferRun(
            smis, lTask, beamSize, batchSize, 0.0, 1, 150, 1, T, returnNum, device
        );
        auto inferSmis = std::get<0>(inferRes);
        std::for_each(inferSmis.begin(), inferSmis.end(), [&solver](str &smi){smi = std::get<0>(solver.molHandler.canonicalizeSmiles(smi));});

        for (int64_t cnt=0; cnt < batchSize; cnt++){
            const int64_t inferFinishCount = cnt * returnNum;
            for (int64_t returnCnt=0; returnCnt < returnNum; returnCnt++){
                if (inferSmis[inferFinishCount + returnCnt] == tgtSmis[cnt]){
                    // topnCount[returnCnt]++;
                    std::for_each(topnCount.begin() + returnCnt, topnCount.end(), [](int64_t &a){a+=1;});
                    break;
                }
            }
        }
        auto batchEnd = std::chrono::high_resolution_clock::now();
        auto batchSpend = std::chrono::duration_cast<std::chrono::milliseconds>(batchEnd - batchBegin);
        timeCount += batchSpend.count() * 1e-3;
        std::cout << "batch " << bcnt + 1 << "(" << finishCount + batchSize << ")is finished, cost " << batchSpend.count() * 1e-3 << " seconds.\n";
        for (int i=0; i < returnNum; i++){
            std::cout << "Top-" << i + 1 << " C\t";
        }
        std::cout << std::endl;
        for (int i=0; i < returnNum; i++){
            std::printf("%.lld%s", topnCount[i], "  ");
        }
        std::cout << std::endl;
    }
    auto timeEnd = std::chrono::high_resolution_clock::now();
    auto timeSpend = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeBegin);
    std::transform(topnCount.begin(), topnCount.end(), topnAcc.begin(), [&datasetSize](int64_t &a){return float(a / datasetSize);});
    std::cout << "finish " << prods.size() << " reaction predictions, cost " << timeSpend.count() * 1e-3 << " seconds, avg time:" << timeCount / prods.size() <<"\n";
    for (int i=0; i < returnNum; i++){
        std::cout << "Top-" << i + 1 << " A%\t";
    }
    std::cout << std::endl;
    for (int i=0; i < returnNum; i++){
        std::printf("%.4f%s", topnAcc[i], "%  ");
    }
}