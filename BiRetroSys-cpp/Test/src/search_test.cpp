#include <Test/include_head.h>
#include <Search/tree_utils.h>

int main(){
    // str test_smi = "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C";

    // std::unordered_set<str> tm;
    // Search::loadTerminalMols(tm);

    // auto searchProcess = Search::searchTree(test_smi, "Nv", &tm, 20, 20, 150, 1.0f);
    // searchProcess.multiStepSearch(100, -1, 0.05);
    
    str testDir = std::filesystem::current_path().parent_path();
    testDir += "/Models/routes_test.txt";
    str testLogDir = testDir + "/Search/test_log.log";
    std::unordered_set<str> terminals;
    Search::loadTerminalMols(terminals);

    std::ifstream testData;
    std::ofstream testLog;
    testData.open(testDir, std::ios::in);
    testLog.open(testLogDir, std::ios::out | std::ios::trunc);

    str tgt;
    int count = 0;
    int succCount = 0;
    int stepCount = 0;
    double timeCount = 0.0;
    Search::searchTree *searchProcess = nullptr;
    while (std::getline(testData, tgt)){
        auto pstart = std::chrono::high_resolution_clock::now();
        searchProcess = new Search::searchTree(tgt, std::to_string(count), &terminals, 20, 20, 150, 1.0f);
        auto [succ, step] = searchProcess->multiStepSearch(100, -1, 0.01);
        auto pend = std::chrono::high_resolution_clock::now();
        auto pcost = std::chrono::duration_cast<std::chrono::milliseconds>(pend - pstart).count() * 1e-3;

        if (succ){
            succCount++;
            stepCount += step;
            timeCount += pcost;
        }

        str logStr = std::to_string(succCount) + " | " + std::to_string(count+1) + " " + tgt + " search " + (succ ? "successed" : "failed");
        outputLog(logStr, testLog);

        delete searchProcess;
        count++;
        // if (count == 20) break;
    }

    str logStr = std::to_string(count) + " planning finish, success: " + std::to_string(succCount) + " | " + std::to_string((succCount / count) * 100) + "%, " + "lengths: " + std::to_string(stepCount / count) + ", " + "times: " + std::to_string(timeCount / count) + "s/mol.\n";
    outputLog(logStr, testLog);
    testLog.close();
    testData.close();
}