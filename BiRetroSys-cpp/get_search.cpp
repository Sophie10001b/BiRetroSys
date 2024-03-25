#include <Search/tree_utils.h>

int main(){
    str test_smi = "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C";

    std::unordered_set<str> tm;
    Search::loadTerminalMols(tm);

    auto searchProcess = Search::searchTree(test_smi, "Nv", &tm, 20, 20, 150, 1.0f);
    searchProcess.multiStepSearch(100, -1, 0.1, true, 0.01);
}