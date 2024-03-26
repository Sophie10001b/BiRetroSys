#include <Search/tree_utils.h>

int main(){
    // str test_smi = "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C";

    printf("loading terminal molecules..\n");
    std::unordered_set<str> tm;
    Search::loadTerminalMols(tm);
    
    str molName, testSmi;
    Search::searchTree *searchProcess = nullptr;
    while (1){
        printf("please input the molecule's NAME SMILES(e.g. name1 CCC), when NAME is EXIT, the search will be terminated.\n");
        std::cin >> molName;
        if (molName == "EXIT") break;
        std::cin >> testSmi;
        if (!RDKit::SmilesToMol(testSmi, 0, false)){
            printf("the input molecule %s is invalid, please check the SMILES string.\n", testSmi.data());
            continue;
        }

        searchProcess = new Search::searchTree(testSmi, molName, &tm, 20, 20, 150, 1.0f);
        searchProcess->multiStepSearch(100, -1, 0.1, true, 0.01);
        delete searchProcess;
        printf("current search is finish, please check the result and log in /Search/%s.\n", testSmi.data());
    }
}