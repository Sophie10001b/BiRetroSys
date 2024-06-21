#include "synthesis.h"

multiStepSynthesis::multiStepSynthesis(const str &targetSmi, const std::unordered_set<str> *terminalMols, const searchSettings &ss, QObject *parent): treeSettings(ss), QObject(parent){
    this->tree = new Search::searchTree(targetSmi, targetSmi, terminalMols, this->treeSettings.expansionWidth, this->treeSettings.checkWidth, this->treeSettings.singleSearchSteps, this->treeSettings.temperature);
}

multiStepSynthesis::~multiStepSynthesis(){
    delete this->tree;
}

void multiStepSynthesis::startSearch(){
    int step = 0;
    auto searchBegin = std::chrono::high_resolution_clock::now();
    if (!this->tree->hasFound){
        for (; step < this->treeSettings.multiSearchSteps; step++){
            Search::moleculeNode *nextMol = nullptr;
            float minCost = MAXFLOAT;
            for (auto p : this->tree->molNodes){
                if (p->isOpen && p->vmt < minCost){
                    minCost = p->vmt;
                    nextMol = p;
                }
            }

            // no open nodes
            if (!nextMol){
                outputLog("No open nodes, search terminate.", this->tree->searchLog);
                this->terminateClass = 1;
                break;
            }
            outputLog("Step " + std::to_string(step+1) + "/" + std::to_string(this->treeSettings.multiSearchSteps) + ": Trying to expand " + nextMol->mol, this->tree->searchLog);

            if (!this->runSynthesis){
                this->terminateClass = 2;
                break;
            }

            auto [expandSmis, expandScores] = this->tree->filterRun(nextMol->mol, this->tree->inferFun({nextMol->mol}), this->treeSettings.expansionLowerBound, this->treeSettings.needConsistCheck, this->treeSettings.checkLowerBound);

            // get top-k results
            emit this->finishEachStep(step, this->treeSettings.multiSearchSteps, nextMol->mol, expandSmis, expandScores);
            
            bool routeFound = this->tree->expandTree(nextMol, expandSmis, expandScores);
            if (routeFound && !this->tree->hasFound){
                this->tree->hasFound = true;
                outputLog("Route has found!", this->tree->searchLog);
                emit this->findRoute();
            }
            if (this->tree->hasFound && this->treeSettings.isEarlyStop) break;
        }
    }
    auto searchEnd = std::chrono::high_resolution_clock::now();
    auto searchSpend = std::chrono::duration_cast<std::chrono::milliseconds>(searchEnd - searchBegin).count() * 1e-3;
    if (!this->terminateClass) this->tree->finishSearch();
    emit this->finishSearch(this->terminateClass, multiStepSynthesisLog(this->tree->target, step, searchSpend));
}