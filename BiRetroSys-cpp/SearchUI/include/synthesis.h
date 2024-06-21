#pragma once

#include "include_head.h"
#include "settings.h"

struct multiStepSynthesisLog{
    str targetSmi;
    int spendSteps;
    float spendTimes;

    multiStepSynthesisLog(const str &smi, int step, float spendTimes): targetSmi(smi), spendSteps(step), spendTimes(spendTimes){}
};

class multiStepSynthesis: public QObject{
    Q_OBJECT

    public:
    multiStepSynthesis(const str &targetSmi, const std::unordered_set<str> *terminalMols, const searchSettings &ss, QObject *parent=nullptr);
    ~multiStepSynthesis();

    Search::searchTree *tree = nullptr;

    bool runSynthesis = true;
    int terminateClass = 0;

    public slots:
    void startSearch();

    signals:
    void finishSearch(int status, multiStepSynthesisLog searchStatus);
    void findRoute();
    void finishEachStep(int curStep, int totalStep, const str &curSmi, std::vector<std::vector<str>> topSmi, std::vector<float> topSmiScore);

    private:
    const searchSettings treeSettings;
};