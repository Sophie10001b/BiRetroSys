#pragma once
#include <Search/value_fun.h>

namespace Search {
    class reactionNode;

    class moleculeNode {
        public:
        const int id;
        const str mol;
        reactionNode *parent;
        std::vector<reactionNode*> children;

        bool hasFound;
        bool isTerminal;
        bool isOpen;

        const float cost;
        float rn;
        float vmt;

        moleculeNode(const int id, const str &mol, const float value, reactionNode *parent, bool isTerminal);

        void init();
        void update();
        void updateVmt();
        void close();
        void getAncestor(std::unordered_set<str> &ancestor);
        void upToDownUpdate();
        str getStrFeature();

        ~moleculeNode();
    };

    class reactionNode {
        public:
        const int id;
        moleculeNode *parent;
        std::vector<moleculeNode*> children;

        bool hasFound;
        bool isOpen;

        const float cost;
        float costs;
        float rn;

        reactionNode(const int id, const float value, moleculeNode *parent);

        void init();
        void update(float dRn);
        void updateVmt();
        str getStrFeature();

        ~reactionNode();

        private:
        float getCosts();
    };

    class searchTree {
        public:
        const str target;
        const str targetName;
        const std::unordered_set<str> terminalMol;
        const int expansionWidth;
        const int checkWidth;
        const int singleSteps;
        const float T;
        bool hasFound;

        std::vector<moleculeNode*> molNodes;
        std::vector<reactionNode*> reacNodes;
        std::unordered_set<str> excludeMols;

        searchTree(const str &target, const str &targetName, const std::unordered_set<str> &terminalMol, const int expansionWidth=10, const int checkWidth=10, const int singleSteps=150, const float T=1.0);

        int multiStepSearch(const int steps=100, const int earlyStop=-1, const float lowerBound=0.1, const bool consistCheck=true);

        ~searchTree();

        private:
        moleculeNode *root;
        valueModel *valModel;
        Inference::SeqAGraphInfer *inferModel;

        std::ofstream searchLog;

        std::vector<float> valueFun(const std::vector<str> &smis);
        std::vector<std::unordered_map<str, float>> inferFun(const std::vector<str> &smis, const bool isRetro=true);
        std::pair<std::vector<std::vector<str>>, std::vector<float>> filterRun(const str &expandSmi, const std::vector<std::unordered_map<str, float>> &expandResults, const float lowerBound, const bool consistCheck);

        moleculeNode *addMol(const str &mol, reactionNode *parent, float value);
        reactionNode *addReaction(const std::vector<str> &reaction, moleculeNode *parent, float cost, std::unordered_set<str> &ancestor);
        bool expandTree(moleculeNode *startMol, std::vector<std::vector<str>> &expandResults, std::vector<float> &scores);
        bool finishSearch();
        void visualization(const str &name);
        void visualizationBest(const str &name);
    };

    void loadTerminalMols(std::unordered_set<str> &finalSet, str &&molPath="");
}