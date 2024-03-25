#include <Search/tree_utils.h>

namespace Search {
    moleculeNode::moleculeNode(const int id, const str &mol, const float value, reactionNode *parent, bool isTerminal): id(id), mol(mol), cost(value){
        this->parent = parent;
        this->children = std::vector<reactionNode*>(0);

        this->hasFound = isTerminal;
        this->isTerminal = isTerminal;
        this->isOpen = !isTerminal;
        this->rn = isTerminal ? 0 : value;
        this->vmt = 0;

        this->gvnode = nullptr;

        if (parent) parent->children.push_back(this);
    }

    void moleculeNode::init(){
        assert(this->isOpen);
        this->isOpen = false;
        this->update();
    }

    void moleculeNode::update(){
        assert(!this->isOpen);
        float newRn = std::numeric_limits<float>::max();
        for (const auto reaction : this->children){
            newRn = std::min(newRn, reaction->rn);
            this->hasFound |= reaction->hasFound;
        }
        float dRn = newRn - this->rn;
        this->rn = newRn;
        if (this->parent) this->parent->update(dRn);
    }

    void moleculeNode::updateVmt(){
        this->vmt = 0;
        if (!this->parent) this->vmt = this->rn;
        else{
            this->vmt += this->parent->costs;
            reactionNode *temp = this->parent;
            while (temp){
                for (const auto mol : temp->children) this->vmt += mol->rn;
                temp = temp->parent->parent;
            }
            for (const auto &reaction : this->children) reaction->updateVmt();
        }
    }

    void moleculeNode::close(){
        this->isOpen = false;
        this->rn = std::numeric_limits<float>::max();
        if (this->parent) this->parent->update(this->rn);
    }

    void moleculeNode::getAncestor(std::unordered_set<str> &ancestor){
        if (this->parent) this->parent->parent->getAncestor(ancestor);
        ancestor.insert(this->mol);
    }

    void moleculeNode::upToDownUpdate(){
        if (!this->parent){
            for (const auto &child : this->children) child->updateVmt();
        }
        else if (!this->parent->parent->parent) this->parent->updateVmt();
        else this->parent->parent->upToDownUpdate();
    }

    void moleculeNode::vizPrepare(Agraph_t *curGraph){
        str nodeName = std::to_string(this->id) + " | " + this->mol;
        this->gvnode = agnode(curGraph, nodeName.data(), 1);
        
        str nodeColor = this->isOpen ? "lightgrey" : "aquamarine";

        if (this->hasFound) nodeColor = "lightblue";
        else if (this->isTerminal) nodeColor = "lightyellow";
        agsafeset(this->gvnode, "color", nodeColor.c_str(), "");
        agsafeset(this->gvnode, "shape", "box", "");
        agsafeset(this->gvnode, "style", "filled", "");
    }

    moleculeNode::~moleculeNode(){}

    reactionNode::reactionNode(const int id, const float value, moleculeNode *parent): id(id), cost(value){
        this->parent = parent;
        this->children = std::vector<moleculeNode*>(0);

        this->hasFound = false;
        this->isOpen = true;
        this->costs = 0;
        this->rn = 0;

        this->gvnode = nullptr;

        if (parent) parent->children.push_back(this);
    }

    float reactionNode::getCosts(){
        return this->parent->parent ? this->cost + this->parent->parent->getCosts() : this->cost;
    }

    void reactionNode::init(){
        assert(this->isOpen);
        this->isOpen = false;

        this->hasFound = true;
        this->rn = 0;
        for (const auto mol : this->children){
            this->rn += mol->rn;
            this->hasFound &= mol->hasFound;
        }
        this->costs = this->getCosts();
    }

    void reactionNode::update(float dRn){
        assert(!this->isOpen);
        this->rn += dRn;
        this->hasFound = true;
        for (const auto mol : this->children) this->hasFound &= mol->hasFound;
        this->parent->update();
    }

    void reactionNode::updateVmt(){
        for (const auto mol : this->children) mol->updateVmt();
    }

    void reactionNode::vizPrepare(Agraph_t *curGraph){
        this->gvnode = agnode(curGraph, std::to_string(this->id).data(), 1);
        
        str nodeColor = this->isOpen ? "lightgrey" : "aquamarine";

        if (this->hasFound) nodeColor = "lightblue";
        agsafeset(this->gvnode, "color", nodeColor.c_str(), "");
        agsafeset(this->gvnode, "shape", "rarrow", "");
        agsafeset(this->gvnode, "style", "filled", "");
    }

    reactionNode::~reactionNode(){}

    // searchTree
    searchTree::searchTree(const str &target, const str &targetName, const std::unordered_set<str> *terminalMol, const int expansionWidth, const int checkWidth, const int singleSteps, const float T): target(target), targetName(targetName), terminalMol(terminalMol), expansionWidth(expansionWidth), checkWidth(checkWidth), singleSteps(singleSteps), T(T){
        this->hasFound = false;
        if (this->terminalMol->find(this->target) != this->terminalMol->end()){
            this->hasFound = true;
            outputLog("Target Molecule already in terminal Molecules.", this->searchLog);
        }

        this->valModel = new valueModel();
        this->inferModel = new Inference::SeqAGraphInfer();
        this->root = this->addMol(this->target, nullptr, this->valueFun({this->target})[0]);
        this->excludeMols = {"", "CC"};

        // load LOG
        str logName = std::filesystem::current_path().parent_path();
        logName += "/Search/" + targetName + "/";
        if (!std::filesystem::exists(logName)) std::filesystem::create_directories(logName);

        this->savePath = logName;
        logName += targetName + ".log";
        this->searchLog.open(logName, std::ios::out | std::ios::trunc);
    }

    std::pair<bool, int> searchTree::multiStepSearch(const int steps, const int earlyStop, const float lowerBound, const bool consistCheck, const float checkLowerBound){
        int step = 0;
        if (!this->hasFound){
            for (; step < steps; step++){
                moleculeNode *nextMol = nullptr;
                float minCost = MAXFLOAT;
                for (auto p : this->molNodes){
                    if (p->isOpen && p->vmt < minCost){
                        minCost = p->vmt;
                        nextMol = p;
                    }
                }

                if (!nextMol){
                    outputLog("No open nodes, search terminate.", this->searchLog);
                    break;
                }
                outputLog("Step " + std::to_string(step+1) + "/" + std::to_string(steps) + ": Trying to expand " + nextMol->mol, this->searchLog);

                auto [expandSmis, expandScores] = this->filterRun(nextMol->mol, this->inferFun({nextMol->mol}), lowerBound, consistCheck, checkLowerBound);

                bool routeFound = this->expandTree(nextMol, expandSmis, expandScores);
                if (routeFound && !this->hasFound){
                    this->hasFound = true;
                    outputLog("Route has found!", this->searchLog);
                }
                if (this->hasFound && step+1 > earlyStop) break;
            }
        }
        return std::make_pair(this->finishSearch(), step+1);
    }

    moleculeNode *searchTree::addMol(const str &mol, reactionNode *parent, float value){
        moleculeNode *newMol = new moleculeNode(this->molNodes.size(), mol, value, parent, this->terminalMol->find(mol) != this->terminalMol->end());
        this->molNodes.push_back(newMol);
        return newMol;
    }

    reactionNode *searchTree::addReaction(const std::vector<str> &reaction, moleculeNode *parent, float cost, std::unordered_set<str> &ancestor){
        for (const auto &mol : reaction){
            if (ancestor.find(mol) != ancestor.end()) return (reactionNode*)nullptr;
        }
        auto values = this->valueFun(reaction);
        reactionNode *newReaction = new reactionNode(this->reacNodes.size(), cost, parent);
        for (int i=0; i < reaction.size(); i++) this->addMol(reaction[i], newReaction, values[i]);
        newReaction->init();
        this->reacNodes.push_back(newReaction);
        return newReaction;
    }

    bool searchTree::expandTree(moleculeNode *startMol, std::vector<std::vector<str>> &expandResults, std::vector<float> &scores){
        assert(!startMol->hasFound);
        if (scores.size() == 0){
            startMol->close();
            return false;
        }
        else{
            std::for_each(scores.begin(), scores.end(), [](float &a){
                a = std::max(std::min(1.0f, a), 1e-3f);
                a = -log(a);
            });
            std::unordered_set<str> ancestor;
            startMol->getAncestor(ancestor);
            for (int i=0; i < scores.size(); i++) this->addReaction(expandResults[i], startMol, scores[i], ancestor);

            if (startMol->children.size() == 0){
                startMol->close();
                return false;
            }
            else{
                startMol->init();
                startMol->upToDownUpdate();
                return this->root->hasFound;
            }
        }
    }

    std::pair<std::vector<std::vector<str>>, std::vector<float>> searchTree::filterRun(const str &expandSmi, const std::vector<std::unordered_map<str, float>> &expandResults, const float lowerBound, const bool consistCheck, const float checkLowerBound){
        std::vector<std::vector<str>> tempRes1;
        std::vector<str> checkInput;
        std::vector<float> tempScore1;
        auto strSplit = [](const str &s, const char split){
            std::vector<str> res;
            int i = 0, startPos = 0;
            for (; i < s.size(); i++){
                if (s[i] == split){
                    if (i > startPos) res.push_back(s.substr(startPos, i - startPos));
                    startPos = i + 1;
                }
            }
            if (i > startPos) res.push_back(s.substr(startPos, i - startPos));
            return res;
        };

        for (auto &[r, s] : expandResults[0]){
            if (s < lowerBound) continue;
            tempRes1.push_back(strSplit(r, '.'));
            tempScore1.push_back(s);
            checkInput.push_back(r);
        }

        // consistency check
        if (consistCheck && checkInput.size()){
            std::vector<std::vector<str>> tempRes2;
            std::vector<float> tempScore2;
            auto checkRes = this->inferFun(checkInput, false);
            int count = 0;
            for (auto &check : checkRes){
                auto findRes = check.find(expandSmi);
                if (findRes != check.end() && findRes->second >= checkLowerBound){
                    tempRes2.push_back(tempRes1[count]);
                    tempScore2.push_back(tempScore1[count] * findRes->second);
                }
                count++;
            }

            // if no results pass consistency check at first step, ignore it
            if (!tempScore2.size() && expandSmi == this->target){
                tempRes2.swap(tempRes1);
                tempScore2.swap(tempScore1);
            }
            if (tempRes2.size() != expandResults[0].size()){
                float accumRes = std::reduce(tempScore2.begin(), tempScore2.end());
                std::for_each(tempScore2.begin(), tempScore2.end(), [accumRes](auto &a){a /= accumRes;});
            }
            return std::make_pair(tempRes2, tempScore2);
        }
        else{
            if (tempScore1.size() != expandResults[0].size()){
                float accumRes = std::reduce(tempScore1.begin(), tempScore1.end());
                std::for_each(tempScore1.begin(), tempScore1.end(), [accumRes](auto &a){a /= accumRes;});
            }
            return std::make_pair(tempRes1, tempScore1);
        }
    }

    std::vector<float> searchTree::valueFun(const std::vector<str> &smis){
        return this->valModel->valueRun(smis);
    }

    std::vector<std::unordered_map<str, float>> searchTree::inferFun(const std::vector<str> &smis, const bool isRetro){
        std::vector<int64_t> lTask = std::vector<int64_t>(smis.size(), isRetro ? 0 : 1);
        int64_t beamSize = isRetro ? this->expansionWidth : this->checkWidth;
        auto [inferRes, inferScore] = this->inferModel->inferRun(smis, lTask, beamSize, smis.size(), 0.0, 1, this->singleSteps, 1, this->T, beamSize); // [batchSize * beamSize]

        for (auto &p : smis) this->excludeMols.insert(p);

        // canonical && filter
        std::vector<std::unordered_map<str, float>> filterRes(smis.size(), std::unordered_map<str, float>());
        for (int i=0; i < inferRes.size(); i++){
            int batchId = i / beamSize;
            auto [canoSmi, isValid] = this->inferModel->molHandler.canonicalizeSmiles(inferRes[i], false);
            if (isValid && this->excludeMols.find(canoSmi) == this->excludeMols.end() && filterRes[batchId].find(canoSmi) == filterRes[batchId].end()) filterRes[batchId][canoSmi] = inferScore[i];
        }
        for (auto &m : filterRes){
            std::for_each(m.begin(), m.end(), [](auto &p){p.second = exp(p.second);});
            float scoreSum = std::reduce(m.begin(), m.end(), 0.0f, [](float curSum, const auto &p){return curSum + p.second;});
            std::for_each(m.begin(), m.end(), [&scoreSum](auto &p){p.second /= scoreSum;});
        }
        for (auto &p : smis) this->excludeMols.erase(p);
        return filterRes;
    }

    void searchTree::visualization(const str &name){
        GVC_t *gvc = gvContext();
        Agraph_t *G = agopen("g", Agdirected, 0);

        std::queue<std::pair<std::variant<moleculeNode*, reactionNode*>, std::variant<moleculeNode*, reactionNode*>>> qNode;
        reactionNode *tempNode = nullptr;
        this->root->vizPrepare(G);
        qNode.push(std::make_pair(this->root, tempNode));

        while (!qNode.empty()){
            auto [__node, __parent] = qNode.front(); qNode.pop();
            // reactionNode -> moleculeNode
            if (std::get_if<moleculeNode*>(&__node)){
                auto node = std::get<moleculeNode*>(__node);
                auto parent = std::get<reactionNode*>(__parent);
                if (parent) auto bond = agedge(G, parent->gvnode, node->gvnode, "", 1);
                if (node->children.size() > 0){
                    for (auto &p : node->children){
                        p->vizPrepare(G);
                        qNode.push(std::make_pair(p, node));
                    }
                }
            }
            else if (std::get_if<reactionNode*>(&__node)){
                // moleculeNode -> reactionNode
                auto node = std::get<reactionNode*>(__node);
                auto parent = std::get<moleculeNode*>(__parent);
                str bondLabel = std::to_string(exp(-node->cost));
                bondLabel = bondLabel.substr(0, bondLabel.find('.') + 3);
                auto bond = agedge(G, parent->gvnode, node->gvnode, bondLabel.data(), 1);
                if (node->children.size() > 0){
                    for (auto &p : node->children){
                        p->vizPrepare(G);
                        qNode.push(std::make_pair(p, node));
                    }
                }
            }
        }
        agsafeset(G, "dpi", "600", "");
        agsafeset(G, "rankdir", "LR", "");
        gvLayout(gvc, G, "dot");
        gvRenderFilename(gvc, G, "pdf", (name + ".pdf").c_str());
        gvFreeLayout(gvc, G);
        agclose(G);
        gvFreeContext(gvc);
    }

    void searchTree::visualizationBest(const str &name){
        GVC_t *gvc = gvContext();
        Agraph_t *G = agopen("g", Agdirected, 0);

        std::queue<moleculeNode*> qNode;
        this->root->vizPrepare(G);
        qNode.push(this->root);

        while (!qNode.empty()){
            auto node = qNode.front(); qNode.pop();
            if (node->hasFound){
                reactionNode *bestReaction = nullptr;
                if (node->children.size() > 0){
                    for (auto p : node->children){
                        if (p->hasFound && (!bestReaction || p->cost < bestReaction->cost)) bestReaction = p;
                    }
                    str bondLabel = std::to_string(exp(-bestReaction->cost));
                    bondLabel = bondLabel.substr(0, bondLabel.find('.') + 3);
                    bestReaction->vizPrepare(G);
                    auto bond = agedge(G, node->gvnode, bestReaction->gvnode, bondLabel.data(), 1);

                    for (auto p : bestReaction->children){
                        p->vizPrepare(G);
                        auto chbond = agedge(G, bestReaction->gvnode, p->gvnode, "", 1);
                        qNode.push(p);
                    }
                }
            }
        }
        agsafeset(G, "dpi", "600", "");
        agsafeset(G, "rankdir", "LR", "");
        gvLayout(gvc, G, "dot");
        gvRenderFilename(gvc, G, "pdf", (name + ".pdf").c_str());
        gvFreeLayout(gvc, G);
        agclose(G);
        gvFreeContext(gvc);
    }

    bool searchTree::finishSearch(){
        this->visualization(this->savePath + this->targetName + "-Tree");
        if (this->hasFound) this->visualizationBest(this->savePath + this->targetName + "-BestRoute");
        return this->hasFound;
    }

    searchTree::~searchTree(){
        this->searchLog.close();
        for (auto p : this->molNodes) delete p;
        for (auto p : this->reacNodes) delete p;
        std::vector<moleculeNode*>().swap(this->molNodes);
        std::vector<reactionNode*>().swap(this->reacNodes);

        delete this->valModel;
        delete this->inferModel;
    }

    void loadTerminalMols(std::unordered_set<str> &finalSet, str &&molPath){
        if (molPath == "" || !std::filesystem::exists(molPath)){
            molPath = std::filesystem::current_path().parent_path();
            molPath += "/Models/origin_dict.csv";
        }
        auto loadBegin = std::chrono::high_resolution_clock::now();
        
        #if (defined _OPENMP) && (_RDPAL)
        #pragma omp parallel
        {
            std::ifstream terminalMols;
            terminalMols.open(molPath, std::ios::in);
            std::unordered_set<str> localSet;
            str line;
            std::getline(terminalMols, line);

            int tid = omp_get_thread_num();
            int tnum = omp_get_num_threads();
            
            // skip
            for (int i=0; i < tid; i++) std::getline(terminalMols, line);
            while (std::getline(terminalMols, line)){
                int start = 1, end=line.size() - 1;
                for (; start < end && line[end] != ','; end--){}
                if (start < end) localSet.insert(line.substr(start+1, end-start-1));
                // skip
                for (int i=0; i < tnum && std::getline(terminalMols, line); i++){}
            }

            // accumulate
            #pragma omp critical
            {
                finalSet.insert(localSet.begin(), localSet.end());
            }
            terminalMols.close();
        }
        #else
        std::ifstream terminalMols;
        terminalMols.open(molPath, std::ios::in);
        str line;
        std::getline(terminalMols, line);

        while (std::getline(terminalMols, line)){
            int start = 0, end=line.size() - 1;
            for (; start < end && line[start] != ','; start++){}
            for (; start < end && line[end] != ','; end--){}
            if (start < end) finalSet.insert(line.substr(start+1, end-start-1));
        }
        terminalMols.close();
        #endif

        auto loadEnd = std::chrono::high_resolution_clock::now();
        auto loadCost = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadBegin).count() * 1e-3;
        printf("load %lu molecules from %s, cost %.2f second\n", finalSet.size(), molPath.data(), loadCost);
    }
}