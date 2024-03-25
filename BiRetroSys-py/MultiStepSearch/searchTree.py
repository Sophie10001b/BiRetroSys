import os
import logging
import math
import numpy as np

from tqdm.std import trange
from MultiStepSearch.moleculeNode import molNode
from MultiStepSearch.reactionNode import reactionNode
from queue import Queue
from graphviz import Digraph

curDir = os.path.dirname(os.path.realpath(__file__))
saveDir = os.path.join(curDir, "SearchResults")

class molTree:
    def __init__(self, targrtMol: str, terminalMol: list[str], valueFun, inferFun, checkFun)-> None:
        self.targetMol = targrtMol
        self.terminalMol = terminalMol
        self.valueFun = valueFun
        self.inferFun = inferFun
        self.checkFun = checkFun

        self.molNodes = []
        self.reactionNodes = []
        self.hasFound = targrtMol in terminalMol

        if self.hasFound:
            logging.info("Target Molecule already in terminal Molecules.")
        
        self.root = self.__addMol(targrtMol, None, list(self.valueFun([targrtMol]))[0])

    def __addMol(self, mol: str, parent: reactionNode, value: float):
        hasFound = mol in self.terminalMol

        mNode = molNode(
            mol=mol,
            value=value,
            parent=parent,
            molId=len(self.molNodes),
            isTerminal=hasFound
        )
        self.molNodes.append(mNode)
        return mNode
    
    def __addReaction(self, reaction: list[str], parent: molNode, cost: float, ancestor: set):
        for mol in reaction:
            if mol in ancestor: return None
        
        values = list(self.valueFun(reaction))
        rNode = reactionNode(cost, parent, len(self.reactionNodes))
        for mol, value in zip(reaction, values):
            self.__addMol(mol, rNode, value)
        
        rNode.initValue()
        self.reactionNodes.append(rNode)
        return rNode

    def __expand(self, startMol: molNode, smiResult: list[list[str]], scores: list[float]):
        assert not startMol.hasFound and len(startMol.children) == 0

        if len(scores) == 0:
            startMol.closeNode()
            return False
        
        scores = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
        scores = list(scores)
        for i in range(len(scores)):
            self.__addReaction(smiResult[i], startMol, scores[i], startMol.getAncestor())
        
        if len(startMol.children) == 0:
            startMol.closeNode()
            return False
        
        startMol.initValue(propagate=True)
        startMol.firstReactionUpdateVmt()
        return self.root.hasFound
    
    #graphviz visualization with BFS
    def __visualization(self, name: str):
        G = Digraph("searchTree", filename=os.path.join(saveDir, name))
        G.attr(rankdir="LR")
        G.attr("node", shape="box")
        G.format = "pdf"

        qNode = Queue()
        qNode.put((self.root, None)) #node & parent
        while not qNode.empty():
            node, parent = qNode.get()
            color = "lightgrey" if node.isOpen else "aquamarine"
            shape = "box" if isinstance(node, molNode) else "rarrow"

            if isinstance(node, molNode) and node.isTerminal:
                color = "lightyellow"
            elif node.hasFound:
                color = "lightblue"
            
            G.node(node.getStrFeature(), shape=shape, color=color, style="filled")

            label = "{:.4f}".format(math.exp(-node.cost)) if isinstance(node, reactionNode) else ""
            if parent is not None:
                G.edge(parent.getStrFeature(), node.getStrFeature(), label=label)
            if len(node.children) > 0:
                for child in node.children:
                    qNode.put((child, node))
        G.render()
    
    def __visualizationBest(self, name: str):
        G = Digraph("searchTree", filename=os.path.join(saveDir, name))
        G.attr(rankdir="LR")
        G.attr("node", shape="box")
        G.format = "pdf"

        color, terminalColor = "lightblue", "lightyellow"
        molShape, reactionShape = "box", "rarrow"

        qNode = Queue()
        qNode.put(self.root) #node & parent
        G.node(self.root.getStrFeature(), shape=molShape, color=terminalColor if self.root.isTerminal else color, style="filled")
        
        while True:
            mol = qNode.get()
            if mol.hasFound:
                bestReaction = None
                if len(mol.children) > 0:
                    for reaction in mol.children:
                        if (reaction.hasFound) and (bestReaction is None or reaction.cost < bestReaction.cost):
                            bestReaction = reaction

                    label = "{:.4f}".format(math.exp(-bestReaction.cost))
                    G.node(bestReaction.getStrFeature(), shape=reactionShape, color=color, style="filled")
                    G.edge(mol.getStrFeature(), bestReaction.getStrFeature(), label=label)
                
                    for nextmol in bestReaction.children:
                        qNode.put(nextmol)
                        G.node(nextmol.getStrFeature(), shape=molShape, color=terminalColor if nextmol.isTerminal else color, style="filled")
                        G.edge(bestReaction.getStrFeature(), nextmol.getStrFeature(), label="")

            if qNode.empty(): break
        G.render()
    
    def __finish(self, name: str="Tree", bestname: str="BestRoute"):
        if not os.path.exists(saveDir): os.makedirs(saveDir)

        self.__visualization(name)
        if self.hasFound:
            self.__visualizationBest(bestname)

        return self.hasFound

    def search(self, steps: int, name: str="Tree", bestName: str="bestRoute", earlyStop: int=0, lowerbound: float=0.1, consistCheck: bool=True, checkLowerBound: float=0.01):
        if not self.hasFound:
            tqdmSteps = trange(steps)
            for step in tqdmSteps:
                scores = []
                for i in range(len(self.molNodes)):
                    scores.append(self.molNodes[i].vmt) if self.molNodes[i].isOpen else scores.append(float("inf"))
                
                if min(scores) == float("inf"):
                    logging.info("No open nodes!")
                    break

                nextNodeIdx = scores.index(min(scores))
                nextNode = self.molNodes[nextNodeIdx]

                logging.info(f"Step{step}: Trying to expand {nextNode.mol}")
                assert nextNode.isOpen

                inferRes, inferScore = self.inferFun([nextNode.mol], nextNode.mol)

                filterMap = ("CC", "")

                #hard-coded filtering
                #1. filter invalid results and rescale
                filterRes, filterScore = [], []
                for smiRes, smiScore in zip(inferRes[0], inferScore[0]):
                    if smiRes in filterMap: continue
                    filterRes.append(smiRes)
                    filterScore.append(smiScore)
                scoreSum = sum(filterScore)
                filterScore = [_ / scoreSum for _ in filterScore]

                #2. split reactants & lowerbound
                splitRes, splitScore, checkInput = [], [], []
                for smiRes, smiScore in zip(filterRes, filterScore):
                    reactantSplit = smiRes.split(".")
                    if smiScore < lowerbound: continue

                    splitRes.append(reactantSplit)
                    splitScore.append(smiScore)
                    checkInput.append(smiRes)
                
                #3. consistency check
                if consistCheck and len(checkInput) > 0:
                    consistRes, consistScore = [], []
                    checkRes, checkScore = self.checkFun(checkInput, "")
                    for i, (res, score) in enumerate(zip(checkRes, checkScore)):
                        if nextNode.mol in res and score[res.index(nextNode.mol)] >= checkLowerBound:
                            idx = res.index(nextNode.mol)
                            consistRes.append(splitRes[i])
                            consistScore.append(splitScore[i] * score[idx])
                
                    #rescale
                    if len(consistScore) == 0 and nextNode == self.root:
                        consistRes = splitRes
                        consistScore = splitScore
                        logging.info(f"{nextNode.mol} no results pass consistency check, now trying more flexible settings")

                    if len(consistScore) != len(inferScore[0]):
                        scoreSum = sum(consistScore)
                        consistScore = [score / scoreSum for score in consistScore]
                    
                    hasFound = self.__expand(nextNode, consistRes, consistScore)
                
                else:
                    #rescale
                    if len(splitScore) != len(inferScore[0]):
                        scoreSum = sum(splitScore)
                        splitScore = [score / scoreSum for score in splitScore]
                    
                    hasFound = self.__expand(nextNode, splitRes, splitScore)

                if hasFound and not self.hasFound:
                    self.hasFound = hasFound
                    logging.info("Route has found!")
                
                if self.hasFound and step + 1 > earlyStop:
                    break
                
        return self.__finish(name, bestName), step + 1