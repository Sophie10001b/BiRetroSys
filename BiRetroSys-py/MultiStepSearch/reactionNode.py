import numpy as np

from typing import Optional

class reactionNode:
    def __init__(
        self,
        cost: float, parent,
        reactionId: int=0,
        isTerminal: bool=False
    ) -> None:
        self.parent = parent
        self.depth = 0 if self.parent is None else self.parent.depth + 1
        #function G in A*
        self.cost = cost
        self.costs = None
        #function H in A*
        self.rn = None
        self.reactionId = reactionId
        self.hasFound = isTerminal
        self.isOpen = True
        self.children = []
        self.parent.children.append(self)
    
    def __calculateCost(self):
        return self.cost if self.parent.parent is None else self.cost + self.parent.parent.__calculateCost()
    
    #only for the 1st reaction layer, up to down
    def updateVmt(self, vmt: Optional[float]=None):
        for i in range(len(self.children)):
            if vmt is None:
                self.children[i].updateVmt(self.rn)
            else:
                self.children[i].updateVmt(vmt)
    
    def update(self, propagate: bool=False, drn: float=0):
        assert not self.isOpen
        self.rn += drn
        self.hasFound = True
        for mol in self.children:
            self.hasFound &= mol.hasFound
        
        if propagate: self.parent.update(propagate)

    def initValue(self):
        assert self.isOpen

        self.isOpen = False
        self.rn = 0
        self.hasFound = True
        for mol in self.children:
            self.rn += mol.rn
            self.hasFound &= mol.hasFound
        self.costs = self.__calculateCost()
    
    def getStrFeature(self):
        return f"{self.reactionId}"