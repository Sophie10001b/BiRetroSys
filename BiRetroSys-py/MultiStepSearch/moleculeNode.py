import numpy as np

from typing import Optional

class molNode:
    def __init__(
        self, mol: str,
        value: float,
        parent=None, molId: int=0,
        isTerminal: bool=False
    ) -> None:
        self.mol = mol
        self.molId = molId
        
        self.parent = parent
        self.children = []
        self.vm = value #Vm for molecule
        self.depth = 0 if self.parent is None else self.parent.depth + 1
        self.hasFound = isTerminal
        self.isTerminal = isTerminal
        self.isOpen = not isTerminal

        if self.parent is not None:
            self.parent.children.append(self)

        self.rn = value if self.isOpen else 0
        self.vmt = None

    def update(self, propagate: bool=False):
        assert not self.isOpen
        newrn = float("inf")
        for reaction in self.children:
            newrn = min(newrn, reaction.rn)
            self.hasFound |= reaction.hasFound
        drn = newrn - self.rn
        self.rn = newrn
        
        if self.parent is not None and propagate: self.parent.update(propagate, drn)

    #for the expand node
    def initValue(self, propagate: bool=False):
        assert self.isOpen

        self.isOpen = False
        self.update(propagate)
    
    #for the expand node
    def updateVmt(self, vmt: Optional[float]=None):
        # self.vmt = vmt + self.parent.costs if self.parent is not None else vmt
        # for i in range(len(self.children)):
        #     self.children[i].updateVmt(vmt)
        self.vmt = 0
        if self.parent is None:
            self.vmt = self.rn
        else:
            self.vmt += self.parent.costs
            parent = self.parent
            while parent is not None:
                for child in parent.children: self.vmt += child.rn
                parent = parent.parent.parent
            for i in range(len(self.children)):
                self.children[i].updateVmt()
    
    def closeNode(self):
        assert self.isOpen
        self.isOpen = False
        self.rn = float("inf")
        if self.parent is not None: self.parent.update(True, self.rn)
        
    def getAncestor(self):
        if self.parent is None:
            return {self.mol}
        
        ancestor = self.parent.parent.getAncestor()
        ancestor.add(self.mol)
        return ancestor
    
    def firstReactionUpdateVmt(self):
        if self.parent is None: #root
            for child in self.children:
                child.updateVmt()
        elif self.parent.parent.parent is None: #child for 1st reaction
            self.parent.updateVmt()
        else: #others
            self.parent.parent.firstReactionUpdateVmt()

    def getStrFeature(self):
        return f"{self.molId} | {self.mol}"