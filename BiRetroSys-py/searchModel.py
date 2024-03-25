import os

from typing import Optional
from Inference.onnxInference import onnxInfer
from MultiStepSearch.valueFunction import valueModel
from MultiStepSearch.searchTree import molTree
from MultiStepSearch.searchUtils import setLogger, loadTerminalMols

curDir = os.path.dirname(os.path.realpath(__file__))

class retroSearch:
    def __init__(
        self,
        modelClass: Optional[str]="full"
    ) -> None:
        self.valueModel = valueModel(2048)
        self.inferModel = onnxInfer(modelClass)

        setLogger()
        self.terminalMols = loadTerminalMols()
    
    def multiStepRetroSearch(self, targetMol: str, expansionWidth: int=20, checkWidth: int=20, singleSteps: int=150, multiSteps: int=100, T: float=1.0):
        valueFun = lambda x: self.valueModel.getValue(x) # list[str]
        inferFun = lambda smi, excludeMol: self.inferModel.Inference(smi, [0 for _ in range(len(smi))], expansionWidth, expansionWidth, singleSteps, T, excludeMol) #return list[list], batch*beam
        checkFun = lambda smi, excludeMol: self.inferModel.Inference(smi, [1 for _ in range(len(smi))], checkWidth, checkWidth, singleSteps, T, excludeMol) #return list[list], batch*beam

        searchTree = molTree(targetMol, self.terminalMols, valueFun, inferFun, checkFun)
        res = searchTree.search(multiSteps, lowerbound=0.1, earlyStop=multiSteps)
        pass

if __name__ == "__main__":
    search = retroSearch()
    # example_smi = "CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C"

    example_smi = "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C"
    # example_smi = "C1CC(=O)NC(=O)C1N2CC3=C(C2=O)C=CC=C3N"
    search.multiStepRetroSearch(example_smi, expansionWidth=20, checkWidth=20, T=1.0)
    pass