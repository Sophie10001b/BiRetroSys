import os
import time
import math
import numpy as np
import pickle
import logging

from MultiStepSearch.searchUtils import setLogger, loadTerminalMols
from Inference.onnxInference import onnxInfer
from MultiStepSearch.valueFunction import valueModel
from MultiStepSearch.searchTree import molTree

curDir = os.path.dirname(os.path.realpath(__file__))

def datasetTest(
    name: str="test",
    expansionWidth: int=10,
    checkWidth: int=10, 
    singleSteps: int=150, 
    multiSteps: int=50, 
    T: float=1.0,
    consistCheck: bool=True
):
    setLogger(name=name + ".log")
    terminalMols = loadTerminalMols()

    valModel = valueModel(2048)
    inferModel = onnxInfer()

    valueFun = lambda x: valModel.getValue(x) # list[str]

    inferFun = lambda smi, excludeMol: inferModel.Inference(smi, [0 for _ in range(len(smi))], expansionWidth, expansionWidth, singleSteps, T, excludeMol) #return list[list], batch*beam

    checkFun = lambda smi, excludeMol: inferModel.Inference(smi, [1 for _ in range(len(smi))], checkWidth, checkWidth, singleSteps, T, excludeMol) #return list[list], batch*beam

    routes = pickle.load(open(os.path.join(curDir, "Models", "routes_possible_test_hard.pkl"), "rb"))

    totalCount = len(routes)
    successCount = 0
    costs = 0
    costTimes = 0

    for i, route in enumerate(routes):
        start = time.perf_counter()
        targetMol = route[0].split(">")[0]
        searchTree = molTree(targetMol, terminalMols, valueFun, inferFun, checkFun)

        res, cost = searchTree.search(multiSteps, f"Tree{i}", f"bestRoute{i}", 0, lowerbound=0.01, consistCheck=consistCheck)
        costTime = time.perf_counter() - start
        
        if res:
            successCount += 1
            costs += cost
            costTimes += costTime
        
        logging.info("{0} | {1} | {2}, {3} search {4}.\n".format(successCount, i + 1, totalCount, targetMol, "successed" if res else "failed"))
    
    logging.info("{0} planning finish, success: {1} | {2:.6f}%, lengths: {3:.6f}, times: {4:.6f}s/mol.".format(totalCount, successCount, (successCount / totalCount) * 100, costs / totalCount, costTimes / totalCount))

if __name__ == "__main__":
    datasetTest(
        expansionWidth=10,
        checkWidth=10,
        multiSteps=100,
        T=1.0,
        consistCheck=True,
    )