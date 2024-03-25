import os
import math
import time
import onnxruntime
import numpy as np

from typing import Optional
from Inference.pyPreprocess import pyMolHandler, ATOMFEATDIM, BONDFEATDIM
from Inference.huggingface_infer import Beam_Generate as Huggingface_Beam

curDir = os.path.dirname(os.path.realpath(__file__))

testDir = "/Users/sophie/Code/BiRetroSys/BiRetroSys-py/Models/50k/token(test).txt"

def onhotToTensor(src: np.ndarray, oneHotList: list[int]):
    assert sum(oneHotList) == src.shape[-1]
    
    tgt = np.full((src.shape[0], len(oneHotList)), -1)
    idx = np.where(src == 1)
    idxAccum = np.cumsum(np.array([0] + oneHotList))

    for cnt in range(idx[0].shape[0]):
        batchIdx = idx[0][cnt]
        label = idx[1][cnt]
        for i in range(len(oneHotList)):
            if label >= idxAccum[i] and label < idxAccum[i+1]:
                tgt[batchIdx][i] = label - idxAccum[i]

    return tgt

def nodePadding(src: np.ndarray, srcLength: np.ndarray):
    maxLength = srcLength.max()
    dModel = src.shape[-1]
    tgt = np.zeros((srcLength.shape[-1], maxLength, dModel), dtype=src.dtype)
    preLength = 0
    for i, length in enumerate(srcLength):
        tgt[i][:length] = src[preLength:preLength + length]
        preLength += length
    return tgt

def getMask(queryLength: int, keyLengh: int, validLength: np.ndarray):
    batchSize = validLength.shape[-1]
    masks = np.zeros((batchSize, queryLength, keyLengh), dtype=bool)
    for i, length in enumerate(validLength):
        masks[i][:length] = True
    return np.expand_dims(masks, 1)


class onnxInfer():
    def __init__(self, modelClass: Optional[str]="50k"):
        assert modelClass in ["50k", "full"]
        execution = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        onnxDir = os.path.join(os.path.dirname(curDir), "Models", modelClass)
        encoderDir = os.path.join(onnxDir, "encoder.onnx")
        extraEmbDir = os.path.join(onnxDir, "extra_embedding.onnx")
        decoderDir = os.path.join(onnxDir, "decoder.onnx")
        vocabDir = os.path.join(onnxDir, "vocabulary(uspto_50k).txt" if modelClass=="50k" else "vocabulary(uspto_full).txt")

        self.preprocesser = pyMolHandler(vdir=vocabDir)
        self.encInfer = onnxruntime.InferenceSession(encoderDir, providers=execution)
        self.extraInfer = onnxruntime.InferenceSession(extraEmbDir, providers=execution)
        self.decInfer = onnxruntime.InferenceSession(decoderDir, providers=execution)

        self.vocab = self.preprocesser.vocab
        self.rvocab = self.preprocesser.rvocab

    def __inference(
        self,
        smis: list[str], lTask: list[int],
        beamSize, returnNum, maxStep, T
    ):
        batchSize = len(smis)
        inputs = self.preprocesser.generateBatch(
            smis=smis, lTask=lTask
        )
        encInputs = {
            "atomFeat": inputs.atomFeat,
            "queryIdx": inputs.queryIdx,
            "keyIdx": inputs.keyIdx,
            "deg": inputs.deg,
            "dist": inputs.dist,
            "bondFeat0": inputs.bondFeat[0],
            "bondFeat1": inputs.bondFeat[1],
            "bondFeat2": inputs.bondFeat[2],
            "bondFeat3": inputs.bondFeat[3],
            "bondIdx0": inputs.bondIdx[0],
            "bondIdx1": inputs.bondIdx[1],
            "bondIdx2": inputs.bondIdx[2],
            "bondIdx3": inputs.bondIdx[3],
            "attnBondIdx0": inputs.attnBondIdx[0],
            "attnBondIdx1": inputs.attnBondIdx[1],
            "attnBondIdx2": inputs.attnBondIdx[2],
            "attnBondIdx3": inputs.attnBondIdx[3],
            "bondSplit": inputs.bondSplit
        }
        extraTokenInputs = {
            "Task": inputs.lTask,
            "Class": inputs.lClass
        }

        graphEmb = self.encInfer.run(None, encInputs)[0]
        graphEmb = nodePadding(graphEmb, inputs.graphLength)
        
        extraEmb = self.extraInfer.run(None, extraTokenInputs)[0]
        
        beamSearch = Huggingface_Beam(
            beam_size=beamSize,
            batch_size=batchSize,
            bos_token_ids=self.vocab['<BOS>'],
            pad_token_ids=self.vocab['<PAD>'],
            eos_token_ids=self.vocab['<EOS>'],
            vocab=self.vocab,
            rvocab=self.rvocab,
            length_penalty=0.,
            min_len=1,
            max_len=maxStep,
            beam_group=1,
            temperature=T,
            top_k=0,
            top_p=0.,
            return_num=returnNum,
            remove_finish_batch=True
        )
        graphEmb = graphEmb.repeat(beamSize, 0)
        extraEmb = extraEmb.repeat(beamSize, 0)
        msaCache = np.array([], dtype=graphEmb.dtype).reshape((8, graphEmb.shape[0], 0, graphEmb.shape[-1]))
        graphLength = inputs.graphLength.repeat(beamSize, 0)
        taskCount = inputs.lTask.repeat(beamSize, 0)
        numList = np.bincount(taskCount, minlength=2)
        contextMask = getMask(3, graphEmb.shape[1], graphLength)
        extraQ, extraK = np.array([2], dtype=np.int64), np.array([2], dtype=np.int64)
        step = np.zeros((2), dtype=np.int64)

        for i in range(maxStep):
            step[0] = i
            decInputs = {
                "tokens": beamSearch.current_token.reshape(-1, 1),
                "extraTokenEmb": extraEmb,
                "msaCache": msaCache,
                "mcaCache": graphEmb,
                "contextMask": contextMask,
                "extraQ": extraQ,
                "extraK": extraK,
                "numList": numList,
                "step": step
            }
            decOut, msaCache = self.decInfer.run(None, decInputs)
            if i == 0:
                decOut = decOut[:, 2:3]
                contextMask = contextMask[:, :, :1]
                extraEmb = np.array([], dtype=extraEmb.dtype).reshape(beamSearch.current_token.shape[-1], 0, graphEmb.shape[-1])
                extraQ[0] = 0
            beamSearch.generate(decOut)
            if beamSearch.is_done: break

            unfinishIdx = beamSearch.mem_ids
            msaCache = msaCache[:, unfinishIdx]
            graphEmb = graphEmb[unfinishIdx]
            taskCount = taskCount[unfinishIdx]
            numList = np.bincount(taskCount, minlength=2)
            contextMask = contextMask[unfinishIdx]
  
        return beamSearch.finish_generate()
    
    def Inference(
        self, 
        smis: list[str], lTask: list[int],
        beamSize: Optional[int]=20,
        returnNum: Optional[int]=10,
        maxStep: Optional[int]=150,
        T: Optional[int]=1.0,
        excludeMol: Optional[str]=""
    ):
        beamRes, beamScore = self.__inference(smis, lTask, beamSize, beamSize, maxStep, T)
        # filtering
        filterRes, filterScore = [[] for _ in range(len(smis))], [[] for _ in range(len(smis))]
        for i in range(len(beamRes)):
            for j in range(len(beamRes[0])):
                res = beamRes[i][j]
                strRes = [self.rvocab.get(res[_]) for _ in range(res.shape[-1])]
                strRes = "".join(strRes)
                canoRes, isValid = self.preprocesser.canonicalizeSmiles(strRes)
                if isValid and canoRes not in filterRes[i] and canoRes != excludeMol:
                    filterRes[i].append(canoRes)
                    filterScore[i].append(beamScore[i][j])

        for i in range(len(smis)):
            filterRes[i] = filterRes[i][:returnNum]
            filterScore[i] = filterScore[i][:returnNum]

            filterScore[i] = [math.exp(filterScore[i][j]) for j in range(len(filterScore[i]))]
            scoreSum = sum(filterScore[i])
            filterScore[i] = [filterScore[i][j] / scoreSum for j in range(len(filterScore[i]))]
        return filterRes, filterScore


if __name__ == "__main__":
    Infer = onnxInfer()
    assumeBatch = 16
    returnNum = 10
    prods, reacs = [], []
    with open(testDir, "r") as f:
        for reaction in f.readlines():
            reaction = reaction.strip("\n").split("\t")
            prods.append(reaction[0]), reacs.append(reaction[1])
    
    topnCount = [0 for _ in range(returnNum)]
    topnAcc = [0 for _ in range(returnNum)]

    processCount = math.ceil(len(prods) / assumeBatch)

    processBegin = time.perf_counter()
    for bcnt in range(processCount):
        batchBegin = time.perf_counter()
        finishCount = bcnt * assumeBatch
        batchSize = assumeBatch if len(prods) - finishCount > assumeBatch else len(prods) - finishCount

        lTask = [0 for _ in range(batchSize)]
        smis = prods[finishCount:finishCount + batchSize]
        tgtSmis = reacs[finishCount:finishCount + batchSize]

        res, score = Infer.Inference(smis, lTask)

        # strRes = [[] for _ in range(batchSize)]
        # for bszcnt in range(batchSize):
        #     batchRes = res[bszcnt]
        #     for beamcnt in range(returnNum):
        #         beamRes = batchRes[beamcnt]
        #         beamStrRes = "".join([Infer.rvocab.get(beamRes[_]) for _ in range(beamRes.shape[0])])
        #         beamStrRes = beamStrRes.split("<EOS>")[0]
        #         strRes[bszcnt].append(beamStrRes)
        
        for bszcnt in range(batchSize):
            gt = tgtSmis[bszcnt]
            batchRes = res[bszcnt]
            for beamcnt in range(returnNum):
                canoSmi = batchRes[beamcnt]
                if (canoSmi == gt):
                    for _ in range(beamcnt, returnNum):
                        topnCount[_] += 1 
                    break
        print("Batch{0}({1}) finished, spend {2:.4f} seconds, correct count:".format(bcnt, finishCount + batchSize, time.perf_counter() - batchBegin))
        print("{0}\n".format(topnCount))
    
    topnAcc = [_ / len(prods) for _ in topnAcc]
    print("All {0} reactions finish, spend {1:.4f} seconds, accuracy:".format(len(prods), time.perf_counter() - processBegin))
    print("{0}\n".format(topnAcc))
    pass