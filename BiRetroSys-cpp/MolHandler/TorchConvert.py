import torch
import numpy as np
import MolHandler

from typing import Optional

class inputData:
    def __init__(
        self, atomFeat, queryIdx, keyIdx, deg, dist, lTask, lClass, graphLength,
        bondFeat: list, bondIdx: list, attnBondIdx: list, bondSplit, seqFeat=None, seqLength=None, needConvert: Optional[bool]=True, device: Optional[str]="cpu"
    ):
        self.atomFeat = torch.tensor(atomFeat, dtype=torch.float, device=device) if needConvert else atomFeat.astype(np.float32)
        self.queryIdx = torch.tensor(queryIdx.reshape(-1), dtype=torch.long, device=device) if needConvert else queryIdx.reshape(-1).astype(np.int64)
        self.keyIdx = torch.tensor(keyIdx.reshape(-1), dtype=torch.long, device=device) if needConvert else keyIdx.reshape(-1).astype(np.int64)
        self.deg = torch.tensor(deg.reshape(-1), dtype=torch.long, device=device) if needConvert else deg.reshape(-1).astype(np.int64)
        self.dist = torch.tensor(dist.reshape(-1), dtype=torch.long, device=device) if needConvert else dist.reshape(-1).astype(np.int64)
        self.lTask = torch.tensor(lTask.reshape(-1), dtype=torch.long, device=device) if needConvert else lTask.reshape(-1).astype(np.int64)
        self.lClass = torch.tensor(lClass.reshape(-1), dtype=torch.long, device=device) if needConvert else lClass.reshape(-1).astype(np.int64)
        self.graphLength = torch.tensor(graphLength.reshape(-1), dtype=torch.long, device=device) if needConvert else graphLength.reshape(-1).astype(np.int64)

        self.bondFeat = []
        for i, _ in enumerate(bondFeat):
            if needConvert:
                self.bondFeat.append(torch.tensor(_, dtype=torch.float, device=device) if i == 0 else torch.tensor(_.reshape(-1), dtype=torch.long, device=device))
            else:
                self.bondFeat.append(_.astype(np.float32) if i == 0 else _.reshape(-1).astype(np.int64))
        self.bondIdx = [torch.tensor(_, dtype=torch.long, device=device) for _ in bondIdx] if needConvert else [_.astype(np.int64) for _ in bondIdx]
        self.attnBondIdx = [torch.tensor(_.reshape(-1), dtype=torch.long, device=device) for _ in attnBondIdx] if needConvert else [_.reshape(-1).astype(np.int64) for _ in attnBondIdx]

        self.bondSplit = torch.tensor(bondSplit.reshape(-1), dtype=torch.long, device=device) if needConvert else bondSplit.reshape(-1).astype(np.int64)

        if seqFeat.size > 0:
            self.seqFeat = torch.tensor(seqFeat, dtype=torch.long, device=device) if needConvert else seqFeat.astype(np.int64)
            self.seqLength = torch.tensor(seqLength.reshape(-1), dtype=torch.long, device=device) if needConvert else seqLength.reshape(-1).astype(np.int64)
        
        self.atomDim = MolHandler.atomDim
        self.bondDim = MolHandler.bondDim


class pyMolHandler:
    def __init__(
        self, vdir: str, extraToken: Optional[list[str]]=["<BOS>", "<EOS>", "<PAD>", "<UNK>"], maxDeg: Optional[int]=9, maxK: Optional[int]=4, maxPath: Optional[int]=15
    ):
        self.molPreprocess = MolHandler.molPreprocess(vdir, maxDeg, maxK, maxPath)
        self.vocab = self.molPreprocess.vocab
    
    def generateBatch(
        self, smis: list[str], lTasks: list[int], needSeq: Optional[bool]=False,
        needConvert: Optional[bool]=True,
        device: Optional[str]="cpu"
    ) -> inputData:
        batch = self.molPreprocess.generateBatch(smis, lTasks, needSeq)
        return inputData(
            batch.atomFeat, batch.queryIdx, batch.keyIdx, batch.deg, batch.dist, batch.lTask, batch.lClass, batch.graphLength, batch.bondFeat, batch.bondIdx, batch.attnBondIdx, batch.bondSplit, batch.seqFeat, batch.seqLength, needConvert, device
        )
    
    def canonicalize(self, smi: str):
        canoSmi, isValid = self.molPreprocess.canonicalizeSmiles(smi)
        return canoSmi if isValid else smi, isValid
    
if __name__ == "__main__":
    handler = pyMolHandler(
        vdir="/Users/sophie/Code/BiRetroSys/Preprocess/vocabulary(uspto_full).txt"
    )
    batch = handler.generateBatch(
        smis=["[CH3:1][C:2]([N:4]([C:9](=[O:10])[CH3:8])[CH2:5][CH2:6][NH2:7])=[O:3]", "CCC.CCCC"],
        lTasks=[0, 1],
        device="mps"
    )
    pass