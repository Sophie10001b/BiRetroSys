import os
import numpy as np
import onnxruntime

from rdkit import Chem
from rdkit.Chem import AllChem

curDir = os.path.dirname(os.path.realpath(__file__))
onnxDir = os.path.join(os.path.dirname(curDir), "Models", "valueMLP.onnx")

class valueModel:
    def __init__(self, dFP=2048) -> None:
        execution = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.model = onnxruntime.InferenceSession(onnxDir, providers=execution)
        self.dFP = dFP
    
    def getValue(self, smis: list[str]):
        fps = []
        for smi in smis:
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=self.dFP)

            onbits = list(fp.GetOnBits())
            arr = np.zeros(fp.GetNumBits(), dtype=bool)
            arr[onbits] = True
            fps.append(arr)
        fps = np.array(fps, dtype=np.float32)

        return self.model.run(None, {"molFP": fps})[0].reshape(-1)

if __name__ == "__main__":
    smis = ["CCC", "CC(=O)c1ccc2[nH]ccc2c1.CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "CC"]
    
    valModel = valueModel()
    res = valModel.getValue(smis)
    pass