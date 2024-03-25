import copy
import torch
import numpy as np

from rdkit import Chem

# basic
MAXK = 4
MAXDEG = 9
MAXPATH = 15
DISTBLOCK = [[0], [1], [2], [3], [4], [5], [6], [7], [8, 15], [15, 2048]]

SMIREGEX = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

# atom
ATOMLIST = ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "W", "Ru", "Nb", "Re", "Te", "Rh", "Ta", "Tc", "Ba", "Bi", "Hf", "Mo", "U", "Sm", "Os", "Ir", "Ce", "Gd", "Ga", "Cs", "<unk>"]
atomMap = {atom:i for (i, atom) in enumerate(ATOMLIST)}

HYBRIDLIST = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
hybridMap = {feat:i for (i, feat) in enumerate(HYBRIDLIST)}

CHIRALIST = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
              Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
chiralMap = {feat:i for (i, feat) in enumerate(CHIRALIST)}

RSLIST = ["R", "S", "None"]
rsMap = {feat:i for (i, feat) in enumerate(RSLIST)}

NDEGREE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

NCHARGE = [-1, -2, 1, 2, 0]
chargeMap = {feat:i for (i, feat) in enumerate(NCHARGE)}

NVALENCE = [0, 1, 2, 3, 4, 5, 6]
valenceMap = {feat:i for (i, feat) in enumerate(NVALENCE)}

NHS = [0, 1, 3, 4, 5]
hsMap = {feat:i for (i, feat) in enumerate(NHS)}

ATOMFEATNUM = 13
ATOMFEATDIM = [len(ATOMLIST), len(NDEGREE), len(NCHARGE), len(NVALENCE), len(NHS), len(CHIRALIST), len(RSLIST), len(HYBRIDLIST), 2, 10, 2, 2, 10]

def getAtomFeat(atom, lTask=0, lClass=-1, lRoot=0, lShuffle=0):
    feat = [-1 for _ in range(ATOMFEATNUM)]
    feat[0] = atomMap.get(atom.GetSymbol(), atomMap.get("<unk>"))

    if feat[0] != atomMap.get("<unk>"):
        feat[1] = atom.GetDegree() if atom.GetDegree() in NDEGREE else 9
        feat[2] = chargeMap.get(atom.GetFormalCharge(), 4)
        feat[3] = valenceMap.get(atom.GetTotalValence(), 6)
        feat[4] = hsMap.get(atom.GetTotalNumHs(), 4)
        feat[5] = chiralMap.get(atom.GetChiralTag(), 2)
        feat[6] = rsMap.get(atom.GetPropsAsDict().get('_CIPCode', 'None'), 2)
        feat[7] = hybridMap.get(atom.GetHybridization(), 4)
        feat[8] = int(atom.GetIsAromatic())
        feat[9] = lRoot
        feat[10] = lShuffle
        feat[11] = lTask
        feat[12] = lClass
    
    oneHot = np.zeros((sum(ATOMFEATDIM)), np.float32)
    dimAccum = 0
    for i, featDim in enumerate(ATOMFEATDIM):
        if (feat[i] >= 0):
            oneHot[dimAccum + feat[i]] = 1
        dimAccum += featDim
    return oneHot.reshape((1, -1))

# bond
BONDLIST = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC,]
bondMap = {bond:i for (i, bond) in enumerate(BONDLIST)}

STEREOLIST = [Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ]
stereoMap = {feat:i for (i, feat) in enumerate(STEREOLIST)}

BONDFEATNUM = 4
BONDFEATDIM = [len(BONDLIST), len(STEREOLIST), 2, 2]

def getBondFeat(bond):
    feat = [-1 for _ in range(BONDFEATNUM)]
    if (bond.GetBondType() in BONDLIST):
        feat[0] = bondMap.get(bond.GetBondType())
        feat[1] = stereoMap.get(bond.GetStereo(), 0)
        feat[2] = int(bond.GetIsConjugated())
        feat[3] = int(bond.IsInRing())
    
    oneHot = np.zeros((sum(BONDFEATDIM)), np.float32)
    dimAccum = 0
    for i, featDim in enumerate(BONDFEATDIM):
        if (feat[i] >= 0):
            oneHot[dimAccum + feat[i]] = 1
        dimAccum += featDim
    return oneHot.reshape((1, -1))


class batchData:
    def __init__(self, num: int, k: int=MAXK) -> None:
        self.atomFeat = None
        self.bondFeat = [[] for _ in range(k)]
        self.bondIdx = [[] for _ in range(k)]
        self.attnBondIdx = [[] for _ in range(k)]
        self.queryIdx = None
        self.keyIdx = None
        self.deg = None
        self.dist = None
        self.lTask = np.full((num), -1, np.int64)
        self.lClass = np.full((num), -1, np.int64)
        self.graphLength = np.zeros((num), np.int64)
        self.bondSplit = np.zeros((k), np.int64)
    
    def convertTensor(self, device: str="cpu"):
        self.atomFeat = torch.tensor(self.atomFeat, dtype=torch.float, device=device)
        self.queryIdx = torch.tensor(self.queryIdx, dtype=torch.long, device=device)
        self.keyIdx = torch.tensor(self.keyIdx, dtype=torch.long, device=device)
        self.deg = torch.tensor(self.deg, dtype=torch.long, device=device)
        self.dist = torch.tensor(self.dist, dtype=torch.long, device=device)
        self.lTask = torch.tensor(self.lTask, dtype=torch.long, device=device)
        self.lClass = torch.tensor(self.lClass, dtype=torch.long, device=device)
        self.graphLength = torch.tensor(self.graphLength, dtype=torch.long, device=device)
        self.bondSplit = torch.tensor(self.bondSplit, dtype=torch.long, device=device)

        for i in range(len(self.bondFeat)):
            self.bondFeat[i] = torch.tensor(self.bondFeat[i], dtype=torch.float if i == 0 else torch.long, device=device)
            self.bondIdx[i] = torch.tensor(self.bondIdx[i], dtype=torch.long, device=device)
            self.attnBondIdx[i] = torch.tensor(self.attnBondIdx[i], dtype=torch.long, device=device)
        return self


class pyMolHandler:
    def __init__(self, vdir: str) -> None:
        self.vocab, self.rvocab = self.getVocabulary(vdir)
    
    def canonicalizeSmiles(self, smi: str):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            cano_smi = smi
            valid = False
        else:
            valid = True
            # if mol.GetNumHeavyAtoms() < 2:
            #     cano_smi = 'CC'
            for a in mol.GetAtoms(): a.ClearProp('molAtomMapNumber')

            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        return cano_smi, valid

    def getVocabulary(self, vDir: str, extraToken: list[str] = ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]):
        vocab, rVocab = {}, {}
        with open(vDir, "r") as f:
            for id, tokens in enumerate(f.readlines()):
                token, _ = tokens.strip("\n").split("\t")
                vocab[token] = id
                rVocab[id] = token

        for extra in extraToken:
            vocab[extra] = len(vocab)
            rVocab[len(rVocab)] = extra
        return vocab, rVocab

    def getAttentionBias(self, mol):
        adjMatrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        spdMatrix = Chem.rdmolops.GetDistanceMatrix(mol)

        degMat = adjMatrix.sum(0).astype(np.int64)
        degMat[degMat > MAXDEG] = MAXDEG

        distMat = np.zeros(spdMatrix.shape, np.int64)
        for dist, blk in enumerate(DISTBLOCK):
            start = blk[0]
            if len(blk) == 1:
                distMat[spdMatrix == dist] = dist
            elif len(blk) > 1:
                end = blk[1]
                distMat[np.logical_and((spdMatrix >= start), (spdMatrix < end))] = dist
        distMat[np.logical_or((spdMatrix < DISTBLOCK[0][0]), (spdMatrix >= DISTBLOCK[-1][-1]))] = len(DISTBLOCK)
        return degMat, distMat.reshape(-1)

    def getKhopFeat(self, mol, k=MAXK):
        kAdjMat = []
        kBondIdx = [None for _ in range(k-1)]
        kBondFeat = [None for _ in range(k-1)]

        adjMatrix = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(np.int64)
        kAdjMat.append(copy.deepcopy(adjMatrix))
        for _ in range(k - 1):
            kAdjMat.append(kAdjMat[-1] @ adjMatrix)
        for i in range(len(kAdjMat)):
            np.fill_diagonal(kAdjMat[i], 0)
        
        lastPath = copy.deepcopy(kAdjMat[0])
        for i in range(1, len(kAdjMat)):
            kAdjMat[i][lastPath > 0] = 0
            lastPath += kAdjMat[i]
        
        for i in range(1, len(kAdjMat)):
            iBondIdx = np.array(np.where(kAdjMat[i] > 0), np.int64)
            iBondFeat = kAdjMat[i][list(iBondIdx[0]), list(iBondIdx[1])]
            iBondFeat[iBondFeat > MAXPATH] = MAXPATH
            kBondIdx[i-1] = iBondIdx
            kBondFeat[i-1] = iBondFeat
        return kBondIdx, kBondFeat

    def getQKVIndex(self, preLength, curLength):
        tempMat = np.arange(preLength, preLength + curLength, dtype=np.int64)
        qIdx = tempMat.repeat(curLength, 0)
        kvIdx = np.concatenate([tempMat for _ in range(curLength)], 0)
        return qIdx, kvIdx

    def getBatchBond(
        self,
        bData: batchData,
        curBondFeat: list[np.ndarray],
        curBondIdx: list[np.ndarray],
        preCumLength: int, preLength: int, curLength: int, k: int=MAXK
    ):
        for i in range(k):
            if not isinstance(curBondFeat[i], np.ndarray): continue
            feat = curBondFeat[i]
            idx = curBondIdx[i]
            if feat.shape[0] > 0:
                idx += preLength
                attnIdx = (idx[0] - preLength) * curLength + (idx[1] - preLength) + preCumLength
                bData.bondIdx[i] = np.concatenate([bData.bondIdx[i], idx], -1) if isinstance(bData.bondIdx[i], np.ndarray) else idx
                bData.bondFeat[i] = np.concatenate([bData.bondFeat[i], feat], 0) if isinstance(bData.bondFeat[i], np.ndarray) else feat
                bData.attnBondIdx[i] = np.concatenate([bData.attnBondIdx[i], attnIdx], -1) if isinstance(bData.attnBondIdx[i], np.ndarray) else attnIdx


    def generateGraph(
        self,
        smi: str, bData: batchData,
        preCumLength: int, preLength: int, batchId: int,
        lTask: int, lClass: int, lRoot: int, lShuffle: int
    ):
        mol = Chem.MolFromSmiles(smi)
        atomNum = mol.GetNumAtoms()

        atomFeat = np.ndarray((0, sum(ATOMFEATDIM)), np.float32)
        bondFeat = np.ndarray((0, sum(BONDFEATDIM)), np.float32)
        bondIdx = [[], []]

        for atom in mol.GetAtoms():
            atomFeat = np.concatenate([atomFeat, getAtomFeat(atom, lTask, lClass, lRoot, lShuffle)], 0)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bondIdx[0].extend([start, end])
            bondIdx[1].extend([end, start])
            tempBondFeat = getBondFeat(bond)
            bondFeat = np.concatenate([bondFeat, tempBondFeat, tempBondFeat], 0)
        
        bondIdx = np.array(bondIdx, np.int64)
        sortIdx = (bondIdx[0] * atomNum + bondIdx[1]).argsort()
        bondIdx = bondIdx[:, sortIdx]
        bondFeat = bondFeat[sortIdx]

        degMat, distMat = self.getAttentionBias(mol)
        kBondIdx, kBondFeat = self.getKhopFeat(mol)
        qIdx, kvIdx = self.getQKVIndex(preLength, atomNum)
        kBondIdx.insert(0, bondIdx)
        kBondFeat.insert(0, bondFeat)

        # generate batch
        bData.atomFeat = np.concatenate([bData.atomFeat, atomFeat], 0) if isinstance(bData.atomFeat, np.ndarray) else atomFeat
        bData.deg = np.concatenate([bData.deg, degMat], -1) if isinstance(bData.deg, np.ndarray) else degMat
        bData.dist = np.concatenate([bData.dist, distMat], -1) if isinstance(bData.dist, np.ndarray) else distMat
        bData.queryIdx = np.concatenate([bData.queryIdx, qIdx], -1) if isinstance(bData.queryIdx, np.ndarray) else qIdx
        bData.keyIdx = np.concatenate([bData.keyIdx, kvIdx], -1) if isinstance(bData.keyIdx, np.ndarray) else kvIdx

        self.getBatchBond(bData, kBondFeat, kBondIdx, preCumLength, preLength, atomNum)

        bData.graphLength[batchId] = atomNum
        preCumLength += atomNum ** 2
        preLength += atomNum
        return preCumLength, preLength

    def generateBatch(self, smis: list[str], lTask: list[int]):
        preCumLength, preLength = 0, 0
        bData = batchData(len(lTask))

        for id, smi in enumerate(smis):
            canoSmi, isValid = self.canonicalizeSmiles(smi)
            if not isValid: canoSmi = "CC"
            preCumLength, preLength = self.generateGraph(canoSmi, bData, preCumLength, preLength, id, lTask[id], -1, 0, 0)

        bondSplit = []
        for _ in bData.bondFeat:
            bondSplit.append(_.shape[0] if isinstance(_, np.ndarray) else 0)
        bData.bondSplit = np.array(bondSplit, np.int64)
        bData.lTask = np.array(lTask, np.int64)
        bData.lClass = np.zeros((len(lTask)), np.int64)

        #check
        if isinstance(bData.bondIdx[0], list):
            bData.bondFeat[0] = np.array(bData.bondFeat[0], np.float32).reshape(0, sum(BONDFEATDIM))
            bData.bondIdx[0] = np.array(bData.bondIdx[0], np.int64).reshape(2, 0)
            bData.attnBondIdx[0] = np.array(bData.attnBondIdx[0], np.int64).reshape(0)

        for i in range(1, len(bData.bondFeat)):
            if isinstance(bData.bondIdx[i], list):
                bData.bondFeat[i] = np.array(bData.bondFeat[i], np.int64).reshape(0)
                bData.bondIdx[i] = np.array(bData.bondIdx[i], np.int64).reshape(2, 0)
                bData.attnBondIdx[i] = np.array(bData.attnBondIdx[i], np.int64).reshape(0)
        return bData


if __name__ == "__main__":
    smis = ["CC"]
    lTask = [0]
    handler = pyMolHandler("/Users/sophie/Code/BiRetroSys/BiRetroSys-py/Models/full/vocabulary(uspto_full).txt")
    data = handler.generateBatch(smis, lTask)
    pass