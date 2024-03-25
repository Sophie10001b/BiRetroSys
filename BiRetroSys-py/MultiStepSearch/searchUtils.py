import os
import logging
import pandas as pd

from typing import Optional

curDir = os.path.dirname(os.path.realpath(__file__))

def setLogger(name: str=None):
    if name is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(name)-4s %(asctime)-1s: %(levelname)-4s %(message)s",
            datefmt="%m-%d %H:%M:%S",
            filemode="w"
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(name)-4s %(asctime)-1s: %(levelname)-4s %(message)s",
            datefmt="%m-%d %H:%M:%S",
            filename=name,
            filemode="w"
        )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-4s %(asctime)-1s: %(levelname)-4s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def loadTerminalMols(terminalMolDir:Optional[str]=None):
    terminalMols = []
    if terminalMolDir is None:
        terminalMolDir = os.path.join(os.path.dirname(curDir), "Models", "origin_dict.csv")

    for chunk in pd.read_csv(terminalMolDir, chunksize=1e8):
        terminalMols.extend(list(chunk["mol"]))
    terminalMols = set(terminalMols)
    logging.info("{0} terminal molecules have been loaded.".format(len(terminalMols)))
    
    return terminalMols