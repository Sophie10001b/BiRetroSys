# BiRetroSys

## Working in Progress
The BiRetroSys includes **1.Preprocess**, **2.Single-Step Template-Free Prediction**, **3.Beam Search**, and **4.Multi-Step Retrosynthesis Search** modules. Specifically, the single-step model is a dual-task SeqAGraph(https://github.com/AILBC/BiG2S) trained on USPTO-full, and the multi-step search is a custom Retro*(https://github.com/binghong-ml/retro_star) with a dynamic molecule filter set, a lower-bound beam search score, and a consistency checking mechanism based on forward synthesis.

The Single-Step autoregressive model is fully implemented by ONNX and ONNXRuntime(ORT), which means it will have a noticeable speed-up compared with naive PyTorch implementation in CPU (**about 1 sec. per molecule per step**).

## How To Start(Python)
1. Make a copy of the source code, and download the molecule set, encoder ONNX file, and decoder ONNX file from the link below, then put them into the `/Models/`(for molecule set) and `/Models/full/`(for ONNX models).
https://drive.google.com/drive/folders/1R5I4Yb1Ss8hBpgg7H4jLzpM1g719doJE?usp=drive_link
2. Run`searchModel.py`, your target molecule is "example_smi".
3. Find your results and the corresponding GraphViz file in `/MultiStepSearch/SearchResults`.

### To Do Lists
1. C++ version of BiRetrosys. Now we are working on the final Search Tree Class and visualization with GraphViz libraries.
2. A simple interface of BiRetroSys.
3. An installation channel, maybe pip or homebrew?
