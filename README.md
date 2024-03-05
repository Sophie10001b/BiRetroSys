# BiRetroSys

## Working in Progress
The BiRetroSys includes **1.Preprocess**, **2.Single-Step Template-Free Prediction**, **3.Beam Search**, and **4.Multi-Step Retrosynthesis Search** modules. Specifically, the single-step model is a dual-task SeqAGraph trained on USPTO-full, and the multi-step search is a custom Retro* model with a dynamic molecule filter set, a lower-bound beam search score, and a consistency checking mechanism based on forward synthesis.

The Single-Step autoregressive model is fully implemented by ONNX and ONNXRuntime(ORT), which means it will have a noticeable speed-up compared with naive PyTorch implementation in CPU (**about 1 sec. per molecule per step**).

### To Do Lists
1. C++ version of BiRetrosys. Now we are working on the final Search Tree Class and visualization with GraphViz libraries.
2. A simple interface of BiRetroSys.
3. An installation channel, maybe pip or homebrew?
