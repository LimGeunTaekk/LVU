# Long Video Understanding (LVU)

## Problem 1. Keyframe Selection
**Motivation**
* Current video question answering problems can be addressed with a few keyframes (LongVideoBench, VideoMME, EgoSchema)
* Current Query-matching paradigms are sub-optimal (CLIP, BLIP)

**Goal** : We aim to build a spatio-temporal composition-aware method that supports diverse VQA benchmark

---

**Uniform Sampling (Baseline)**

To do. 
* a
* b
* c

|     VideoLLMs     |  LLM size  | Input frames | [LongVideoBench](https://github.com/longvideobench/LongVideoBench) | [VideoMME](https://github.com/MME-Benchmarks/Video-MME) |  [MLVU](https://github.com/JUNJIE99/MLVU)  | [EgoTempo](https://github.com/google-research-datasets/egotempo) | [VidComposition](https://github.com/yunlong10/VidComposition) |
|:-----------------|:----------:|:------------:|:--------------:|:--------:|:------:|:--------:|:--------------:|
|   [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)     |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|    [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)     |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
| [LLaVa-Onevision](https://github.com/LLaVA-VL/LLaVA-NeXT)   |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|     [Qwen2-VL](https://github.com/QwenLM/Qwen-VL)      |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |


---

**Keyframe Sampling Method**

Build a unified environment (CUDA 11.8, Ubuntu 18.04)

```Shell
# For AKS build
git clone https://github.com/Yui010206/SeViLA.git
cd SeViLA
pip install -e . # remove sapcy in requirements.txt
pip install numpy==1.24.4
pip install spacy==3.7.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```


* Build a unified code framework

1. [[paper](https://arxiv.org/abs/2502.21271)][[code](https://github.com/ncTimTang/AKS)] Adaptive Keyframe Sampling for Long Video Understanding (AKS, CVPR 2025) 
2. [[paper](https://arxiv.org/abs/2504.02259)][[code](https://github.com/LongVideoHaystack/TStar)] Re-thinking Temporal Search for Long-Form Video Understanding (Tstar, CVPR 2025)
3. [[paper](https://arxiv.org/abs/2405.19209)][[code](https://github.com/Ziyang412/VideoTree)] VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos (VideoTree, CVPR 2025)
