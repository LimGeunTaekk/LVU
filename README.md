# Long Video Understanding (LVU)

## Problem 1. Keyframe Selection
**Motivation**
* Current video question answering problems can be addressed with a few keyframes (LongVideoBench, VideoMME, EgoSchema)
* Current Query-matching paradigms are sub-optimal (CLIP, BLIP)

**Goal** : We aim to build a spatio-temporal composition-aware method that supports diverse VQA benchmark

---

**Uniform Sampling (Baseline)**

|     VideoLLMs     |  LLM size  | Input frames | [LongVideoBench](https://github.com/longvideobench/LongVideoBench) | [VideoMME](https://github.com/MME-Benchmarks/Video-MME) |  [MLVU](https://github.com/JUNJIE99/MLVU)  | [EgoTempo](https://github.com/google-research-datasets/egotempo) | [VidComposition](https://github.com/yunlong10/VidComposition) |
|:-----------------|:----------:|:------------:|:--------------:|:--------:|:------:|:--------:|:--------------:|
|   [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)     |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|    [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)     |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
| [LLaVa-Onevision](https://github.com/LLaVA-VL/LLaVA-NeXT)   |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|     [Qwen2-VL](https://github.com/QwenLM/Qwen-VL)      |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |


**Adaptive Keyframe Sampling (AKS, CVPR 2025)**

|     VideoLLMs     |  LLM size  | Input frames | [LongVideoBench](https://github.com/longvideobench/LongVideoBench) | [VideoMME](https://github.com/MME-Benchmarks/Video-MME) |  [MLVU](https://github.com/JUNJIE99/MLVU)  | [EgoTempo](https://github.com/google-research-datasets/egotempo) | [VidComposition](https://github.com/yunlong10/VidComposition) |
|:-----------------|:----------:|:------------:|:--------------:|:--------:|:------:|:--------:|:--------------:|
|   [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) + [AKS](https://github.com/ncTimTang/AKS)    |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|    [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) + [AKS](https://github.com/ncTimTang/AKS)     |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
| [LLaVa-Onevision](https://github.com/LLaVA-VL/LLaVA-NeXT) + [AKS](https://github.com/ncTimTang/AKS)    |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|     [Qwen2-VL](https://github.com/QwenLM/Qwen-VL) + [AKS](https://github.com/ncTimTang/AKS)      |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
