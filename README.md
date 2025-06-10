# Long Video Understanding (LVU)

## Problem 1. Keyframe Selection
**Motivation**
* Current video question answering problems can be addressed with a few keyframes (LongVideoBench, VideoMME, EgoSchema)
* Current Query-matching paradigms are sub-optimal (CLIP, BLIP)

**Goal** : We aim to build a spatio-temporal composition-aware method that supports diverse VQA benchmark


### [As-Is] : 

Uniform Sampling 

|     VideoLLMs     |  LLM size  | Input frames | LongVideoBench | VideoMME |  MLVU  | EgoTempo | VidComposition |
|:-----------------|:----------:|:------------:|:--------------:|:--------:|:------:|:--------:|:--------------:|
|   Video-LLaVA     |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|    VideoChat2     |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
| LLaVa-Onevision   |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|     Qwen2-VL      |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |


Adaptive Keyframe Sampling (AKS, CVPR 2025)

|     VideoLLMs     |  LLM size  | Input frames | LongVideoBench | VideoMME |  MLVU  | EgoTempo | VidComposition |
|:-----------------|:----------:|:------------:|:--------------:|:--------:|:------:|:--------:|:--------------:|
|   Video-LLaVA + AKS   |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|    VideoChat2 + AKS     |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
| LLaVa-Onevision + AKS   |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
|     Qwen2-VL + AKS      |      7B    |       32     |        -       |     -    |    -   |     -    |        -       |
