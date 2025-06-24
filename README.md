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

Build a unified environment 
* CUDA 11.8
* Ubuntu 18.04
* Python 3.9

```Shell
# For AKS build
git clone https://github.com/Yui010206/SeViLA.git
cd SeViLA
pip install -e . # remove sapcy in requirements.txt
pip install numpy==1.24.4
pip install spacy==3.7.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

```Shell
# For VideoTree build
pip install openai
pip install pandas
pip install accelerate
git clone https://github.com/subhadarship/kmeans_pytorch # replace the init file in "kmeans_pytorch" folder with the file we provide in "./kmeans_pytorch" folder (this repo)
cd kmeans_pytorch
pip install --editable .
```
```Shell
# For Tstar build
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git

# GCC 9.0 version required (Optional)
sudo apt install -y software-properties-common
sudo apt update
sudo apt install -y gcc-9 g++-9

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

sudo update-alternatives --config gcc
sudo update-alternatives --config g++

cd YOLO-World && pip install -e . && cd .. 

pip install -r requirements.txt # do not use official file

# Fix mmdet/mmyolo related issues
sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/g" $(python -c "import importlib.util; filename=importlib.util.find_spec('mmdet').origin;print(filename)")
sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/g" $(python -c "import importlib.util; filename=importlib.util.find_spec('mmyolo').origin;print(filename)")
pip install --upgrade setuptools

# download model
cd ./pretrained/YOLO-World && wget https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth && cd ../..

# download data
mkdir -p data/coco/lvis
wget -O data/coco/lvis/lvis_v1_minival_inserted_image_name.json https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json
mkdir -p data/texts
wget -O data/texts/lvis_v1_class_texts.json https://github.com/AILab-CVC/YOLO-World/raw/refs/heads/master/data/texts/lvis_v1_class_texts.json

# mmcv, transformer version issue check (optional)
pip install -U git+https://github.com/huggingface/transformers,
pip install "mmcv==v2.0.0rc4" # -std=c++17 should be modified (refer to. https://github.com/open-mmlab/mmcv/issues/2860)

python run_TStar_onDataset.py     --video_path ../kfs-train-clip/0a060760-c33f-4160-8719-25725b570043.mp4     --question "What color is my gloves?"     --options "A) Green\nB) Yellow\nC) Blue\nD) Brown\n"
```


Build a unified code framework

* Input : Raw Video + Question
* Output : Keyframe Index

1. [[paper](https://arxiv.org/abs/2502.21271)][[code](https://github.com/ncTimTang/AKS)] Adaptive Keyframe Sampling for Long Video Understanding (AKS, CVPR 2025) 
2. [[paper](https://arxiv.org/abs/2504.02259)][[code](https://github.com/LongVideoHaystack/TStar)] Re-thinking Temporal Search for Long-Form Video Understanding (Tstar, CVPR 2025)
3. [[paper](https://arxiv.org/abs/2405.19209)][[code](https://github.com/Ziyang412/VideoTree)] VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos (VideoTree, CVPR 2025)
