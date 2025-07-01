# Videos tasks:

- [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) (activitynetqa_generation)
- [SeedBench](https://github.com/AILab-CVC/SEED-Bench) (seedbench)
- [SeedBench 2](https://github.com/AILab-CVC/SEED-Bench) (seedbench_2)
- [CVRR-ES](https://github.com/mbzuai-oryx/CVRR-Evaluation-Suite) (cvrr)
  - cvrr_continuity_and_object_instance_count
  - cvrr_fine_grained_action_understanding
  - cvrr_interpretation_of_social_context
  - cvrr_interpretation_of_visual_context
  - cvrr_multiple_actions_in_a_single_video
  - cvrr_non_existent_actions_with_existent_scene_depictions
  - cvrr_non_existent_actions_with_non_existent_scene_depictions
  - cvrr_partial_actions
  - cvrr_time_order_understanding
  - cvrr_understanding_emotional_context
  - cvrr_unusual_and_physically_anomalous_activities
- [EgoSchema](https://github.com/egoschema/EgoSchema) (egoschema)
  - egoschema_mcppl
  - egoschema_subset_mcppl
  - egoschema_subset
- [LongVideoBench](https://github.com/longvideobench/LongVideoBench)
- [MovieChat](https://github.com/rese1f/MovieChat) (moviechat)
  - Global Mode for entire video (moviechat_global)
  - Breakpoint Mode for specific moments (moviechat_breakpoint)
- [MLVU](https://github.com/JUNJIE99/MLVU) (mlvu)
- [MMT-Bench](https://mmt-bench.github.io/) (mmt)
  - MMT Validation (mmt_val)
  - MMT Test (mmt_test)
- [MVBench](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) (mvbench)

  - mvbench_action_sequence
  - mvbench_moving_count
  - mvbench_action_prediction
  - mvbench_episodic_reasoning
  - mvbench_action_antonym
  - mvbench_action_count
  - mvbench_scene_transition
  - mvbench_object_shuffle
  - mvbench_object_existence
  - mvbench_fine_grained_pose
  - mvbench_unexpected_action
  - mvbench_moving_direction
  - mvbench_state_change
  - mvbench_object_interaction
  - mvbench_character_order
  - mvbench_action_localization
  - mvbench_counterfactual_inference
  - mvbench_fine_grained_action
  - mvbench_moving_attribute
  - mvbench_egocentric_navigation

- [NExT-QA](https://github.com/doc-doc/NExT-QA) (nextqa)

  - NExT-QA Multiple Choice Test (nextqa_mc_test)
  - NExT-QA Open Ended Validation (nextqa_oe_val)
  - NExT-QA Open Ended Test (nextqa_oe_test)

- [PerceptionTest](https://github.com/google-deepmind/perception_test)

  - PerceptionTest Test
    - perceptiontest_test_mc
    - perceptiontest_test_mcppl
  - PerceptionTest Validation
    - perceptiontest_val_mc
    - perceptiontest_val_mcppl

- [TempCompass](https://github.com/llyx97/TempCompass) (tempcompass)

  - tempcompass_multi_choice
  - tempcompass_yes_no
  - tempcompass_caption_matching
  - tempcompass_captioning


- [TemporalBench](https://huggingface.co/datasets/microsoft/TemporalBench) (temporalbench)

  - temporalbench_short_qa
  - temporalbench_long_qa
  - temporalbench_short_caption


- [Vatex](https://eric-xw.github.io/vatex-website/index.html) (vatex)

  - Vatex Chinese (vatex_val_zh)
  - Vatex Test (vatex_test)

- [VideoDetailDescription](https://huggingface.co/datasets/lmms-lab/VideoDetailCaption) (video_dc499)
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT) (videochatgpt)
  - Video-ChatGPT Generic (videochatgpt_gen)
  - Video-ChatGPT Temporal (videochatgpt_temporal)
  - Video-ChatGPT Consistency (videochatgpt_consistency)
- [Video-MME](https://video-mme.github.io/) (videomme)
- [Vinoground](https://vinoground.github.io) (vinoground)
- [VITATECS](https://github.com/lscpku/VITATECS) (vitatecs)

  - VITATECS Direction (vitatecs_direction)
  - VITATECS Intensity (vitatecs_intensity)
  - VITATECS Sequence (vitatecs_sequence)
  - VITATECS Compositionality (vitatecs_compositionality)
  - VITATECS Localization (vitatecs_localization)
  - VITATECS Type (vitatecs_type)

- [WorldQA](https://zhangyuanhan-ai.github.io/WorldQA/) (worldqa)

  - WorldQA Generation (worldqa_gen)
  - WorldQA Multiple Choice (worldqa_mc)

- [YouCook2](http://youcook2.eecs.umich.edu/) (youcook2_val)

- [VDC](https://github.com/rese1f/aurora) (vdc)
  - VDC Detailed Caption (detailed_test)
  - VDC Camera Caption (camera_test)
  - VDC Short Caption (short_test)
  - VDC Background Caption (background_test)
  - VDC Main Object Caption (main_object_test)

- [VideoEval-Pro](https://tiger-ai-lab.github.io/VideoEval-Pro/) (videoevalpro)

# Video Model

### LLaVA-VID

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

cd /path/to/LLaVA-NeXT;
python3 -m pip install -e ".[train]";

python3 -m pip install flash-attn --no-build-isolation;

python3 -m pip install av;


TASK=$1
CKPT_PATH=$2
CONV_TEMPLATE=$3
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llavavid \
    --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,video_decode_backend=decord,max_frames_num=32 \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/

```


### LLaMA-VID

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

# Notice that you should not leave the folder of LLaMA-VID when calling lmms-eval
# Because they left their processor's config inside the repo
cd /path/to/LLaMA-VID;
python3 -m pip install -e .

python3 -m pip install av sentencepiece;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llama_vid \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
```

### Video-LLaVA

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python3 -m pip install transformers --upgrade;
python3 -m pip install av sentencepiece;


TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model video_llava \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/

```


### MPlug-Owl
Notice that this model will takes long time to load, please be patient :)

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

# It has to use an old transformers version to run
python3 -m pip install av sentencepiece protobuf==3.20 transformers==4.28.1 einops;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model mplug_owl_video \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 

```


### Video-ChatGPT

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python3 -m pip install sentencepiece av;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model video_chatgpt \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```

### MovieChat

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/rese1f/MovieChat.git
mv /path/to/MovieChat/MovieChat /path/to/lmms-eval/lmms_eval/models

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model moviechat \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```

### LLaVA-OneVision-MovieChat

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

git clone https://github.com/rese1f/MovieChat.git
mv /path/to/MovieChat/MovieChat_OneVision/llava /path/to/lmms-eval/lmms_eval/models

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava_onevision_moviechat \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```

### AuroraCap

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

git clone https://github.com/rese1f/aurora.git
mv /path/to/aurora/src/xtuner/xtuner /path/to/lmms-eval/lmms_eval/models/xtuner-aurora

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model auroracap \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```


### SliME

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

git clone https://github.com/yfzhang114/SliME.git

cd SliME
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
cd ..

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model slime \
    --tasks $TASK \
    --model_args pretrained="yifanzhang114/SliME-Llama3-8B,conv_template=llama3,model_name=slime" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```
