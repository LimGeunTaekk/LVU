export HF_HOME="~/.cache/huggingface"
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export PATH=$PATH:~/.local/bin
export CUDA_VISIBLE_DEVICES=4,5,6,7

# nframes=32
# # minerva
# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \
#     --tasks minerva \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/minerva/qwen2_5_vl_nframes_32 \
#     --verbosity DEBUG \
#     --log_samples \

# # videoholmes
# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \
#     --tasks videoholmes \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/videoholmes/qwen2_5_vl_nframes_32 \
#     --verbosity DEBUG \
#     --log_samples \

# # videoevalpro
# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \
#     --tasks videoevalpro \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/videoevalpro/qwen2_5_vl_nframes_32 \
#     --verbosity DEBUG \
#     --log_samples \

# rtvbench
accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \
    --tasks rtvbench \
    --batch_size 1 \
    --output_path ./results/qwen2_5_vl/rtvbench/qwen2_5_vl_nframes_32 \
    --verbosity DEBUG \
    --log_samples \

# nframes=16
# minerva
# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=16 \
#     --tasks minerva \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/minerva/qwen2_5_vl_nframes_16 \
#     --verbosity DEBUG \
#     --log_samples \

# # videoholmes
# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=16 \
#     --tasks videoholmes \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/videoholmes/qwen2_5_vl_nframes_16 \
#     --verbosity DEBUG \
#     --log_samples \

# # videoevalpro
# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=16 \
#     --tasks videoevalpro \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/videoevalpro/qwen2_5_vl_nframes_16 \
#     --verbosity DEBUG \
#     --log_samples \

# rtvbench
accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=16 \
    --tasks rtvbench \
    --batch_size 1 \
    --output_path ./results/qwen2_5_vl/rtvbench/qwen2_5_vl_nframes_16 \
    --verbosity DEBUG \
    --log_samples \

# nframes=8
# minerva
# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=8 \
#     --tasks minerva \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/minerva/qwen2_5_vl_nframes_8 \
#     --verbosity DEBUG \
#     --log_samples \

# # videoholmes
# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=8 \
#     --tasks videoholmes \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/videoholmes/qwen2_5_vl_nframes_8 \
#     --verbosity DEBUG \
#     --log_samples \

# # videoevalpro
# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=8 \
#     --tasks videoevalpro \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/videoevalpro/qwen2_5_vl_nframes_8 \
#     --verbosity DEBUG \
#     --log_samples \

# rtvbench
accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=8 \
    --tasks rtvbench \
    --batch_size 1 \
    --output_path ./results/qwen2_5_vl/rtvbench/qwen2_5_vl_nframes_8 \
    --verbosity DEBUG \
    --log_samples \