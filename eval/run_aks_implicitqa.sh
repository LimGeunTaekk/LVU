export HF_HOME="~/.cache/huggingface"
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export PATH=$PATH:~/.local/bin
export CUDA_VISIBLE_DEVICES=0,1,2,3

# # Ours
# python utils/frame_idx.py \
#     --original_file /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
#     --input_file /data3/gtlim/workspace/LVU/src/aks/selected_frames/implicitqa/siglip/implicitqa_selected_frames_16_1fps.json \
#     --output_file /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_buffer.json \

# accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False \
#     --tasks implicitqa \
#     --batch_size 1 \
#     --output_path ./results/qwen2_5_vl/implicitqa/qwen2_5_vl_siglip_aks_nframes_16 \
#     --verbosity DEBUG \
#     --log_samples \


# Ours
python utils/frame_idx.py \
    --original_file /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --input_file /data3/gtlim/workspace/LVU/src/aks/selected_frames/implicitqa/care/implicitqa_selected_frames_16_1fps.json \
    --output_file /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_buffer.json \

accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False \
    --tasks implicitqa \
    --batch_size 1 \
    --output_path ./results/qwen2_5_vl/implicitqa/qwen2_5_vl_care_aks_nframes_16 \
    --verbosity DEBUG \
    --log_samples \