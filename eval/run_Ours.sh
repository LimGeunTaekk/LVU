# Run and exactly reproduce qwen2vl results!
# mlvu as an example

export HF_HOME="~/.cache/huggingface"

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export PATH=$PATH:~/.local/bin

# Ours
python utils/frame_idx.py \
    --original_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_original.pkl \
    --input_file /data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/aks/selected_frames/MLVU/blip/Context_aware_AKS_selected_frames_32_coef_0.9.json \
    --output_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_buffer.pkl


accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \
    --tasks mlvu_test \
    --batch_size 1 \
    --output_path ./results/qwen2_5_vl_ours/mlvu_test/AKS_coef_0.9 \
    --verbosity DEBUG \
    --log_samples \

# Ours
python utils/frame_idx.py \
    --original_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_original.pkl \
    --input_file /data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/aks/selected_frames/MLVU/blip/Context_aware_AKS_selected_frames_32_coef_0.75.json \
    --output_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_buffer.pkl


accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \
    --tasks mlvu_test \
    --batch_size 1 \
    --output_path ./results/qwen2_5_vl_ours/mlvu_test/AKS_coef_0.75 \
    --verbosity DEBUG \
    --log_samples \

# Ours
python utils/frame_idx.py \
    --original_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_original.pkl \
    --input_file /data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/aks/selected_frames/MLVU/blip/Context_aware_AKS_selected_frames_32_coef_0.5.json \
    --output_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_buffer.pkl


accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \
    --tasks mlvu_test \
    --batch_size 1 \
    --output_path ./results/qwen2_5_vl_ours/mlvu_test/AKS_coef_0.5 \
    --verbosity DEBUG \
    --log_samples \

# Ours
python utils/frame_idx.py \
    --original_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_original.pkl \
    --input_file /data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/aks/selected_frames/MLVU/blip/Context_aware_AKS_selected_frames_32_coef_0.25.json \
    --output_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_buffer.pkl


accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \
    --tasks mlvu_test \
    --batch_size 1 \
    --output_path ./results/qwen2_5_vl_ours/mlvu_test/AKS_coef_0.25 \
    --verbosity DEBUG \
    --log_samples \

# Ours
python utils/frame_idx.py \
    --original_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_original.pkl \
    --input_file /data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/aks/selected_frames/MLVU/blip/Context_aware_AKS_selected_frames_32_coef_0.1.json \
    --output_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_buffer.pkl


accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \
    --tasks mlvu_test \
    --batch_size 1 \
    --output_path ./results/qwen2_5_vl_ours/mlvu_test/AKS_coef_0.1 \
    --verbosity DEBUG \
    --log_samples \