

CUDA_VISIBLE_DEVICES=0 python run_alignment.py \
    --dataset_name MLVU \
    --dataset_path ../../data/recent_bench/25CVPR_MLVU/mlvu_test \
    --extract_feature_model blip \
    --fps 1 \
    --output_file ./outscores


# python run_frame_select.py \
#     --dataset_name MLVU \
#     --extract_feature_model blip \
#     --score_path ./outscores/MLVU_1.0fps/blip/scores.json \
#     --frame_path ./outscores/MLVU_1.0fps/blip/frames.json \
#     --all_depth 5 \
#     --max_num_frames 32 \
#     --file_name selected_frames_32_1fps.json \