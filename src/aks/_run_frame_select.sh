python run_frame_select.py \
    --dataset_name implicitqa \
    --extract_feature_model siglip \
    --score_path /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/siglip/scores.json \
    --all_depth 4 \
    --max_num_frames 16 \
    --file_name implicitqa_selected_frames_16_1fps.json \
    --coef 1.0

python run_frame_select.py \
    --dataset_name implicitqa \
    --extract_feature_model care \
    --score_path /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_0.5fps/care/scores.json \
    --all_depth 4 \
    --max_num_frames 16 \
    --file_name implicitqa_selected_frames_16_1fps.json \
    --coef 1.0

