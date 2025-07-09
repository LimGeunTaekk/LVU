# export OPENAI_API_KEY=''

# python src/extract_images.py

python adaptive_breath_expansion.py \
    --output_filename relevance_score.json \
    --output_base_path ./outputs/dynamic_width_expansion \
    --model gpt-4o-2024-11-20 \
    --prompt_type cap_score \
    --frame_feat_path /data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/keyframe/videotree/egoschema/frame_features \
    # --frame_feat_path /data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/recent_bench/25CVPR_MLVU/MLVU_Test/frames \

# python depth_expansion.py
