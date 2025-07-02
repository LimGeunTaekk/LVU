export OPENAI_API_KEY=''

python data_extraction/extract_images.py

python data_extraction/extract_features.py

python adaptive_breath_expansion.py \
    --output_filename relevance_score.json \
    --output_base_path ./outputs/dynamic_width_expansion \
    --model gpt-4o-2024-11-20 \
    --prompt_type cap_score \
    --frame_feat_path ./data/egoschema/frame_features \

python depth_expansion.py
