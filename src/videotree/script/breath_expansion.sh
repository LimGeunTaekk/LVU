export OPENAI_API_KEY=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

python adaptive_breath_expansion.py \
    --output_filename relevance_score.json \
    --output_base_path ./outputs/dynamic_width_expansion \
    --model gpt-4o-2024-11-20 \
    --prompt_type cap_score \
    --frame_feat_path ./data/egoschema/frame_features \
