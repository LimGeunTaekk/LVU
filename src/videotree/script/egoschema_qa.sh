export OPENAI_API_KEY=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

python main_qa.py \
    --output_filename standard_qa.json \
    --fewshot_example_path ./data/egoschema/few_shot_6.json \
    --output_base_path ./outputs/dynamic_width_expansion \
    --tree_node_idx ./outputs/dynamic_width_expansion/depth_expansion_res.json \
    --model gpt-4o-2024-11-20 \
    --prompt_type qa_standard \
    --temperature 0.0 \

