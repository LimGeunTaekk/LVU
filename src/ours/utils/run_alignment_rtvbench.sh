CUDA_VISIBLE_DEVICES=3 python run_alignment_missing.py \
    --dataset_name rtvbench \
    --dataset_path ../../data/benchmarks/rtvbench/ \
    --extract_feature_model blip \
    --fps 1 \
    --output_file ./outscores \
    --split_num 1 \
    --batch_idx 0 \
