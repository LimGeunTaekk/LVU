CUDA_VISIBLE_DEVICES=3 python run_alignment.py \
    --dataset_name implicitqa \
    --dataset_path ../../data/benchmarks/implicitqa/ \
    --extract_feature_model care \
    --fps 0.5 \
    --output_file ./outscores \
    --split_num 2 \
    --batch_idx 1 \
    --batch_size 8 \