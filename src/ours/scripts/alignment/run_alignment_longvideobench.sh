CUDA_VISIBLE_DEVICES=3 python run_alignment.py \
    --dataset_name longvideobench \
    --dataset_path ../../data/benchmarks/longvideobench/ \
    --extract_feature_model blip \
    --fps 1 \
    --output_file ./outscores \
    --split_num 1 \
    --batch_idx 0 \