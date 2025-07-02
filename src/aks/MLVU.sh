CUDA_VISIBLE_DEVICES=0 python run_alignment.py \
    --dataset_name MLVU \
    --dataset_path ../../data/recent_bench/25CVPR_MLVU/MLVU_Test \
    --extract_feature_model blip \
    --fps 1 \
    --output_file ./outscores