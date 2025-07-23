CUDA_VISIBLE_DEVICES=3 python run_alignment.py \
    --dataset_name videomme \
    --dataset_path ../../data/prior_bench/25CVPR_VideoMME/videomme \
    --extract_feature_model blip \
    --response_file ./llms/videomme.json \
    --fps 1 \
    --output_file ./outscores
