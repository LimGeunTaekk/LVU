CUDA_VISIBLE_DEVICES=3 python run_alignment.py \
    --dataset_name longvideobench \
    --dataset_path ../../data/prior_bench/24NIPS_LongVideoBench \
    --extract_feature_model blip \
    --response_file ./llms/lvbench.json \
    --fps 1 \
    --output_file ./outscores

