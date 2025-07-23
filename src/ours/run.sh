# python feature_extract.py \
#     --backbone facebook/dinov2-giant

# python run_scene_detect.py

# python run_planning.py

# CUDA_VISIBLE_DEVICES=2 python run_alignment.py \
#     --dataset_name MLVU \
#     --dataset_path ../../data/recent_bench/25CVPR_MLVU/mlvu_test \
#     --extract_feature_model blip \
#     --fps 1 \
#     --output_file ./outscores

# python main.py \
#     --num_of_scene 10 \
#     --max_frames 32 \
#     --scene_detector pyscene \
#     --coef 0.7 \
