# Scene 5 (Mean) (Fine)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode mean \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode mean \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode mean \
    --planning_mode fine \
    --output_folder implicitqa_blip \

# Scene 5 (Max) (Fine)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

# Scene 5 (Mean) (Coarse)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode mean \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode mean \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode mean \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

# Scene 5 (Max) (Coarse)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode max \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode max \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 5 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode max \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

# Scene 5 (Mean) (Fine)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode mean \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode mean \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode mean \
    --planning_mode fine \
    --output_folder implicitqa_blip \

# Scene 5 (Max) (Fine)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

# Scene 5 (Mean) (Coarse)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode mean \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode mean \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode mean \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

# Scene 5 (Max) (Coarse)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode max \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode max \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 10 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode max \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

# Scene 5 (Mean) (Fine)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode mean \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode mean \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode mean \
    --planning_mode fine \
    --output_folder implicitqa_blip \

# Scene 5 (Max) (Fine)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode max \
    --planning_mode fine \
    --output_folder implicitqa_blip \

# Scene 5 (Mean) (Coarse)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode mean \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode mean \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode mean \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

# Scene 5 (Max) (Coarse)
python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.5 \
    --context_merge_mode max \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 0.75 \
    --context_merge_mode max \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

python run_retrieve.py \
    --dataset /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json \
    --video_dir /data3/gtlim/workspace/LVU/data/benchmarks/implicitqa \
    --llm_output /data3/gtlim/workspace/LVU/src/ours/llms/gpt-4o-2024-11-20/implicitqa.json \
    --matching_score /data3/gtlim/workspace/LVU/src/ours/outscores/implicitqa_1.0fps/blip/scores.json \
    --num_of_scene 15 \
    --max_frames 16 \
    --scene_detector pyscene \
    --coef 1.0 \
    --context_merge_mode max \
    --planning_mode coarse \
    --output_folder implicitqa_blip \

