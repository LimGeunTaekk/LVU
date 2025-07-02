# export OPENAI_API_KEY=''

python run_TStarDemo.py \
    --video_path /data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/recent_bench/25CVPR_MLVU/MLVU_Test/video/test_AWB-6.mp4 \
    --question "In what environment does the scene in the video take place?" \
    --options "A) Ocean, B) City, C) Rainforest, D) Wasteland, E) Snow Mountain, F) Desert" \
    --grounder gpt-4o \
    --heuristic owl-vit \
    --search_nframes 8