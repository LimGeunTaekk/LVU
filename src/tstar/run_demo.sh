export OPENAI_API_KEY=''

python run_TStarDemo.py \
    --video_path /data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/TStar/LVHaystackBench/playground/03e90bbc-7d6b-423c-84d9-b5be3eff11c5.mp4 \
    --question "What is the color of the couch?" \
    --options "A) Red, B) Blue, C) Green, D) Yellow" \
    --grounder gpt-4o \
    --heuristic owl-vit \
    --search_nframes 8