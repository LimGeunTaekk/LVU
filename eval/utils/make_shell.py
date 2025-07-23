#! /usr/bin/env python

if __name__ == '__main__':

    n_scenes = [3, 5, 10]
    detectors = ['pyscene','gebd_dino','gebd_clip']
    coefs = [0.0, 1.0, 0.25, 0.5, 0.75, 0.9, 1.0]

    for n_scene in n_scenes:
        for detector in detectors:
            for coef in coefs:
                expname = f'scenes_{n_scene}_detector_{detector}_coef_{coef}'
                with open ('./scripts/ablation/' + expname + '.sh', 'w') as rsh:
                    rsh.write(f'''\
export HF_HOME="~/.cache/huggingface"

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export PATH=$PATH:~/.local/bin

# Ours
python utils/frame_idx.py \\
    --original_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_original.pkl \\
    --input_file ../src/results/Ours_Context/Ours_frames_32_scenes_{n_scene}_detector_{detector}_coef_{coef}.json \\
    --output_file ../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_buffer.pkl \\

accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \\
    --model qwen2_5_vl \\
    --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False,max_num_frames=32 \\
    --tasks mlvu_test \\
    --batch_size 1 \\
    --output_path ./results/qwen2_5_vl_ours/mlvu_test/Ours_frames_32_scenes_{n_scene}_detector_{detector}_coef_{coef} \\
    --verbosity DEBUG \\
    --log_samples \\''')
                with open(f'run_Ours_ablation.sh', 'a') as total_sh:
                    total_sh.write(f"bash ./scripts/ablation/{expname}.sh\n")
