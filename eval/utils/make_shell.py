#! /usr/bin/env python
import argparse


def get_task_info(task_name, selector_name):
    if task_name == 'longvideobench_val_v':
        info = {
            "original_file" : '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/24NIPS_LongVideoBench/longvideobench/include_frame_idx.json',
            "input_file" : '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/results/Ours_Context/longvideobench',
            "output_file" : '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/24NIPS_LongVideoBench/longvideobench/include_frame_idx_buffer.json',
        }
    elif task_name == 'videomme':
        info = {
            "original_file" : '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/25CVPR_VideoMME/videomme/include_frame_idx.json',
            "input_file" : '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/results/Ours_Context/videomme',
            "output_file" : '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/25CVPR_VideoMME/videomme/include_frame_idx_buffer.json',
        }
    elif task_name == 'mlvu_test':
        info = {
            "original_file" : '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_original.pkl',
            "input_file" : '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/results/Ours_Context/mlvu',
            "output_file" : '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_buffer.pkl',
        }
    elif task_name == 'implicitqa':
        info = {
            "original_file" : '/data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_original.json',
            "input_file" : f'/data3/gtlim/workspace/LVU/src/ours/selected_frames/implicitqa_{selector_name}',
            "output_file" : '/data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/implicitqa_buffer.json',
        }
    return info


def get_model_info(model_name):
    if model_name == 'qwen2_5_vl':
        info = {
            "args" : 'pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_implementation=flash_attention_2,max_pixels=12845056,interleave_visuals=False'
        }
    elif model_name == 'llava_onevision':
        info = {
            "args" : 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen',
        }
    return info


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Ablation")

    parser.add_argument('--task_name', type=str, choices=['implicitqa','longvideobench_val_v','mlvu_test','videomme'], required=True)
    parser.add_argument('--model_name', type=str, choices=['qwen2_5_vl','llava_onevision'], required=True)
    parser.add_argument('--selector_name', type=str, choices=['blip','siglip','care'], required=True)

    args = parser.parse_args()

    task_info = get_task_info(args.task_name, args.selector_name)
    model_info = get_model_info(args.model_name)

    nframes = [16]
    n_scenes = [3]
    detectors = ['pyscene']
    coefs = [0.5]
    context_merge_modes = ['mean','max']
    planning_modes = ['fine','coarse']

    for nframe in nframes:
        for n_scene in n_scenes:
            for detector in detectors:
                for coef in coefs:
                    for context_merge_mode in context_merge_modes:
                        for planning_mode in planning_modes:
                            expname = f'task_{args.task_name}_model_{args.model_name}_scenes_{n_scene}_detector_{detector}_coef_{coef}_context_mode_{context_merge_mode}_planning_mode_{planning_mode}'
                            with open (f'./scripts/ablation_{args.selector_name}/' + expname + '.sh', 'w') as rsh:
                                rsh.write(f'''\
export HF_HOME="~/.cache/huggingface"

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export PATH=$PATH:~/.local/bin

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Ours
python utils/frame_idx.py \\
    --original_file {task_info['original_file']} \\
    --input_file {task_info['input_file']}/Ours_frames_{nframe}_scenes_{n_scene}_detector_{detector}_coef_{coef}_context_mode_{context_merge_mode}_planning_mode_{planning_mode}.json \\
    --output_file {task_info['output_file']} \\

accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \\
    --model {args.model_name} \\
    --model_args={model_info['args']} \\
    --tasks {args.task_name} \\
    --batch_size 1 \\
    --output_path ./results/{args.model_name}/{args.task_name}_{args.selector_name}/{args.model_name}_n_scenes_{n_scene}_detectors_{detector}_coefs_{coef}_context_mode_{context_merge_mode}_planning_mode_{planning_mode} \\
    --verbosity DEBUG \\
    --log_samples \\''')
                            with open(f'run_ours_{args.task_name}_ablation.sh', 'a') as total_sh:
                                total_sh.write(f"bash ./scripts/ablation_{args.selector_name}/{expname}.sh\n")
