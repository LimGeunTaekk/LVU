import os
import json
import torch
import tqdm
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from decord import VideoReader
from decord import cpu, gpu

from scenedetect.frame_timecode import FrameTimecode

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--dataset",
        default='/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt.json',
    )
    parser.add_argument(
        "--video_dir",
        default='/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/25CVPR_MLVU/mlvu_test',
    )
    parser.add_argument(
        "--llm_output",
        default='./llms/mlvu.json',
    ) 
    parser.add_argument(
        "--matching_score",
        default='/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/ours/outscores/MLVU_1.0fps/blip/scores.json',
    )
    parser.add_argument(
        "--scene_detector",
        default='pyscene', choices=['pyscene','gebd_dino', 'gebd_clip']
    )
    parser.add_argument(
        "--context_merge_mode",
        default='mean', choices=['mean','max']
    )
    parser.add_argument(
        "--planning_mode",
        default='fine', choices=['fine','coarse']
    )
    parser.add_argument(
        "--coef",
        type=float, default=0.5,
    )
    parser.add_argument(
        "--num_of_scene",
        type=int, default=3,
    )
    parser.add_argument(
        "--max_frames",
        type=int, default=32,
    )
    parser.add_argument(
        "--output_folder",
        default='mlvu',
    )
    args = parser.parse_args()
    return args


def get_scene_boundary(detector='pyscene', video_dir=None, video_name=None, question_id=None, vr=None):
    path = os.path.join(video_dir, detector, video_name, 'scene_list.pk')
    with open(path, 'rb') as f:
        seg_idx = pickle.load(f)
    return np.array(seg_idx['list'])

def get_matching_score(args, score_info=None, coef=0.7, after_opt=False):
    if after_opt == False:
        score=[]
        for sl in score_info:
            qs = sl['question_score']
            if args.context_merge_mode == 'mean':
                cs = np.mean(sl['context_score'])
            elif args.context_merge_mode == 'max':
                cs = np.max(sl['context_score'])
            sc = coef * qs + (1-coef) * cs
            score.append(sc)
        return np.array(score)
    else:
        qs = np.array(score_info['question_score'])
        cs = np.array(score_info['context_score'])
        if args.context_merge_mode == 'mean':
            cs = np.mean(cs,axis=0)
        elif args.context_merge_mode == 'max':
            cs = np.max(cs,axis=0)
        score = coef * qs + (1-coef) * cs
        return score

def parse_llm_answer(question=None, response=None):
    plan = dict()

    find = False
    for sampling_strategy in ['Global-level','Scene-level','Frame-level']:
        if sampling_strategy in response:
            find = True
            break

    if find == False:
        return None

    response_list = response.split('\n')
    context = list()
    for item in response_list:
        if 'Context' in item:
            context.append(item.split(':')[-1])

    plan['sampling_strategy'] = sampling_strategy
    plan['context'] = context
    return plan

def get_meta(args, anno_dict):

    if '.mp4' not in anno_dict['video_id']:
        anno_dict['video_id'] = anno_dict['video_id'] + '.mp4'

    question = anno['question']
    vid = anno['video_id']
    qid = anno['question_id']

    if args.output_folder == 'rtvbench':
        vr = VideoReader(os.path.join(args.video_dir,'videos', anno_dict['field'], anno_dict['video_id']), ctx=cpu(0))
    else:
        vr = VideoReader(os.path.join(args.video_dir,'videos', anno_dict['video_id']), ctx=cpu(0))

    return question, vid, qid, vr


def get_oracle(oracle_list, vr, qid):
    for anno_dict in oracle_list:
        if anno_dict['question_id'] == qid:

            start_time = anno_dict['question_start_time']
            end_time = anno_dict['question_stop_time']
            fps = vr.get_avg_fps()

            start_idx = int(anno_dict['question_start_time'] * fps)
            end_idx = int(anno_dict['question_stop_time'] * fps)
            end_idx = min(end_idx, len(vr)-1)

    return start_idx, end_idx

def get_visualization(args, anno, score_info, vr, seg_idx, keyframe_idx, oracle_list=None):
    vid = anno['video_id']
    qid = anno['question_id']
    question = anno['question']

    if oracle_list is not None:
        start_idx, end_idx = get_oracle(oracle_list, vr, qid)

    frame_sample_idx = score_info['frame_idx']
    scene_bound_idx = [idx[-1] for idx in seg_idx]

    question_score = list()
    context_score = list()
    gt_frame_idx = list()

    for i in range(len(frame_sample_idx)):
        question_score.append(score_info['score'][i]['question_score'])

        if args.context_merge_mode == 'mean':
            context_score.append(np.mean(score_info['score'][i]['context_score']))
        elif args.context_merge_mode == 'max':
            context_score.append(np.max(score_info['score'][i]['context_score']))

        if frame_sample_idx[i] >= start_idx and frame_sample_idx[i] <= end_idx:
            gt_frame_idx.append(1)
        else:
            gt_frame_idx.append(0)

    plt.figure(figsize=(10, 4))

    plt.plot(gt_frame_idx, label='Ground Truth Region', drawstyle='steps-post', color='green')
    plt.plot(question_score, label='VLM Matching Score (Question)', color='blue')
    plt.plot(context_score, label='VLM Matching Score (Context)', color='red')

    plt.xlabel('Time (s)')
    plt.ylabel('Matching Score (Cosine)')
    plt.title(f'Question : {question}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if os.path.isdir(f'selected_frames/{args.output_folder}/visualization_{args.context_merge_mode}/{vid}') == False:
        os.mkdir(f'selected_frames/{args.output_folder}/visualization_{args.context_merge_mode}/{vid}')

    question = question.replace('/',' ')

    plt.savefig(f'selected_frames/{args.output_folder}/visualization_{args.context_merge_mode}/{vid}/{question}.png', dpi=300)
    plt.close()


if __name__ == '__main__':

    args = parse_eval_args()
    annotation_list = json.load(open(args.dataset))
    answer_list = json.load(open(args.llm_output)) # question_id로 dict 정리
    score_list = json.load(open(args.matching_score)) # question_id로 dict 정리

    output_list = list()

    if os.path.isdir(os.path.join('./selected_frames',args.output_folder)) == False:
        os.mkdir(os.path.join('./selected_frames',args.output_folder))

    if args.output_folder == 'implicitqa':
        with open("/data3/gtlim/workspace/LVU/data/benchmarks/implicitqa/anno/ImplicitQAv0.1.2.jsonl", "r") as f:
            oracle_list = []
            for line in f:
                oracle_list.append(json.loads(line))
    else:
        oracle_list = None

    for anno in tqdm.tqdm(annotation_list):
        question, vid, qid, vr = get_meta(args, anno)

        fps = int(vr.get_avg_fps())

        seg_idx = list()

        score_info = score_list[qid]
        answer_info = answer_list[qid]

        # step1. scene boundary preprocessing
        seg_idx = get_scene_boundary(args.scene_detector, args.video_dir, vid, qid, vr)

        # step2. LLM planning preprocessing
        try:
            score = get_matching_score(args, score_info['score'], args.coef)
        except:
            score = get_matching_score(args, score_info, args.coef, after_opt=True)

        frame_idx = np.array(score_info['frame_idx'])
        llm_answer = answer_info['answer']
        plan_dict = parse_llm_answer(question, llm_answer)

        if len(score) != len(frame_idx):
            if len(frame_idx) % len(score) == 0:
                iter_ = len(frame_idx) // len(score)
                score = np.repeat(score,iter_)
            else:
                iter_ = len(frame_idx) // len(score) + 1
                score = np.repeat(score,iter_)
                score = score[:len(frame_idx)]

        if plan_dict is None:
            continue

        # step3. Matching based frame retrieval
        scene_score_list = list()
        for seg in seg_idx:
            start_idx = seg[0] <= frame_idx
            end_idx = seg[1] >= frame_idx
            scene_idx = np.logical_and(start_idx,end_idx)
            try:
                scene_score_list.append(np.max(score[scene_idx]))
            except: # average boundary score (less than fps)
                try:
                    start_idx = min(seg[0] // fps, len(score)-1)
                    end_idx = min(seg[1] // fps, len(score)-1)
                    start_score = score[start_idx]
                    end_score = score[end_idx]
                    scene_score_list.append(np.mean([start_score,end_score]))
                except:
                    import pdb;pdb.set_trace()

        scene_score_list = np.array(scene_score_list)
        topk_idx = np.argsort(scene_score_list)[::-1]

        # step4. Scene based sampling
        if plan_dict['sampling_strategy'] == 'Global-level':
            if args.planning_mode == 'fine':
                target_scene = seg_idx[topk_idx[:args.num_of_scene]]
                target_frame_list = list()
                for candidate_scene in target_scene:
                    start = candidate_scene[0]
                    end = candidate_scene[1]
                    if end == len(vr):
                        end = end -1
                    target_frame_list.extend(range(start,end))

                target_frame_list = np.array(target_frame_list)
                target_idx = np.linspace(0,len(target_frame_list)-1, args.max_frames, dtype=int)
                keyframe_idx = target_frame_list[target_idx]
            elif args.planning_mode == 'coarse':
                keyframe_idx = np.linspace(0,len(vr)-1, args.max_frames, dtype=int)
        else:
            target_scene = seg_idx[topk_idx[0]]
            start = target_scene[0]
            end = target_scene[1]
            if end == len(vr):
                end = end -1

            start_idx = start <= frame_idx
            end_idx = end >= frame_idx
            scene_idx = np.logical_and(start_idx, end_idx)
    
            try:
                best_idx = np.argmax(score[scene_idx])
                best_f_idx = frame_idx[scene_idx][best_idx]
            except:
                best_f_idx = min(start // fps, len(score)-1)

            if plan_dict['sampling_strategy'] == 'Scene-level':
                if args.planning_mode == 'fine':
                    keyframe_idx = np.linspace(start, end, args.max_frames, dtype=int)
                elif args.planning_mode == 'coarse':
                    target_scene = seg_idx[topk_idx[:args.num_of_scene]]
                    target_frame_list = list()
                    for candidate_scene in target_scene:
                        start = candidate_scene[0]
                        end = candidate_scene[1]
                        if end == len(vr):
                            end = end -1
                        target_frame_list.extend(range(start,end))

                    target_frame_list = np.array(target_frame_list)
                    target_idx = np.linspace(0,len(target_frame_list)-1, args.max_frames, dtype=int)
                    keyframe_idx = target_frame_list[target_idx]

            elif plan_dict['sampling_strategy'] == 'Frame-level':
                if args.planning_mode == 'fine':
                    keyframe_idx = np.array([best_f_idx])
                elif args.planning_mode == 'coarse':
                    keyframe_idx = np.linspace(start, end, args.max_frames, dtype=int)

        
        keyframe_idx = list(set(keyframe_idx.tolist()))
        keyframe_idx.sort()

        if np.max(keyframe_idx) >= len(vr) or len(keyframe_idx) > args.max_frames:
            import pdb;pdb.set_trace()

        output_list.append({
            "vid" : vid,
            "qid" : qid,
            "sampling_strategy" : plan_dict['sampling_strategy'],
            "frame_idx" : keyframe_idx
        })

        # get_visualization(args, anno, score_info, vr, seg_idx, keyframe_idx, oracle_list)

    with open(f"selected_frames/{args.output_folder}/Ours_frames_{args.max_frames}_scenes_{args.num_of_scene}_detector_{args.scene_detector}_coef_{args.coef}_context_mode_{args.context_merge_mode}_planning_mode_{args.planning_mode}.json", "w") as json_file:
        json.dump(output_list, json_file, indent=4)
