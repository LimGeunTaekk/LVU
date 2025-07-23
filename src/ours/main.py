import os
import json
import torch
import tqdm
import pickle
import argparse
import numpy as np

from decord import VideoReader
from decord import cpu, gpu

from scenedetect.frame_timecode import FrameTimecode

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--dataset",
        default='/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt.json',
    )
    parser.add_argument(
        "--video_dir",
        default='/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/recent_bench/25CVPR_MLVU/mlvu_test',
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
    args = parser.parse_args()
    return args


def get_scene_boundary(detector='pyscene', video_dir=None, video_name=None, question_id=None, vr=None):
    path = os.path.join(video_dir, detector, video_name, 'scene_list.pk')
    with open(path, 'rb') as f:
        seg_idx = pickle.load(f)
    return np.array(seg_idx['list'])

def get_matching_score(score_info=None, coef=0.7):
    score=[]
    for sl in score_info:
        qs = sl['question_score']
        cs = np.mean(sl['context_score'])
        sc = coef * qs + (1-coef) * cs
        score.append(sc)
    return np.array(score)

def parse_llm_answer(question=None, response=None):
    plan = dict()

    find = False
    for sampling_strategy in ['Global-level','Scene-level','Frame-level']:
        if sampling_strategy in response:
            find = True
            break

    if find == False:
        import pdb;pdb.set_trace()

    response_list = response.split('\n')
    context = list()
    for item in response_list:
        if 'Context' in item:
            context.append(item.split(':')[-1])

    plan['sampling_strategy'] = sampling_strategy
    plan['context'] = context
    return plan

if __name__ == '__main__':

    args = parse_eval_args()
    annotation_list = json.load(open(args.dataset))
    answer_list = json.load(open(args.llm_output)) # question_id로 dict 정리
    score_list = json.load(open(args.matching_score)) # question_id로 dict 정리

    output_list = list()
    
    for anno in tqdm.tqdm(annotation_list):

        question = anno['question']
        vid = anno['video']
        qid = anno['question_id']

        vr = VideoReader(os.path.join(args.video_dir,'video',vid), ctx=cpu(0))
        fps = int(vr.get_avg_fps())

        seg_idx = list()

        score_info = score_list[qid]
        answer_info = answer_list[qid]

        # step1. scene boundary preprocessing
        seg_idx = get_scene_boundary(args.scene_detector, args.video_dir, vid, qid, vr)

        # step2. LLM planning preprocessing
        score = get_matching_score(score_info['score'], args.coef)
        frame_idx = np.array(score_info['frame_idx'])
        llm_answer = answer_info['answer']
        plan_dict = parse_llm_answer(question, llm_answer)

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
                keyframe_idx = np.linspace(start, end, args.max_frames, dtype=int)
            elif plan_dict['sampling_strategy'] == 'Frame-level':
                keyframe_idx = np.array([best_f_idx])
        
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

    with open(f"../results/Ours_Context/Ours_frames_{args.max_frames}_scenes_{args.num_of_scene}_detector_{args.scene_detector}_coef_{args.coef}.json", "w") as json_file:
        json.dump(output_list, json_file, indent=4)
