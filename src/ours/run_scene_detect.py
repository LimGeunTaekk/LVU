
import glob
import os
import tqdm
import numpy as np
import pickle
import argparse
import torch
import torch.nn.functional as F

from decord import VideoReader
from decord import cpu, gpu
from torch import nn
from torch.distributions import Categorical
from scenedetect import detect, ContentDetector
from scenedetect.frame_timecode import FrameTimecode

UNIT_SQUARE_SIZE = 9
MINIMUN_SQUARE_SIZE = 5
TOPK_RATIO = 0.2
#MIN_STD = 0.35
MIN_DIFF = 0.8
BIG_MINUS = -100.


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--scene_detector",
        default='pyscene', choices=['pyscene','gebd_dino', 'gebd_clip']
    )
    parser.add_argument(
        "--feat_path",
        default='../../data/benchmarks/25CVPR_MLVU/mlvu_test/CLIP_frame_features',
    )
    parser.add_argument(
        "--video_path",
        default='../../data/benchmarks/25CVPR_MLVU/mlvu_test/video',
    )
    parser.add_argument(
        "--output_path",
        default='../../data/benchmarks/25CVPR_MLVU/mlvu_test/pyscene',
    )
    args = parser.parse_args()
    return args

def g(unit_square):
    const = UNIT_SQUARE_SIZE//2
    p1 = unit_square[:const, :const]
    p2 = unit_square[const+1:, const+1:]
    n1 = unit_square[:const, const+1:]
    n2 = unit_square[const+1:, :const]
    score = torch.mean((p1+p2)-(n1+n2))
    return score

def f(tsm, start_index):
    #tsm: [l, l]
    original_len = tsm.size(0)
    #print(f'input: {tsm.size()}')
    if original_len < MINIMUN_SQUARE_SIZE:
        return []

    topk = int(original_len*TOPK_RATIO)
    score_list = []
    pad_tsm = nn.ZeroPad2d(UNIT_SQUARE_SIZE//2)(tsm)

    for i in range(UNIT_SQUARE_SIZE//2, original_len+(UNIT_SQUARE_SIZE//2)):
        unit_square = pad_tsm[i-(UNIT_SQUARE_SIZE//2):i+(UNIT_SQUARE_SIZE//2)+1, i-(UNIT_SQUARE_SIZE//2):i+(UNIT_SQUARE_SIZE//2)+1]
        score_list.append(g(unit_square))

    scores = torch.stack(score_list)

    if torch.max(scores) - torch.mean(scores) < MIN_DIFF:
        return []

    threshold = -torch.kthvalue(-scores, topk).values
    modified_scores = torch.where(scores >= threshold, scores, torch.full_like(scores, BIG_MINUS))
    distribution = Categorical(logits=modified_scores)
    index = distribution.sample().cpu().numpy().item()

    pre_indice = f(tsm[:index,:index], start_index)
    post_indice = f(tsm[index+1:,index+1:], start_index+index+1)

    ret = pre_indice + post_indice
    ret.append(start_index + index)

    return ret

if __name__ == '__main__':

    args = parse_eval_args()
    if os.path.isdir(os.path.join(args.output_path))==False:
        os.mkdir(os.path.join(args.output_path))

    root_dir = args.video_path
    vid_files = glob.glob(os.path.join(root_dir, '**', '*.mp4'), recursive=True)
    vid_files = [os.path.abspath(p) for p in vid_files]

    for video_path in tqdm.tqdm(vid_files):
        vid = video_path.split('/')[-1]
        if os.path.isdir(os.path.join(args.output_path,vid))==False:
            try:
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
                fps = vr.get_avg_fps()
            except:
                continue

            if args.scene_detector == 'pyscene':
                scene_list = detect(video_path, ContentDetector())
                seg_idx = []
                if len(scene_list) != 0:
                    for seg in scene_list:
                        seg_idx.append((FrameTimecode.get_frames(seg[0]), FrameTimecode.get_frames(seg[1])))
                else:
                    seg_idx = [(0, len(vr)-1)]
            else:
                feat_file = os.path.join(args.feat_path,vid.replace('mp4','pt'))
                video_feat = torch.load(feat_file).cuda()
                video_feat = torch.nn.functional.normalize(video_feat,dim=-1)
                tsm = torch.matmul(video_feat, video_feat.T)
                scene_list = f(tsm, 0)
                if len(scene_list) == 0:
                    seg_idx = [(0, len(vr)-1)]
                else:
                    scene_list.sort()
                    scene_list = np.array(scene_list) * int(fps)
                    seg_idx = [(0,scene_list[0])]
                    for i in range(len(scene_list)-1):
                        seg_idx.append((scene_list[i],scene_list[i+1]))
            results = {
                'video' : vid,
                'list' : seg_idx
            }
            os.mkdir(os.path.join(args.output_path,vid))
            with open(os.path.join(args.output_path, vid, "scene_list.pk"), 'wb') as pk:
                pickle.dump(results, pk, protocol=pickle.HIGHEST_PROTOCOL)
