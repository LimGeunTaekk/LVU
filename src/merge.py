import os
import json
import pickle
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TStarSearcher: Simplified Video Frame Search and QA Tool")

    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--n_frames', type=int, default=32, required=True)

    args = parser.parse_args()

    method = args.method
    n_frames = args.n_frames

    frame_idx = list()
    anno_list = json.load(open("../data/recent_bench/25CVPR_MLVU/mlvu_test/test-ground-truth/test_mcq_gt_format.json"))

    if method == 'AKS':
        video_ = './aks/outscores/MLVU_1.0fps/blip/videos.json'
        frame_ = f'./aks/selected_frames/MLVU/blip/selected_frames_{n_frames}_1fps.json'

        video_list = json.load(open(video_))
        frame_list = json.load(open(frame_))

        for anno, qid, indices in zip(anno_list,video_list,frame_list):
            indices.sort()

            frame_idx.append({
                "vid" : anno['video_id'],
                "qid" : qid,
                "frame_idx" : indices
            })

    elif method == 'Tstar':
        frame_ = f'./tstar/output/MLVU_tstar_nframes_{n_frames}.pkl'
        with open(frame_, 'rb') as f:
            frame_list = pickle.load(f)

        for frame in frame_list:
            frame['frame_idx'].sort()

            frame_idx.append({
                "vid" : frame['vid'],
                "qid" : frame['qid'],
                "frame_idx" : frame['frame_idx']
            })

    elif method == 'Videotree':
        exit() # to be continue

    with open(f"./results/{method}_{n_frames}.json", "w") as json_file:
        json.dump(frame_idx, json_file, indent=4)
