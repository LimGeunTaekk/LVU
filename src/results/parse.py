import json
import os

if __name__ == '__main__':

    anno_list = json.load(open("/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/25CVPR_VideoMME/videomme/videomme.json"))

    key_frame_list = json.load(open("/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/results/Previous/videomme_32_AKS.json"))

    new_keyframe_list = []

    for i in range(len(anno_list)):
        vid = anno_list[i]['videoID']
        qid = anno_list[i]['question_id']
        key_idx = key_frame_list[i]['frame_idx']

        key_idx = [int(idx) for idx in key_idx]

        info = {
            "vid" : vid,
            "qid" : qid,
            "frame_idx" : key_idx
        }

        new_keyframe_list.append(info)

    with open("/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/src/results/Previous/videomme_32_AKS.json", "w") as json_file:
        json.dump(new_keyframe_list, json_file,indent=4)

