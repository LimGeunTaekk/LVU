import torch
import os
import cv2
import json
import time

import numpy as np
import pickle
import tqdm
import argparse
import json

from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

from decord import VideoReader
from decord import cpu, gpu

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')
    parser.add_argument('--split_num', type=int, default=2)
    parser.add_argument('--batch_idx', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='longvideobench', help='support longvideobench and videomme')
    parser.add_argument('--dataset_path', type=str, default='./datasets/longvideobench',help='your path of the dataset')
    parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip/sevila')
    parser.add_argument('--output_file', type=str, default='./outscores',help='path of output scores and frames')
    parser.add_argument('--fps', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def main(args):
    batch_size = 256
    label_path = os.path.join(args.dataset_path,'anno',f'{args.dataset_name}_original.json')
    video_path = os.path.join(args.dataset_path,'videos')
    if os.path.exists(label_path):
        with open(label_path,'r') as f:
            datas = json.load(f)
    else:
        raise OSError("the label file does not exist")

    video_list = os.listdir(video_path)

    interval = len(video_list)//args.split_num
    video_list = video_list[args.batch_idx * interval : (args.batch_idx+1) * interval]

    device = args.device
    
    if args.extract_feature_model == 'blip':
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
    elif args.extract_feature_model == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif args.extract_feature_model == 'sevila':
        model, vis_processors, text_processors = load_model_and_preprocess(name="sevila", model_type="pretrain_flant5xl", is_eval=True, device=device)
    else:
        raise ValueError("model not support")

    out_score_path = os.path.join(args.output_file, args.dataset_name + '_' + str(args.fps) + 'fps', args.extract_feature_model)
    os.makedirs(out_score_path, exist_ok=True)

    score_path = os.path.join(out_score_path,f'scores_{args.batch_idx}.json')
    if os.path.isfile(score_path):
        scores = json.load(open(score_path))
    else:
        scores = dict()

    transform_preprocess = transforms.Compose([transforms.ToPILImage(), vis_processors["eval"]])

    for vid in tqdm.tqdm(video_list):
        question_id_list = []
        question_list = []

        for data in datas:
            if '.mp4' not in data['video_id']:
                data['video_id'] = data['video_id'] + '.mp4'
            if vid == data['video_id']:
                question_list.append(text_processors["eval"](data['question']))
                question_id_list.append(data['question_id'])

        vr = VideoReader(os.path.join(video_path, vid), ctx=cpu(0), num_threads=8)
        fps = vr.get_avg_fps()
        fps = args.fps * fps

        sample_idx = np.arange(0, len(vr)-1, fps, dtype=int)
        raw_image_np = vr.get_batch(sample_idx).numpy()
        img_list = torch.stack([transform_preprocess(frame) for frame in raw_image_np]).to(device)
        
        iteration = img_list.size(0) // batch_size

        with torch.no_grad():
            for q_idx in range(len(question_list)):
                blip_output_list = list()
                for iter_ in range(iteration+1):
                    blip_output = model({"image": img_list[iter_*batch_size:(iter_+1)*batch_size], "text_input": question_list[q_idx]}, match_head="itm")
                    blip_output_list.append(blip_output)
                blip_output_list = torch.cat(blip_output_list, dim=0)          
                blip_scores = torch.nn.functional.softmax(blip_output_list, dim=1)

                score = blip_scores[:,1].detach().cpu().numpy().tolist()
                frame_num = sample_idx.tolist()

                scores[question_id_list[q_idx]] = {
                    "frame_idx" : frame_num,
                    "score" : score
                }

    with open(score_path,'w') as f:
        json.dump(scores,f)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)