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
from transformers import SiglipProcessor, SiglipModel
from transformers import TorchAoConfig, AutoImageProcessor, AutoModel
from torchvision import transforms
from torch.nn.functional import cosine_similarity

from decord import VideoReader
from decord import cpu, gpu
from torch.cuda.amp import autocast

from models.CaReBench.utils.video import read_frames_decord
from models.CaReBench.models.modeling_encoders import AutoEncoder

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
    parser.add_argument('--batch_size', type=int, default=256)

    return parser.parse_args()


def main(args):
    batch_size = args.batch_size
    label_path = os.path.join(args.dataset_path,'anno',f'{args.dataset_name}_original.json')
    response_path = os.path.join('llms','gpt-4o-2024-11-20',f'{args.dataset_name}.json')
    video_path = os.path.join(args.dataset_path,'videos')

    if os.path.exists(label_path):
        with open(label_path,'r') as f:
            datas = json.load(f)
    else:
        raise OSError("the label file does not exist")

    if os.path.exists(response_path):
        with open(response_path,'r') as f:
            responses = json.load(f)
    else:
        raise OSError("the answer file does not exist")

    if args.dataset_name != 'rtvbench':
        video_list = os.listdir(video_path)
    else:
        video_list = []
        for sub_dir in os.listdir(video_path):
            video_list.extend(os.listdir(os.path.join(video_path,sub_dir)))

    interval = len(video_list)//args.split_num
    video_list = video_list[args.batch_idx * interval : (args.batch_idx+1) * interval]

    device = args.device
    
    if args.extract_feature_model == 'blip':
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
        transform_preprocess = transforms.Compose([transforms.ToPILImage(), vis_processors["eval"]])

    elif args.extract_feature_model == 'siglip':
        model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384", attn_implementation="flash_attention_2", torch_dtype=torch.float16, device_map=device,)
        model.to(device)
        processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        transform_preprocess = transforms.Compose([transforms.ToPILImage()])

    elif args.extract_feature_model == 'care':
        model = AutoEncoder.from_pretrained('./models/CaReBench/CaRe_7B')

    out_score_path = os.path.join(args.output_file, args.dataset_name + '_' + str(args.fps) + 'fps', args.extract_feature_model)
    os.makedirs(out_score_path, exist_ok=True)

    score_path = os.path.join(out_score_path,f'scores_{args.batch_idx}.json')
    if os.path.isfile(score_path):
        scores = json.load(open(score_path))
    else:
        scores = dict()

    for vid in tqdm.tqdm(video_list):
        question_id_list = []
        question_list = []
        context_list = []

        for data in datas:
            if type(data['question_id']) == int:
                data['question_id'] = str(data['question_id'])

            if '.mp4' not in data['video_id']:
                data['video_id'] = data['video_id'] + '.mp4'

            if vid == data['video_id']:
                response = responses[data['question_id']]
                response_list = response['answer'].split('\n')
                context = list()
                for item in response_list:
                    if 'Context' in item:
                        cxt = item.split(':')[-1]
                        context.append(cxt)

                question_list.append(data['question'])
                question_id_list.append(data['question_id'])
                context_list.append(context)

                if args.dataset_name == 'rtvbench':
                    sub_field = data['field']

        if args.dataset_name != 'rtvbench':
            vr = VideoReader(os.path.join(video_path, vid), ctx=cpu(0), num_threads=8)
        else:
            vr = VideoReader(os.path.join(video_path, sub_field, vid), ctx=cpu(0), num_threads=8)

        fps = vr.get_avg_fps()
        fps = args.fps * fps

        sample_idx = np.arange(0, len(vr)-1, fps, dtype=int)

        if args.extract_feature_model == 'blip':
            raw_image_np = vr.get_batch(sample_idx).numpy()
            img_list = torch.stack([transform_preprocess(frame) for frame in raw_image_np]).to(device)
        elif args.extract_feature_model == 'siglip':
            raw_image_np = vr.get_batch(sample_idx).numpy()
            img_list = [transform_preprocess(frame) for frame in raw_image_np]
        elif args.extract_feature_model == 'care':
            img_list = vr.get_batch(sample_idx)
            img_list = img_list.permute(0,3,1,2)

        if len(img_list) % batch_size == 0:
            iteration = len(img_list) // batch_size
        else:
            iteration = len(img_list) // batch_size
            iteration = iteration + 1

        with torch.no_grad():
            for q_idx in tqdm.tqdm(range(len(question_list))):
                output_list_q = list()
                output_list_cxt = list()

                for iter_ in range(iteration):
                    # Question
                    if args.extract_feature_model == 'blip': 
                        output_q = model({"image": img_list[iter_*batch_size:(iter_+1)*batch_size], "text_input": text_processors["eval"](question_list[q_idx])}, match_head="itm")
                        output_q = torch.nn.functional.softmax(output_q, dim=-1)
                        output_q = output_q[:,1]

                    elif args.extract_feature_model == 'siglip':
                        inputs = processor(text=question_list[q_idx], images=img_list[iter_*batch_size:(iter_+1)*batch_size], padding="max_length", return_tensors="pt").to(device)
                        with torch.no_grad():
                            with autocast():
                                outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        output_q = torch.sigmoid(logits_per_image).squeeze(1)

                    elif args.extract_feature_model == 'care': 
                        vision_emb = model.encode_vision(img_list[iter_*batch_size:(iter_+1)*batch_size].unsqueeze(0))
                        text_emb = model.encode_text(question_list[q_idx])
                        with autocast(dtype=torch.float16):
                            output_q = cosine_similarity(vision_emb, text_emb)

                    # Context
                    if args.extract_feature_model == 'blip': 
                        output_cxt = []
                        for cxt_idx in range(len(context_list[q_idx])):
                            output_cxt.append(model({"image": img_list[iter_*batch_size:(iter_+1)*batch_size], "text_input": text_processors["eval"](context_list[q_idx][cxt_idx])}, match_head="itm"))
                        output_cxt = torch.stack(output_cxt)
                        output_cxt = torch.nn.functional.softmax(output_cxt, dim=-1)
                        output_cxt = output_cxt[:,:,1]

                    elif args.extract_feature_model == 'siglip':
                        inputs = processor(text=context_list[q_idx], images=img_list[iter_*batch_size:(iter_+1)*batch_size], padding="max_length", return_tensors="pt").to(device)
                        with torch.no_grad():
                            with autocast():
                                outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        output_cxt = torch.sigmoid(logits_per_image).permute(1,0)

                    elif args.extract_feature_model == 'care': 
                        text_emb = model.encode_text(context_list[q_idx])
                        with autocast(dtype=torch.float16):
                            output_cxt = cosine_similarity(vision_emb, text_emb)

                    output_list_q.append(output_q)
                    output_list_cxt.append(output_cxt)

                output_list_q = torch.cat(output_list_q, dim=0)   
                if args.extract_feature_model != 'care': 
                    output_list_cxt = torch.cat(output_list_cxt, dim=1) 
                else:
                    output_list_cxt = torch.stack(output_list_cxt).permute(1,0)

                score_q = output_list_q.detach().cpu().numpy().tolist() # (T)
                score_cxt = output_list_cxt.detach().cpu().numpy().tolist() # (T,C)

                frame_num = sample_idx.tolist()

                scores[question_id_list[q_idx]] = {
                    "frame_idx" : frame_num,
                    "question_score" : score_q,
                    "context_score" : score_cxt
                }

    with open(score_path,'w') as f:
        json.dump(scores,f)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)