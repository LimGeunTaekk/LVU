import torch
import os
import cv2
import json

import numpy as np
import pickle
import tqdm
import argparse
import json

from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import CLIPProcessor, CLIPModel

from decord import VideoReader
from decord import cpu, gpu

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')
    parser.add_argument('--dataset_name', type=str, default='longvideobench', help='support longvideobench and videomme')
    parser.add_argument('--dataset_path', type=str, default='./datasets/longvideobench',help='your path of the dataset')
    parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip/sevila')
    parser.add_argument('--output_file', type=str, default='./outscores',help='path of output scores and frames')
    parser.add_argument('--fps', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def main(args):
    if args.dataset_name =="longvideobench":
       label_path = os.path.join(args.dataset_path,'lvb_val.json')
       video_path = os.path.join(args.dataset_path,'videos')
    elif args.dataset_name =="videomme":
       label_path = os.path.join(args.dataset_path,'videomme.json')
       video_path = os.path.join(args.dataset_path,'data')
    elif args.dataset_name =="egotempo":
       label_path = os.path.join(args.dataset_path,'egotempo_openQA.json')
       video_path = os.path.join(args.dataset_path,'trimmed_clips')
    elif args.dataset_name =="MLVU":
       label_path = os.path.join(args.dataset_path,'test-ground-truth','test_mcq_gt.json')
       video_path = os.path.join(args.dataset_path,'video')
    elif args.dataset_name =="VidComposition":
       label_path = os.path.join(args.dataset_path,'multi_choice.json')
       video_path = os.path.join(args.dataset_path,'videos')
    else:
       raise ValueError("dataset_name: longvideobench or videomme")
    
    if os.path.exists(label_path):
        with open(label_path,'r') as f:
            datas = json.load(f)
            if args.dataset_name == "egotempo":
                datas = datas['annotations']
    else:
        raise OSError("the label file does not exist")
    
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

    scores = []
    fn = []
    videos = []

    score_path = os.path.join(out_score_path,'scores.json')
    frame_path = os.path.join(out_score_path,'frames.json')
    vid_path = os.path.join(out_score_path,'videos.json')

    for data in tqdm.tqdm(datas):
        
        if args.dataset_name == 'longvideobench':
            video = os.path.join(video_path, data["video_path"])
        elif args.dataset_name =="videomme":
            video = os.path.join(video_path, data["videoID"]+'.mp4')
        elif args.dataset_name =="egotempo":
            video = os.path.join(video_path, data["clip_id"]+'.mp4')
        elif args.dataset_name =="MLVU":
            video = os.path.join(video_path, data["video"])
        elif args.dataset_name =="VidComposition":
            video = os.path.join(video_path, data["video"], data["segment"])

        try:
            vr = VideoReader(video, ctx=cpu(0), num_threads=1)
            fps = vr.get_avg_fps() / args.fps
            frame_nums = int((len(vr)/int(fps)))
        except:
            continue

        score = []
        frame_num = []
        text = data['question']  

        if args.extract_feature_model == 'blip':
            txt = text_processors["eval"](text)
            for j in range(frame_nums):
                raw_image = np.array(vr[j*int(fps)])
                raw_image = Image.fromarray(raw_image)
                img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    blip_output = model({"image": img, "text_input": txt}, match_head="itm")               
                blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
                score.append(blip_scores[:, 1].item())
                frame_num.append(j*int(fps))

        elif args.extract_feature_model == 'clip':
            inputs_text = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
            text_features = model.get_text_features(**inputs_text)
            for j in range(frame_nums):
                raw_image = np.array(vr[j*int(fps)])
                raw_image = Image.fromarray(raw_image)
                inputs_image = processor(images=raw_image, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs_image)
                clip_score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
                score.append(clip_score.item())
                frame_num.append(j*int(fps))

        else:
            text = 'Question: ' + data['question'] + ' Candidate: ' 
            if args.dataset_name == 'longvideobench':
                for j,cad in enumerate(data['candidates']):
                    text = text + ". ".join([chr(ord("A")+j), cad]) + ' '
            else:   
                for j in data['options']:
                    text = text + j
            text = text + '. Is this a good frame can answer the question?'
            txt = text_processors["eval"](text)
            for j in range(frame_nums):
                raw_image = np.array(vr[j*int(fps)])
                raw_image = Image.fromarray(raw_image)
                img = vis_processors["eval"](raw_image).unsqueeze(0).unsqueeze(0).to(device)
                samples = {'video':img,'loc_input':txt}
                sevila_score = float(model.generate_score(samples).squeeze(0).squeeze(0))
                score.append(sevila_score)
                frame_num.append(j*int(fps))

        fn.append(frame_num)
        scores.append(score)
        if args.dataset_name =="egotempo":
            videos.append(data['question_id'])
        if args.dataset_name =="MLVU":
            videos.append(data['question_id'])


    with open(frame_path,'w') as f:
        json.dump(fn,f)

    with open(score_path,'w') as f:
        json.dump(scores,f)

    with open(vid_path,'w') as f:
        json.dump(videos,f)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)