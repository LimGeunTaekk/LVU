import cv2
import os
import torch
import transformers
import json
import argparse
import numpy as np
from PIL import Image

from pathlib import Path
from tqdm import tqdm

from decord import VideoReader
from decord import cpu, gpu

from transformers import AutoModel, AutoConfig
from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--input_base_path",
        default='/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/25CVPR_MLVU/MLVU_Test/video',
    )
    parser.add_argument(
        "--output_base_path",
        default='/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/benchmarks/25CVPR_MLVU/MLVU_Test/frames',
    )
    args = parser.parse_args()
    return args


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def save_image_features(img_feats, name_ids, save_folder):
    """
    Save image features to a .pt file in a specified folder.

    Args:
    - img_feats (torch.Tensor): Tensor containing image features
    - name_ids (str): Identifier to include in the filename
    - save_folder (str): Path to the folder where the file should be saved

    Returns:
    - None
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    torch.save(img_feats, filepath)


def extract_es(args=None):
    # image_path = "CLIP.png"
    model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-8B
    image_size = 224
    batch_size = 16
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()

    input_base_path = Path(args.input_base_path)
    output_base_path = Path(args.output_base_path)
    fps = 1
    pbar = tqdm(total=len(list(input_base_path.iterdir())))

    for video_fp in input_base_path.iterdir():
        # output_path = output_base_path / video_fp.stem
        # output_path.mkdir(parents=True, exist_ok=True)
        vidcap = VideoReader(str(video_fp), ctx=cpu(0))

        fps_ori = int(vidcap.get_avg_fps())
        frame_interval = int(1 / fps * fps_ori)
        total_frames = len(vidcap)
        sample_idx = np.arange(0,total_frames,frame_interval)

        frames = vidcap.get_batch(sample_idx).asnumpy()
        img_feature_list = []

        for i in tqdm(range(0, len(frames), batch_size)):
            frame = frames[i:i+batch_size]
            input_pixels = processor(images=frame, return_tensors="pt", padding=True).pixel_values.to('cuda')

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(input_pixels)
                img_feature_list.extend(image_features)

        img_feature_tensor = torch.stack(img_feature_list)
        img_feats = img_feature_tensor.squeeze(1).detach().cpu()

        save_image_features(img_feats, video_fp.stem, output_base_path)
        pbar.update(1)



if __name__ == '__main__':
    args = parse_eval_args()
    extract_es(args)
