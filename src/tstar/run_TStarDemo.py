import argparse
import warnings
import os
import json
import tqdm
import pickle

from TStar.TStarFramework import run_tstar


def list_to_labeled_string(input_list):
    """
    ['Fighting', 'Shoplifting', ...] 형태의 리스트를
    "A) Fighting, B) Shoplifting, ..." 형태의 문자열로 변환하는 함수
    """
    labels = [f"{chr(65 + idx)}) {item}" for idx, item in enumerate(input_list)]
    return ', '.join(labels)

def main():
    """
    TStarSearcher: Simplified Video Frame Search Tool
    
    Example usage:
        python searcher.py --video_path path/to/video.mp4 --question "Your question here" --options "A) Option1\nB) Option2\nC) Option3\nD) Option4"
    """
    parser = argparse.ArgumentParser(description="TStarSearcher: Simplified Video Frame Search and QA Tool")
    # parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    # parser.add_argument('--question', type=str, required=True, help='Question for video content QA.')
    # parser.add_argument('--options', type=str, required=True, help='Multiple-choice options for the question.')

    parser.add_argument('--dataset_name', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Question for video content QA.')

    # search tools
    parser.add_argument('--grounder', type=str, default='gpt-4o', help='Directory to save outputs.')
    parser.add_argument('--heuristic', type=str, default='owl-vit', help='Directory to save outputs.')
    
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for model inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--search_nframes', type=int, default=8, help='Number of top frames to return.')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows in the image grid.')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of columns in the image grid.')
    parser.add_argument('--confidence_threshold', type=float, default=0.6, help='YOLO detection confidence threshold.')
    parser.add_argument('--search_budget', type=float, default=0.5, help='Maximum ratio of frames to process during search.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs.')
    
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir,args.dataset_name + f'_search_nframes_{args.search_nframes}')

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

    search_results = list()

    for data in tqdm.tqdm(datas):
        # Run the TStar search process
        if os.path.isdir(os.path.join(args.output_dir, data['video'].split('.')[0], data['question_id'], 'frames')) == True:
            continue

        try:
            results = run_tstar(
                video_path=os.path.join(video_path,data['video']),
                question=data['question'],
                question_id=data['question_id'],
                options=list_to_labeled_string(data['candidates']),
                grounder=args.grounder,
                heuristic=args.heuristic,
                search_nframes=args.search_nframes,
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
                confidence_threshold=args.confidence_threshold,
                search_budget=args.search_budget,
                output_dir=args.output_dir,
            )

            # Display the results
            print("#"*20)
            print(f"Input Quetion: {data['question']}")
            print(f"Input Options: {list_to_labeled_string(data['candidates'])}")
            print("#"*20)
            print("T* Searching Results:")
            print(f"Grounding Objects: {results['Grounding Objects']}")
            print(f"Frame Timestamps: {results['Frame Timestamps']}")
            # print(f"Answer: {results['Answer']}") # remove QA part (money issue..)

            search_results.append({
                "question" : data['question'],
                "question_id" : data['question_id'],
                "video" : data['video'],
                "keyframe_idx" : results['Frame Timestamps'],
                "grounding_objects" : results['Grounding Objects']
            })
        except:
            continue

    # with open(os.path.join(args.output_dir,'results.pkl'), 'wb') as f:
    #     pickle.dump(search_results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore all warnings
        main()
