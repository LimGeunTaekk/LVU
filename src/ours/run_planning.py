import re
import os
import argparse
import json
import tqdm
import pyarrow.parquet as pq
import ast

from openai import OpenAI

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--annotation_file",
        default='../../data/benchmarks/mlvu/anno/test_mcq_gt.json',
    )
    parser.add_argument(
        "--gpt_version",
        default='gpt-4o-2024-11-20',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_eval_args()
    ext = args.annotation_file.split('.')[-1]
    dataset = args.annotation_file.split('/')[4]
    output_path = f'./llms/{args.gpt_version}/{dataset}.json'

    if os.path.isdir(f'./llms/{args.gpt_version}')==False:
        os.mkdir(f'./llms/{args.gpt_version}')

    if ext == 'json':
        annotation_list = json.load(open(args.annotation_file))
        if 'question' not in annotation_list[0].keys():
            for anno in annotation_list:
                anno['question'] = anno['Question']
                del anno['Question']

    elif ext =='jsonl':
        with open(args.annotation_file) as f: # implicitqa
            annotation_list = []
            for line in f:
                anno = json.loads(line)
                anno['question'] = anno['question_text']
                del anno['question_text']
                annotation_list.append(anno)

    elif ext == 'parquet':
        annotation_list = pq.read_table(args.annotation_file).to_pandas().to_dict(orient='records')
        for anno in annotation_list:
            meta_dict = ast.literal_eval(anno['meta'])
            if 'id' in meta_dict.keys():
                anno['question_id'] = meta_dict['id']
            elif 'question_id' in meta_dict.keys():
                anno['question_id'] = meta_dict['question_id']
            elif 'key' in meta_dict.keys():
                anno['question_id'] = meta_dict['key']

    if os.path.isfile(output_path):
        output = json.load(open(output_path))
    else:
        output = dict()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for anno in tqdm.tqdm(annotation_list):

        identify_key = 'question_id'
        if identify_key not in anno.keys():
            if 'id' in anno.keys(): # longvideobench
                identify_key = 'id'
            elif 'key' in anno.keys(): # minerva
                identify_key = 'key'
            elif 'Question ID' in anno.keys(): #videoholmes
                identify_key = 'Question ID'
            elif 'questionID' in anno.keys(): # rtvbench
                identify_key = 'questionID'

        if anno[identify_key] not in output.keys():
            messages = list()
            query = (
                "You are a VideoQA reasoning assistant.\n\n"
                "Given a video and a question, you will analyze the question and reason step-by-step to determine the best approach for answering it.\n"
                "As part of your reasoning, you will identify the question type, select the appropriate video sampling strategy, and define the specific contexts to extract from the video for answering the question.\n\n"
                "Follow these steps:\n\n"
                "Step 1: Identify the question type. Choose the most suitable from:\n"
                "- Object property (e.g., color, shape, size)\n"
                "- Object localization (e.g., where an object is located)\n"
                "- Object state change (e.g., opened vs. closed)\n"
                "- Action recognition (e.g., what action is being performed)\n"
                "- Action prediction (e.g., what will happen next)\n"
                "- Human-object interaction (e.g., how the user interacts with an object)\n"
                "- Global sentiment or emotion (e.g., overall mood of the video)\n"
                "- Scene context understanding (e.g., what is happening in the scene)\n"
                "- Other (specify clearly if needed)\n\n"
                "Step 2: Based on the question type and the expected information required to answer it, select the **video sampling strategy:\n"
                "- Frame-level (the question can be answered by looking at a single frame)\n"
                "- Scene-level (the question can be answered by looking at a single scene)\n"
                "- Global-level (the question requires looking at multiple scenes throughout the video)\n\n"
                "**Important: You must choose the sampling strategy exactly as one of these three words: 'Frame-level', 'Scene-level', or 'Global-level'.**\n\n"
                "Step 3: Based on the question type and the selected sampling strategy, define the specific contexts that should be extracted from the video to answer the question.\n"
                "Use the following structured style:\n"
                " - Context 1: [brief description]\n"
                " - Context 2: [brief description]\n"
                " - Context 3: [brief description]\n"
                "...\n\n"
                "**Note:** The number of contexts should be flexible and determined by what is necessary based on the question type and the selected sampling strategy.\n\n"
                "Example:\n"
                "Question: What will the user likely do with the blue microfiber cloth?\n"
                "Step 1:\n"
                " - Question type: Action prediction\n"
                "Step 2:\n"
                " - Sampling strategy: Scene-level\n"
                "Step 3:\n"
                " - Context 1: The scene where the user is holding or picking up the blue microfiber cloth.\n"
                " - Context 2: The scene showing what the user does immediately after picking up the blue microfiber cloth, such as wiping or cleaning an object.\n\n"
                "Please follow the above example format when answering.\n\n"
                f"Question: {anno['question']}"
            )
            messages.append({"role": "user", "content": f"{query}"})

            try:
                if 'o3' not in args.gpt_version:
                    completion = client.chat.completions.create(model=args.gpt_version, messages=messages, temperature=0, seed=42)
                else:
                    completion = client.chat.completions.create(model=args.gpt_version, messages=messages, seed=42)
                answer_content = completion.choices[0].message.content
                output[anno[identify_key]] = {
                    "question" : anno['question'],
                    "answer" : answer_content
                }
            except Exception as e:
                print(f"QID : {anno[identify_key]} || ERROR : {e}")
        else:
            continue

    with open(output_path, "w") as json_file:
        json.dump(output, json_file, indent=4)
