import json
import os
import numpy as np

from pathlib import Path
from collections import OrderedDict

videoevalpro = {
    'Local Reasoning' : 'LR', 
    'Local Perception' : 'LP', 
    'Holistic Reasoning' : 'HR', 
    'Holistic Perception' : 'HP'
}

videoholmes = {
    'SR' : 'Social Reasoning',
    'IMC' : 'Intention & Motive Chaining',
    'TCI' : 'Temporal Causal Inference',
    'TA' : 'Timeline Analysis',
    'MHR' : 'Multimodal Hint Reasoning',
    'PAR' : 'Physical Anomaly Reasoning',
    'CTI' : 'Core Theme Inference'
}

rtvbench = [
    'Event-PU', 
    'Event-GU', 
    'Event-SR', 
    'Action-SP', 
    'Event-TP', 
    'Event-SP', 
    'Action-PU', 
    'Object-GU', 
    'Action-GU', 
    'Action-VP', 
    'Object-VP', 
    'Action-TP', 
    'Object-IA', 
    'Action-SR', 
    'Object-SP', 
    'Object-PU', 
    'Event-IA', 
    'Object-SR', 
    'Action-IA', 
    'Object-FP', 
    'Event-VP', 
    'Event-FP', 
    'Action-FP', 
    'Object-TP'
]

mlvu_test = {
    "topic_reasoning" : "TR",
    "anomaly_reco" : "AR",
    "needleQA" : "NQA",
    "ego" : "ER",
    "plotQA" : "PQA",
    "order" : "AO",
    "count" : "AC",
}

minerva = [
    'Reading', 
    'Cause and Effect', 
    'Spatial Perception', 
    'State Changes', 
    'Temporal Reasoning', 
    'Event Occurence', 
    'Goal Reasoning', 
    'Situational Awareness', 
    'Listening', 
    'Object Recognition', 
    'Numerical Reasoning', 
    'Counterfactual', 
    'Counting'
]

longvideobench_val_v = [
    ("Scene-referred Event" , "S2E"),
    ("Scene-referred Object" , "S2O"),
    ("Scene-referred Object Attribute" , "S2A"),
    ("Event-referred Object" , "E2O"),
    ("Object-referred Event" , "O2E"),
    ("Text-referred Event" , "T2E"),
    ("Text-referred Object" , "T2O"),
    ("Text-referred Object Attribute" , "T2A"),
    ("Event before/after Event" , "E3E"),
    ("Object before/after Object" , "O3O"),
    ("Scene-referred Event" , "SSS"),
    ("Scene-referred Object Tracking" , "SOS"),
    ("Sequence-referred Object Attribute Change" , "SAA"),
    ("Event before/after Text" , "T3E"),
    ("Object before/after Text" , "T3O"),
    ("Text-referred Object Tracking" , "TOS"),
    ("Text-referred Object Attribute Change" , "TAA"),
]

implicitqa = {
    'Causal and Motivational Reasoning' : 'CMR', 
    'Social Interaction and Relationships' : 'SIR', 
    'Viewpoint and Visibility' : 'VV', 
    'Lateral Spatial Reasoning' : 'LSR', 
    'Motion and Trajectory Dynamics' : 'MTD', 
    'Inferred Counting' : 'IC', 
    'Physical and Environmental Context' : 'PEC', 
    'Relative Depth and Proximity' : 'RDP', 
    'Vertical Spatial Reasoning' : 'VSR' 
}

def get_exp_dict(TASK_TYPES):
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}
    return category2score

if __name__ == '__main__':

    save_dir = 'implicitqa'
    task = 'implicitqa'

    root_dir = 'results'

    if task == 'mlvu_test':
        task_dict = mlvu_test
    elif task == 'longvideobench_val_v':
        task_dict = longvideobench_val_v
    elif task == 'implicitqa':
        task_dict = implicitqa
    elif task == 'videoholmes':
        task_dict = videoholmes
    elif task == 'minerva':
        task_dict = minerva
    elif task == 'videoevalpro':
        task_dict = videoevalpro

    for mllm in os.listdir(root_dir):
        for file in sorted(list(Path(os.path.join(root_dir,mllm,save_dir)).rglob('*.jsonl'))):

            total_correct = 0
            total_answered = 0

            with open(str(file), 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip() != '']
            
            type_ = set()
            for res in results:
                if task == 'mlvu_test':
                    type_.add(res['mlvu_percetion_score']['task_type'])
                elif task == 'longvideobench_val_v':
                    type_.add(res['lvb_acc']['question_category'])
                elif task == 'implicitqa':
                    try:
                        type_.add(res['videomme_perception_score']['task_type'])
                    except:
                        type_.add(res['implicitqa_perception_score']['task_type'])

                elif task == 'videoholmes':
                    type_.add(res['videoholmes_perception_score']['task_type'])
                elif task == 'minerva':
                    type_.add(res['minerva_perception_score']['task_type'])
                elif task == 'videoevalpro':
                    type_.add(res['videoevalpro_perception_score']['task_type'])

            exp_dict = get_exp_dict(type_)

            for res in results:
                if task == 'mlvu_test':
                    pred = res['mlvu_percetion_score']['pred_answer']
                    answer = res['mlvu_percetion_score']['answer']
                    task_type = res['mlvu_percetion_score']['task_type']
                elif task == 'longvideobench_val_v':
                    pred = res['lvb_acc']['parsed_pred']
                    answer = res['lvb_acc']['answer']
                    task_type = res['lvb_acc']['question_category']
                elif task == 'implicitqa':
                    try:
                        pred = res['videomme_perception_score']['pred_answer']
                        answer = res['videomme_perception_score']['answer']
                        task_type = res['videomme_perception_score']['task_type']
                    except:
                        pred = res['implicitqa_perception_score']['pred_answer']
                        answer = res['implicitqa_perception_score']['answer']
                        task_type = res['implicitqa_perception_score']['task_type']
                elif task == 'videoholmes':
                    pred = res['videoholmes_perception_score']['pred_answer']
                    answer = res['videoholmes_perception_score']['answer']
                    task_type = res['videoholmes_perception_score']['task_type']
                elif task == 'minerva':
                    pred = res['minerva_perception_score']['pred_answer']
                    answer = res['minerva_perception_score']['answer']
                    task_type = res['minerva_perception_score']['task_type']
                elif task == 'videoevalpro':
                    pred = res['videoevalpro_perception_score']['pred_answer']
                    answer = res['videoevalpro_perception_score']['answer']
                    task_type = res['videoevalpro_perception_score']['task_type']
                total_answered += 1
                exp_dict[task_type]['answered'] += 1

                if pred == answer:
                    total_correct += 1
                    exp_dict[task_type]['correct'] += 1

            exp_name = str(file).split('/')[3]
            total_acc = (total_correct / total_answered) * 100
            macro_acc = []
            print(f"\n=== {exp_name} ===")
            print(f"Total Accuracy: {total_acc:.1f}")

            for type_, type_2 in task_dict.items():
                if type_ in exp_dict.keys():
                    task_acc = (exp_dict[type_]['correct'] / exp_dict[type_]['answered']) * 100
                    # print(f"{type_:<15}: {task_acc:6.1f}")
                else:
                    task_acc = (exp_dict[type_2]['correct'] / exp_dict[type_2]['answered']) * 100
                    # print(f"{type_2:<15}: {task_acc:6.1f}")
                macro_acc.append(task_acc)

            print(f"Macro Accuracy: {np.mean(macro_acc):.1f}")


            # for type_ in task_dict:
            #     task_acc = (exp_dict[type_]['correct'] / exp_dict[type_]['answered']) * 100
            #     print(f"{type_:<15}: {task_acc:6.1f}")
