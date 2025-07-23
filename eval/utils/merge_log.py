import json
import os

from pathlib import Path

task_dict = {
    "topic_reasoning" : "TR",
    "anomaly_reco" : "AR",
    "needleQA" : "NQA",
    "ego" : "ER",
    "plotQA" : "PQA",
    "order" : "AO",
    "count" : "AC",
}

def get_exp_dict():
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needleQA", "order", "plotQA", "sportsQA", "topic_reasoning", "tutorialQA"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}
    return category2score

if __name__ == '__main__':

    root_dir = 'results'
    task = 'mlvu_test'

    for mllm in os.listdir(root_dir):
        for file in sorted(list(Path(os.path.join(root_dir,mllm,task)).rglob('*.jsonl'))):
            exp_dict = get_exp_dict()

            total_correct = 0
            total_answered = 0

            with open(str(file), 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip() != '']

            for res in results:
                pred = res['mlvu_percetion_score']['pred_answer']
                answer = res['mlvu_percetion_score']['answer']
                task_type = res['mlvu_percetion_score']['task_type']

                total_answered += 1
                exp_dict[task_type]['answered'] += 1

                if pred == answer:
                    total_correct += 1
                    exp_dict[task_type]['correct'] += 1

            exp_name = str(file).split('/')[3]
            total_acc = (total_correct / total_answered) * 100

            print(f"\n=== {exp_name} ===")
            print(f"Total Accuracy: {total_acc:.1f}")

            # for type_, type_2 in task_dict.items():
            #     task_acc = (exp_dict[type_]['correct'] / exp_dict[type_]['answered']) * 100
            #     print(f"{type_2:<15}: {task_acc:6.1f}")
