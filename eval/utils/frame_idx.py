import os
import json
import pickle
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TStarSearcher: Simplified Video Frame Search and QA Tool")

    parser.add_argument('--original_file', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()

    frame_idx_list = json.load(open(args.input_file))
    ext = args.original_file.split('.')[-1]
    if ext == 'pk':
        original_dataset = pickle.load(open(args.original_file,'rb'))
    elif ext == 'json':
        original_dataset = json.load(open(args.original_file))

    for i in range(len(frame_idx_list)):
        find = False
        for j in range(len(original_dataset)):
            q_key = 'question_id'
            qid = frame_idx_list[i]['qid']
            if q_key not in original_dataset[j].keys():
                q_key = 'id'
            qid_o = original_dataset[j][q_key]
            if type(qid_o) == int:
                qid_o = str(qid_o)
            if qid == qid_o:
                frame_idx = frame_idx_list[i]['frame_idx']
                frame_idx = [str(idx) for idx in sorted(frame_idx)]
                original_dataset[j]['indices'] = frame_idx
                find = True
                break
            else:
                continue
        if find == False:
            import pdb;pdb.set_trace()

    if ext == 'pk':
        with open(args.output_file, 'wb') as f:
            pickle.dump(original_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif ext == 'json':
        with open(args.output_file, "w") as json_file:
            json.dump(original_dataset, json_file)