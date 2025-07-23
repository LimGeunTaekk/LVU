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
    original_dataset = pickle.load(open(args.original_file,'rb'))

    for i in range(len(frame_idx_list)):
        find = False
        for j in range(len(original_dataset)):
            qid = frame_idx_list[i]['qid']
            qid_o = original_dataset[j]['question_id']
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

    with open(args.output_file, 'wb') as f:
        pickle.dump(original_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
