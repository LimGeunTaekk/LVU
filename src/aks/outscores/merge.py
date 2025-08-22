import os
import json

if __name__ == '__main__':
    
    root_dir = '/data3/gtlim/workspace/LVU/src/aks/outscores/videoevalpro_1.0fps/blip'

    scores = dict()

    for file_ in sorted(os.listdir(root_dir)):
        score_dict = json.load(open(os.path.join(root_dir,file_)))
        print(len(score_dict))
        for key in score_dict.keys():
            scores[key] = score_dict[key]

    with open(os.path.join(root_dir,'scores.json'),'w') as f:
        json.dump(scores,f)

    ori_anno = json.load(open("/data3/gtlim/workspace/LVU/data/benchmarks/videoevalpro/anno/videoevalpro_original.json"))

    cnt = 0

    qid=set()

    for anno in ori_anno:
        qid.add(anno['question_id'])
        if anno['question_id'] not in scores.keys():
            print(anno['question_id'])
            cnt += 1

    print(cnt)
    import pdb;pdb.set_trace()