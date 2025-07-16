
import os
import tqdm
import pickle

from scenedetect import detect, ContentDetector

if __name__ == '__main__':

    root_dir = '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/recent_bench/25CVPR_MLVU/mlvu_test/video'
    save_dir = '/data3/gtlim/workspace/26CVPR_VideoLLM/LVU/data/recent_bench/25CVPR_MLVU/mlvu_test/pyscenedetect'

    for vid in tqdm.tqdm(os.listdir(root_dir)):
        try:
            scene_list = detect(os.path.join(root_dir,vid), ContentDetector())

            if os.path.isdir(os.path.join(save_dir,vid))==False:
                os.mkdir(os.path.join(save_dir,vid))
                results = {
                    'video' : vid,
                    'list' : scene_list
                }

                with open(os.path.join(save_dir, vid, "scene_list.pk"), 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            continue
                    