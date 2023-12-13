import numpy as np
from numpy import load
import os

test_scene_info = "./assets/megadepth_test_1500_scene_info"

# change test_scene_info
for filename in os.listdir(test_scene_info):
    f_npz = os.path.join(test_scene_info, filename)
    if not f_npz.endswith(".npz"):
        print("skip file: ", f_npz)
        continue
    data = load(f_npz, allow_pickle=True)
    for count, image_path in enumerate(data['image_paths']):
        if image_path is not None:
            if 'Undistorted_SfM' in image_path:
                data['image_paths'][count] = data['depth_paths'][count].replace('depths', 'imgs').replace('h5', 'jpg')
    data['pair_infos'] = np.asarray(data['pair_infos'], dtype=object)
    np.savez(f_npz, **data)
    print("Saved to ", f_npz)
