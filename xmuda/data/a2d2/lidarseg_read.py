import os
import numpy as np
patch = '/home/viplab/sdb1_dir/datasets/Lidarseg/lidarseg/v1.0-trainval/'
lidarseg_labels_filename = patch+"fff6a4193f0d41f3931a90364e72c137_lidarseg.bin"
points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)  # [num_points]
print(points_label)
