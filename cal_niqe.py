import cv2
from basicsr.metrics import calculate_niqe
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def cal_niqe(folder):
    niqes = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg','tif')):
            img_path = os.path.join(folder, filename)
            crop_border = 4
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            unique_colors, counts = np.unique(img[0].reshape(-1), axis=0, return_counts=True)
            if len(unique_colors)==1:
                continue
            niqe = calculate_niqe(img, crop_border=crop_border, input_order='HWC', convert_to='y')
            niqes.append(niqe)

    return np.mean(niqes)

img_path = 'result_images'
niqe = cal_niqe(img_path)
print("NIQE : %.4f" % niqe)
