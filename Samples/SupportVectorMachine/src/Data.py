import os

import cv2
import numpy as np
from skimage.transform import resize

def ReadData(data_folder_path):
    img_list = []
    labels = []
    fns = os.listdir(data_folder_path)
    for fn in fns:
        file_path = os.path.join(data_folder_path, fn)
        img = cv2.imread(file_path)
        img_list.append(resize(img, (128, 64)))

        pre = fn.split('_')[0]
        if pre == 'Cir':
            labels.append(0)
        elif pre == 'Squa':
            labels.append(1)
        elif pre == 'tri':
            labels.append(2)
    
    img_list = np.array(img_list)
    n = len(labels)
    labels = np.array(labels).reshape(n, 1)
    return img_list, labels