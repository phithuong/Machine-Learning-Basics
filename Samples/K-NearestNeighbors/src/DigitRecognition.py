import os
import sys
sys.path.append('../')

from Parameter import read_knn_param
from Data import get_train_data, get_test_data

import cv2
import numpy as np

class DigitRecognition:
    def __init__(self, knn_prm, train_data_x, train_data_y, test_data_x, test_data_y):
        self._k = knn_prm['k']
        self._cls = knn_prm['cls']
        self._img_width = knn_prm['img_width']
        self._img_height = knn_prm['img_height']
        self._train_data_x = train_data_x
        self._train_data_y = train_data_y
        self._test_data_x = test_data_x
        self._test_data_y = test_data_y

    def predict(self):
        predict = []
        N1 = len(self._train_data_x)
        N2 = len(self._test_data_x)

        for ii in range(N2):
            variance = np.var(self._train_data_x - self._test_data_x[ii, :], axis=1)
            tmp_idx = np.argpartition(variance, self._k)[:self._k]
            tmp_cls = self._train_data_y[tmp_idx].tolist()
            
            counts = [0] * len(self._cls)
            for tmp in tmp_cls:
                for jj, cls in enumerate(self._cls):
                    if cls == tmp:
                        counts[jj] += 1
                        break
            cls = self._cls[counts.index(max(counts))]
            predict.append(cls)

        return predict


if __name__=="__main__":
    # Read data and parameters
    knn_parameter_file_path = os.path.join(os.path.dirname(__file__), '../knn_prm.json')
    knn_prm = read_knn_param(knn_parameter_file_path)

    train_data_img_path = os.path.join(os.path.dirname(__file__), '../train_data/digits.png')
    train_data_file_path = os.path.join(os.path.dirname(__file__), '../train_data/train_data.csv')
    train_data_x, train_data_y = get_train_data(knn_prm['img_width'], knn_prm['img_height'],
                                        train_data_img_path, train_data_file_path)

    test_data_img_path = os.path.join(os.path.dirname(__file__), '../test_data')
    test_data_x, test_data_y = get_test_data(knn_prm['img_width'], knn_prm['img_height'],
                                        test_data_img_path)

    # Create instance
    dr = DigitRecognition(knn_prm, train_data_x, train_data_y, test_data_x, test_data_y)
    predict = dr.predict()

    print(predict)
