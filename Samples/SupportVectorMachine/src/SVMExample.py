import os

import cv2
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from Data import ReadData
from Parameter import ReadTrainParameters
from ExtractFeatures import GetGradientFromImg, GetHogFeatures


class SVM:
    def __init__(self, train_data_x, train_data_y, test_data_x, test_data_y):
        self._train_data_x = train_data_x
        self._train_data_y = train_data_y
        self._test_data_x = test_data_x
        self._test_data_y = test_data_y
        self._clf = None

    def train(self):
        self._clf = OneVsRestClassifier(SVC(kernel='linear', decision_function_shape='ovr'))
        self._clf.fit(self._train_data_x, self._train_data_y)

    def predict(self):
        pred = self._clf.predict(self._test_data_x)
        return pred

    def get_precision(self, pred):
        arr = (pred == self._test_data_y)
        vals, counts = np.unique(arr, return_counts=True)
        accuracy = counts[np.where(vals == True)] / len(self._test_data_y)
        return accuracy


if __name__=="__main__":

    print('SVM is started.')
    train_prm_file = 'train_param.json'
    train_prm_file_path = os.path.join(os.path.dirname(__file__), train_prm_file)
    train_prm = ReadTrainParameters(train_prm_file_path)

    train_img_list, train_cls = ReadData(os.path.join(os.path.dirname(__file__), train_prm['train_img_folder_path']))
    train_fd_list, train_hog_img_list = GetHogFeatures(train_img_list)

    predict_img_list, predict_cls = ReadData(os.path.join(os.path.dirname(__file__), train_prm['predict_img_folder_path']))
    predict_fd_list, predict_hog_img_list = GetHogFeatures(predict_img_list)

    train_data_frame = np.hstack((train_fd_list, train_cls))
    predict_data_frame = np.hstack((predict_fd_list, predict_cls))

    x_train, y_train = train_data_frame[:, :-1], train_data_frame[:, -1:].ravel()
    x_test, y_test = predict_data_frame[:, :-1], predict_data_frame[:, -1:].ravel()

    svm_ = SVM(x_train, y_train, x_test, y_test)
    svm_.train()
    y_pred = svm_.predict()
    np.savetxt(os.path.join(os.path.dirname(__file__), train_prm['predict_output_file_path']),
               y_pred, fmt='%d', delimiter=',')

    print("Accuracy: " + str(svm_.get_precision(y_pred)))
    print(classification_report(y_test, y_pred))

    print('End')