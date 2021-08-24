# Logistic regression
# Logistic function (sigmoid function): y = 1 / (1 + e^-n) = w0 + w1.x
# Loss function: L = −(y ∗ np.log(y_pred) + (1 − y) ∗ np.log(1 − y_pred))
# dL/dw0 = y_pred - y
# dL/dw1 = x ∗ (y_pred - y)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, lr=0.001, iter_num=10000):
        self._lr = lr
        self._iter_num = iter_num

    def sigmoid(self, x):
        x_ = np.array(x).astype(np.float64)
        y = 1 / (1 + np.e**(-x))
        return y

    def loss(self, y, y_pred):
        L = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return L

    def update_weight(self, w, x, y, y_pred):
        w = w - self._lr * np.dot(x.T, y_pred - y)
        return w


def split_data(data_x, data_y, train_data_percent=0.75):
    data_len = len(data_x)
    idxes = np.arange(data_len)
    np.random.shuffle(idxes)

    idx = int(data_len * train_data_percent)
    train_data_x = data_x[idxes[0:idx]]
    train_data_y = data_y[idxes[0:idx]]
    test_data_x = data_x[idxes[idx:-1]]
    test_data_y = data_y[idxes[idx:-1]]

    return train_data_x, train_data_y, test_data_x, test_data_y


def main():
    reg = LogisticRegression()

    # #sample data 1
    # data_len = 100
    # x = np.arange(1, data_len+1, 1).reshape(data_len, 1)
    # ones = np.ones((1, data_len)).reshape(data_len, 1)
    # x = np.hstack((ones, x))
    # y = []
    # [y.extend([0, 1]) for i in range(50)]
    # y = np.array(y)

    # #sample data 2
    data_len = 100
    value_less_than_100 = np.arange(0, int(data_len), 1)
    value_greater_than_100 = np.arange(data_len, 2*data_len, 1)

    data_x = np.concatenate((value_less_than_100, value_greater_than_100))
    data_x = data_x.reshape(2*data_len, 1)
    ones = np.ones((1, 2*data_len)).reshape(2*data_len, 1)
    data_x = np.hstack((ones, data_x))
    data_y = []
    [data_y.append(0) for i in range(100)]
    [data_y.append(1) for i in range(100)]
    data_y = np.array(data_y)

    train_data_x, train_data_y, test_data_x, test_data_y = split_data(data_x, data_y, train_data_percent=0.75)

    # sample data 3
    # data_file_path = os.path.join(os.path.dirname(__file__), 'Social_Network_Ads.csv')
    # data = np.genfromtxt(data_file_path, delimiter=',', skip_header=1)
    # data_len = len(data)
    # x = data[:, 2].reshape(data_len, 1)
    # ones = np.ones((1, data_len)).reshape(data_len, 1)
    # x = np.hstack((ones, x))
    # y = data[:, 4]

    weights = []
    loss_vals = []

    w = np.array([0.0, 0.1])

    for i in range(reg._iter_num):
        fx = np.dot(train_data_x, w)
        y_pred = reg.sigmoid(fx)

        idxes = np.where(y_pred == 0)
        y_pred[idxes] = 10e-6

        L = reg.loss(train_data_y, y_pred)
        w = reg.update_weight(w, train_data_x, train_data_y, y_pred)

        loss_vals.append(L)
        weights.append(w)

    print(w)
    print(y_pred)
    fx = np.dot(test_data_x, w)
    y_pred = reg.sigmoid(fx)
    plt.scatter(test_data_x[:, 1], test_data_y, marker='o')
    plt.scatter(test_data_x[:, 1], y_pred, marker='o')
    plt.show()


if __name__=='__main__':
    main()