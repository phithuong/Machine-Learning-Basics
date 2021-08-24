'''
Linear model: Y = w0 + w1 * X
Lost function: L = (1/2) * (Y_pred - Y)**2 = (1/2) * ((w0 + w1 * X) - Y)**2
dL/dw0 = (w0 + w1 * X) - Y
dL/dw0 = X * ((w0 + w1 * X) - Y)
'''

import os
import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(data_x, data_y, lr=0.000001, iter=100):
    """Calculate weights of linear regression model

    Args:
        data_x ([ndarray]): [input data]
        data_y ([ndarray]): [output data]
        lr (float, optional): [learning rate]. Defaults to 0.001.
        iter (int, optional): [numbẻ of interator]. Defaults to 1000.

    Returns:
        [ndarray]: [calculated weights]
    """
    n = len(data_x)
    in_data = np.append(np.ones((1, n)).reshape(n, 1), data_x, axis=1)

    loss_vals = np.zeros(iter)
    w = np.array([0.0, 1.0])

    for i in range(iter):
        Y_pred = np.matmul(in_data, w)
        grad_w0 = Y_pred - data_y
        grad_w1 = np.dot(data_x.reshape(n), (Y_pred - data_y))

        loss_vals[i] = (1/2) * np.sum((Y_pred - data_y) ** 2)
        print(loss_vals[i])

        w[0] -= lr * np.sum(grad_w0)
        w[1] -= lr * np.sum(grad_w1)

    return w, loss_vals


if __name__ == '__main__':
    data_file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    data = np.genfromtxt(data_file_path, dtype=np.float, delimiter=',', skip_header=1)
    data_x = data[:, 0:data.shape[1] - 1]
    data_y = data[:, data.shape[1] - 1]
    plt.scatter(data_x, data_y)

    w, loss_vals = gradient_descent(data_x, data_y)
    in_data = np.append(np.ones((1, data_x.shape[0])).reshape(data_x.shape[0], 1), data_x, axis=1)
    Y_pred = np.matmul(in_data, w)
    plt.scatter(data_x, Y_pred)
    plt.show()

    print(w)