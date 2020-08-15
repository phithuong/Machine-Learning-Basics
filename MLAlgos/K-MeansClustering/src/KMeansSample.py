import os

import numpy as np
np.random.seed(1)

def ReadData(data_file_path):
    data = np.genfromtxt(data_file_path, dtype=float, delimiter=',')
    data_x = data[:, 1:(data.shape[1])]
    data_y = data[:, 0].astype(int)
    return data_x, data_y


class Kmeans:
    def __init__(self, k, data_x, data_y=None, max_iter=1000):
        self._k = k
        self._max_iter = max_iter
        self._data_x = data_x
        self._data_y = data_y
        self._centers = None
        self._labels = None

    def cluster(self):
        labels = None
        centers = self._data_x[np.random.choice(self._data_x.shape[0], self._k, replace=False)]

        ii = 0
        while True:
            previous_centers = centers
            labels = Kmeans._assign_labels(self._data_x, centers)
            centers = Kmeans._update_centers(self._data_x, labels, self._k)

            ii += 1
            if np.allclose(previous_centers, centers) or (ii == self._max_iter):
                break

        self._centers = centers
        self._labels = labels

        return centers, labels, ii

    def precision(self):
        if self._data_y is None:
            print('No labels for comparision.')
        else:
            labels = self._labels.copy()
            pair_label_replace = []
            n = 0
            for ii in range(self._k):
                previous_n = n
                n = n + len(self._data_y[self._data_y == ii + 1]) + 1

                label_replace = labels[previous_n:n]
                unique, counts = np.unique(label_replace, return_counts=True)
                idx = np.argmax(counts)

                pair_label_replace.append((ii + 1, unique[idx]))

            for ii in range(len(self._labels)):
                for pair in pair_label_replace:
                    if pair[1] == self._labels[ii]:
                        labels[ii] = pair[0]
                        break

            arr_eval = (labels == self._data_y)
            vals, nums = np.unique(arr_eval, return_counts=True)
            true_num = nums[np.where(vals == True)]

            self._labels = labels
            precision = true_num / len(self._labels)
            return precision


    @staticmethod
    def _assign_labels(data_x, centers):
        n = data_x.shape[0]
        k = centers.shape[0]
        labels = []
        for ii in range(n):
            distance_to_centers = np.sum((np.abs(data_x[ii] - centers)) ** 2, axis=1)
            labels.append(np.argmin(distance_to_centers))
        labels = np.array(labels)
        return labels

    @staticmethod
    def _update_centers(data_x, labels, k):
        new_centers = np.empty((0, data_x.shape[1]))
        for ii in range(k):
            center = np.mean(data_x[labels == ii, :], axis=0)
            new_centers = np.append(new_centers, center)

        new_centers = np.reshape(new_centers, (k, data_x.shape[1]))
        return new_centers


if __name__=="__main__":
    fn = 'wine.data'
    data_file_path = os.path.join(os.path.dirname(__file__), '../data/{}'format(fn))
    data_x, data_y = ReadData(data_file_path)
    km = Kmeans(3, data_x, data_y=data_y, max_iter=10)
    km.cluster()
    precision = km.precision()

    print('Precision: %.2f' % (precision * 100) + '%')
