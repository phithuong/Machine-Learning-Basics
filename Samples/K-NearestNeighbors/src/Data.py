import os
import sys

import cv2
import numpy as np

def get_train_data(img_width, img_height, train_data_img_path, train_data_file_path):
    train_data = None

    if os.path.isfile(train_data_file_path):
        train_data = np.genfromtxt(train_data_file_path, dtype=int, delimiter=',')
    else:
        if os.path.isfile(train_data_img_path):
            data_img = np.empty((0, img_width * img_height), dtype=int)
            digits = []
            keys = [i for i in range(48,58)]

            img = cv2.imread(train_data_img_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img, 127, 255, 0)

            # opencv (version 3.3): findContours() return img,, contours, hierachy
            # opencv (version 4.2): findContours() return contours, hierachy
            img, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for ii, cnt in enumerate(contours):
                area_cnt = cv2.contourArea(cnt)
                if area_cnt > 120 and area_cnt < 900:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h > 30:
                        tmp_img = thresh[y:y+h, x:x+w]

                        cv2.destroyAllWindows()
                        cv2.imshow('tmp', tmp_img)
                        key = cv2.waitKey(0)

                        if key == 27:
                            sys.exit()
                        elif key in keys:
                            # Normalize image size (10 x 10)
                            tmp_img_norm = cv2.resize(tmp_img, (img_width, img_height))
                            tmp_img_norm = np.reshape(tmp_img_norm, (1, img_width * img_height))

                            # Save data and label of digits
                            data_img = np.append(data_img, tmp_img_norm)
                            digits.append(int(chr(key)))

            # Save train data
            new_shape = (len(data_img)// (img_width * img_height), (img_width * img_height))
            data_img = np.reshape(data_img, new_shape)
            digits = np.array([digits]).T

            train_data = np.concatenate((data_img, digits), axis=1)
            np.savetxt(train_data_file_path, train_data, fmt='%d', delimiter=',')

    train_data_x = train_data[:, 0:(img_width*img_height)]
    train_data_y = train_data[:, (img_width*img_height)]

    return train_data_x, train_data_y

def get_test_data(img_width, img_height, test_data_dir_path):
    test_data_x = np.empty((0, img_width*img_height), dtype=int)
    test_data_y = np.empty((0,1), dtype=int)

    fns = os.listdir(test_data_dir_path)
    for fn in fns:
        test_data_img_path = os.path.join(test_data_dir_path, fn)

        img = cv2.imread(test_data_img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img, 127, 255, 0)

        img, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_cnts = [cv2.contourArea(cnt) for cnt in contours]

        # Remove contour bounding image
        idx = np.argmax(area_cnts)
        area_cnts.pop(idx)
        contours.pop(idx)

        # Get contour bounding digit
        idx = np.argmax(area_cnts)

        # Get digit
        x,y,w,h = cv2.boundingRect(contours[idx])
        data_img = thresh[y:y+h, x:x+w]
        data_img = cv2.resize(data_img, (img_width, img_height))

        data_img = np.reshape(data_img, (1, img_width*img_height))

        # Append to test_data_x and test_data_y
        test_data_x = np.append(test_data_x, data_img)
        test_data_y = np.append(test_data_y, int(fn.split('.')[0]))

    new_shape = (len(test_data_x)//(img_width*img_height), img_width*img_height)
    test_data_x.resize(new_shape)

    return test_data_x, test_data_y
