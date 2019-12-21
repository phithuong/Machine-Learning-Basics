import json

KNN_PARAM_KEY = ['k', 'cls', 'img_width', 'img_height']


def read_knn_param(knn_parameter_file_path):
    knn_prm = {}
    with open(knn_parameter_file_path, 'r') as f:
        prms = json.load(f)
        for key, value in prms.items():
            if key not in KNN_PARAM_KEY:
                msg = '{} is not neccessary key in knn parameters.'.format(key)
                raise Exception(msg)
            knn_prm[key] = value

    return knn_prm
