import json

TRAIN_PARAMETER_KEYS = {
    "TrainImgFolderPath": "train_img_folder_path",
    "PredictImgFolderPath": "predict_img_folder_path",
    "PredictOutputFilePath": "predict_output_file_path"
}

def ReadTrainParameters(train_parameter_file_path):
    with open(train_parameter_file_path, 'r') as f:
        train_prm = json.load(f)

        for key, value in train_prm.items():
            if key not in TRAIN_PARAMETER_KEYS.values():
                msg = '{} key is not nesscessory in train parameters.'
                raise Exception(msg)
    return train_prm