import os

ZENODO_URL_ = "https://zenodo.org/api/records/6326409/files/datasets.zip/content"
MODEL_PATH_IN_ZIP_ = "datasets/deeplearning/train_gru/f0_c0/train/model0_best.h5"
SEQ_LEN_ = 12288

cur_file_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_WEIGHTS_PATH_ = os.path.join(cur_file_dir, "model", "model0_best.h5")
