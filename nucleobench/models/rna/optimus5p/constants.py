import os

ZENODO_HDF5_URL_ = "https://zenodo.org/record/1481198/files/retrained_main_MRL_model.hdf5?download=1"
SEQ_LEN_ = 50

cur_file_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_WEIGHTS_ = os.path.join(cur_file_dir, "optimus_5p_weights.pt")
