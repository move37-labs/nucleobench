from . import constants
from .model.saluki import Saluki

# Weights are committed at nucleobench/models/rna/saluki/model/model0_best.h5
# They originate from Zenodo DOI 10.5281/zenodo.6326409 (CC BY 4.0, Agarwal & Kelley 2022).
# The file lives inside datasets.zip at:
#   datasets/deeplearning/train_gru/f0_c0/train/model0_best.h5
#
# To download them independently (e.g. if the committed file is removed):
#
#   import fsspec, zipfile, shutil
#   ZENODO_URL = "https://zenodo.org/api/records/6326409/files/datasets.zip/content"
#   MODEL_IN_ZIP = "datasets/deeplearning/train_gru/f0_c0/train/model0_best.h5"
#   with fsspec.open(ZENODO_URL, "rb", block_size=2**22) as f:
#       zf = zipfile.ZipFile(f)  # Range requests; only fetches central dir + target member
#       with zf.open(MODEL_IN_ZIP) as src, open(dest, "wb") as out:
#           shutil.copyfileobj(src, out)
#
# TODO(joelshor): switch load_model() to fetch live from Zenodo (using the fsspec
# approach above) and remove the committed model0_best.h5 binary.


def load_model(weights_path: str | None = None) -> Saluki:
    if weights_path is None:
        weights_path = constants.LOCAL_WEIGHTS_PATH_
    return Saluki(weights_path=weights_path)
