import os

FT_ZENODO_URL_ = 'https://zenodo.org/records/15043668/files/rinalmo_giga_mrl_ft.pt'
FT_GCP_URL_ = 'https://storage.googleapis.com/nucleobench-models/rinalmo_giga_mrl_ft.pt'

cur_file_dir = os.path.dirname(os.path.abspath(__file__))
CACHE_FT_ = os.path.join(cur_file_dir, 'rinalmo_giga_mrl_ft.pt')  # Optional local location for large pt file.
