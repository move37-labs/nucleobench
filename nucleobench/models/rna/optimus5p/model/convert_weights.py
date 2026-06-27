import os
import h5py
import torch
import numpy as np
import requests
from tqdm import tqdm

MODEL_URL = "https://zenodo.org/record/1481198/files/retrained_main_MRL_model.hdf5?download=1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "optimus_5p_weights.pt")
DOWNLOAD_PATH = os.path.join(SCRIPT_DIR, "retrained_main_MRL_model.hdf5")

# Post-processing constants from Kipoi model.yaml
# https://raw.githubusercontent.com/kipoi/models/master/Optimus_5Prime/model.yaml
POSTPROC_MEAN = 5.58621521
POSTPROC_SD = 1.34657403

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return
    
    print(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as t:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    print("Download complete.")

def traverse_h5(obj, datasets):
    if isinstance(obj, h5py.Dataset):
        datasets.append(obj)
    elif isinstance(obj, h5py.Group):
        for key in obj:
            traverse_h5(obj[key], datasets)

def convert_weights(output_path=None):
    if output_path is None:
        output_path = OUTPUT_WEIGHTS_PATH
        
    # 1. Download
    download_file(MODEL_URL, DOWNLOAD_PATH)
    
    # 2. Open HDF5
    f = h5py.File(DOWNLOAD_PATH, 'r')
    
    # 3. Collect all datasets (weights)
    all_datasets = []
    traverse_h5(f, all_datasets)
    
    # We expect 10 datasets: 5 layers * (kernel + bias)
    # They usually come in order if we sort by name, but Keras naming can be tricky (conv1d, conv1d_1, etc.)
    # Let's inspect names to map them correctly.
    
    # Keras layer names often look like 'model_weights/conv1d_1/conv1d_1/kernel:0'
    # We need to sort them to match our architecture order:
    # Conv1 -> Conv2 -> Conv3 -> Dense1 -> Dense2
    
    # Filter for kernel and bias
    kernels = [d for d in all_datasets if 'kernel' in d.name or 'W' in d.name] # 'W' is older Keras sometimes
    biases = [d for d in all_datasets if 'bias' in d.name or 'b' in d.name]
    
    # Sort by name. Assuming lexicographical order roughly matches layer order or numbers help.
    # conv1d_1, conv1d_2, conv1d_3, dense_1, dense_2
    kernels.sort(key=lambda x: x.name)
    biases.sort(key=lambda x: x.name)
    
    print("Found kernels:", [k.name for k in kernels])
    print("Found biases:", [b.name for b in biases])
    
    if len(kernels) != 5 or len(biases) != 5:
        print("WARNING: unexpected number of weight tensors. Please check the structure.")
    
    # Map to PyTorch state dict
    state_dict = {}
    
    # Layers in our PyTorch model
    layer_names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
    
    for i, layer_name in enumerate(layer_names):
        print(f"Processing {layer_name}...")
        k_dset = kernels[i]
        b_dset = biases[i]
        
        k_np = k_dset[:]
        b_np = b_dset[:]
        
        print(f"  Original kernel shape: {k_np.shape}")
        print(f"  Original bias shape: {b_np.shape}")
        
        # Transpose logic
        if 'conv' in layer_name:
            # Keras Conv1D: (Kernel, Input, Output) -> PyTorch: (Output, Input, Kernel)
            # Permute (2, 1, 0)
            k_torch = torch.tensor(k_np).permute(2, 1, 0)
        else:
            # Keras Dense: (Input, Output) -> PyTorch: (Output, Input)
            # Permute (1, 0)
            k_torch = torch.tensor(k_np).permute(1, 0)
            
        b_torch = torch.tensor(b_np)
        
        print(f"  Transposed kernel shape: {k_torch.shape}")
        
        # Apply scaling to the last layer (fc2)
        if layer_name == 'fc2':
            print(f"Applying post-processing scaling to {layer_name}...")
            # y_final = (Wx + b) * SD + MEAN = W(x * SD) + (b * SD + MEAN)
            # W_scaled = W * SD
            # b_scaled = b * SD + MEAN
            k_torch = k_torch * POSTPROC_SD
            b_torch = b_torch * POSTPROC_SD + POSTPROC_MEAN
        
        state_dict[f'{layer_name}.weight'] = k_torch
        state_dict[f'{layer_name}.bias'] = b_torch
        
    # Save
    print(f"Saving weights to {output_path}...")
    torch.save(state_dict, output_path)
    print("Done.")

if __name__ == "__main__":
    convert_weights()
