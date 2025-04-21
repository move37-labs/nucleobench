"""Fetch and load BPNets trained for the Ledidi paper.

This includes a set of BPNet models trained using bpnet-lite along with the 
training parameter files used for evaluating Ledidi. The training parameter files 
include the ENCODE accessions for the signal and peak files used for training. 
The models were trained using PyTorch and can be loaded and used with the bpnet-lite 
repository, where they are instances of the `BPNet` object.

https://zenodo.org/records/14604495

To test one locally:
```zsh
python -m nucleobench.models.bpnet.load_model
```
"""

import os 
import subprocess
import tempfile

import torch

from nucleobench.models.bpnet import constants as bp_constants

RECORDS_ = 'https://zenodo.org/records/14604495'
# curl https://zenodo.org/records/14604495/files/ATAC.torch --output 'ATAC.torch'


def get_url(model_name: str) -> str:
    assert model_name in bp_constants.AVAILABLE_MODELS_
    return f'{RECORDS_}/files/{model_name}.torch'

def download(model_name: str):
    assert model_name in bp_constants.AVAILABLE_MODELS_
    url = get_url(model_name)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, f'{model_name}.torch')
        subprocess.run(['curl', url, '--output', model_path])
        model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
    
    # Text copied from `https://github.com/jmschrei/ledidi/blob/master/tutorials/Tutorial%201%20-%20Design%20of%20Protein%20Binding%20Sites.ipynb`:
    # There are two technical details of the BPNet models we need to account for before they 
    # can be used. First, they take a control track in addition to genome sequence to adjust 
    # for biases in the mapping step. Because Ledidi assumes only a single input to the model, 
    # we need to use ControlWrapper, which just automatically creates and passes in an all-zeroes 
    # control of the same batch size as the data, removing our need to account for it. Second, the 
    # model makes predictions for both profile and counts, so we need to use CountWrapper to remove 
    # the profile output. After using these two wrappers, we have an object that only takes in sequence 
    # and outputs a single number per example. Most model complexities can be removed through the use 
    # of a wrapper like this.
    model = CountWrapper(ControlWrapper(model))

    return model


# TODO(joelshor): Figure out how to import these directly from bpnet.
class CountWrapper(torch.nn.Module):
	"""A wrapper class that only returns the predicted counts.

    Copied from https://github.com/jmschrei/bpnet-lite/blob/master/bpnetlite/bpnet.py,
    under the MIT license.
    
	This class takes in a trained model and returns only the second output.
	For BPNet models, this means that it is only returning the count
	predictions. This is for convenience when using captum to calculate
	attribution scores.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super(CountWrapper, self).__init__()
		self.model = model

	def forward(self, X, X_ctl=None, **kwargs):
		return self.model(X, X_ctl, **kwargs)[1]


# TODO(joelshor): Figure out how to import these directly from bpnet.
class ControlWrapper(torch.nn.Module):
	"""This wrapper automatically creates a control track of all zeroes.
 
    Copied from https://github.com/jmschrei/bpnet-lite/blob/master/bpnetlite/bpnet.py,
    under the MIT license.

	This wrapper will check to see whether the model is expecting a control
	track (e.g., most BPNet-style models) and will create one with the expected
	shape. If no control track is expected then it will provide the normal
	output from the model.
	"""

	def __init__(self, model):
		super(ControlWrapper, self).__init__()
		self.model = model

	def forward(self, X, X_ctl=None):
		if X_ctl != None:
			return self.model(X, X_ctl)

		if self.model.n_control_tracks == 0:
			return self.model(X)

		X_ctl = torch.zeros(X.shape[0], self.model.n_control_tracks,
			X.shape[-1], dtype=X.dtype, device=X.device)
		return self.model(X, X_ctl)
    

if __name__ == '__main__':
    download('GATA2')