"""Gene expression from DNA: https://www.nature.com/articles/s41592-021-01252-x

Accessed through gRelu: https://github.com/Genentech/gReLU

Usage follows this tutorial:
https://github.com/Genentech/gReLU/blob/main/docs/tutorials/4_design.ipynb

To test on real data:
```zsh
python -m nucleobench.models.grelu.enformer.model_def
```
"""

from typing import Optional, Union

import argparse
import numpy as np
import torch

from nucleobench.models.grelu import model_def as grelu_md
from nucleobench.models.grelu.enformer import constants


class Enformer(grelu_md.GReluModel):
    """Predicts gene expression from DNA.
    https://www.nature.com/articles/s41592-021-01252-x
    """

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("Enformer init args")
        group.add_argument("--aggregation_type", type=str, required=True, 
                           choices=['muscle_CAGE', 'muscle_not_liver'])
        return parser
    
    @staticmethod
    def debug_init_args():
        return {
            'aggregation_type': 'muscle_CAGE',
            'run_sanity_checks': False,
        }
        
    # List of possible tasks.
    # Set in child models.
    POSSIBLE_TASKS_ = constants.ENFORMER_TASKS_

    def __init__(
        self,
        aggregation_type: str,
        override_model: Optional[torch.nn.Module] = None,
        override_aggregation = None,
        run_sanity_checks: bool = True,
    ):
        """Init.
        
        Aggregation is complicated. We enumerate the options as strings, and use premade aggregation
        functions for well thought-out options, instead of asking the user to provide them, so the
        aggregation can be easily tracked and tested.
        """
        super().__init__(
            project=constants.ENFORMER_PROJECT_,
            model_name=constants.ENFORMER_MODEL_NAME,
            override_model=override_model,
            expected_sequence_length=constants.ENFORMER_TRAIN_LEN_,
        )
        
        if aggregation_type not in ['muscle_CAGE', 'muscle_not_liver']:
            raise ValueError(f'Unknown aggregation type: {aggregation_type}')

        # Create aggregation method.
        if override_aggregation is None:
            positive_idxs, negative_idxs = idxs_by_name(aggregation_type)
            def _aggregation(model_out: torch.Tensor) -> torch.Tensor:
                assert model_out.ndim == 3
                assert model_out.shape[1] == len(constants.ENFORMER_TASKS_)
                
                ret = (torch.sum(model_out[:, positive_idxs], axis=(1, 2)) - 
                       torch.sum(model_out[:, negative_idxs], axis=(1, 2)))
                assert ret.ndim == 1
                return ret
            self.aggregation = _aggregation
        else:
            self.aggregation = override_aggregation
                    
        # Sanity check inference.
        if run_sanity_checks:
            ret = self.model(self.string_to_onehot(['A' * self.sequence_length]))
            assert ret.shape == (1, 5313, 896), ret.shape
            # Apparently 
            ret = self.model(self.string_to_onehot(['A' * 82_000]))
            assert ret.shape == (1, 5313, 1), ret.shape
            
            ret = self.inference_on_strings(['A' * self.sequence_length])
            assert ret.ndim == 1
        


    def inference_on_tensor(
        self, 
        x: torch.Tensor,
        return_debug_info: bool = False,
        ) -> Union[torch.Tensor, tuple[torch.Tensor, np.ndarray]]:
        """Run inference on a one-hot tensor."""
        assert x.ndim == 3  # Batched.
        assert x.shape[1] == 4
        #assert x.shape[2] == self.sequence_length, x.shape

        m_out = self.model(x)
        assert m_out.ndim == 3
        assert m_out.shape[1] == len(constants.ENFORMER_TASKS_), m_out.shape
        
        ret = self.aggregation(m_out)
        assert ret.ndim == 1, ret.shape
        
        # Always return something that should be minimized, so flip the sign.
        ret *= -1
        
        if return_debug_info:
            return (ret, m_out.clone().detach().numpy())
        else: 
            return ret
    
    
def idxs_by_name(aggregation_type: str) -> tuple[list[int], list[int]]:
    if aggregation_type == 'muscle_CAGE':
        positive_idxs = constants.muscle_cage_idx()
        negative_idxs = []
    elif aggregation_type == 'muscle_not_liver':
        positive_idxs = constants.activate_muscle_idx() + constants.deactivate_liver_idx()
        negative_idxs = constants.activate_liver_idx() + constants.deactivate_muscle_idx()
    else:
        raise ValueError(f'Unknown aggregation type: {aggregation_type}')
    return positive_idxs, negative_idxs


if __name__ == "__main__":
    # Test with a real model.
    import time, tqdm
    print(f'Starting Enformer...')
    m = Enformer(aggregation_type='muscle_not_liver')
    ntimes = 10
    s_time = time.time()
    # Runs at roughly 15s / iteration, on my macbook.
    for _ in tqdm.trange(ntimes):
        ret = m(["A" * 196_608])
    e_time = time.time()
    print(f'Finished in {e_time - s_time} seconds: {(e_time - s_time) / ntimes} s / iter')