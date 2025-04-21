"""Tests for model_def.py

To test:
pytest nucleobench/models/malinois/model_def_test.py
"""

import pytest

import numpy as np
import torch

from nucleobench.common import string_utils
from nucleobench.common import testing_utils

from nucleobench.models.malinois import model_def


model_args = {'extra_channels': 2, 'call_is_on_strings': False}

def test_energy_calc_from_output_tensor_sanity():
    for target_feature in [0, 1, 2]:
        _ = model_def.energy_calc_from_output_tensor(
            torch.Tensor([[.2, .3, 1.0],
                          [1.5, 2., -20]]),
            target_feature=target_feature,
            bending_factor=1.0,
        )


def test_energy_calc_from_output_tensor_correctness():
    energy = model_def.energy_calc_from_output_tensor(
        torch.Tensor([[.2, .3, 1.0],
                      [1.5, 7., -20]]),
        target_feature=0,
        bending_factor=0.0,
        a_min=0.5,
        a_max=5,
    )
    ret = energy.detach().numpy()
    assert np.allclose(ret, np.array([1.0 - 0.5, 5. - 1.5]))


@pytest.mark.parametrize('return_debug_info', [True, False])
def test_malinois_sanity(return_debug_info):
    m = model_def.Malinois(
         model_artifact=None,
         target_feature=0,
         bending_factor=1.0,
         override_model=testing_utils.CountLetterModel(**model_args))
    ret = m.inference_on_strings(['AAA', 'CCC', 'TTT', 'GGG', 'ACT'], return_debug_info)
    if return_debug_info:
         assert list(ret[0].shape) == [5]
         assert list(ret[1]['malinois_output'].shape) == [5, 3]
    else:
     assert list(ret.shape) == [5]


def test_add_flank():
    m = model_def.Malinois(
         model_artifact=None,
         target_feature=0,
         bending_factor=1.0,
         override_model=testing_utils.CountLetterModel(**model_args),
         flank_length=2,
         )
    dnas = ['AAA', 'CCC', 'TTT', 'GGG', 'ACT']
    tensor = string_utils.dna2tensor_batch(dnas)

    ret = m.add_flanks_tensor(tensor)
    assert list(ret.shape) == [5, 4, 7]


def test_smoothgrad_sanity():
     """Smoke test, and that smoothgrad inference is the same as normal inference."""
     m = model_def.Malinois(
          model_artifact=None,
          target_feature=0,
          bending_factor=1.0,
          override_model=testing_utils.CountLetterModel(**model_args))
     y1, smooth_grad = m.smoothgrad_on_string('ATAAG')
     y2 = m.inference_on_strings(['ATAAG'])
     assert y1 == y2
     assert isinstance(smooth_grad, list)
     assert len(smooth_grad) == 5


def test_smoothgrad_correctness():
     """Check that smoothgrad on an C-count network knows that Cs are important."""
     m = model_def.Malinois(
          model_artifact=None,
          target_feature=0,
          bending_factor=0.0,
          a_min=None,
          a_max=None,
          smoothgrad_stdev=0.01,
          smoothgrad_times=2,
          vocab=['A', 'C', 'G', 'T'],
          override_model=testing_utils.CountLetterModel(vocab_i=1, **model_args))
     _, smooth_grad = m.smoothgrad_on_string('ATCCA')
     for smooth_grad_dict in smooth_grad:
         # Output is energy, which is attempted to be minimized, so sign is negative.
         assert smooth_grad_dict['C'] < smooth_grad_dict['A']
         assert smooth_grad_dict['C'] < smooth_grad_dict['G']
         assert smooth_grad_dict['C'] < smooth_grad_dict['T']


def test_tism_correctness():
     """Check that TISM on an C-count network knows that Cs are important."""
     m = model_def.Malinois(
          model_artifact=None,
          target_feature=0,
          bending_factor=0.0,
          a_min=None,
          a_max=None,
          smoothgrad_stdev=0.01,
          smoothgrad_times=2,
          vocab=['A', 'C', 'G', 'T'],
          override_model=testing_utils.CountLetterModel(vocab_i=1, **model_args))
     base_str = 'ATCCA'
     _, tism = m.tism(base_str)
     for base_nt, tism_dict in zip(base_str, tism):
         assert base_nt not in tism_dict
         if base_nt == 'C':
              # Everything should be the same.
              assert tism_dict['A'] == tism_dict['T'] == tism_dict['G']
              assert tism_dict['A'] > 0  # decrease the count, increase the energy.
         else:
              # TISM should show that the greatest change comes from adding a 'C'.
              for nt in ['A', 'T', 'G']:
                   if nt == base_nt: continue
                   assert tism_dict[nt] == 0  # changing to a non-C should be no change.
              assert tism_dict['C'] < 0


def test_tism_consistency():
     """TISM on a single nucleotide should be the same as the string.."""
     m = model_def.Malinois(
          model_artifact=None,
          target_feature=0,
          bending_factor=0.0,
          a_min=None,
          a_max=None,
          smoothgrad_stdev=0.01,
          smoothgrad_times=2,
          vocab=['A', 'C', 'G', 'T'],
          override_model=testing_utils.CountLetterModel(vocab_i=1, **model_args))
     base_str = 'ATCCA'
     v1, tism1 = m.tism(base_str)
     single_bp_tisms = [m.tism(base_str, idx) for idx in range(len(base_str))]
     
     for idx in range(len(single_bp_tisms)):
         v2, tism2 = single_bp_tisms[idx]
         assert v1 == v2
         assert len(tism2) == 1
         for k, v in tism2[0].items():
               assert v == tism1[idx][k]
               
@pytest.mark.parametrize("flank_length", [0, 100, 200])
def test_flank_length(flank_length: int):
     """Check that inference works with various flank sizes."""
     seq_len = 600 - 2 * flank_length
     
     m = model_def.Malinois(
          model_artifact=None,
          flank_length=flank_length,
          target_feature=0,
          bending_factor=0.0,
          a_min=None,
          a_max=None,
          smoothgrad_stdev=0.01,
          smoothgrad_times=2,
          vocab=['A', 'C', 'G', 'T'],
          override_model=testing_utils.CountLetterModel(**model_args),
          check_input_shape=True)
     m(['A' * seq_len])