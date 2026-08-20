"""Tests for ChromBPNetOracle.

Uses override_model so weights are not downloaded. Run with:
    pytest nucleobench/models/chrombpnet/model_def_test.py
"""

import pytest

from nucleobench.common import testing_utils
from nucleobench.models.chrombpnet import constants as cb_constants
from nucleobench.models.chrombpnet import model_def

model_args = {
    "add_unsqueeze_to_output": True,
    "call_is_on_strings": False,
    "flip_sign": False,
}


def test_model_def_sanity():
    m = model_def.ChromBPNetOracle(
        cell_type="Adrenal_c0",
        override_model=testing_utils.CountLetterModel(**model_args),
    )
    ret = m.inference_on_strings(["AAA", "CCC", "TTT", "GGG", "ACT"])
    assert list(ret.shape) == [5]


def test_tism_correctness():
    """Check that TISM on a C-count network knows that Cs are important."""
    m = model_def.ChromBPNetOracle(
        cell_type="Adrenal_c0",
        override_model=testing_utils.CountLetterModel(vocab_i=1, **model_args),
    )
    base_str = "ATCCA"
    _, tism = m.tism(base_str)
    for base_nt, tism_dict in zip(base_str, tism):
        assert base_nt not in tism_dict
        if base_nt == "C":
            assert tism_dict["A"] == tism_dict["T"] == tism_dict["G"]
            assert tism_dict["A"] > 0
        else:
            for nt in ["A", "T", "G"]:
                if nt == base_nt:
                    continue
                assert tism_dict[nt] == 0
            assert tism_dict["C"] < 0


def test_tism_consistency():
    """TISM on a single nucleotide should match the full-string TISM."""
    m = model_def.ChromBPNetOracle(
        cell_type="Adrenal_c0",
        override_model=testing_utils.CountLetterModel(**model_args),
    )
    base_str = "ATCCA"
    v1, tism1 = m.tism(base_str)
    single_bp_tisms = [m.tism(base_str, [idx]) for idx in range(len(base_str))]

    for idx in range(len(single_bp_tisms)):
        v2, tism2 = single_bp_tisms[idx]
        assert v1 == v2
        assert len(tism2) == 1
        for k, v in tism2[0].items():
            assert v == tism1[idx][k]


def test_debug_init_args():
    args = model_def.ChromBPNetOracle.debug_init_args()
    assert args["cell_type"] == "Adrenal_c0"
    model_def.ChromBPNetOracle(
        **args, override_model=testing_utils.CountLetterModel(**model_args)
    )


def test_init_parser_accepts_adrenal_c0():
    parser = model_def.ChromBPNetOracle.init_parser()
    parsed = parser.parse_args(["--cell_type", "Adrenal_c0"])
    assert parsed.cell_type == "Adrenal_c0"


def test_available_models_count():
    assert len(cb_constants.AVAILABLE_MODELS_) == 94
    assert "Adrenal_c0" in cb_constants.AVAILABLE_MODELS_
    assert "Muscle_c4" in cb_constants.AVAILABLE_MODELS_


def test_rejects_bare_string():
    m = model_def.ChromBPNetOracle(
        cell_type="Adrenal_c0",
        override_model=testing_utils.CountLetterModel(**model_args),
    )
    with pytest.raises(ValueError, match="list of strings"):
        m("AAA")
