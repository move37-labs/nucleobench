import pytest

from nucleobench.common import string_utils, testing_utils
from nucleobench.models.rna.optimus5p import model_def

model_args = {
    "add_unsqueeze_to_output": True,
    "call_is_on_strings": False,
    "flip_sign": False,
}


def test_optimus5p_sanity():
    """Basic sanity test with override model."""
    m = model_def.Optimus5P(override_model=testing_utils.CountLetterModel(**model_args))
    ret = m.inference_on_strings(["AAA", "CCC", "TTT", "GGG", "ACT"])
    assert list(ret.shape) == [5]


def test_optimus5p_tism_correctness():
    """Check that TISM on a C-count network knows that Cs are important."""
    # Count vocab index 1 (which is 'C' in standard ACGT vocab)
    m = model_def.Optimus5P(
        override_model=testing_utils.CountLetterModel(vocab_i=1, **model_args)
    )
    base_str = "ATCCA"
    _, tism = m.tism(base_str)
    for base_nt, tism_dict in zip(base_str, tism):
        assert base_nt not in tism_dict
        if base_nt == "C":
            # Everything should be the same.
            assert tism_dict["A"] == tism_dict["T"] == tism_dict["G"]
            assert (
                tism_dict["A"] > 0
            )  # decrease the count, increase the energy (due to -1 multiplier).
        else:
            # TISM should show that the greatest change comes from adding a 'C'.
            for nt in ["A", "T", "G"]:
                if nt == base_nt:
                    continue
                assert tism_dict[nt] == 0  # changing to a non-C should be no change.
            assert tism_dict["C"] < 0


def test_optimus5p_gradient_flow():
    """Test that gradients can flow through the model for gradient-based optimization."""
    m = model_def.Optimus5P(override_model=testing_utils.CountLetterModel(**model_args))

    # Create a simple test sequence
    test_seq = "ACGT"
    onehot = string_utils.dna2tensor(test_seq, vocab_list=m.vocab)
    batch_onehot = onehot.unsqueeze(0).float()  # Add batch dimension and ensure float
    batch_onehot.requires_grad = True

    # Forward pass
    output = m.inference_on_tensor(batch_onehot)
    loss = output.sum()

    # Backward pass
    try:
        loss.backward()
        assert batch_onehot.grad is not None, "No gradients computed"
    except Exception as e:
        pytest.fail(f"Backward pass failed: {e}")


def test_optimus5p_sliding_window():
    """Test that sliding window logic works correctly for longer sequences."""
    m = model_def.Optimus5P(
        window_size=50,
        stride=5,
        override_model=testing_utils.CountLetterModel(**model_args),
    )

    # Sequence of length 100
    seq = "A" * 100
    ret = m([seq])
    assert list(ret.shape) == [1]
