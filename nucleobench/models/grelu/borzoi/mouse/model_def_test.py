"""Offline tests for the mouse Borzoi readout, using an injected model."""

import numpy as np
import pytest
import torch

from nucleobench.models.grelu.borzoi import mouse
from nucleobench.models.grelu.borzoi.mouse import BorzoiMouse

ORIGIN_ = mouse.OUTPUT_ORIGIN_


def _task_table():
    """Minimal stand-in for a real checkpoint's `data_params["tasks"]`."""
    names = [f"track_{i}" for i in range(mouse.N_MOUSE_TASKS_)]
    for index, name in [
        (2237, "ENCFF427XNM"),
        (2287, "ENCFF968GGL+"),
        (2288, "ENCFF968GGL-"),
    ]:
        names[index] = name
    return {
        "name": names,
        "sum_stat": ["sum_sqrt"] * mouse.N_MOUSE_TASKS_,
        "scale": [0.3] * mouse.N_MOUSE_TASKS_,
    }


class TinyMouse(torch.nn.Module):
    """Emits a constant profile per track, proportional to `C` content."""

    def __init__(self, tasks=None):
        super().__init__()
        self.data_params = {
            "tasks": _task_table() if tasks is None else tasks,
            "train_seq_len": mouse.BorzoiMouse.sequence_length,
        }

    def forward(self, x):
        value = x[:, 1, ORIGIN_ : ORIGIN_ + 32].mean(dim=1)
        output = torch.zeros(
            (x.shape[0], mouse.N_MOUSE_TASKS_, mouse.N_SPATIAL_BINS_),
            device=x.device,
            dtype=x.dtype,
        )
        scaled = (value[:, None] * torch.tensor([3, 1, 2], device=x.device))[..., None]
        return output.index_copy(
            1,
            torch.tensor([2237, 2287, 2288], device=x.device),
            scaled.expand(-1, 3, mouse.N_SPATIAL_BINS_),
        )


def _model(readout="es_e14", strand="+", exons=None, tasks=None):
    return BorzoiMouse(
        readout,
        strand,
        exons if exons is not None else [(ORIGIN_, ORIGIN_ + 32)],
        override_model=TinyMouse(tasks),
        device="cpu",
        run_sanity_checks=False,
    )


def test_readout_gradient_and_sign():
    model = _model(exons=[(ORIGIN_, ORIGIN_ + 16)])
    x = torch.zeros((1, 4, model.sequence_length), requires_grad=True)
    score = model.inference_on_tensor(x)
    assert score.shape == (1,)
    score.sum().backward()
    # More coverage means a lower (better) score.
    assert x.grad[0, 1, ORIGIN_] < 0
    assert model.track_name == "ENCFF968GGL+"


def test_strand_and_readout_select_different_tracks():
    x = torch.zeros((1, 4, BorzoiMouse.sequence_length))
    x[0, 1, ORIGIN_ : ORIGIN_ + 32] = 1
    assert _model(strand="-").inference_on_tensor(x).item() == pytest.approx(
        -np.log1p(2)
    )
    assert _model("es_bruce4", "+").inference_on_tensor(x).item() == pytest.approx(
        -np.log1p(3)
    )
    # Bruce4 is unstranded, so both strands map to the same track.
    assert _model("es_bruce4", "-").track_index == _model("es_bruce4", "+").track_index


def test_rejects_exons_outside_the_predicted_window():
    with pytest.raises(ValueError, match="outside"):
        _model(exons=[(0, 1)])


def test_rejects_unknown_readout_or_strand():
    with pytest.raises(ValueError, match="named readout"):
        _model(readout="nonexistent")
    with pytest.raises(ValueError, match="named readout"):
        _model(strand="?")


def test_rejects_mismatched_task_table():
    shifted = _task_table()
    shifted["name"][2287] = "some other track"
    with pytest.raises(ValueError, match="track indices"):
        _model(tasks=shifted)
    rescaled = _task_table()
    rescaled["scale"][2287] = 1.0
    with pytest.raises(ValueError, match="transformation differs"):
        _model(tasks=rescaled)


def test_strings_require_the_exact_input_length():
    model = _model()
    with pytest.raises(ValueError, match="length"):
        model(["ACGT"])


def test_rejects_corrupt_local_checkpoint(tmp_path):
    corrupt = tmp_path / "mouse_rep0.ckpt"
    corrupt.write_bytes(b"incomplete checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        BorzoiMouse(
            "es_e14",
            "+",
            [(ORIGIN_, ORIGIN_ + 32)],
            override_weights_local_path=str(corrupt),
        )


def test_tism_matches_the_input_gradient():
    model = _model(exons=[(ORIGIN_, ORIGIN_ + 32)])
    sequence = "A" * model.sequence_length
    pairs, values = model.get_tism(sequence, [ORIGIN_])
    assert pairs == [(ORIGIN_, "C"), (ORIGIN_, "G"), (ORIGIN_, "T")]
    # Only `C` drives the injected track, and more of it lowers the score.
    assert values[0] < 0
    np.testing.assert_allclose(values[1:], [0, 0], atol=1e-7)


class RampMouse(torch.nn.Module):
    """Profile that varies by bin, so which exon window is scored matters.

    `TinyMouse` emits the same value in every bin, which cannot distinguish a
    correct exon window from a stale one.
    """

    def __init__(self):
        super().__init__()
        self.data_params = {
            "tasks": _task_table(),
            "train_seq_len": mouse.BorzoiMouse.sequence_length,
        }

    def forward(self, x):
        out = torch.zeros(
            (x.shape[0], mouse.N_MOUSE_TASKS_, mouse.N_SPATIAL_BINS_),
            device=x.device,
            dtype=x.dtype,
        )
        out[:, 2287, :] = torch.arange(mouse.N_SPATIAL_BINS_, dtype=x.dtype)
        return out


def test_reassigning_exons_rescores_the_new_window():
    """The cached coverage weights must not go stale when exons change."""
    resolution = mouse.OUTPUT_RESOLUTION_
    model = BorzoiMouse(
        "es_e14",
        "+",
        [(ORIGIN_, ORIGIN_ + resolution)],
        override_model=RampMouse(),
        device="cpu",
        run_sanity_checks=False,
    )
    x = torch.zeros((1, 4, model.sequence_length))
    first = model.inference_on_tensor(x).item()

    model.exons = [(ORIGIN_ + 1000 * resolution, ORIGIN_ + 1001 * resolution)]
    second = model.inference_on_tensor(x).item()

    # Bin 0 holds 0.0 and bin 1000 holds 1000.0.
    assert first == pytest.approx(0.0)
    assert second == pytest.approx(-np.log1p(1000.0))
    assert model.exons == [(ORIGIN_ + 32000, ORIGIN_ + 32032)]


def test_exons_setter_validates():
    model = _model()
    with pytest.raises(ValueError, match="outside"):
        model.exons = [(0, 1)]
