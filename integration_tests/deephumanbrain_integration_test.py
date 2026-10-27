"""
To run:
```zsh
KERAS_BACKEND=torch pytest -v -m deephumanbrain \
    integration_tests/deephumanbrain_integration_test.py
```

Compares the PyTorch-backend CREsted model against TensorFlow-generated
reference predictions for all 76 outputs.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import nucleobench.models.deephumanbrain
from nucleobench.common.genomic_readout import one_hot
from nucleobench.models.deephumanbrain.model_def import DeepHumanBrain

FIXTURE_ = (
    Path(nucleobench.models.deephumanbrain.__file__).parent
    / "testdata/deephumanbrain_tensorflow.json"
)


@pytest.mark.deephumanbrain
def test_tensorflow_fixture_public_api_and_input_gradient():
    crested = pytest.importorskip("crested")
    sequences = ["ACGT" * 528 + "AC", "A" * 2114, "C" * 2114]
    fixture = json.loads(FIXTURE_.read_text())
    assert [
        hashlib.sha256(sequence.encode()).hexdigest() for sequence in sequences
    ] == [row["sha256"] for row in fixture["sequences"]]
    model = DeepHumanBrain(["AMY"])
    assert model.metadata_sha256 == fixture["labels_sha256"]
    assert model.weights_sha256 == fixture["model_sha256"]
    expected = np.asarray(fixture["predictions"])
    actual = model.predict_raw_on_strings(sequences)
    # Established on macOS arm64: max observed absolute difference was 2.96e-5.
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=4e-5)
    public_pytorch = crested.tl.predict(sequences, model.model, verbose=0)
    np.testing.assert_allclose(actual, public_pytorch, rtol=1e-5, atol=1e-6)
    x = one_hot([sequences[0]], model.sequence_length).requires_grad_(True)
    score = model.inference_on_tensor(x)
    score.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
