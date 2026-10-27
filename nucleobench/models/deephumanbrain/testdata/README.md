# DeepHumanBrain testdata

## deephumanbrain_tensorflow.json

Reference predictions for `DeepHumanBrain`, produced by the same CREsted model
running on the **TensorFlow** Keras backend. NucleoBench runs it on the
**PyTorch** backend so that inputs stay differentiable, so this golden checks
that the two backends agree numerically.

| Field | Value |
|---|---|
| Model | CREsted `DeepHumanBrain` |
| Fetched with | `crested.get_model("DeepHumanBrain")` |
| Weights SHA256 | `b2dd34f000da392de80a970a9e5dbc434b5a11f68f0bc491614e314e2fa722f6` |
| Output labels SHA256 | `c90008790a163423cf3d178db61156dfcf1c57c938826cca799ca8ba4ee18530` |
| CREsted version | 1.10.0 |
| Keras / TensorFlow versions | 3.15.1 / 2.20.0 |
| Load call | `crested.utils.load_model(path)` with `KERAS_BACKEND=tensorflow` |
| Keras input shape | `(1, 2114, 4)` — channels last |
| Outputs | All 76 normalized accessibility values, unnegated |

`predictions` is a 3 x 76 array of the **raw** model outputs.
`DeepHumanBrain` negates the mean over the selected labels:

```
oracle_score = -mean(predictions[:, selected_indices])
```

### Tolerance

`integration_tests/deephumanbrain_integration_test.py` compares with
`rtol=5e-5, atol=4e-5`. The largest absolute difference observed on macOS
arm64 was `2.96e-5`.

### Sequences

All three are synthetic, so this establishes agreement between the two Keras
backends, not agreement with measured brain-peak annotations.

| Name | Source | Sequence |
|---|---|---|
| `ACGT_repeat` | Synthetic | `"ACGT" * 528 + "AC"` (2114 bp) |
| `all_A` | Synthetic | 2114 x A |
| `all_C` | Synthetic | 2114 x C |

### Regeneration

The repo environment cannot import TensorFlow alongside its own dependencies,
so generate the goldens in a separate virtualenv built from the same Python:

```bash
python -m venv /tmp/tf-venv
/tmp/tf-venv/bin/pip install "crested==1.10.0" tensorflow
KERAS_BACKEND=tensorflow /tmp/tf-venv/bin/python -c "
import crested, json, hashlib, numpy as np
path, labels = crested.get_model('DeepHumanBrain')
model = crested.utils.load_model(path)
seqs = ['ACGT' * 528 + 'AC', 'A' * 2114, 'C' * 2114]
preds = np.asarray(crested.tl.predict(seqs, model, verbose=0))
print(json.dumps({'predictions': preds.tolist()}))
"
```
