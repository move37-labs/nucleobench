# ChromBPNet testdata

## adrenal_c0_keras_goldens.json

Official Keras I/O pairs for `ChromBPNetOracle(cell_type="Adrenal_c0")`.

| Field | Value |
|---|---|
| Model | HDMA ChromBPNet, Part 1 |
| Cell type | `Adrenal_c0` (adrenal cortex cluster 0) |
| Fold | 0 |
| Weights file | `Adrenal_c0__fold_0__chrombpnet_nobias.h5` |
| Zenodo record | https://zenodo.org/records/15048278 |
| TF version used to generate | 2.11.0 |
| Load call | `tf.keras.models.load_model(path, compile=False)` |
| Counts head | `outputs[1][0, 0]` |
| Keras input shape | `(1, 2114, 4)` — channels last |

`keras_counts` is the **raw counts head scalar** (log scale, not negated).  
NucleoBench's `ChromBPNetOracle` negates this:  
```
oracle_score = -keras_counts
```

### Regeneration

Run `generate_keras_goldens.py` in a Linux TF environment (TF 2.8+ required;
no macOS arm64 wheel):

```bash
docker run --rm \
    -v ~/.cache/nucleobench/chrombpnet:/weights:ro \
    -v $(pwd):/repo \
    tensorflow/tensorflow:2.11.0 \
    python /repo/nucleobench/models/chrombpnet/generate_keras_goldens.py \
        --weights /weights/Adrenal_c0.h5 \
        --out /repo/nucleobench/models/chrombpnet/testdata/adrenal_c0_keras_goldens.json \
        --also-pytorch
```

The `--also-pytorch` flag prints `max |keras - pytorch|` across all sequences,
which should be used to validate or tighten the `atol` in the integration test.

### Sequences

| Name | Source | Coordinates |
|---|---|---|
| `polya` | Synthetic | 2114 × A |
| `gc_rich` | Synthetic | 2114 × GC repeat |
| `cyp11a1` | hg38 UCSC 2026-08-21 | chr15:74366827–74368940 (1-based), centered on CYP11A1 TSS (Ensembl ENSG00000140459) |
| `gene_desert` | hg38 UCSC 2026-08-21 | chr8:127148943–127151056 (1-based), same locus as Enformer/Borzoi integration tests |
