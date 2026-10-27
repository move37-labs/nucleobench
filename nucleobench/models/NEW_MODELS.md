# Additional genomic models

Model adapters for mouse ES-cell and human brain readouts. All require Python 3.11.

| Registry key | Input | Scalar objective |
| --- | --- | --- |
| `borzoi_mouse` | 524,288 bp | Negative `log1p` of union-exon mean RNA-seq coverage |

Lower is better in every case, matching the rest of the repository.

## Input conventions

All string inputs must have the exact model length. Input tensors use float
`(batch, 4, length)` in A, C, G, T order. Lowercase sequence is accepted, and
`N` is encoded as four zeros. Exon coordinates are zero-based, half-open
offsets into the **supplied input sequence**, not genomic coordinates;
overlapping exons are unioned before overlap-weighted coverage scoring.
Shared helpers live in `nucleobench/common/genomic_readout.py`.

```python
from nucleobench.models.grelu.borzoi.mouse import BorzoiMouse

model = BorzoiMouse(readout="es_e14", strand="+", exons=[(163840, 163900)])
score = model(["A" * 524288])            # shape (1,); lower means more coverage
raw = model.predict_raw_on_strings(["A" * 524288])   # (1, 2608, 6144)
```

## `borzoi_mouse`

Converted mouse replicate 0, loaded through gReLU from `Genentech/borzoi-model`
and verified against a pinned checkpoint SHA256. Needs no dependencies beyond
the ones the human Borzoi already requires.

Its 2,608 tracks are in 32-bp bins across the center 196,608 bp, starting at
input offset 163,840. E14 is stranded `ENCFF968GGL` (indices 2287/2288);
Bruce4 is unstranded `ENCFF427XNM` (index 2237). The adapter checks that the
checkpoint's target table still has those names at those indices and still
declares the `sum_sqrt` / scale-0.3 transformation, so scores are on the
checkpoint's training scale and are not physical read counts.

## Verification status

Offline tests with injected models run as part of the normal unit-test suite
and live beside each adapter (`grelu/borzoi/mouse_test.py`,
`nucleobench/common/genomic_readout_test.py`).

- **`borzoi_mouse`: accepted.** On real mm10 sequence the converted checkpoint
  separates ES-cell genes from silent loci on the E14 RNA track by two to three
  orders of magnitude (Nanog 0.818, Pou5f1 0.641, Alb 0.0080, gene desert
  0.0017). This is the same bar the human Borzoi and Enformer integrations
  meet. An earlier comparison against CREsted's independent Keras conversion
  was run on an all-A sequence, where both implementations output
  gene-desert-level noise (~0.002); that comparison carried no signal and was
  not evidence of a conversion error.

## Running the integration tests

These are outside ordinary test discovery (`testpaths = ["nucleobench"]`) and
must be selected explicitly. They download real weights and genomes.

```sh
pytest -v -m borzoi_mouse integration_tests/borzoi_mouse_integration_test.py
```

The mouse Borzoi test downloads mm10 through
`integration_tests/data_loaders/mouse_es_expression_by_bucket.py`.
