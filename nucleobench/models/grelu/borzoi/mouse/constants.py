"""Constants for the mouse Borzoi model.

Mouse-specific: the replicate 0 checkpoint, its pinned digest, the output grid
geometry, and the ES-cell RNA-seq tracks this adapter scores. Values shared
with the human variant live in `nucleobench.models.grelu.borzoi.constants`.

To test some parts of this:
```zsh
python -m nucleobench.models.grelu.borzoi.mouse.constants
```
"""

BORZOI_MOUSE_FILENAME = "mouse_rep0.ckpt"
# Pinned so that a silently re-uploaded checkpoint is caught rather than used.
BORZOI_MOUSE_SHA256 = "9c8aeaa8633d1405913afe95077dd78c5a269100ee2e61c47518f1fa4b102b57"

BORZOI_MOUSE_N_TASKS_ = 2608
BORZOI_MOUSE_N_SPATIAL_BINS_ = 6144
# Borzoi predicts the center 196,608 bp of its input in 32-bp bins.
BORZOI_MOUSE_OUTPUT_RESOLUTION_ = 32
BORZOI_MOUSE_OUTPUT_ORIGIN_ = 163_840

# ENCODE RNA-seq tracks for the two mouse ES lines. E14 is stranded, so the
# index depends on the gene's strand; Bruce4 is unstranded.
BORZOI_MOUSE_TRACKS_ = {
    "es_e14": {"+": (2287, "ENCFF968GGL+"), "-": (2288, "ENCFF968GGL-")},
    "es_bruce4": {"+": (2237, "ENCFF427XNM"), "-": (2237, "ENCFF427XNM")},
}

# Flattened track names, mirroring `BORZOI_TASKS_` on the human side.
BORZOI_MOUSE_TASKS_ = sorted(
    {name for strands in BORZOI_MOUSE_TRACKS_.values() for _, name in strands.values()}
)


if __name__ == "__main__":
    print(f"Readouts: {list(BORZOI_MOUSE_TRACKS_)}")
    print(f"Tracks: {BORZOI_MOUSE_TASKS_}")
    assert len(BORZOI_MOUSE_TASKS_) == 3, BORZOI_MOUSE_TASKS_
