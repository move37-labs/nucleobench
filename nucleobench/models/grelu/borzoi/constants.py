"""Constants shared by every Borzoi variant.

The human and mouse checkpoints live in the same Hugging Face repository and
take the same input length. Everything else is variant-specific and lives in
`human/constants.py` or `mouse/constants.py`.
"""

BORZOI_REPO_ID = "Genentech/borzoi-model"
BORZOI_TRAIN_LEN_ = 524_288
