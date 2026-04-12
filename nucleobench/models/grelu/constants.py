import os

from dotenv import load_dotenv

load_dotenv()  # loads .env if present; real env vars (e.g. GitHub Actions secrets) take precedence

# Must be this, to be consistent with gRelu.
VOCAB_ = ["A", "C", "G", "T"]

WANDB_API_KEY_ = os.getenv("WANDB_API_KEY")

# "Auto" device option.
AUTO_DEVICE = "auto"