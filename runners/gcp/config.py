"""
Configuration for Google Batch runner.

USAGE:
    This file contains configuration constants for the Google Batch runner.
    Modify these values to customize your setup, or set environment variables:

    For production use, set these environment variables rather than modifying
    the defaults in this file.
"""

# Google Cloud configuration
PROJECT_ID = ""
REGION = ""
BUCKET_NAME = ""
# Optional: service account for the job. If empty, uses the default Compute Engine service account.
SERVICE_ACCOUNT_EMAIL = ""

# Docker image configuration
DOCKER_IMAGE = "joelshor/nucleobench:latest"

# Job configuration
DEFAULT_DISK_SIZE_GB = 20

# Not enformer.
DEFAULT_MACHINE_TYPE = "e2-standard-4"
DEFAULT_CPU_COUNT = 1
DEFAULT_MEMORY_GB = 1

# Not enformer.
ENFORMER_MACHINE_TYPE = "e2-custom-4-30720"
ENFORMER_CPU_COUNT = 4
ENFORMER_MEMORY_GB = 30

_REQUIRED_FIELDS = ["PROJECT_ID", "REGION", "BUCKET_NAME", "DOCKER_IMAGE"]


def validate() -> None:
    """Raise ValueError listing every required config field that is still empty."""
    import sys

    current_module = sys.modules[__name__]
    missing = [f for f in _REQUIRED_FIELDS if not getattr(current_module, f, None)]
    if missing:
        fields_str = "\n".join(f"  - {f}" for f in missing)
        raise ValueError(
            f"The following required values are not set in runners/gcp/config.py:\n"
            f"{fields_str}\n"
            f"Please edit runners/gcp/config.py and fill in these values before running."
        )
