"""AdaBeam.

Adaptive beam, adaptive mutation rate, adaptive directed evolution.
"""

import argparse

import gradabeam

from nucleobench.common import argparse_lib, testing_utils
from nucleobench.optimizations import optimization_class as oc


class AdaBeam(gradabeam.AdaBeam, oc.SequenceOptimizer):
    """AdaBeam designer wrapper around gradabeam package."""

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group("AdaBeam init args")

        group.add_argument(
            "--beam_size",
            type=int,
            default=10,
            required=False,
            help="Number of sequences to propose for measurement from ground truth per round",
        )
        group.add_argument(
            "--mutations_per_sequence",
            type=float,
            required=False,
            help="The expected number of mutations per sequence.",
        )
        group.add_argument(
            "--n_rollouts_per_root",
            type=int,
            default=4,
            required=False,
            help="Number of rollouts to perform per parent node (per round)",
        )
        group.add_argument(
            "--skip_repeat_sequences",
            type=argparse_lib.str_to_bool,
            default=True,
            required=False,
            help="",
        )
        group.add_argument(
            "--eval_batch_size",
            type=int,
            default=1,
            required=False,
            help="For code optimization; size of batches sent to model",
        )
        group.add_argument(
            "--rng_seed",
            type=int,
            default=42,
            required=False,
            help="Seed for the pseudo-random number generator",
        )
        group.add_argument(
            "--debug",
            type=argparse_lib.str_to_bool,
            default=None,
            required=False,
            help="Debug info.",
        )

        return parser

    @staticmethod
    def debug_init_args():
        return {
            "model_fn": testing_utils.CountLetterModel(),
            "start_sequence": "AAAAAA",
            "beam_size": 10,
            "mutations_per_sequence": 1,
            "n_rollouts_per_root": 4,
            "eval_batch_size": 1,
            "skip_repeat_sequences": False,  # avoids infinite loops.
            "rng_seed": 42,
        }

    def is_finished(self) -> bool:
        return False
