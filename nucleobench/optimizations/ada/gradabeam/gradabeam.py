"""GradaBeamPBT.

Gradient-guided adaptive beam, adaptive mutation rate (PBT), adaptive directed evolution.
"""

import argparse

import gradabeam

from nucleobench.common import argparse_lib, testing_utils
from nucleobench.optimizations import optimization_class as oc


class GradaBeam(gradabeam.GradaBeam, oc.SequenceOptimizer):
    """GrAdaBeam wrapper around gradabeam package."""

    @staticmethod
    def init_parser():
        parser = argparse.ArgumentParser(description="", add_help=False)
        group = parser.add_argument_group("GradaBeam init args")

        group.add_argument("--beam_size", type=int, default=10, required=False)
        group.add_argument("--mutations_per_sequence", type=float, required=False)
        group.add_argument("--n_rollouts_per_root", type=int, default=4, required=False)
        group.add_argument("--eval_batch_size", type=int, default=1, required=False)
        group.add_argument("--rng_seed", type=int, default=42, required=False)
        group.add_argument("--exploration_alpha", type=float, required=True)
        group.add_argument("--max_rollout_len", type=int, default=200, required=False)
        group.add_argument(
            "--use_pbt",
            type=argparse_lib.str_to_bool,
            default=True,
            required=False,
            help="Enable Population Based Training for adaptive mutation rates.",
        )
        group.add_argument(
            "--debug", type=argparse_lib.str_to_bool, default=None, required=False
        )
        return parser

    @staticmethod
    def debug_init_args():
        return {
            "model_fn": testing_utils.CountLetterModel(flip_sign=True),
            "start_sequence": "AAAAAA",
            "beam_size": 10,
            "mutations_per_sequence": 1,
            "n_rollouts_per_root": 4,
            "eval_batch_size": 1,
            "rng_seed": 42,
            "exploration_alpha": 0.05,
            "use_pbt": True,
        }

    def is_finished(self) -> bool:
        return False
