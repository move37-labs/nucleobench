"""Parent class for optimizers."""

from typing import Optional

from nucleobench.optimizations.typing import ModelType, SequenceType, SamplesType, PositionsToMutateType

class SequenceOptimizer(object):
    def __init__(
        self, 
        model_fn: ModelType, 
        start_sequence: SequenceType,
        positions_to_mutate: Optional[PositionsToMutateType] = None,
        ):
        raise NotImplementedError("Not implemented.")

    def run(self, n_steps: int):
        raise NotImplementedError("Not implemented.")

    def get_samples(self, n_samples: int) -> SamplesType:
        raise NotImplementedError("Not implemented.")

    @staticmethod
    def init_parser():
        raise ValueError("Not implemented.")

    @staticmethod
    def run_parser():
        raise ValueError("Not implemented.")
    
    def is_finished(self) -> bool:
        raise ValueError("Not implemented.")
