"""Parent class for optimizers."""

from nucleobench.models import model_class

from typing import Optional, Union

SequenceType = str
SamplesType = list[str]


class SequenceOptimizer(object):
    def __init__(
        self, 
        model_fn: Union[model_class.ModelClass, callable], 
        seed_sequence: SequenceType,
        positions_to_mutate: Optional[list[int]] = None,
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
