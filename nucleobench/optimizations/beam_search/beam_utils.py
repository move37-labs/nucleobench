
from typing import Any, Generator

BeamValue = float

# TODO(joelshor): Consider storing string deltas instead of full strings, to decrease
# memory usage.
class Beam(object):
    """Simple API for heap-based beam access.
    
    Note: Optimizers try to MINIMIZE, so the value from networks are better if LOW.
    This means we should REMOVE LARGE VALUES from the beam as we go.
    
    TODO(joelshor): Make this more efficient with a heap / heapq.
    """
    def __init__(self, max_items: int):
        self.max_items = max_items
        self.beam = []
        
    def put(self, itms: list[tuple[BeamValue, Any]]):
        self.beam.extend(itms)
        self.beam.sort(key=lambda x: x[0], reverse=False)
        if len(self.beam) > self.max_items:
            self.beam = self.beam[:self.max_items]
        
    def get_items(self) -> Generator[str, None, None]:
        for x in self.beam:
            yield x[1]
            
    
    def get_best_state(self) -> Any:
        return self.beam[0][1]
    
    def get_best_val_and_state(self) -> Any:
        return self.beam[0]