"""Generic priority queue with supported interface."""

import dataclasses
import heapq
import bisect


@dataclasses.dataclass(order=True)
class SearchQItem:
    fitness: float
    state: str
    num_edits: int


class TwoSidedPriorityQueue(object):
    """Keeps low and high items.

    Uses a simple list.

    We need access to the "best" (to expand) and the "worst" (to remove).

    Min is better for the actual problem, but provide -{energy} for the problems
    we care about, so max is better.

    TODO(joelshor): Consider using two heaps, but this would use more memory.
    """

    def __init__(self, max_items: int):
        # Max items allowed in the queue. Internally, we will at times
        # have more elements in the queue than this.
        self.max_items = max_items
        self.q  = []

    def push(self, itm: SearchQItem):
        if len(self.q) == self.max_items and itm.fitness < self.q[0].fitness:
            return
        elif len(self.q) < self.max_items:
            bisect.insort(self.q, itm)
        else:
            bisect.insort(self.q, itm)
            self.q.pop(0)  # Remove the "worst"


    def push_batch(self, itms: list[SearchQItem]):
        """Push a batch of items.

        TODO(joelshor): Find a more efficient way to take advantage of batching.
        """
        for itm in itms:
            self.push(itm)


    def pop(self) -> SearchQItem:
        """Pops the 'best' item."""
        return self.q.pop(-1)


    def get(self, n_samples: int) -> list[SearchQItem]:
        if n_samples > self.max_items:
            raise ValueError('Too many items requested: {n_samples} vs {self.max_items}')
        return [x.state for x in self.q[-n_samples:]]


class OneSidedPriorityQueue(object):
    """Priority queue.

    Uses fitness instead of energy (keep high items, low is bad).
    """

    def __init__(self, max_items: int):
        # Max items allowed in the queue. Internally, we will at times
        # have more elements in the queue than this.
        self.max_items = max_items
        self.q  = []
        
        
    def reset_queue(self, itms: list[SearchQItem]):
        del self.q
        heapq.heapify(itms)
        self.q = itms


    def push(self, itm: SearchQItem):
        if len(self.q) == self.max_items and itm.fitness < self.q[0].fitness:
            return
        elif len(self.q) < self.max_items:
            heapq.heappush(self.q, itm)
        else:
            heapq.heappushpop(self.q, itm)


    def push_batch(self, itms: list[SearchQItem]):
        """Push a batch of items.

        TODO(joelshor): Find a more efficient way to take advantage of batching.
        """
        for itm in itms:
            self.push(itm)


    def get(self, n_samples: int) -> list[str]:
        if n_samples > self.max_items:
            raise ValueError('Too many items requested: {n_samples} vs {self.max_items}')
        return [x.state for x in heapq.nlargest(n_samples, self.q)]
