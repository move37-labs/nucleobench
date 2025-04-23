from nucleobench.optimizations import optimization_class as oc

from nucleobench.optimizations.beam_search import beam_ordered, beam_unordered
from nucleobench.optimizations.dummy import random_perturbation
from nucleobench.optimizations.fastseqprop_torch import fs
from nucleobench.optimizations.ledidi import ledidi
from nucleobench.optimizations.directed_evolution import directed_evolution
from nucleobench.optimizations.ada.adalead import adalead_ref
from nucleobench.optimizations.simulated_annealing import simulated_annealing

OPTIMIZATIONS_REQUIRING_TISM_ = {
    "beam_search": beam_ordered.OrderedBeamSearch,
    "beam_search_unordered": beam_unordered.UnorderedBeamSearch,
}

OPTIMIZATIONS_REQUIRING_PYTORCH_DIFF_ = {
    "fastseqprop": fs.FastSeqProp,
    "ledidi": ledidi.Ledidi,
}

GENERAL_OPTIMIZATIONS_ = {
    "adalead": adalead_ref.AdaLeadRef,
    "directed_evolution": directed_evolution.DirectedGreedEvolution,
    "dummy": random_perturbation.RandomPerturbation,
    "simulated_annealing": simulated_annealing.SimulatedAnnealing,
}

OPTIMIZATIONS_ = {}
OPTIMIZATIONS_.update(OPTIMIZATIONS_REQUIRING_TISM_)
OPTIMIZATIONS_.update(OPTIMIZATIONS_REQUIRING_PYTORCH_DIFF_)
OPTIMIZATIONS_.update(GENERAL_OPTIMIZATIONS_)


def get_optimization(opt_name: str) -> oc.SequenceOptimizer:
    return OPTIMIZATIONS_[opt_name]
