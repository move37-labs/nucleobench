"""Typing for NucleoBench"""

from collections.abc import Callable

from nucleobench.optimizations import model_class

SequenceType = model_class.SequenceType
SamplesType = list[SequenceType]
PositionsToMutateType = list[int] | None
TISMType = list[dict[str, float]]

ModelType = model_class.ModelClass | Callable
TISMModelClass = model_class.TISMModelClass
PyTorchDifferentiableModel = model_class.PyTorchDifferentiableModel
