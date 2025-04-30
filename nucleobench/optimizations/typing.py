"""Typing for NucleoBench"""

from typing import Union

from nucleobench.optimizations import model_class

SequenceType = model_class.SequenceType
SamplesType = list[SequenceType]
PositionsToMutateType = list[int]
TISMType = list[dict[str, float]]

ModelType = Union[model_class.ModelClass, callable]
TISMModelClass = model_class.TISMModelClass
PyTorchDifferentiableModel = model_class.PyTorchDifferentiableModel