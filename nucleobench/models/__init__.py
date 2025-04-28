from nucleobench.optimizations import model_class as mc
from nucleobench.models.bpnet import model_def as bpnet_model_def
from nucleobench.models.grelu.enformer import model_def as enformer_model_def
from nucleobench.models.dummy import model_def as dummy_model_def
from nucleobench.models.malinois import model_def as malinois_model_def
from nucleobench.models.substring_count_net import model_def as substring_model_def


MODELS_ = {
    'dummy': dummy_model_def.DummyModel,
    'enformer': enformer_model_def.Enformer,
    'malinois': malinois_model_def.Malinois,
    'substring_count': substring_model_def.CountSubstringModel,
    'bpnet': bpnet_model_def.BPNet,
}

def get_model(model_name: str) -> mc.ModelClass:
    return MODELS_[model_name]