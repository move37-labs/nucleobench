from nucleobench.models.bpnet import model_def as bpnet_model_def
from nucleobench.models.dummy import model_def as dummy_model_def
from nucleobench.models.grelu.borzoi import model_def as borzoi_model_def
from nucleobench.models.grelu.enformer import model_def as enformer_model_def
from nucleobench.models.malinois import model_def as malinois_model_def
from nucleobench.models.rna.optimus5p import model_def as optimus5p_model_def
from nucleobench.models.rna.rinalmo_mrl import model_def as rinalmo_mrl_model_def
from nucleobench.models.rna.saluki import model_def as saluki_model_def
from nucleobench.models.substring_count_net import model_def as substring_model_def
from nucleobench.optimizations import model_class as mc

MODELS_ = {
    "dummy": dummy_model_def.DummyModel,
    "enformer": enformer_model_def.Enformer,
    "malinois": malinois_model_def.Malinois,
    "substring_count": substring_model_def.CountSubstringModel,
    "bpnet": bpnet_model_def.BPNet,
    "rinalmo_mrl": rinalmo_mrl_model_def.RinalmoMRL,
    "optimus5p": optimus5p_model_def.Optimus5P,
    "saluki": saluki_model_def.SalukiModel,
    "borzoi": borzoi_model_def.Borzoi,
}


def get_model(model_name: str) -> mc.ModelClass:
    return MODELS_[model_name]
