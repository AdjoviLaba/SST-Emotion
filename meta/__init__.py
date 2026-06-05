from .reptile import Reptile
from .fomaml import FOMAML
from .maml import MAML
from .anil import ANIL
from .none import NoMeta

REGISTRY = {
    "reptile": Reptile,
    "fomaml": FOMAML,
    "maml": MAML,
    "anil": ANIL,
    "none": NoMeta,
}


def build(name: str, model, cfg: dict):
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(f"Unknown meta algorithm '{name}'. Choose from: {list(REGISTRY)}")
    return REGISTRY[name](model, cfg)
