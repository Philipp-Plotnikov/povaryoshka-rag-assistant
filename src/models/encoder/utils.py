import torch

from models.encoder.encoder import PovaryoshkaEncoder
from models.utils import get_model_device


def load_encoder():
    encoder = PovaryoshkaEncoder(matryoshka_dims=[384])
    device = get_model_device(encoder)
    state_dict = torch.load("../models/encoder/povaryoshka_encoder_weights.pth", map_location=device)
    encoder.load_state_dict(state_dict)
    return encoder
