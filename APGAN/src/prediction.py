import numpy as np
import torch
from src.model import LatentDecoder, Discriminator, GraphEncoder, Generator, Predictor, proPredictor
from data.data_loader import get_data_loader
import time
from src.utils import check_validity, save_mol_png, check_novelty, get_real_label, get_fake_label, \
    compute_gradient_penalty, get_data_par, \
    get_loss_curve, calculate_sa_score, simplified_function


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor_model = torch.load('path/to/your/trained_predictor_model.pth').to(device)
predictor_model.eval()

