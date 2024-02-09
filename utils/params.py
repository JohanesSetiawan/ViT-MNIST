import torch
import numpy as np
import random


class Parameters:
    def __init__(self):
        # stay
        self.ACTIVATION = "gelu"
        self.ADAM_BETAS = (0.9, 0.999)
        self.ADAM_WEIGHT_DECAY = 0
        self.LEARNING_RATE = 1e-4
        self.DROPOUT = 0.001
        self.IMG_SIZE = 28
        self.IN_CHANNELS = 1
        self.PATCH_SIZE = 4
        self.EMBED_DIM = 16
        self.RANDOM_SEED = 42
        self.NUM_PATCHES = 49
        self.NUM_CLASSES = 10

        # change it
        self.BATCH_SIZE = 32
        self.NUM_HEADS = 16
        self.HIDDEN_DIM = 768
        self.NUM_ENCODERS = 8
        self.PATH_MODELS = "./models/trained_model_1.pth"

        random.seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed_all(self.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
