"""UltraNet — ультимативная нейросетевая библиотека на чистом NumPy.

Содержит собственный автоград, слои (Linear/Conv2d/LayerNorm/Attention),
оптимизаторы (SGD/Adam/AdamW/RMSprop), тренер и готовые модели (MLP, CNN, GPT).
"""
from . import data, functional, models, nn, optim
from .data import CharTokenizer, DataLoader, make_lm_batches, make_moons, make_spirals, train_test_split
from .functional import accuracy, binary_cross_entropy, cross_entropy, mse_loss
from .models import GPT, MLP, ConvNet, GPTConfig
from .optim import SGD, Adam, AdamW, CosineWarmup, RMSprop
from .tensor import Tensor, cat, ones, randn, stack, tensor, zeros
from .trainer import Trainer

__version__ = "1.0.0"

__all__ = [
    "Tensor", "tensor", "zeros", "ones", "randn", "stack", "cat",
    "nn", "optim", "data", "functional", "models",
    "MLP", "ConvNet", "GPT", "GPTConfig",
    "SGD", "Adam", "AdamW", "RMSprop", "CosineWarmup",
    "Trainer", "DataLoader", "CharTokenizer", "make_spirals", "make_moons",
    "make_lm_batches", "train_test_split",
    "cross_entropy", "binary_cross_entropy", "mse_loss", "accuracy",
]
