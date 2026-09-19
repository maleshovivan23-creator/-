"""UltraNet — ультимативная нейросетевая библиотека на чистом NumPy.

Собственный автоград, современные слои (RoPE, RMSNorm, SwiGLU, MHA с KV-кэшем),
оптимизаторы (AdamW, Lion, Lookahead, EMA), тренер и модели вплоть до GPT.
"""
from . import checkpoint, data, functional, models, nn, optim, proteus, tokenizer
from .data import (
    CharTokenizer,
    DataLoader,
    Dataset,
    WordTokenizer,
    make_lm_batches,
    make_moons,
    make_regression,
    make_spirals,
    normalize,
    train_test_split,
)
from .functional import (
    accuracy,
    bce_with_logits,
    binary_cross_entropy,
    confusion_matrix,
    cross_entropy,
    f1_score,
    focal_loss,
    huber_loss,
    mae_loss,
    mse_loss,
    nll_loss,
    perplexity,
    top_k_accuracy,
)
from .models import GPT, MLP, ConvNet, GPTConfig, ResNet, TextClassifier
from .optim import (
    EMA,
    SGD,
    Adagrad,
    Adam,
    AdamW,
    CosineWarmup,
    Lion,
    Lookahead,
    OneCycleLR,
    ReduceLROnPlateau,
    RMSprop,
    StepLR,
)
from .tensor import (
    Tensor,
    cat,
    get_rng,
    manual_seed,
    no_grad,
    ones,
    randn,
    stack,
    tensor,
    zeros,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .gradcheck import gradcheck, numeric_grad
from .tokenizer import BPETokenizer
from .trainer import Trainer

__version__ = "4.0.0"

__all__ = [
    # ядро
    "Tensor", "tensor", "zeros", "ones", "randn", "stack", "cat", "no_grad",
    "manual_seed", "get_rng",
    # пакеты
    "nn", "optim", "data", "functional", "models", "proteus",
    # модели
    "MLP", "ConvNet", "ResNet", "GPT", "GPTConfig", "TextClassifier",
    # оптимизаторы и расписания
    "SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Lion", "Lookahead", "EMA",
    "CosineWarmup", "OneCycleLR", "StepLR", "ReduceLROnPlateau",
    # обучение и данные
    "Trainer", "DataLoader", "Dataset", "CharTokenizer", "WordTokenizer", "BPETokenizer",
    "save_checkpoint", "load_checkpoint", "gradcheck", "numeric_grad",
    "make_spirals", "make_moons", "make_regression", "make_lm_batches",
    "train_test_split", "normalize",
    # потери и метрики
    "cross_entropy", "nll_loss", "binary_cross_entropy", "bce_with_logits", "focal_loss",
    "mse_loss", "mae_loss", "huber_loss",
    "accuracy", "top_k_accuracy", "perplexity", "confusion_matrix", "f1_score",
]
