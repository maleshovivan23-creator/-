"""Proteus — подсистема UltraNet: вложенная байтовая модель, живущая на любом железе."""
from .core import Proteus, Response
from .bytes import ByteTokenizer, make_byte_batches
from .controller import MetaController, Plan
from .devices import DEVICES, Device, DeviceState, get_device, list_devices
from .memory import Doc, LocalMemory
from .swarm import Node, Swarm
from .matformer import MatBlock, MatConfig, MatFormer
from .train import evaluate_widths, train_matryoshka
from .quant import QuantStats, TernaryModel, quantization_error, ternary_quantize

__all__ = [
    "Proteus", "Response",
    "ByteTokenizer", "make_byte_batches",
    "MatConfig", "MatFormer", "MatBlock",
    "Device", "DeviceState", "DEVICES", "get_device", "list_devices",
    "MetaController", "Plan",
    "LocalMemory", "Doc", "Swarm", "Node",
    "train_matryoshka", "evaluate_widths",
    "TernaryModel", "QuantStats", "ternary_quantize", "quantization_error",
]
