"""Proteus — подсистема UltraNet: вложенная байтовая модель, живущая на любом железе."""
from .adapt import Feedback, LoRALayer, PersonalAdapter
from .assembly import Assembly, ProteusBody, detect_triggers
from .capsules import Capsule, CapsuleRegistry, Level
from .catalog import CATALOG, BY_KEY, CapsuleSpec, Group, Stage, by_group
from .context import Activity, AdaptationLoops, Context, Place, Style, style_for
from .core import Proteus, Response
from .growth import (LADDER, GrowingProteus, GrowthState, HardwareProfile,
                     ladder_table, next_stage, probe_hardware, stage_at)
from .experts import MoE, Expert
from .shards import Shard, ShardedSwarm, SwarmNode
from .steering import SteeringLibrary, SteeringVector
from .weights import (RVQ, ComplexWeight, FisherImportance, FractalStack,
                      HyperNetwork, LogPolarWeight, SphericalWeight,
                      compare_methods, cosine, rel_error, slerp)
from .bytes import ByteTokenizer, make_byte_batches
from .controller import RLMetaController, MetaController, Plan
from .devices import DEVICES, Device, DeviceState, get_device, list_devices
from .memory import Doc, LocalMemory
from .swarm import Node, Swarm
from .matformer import MatBlock, MatConfig, MatFormer
from .train import evaluate_widths, train_matryoshka
from .quant import TernaryLinear, ste_round, QuantStats, TernaryModel, quantization_error, ternary_quantize

__all__ = [
    "TernaryLinear",
    "ste_round",
    "RLMetaController",
    "Proteus", "Response",
    "MoE", "Expert", "SteeringLibrary", "SteeringVector",
    "PersonalAdapter", "LoRALayer", "Feedback",
    "Capsule", "CapsuleRegistry", "Level",
    "CATALOG", "BY_KEY", "CapsuleSpec", "Group", "Stage", "by_group",
    "ProteusBody", "Assembly", "detect_triggers",
    "SphericalWeight", "LogPolarWeight", "RVQ", "FisherImportance",
    "FractalStack", "ComplexWeight", "HyperNetwork", "compare_methods",
    "rel_error", "cosine", "slerp",
    "GrowingProteus", "GrowthState", "HardwareProfile", "probe_hardware",
    "LADDER", "stage_at", "next_stage", "ladder_table",
    "Shard", "ShardedSwarm", "SwarmNode",
    "Context", "Activity", "Place", "Style", "style_for", "AdaptationLoops",
    "ByteTokenizer", "make_byte_batches",
    "MatConfig", "MatFormer", "MatBlock",
    "Device", "DeviceState", "DEVICES", "get_device", "list_devices",
    "MetaController", "Plan",
    "LocalMemory", "Doc", "Swarm", "Node",
    "train_matryoshka", "evaluate_widths",
    "TernaryModel", "QuantStats", "ternary_quantize", "quantization_error",
]
