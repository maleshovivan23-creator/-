"""Рой: один Протей, разлитый по нескольким устройствам.

Реализует Часть 4: «Наушники слышат, часы думают, телефон помнит,
ноутбук считает. Inference идёт по цепочке. Нет центра».

Слои трансформера распределяются между узлами пропорционально их мощности;
активация передаётся по цепочке. Отключение узла вызывает перераспределение
(«Протей отступает»), подключение — расширение («растекается»).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..tensor import Tensor, no_grad
from .controller import MetaController
from .devices import Device, DeviceState
from .matformer import MatFormer


@dataclass
class Node:
    """Узел роя — устройство с его состоянием."""

    state: DeviceState
    layers: List[int] = field(default_factory=list)
    bytes_moved: int = 0

    @property
    def name(self) -> str:
        return self.state.device.name

    @property
    def capacity(self) -> float:
        """Пропускная способность с учётом троттлинга."""
        return max(self.state.device.gflops * self.state.throttle, 1e-6)


class Swarm:
    """Инференс по цепочке устройств без центра и облака.

    >>> sw = Swarm(model, [DeviceState(get_device("earbuds")),
    ...                    DeviceState(get_device("phone"))])
    >>> logits = sw.forward(np.array([[1, 2, 3]]))
    """

    def __init__(self, model: MatFormer, states: Sequence[DeviceState],
                 width: float = 1.0) -> None:
        self.model = model
        self.width = width
        self.nodes: List[Node] = [Node(s) for s in states]
        self.hops = 0
        self.bytes_total = 0
        self._assign()

    # ----------------------------------------------------------- топология
    def _assign(self) -> None:
        """Распределить слои пропорционально мощности узлов."""
        n_layers = self.model.cfg.n_layer
        for n in self.nodes:
            n.layers = []
        if not self.nodes:
            raise ValueError("рой пуст — некому считать")
        caps = np.array([n.capacity for n in self.nodes], dtype=float)
        share = caps / caps.sum()
        counts = np.floor(share * n_layers).astype(int)
        # каждый узел получает хотя бы один слой, если слоёв хватает:
        # даже слабые наушники участвуют в общем организме
        if n_layers >= len(self.nodes):
            counts = np.maximum(counts, 1)
        while counts.sum() < n_layers:
            counts[int(np.argmax(share - counts / max(n_layers, 1)))] += 1
        while counts.sum() > n_layers:
            # забираем у самых мощных, не опуская никого ниже минимума
            floor = 1 if n_layers >= len(self.nodes) else 0
            cand = np.where(counts > floor)[0]
            counts[cand[int(np.argmax(counts[cand]))]] -= 1
        # узлы идут от слабого к сильному: слышит -> думает -> считает
        order = np.argsort(caps)
        layer = 0
        for idx in order:
            take = int(counts[idx])
            self.nodes[idx].layers = list(range(layer, layer + take))
            layer += take

    @property
    def active_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.layers]

    def topology(self) -> str:
        rows = []
        for n in sorted(self.nodes, key=lambda x: (x.layers or [99])[0]):
            span = f"{n.layers[0]}–{n.layers[-1]}" if n.layers else "—"
            rows.append(f"  {n.name:<22} слои {span:<8} мощность {n.capacity:>7.2f}")
        return "\n".join(rows)

    # -------------------------------------------------------------- события
    def join(self, state: DeviceState) -> None:
        """Появилось новое устройство — Протей растекается."""
        self.nodes.append(Node(state))
        self._assign()

    def leave(self, name: str) -> None:
        """Устройство исчезло — Протей стягивается, ничего не теряя."""
        before = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.name != name]
        if len(self.nodes) == before:
            raise KeyError(f"узел '{name}' не найден в рое")
        if not self.nodes:
            raise ValueError("нельзя удалить последний узел роя")
        self._assign()

    # ------------------------------------------------------------- инференс
    @no_grad()
    def forward(self, idx) -> Tensor:
        """Прогон по цепочке узлов; между узлами передаётся только активация."""
        d = self.model.width_dim(self.width)
        x = self.model.embed(idx, d)
        self.hops = 0
        self.bytes_total = 0
        chain = sorted(self.active_nodes, key=lambda n: n.layers[0])
        for i, node in enumerate(chain):
            for li in node.layers:
                x = self.model.run_block(li, x, d)
            if i < len(chain) - 1:
                payload = int(np.prod(x.shape)) * 4      # fp32-активация
                node.bytes_moved += payload
                self.bytes_total += payload
                self.hops += 1
        return self.model.readout(x, d)

    def transfer_report(self, idx) -> str:
        """Сколько данных ушло по сети против размера самой модели."""
        self.forward(idx)
        weights = self.model.active_params(self.width) * 2 // 8   # 1.58-bit упаковка
        return (f"прыжков между устройствами: {self.hops}, "
                f"передано {self.bytes_total / 1024:.1f} КБ активаций "
                f"(веса модели: {weights / 1024:.1f} КБ — они НЕ передаются)")
