"""Шарды и рой без лидера.

«Каждое устройство хранит свой шард — не копию модели, а кусок одного
организма. Шарды пересекаются: маленькие устройства хранят только маленькие
уровни вложенности, большие — все».

Четыре правила консенсуса из спецификации:
  1. Кто ближе — тот и роутер.   2. Кто свободнее — тот и считает.
  3. Кто голоднее — тот и отдаёт. 4. Кто умнее — тот и решает сложное.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .devices import Device, DeviceState


@dataclass
class Shard:
    """Кусок Протея, живущий на узле.

    Шард описывается диапазоном уровней вложенности [0, max_width] и
    множеством слоёв. Маленькое устройство хранит только узкие уровни.
    """

    layers: Set[int] = field(default_factory=set)
    max_width: float = 1.0
    skills: Set[str] = field(default_factory=set)
    bytes_stored: int = 0

    def covers(self, layer: int, width: float) -> bool:
        return layer in self.layers and width <= self.max_width + 1e-9

    def __str__(self) -> str:
        ls = f"{min(self.layers)}–{max(self.layers)}" if self.layers else "—"
        return f"слои {ls}, ширина ≤{self.max_width:.2f}, {self.bytes_stored / 1024:.0f} КБ"


@dataclass
class SwarmNode:
    """Узел роя со своим шардом и состоянием."""

    state: DeviceState
    shard: Shard = field(default_factory=Shard)
    online: bool = True
    router_score: float = 0.0

    @property
    def name(self) -> str:
        return self.state.device.name

    @property
    def power(self) -> float:
        return self.state.device.gflops * self.state.throttle

    @property
    def hunger(self) -> float:
        """Насколько узел «голоден» (мало батареи) — тем меньше берёт работы."""
        if not self.state.device.battery:
            return 0.0
        return max(0.0, (100.0 - self.state.battery_pct) / 100.0)


class ShardedSwarm:
    """Рой без лидера: шарды, консенсус по правилам, отказоустойчивость.

    >>> sw = ShardedSwarm(model, states)
    >>> sw.elect_router(["код"])      # роутером станет узел с кодовым шардом
    >>> sw.forward(ids)               # результат равен монолиту
    """

    def __init__(self, model, states: Sequence[DeviceState],
                 width: float = 1.0, skills_by_layer: Optional[Dict[int, Set[str]]] = None) -> None:
        self.model = model
        self.width = width
        self.skills_by_layer = skills_by_layer or {}
        self.nodes: List[SwarmNode] = [SwarmNode(s) for s in states]
        self.hops = 0
        self.bytes_moved = 0
        self.rebuilds = 0
        self.current_router: Optional[str] = None
        self.rebalance()

    # --------------------------------------------------------- распределение
    def rebalance(self) -> None:
        """Пересобрать карту роя. Вызывается при любом изменении состава."""
        live = [n for n in self.nodes if n.online]
        if not live:
            raise ValueError("рой пуст: нет онлайн-узлов")
        n_layers = self.model.cfg.n_layer

        # вес узла: мощность, уменьшенная голодом (правило 3)
        weights = np.array([n.power * (1.0 - 0.5 * n.hunger) for n in live], dtype=float)
        weights = np.clip(weights, 1e-6, None)
        share = weights / weights.sum()

        counts = np.floor(share * n_layers).astype(int)
        if n_layers >= len(live):
            counts = np.maximum(counts, 1)
        while counts.sum() < n_layers:
            counts[int(np.argmax(share - counts / max(n_layers, 1)))] += 1
        while counts.sum() > n_layers:
            floor = 1 if n_layers >= len(live) else 0
            cand = np.where(counts > floor)[0]
            counts[cand[int(np.argmax(counts[cand]))]] -= 1

        # слабые узлы идут первыми (вход), мощные — в конце (тяжёлые слои)
        order = np.argsort(weights)
        cursor = 0
        for i in order:
            node = live[i]
            take = int(counts[i])
            layers = set(range(cursor, cursor + take))
            cursor += take
            # ширина шарда ограничена памятью устройства
            max_w = self._max_width_for(node, len(layers))
            skills: Set[str] = set()
            for l in layers:
                skills |= self.skills_by_layer.get(l, set())
            node.shard = Shard(layers, max_w, skills,
                               self._shard_bytes(len(layers), max_w))
        for n in self.nodes:
            if not n.online:
                n.shard = Shard()
        self.rebuilds += 1

    def _max_width_for(self, node: SwarmNode, n_layers: int) -> float:
        """Какую ширину узел способен хранить в своей памяти."""
        budget = node.state.available_bytes
        for w in (1.0, 0.75, 0.5, 0.25, 0.125):
            if self._shard_bytes(n_layers, w) <= budget:
                return w
        return 0.125

    def _shard_bytes(self, n_layers: int, width: float) -> int:
        d = self.model.width_dim(width)
        per_layer = self.model.blocks[0].active_params(d)
        return int((per_layer * n_layers + self.model.cfg.vocab_size * d) * 2 / 8)

    # ------------------------------------------------------------- консенсус
    def elect_router(self, need_skills: Sequence[str] = (),
                     complexity: float = 0.5) -> SwarmNode:
        """Выбрать временного роутера по четырём правилам. Лидера нет."""
        live = [n for n in self.nodes if n.online]
        best, best_score = None, -np.inf
        for n in live:
            # правило 1: кто ближе к задаче (есть нужные навыки в шарде)
            skill = n.shard.skills & set(need_skills)
            s_skill = len(skill) / max(len(set(need_skills)), 1) if need_skills else 0.0
            # правило 2: кто свободнее
            s_free = n.state.ram_free_frac
            # правило 3: кто голоднее — тот меньше хочет быть роутером
            s_hunger = -n.hunger
            # правило 4: кто умнее — для сложных задач
            s_power = np.log1p(n.power) / 5.0 * complexity
            score = 2.0 * s_skill + 0.5 * s_free + 0.5 * s_hunger + s_power
            n.router_score = float(score)
            if score > best_score:
                best, best_score = n, score
        self.current_router = best.name
        return best

    def consensus(self, proposal: str, votes_needed: float = 0.5) -> bool:
        """Резонанс вместо голосования: изменение проходит при поддержке большинства.

        Узел «за», если у него есть ресурс (не голодает и не перегрет).
        """
        live = [n for n in self.nodes if n.online]
        if not live:
            return False
        agree = sum(1 for n in live if n.state.throttle > 0.5)
        return agree / len(live) > votes_needed

    # ---------------------------------------------------------------- жизнь
    def join(self, state: DeviceState) -> SwarmNode:
        node = SwarmNode(state)
        self.nodes.append(node)
        self.rebalance()
        return node

    def leave(self, name: str, permanent: bool = False) -> None:
        """Узел ушёл. Шарды перераспределяются, ничего не теряется."""
        found = [n for n in self.nodes if n.name == name]
        if not found:
            raise KeyError(f"узел '{name}' не найден")
        if len([n for n in self.nodes if n.online]) <= 1:
            raise ValueError("нельзя отключить последний узел роя")
        if permanent:
            self.nodes = [n for n in self.nodes if n.name != name]
        else:
            found[0].online = False
        self.rebalance()

    def rejoin(self, name: str) -> None:
        """Узел вернулся — Протей растекается обратно."""
        for n in self.nodes:
            if n.name == name:
                n.online = True
        self.rebalance()

    # -------------------------------------------------------------- инференс
    def forward(self, idx):
        """Прогон по цепочке шардов. По сети идут активации, не веса."""
        from ..tensor import no_grad

        with no_grad():
            d = self.model.width_dim(self.width)
            x = self.model.embed(idx, d)
            self.hops = 0
            self.bytes_moved = 0
            chain = sorted([n for n in self.nodes if n.online and n.shard.layers],
                           key=lambda n: min(n.shard.layers))
            for i, node in enumerate(chain):
                for li in sorted(node.shard.layers):
                    x = self.model.run_block(li, x, d)
                if i < len(chain) - 1:
                    payload = int(np.prod(x.shape)) * 4
                    self.bytes_moved += payload
                    self.hops += 1
            return self.model.readout(x, d)

    # --------------------------------------------------------------- отчёты
    def topology(self) -> str:
        rows = []
        for n in sorted(self.nodes, key=lambda x: (min(x.shard.layers) if x.shard.layers else 99)):
            status = "●" if n.online else "○ offline"
            rows.append(f"  {status} {n.name:<20} {n.shard}"
                        + (f"  [{', '.join(sorted(n.shard.skills))}]" if n.shard.skills else ""))
        return "\n".join(rows)

    def coverage(self) -> bool:
        """Все ли слои модели покрыты живыми шардами."""
        covered: Set[int] = set()
        for n in self.nodes:
            if n.online:
                covered |= n.shard.layers
        return covered == set(range(self.model.cfg.n_layer))
