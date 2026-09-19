"""Proteus — единый фасад: один объект, живущий на любом железе.

Собирает вместе байтовый вход, вложенную модель, мета-контроллер,
локальную память и рой. Это реализация цикла из Части 2 спецификации:
приём -> профилирование -> сборка -> мышление -> память -> ранний выход -> ответ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..tensor import no_grad
from .adapt import Feedback, PersonalAdapter
from .bytes import ByteTokenizer
from .assembly import Assembly, ProteusBody
from .capsules import CapsuleRegistry
from .controller import MetaController, Plan
from .devices import Device, DeviceState, get_device
from .matformer import MatConfig, MatFormer
from .memory import LocalMemory
from .quant import TernaryModel
from .shards import ShardedSwarm
from .steering import SteeringLibrary
from .swarm import Swarm


@dataclass
class Response:
    """Результат одного запроса — со всей телеметрией «как думал»."""

    text: str
    plan: Plan
    layers_used: int
    context_used: str = ""
    swarm_hops: int = 0
    stayed_local: bool = True
    steering: str = ""
    experts_active: Optional[int] = None
    capsules_active: float = 1.0
    adapter_applied: bool = False

    def __str__(self) -> str:
        return self.text

    def trace(self) -> str:
        lines = [
            f"  план      : {self.plan}",
            f"  слоёв     : {self.layers_used}",
            f"  локально  : {'да' if self.stayed_local else 'НЕТ'}",
        ]
        if self.context_used:
            lines.append(f"  из памяти : {self.context_used[:60]}...")
        if self.steering:
            lines.append(f"  вектор    : {self.steering}")
        if self.experts_active is not None:
            lines.append(f"  экспертов : {self.experts_active}")
        if self.capsules_active < 1.0:
            lines.append(f"  капсул    : {self.capsules_active * 100:.0f}% активно")
        if self.adapter_applied:
            lines.append(f"  адаптер   : LoRA применён")
        if self.swarm_hops:
            lines.append(f"  прыжков   : {self.swarm_hops}")
        return "\n".join(lines)


class Proteus:
    """Один Протей: байтовый вход, вложенная модель, адаптация под железо.

    >>> p = Proteus.tiny()
    >>> p.attach("phone")
    >>> r = p.respond("привет", max_new_tokens=8)
    >>> r.stayed_local
    True
    """

    def __init__(self, model: Optional[MatFormer] = None,
                 memory: Optional[LocalMemory] = None,
                 device: str | Device = "phone") -> None:
        self.tok = ByteTokenizer()
        self.model = model or MatFormer(MatConfig())
        self.memory = memory or LocalMemory()
        self.controller = MetaController(self.model)
        self.state = DeviceState(get_device(device) if isinstance(device, str) else device)
        self.swarm: Optional[Swarm] = None
        self.sharded: Optional[ShardedSwarm] = None
        self.cloud_enabled = False        # по умолчанию — никогда
        self.steering = SteeringLibrary(self.model)
        self.adapter: Optional[PersonalAdapter] = None
        self.capsules: Optional[CapsuleRegistry] = None
        self.task_vectors: Dict[str, str] = {}   # тип задачи -> имя steering-вектора
        self.body = ProteusBody(self.state.device, self.state)  # каталог 48 капсул

    # ------------------------------------------------------------ конструкторы
    @classmethod
    def tiny(cls, **kw) -> "Proteus":
        """Маленький Протей для тестов и микроконтроллеров."""
        cfg = MatConfig(n_layer=4, n_embd=96, head_dim=32, block_size=48,
                        widths=(0.33, 0.67, 1.0), exit_layers=(1,))
        return cls(MatFormer(cfg), **kw)

    @classmethod
    def standard(cls, **kw) -> "Proteus":
        cfg = MatConfig(n_layer=8, n_embd=256, head_dim=32, block_size=128,
                        widths=(0.25, 0.5, 1.0), exit_layers=(2, 5))
        return cls(MatFormer(cfg), **kw)

    @classmethod
    def with_experts(cls, n_experts: int = 4, top_k: int = 2, **kw) -> "Proteus":
        """Протей с MoE: разные эксперты для разных задач."""
        cfg = MatConfig(n_layer=4, n_embd=128, head_dim=32, block_size=64,
                        widths=(0.25, 0.5, 1.0), exit_layers=(1,),
                        n_experts=n_experts, top_k_experts=top_k)
        return cls(MatFormer(cfg), **kw)

    # ------------------------------------------------------------ подсистемы
    def enable_adapter(self, rank: int = 4) -> PersonalAdapter:
        """Включить персонализацию (LoRA поверх замороженного ядра)."""
        self.adapter = PersonalAdapter(self.model, rank=rank)
        return self.adapter

    def plan_capsules(self, prompt: str) -> Assembly:
        """Какие из 48 капсул каталога нужны этому запросу на этом железе."""
        return self.body.assemble(prompt)

    def enable_capsules(self, skills_by_expert: Optional[Dict[int, set]] = None
                        ) -> CapsuleRegistry:
        self.capsules = CapsuleRegistry.from_model(self.model, skills_by_expert)
        return self.capsules

    def bind_vector(self, task: str, vector_name: str) -> None:
        """Связать тип задачи со steering-вектором."""
        self.task_vectors[task] = vector_name

    @staticmethod
    def classify_task(prompt: str) -> str:
        """Первые слои — классификатор задачи (из спецификации, часть 2)."""
        p = prompt.lower()
        if any(k in p for k in ("код", "функци", "def ", "class ", "code", "напиши программ")):
            return "код"
        if any(k in p for k in ("стих", "поэм", "придумай", "сочини", "poem")):
            return "творчество"
        if any(k in p for k in ("посчитай", "сколько будет", "реши", "уравнен", "+", "=")):
            return "математика"
        if any(k in p for k in ("почему", "объясни", "проанализируй", "разбер", "why", "explain")):
            return "рассуждение"
        return "диалог"

    # ------------------------------------------------------------- устройства
    def attach(self, device: str | Device, **state_kw) -> "Proteus":
        """Переехать на устройство (или описать его текущее состояние)."""
        dev = get_device(device) if isinstance(device, str) else device
        self.state = DeviceState(dev, **state_kw)
        return self

    def form_swarm(self, devices: Sequence[str], width: float = 1.0) -> Swarm:
        """Собрать рой из нескольких устройств."""
        states = [DeviceState(get_device(d)) for d in devices]
        self.swarm = Swarm(self.model, states, width=width)
        return self.swarm

    def form_sharded_swarm(self, devices: Sequence[str],
                           skills_by_layer: Optional[Dict[int, set]] = None,
                           width: float = 1.0) -> ShardedSwarm:
        """Рой с шардами, консенсусом и выбором роутера без лидера."""
        states = [DeviceState(get_device(d)) for d in devices]
        self.sharded = ShardedSwarm(self.model, states, width=width,
                                    skills_by_layer=skills_by_layer)
        return self.sharded

    def dissolve_swarm(self) -> None:
        self.swarm = None
        self.sharded = None

    # ------------------------------------------------------------ обратная связь
    def feedback(self, prompt: str, reward: float = 1.0) -> None:
        """Лайк/правка/игнор — сигнал для средней петли адаптации."""
        if self.adapter is None:
            self.enable_adapter()
        self.adapter.observe(Feedback(self.tok.encode(prompt), reward=reward))

    def consolidate(self, steps: int = 20, **kw) -> Dict[str, List[float]]:
        """Средняя петля: дообучить LoRA на накопленных сигналах."""
        if self.adapter is None:
            return {"loss": []}
        return self.adapter.consolidate(steps=steps, **kw)

    # ------------------------------------------------------------------ знание
    def remember(self, *facts: str) -> None:
        """Положить знание в локальную память (вне весов)."""
        for f in facts:
            self.memory.add(f)

    # ---------------------------------------------------------------- ответ
    @no_grad()
    def respond(self, prompt: str, max_new_tokens: int = 48,
                task_complexity: Optional[float] = None,
                temperature: float = 0.7, top_k: int = 40,
                use_memory: bool = True, seed: Optional[int] = None) -> Response:
        """Полный цикл: профилирование -> план -> (память) -> генерация."""
        complexity = self.estimate_complexity(prompt) if task_complexity is None else task_complexity
        task = self.classify_task(prompt)
        plan = self.controller.plan(self.state, complexity)

        # мета-контроллер крутит число экспертов: простая задача — меньше
        experts_active = None
        if self.model.cfg.n_experts > 1:
            experts_active = int(np.clip(
                round(1 + (self.model.cfg.n_experts - 1) * complexity),
                1, self.model.cfg.n_experts))
            self.model.set_active_experts(experts_active)

        # капсулы: просыпаются только нужные
        capsule_frac = 1.0
        if self.capsules is not None:
            self.capsules.assemble([task])
            capsule_frac = self.capsules.active_fraction()

        # steering: направление мышления под тип задачи
        steer_name = self.task_vectors.get(task, "")
        if steer_name and steer_name in self.steering:
            self.steering.apply(steer_name)

        context = ""
        if use_memory and len(self.memory):
            context = self.memory.context_for(prompt, top_k=2, max_chars=200)

        full = f"{context}\n{prompt}" if context else prompt
        ids = self.tok.encode(full)[-self.model.cfg.block_size:]
        if not ids:
            ids = [self.tok.bos_id]

        if self.swarm is not None:
            logits = self.swarm.forward(np.array([ids]))
            out_ids = self._sample_loop(ids, plan, max_new_tokens, temperature, top_k, seed)
            hops = self.swarm.hops
        else:
            out_ids = self._sample_loop(ids, plan, max_new_tokens, temperature, top_k, seed)
            hops = 0

        text = self.tok.decode(out_ids[len(ids):])
        if steer_name:
            self.steering.clear()
        self.model.set_active_experts(None)

        # быстрая петля: запоминаем взаимодействие для будущей консолидации
        if self.adapter is not None:
            self.adapter.observe(Feedback(self.tok.encode(full), reward=0.0, tag=task))

        return Response(text=text, plan=plan, layers_used=plan.n_layers,
                        context_used=context, swarm_hops=hops,
                        stayed_local=not self.cloud_enabled,
                        steering=steer_name, experts_active=experts_active,
                        capsules_active=capsule_frac,
                        adapter_applied=self.adapter is not None)

    def _sample_loop(self, ids: List[int], plan: Plan, n: int,
                     temperature: float, top_k: int, seed: Optional[int]) -> List[int]:
        return self.model.generate(ids, max_new_tokens=n, width=plan.width,
                                   temperature=temperature, top_k=top_k,
                                   n_layers=plan.n_layers, seed=seed)

    @staticmethod
    def estimate_complexity(prompt: str) -> float:
        """Грубая оценка сложности запроса: команда или размышление."""
        p = prompt.lower().strip()
        simple_markers = ("включи", "выключи", "открой", "закрой", "стоп", "да", "нет",
                          "on", "off", "open", "close", "yes", "no")
        if any(p.startswith(m) for m in simple_markers) or len(p) < 12:
            return 0.15
        hard_markers = ("почему", "объясни", "проанализируй", "докажи", "напиши код",
                        "why", "explain", "analyze", "prove")
        if any(m in p for m in hard_markers) or len(p) > 120:
            return 0.95
        return 0.5

    # --------------------------------------------------------------- отчёты
    def footprint(self) -> str:
        """Сколько Протей весит на текущем устройстве."""
        q = TernaryModel(self.model)
        plan = self.controller.plan(self.state, 1.0)
        return (f"{self.state}\n"
                f"  полная модель : {q.stats}\n"
                f"  активная часть: {plan.active_params:,} параметров "
                f"({plan.weight_bytes / 1024:.0f} КБ)\n"
                f"  память знаний : {self.memory.size_bytes() / 1024:.1f} КБ "
                f"({len(self.memory)} фактов, вне весов)")

    def device_matrix(self, devices: Optional[Sequence[str]] = None,
                      task_complexity: float = 0.8) -> str:
        """Таблица: как Протей выглядит на разных устройствах."""
        keys = list(devices or ["smartcard", "esp32", "ring", "earbuds", "smartwatch",
                                "phone", "laptop", "workstation", "server"])
        rows = [f"{'устройство':<22} {'RAM':>9} {'активно':>12} {'вес':>10} {'слоёв':>6}"]
        rows.append("-" * 64)
        for k in keys:
            dev = get_device(k)
            plan = self.controller.plan(DeviceState(dev), task_complexity)
            ram = (f"{dev.ram_bytes / 1024**3:.0f}ГБ" if dev.ram_bytes >= 1024**3
                   else f"{dev.ram_bytes / 1024**2:.0f}МБ" if dev.ram_bytes >= 1024**2
                   else f"{dev.ram_bytes / 1024:.0f}КБ")
            w = (f"{plan.weight_bytes / 1024**2:.1f}МБ" if plan.weight_bytes >= 1024**2
                 else f"{plan.weight_bytes / 1024:.0f}КБ")
            mark = "" if plan.fits else "  (не влезает!)"
            rows.append(f"{dev.name:<22} {ram:>9} {plan.active_params:>12,} {w:>10} "
                        f"{plan.n_layers:>6}{mark}")
        return "\n".join(rows)
