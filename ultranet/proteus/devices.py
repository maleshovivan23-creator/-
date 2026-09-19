"""Профили устройств — от ESP32 до датацентра.

Числа взяты из спецификации Протея (Часть 3) и используются мета-контроллером
для подбора конфигурации. Бюджет памяти — реальный ограничитель: если
упакованная модель не влезает, конфигурация отвергается.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

KB = 1024
MB = 1024 * KB
GB = 1024 * MB


@dataclass
class Device:
    """Описание железа."""

    name: str
    tier: str                 # micro | small | medium | large | datacenter
    ram_bytes: int
    ram_for_model: float = 0.35     # какую долю RAM можно отдать весам
    gflops: float = 0.1             # грубая оценка вычислительной мощности
    battery: bool = True
    always_on: bool = False

    @property
    def weight_budget(self) -> int:
        return int(self.ram_bytes * self.ram_for_model)

    def __str__(self) -> str:
        r = self.ram_bytes
        h = f"{r / GB:.0f} ГБ" if r >= GB else (f"{r / MB:.0f} МБ" if r >= MB else f"{r / KB:.0f} КБ")
        return f"{self.name} [{self.tier}, RAM {h}]"


# ------------------------------------------------------------------ каталог
DEVICES: Dict[str, Device] = {
    # микроуровень
    "esp32":        Device("ESP32", "micro", 520 * KB, 0.45, 0.001, True, True),
    "stm32":        Device("STM32", "micro", 256 * KB, 0.45, 0.0005, True, True),
    "rp2040":       Device("RP2040", "micro", 264 * KB, 0.45, 0.0005, True, True),
    "smartcard":    Device("Смарт-карта", "micro", 32 * KB, 0.5, 0.0001, False, True),
    "implant":      Device("Имплантат", "micro", 128 * KB, 0.4, 0.0002, True, True),
    # малый
    "earbuds":      Device("Наушники", "small", 256 * MB, 0.25, 0.05, True, True),
    "smartwatch":   Device("Умные часы", "small", 2 * GB, 0.25, 0.3, True, True),
    "glasses":      Device("Умные очки", "small", 4 * GB, 0.3, 0.5, True, True),
    "ring":         Device("Умное кольцо", "small", 64 * MB, 0.25, 0.01, True, True),
    "speaker":      Device("Умная колонка", "small", 1 * GB, 0.4, 0.2, False, True),
    "drone":        Device("Дрон", "small", 512 * MB, 0.3, 0.3, True, False),
    # средний
    "phone":        Device("Смартфон", "medium", 8 * GB, 0.35, 2.0, True, True),
    "phone_hi":     Device("Смартфон (флагман)", "medium", 16 * GB, 0.4, 4.0, True, True),
    "tablet":       Device("Планшет", "medium", 8 * GB, 0.4, 2.5, True, False),
    "laptop":       Device("Ноутбук", "medium", 32 * GB, 0.5, 8.0, True, False),
    "console":      Device("Игровая консоль", "medium", 16 * GB, 0.5, 10.0, False, False),
    "car":          Device("Автомобиль", "medium", 16 * GB, 0.5, 6.0, False, True),
    "tv":           Device("Smart TV", "medium", 4 * GB, 0.4, 1.0, False, True),
    # большой
    "workstation":  Device("Рабочая станция", "large", 128 * GB, 0.6, 50.0, False, False),
    "server":       Device("Сервер", "datacenter", 512 * GB, 0.7, 200.0, False, True),
    "cluster":      Device("Кластер", "datacenter", 4096 * GB, 0.8, 2000.0, False, True),
}


@dataclass
class DeviceState:
    """Мгновенное состояние устройства — меняется между запросами."""

    device: Device
    battery_pct: float = 100.0
    temperature_c: float = 35.0
    ram_free_frac: float = 1.0        # доля бюджета, реально доступная сейчас
    latency_budget_ms: float = 1000.0

    @property
    def available_bytes(self) -> int:
        return int(self.device.weight_budget * max(self.ram_free_frac, 0.0))

    @property
    def throttle(self) -> float:
        """Коэффициент 0..1: насколько урезать вычисления.

        Низкий заряд и перегрев заставляют Протея сжиматься.
        """
        f = 1.0
        if self.device.battery:
            if self.battery_pct < 10:
                f *= 0.25
            elif self.battery_pct < 20:
                f *= 0.4
            elif self.battery_pct < 50:
                f *= 0.75
        if self.temperature_c > 80:
            f *= 0.3
        elif self.temperature_c > 65:
            f *= 0.6
        return f

    def __str__(self) -> str:
        return (f"{self.device} батарея {self.battery_pct:.0f}% "
                f"t={self.temperature_c:.0f}°C throttle={self.throttle:.2f}")


def get_device(key: str) -> Device:
    if key not in DEVICES:
        raise KeyError(f"неизвестное устройство '{key}'. Доступны: {', '.join(sorted(DEVICES))}")
    return DEVICES[key]


def list_devices(tier: Optional[str] = None) -> List[Device]:
    return [d for d in DEVICES.values() if tier is None or d.tier == tier]
