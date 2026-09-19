"""Собрать весь проект в один .py файл для Colab/Kaggle.

Склеивать модули подряд нельзя: внутри пакета 35 файлов со взаимными
относительными импортами (`from .tensor import Tensor`), и порядок
склейки пришлось бы угадывать вручную.

Вместо этого исходники встраиваются как строки и регистрируются в
sys.modules настоящими модулями. Импорты продолжают работать как есть,
поведение совпадает с установленным пакетом.

    python tools/make_bundle.py -o proteus_colab.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "ultranet"

HEADER = '''"""Протей — весь проект в одном файле. Для Google Colab и Kaggle.

Это автосборка: {n_files} модулей пакета ultranet ({n_lines} строк) встроены
ниже как исходники и регистрируются в sys.modules. Импорты работают
как у обычной установки:

    from ultranet.models import GPT, GPTConfig
    from ultranet.proteus import Proteus, MatFormer

Обучение языковой модели на TinyStories:

    !pip install -q datasets
    !python proteus_colab.py --steps 200                    # проба
    !python proteus_colab.py --steps 60000 --max-hours 11   # полный прогон

Проба обязательна. Единственный критерий: стартовый loss должен быть
примерно ln(vocab_size) — для словаря 4096 это 8.3. Сильно больше —
сломана инициализация, сильно меньше — словарь не тот.

Без аргументов запускает самопроверку: обучает крошечный GPT и печатает
сводку по возможностям.

Сгенерировано tools/make_bundle.py — правьте исходники, не этот файл.
"""
from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

# ═════════════════════════════════════════════════════════════════════
# Исходники пакета. Каждый модуль хранится строкой и исполняется ниже
# в собственном пространстве имён.
# ═════════════════════════════════════════════════════════════════════
_SOURCES = {{}}

'''

LOADER = '''

# ═════════════════════════════════════════════════════════════════════
# Регистрация пакета в sys.modules
# ═════════════════════════════════════════════════════════════════════
class _BundleLoader:
    """Загрузчик, отдающий модули из _SOURCES.

    Заранее исполнять модули в угаданном порядке нельзя: между ними
    взаимные зависимости (например ultranet.__main__ импортирует
    ultranet.cli). Поэтому модуль исполняется в момент первого импорта —
    порядок разрешает сам Python.
    """

    PACKAGES = {"ultranet", "ultranet.proteus"}

    def find_module(self, fullname, path=None):      # старый протокол
        return self if fullname in _SOURCES else None

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _SOURCES:
            return None
        return ModuleSpec(fullname, self,
                          is_package=fullname in self.PACKAGES)

    def create_module(self, spec):
        return None                                   # обычный модуль

    def exec_module(self, module):
        name = module.__name__
        module.__file__ = f"<bundled {name}>"
        if name in self.PACKAGES:
            module.__path__ = []
            module.__package__ = name
        else:
            module.__package__ = name.rsplit(".", 1)[0]
        exec(compile(_SOURCES[name], module.__file__, "exec"), module.__dict__)

    def load_module(self, fullname):                  # старый протокол
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__loader__ = self
        sys.modules[fullname] = mod
        try:
            self.exec_module(mod)
        except Exception:
            sys.modules.pop(fullname, None)
            raise
        return mod


sys.meta_path.insert(0, _BundleLoader())

'''


def collect() -> dict:
    """Собрать {имя_модуля: исходник}, пакеты в конце."""
    out = {}
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG.parent)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts)] = path.read_text(encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="proteus_colab.py")
    args = ap.parse_args()

    sources = collect()
    trainer = (ROOT / "proteus_train.py").read_text(encoding="utf-8")

    n_lines = sum(len(s.splitlines()) for s in sources.values())
    parts = [HEADER.format(n_files=len(sources), n_lines=f"{n_lines:,}")]

    for name, src in sources.items():
        parts.append(f"_SOURCES[{name!r}] = {src!r}\n\n")

    parts.append(LOADER)
    parts.append(_trainer_section(trainer))
    snippet = (ROOT / "tools" / "selftest_snippet.py").read_text(encoding="utf-8")
    # выкинуть пояснительную шапку сниппета — она про генератор, не про бандл
    snippet = "\n".join(ln for ln in snippet.splitlines()
                        if not ln.startswith("# Эта часть")
                        and not ln.startswith("# Держим её")
                        and not ln.startswith("# кавычки"))
    parts.append("\n" + snippet)

    text = "".join(parts)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"{args.out}: {len(text.splitlines()):,} строк, "
          f"{len(sources)} модулей, {len(text) / 1024:.0f} КБ")


def _trainer_section(trainer: str) -> str:
    """Код обучения: выкинуть его собственный docstring и дубли импортов."""
    lines = trainer.splitlines(keepends=True)
    # пропустить модульный docstring
    start = 0
    if lines and lines[0].lstrip().startswith('"""'):
        for i, ln in enumerate(lines[1:], 1):
            if '"""' in ln:
                start = i + 1
                break
    # `from __future__` допустим только в начале файла, а здесь мы уже
    # в середине — в шапке он объявлен один раз и действует на весь модуль
    body = "".join(ln for ln in lines[start:]
                   if not ln.startswith("from __future__ import"))
    # своя точка входа добавляется в SELFTEST: без аргументов — самопроверка
    body = body.replace('if __name__ == "__main__":\n    main()\n', "")
    return (
        "\n# ══════════════════════════════════════════════════════════"
        "═══════════\n"
        "# Обучение на TinyStories\n"
        "# ══════════════════════════════════════════════════════════"
        "═══════════\n" + body
    )


if __name__ == "__main__":
    main()
