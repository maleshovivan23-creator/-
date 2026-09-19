# Запуск: один файл

Есть два самодостаточных файла, оба работают в Colab и Kaggle.

| файл | что внутри | когда брать |
|---|---|---|
| **`proteus_colab.py`** | **весь проект** — 35 модулей, ядро, Протей, обучение | нужен весь Протей |
| `proteus_train.py` | только обучение на TinyStories | нужно лишь обучить модель |

## proteus_colab.py — весь проект

```python
!pip install -q datasets
!wget -q https://raw.githubusercontent.com/maleshovivan23-creator/-/arena/01a0b93c-repo/proteus_colab.py

!python proteus_colab.py           # самопроверка: что внутри и работает ли
!python proteus_colab.py --steps 200                  # проба обучения
!python proteus_colab.py --steps 60000 --max-hours 11 # полный прогон
```

Импорты работают как у установленного пакета:

```python
import proteus_colab                     # регистрирует ultranet.* в sys.modules
from ultranet.models import GPT, GPTConfig
from ultranet.proteus import Proteus, MatFormer, MetaController
```

Проверено: все 460 тестов репозитория проходят против этого файла.

## proteus_train.py — только обучение

Всё в одном: токенизатор, модель, данные, обучение.
Ничего из репозитория не импортирует.

## Kaggle

Settings справа: **Accelerator → GPU T4**, **Internet → On**.

```python
!pip install -q datasets
!wget -q https://raw.githubusercontent.com/maleshovivan23-creator/-/arena/01a0b93c-repo/proteus_train.py

# 1. Проба — обязательно
!python proteus_train.py --steps 200 --name probe

# 2. Полный прогон
!python proteus_train.py --steps 60000 --max-hours 11
```

## Google Colab

Runtime → Change runtime type → **T4 GPU**.

```python
!pip install -q datasets
!wget -q https://raw.githubusercontent.com/maleshovivan23-creator/-/arena/01a0b93c-repo/proteus_train.py

from google.colab import drive; drive.mount('/content/drive')   # чтобы не потерять прогон

!python proteus_train.py --steps 200 --name probe
!python proteus_train.py --steps 60000 --max-hours 11
```

## Что смотреть на пробе

Единственный критерий — **стартовый loss ≈ ln(vocab_size)**. При `--vocab 4096` это **8.3**.

| что видно | что значит |
|---|---|
| 8.3 → ~5.0 за 200 шагов | всё исправно, запускайте полный прогон |
| 200 и больше | сломана инициализация |
| 3–4 | словарь меньше заявленного |
| loss стоит на месте | проблема в данных, не в модели |

## Если сессия оборвалась

Запустите ту же команду ещё раз — обучение продолжится с последнего чекпоинта.
`--max-hours 11` заставляет выйти заранее и сохранить состояние: Kaggle и Colab
рвут сессию без предупреждения.

## Куда пишутся результаты

Определяется само: `/kaggle/working`, `/content/drive/MyDrive/proteus` или `./runs`.
Внутри — `data/` (токенизатор и .bin) и папка прогона с `last.pt`, `log.jsonl`, `samples.txt`.

## Полезные флаги

```
--limit 50000      взять часть историй (быстрая проверка на данных)
--vocab 4096       размер словаря BPE
--dim 256 --layers 6 --heads 8    размер модели (~16M параметров)
--batch 32 --seq 256
--root /куда/писать
```
