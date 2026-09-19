# Запуск обучения: один файл

`proteus_train.py` — всё в одном: токенизатор, модель, данные, обучение.
Ничего из репозитория не импортирует. Работает в Colab и Kaggle.

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
