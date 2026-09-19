# UltraNet 🧠 v4 + Proteus

**Ультимативная нейросеть на Python** — полноценный фреймворк глубокого обучения, написанный с нуля на чистом NumPy: собственный автоград, современные слои (RoPE, RMSNorm, SwiGLU), KV-кэш, оптимизаторы уровня 2023 года, тренер и CLI.

Никакого PyTorch и TensorFlow — только NumPy и математика.

```
GPT-2 style    параметров 1,122,144 | loss 0.1298 | ppl 1.14 | 47.6s
LLaMA style    параметров   668,160 | loss 0.0882 | ppl 1.09 | 26.7s   ← меньше и быстрее
```

## Возможности

| Модуль | Содержимое |
|---|---|
| `tensor.py` | Автоград-движок: динамический граф, broadcasting, **fused softmax/log-softmax**, `no_grad()`, 35+ операций |
| `nn.py` | `Linear`, `Conv2d` (im2col), `MaxPool2d`, `AvgPool2d`, `Embedding`, `LayerNorm`, **`RMSNorm`**, `BatchNorm1d`, `Dropout`, **`RotaryEmbedding` (RoPE)**, `MultiHeadAttention` (**KV-кэш**), **`SwiGLU`**, `TransformerBlock`, `RNNCell`, `GRUCell`, `LSTMCell` |
| `optim.py` | `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, **`Lion`**, **`Lookahead`**, **`EMA`** (с прогревом) + `CosineWarmup`, `OneCycleLR`, `StepLR`, `ReduceLROnPlateau` |
| `functional.py` | **Fused `cross_entropy`** (label smoothing, `ignore_index`), `focal_loss`, `bce_with_logits`, `huber_loss`, `mae_loss`, метрики: `accuracy`, `top_k_accuracy`, `f1_score`, `confusion_matrix`, `perplexity` |
| `models.py` | `MLP`, `ConvNet`, `ResNet`, `TextClassifier`, `GPT` (weight tying, top-k/top-p, repetition penalty, stop-токены) |
| `trainer.py` | Прогресс-бар, валидация, ранняя остановка, **накопление градиентов**, EMA, чекпоинты, коллбэки, ASCII-графики |
| `data.py` | `DataLoader`, `Dataset`, `CharTokenizer`, `WordTokenizer`, синтетические датасеты, `normalize` |
| `tokenizer.py` | **`BPETokenizer`** — байтовый BPE как в GPT-2: обучение, сжатие, save/load, любой Unicode без `<unk>` |
| `checkpoint.py` | **Полные чекпоинты**: веса + моменты оптимизатора + эпоха/шаг + история, строгая валидация форм |
| `gradcheck.py` | **Публичный `gradcheck`** — численная проверка градиентов ваших слоёв |
| `cli.py` | `train-text`, `generate`, `demo`, `bench` |

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## CLI за 30 секунд

```bash
python -m ultranet demo                      # обучение + ASCII-графики
python -m ultranet bench                     # бенчмарк скорости
python -m ultranet train-text --llama --steps 300 --out model.pkl
python -m ultranet generate --checkpoint model.pkl --prompt "нейронная" --top-p 0.9
```

## Быстрый старт

```python
import ultranet as un

x, y = un.make_spirals(n_per_class=400, n_classes=3)
xtr, ytr, xte, yte = un.train_test_split(x, y, test_size=0.2)

model = un.MLP([2, 96, 96, 3], activation="gelu")
opt = un.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)

trainer = un.Trainer(model, opt, grad_clip=1.0, ema_decay=0.99,
                     accum_steps=2, checkpoint_path="best.pkl")
trainer.fit(un.DataLoader(xtr, ytr, 64),
            un.DataLoader(xte, yte, 128, shuffle=False),
            epochs=25, patience=8)
print(trainer.plot_history())   # ASCII-график кривых обучения
```
→ **99.2% accuracy**, с графиком прямо в терминале:

```
 0.6916 ┤*
        │o
        │   *
        │      o
 0.0029 ┤             *   o  o   o  o  o   o  o   o  o   o  o   o
        └────────────────────────────────────────────────────────
         эпохи 1..17    * train  o val
```

### Языковая модель LLaMA-style

```python
import ultranet as un

tok = un.CharTokenizer(text)
ids = tok.encode(text)

cfg = un.GPTConfig.llama_style(tok.vocab_size,   # RoPE + RMSNorm + SwiGLU
                               block_size=48, n_layer=3, n_head=4, n_embd=96)
model = un.GPT(cfg)
opt = un.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
sched = un.CosineWarmup(opt, warmup=30, total=300)

for _ in range(300):
    xb, yb = un.make_lm_batches(ids, cfg.block_size, batch_size=16)
    loss = un.cross_entropy(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.clip_grad_norm(1.0); opt.step(); sched.step()

print(tok.decode(model.generate(tok.encode("нейронная "), 160,
                                temperature=0.7, top_p=0.9,
                                repetition_penalty=1.1)))   # KV-кэш включён по умолчанию
```

> `нейронная сеть учится на данных. градиент течёт назад через граф вычислений. трансформер смотрит на контекст через механизм внимания.`

### Автоград напрямую

```python
from ultranet import Tensor, no_grad

x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
w = Tensor([[0.5], [-1.5]], requires_grad=True)
((x @ w).silu() ** 2).sum().backward()
print(x.grad, w.grad)

with no_grad():          # инференс без построения графа
    y = (x @ w).softmax(-1)
```

## Бенчмарк (чистый NumPy, CPU)

```
▸ Выигрыш no_grad() на инференсе
▸ Слои
  RMSNorm                                    0.52 ms  x1.73 быстрее LayerNorm
▸ Генерация GPT (2.6M параметров)
  без кэша (32 токенов)                    631.68 ms  19.7 ms/токен
  с KV-кэшем (32 токенов)                  103.32 ms   3.2 ms/токен — x6.11
▸ Пропускная способность обучения
  GPT 1,994,240 парам., шаг обучения       164.17 ms  3,119 токенов/с
```

## Примеры

```bash
PYTHONPATH=. python examples/01_classification.py  # MLP + карта границ решений
PYTHONPATH=. python examples/02_convnet.py         # CNN на изображениях
PYTHONPATH=. python examples/03_gpt_text.py        # GPT генерирует текст
PYTHONPATH=. python examples/04_regression.py      # регрессия: MSE vs MAE vs Huber
PYTHONPATH=. python examples/05_llama_vs_gpt2.py   # сравнение архитектур
PYTHONPATH=. python examples/06_bpe_checkpoint.py  # BPE + возобновление обучения
```

## Воспроизводимость

```python
un.manual_seed(1234)    # инициализация весов, dropout, шаффлинг — всё детерминировано
```
Тест `test_full_training_run_reproducible` проверяет, что два независимых прогона обучения дают побитово идентичные кривые loss.

## Надёжность

Ошибки объясняют причину **и** решение:

```
ValueError: Linear: ожидался вход с последней размерностью 4, получен тензор (2, 7).
            Проверьте in_features слоя.
IndexError: Embedding: индексы должны быть в [0, 9], получен диапазон [99, 99].
            Увеличьте num_embeddings или проверьте словарь токенизатора.
```

Паспорт модели одной строкой — `model.summary()`:

```
параметр                    форма                    кол-во
-----------------------------------------------------------
tok_emb.weight              (400, 64)                25,600
blocks.0.attn.qkv.weight    (64, 192)                12,288
...
ИТОГО                                               214,848
память (float32)                                      0.82M
```

## Тесты

```bash
python -m pytest -q     # 83 passed
```

Покрытие включает:
- **численную проверку градиентов** (finite differences) для всех операций, свёрток, attention, RoPE, SwiGLU, RMSNorm, Huber/BCE
- **эквивалентность KV-кэша** полному пересчёту (ошибка ~1e-8) и совпадение генерации cache/no-cache
- **относительность RoPE**: скалярное произведение зависит только от разности позиций; норма векторов сохраняется
- **эквивалентность накопления градиентов** одному большому батчу
- сквозное обучение MLP / CNN / ResNet / TextClassifier / GPT (обе архитектуры), CLI-roundtrip
- **отсутствие алиасинга градиентов** — регрессионный тест на баг, найденный при оптимизации ядра
- **побитовая воспроизводимость** цикла обучения при `manual_seed`
- **восстановление состояния Adam** из чекпоинта и продолжение обучения
- `gradcheck` ловит намеренно сломанный `backward`

## Что нового в v3

- 🔤 **BPE-токенизатор** — байтовый, обучаемый, сжатие x2+, корректный Unicode и эмодзи
- 💾 **Полноценные чекпоинты** — `save_checkpoint`/`load_checkpoint` и `trainer.resume()`: моменты Adam восстанавливаются, loss после загрузки совпадает бит-в-бит
- 🎲 **`manual_seed`** — полная воспроизводимость (веса, dropout, батчи)
- 🔍 **`gradcheck` как публичный API** — проверяйте собственные слои
- 📋 **`model.summary()`** — таблица параметров и объём памяти
- 🛡️ **Понятные ошибки** вместо сырых сообщений NumPy
- ⚡ **Оптимизация ядра** — устранён `np.add.at` на горячем пути attention; найден и закрыт тестом баг с алиасингом градиентов

## Что нового в v2

- 🚀 **KV-кэш** — генерация в 6 раз быстрее
- 🌀 **RoPE** — относительные позиции, экстраполяция за `block_size`
- ⚡ **Fused cross-entropy и softmax** — аналитический градиент одним узлом
- 🦙 **LLaMA-style пресет**: `GPTConfig.llama_style(...)` — RMSNorm + SwiGLU + RoPE
- 🔗 **Weight tying** — экономия `vocab × n_embd` параметров
- 🎲 **top-p (nucleus), repetition penalty, greedy, stop-токены**
- 🏋️ **Lion, Lookahead, EMA, OneCycleLR, ReduceLROnPlateau**
- 🖥️ **CLI** + бенчмарк + ASCII-визуализация кривых обучения

## 🧬 Proteus — подсистема «один Протей на любом железе»

Реализация [спецификации Протея](examples/07_proteus.py) поверх UltraNet. Каждое заявление спеки — **проверяемый код и 57 тестов**, а не декларация.

```bash
PYTHONPATH=. python examples/07_proteus.py
```

| Заявление спецификации | Статус | Как проверено |
|---|---|---|
| «Токенизатор отсутствует, каждый байт — вход» | ✅ | `ByteTokenizer`, `vocab_size=260`; roundtrip для 10 языков, кода, ДНК, нот, эмодзи, рун |
| «Вложенность: одна модель — все размеры» | ✅ | `MatFormer` + matryoshka-обучение: 3 размера из одних весов, все бьют случайный baseline |
| «Квантование 1.58 бит, веса в {-1,0,1}» | ✅ | Измеренная энтропия **1.583 бит** = log₂3; упаковка 2 бит/вес, **x15.7 к fp32** |
| «Ранний выход» | ✅ | `forward_early_exit` — простые входы проходят меньше слоёв |
| «RAG вместо параметров» | ✅ | 305 фактов = 8.9 КБ против 301 КБ весов; обновляется без переобучения |
| «Мета-контроллер пересобирает под железо» | ✅ | План проверяется на бюджет байтов; батарея 5% и перегрев 90°C реально сжимают модель |
| «Рой: нет центра, нет облака» | ✅ | Вывод роя **побитово равен** монолиту; по сети идут активации (5.6 КБ), а не веса (150 КБ) |
| «Устройство умирает — Протей отступает» | ✅ | `leave()`/`join()` перераспределяют слои без потери результата |
| «Все языки одинаково дёшевы» | ⚠️ **уточнено** | Байты убирают штраф *словаря*, но не UTF-8: кириллица = 2 байта/символ, CJK = 3 |
| «Модель 1–3B на смартфоне, 70B на станции» | ❌ не проверено | NumPy на CPU; здесь масштаб 10⁵–10⁷ параметров. Архитектура масштабируется, железо — нет |
| «Обучение на твоих действиях, LoRA между запросами» | ❌ не реализовано | Требует онлайн-обучения; в этой версии нет |

### Что получается на практике

```
Смарт-карта      32КБ RAM  ->     57,696 параметров,  14КБ,  3 слоя
ESP32           520КБ RAM  ->    468,000 параметров, 114КБ,  3 слоя
Смартфон          8ГБ RAM  ->    615,648 параметров, 150КБ,  4 слоя
Смартфон 5% батареи, 85°C  ->     24,800 параметров,   6КБ,  1 слой
```

Рой из четырёх устройств:
```
Наушники     слои 0–0      Смартфон  слои 2–2
Умные часы   слои 1–1      Ноутбук   слои 3–3
вывод роя идентичен монолиту: True
передано 5.6 КБ активаций (веса 150.3 КБ — они НЕ передаются)
```

### Честная оценка

Работает **архитектура**, а не продукт. Модели в демо — 10⁵–10⁶ параметров, обучены на килобайтах за минуты, поэтому генерируют бессвязный текст. Это ожидаемо и честно: демонстрируются механизмы (вложенность, квантование, рой, адаптация), а не качество языковой модели. Механизмы масштабируются на реальные размеры; NumPy на CPU — нет.

```python
from ultranet.proteus import Proteus

p = Proteus.standard(device="phone")
p.remember("Столица Франции — Париж.")
print(p.device_matrix())                    # как выглядит на 9 устройствах
p.form_swarm(["earbuds", "phone", "laptop"])  # рой
r = p.respond("привет")
print(r.trace())                            # чем думал: план, слои, память
```

## Лицензия

MIT
