# UltraNet 🧠

**Ультимативная нейросеть на Python** — полноценный фреймворк глубокого обучения, написанный с нуля на чистом NumPy: собственный автоград, слои, оптимизаторы, тренер и готовые архитектуры вплоть до GPT-трансформера.

Никакого PyTorch и TensorFlow — только NumPy и математика.

```
ppl 71.71 ──► 1.08   (GPT, 1.1M параметров, 400 шагов, CPU)
```

## Что внутри

| Модуль | Содержимое |
|---|---|
| `ultranet/tensor.py` | Автоград-движок: `Tensor` с динамическим графом, broadcasting, 30+ дифференцируемых операций |
| `ultranet/nn.py` | `Linear`, `Conv2d` (im2col), `MaxPool2d`, `Embedding`, `LayerNorm`, `BatchNorm1d`, `Dropout`, `MultiHeadAttention`, `TransformerBlock`, `RNNCell`, `GRUCell`, `Sequential`, `Residual` |
| `ultranet/optim.py` | `SGD` (momentum/Nesterov), `Adam`, `AdamW`, `RMSprop`, `CosineWarmup`, clipping градиентов |
| `ultranet/functional.py` | `cross_entropy` (+ label smoothing), `binary_cross_entropy`, `mse_loss`, `accuracy` |
| `ultranet/models.py` | `MLP`, `ConvNet`, `GPT` с авторегрессионной генерацией (temperature, top-k) |
| `ultranet/trainer.py` | Цикл обучения, валидация, ранняя остановка, история метрик |
| `ultranet/data.py` | `DataLoader`, `CharTokenizer`, синтетические датасеты (спирали, луны, картинки) |

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Быстрый старт

```python
import ultranet as un

x, y = un.make_spirals(n_per_class=400, n_classes=3)
xtr, ytr, xte, yte = un.train_test_split(x, y, test_size=0.2)

model = un.MLP([2, 96, 96, 64, 3], activation="gelu", dropout=0.05)
opt = un.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)

trainer = un.Trainer(model, opt, grad_clip=1.0)
trainer.fit(un.DataLoader(xtr, ytr, 64),
            un.DataLoader(xte, yte, 128, shuffle=False),
            epochs=40, patience=10)
```
→ **100% accuracy** на тесте за считанные секунды.

### Своя языковая модель

```python
import ultranet as un

tok = un.CharTokenizer(text)
ids = tok.encode(text)

cfg = un.GPTConfig(vocab_size=tok.vocab_size, block_size=48,
                   n_layer=3, n_head=4, n_embd=96, dropout=0.05)
model = un.GPT(cfg)
opt = un.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)

for step in range(400):
    xb, yb = un.make_lm_batches(ids, cfg.block_size, batch_size=16)
    loss = un.cross_entropy(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.clip_grad_norm(1.0); opt.step()

print(tok.decode(model.generate(tok.encode("нейронная "), 160,
                                temperature=0.8, top_k=8)))
```

### Автоград напрямую

```python
from ultranet import Tensor

x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
w = Tensor([[0.5], [-1.5]], requires_grad=True)
loss = ((x @ w).gelu() ** 2).sum()
loss.backward()
print(x.grad, w.grad)
```

## Примеры

```bash
PYTHONPATH=. python examples/01_classification.py  # MLP + ASCII-карта границ решений
PYTHONPATH=. python examples/02_convnet.py         # CNN на изображениях
PYTHONPATH=. python examples/03_gpt_text.py        # GPT генерирует текст
```

## Тесты

```bash
PYTHONPATH=. python -m pytest tests -q
```

17 тестов, включая **численную проверку градиентов** (finite differences) для всех операций, свёрток, attention и LayerNorm, плюс сквозные проверки обучения MLP / CNN / GPT.

```
17 passed in 2.23s
```

## Ключевые особенности

- **Обратное распространение** по топологически отсортированному графу с корректным сворачиванием broadcasting-осей
- **Causal self-attention** с масками — проверено тестом, что будущие токены не влияют на прошлые
- **Свёртки через im2col** — векторизованный forward и backward без циклов по батчу
- **Pre-LN трансформер** в стиле GPT-2 с tanh-GELU
- **Сохранение/загрузка весов**: `model.save(path)` / `model.load(path)`
- Полностью типизированный код с документацией на русском

## Лицензия

MIT
