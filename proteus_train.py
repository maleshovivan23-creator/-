"""Протей: обучение языковой модели на TinyStories. Один файл, без зависимостей от репозитория.

Работает и в Google Colab, и в Kaggle — окружение определяется само.

    !pip install -q datasets
    !wget -q https://raw.githubusercontent.com/maleshovivan23-creator/-/arena/01a0b93c-repo/proteus_train.py
    !python proteus_train.py --steps 200          # проба
    !python proteus_train.py --steps 60000 --max-hours 11   # полный прогон

Проба обязательна. Смотреть нужно на одно: стартовый loss должен быть
примерно ln(vocab_size) — для словаря 4096 это 8.3. Если сильно больше,
сломана инициализация; если сильно меньше — словарь не тот.

Всё содержимое — перенос проверенного кода проекта: быстрый BPE,
TorchGPT с инициализацией как у NumPy-референса, цикл обучения с
возобновлением после обрыва сессии.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════ окружение
def detect_root() -> Path:
    """Куда писать данные и чекпоинты: Kaggle, Colab или локально."""
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    if Path("/content").exists():
        drive = Path("/content/drive/MyDrive")
        return drive / "proteus" if drive.exists() else Path("/content/proteus")
    return Path("./runs")


# ═══════════════════════════════════════════════════════════ токенизатор
_CHUNK_RE = re.compile(r"\s*\S+|\s+")
_MAX_CHUNK = 64


def _split_chunks(text: str) -> Iterator[str]:
    """Резать по границам слов: слияния BPE почти не пересекают пробел."""
    for m in _CHUNK_RE.finditer(text):
        chunk = m.group()
        if len(chunk) <= _MAX_CHUNK:
            yield chunk
        else:
            for i in range(0, len(chunk), _MAX_CHUNK):
                yield chunk[i:i + _MAX_CHUNK]


class BPETokenizer:
    """BPE на байтах: 256 базовых + выученные слияния.

    Кодирование идёт по кускам с кэшем. Наивная версия пересчитывала пары
    по всему тексту на каждое слияние — 27k символов/с, то есть 19 часов
    на TinyStories. Здесь 15M символов/с, около двух минут.
    """

    _CACHE_MAX = 200_000

    def __init__(self, merges: Optional[Dict[Tuple[int, int], int]] = None) -> None:
        self.merges: Dict[Tuple[int, int], int] = merges or {}
        self._cache: Dict[str, List[int]] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out, i, n = [], 0, len(ids)
        a, b = pair
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @classmethod
    def train(cls, text: str, vocab_size: int = 4096, verbose: bool = True) -> "BPETokenizer":
        """Учить слияния по тем же кускам, по которым потом кодируем.

        Если учить по сплошному тексту, словарь копит слияния через
        пробел, применить их encode не сможет, и реальное сжатие выйдет
        заметно хуже измеренного при обучении.
        """
        assert vocab_size >= 256, "словарь не может быть меньше 256 байт"
        chunks = [list(c.encode("utf-8")) for c in _split_chunks(text)]
        merges: Dict[Tuple[int, int], int] = {}
        for new_id in range(256, vocab_size):
            counts: Counter = Counter()
            for ids in chunks:
                counts.update(zip(ids, ids[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            chunks = [cls._merge(c, pair, new_id) if len(c) >= 2 else c for c in chunks]
            merges[pair] = new_id
            if verbose and new_id % 500 == 0:
                print(f"    слияние {new_id}/{vocab_size}")
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> List[int]:
        ids = list(chunk.encode("utf-8"))
        if len(ids) < 2:
            return ids
        merges = self.merges
        while True:
            best_rank, best_pos = None, -1
            for i in range(len(ids) - 1):
                rank = merges.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, i
            if best_rank is None:
                return ids
            ids[best_pos:best_pos + 2] = [best_rank]

    def encode(self, text: str) -> List[int]:
        if not self.merges:
            return list(text.encode("utf-8"))
        out: List[int] = []
        for chunk in _split_chunks(text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk)
                if len(self._cache) < self._CACHE_MAX:
                    self._cache[chunk] = cached
            out.extend(cached)
        return out

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def compression_ratio(self, text: str) -> float:
        return len(text.encode("utf-8")) / max(len(self.encode(text)), 1)

    def save(self, path) -> None:
        payload = {"merges": [[list(k), v] for k, v in self.merges.items()]}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({tuple(k): v for k, v in payload["merges"]})


# ═══════════════════════════════════════════════════════════════ модель
import torch                                     # noqa: E402
import torch.nn as tnn                           # noqa: E402
import torch.nn.functional as TF                 # noqa: E402


class GPTConfig:
    def __init__(self, vocab_size=4096, block_size=256, n_layer=6, n_head=8,
                 n_embd=256, dropout=0.0, rope=True, norm="rms", swiglu=True,
                 tie_weights=True):
        self.vocab_size, self.block_size = vocab_size, block_size
        self.n_layer, self.n_head, self.n_embd = n_layer, n_head, n_embd
        self.dropout, self.rope = dropout, rope
        self.norm, self.swiglu, self.tie_weights = norm, swiglu, tie_weights

    def asdict(self) -> dict:
        return dict(self.__dict__)


class RMSNorm(tnn.Module):
    """x / sqrt(mean(x^2) + eps) * gamma — eps ВНУТРИ корня."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = tnn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma


class LayerNorm(tnn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = tnn.Parameter(torch.ones(dim))
        self.beta = tnn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = (x - mu).pow(2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.gamma + self.beta


def gelu_tanh(x):
    """Приближение tanh, как в GPT-2 (не erf)."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


class RotaryEmbedding(tnn.Module):
    """RoPE с чередованием соседних элементов (0,1), (2,3), не half-split."""

    def __init__(self, head_dim: int, max_seq: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        inv = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        freqs = np.outer(np.arange(max_seq, dtype=np.float32), inv)
        self.register_buffer("cos", torch.tensor(np.cos(freqs), dtype=torch.float32))
        self.register_buffer("sin", torch.tensor(np.sin(freqs), dtype=torch.float32))

    def forward(self, x):
        t = x.shape[2]
        cos, sin = self.cos[:t][None, None], self.sin[:t][None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class MultiHeadAttention(tnn.Module):
    def __init__(self, dim, n_heads, dropout, causal, rope, max_seq):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.causal, self.drop_p = causal, dropout
        self.qkv = tnn.Linear(dim, 3 * dim, bias=False)
        self.proj = tnn.Linear(dim, dim, bias=True)
        self.rope = RotaryEmbedding(self.head_dim, max_seq) if rope else None

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        # SDPA сам применяет причинную маску и быстрее ручного softmax
        y = TF.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0,
            is_causal=self.causal)
        y = y.transpose(1, 2).reshape(b, t, c)
        return TF.dropout(self.proj(y), self.drop_p, self.training)


class SwiGLU(tnn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        hidden = int(8 * dim / 3 // 32 * 32) or 4 * dim
        self.w_gate = tnn.Linear(dim, hidden, bias=False)
        self.w_up = tnn.Linear(dim, hidden, bias=False)
        self.w_down = tnn.Linear(hidden, dim, bias=False)
        self.drop_p = dropout

    def forward(self, x):
        return TF.dropout(self.w_down(TF.silu(self.w_gate(x)) * self.w_up(x)),
                          self.drop_p, self.training)


class GELUMLP(tnn.Module):
    def __init__(self, dim, ratio=4, dropout=0.0):
        super().__init__()
        self.fc1 = tnn.Linear(dim, ratio * dim)
        self.fc2 = tnn.Linear(ratio * dim, dim)
        self.drop_p = dropout

    def forward(self, x):
        return TF.dropout(self.fc2(gelu_tanh(self.fc1(x))), self.drop_p, self.training)


class TransformerBlock(tnn.Module):
    def __init__(self, dim, n_heads, dropout, rope, norm, swiglu, max_seq):
        super().__init__()
        mk = (lambda: RMSNorm(dim)) if norm == "rms" else (lambda: LayerNorm(dim))
        self.ln1, self.ln2 = mk(), mk()
        self.attn = MultiHeadAttention(dim, n_heads, dropout, True, rope, max_seq)
        self.mlp = SwiGLU(dim, dropout) if swiglu else GELUMLP(dim, 4, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ProteusGPT(tnn.Module):
    """Декодер-трансформер: RoPE + RMSNorm + SwiGLU, weight tying."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = tnn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = None if cfg.rope else tnn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop_p = cfg.dropout
        self.blocks = tnn.ModuleList([
            TransformerBlock(cfg.n_embd, cfg.n_head, cfg.dropout, cfg.rope,
                             cfg.norm, cfg.swiglu, cfg.block_size * 4)
            for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd) if cfg.norm == "rms" else LayerNorm(cfg.n_embd)
        self.head = None if cfg.tie_weights else tnn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """Embedding N(0, 0.02), Linear kaiming gain=2.0.

        Умолчания torch тут не годятся: Embedding он делает N(0, 1), а при
        weight tying эта же матрица — выходная голова, поэтому логиты на
        старте оказываются примерно в 50 раз крупнее нужного и loss
        начинается с ~240 вместо ln(vocab) ≈ 8.3. Такой прогон либо
        расходится, либо тратит первые тысячи шагов впустую.
        """
        tnn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        if self.pos_emb is not None:
            tnn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, tnn.Linear):
                tnn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / m.weight.shape[1]))
                if m.bias is not None:
                    tnn.init.zeros_(m.bias)
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layer)
        for blk in self.blocks:
            blk.attn.proj.weight.data *= scale
        if self.head is not None:
            tnn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(t, device=idx.device))[None]
        x = TF.dropout(x, self.drop_p, self.training)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T if self.head is None else self.head(x)
        if targets is None:
            return logits
        loss = TF.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ═══════════════════════════════════════════════════════════════ данные
BPE_SAMPLE_CHARS = 2_000_000


def prepare_data(data_dir: Path, vocab_size: int = 4096,
                 limit: Optional[int] = None, val_frac: float = 0.005) -> BPETokenizer:
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_path = data_dir / "tokenizer.json"
    train_path, val_path = data_dir / "train.bin", data_dir / "val.bin"

    if train_path.exists() and tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        n = train_path.stat().st_size // 2
        print(f"данные уже готовы: {n:,} токенов, словарь {tok.vocab_size}")
        return tok

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("нужен datasets:  pip install datasets")

    print("качаю TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"историй: {len(ds):,}")

    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        print(f"токенизатор готов: словарь {tok.vocab_size}")
    else:
        sample, total = [], 0
        for ex in ds:
            sample.append(ex["text"])
            total += len(ex["text"])
            if total >= BPE_SAMPLE_CHARS:
                break
        print(f"учу BPE на {total:,} символах, словарь {vocab_size}...")
        t0 = time.time()
        tok = BPETokenizer.train("\n".join(sample), vocab_size=vocab_size)
        tok.save(tok_path)
        print(f"  готово за {time.time() - t0:.0f} с, словарь {tok.vocab_size}")

    if tok.vocab_size >= 65536:
        raise SystemExit("словарь не влезает в uint16 — уменьшите --vocab")

    print("токенизирую...")
    t0 = time.time()
    sep = tok.encode("\n")
    chunks, total = [], 0
    for i, ex in enumerate(ds):
        ids = tok.encode(ex["text"])
        ids.extend(sep)
        chunks.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        if i and i % 50000 == 0:
            print(f"  {i:>8,}/{len(ds):,}  {total:>12,} токенов  "
                  f"{total / (time.time() - t0):>10,.0f} ток/с")

    ids = np.concatenate(chunks)
    n_val = max(1024, int(len(ids) * val_frac))
    ids[:-n_val].tofile(train_path)
    ids[-n_val:].tofile(val_path)
    print(f"готово за {(time.time() - t0) / 60:.1f} мин | "
          f"train {len(ids) - n_val:,} | val {n_val:,}")
    return tok


# ═══════════════════════════════════════════════════════════════ обучение
def get_batch(data, batch, seq, device, rng):
    ix = rng.integers(0, len(data) - seq - 1, size=batch)
    x = np.stack([data[i:i + seq] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + seq + 1] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


def cosine_warmup(opt, warmup, total):
    def fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def estimate_loss(model, data, batch, seq, device, rng, iters=20):
    model.eval()
    out = []
    for _ in range(iters):
        x, y = get_batch(data, batch, seq, device, rng)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            _, loss = model(x, y)
        out.append(loss.item())
    model.train()
    return float(np.mean(out))


@torch.no_grad()
def sample(model, tok, prompt, max_new, device, block, temperature=0.8, top_k=40, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    ids = tok.encode(prompt) or [10]
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        logits = model(x[:, -block:])[:, -1, :].float()
        if temperature > 0:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=g)
        else:
            nxt = logits.argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    if was_training:
        model.train()
    return tok.decode(x[0].tolist())


def train(args) -> None:
    root = Path(args.root) if args.root else detect_root()
    data_dir, out = root / "data", root / args.name
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ВНИМАНИЕ: GPU не найден. В Kaggle: Settings -> Accelerator -> GPU T4.\n"
              "          В Colab: Runtime -> Change runtime type -> T4 GPU.")
    rng = np.random.default_rng(args.seed)

    tok = prepare_data(data_dir, args.vocab, args.limit)
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.seq,
                    n_layer=args.layers, n_head=args.heads, n_embd=args.dim,
                    dropout=args.dropout)
    model = ProteusGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nмодель {n_params / 1e6:.2f}M параметров | {device} | "
          f"ожидаемый стартовый loss ln({tok.vocab_size}) = {math.log(tok.vocab_size):.2f}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    sched = cosine_warmup(opt, max(100, args.steps // 20), args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    ckpt = out / "last.pt"
    step0 = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step0 = st["step"] + 1
        print(f"продолжаю с шага {step0}")

    def save(step, name="last.pt"):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "cfg": cfg.asdict()}, out / name)

    # Kaggle и Colab рвут сессию без предупреждения: выходим сами, заранее
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    log_path = out / "log.jsonl"
    model.train()
    t0, tokens = time.time(), 0

    for step in range(step0, args.steps):
        x, y = get_batch(train_data, args.batch, args.seq, device, rng)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        tokens += args.batch * args.seq

        if step % args.log_every == 0:
            tps = tokens / max(time.time() - t0, 1e-9)
            print(f"шаг {step:>6} | loss {loss.item():7.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | {tps:,.0f} ток/с")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": round(loss.item(), 4),
                                    "lr": sched.get_last_lr()[0],
                                    "tok_s": round(tps)}) + "\n")

        if step and step % args.eval_every == 0:
            vl = estimate_loss(model, val_data, args.batch, args.seq, device, rng)
            print(f"  val loss {vl:.4f} | ppl {math.exp(min(vl, 20)):.1f}")

        if step and step % args.sample_every == 0:
            text = sample(model, tok, args.prompt, args.sample_len, device, args.seq)
            print(f"  --- сэмпл ---\n{text}\n  -------------")
            with open(out / "samples.txt", "a") as f:
                f.write(f"--- шаг {step} ---\n{text}\n\n")

        if step and step % args.ckpt_every == 0:
            save(step)

        if deadline and time.time() > deadline:
            save(step)
            print(f"\nлимит {args.max_hours} ч исчерпан на шаге {step}. "
                  f"Осталось {args.steps - step - 1:,} шагов.")
            print("Запустите ту же команду ещё раз — продолжит отсюда.")
            return

    save(args.steps - 1, "final.pt")
    print(f"\nготово. Чекпоинты и сэмплы: {out}")
    for p in [args.prompt, "Lily found a", "The little boy was very"]:
        print(f"\n> {sample(model, tok, p, 120, device, args.seq)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение Протея на TinyStories")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None,
                    help="сколько историй взять (для быстрой пробы)")
    ap.add_argument("--max-hours", type=float, default=None, dest="max_hours",
                    help="выйти заранее, сохранив состояние")
    ap.add_argument("--root", default=None, help="куда писать (по умолчанию определяется само)")
    ap.add_argument("--name", default="tinystories-16m")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--sample-len", type=int, default=120, dest="sample_len")
    ap.add_argument("--log-every", type=int, default=100, dest="log_every")
    ap.add_argument("--eval-every", type=int, default=1000, dest="eval_every")
    ap.add_argument("--sample-every", type=int, default=1000, dest="sample_every")
    ap.add_argument("--ckpt-every", type=int, default=500, dest="ckpt_every")
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
