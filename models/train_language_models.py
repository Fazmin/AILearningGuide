"""Train and export the shared bigram, RNN, and 2-layer transformer assets.

Induction curriculum
--------------------
The transformer's training batches mix corpus windows with synthetic repeated strings.
The synthetic strings use a RANGE of repeat periods (``TRAINING_PERIODS``), random
and corpus-text content, and an optional random prefix, so there is no fixed offset to
memorise. The periods in ``induction_diagnostic.HELD_OUT_PERIODS`` are never trained on:
the shipping gate measures induction only on those. See ``induction_diagnostic.py``.

Validation
----------
The last 10% of the corpus is a held-out tail. Transformer training windows come only
from the first 90%; ``validation_loss`` is measured on the tail (the previous pipeline
measured random windows of the training corpus and had no held-out split).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import trange

import induction_diagnostic as induction


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    block_size: int = 64
    model_width: int = 256
    heads: int = 4
    layers: int = 2
    feed_forward_width: int = 768
    dropout: float = 0.0

    @property
    def head_width(self) -> int:
        return self.model_width // self.heads


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_width = config.head_width
        self.qkv = nn.Linear(config.model_width, config.model_width * 3)
        self.output = nn.Linear(config.model_width, config.model_width)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            .view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, sequence, width = inputs.shape
        qkv = self.qkv(inputs)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, sequence, self.heads, self.head_width).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :sequence, :sequence],
            torch.finfo(scores.dtype).min,
        )
        attention = F.softmax(scores, dim=-1)
        mixed = attention @ value
        mixed = mixed.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.dropout(self.output(mixed)), query, key, value, attention


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_width)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.model_width)
        self.mlp = nn.Sequential(
            nn.Linear(config.model_width, config.feed_forward_width),
            nn.GELU(),
            nn.Linear(config.feed_forward_width, config.model_width),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        update, query, key, value, attention = self.attention(
            self.attention_norm(inputs)
        )
        hidden = inputs + update
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden, query, key, value, attention


class TinyTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_width)
        self.position_embedding = nn.Embedding(config.block_size, config.model_width)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.layers)]
        )
        self.final_norm = nn.LayerNorm(config.model_width)
        self.lm_head = nn.Linear(config.model_width, config.vocab_size, bias=False)

    def forward(
        self, input_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        _, sequence = input_ids.shape
        positions = torch.arange(sequence, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        queries: list[Tensor] = []
        keys: list[Tensor] = []
        values: list[Tensor] = []
        patterns: list[Tensor] = []
        residuals: list[Tensor] = []

        for block in self.blocks:
            hidden, query, key, value, attention = block(hidden)
            queries.append(query)
            keys.append(key)
            values.append(value)
            patterns.append(attention)
            residuals.append(hidden)

        logits = self.lm_head(self.final_norm(hidden))
        return (
            logits,
            torch.stack(queries),
            torch.stack(keys),
            torch.stack(values),
            torch.stack(patterns),
            torch.stack(residuals),
        )


class CharacterRnn(nn.Module):
    def __init__(self, vocab_size: int, width: int = 192, layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.rnn = nn.GRU(width, width, layers, batch_first=True)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden, _ = self.rnn(self.embedding(input_ids))
        return self.output(hidden)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


TRAIN_FRACTION = 0.9
# Mix of synthetic content: random symbols (with replacement), random distinct symbols,
# and snippets of the training corpus itself.
SYNTHETIC_KINDS = ("random", "distinct", "corpus")
SYNTHETIC_WEIGHTS = (0.35, 0.25, 0.40)


def load_corpus(path: Path) -> tuple[str, list[str], dict[str, int], Tensor]:
    text = path.read_text(encoding="utf-8")
    characters = sorted(set(text))
    vocabulary = [*characters, "<unk>"]
    stoi = {character: index for index, character in enumerate(characters)}
    unknown = len(vocabulary) - 1
    encoded = torch.tensor(
        [stoi.get(character, unknown) for character in text], dtype=torch.long
    )
    return text, vocabulary, stoi, encoded


def split_corpus(encoded: Tensor) -> tuple[Tensor, Tensor]:
    """Contiguous 90/10 split: transformer training windows come from the head only."""
    boundary = int(len(encoded) * TRAIN_FRACTION)
    return encoded[:boundary], encoded[boundary:]


def _draw(kind: str, count: int, train_tokens: Tensor, vocab_size: int) -> Tensor:
    if kind == "corpus":
        start = int(torch.randint(0, len(train_tokens) - count, (1,)).item())
        return train_tokens[start : start + count]
    if kind == "distinct":
        return torch.randperm(vocab_size - 1)[:count]
    # <unk> (the last id) is never drawn.
    return torch.randint(0, vocab_size - 1, (count,))


def synthetic_window(train_tokens: Tensor, total: int, vocab_size: int) -> Tensor:
    """One repeated-string training window of ``total`` tokens.

    A random string of a random training period is repeated to fill the window, after an
    optional unrelated random prefix. Nothing about the layout is fixed, so a lookup of
    the form "attend N positions back" cannot solve it for more than one period.
    """
    period = random.choice(induction.TRAINING_PERIODS)
    kind = random.choices(SYNTHETIC_KINDS, weights=SYNTHETIC_WEIGHTS)[0]
    longest_prefix = total - 2 * period
    prefix = (
        random.randint(1, longest_prefix)
        if longest_prefix >= 1 and random.random() < 0.5
        else 0
    )
    segment = _draw(kind, period, train_tokens, vocab_size)
    body_length = total - prefix
    body = segment.repeat(math.ceil(body_length / period))[:body_length]
    if not prefix:
        return body
    prefix_kind = "corpus" if kind == "corpus" else "random"
    return torch.cat([_draw(prefix_kind, prefix, train_tokens, vocab_size), body])


def make_batch(
    encoded: Tensor,
    batch_size: int,
    block_size: int,
    vocab_size: int,
    device: torch.device,
    induction_fraction: float,
) -> tuple[Tensor, Tensor]:
    starts = torch.randint(0, len(encoded) - block_size - 1, (batch_size,))
    windows = torch.stack(
        [encoded[start : start + block_size + 1] for start in starts.tolist()]
    )

    induction_count = round(batch_size * induction_fraction)
    for row in range(induction_count):
        windows[row] = synthetic_window(encoded, block_size + 1, vocab_size)

    return windows[:, :-1].to(device), windows[:, 1:].to(device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    encoded: Tensor,
    batch_size: int,
    block_size: int,
    vocab_size: int,
    device: torch.device,
) -> float:
    """Random windows of ``encoded`` (the previous pipeline's validation protocol)."""
    model.eval()
    losses = []
    for _ in range(8):
        inputs, targets = make_batch(
            encoded,
            batch_size,
            block_size,
            vocab_size,
            device,
            induction_fraction=0,
        )
        output = model(inputs)
        logits = output[0] if isinstance(output, tuple) else output
        losses.append(
            F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1)).item()
        )
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def window_loss(
    model: nn.Module,
    tokens: Tensor,
    block_size: int,
    vocab_size: int,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Mean next-character loss (nats) over every non-overlapping window of ``tokens``.

    Deterministic: each character after the first of its window is scored once, with
    between 1 and ``block_size`` characters of context, as in training.
    """
    was_training = model.training
    model.eval()
    windows = (len(tokens) - 1) // block_size
    inputs = tokens[: windows * block_size].view(windows, block_size)
    targets = tokens[1 : windows * block_size + 1].view(windows, block_size)
    total = 0.0
    count = 0
    for start in range(0, windows, batch_size):
        chunk_in = inputs[start : start + batch_size].to(device)
        chunk_out = targets[start : start + batch_size].to(device)
        output = model(chunk_in)
        logits = output[0] if isinstance(output, tuple) else output
        total += F.cross_entropy(
            logits.reshape(-1, vocab_size), chunk_out.reshape(-1), reduction="sum"
        ).item()
        count += chunk_out.numel()
    if was_training:
        model.train()
    return total / count


def bigram_loss(train_tokens: Tensor, eval_tokens: Tensor, vocab_size: int) -> float:
    """Mean surprisal (nats) of an add-one bigram fitted on ``train_tokens``."""
    counts = torch.ones(vocab_size, vocab_size, dtype=torch.float64)
    for current, following in zip(train_tokens[:-1].tolist(), train_tokens[1:].tolist()):
        counts[current, following] += 1
    log_probabilities = (counts / counts.sum(dim=1, keepdim=True)).log()
    pairs = log_probabilities[eval_tokens[:-1], eval_tokens[1:]]
    return float(-pairs.mean().item())


def induction_report(
    model: TinyTransformer,
    heldout_tokens: Tensor,
    vocab_size: int,
    device: torch.device,
    *,
    sequences: int = induction.SEQUENCES_PER_CELL,
    include_reference: bool = True,
) -> dict[str, object]:
    return induction.run_diagnostic(
        induction.torch_forward(model, device),
        heldout_tokens.numpy(),
        vocab_size,
        sequences=sequences,
        include_reference=include_reference,
    )


def learning_rate_at(
    step: int, steps: int, peak: float, warmup: int, schedule: str
) -> float:
    if schedule == "constant":
        return peak
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train_model(
    model: nn.Module,
    encoded: Tensor,
    *,
    steps: int,
    batch_size: int,
    block_size: int,
    vocab_size: int,
    device: torch.device,
    learning_rate: float,
    induction_fraction: float,
    validate: Callable[[], float] | None = None,
    monitor: Callable[[], dict[str, float]] | None = None,
    schedule: str = "constant",
    warmup: int = 0,
    eval_every: int | None = None,
    label: str | None = None,
) -> list[dict[str, float]]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    history: list[dict[str, float]] = []
    progress = trange(steps, desc=label or model.__class__.__name__)
    interval = eval_every or max(1, steps // 10)

    for step in progress:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, steps, learning_rate, warmup, schedule)
        inputs, targets = make_batch(
            encoded,
            batch_size,
            block_size,
            vocab_size,
            device,
            induction_fraction,
        )
        output = model(inputs)
        logits = output[0] if isinstance(output, tuple) else output
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0 or (step + 1) % interval == 0 or step + 1 == steps:
            validation = (
                validate()
                if validate is not None
                else evaluate(model, encoded, batch_size, block_size, vocab_size, device)
            )
            record = {
                "step": float(step + 1),
                "train_loss": float(loss.item()),
                "validation_loss": validation,
            }
            if monitor is not None:
                record.update(monitor())
            history.append(record)
            extra = {
                key.replace("induction_", "ind_"): f"{value:.3f}"
                for key, value in record.items()
                if key.startswith("induction_")
            }
            progress.set_postfix(
                loss=f"{loss.item():.3f}", val=f"{validation:.3f}", **extra
            )
            if monitor is not None:
                print(json.dumps(record), file=sys.stderr, flush=True)

    return history


def write_bigram(encoded: Tensor, vocabulary: list[str], output: Path) -> None:
    counts = torch.ones(len(vocabulary), len(vocabulary), dtype=torch.float64)
    for current, following in zip(encoded[:-1], encoded[1:]):
        counts[int(current), int(following)] += 1
    probabilities = counts / counts.sum(dim=1, keepdim=True)
    payload = {
        "vocabulary": vocabulary,
        "probabilities": probabilities.tolist(),
    }
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def export_transformer(model: TinyTransformer, output: Path) -> None:
    model = copy.deepcopy(model).cpu().eval()
    example = torch.zeros(1, 24, dtype=torch.long)
    torch.onnx.export(
        model,
        (example,),
        output,
        input_names=["input_ids"],
        output_names=["logits", "q", "k", "v", "attention", "residual"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
            "q": {1: "batch", 3: "sequence"},
            "k": {1: "batch", 3: "sequence"},
            "v": {1: "batch", 3: "sequence"},
            "attention": {1: "batch", 3: "query", 4: "key"},
            "residual": {1: "batch", 2: "sequence"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )


def export_rnn(model: CharacterRnn, output: Path) -> None:
    model = copy.deepcopy(model).cpu().eval()
    example = torch.zeros(1, 24, dtype=torch.long)
    torch.onnx.export(
        model,
        (example,),
        output,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shipping_gate(metadata: dict) -> dict[str, object]:
    """Every shipping check that can be decided from the metadata alone.

    (ONNX parity against the PyTorch checkpoint is checked by ``verify_exports.py``.)
    """
    held_out = metadata["evaluation"]["heldout_tail"]
    parameters = metadata["transformer"]["parameters"]
    checks = {
        "parameters_between_1m_and_2m": 1_000_000 <= parameters <= 2_000_000,
        "heldout_loss_is_finite": math.isfinite(held_out["transformer"]),
        "heldout_loss_beats_bigram": held_out["transformer"]
        < held_out["bigram_fitted_on_training_split"],
        "induction_gate": bool(
            metadata["transformer"]["induction_diagnostic"]["passes_shipping_threshold"]
        ),
    }
    return {"checks": checks, "passes": all(checks.values())}


def publish_language_assets(
    exports: Path,
    transformer_publish: Path,
    prediction_publish: Path,
    *,
    include_rnn: bool,
) -> None:
    """Copy the reviewed language assets into the modules that ship them."""
    transformer_publish.mkdir(parents=True, exist_ok=True)
    prediction_publish.mkdir(parents=True, exist_ok=True)
    for name in ["transformer-vocab.json", "tiny-transformer.onnx"]:
        (transformer_publish / name).write_bytes((exports / name).read_bytes())
    metadata_bytes = (exports / "language-models.metadata.json").read_bytes()
    (transformer_publish / "tiny-transformer.metadata.json").write_bytes(metadata_bytes)
    (prediction_publish / "language-models.metadata.json").write_bytes(metadata_bytes)
    for name in ["transformer-vocab.json", "bigram.json"]:
        (prediction_publish / name).write_bytes((exports / name).read_bytes())
    if include_rnn:
        (prediction_publish / "character-rnn.onnx").write_bytes(
            (exports / "character-rnn.onnx").read_bytes()
        )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=Path, default=root / "data" / "tiny-shakespeare.txt"
    )
    parser.add_argument("--output", type=Path, default=root / "exports")
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "checkpoints")
    parser.add_argument(
        "--publish",
        type=Path,
        default=root.parent / "src" / "modules" / "attention" / "assets",
    )
    parser.add_argument(
        "--publish-prediction",
        type=Path,
        default=root.parent
        / "src"
        / "modules"
        / "next-token-prediction"
        / "assets",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Write checkpoints and exports only; do not copy into src/modules.",
    )
    parser.add_argument(
        "--promote",
        type=Path,
        help=(
            "Skip training: copy the checkpoint and exports of an earlier --no-publish "
            "run (a directory holding checkpoints/ and exports/) into place and publish."
        ),
    )
    parser.add_argument(
        "--allow-failing-gate",
        action="store_true",
        help="Publish even if the induction gate failed (the metadata records the failure).",
    )
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--induction-fraction", type=float, default=0.30)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument(
        "--rnn",
        choices=["reuse", "retrain"],
        default="reuse",
        help=(
            "reuse: keep the shipped GRU (its checkpoint is re-scored, its ONNX file and "
            "recorded training history are left untouched). retrain: train and export it."
        ),
    )
    parser.add_argument("--rnn-steps", type=int, default=240)
    parser.add_argument("--rnn-batch-size", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.promote is not None:
        promote(args, root)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using {device}")

    if not args.corpus.exists():
        raise SystemExit(
            f"Missing {args.corpus}. Run `python models/download_corpus.py` first."
        )
    text, vocabulary, stoi, encoded = load_corpus(args.corpus)
    unknown = len(vocabulary) - 1
    train_tokens, heldout_tokens = split_corpus(encoded)
    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vocabulary_payload = {
        "format": 1,
        "kind": "character",
        "stoi": stoi,
        "itos": vocabulary,
        "unk": unknown,
        "corpus": "Tiny Shakespeare",
    }
    vocabulary_json = json.dumps(vocabulary_payload, ensure_ascii=False, indent=2) + "\n"
    (args.output / "transformer-vocab.json").write_text(
        vocabulary_json, encoding="utf-8"
    )
    bigram_path = args.output / "bigram.json"
    write_bigram(encoded, vocabulary, bigram_path)

    config = TransformerConfig(vocab_size=len(vocabulary), dropout=args.dropout)
    transformer = TinyTransformer(config)
    if not 1_000_000 <= parameter_count(transformer) <= 2_000_000:
        raise RuntimeError(
            f"Transformer has {parameter_count(transformer):,} parameters; expected 1–2M."
        )

    def validate() -> float:
        return window_loss(
            transformer, heldout_tokens, config.block_size, config.vocab_size, device
        )

    def monitor() -> dict[str, float]:
        # Logged for the training curve only; the gate is judged once, on the final model.
        report = induction_report(
            transformer, heldout_tokens, config.vocab_size, device,
            sequences=32, include_reference=False,
        )
        random_cells = report["held_out"]["random"]  # type: ignore[index]
        return {
            "induction_best_stripe": float(report["best_score"]),  # type: ignore[arg-type]
            "induction_repeat_loss": float(
                np.mean([cell["loss_second_half"] for cell in random_cells.values()])
            ),
        }

    transformer_history = train_model(
        transformer,
        train_tokens,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        device=device,
        learning_rate=args.learning_rate,
        induction_fraction=args.induction_fraction,
        validate=validate,
        monitor=monitor,
        schedule=args.schedule,
        warmup=args.warmup,
        eval_every=args.eval_every,
    )
    diagnostic = induction_report(transformer, heldout_tokens, config.vocab_size, device)
    transformer_path = args.output / "tiny-transformer.onnx"
    torch.save(
        {"config": asdict(config), "state_dict": transformer.state_dict()},
        args.checkpoint_dir / "tiny-transformer.pt",
    )
    export_transformer(transformer, transformer_path)

    # ---- comparison models --------------------------------------------------------
    rnn = CharacterRnn(len(vocabulary))
    previous_metadata = next(
        (
            candidate
            for candidate in (
                args.output / "language-models.metadata.json",
                root / "exports" / "language-models.metadata.json",
            )
            if candidate.exists()
        ),
        None,
    )
    rnn_checkpoint = args.checkpoint_dir / "character-rnn.pt"
    if args.rnn == "reuse":
        reuse_from = rnn_checkpoint
        if not reuse_from.exists():
            reuse_from = root / "checkpoints" / "character-rnn.pt"
        if not reuse_from.exists() or previous_metadata is None:
            raise SystemExit(
                "--rnn reuse needs checkpoints/character-rnn.pt and the previous "
                "exports/language-models.metadata.json; pass --rnn retrain instead."
            )
        rnn.load_state_dict(torch.load(reuse_from, map_location="cpu", weights_only=True))
        rnn_block = json.loads(previous_metadata.read_text(encoding="utf-8"))["rnn"]
        rnn_block["status"] = (
            "reused: trained once by an earlier run of this pipeline; its ONNX file and "
            "history are unchanged (the history's validation_loss is on random windows "
            "of the whole corpus, not a held-out tail)"
        )
        if reuse_from != rnn_checkpoint:
            shutil.copyfile(reuse_from, rnn_checkpoint)
        rnn.to(device)
    else:
        torch.manual_seed(args.seed + 1)
        rnn_history = train_model(
            rnn,
            encoded,
            steps=args.rnn_steps,
            batch_size=args.rnn_batch_size,
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            device=device,
            learning_rate=5e-4,
            induction_fraction=0,
            label="CharacterRnn",
        )
        torch.save(rnn.state_dict(), rnn_checkpoint)
        export_rnn(rnn, args.output / "character-rnn.onnx")
        rnn_block = {
            "parameters": parameter_count(rnn),
            "history": rnn_history,
            "status": "trained by this run",
        }
    rnn_block["parameters"] = parameter_count(rnn)

    block, vocab = config.block_size, config.vocab_size
    evaluation = {
        "units": "nats per character; multiply by 1.4427 for bits",
        "heldout_tail": {
            "characters": len(heldout_tokens),
            "transformer": window_loss(transformer, heldout_tokens, block, vocab, device),
            "rnn": window_loss(rnn, heldout_tokens, block, vocab, device),
            "bigram_fitted_on_training_split": bigram_loss(
                train_tokens, heldout_tokens, vocab
            ),
        },
        "whole_corpus": {
            "characters": len(encoded),
            "transformer": window_loss(transformer, encoded, block, vocab, device),
            "rnn": window_loss(rnn, encoded, block, vocab, device),
            "bigram_fitted_on_whole_corpus": bigram_loss(encoded, encoded, vocab),
        },
        "note": (
            "Windows are consecutive and non-overlapping, 64 characters each. The "
            "transformer trained only on the first 90% of the corpus, so heldout_tail is a "
            "true held-out figure for it. The GRU trained on random windows of the whole "
            "corpus for 240 steps (about a third of an epoch), so its tail figure is "
            "nearly, but not strictly, held out."
        ),
    }

    metadata = {
        "seed": args.seed,
        "device": str(device),
        "corpus_characters": len(text),
        "vocabulary_size": len(vocabulary),
        "split": {
            "train_fraction": TRAIN_FRACTION,
            "train_characters": len(train_tokens),
            "heldout_characters": len(heldout_tokens),
        },
        "evaluation": evaluation,
        "transformer": {
            "config": asdict(config),
            "parameters": parameter_count(transformer),
            "training": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "windows_per_batch": {
                    "corpus": args.batch_size - round(args.batch_size * args.induction_fraction),
                    "synthetic": round(args.batch_size * args.induction_fraction),
                },
                "learning_rate": args.learning_rate,
                "schedule": args.schedule,
                "warmup_steps": args.warmup,
                "synthetic_curriculum": {
                    "training_periods": list(induction.TRAINING_PERIODS),
                    "held_out_periods": list(induction.HELD_OUT_PERIODS),
                    "content_kinds": dict(zip(SYNTHETIC_KINDS, SYNTHETIC_WEIGHTS)),
                    "random_prefix_probability": 0.5,
                },
            },
            "history": transformer_history,
            "induction_diagnostic": diagnostic,
        },
        "rnn": rnn_block,
    }
    metadata["shipping_gate"] = shipping_gate(metadata)
    metadata_json = json.dumps(metadata, indent=2) + "\n"
    (args.output / "language-models.metadata.json").write_text(
        metadata_json, encoding="utf-8"
    )
    print(induction.summarize(diagnostic))
    print(json.dumps(evaluation, indent=2))
    print(json.dumps(metadata["shipping_gate"], indent=2))

    gate_ok = bool(metadata["shipping_gate"]["passes"])
    if args.no_publish:
        print(f"Not publishing (--no-publish). Exports are in {args.output}.")
    elif gate_ok or args.allow_failing_gate:
        publish_language_assets(
            args.output,
            args.publish,
            args.publish_prediction,
            include_rnn=args.rnn == "retrain",
        )
    if not gate_ok:
        raise SystemExit(
            "The model exported, but it did not pass the shipping gate (see "
            "shipping_gate in the metadata). Do not lower the thresholds; change the "
            "curriculum or --steps."
            + (" Published anyway (--allow-failing-gate)." if args.allow_failing_gate and not args.no_publish else "")
        )


def promote(args: argparse.Namespace, root: Path) -> None:
    """Install the checkpoint and exports of an earlier ``--no-publish`` run, then publish."""
    source_exports = args.promote / "exports"
    source_checkpoints = args.promote / "checkpoints"
    metadata = json.loads(
        (source_exports / "language-models.metadata.json").read_text(encoding="utf-8")
    )
    metadata.setdefault("shipping_gate", shipping_gate(metadata))
    if not metadata["shipping_gate"]["passes"] and not args.allow_failing_gate:
        raise SystemExit(
            "That run did not pass the shipping gate; pass --allow-failing-gate to "
            "publish it with its recorded failure."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in ["transformer-vocab.json", "bigram.json", "tiny-transformer.onnx"]:
        shutil.copyfile(source_exports / name, args.output / name)
    (args.output / "language-models.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(
        source_checkpoints / "tiny-transformer.pt",
        args.checkpoint_dir / "tiny-transformer.pt",
    )
    include_rnn = (source_exports / "character-rnn.onnx").exists()
    if include_rnn:
        shutil.copyfile(
            source_exports / "character-rnn.onnx", args.output / "character-rnn.onnx"
        )
        shutil.copyfile(
            source_checkpoints / "character-rnn.pt",
            args.checkpoint_dir / "character-rnn.pt",
        )
    publish_language_assets(
        args.output, args.publish, args.publish_prediction, include_rnn=include_rnn
    )
    print(f"Promoted {args.promote} (transformer sha256 {sha256_of(args.output / 'tiny-transformer.onnx')}).")


if __name__ == "__main__":
    main()
