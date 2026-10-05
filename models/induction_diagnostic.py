"""The induction diagnostic and shipping gate for the character transformer.

Why this exists
---------------
The first version of this check trained on strings that repeat every 8 characters and
tested strings that repeat every 16. A model that has only memorised "look 7 positions
back" passes that test (16 is a multiple of 8), so the gate could never fail. This
version tests the standard way: on repeat periods the model was never trained on.

What it measures
----------------
Take a random string of length ``P`` and repeat it once, so the sequence is
``s[0..P-1] s[0..P-1]``. A genuine induction head, reading the token at position ``q``
in the second copy, finds the earlier copy of the same token (position ``q - P``) and
attends to the token *after* it (position ``q - P + 1``). Plotted as a ``q`` by ``k``
attention map this is a stripe at offset ``P - 1`` below the diagonal. A fixed look-back
head (always attend 7 positions back) draws a stripe at offset 7 instead, so its score
at offset ``P - 1`` is near zero for every period except 8. The second measurement is
behavioural: the model's loss on the repeated half against the first half, the
"in-context" gain that only copying can produce on a random string.

The rules below are fixed in advance and are not tuned to any particular checkpoint.
``check_induction_gate.py`` asserts that the gate fails the earlier shipped model, passes
an idealised induction pattern, and fails an idealised fixed 7-back pattern.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Periods the model is never trained on. They are coprime with the old period of 8 and
# with each other. Training periods are drawn from TRAINING_PERIODS, which excludes them.
HELD_OUT_PERIODS: tuple[int, ...] = (11, 13, 17, 19)
# Periods inside the training range, scored for reference only (never gating). A model
# that passes these but fails the held-out periods has memorised offsets, not induction.
REFERENCE_PERIODS: tuple[int, ...] = (8, 10, 12, 16)
TRAINING_PERIODS: tuple[int, ...] = tuple(
    period for period in range(4, 32) if period not in HELD_OUT_PERIODS
)

# Shipping gate. ``random`` means random strings of distinct symbols; ``corpus`` means
# real Shakespeare snippets taken from the held-out tail of the corpus.
GATE = {
    # A head qualifies as an induction head when, on EVERY held-out period, its mean
    # attention to the successor of the earlier match is at least this high.
    "random_stripe_min": 0.40,
    "corpus_stripe_min": 0.25,
    # Loss over the repeated half (nats per character), on every held-out period.
    "second_half_loss_max": 1.0,
    # First-half loss minus second-half loss on every held-out period.
    "random_loss_drop_min": 2.0,
    "corpus_loss_drop_min": 1.0,
}

SEQUENCES_PER_CELL = 96
DIAGNOSTIC_SEED = 942

# forward(ids [batch, sequence] int64) -> (logits [batch, sequence, vocab],
#                                          attention [layers, batch, heads, query, key])
Forward = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def repeated_sequences(
    kind: str,
    period: int,
    count: int,
    corpus_tokens: np.ndarray,
    vocab_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``count`` sequences of length ``2 * period``: one string, repeated once."""
    if kind == "random":
        # Distinct symbols, so each token has exactly one earlier occurrence and the
        # induction target is unambiguous. <unk> (the last id) is never drawn.
        base = np.stack(
            [rng.permutation(vocab_size - 1)[:period] for _ in range(count)]
        )
    elif kind == "corpus":
        starts = rng.integers(0, len(corpus_tokens) - period, size=count)
        base = np.stack([corpus_tokens[start : start + period] for start in starts])
    else:
        raise ValueError(kind)
    return np.concatenate([base, base], axis=1).astype(np.int64)


def measure_cell(
    forward: Forward,
    ids: np.ndarray,
    period: int,
) -> dict[str, object]:
    """Per-head stripe scores and first/second half loss for one batch of repeats."""
    logits, attention = forward(ids)
    layers, _, heads, sequence, _ = attention.shape
    assert sequence == 2 * period
    queries = np.arange(period, 2 * period)

    def stripe(offset_from_query: int) -> list[list[float]]:
        keys = queries + offset_from_query
        usable = keys >= 0  # a look-back longer than the query's position has no key
        if not usable.any():
            return np.zeros((layers, heads)).tolist()
        # attention[:, :, :, queries, keys] -> [layers, batch, heads, queries]
        picked = attention[:, :, :, queries[usable], keys[usable]]
        return picked.mean(axis=(1, 3)).tolist()

    log_probabilities = _log_softmax(logits.astype(np.float64))
    nll = -np.take_along_axis(
        log_probabilities[:, :-1], ids[:, 1:, None], axis=-1
    )[..., 0]
    # nll[:, t] is the loss for predicting token t + 1 from position t.
    first_half = float(nll[:, 0 : period - 1].mean())
    second_half = float(nll[:, period : 2 * period - 1].mean())
    return {
        "induction": stripe(-period + 1),
        "previous_token": stripe(-1),
        "lookback_7": stripe(-7),
        "loss_first_half": first_half,
        "loss_second_half": second_half,
        "loss_drop": first_half - second_half,
        "layers": layers,
        "heads": heads,
    }


def run_diagnostic(
    forward: Forward,
    corpus_tokens: np.ndarray,
    vocab_size: int,
    *,
    sequences: int = SEQUENCES_PER_CELL,
    seed: int = DIAGNOSTIC_SEED,
    include_reference: bool = True,
) -> dict[str, object]:
    """Run the full diagnostic and return a JSON-serialisable report with the gate verdict.

    ``corpus_tokens`` should come from text the model was not trained on.
    """
    assert not set(HELD_OUT_PERIODS) & set(TRAINING_PERIODS)
    rng = np.random.default_rng(seed)
    held_out: dict[str, dict[str, dict[str, object]]] = {"random": {}, "corpus": {}}
    reference: dict[str, dict[str, dict[str, object]]] = {"random": {}, "corpus": {}}
    for kind in ("random", "corpus"):
        for period in HELD_OUT_PERIODS:
            ids = repeated_sequences(kind, period, sequences, corpus_tokens, vocab_size, rng)
            held_out[kind][str(period)] = measure_cell(forward, ids, period)
        if include_reference:
            for period in REFERENCE_PERIODS:
                ids = repeated_sequences(
                    kind, period, sequences, corpus_tokens, vocab_size, rng
                )
                reference[kind][str(period)] = measure_cell(forward, ids, period)

    layers = int(held_out["random"][str(HELD_OUT_PERIODS[0])]["layers"])  # type: ignore[arg-type]
    heads = int(held_out["random"][str(HELD_OUT_PERIODS[0])]["heads"])  # type: ignore[arg-type]

    def worst_case(kind: str) -> np.ndarray:
        stacked = np.array(
            [held_out[kind][str(p)]["induction"] for p in HELD_OUT_PERIODS]
        )  # [periods, layers, heads]
        return stacked.min(axis=0)

    random_min = worst_case("random")
    corpus_min = worst_case("corpus")
    qualifies = (random_min >= GATE["random_stripe_min"]) & (
        corpus_min >= GATE["corpus_stripe_min"]
    )

    failures: list[str] = []
    if not qualifies.any():
        best = np.unravel_index(int(random_min.argmax()), random_min.shape)
        failures.append(
            "no head reaches the stripe thresholds on every held-out period "
            f"(best random-string worst case {random_min[best]:.3f} at layer "
            f"{best[0] + 1} head {best[1] + 1}; needs {GATE['random_stripe_min']} "
            f"on random strings and {GATE['corpus_stripe_min']} on corpus text)"
        )
    for kind, drop_key in (("random", "random_loss_drop_min"), ("corpus", "corpus_loss_drop_min")):
        for period in HELD_OUT_PERIODS:
            cell = held_out[kind][str(period)]
            if cell["loss_second_half"] > GATE["second_half_loss_max"]:  # type: ignore[operator]
                failures.append(
                    f"{kind} period {period}: repeated-half loss "
                    f"{cell['loss_second_half']:.2f} nats exceeds {GATE['second_half_loss_max']}"
                )
            if cell["loss_drop"] < GATE[drop_key]:  # type: ignore[operator]
                failures.append(
                    f"{kind} period {period}: in-context loss drop "
                    f"{cell['loss_drop']:.2f} nats is below {GATE[drop_key]}"
                )

    best_flat = int(random_min.argmax())
    best_layer, best_head = divmod(best_flat, heads)
    return {
        "metric": (
            "worst-case (over held-out periods) mean attention from each token in the "
            "second copy of a repeated string to the token after its earlier occurrence "
            "(offset period - 1); plus first-half versus repeated-half loss"
        ),
        "version": 2,
        "held_out_periods": list(HELD_OUT_PERIODS),
        "reference_periods": list(REFERENCE_PERIODS) if include_reference else [],
        "training_periods": list(TRAINING_PERIODS),
        "sequences_per_cell": sequences,
        "seed": seed,
        "gate": GATE,
        # [layer][head]: worst held-out period, random strings. Same shape as the
        # earlier diagnostic's "scores".
        "scores": random_min.tolist(),
        "corpus_scores": corpus_min.tolist(),
        "head_qualifies": qualifies.tolist(),
        "best_layer": best_layer,
        "best_head": best_head,
        "best_score": float(random_min[best_layer, best_head]),
        "held_out": held_out,
        "reference": reference if include_reference else {},
        "layers": layers,
        "heads": heads,
        "failures": failures,
        "passes_shipping_threshold": not failures,
    }


def torch_forward(model: "torch.nn.Module", device: "torch.device") -> Forward:  # noqa: F821
    """Adapt a TinyTransformer to the numpy forward signature."""
    import torch

    @torch.no_grad()
    def forward(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        was_training = model.training
        model.eval()
        outputs = model(torch.from_numpy(ids).to(device))
        if was_training:
            model.train()
        return outputs[0].float().cpu().numpy(), outputs[4].float().cpu().numpy()

    return forward


def onnx_forward(path: str) -> Forward:
    """Adapt a shipped tiny-transformer.onnx file to the numpy forward signature."""
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def forward(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, _q, _k, _v, attention, _residual = session.run(None, {"input_ids": ids})
        return np.asarray(logits), np.asarray(attention)

    return forward


def summarize(report: dict[str, object]) -> str:
    """A short human-readable verdict."""
    lines = []
    for kind in ("random", "corpus"):
        for period, cell in report["held_out"][kind].items():  # type: ignore[index]
            best = max(max(row) for row in cell["induction"])
            lines.append(
                f"{kind:6s} period {int(period):2d}: best head stripe {best:.3f}, "
                f"loss {cell['loss_first_half']:.2f} -> {cell['loss_second_half']:.2f} nats"
            )
    lines.append(
        f"gate: {'PASS' if report['passes_shipping_threshold'] else 'FAIL'}"
    )
    lines.extend(f"  - {reason}" for reason in report["failures"])  # type: ignore[union-attr]
    return "\n".join(lines)
