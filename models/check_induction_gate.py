"""Prove that the induction gate can fail, and that it can pass.

Three controls, none of which is a trained model of the shipping architecture:

1. An idealised induction model (attention on the successor of every earlier match,
   confident copy prediction) must PASS.
2. An idealised fixed "attend 7 back" model, the habit the first shipped transformer
   learned from period-8 training strings, must FAIL.
3. The first shipped transformer itself (kept locally in
   checkpoints/legacy-period8/, identified by SHA-256) must FAIL. It passed the old
   diagnostic with a best score of 0.39.

Run: models/.venv/bin/python models/check_induction_gate.py [--candidate PATH.onnx]
Exits non-zero if any control behaves differently than stated, or if --candidate is
given and does not pass the gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import induction_diagnostic as induction
from train_language_models import load_corpus, split_corpus

LEGACY_ONNX_SHA256 = "0029cae9f83f0c1364974188b7749a1169774ca975e09732c839971ac85596b1"
VOCAB = 66


def ideal_induction(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, length = ids.shape
    attention = np.zeros((2, batch, 4, length, length))
    logits = np.zeros((batch, length, VOCAB))
    for b in range(batch):
        for q in range(length):
            matches = [j for j in range(q) if ids[b, j] == ids[b, q]]
            if not matches:
                attention[:, b, :, q, q] = 1.0
                continue
            for j in matches:
                attention[:, b, :, q, j + 1] += 1.0 / len(matches)
            probabilities = np.full(VOCAB, 0.05 / VOCAB)
            for j in matches:
                probabilities[ids[b, j + 1]] += 0.95 / len(matches)
            logits[b, q] = np.log(probabilities)
    return logits, attention


def fixed_lookback(ids: np.ndarray, offset: int = 7) -> tuple[np.ndarray, np.ndarray]:
    batch, length = ids.shape
    attention = np.zeros((2, batch, 4, length, length))
    logits = np.zeros((batch, length, VOCAB))
    for b in range(batch):
        for q in range(length):
            if q >= offset:
                attention[:, b, :, q, q - offset] = 1.0
                probabilities = np.full(VOCAB, 0.05 / VOCAB)
                probabilities[ids[b, q - offset]] += 0.95
                logits[b, q] = np.log(probabilities)
            else:
                attention[:, b, :, q, q] = 1.0
    return logits, attention


def verdict(name: str, report: dict, expect_pass: bool) -> bool:
    ok = report["passes_shipping_threshold"] == expect_pass
    print(
        f"[{'ok' if ok else 'UNEXPECTED'}] {name}: gate "
        f"{'passes' if report['passes_shipping_threshold'] else 'fails'} "
        f"(expected to {'pass' if expect_pass else 'fail'})"
    )
    return ok


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=root / "data" / "tiny-shakespeare.txt")
    parser.add_argument(
        "--legacy",
        type=Path,
        default=root / "checkpoints" / "legacy-period8" / "tiny-transformer.onnx",
    )
    parser.add_argument("--candidate", type=Path, help="A tiny-transformer.onnx that must pass.")
    parser.add_argument("--write", type=Path, help="Write the legacy and candidate reports here.")
    args = parser.parse_args()

    _, _, _, encoded = load_corpus(args.corpus)
    _, heldout = split_corpus(encoded)
    corpus = heldout.numpy()
    ok = True
    reports: dict[str, object] = {}

    ideal = induction.run_diagnostic(ideal_induction, corpus, VOCAB, sequences=8)
    ok &= verdict("idealised induction model", ideal, True)
    ok &= verdict(
        "idealised fixed 7-back model",
        induction.run_diagnostic(fixed_lookback, corpus, VOCAB, sequences=8),
        False,
    )

    if args.legacy.exists():
        import hashlib

        digest = hashlib.sha256(args.legacy.read_bytes()).hexdigest()
        if digest != LEGACY_ONNX_SHA256:
            raise SystemExit(f"{args.legacy} is not the first shipped transformer ({digest}).")
        legacy = induction.run_diagnostic(induction.onnx_forward(str(args.legacy)), corpus, VOCAB)
        reports["legacy"] = legacy
        ok &= verdict("first shipped transformer (period-8 curriculum)", legacy, False)
        print(induction.summarize(legacy))
        for kind in ("random",):
            for group in ("reference", "held_out"):
                for period, cell in legacy[group][kind].items():
                    best = max(max(row) for row in cell["induction"])
                    lookback = max(max(row) for row in cell["lookback_7"])
                    print(
                        f"    {group:9s} period {int(period):2d}: best induction stripe "
                        f"{best:.3f}, best 7-back {lookback:.3f}"
                    )
    else:
        print(f"[skipped] legacy transformer not found at {args.legacy}")

    if args.candidate is not None:
        candidate = induction.run_diagnostic(
            induction.onnx_forward(str(args.candidate)), corpus, VOCAB
        )
        reports["candidate"] = candidate
        ok &= verdict(f"candidate {args.candidate.name}", candidate, True)
        print(induction.summarize(candidate))

    if args.write:
        args.write.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
