"""Cache attention patterns and activation-patching baselines for canned prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from train_language_models import TinyTransformer, TransformerConfig, choose_device


# Capital letters for the copy pair. A period-13 string is a length the transformer was
# never trained on (see induction_diagnostic.HELD_OUT_PERIODS), so copying it is induction,
# not a memorised offset.
COPY_LETTERS = "QXZRKWMPJVBHD"
COPY_PERIOD = len(COPY_LETTERS)
COPY_LENGTH = 24
COPY_FOIL = "Y"

CANNED_PROMPTS = [
    {
        "id": "induction-letters",
        "text": " A B C D E F A B C D E",
        "description": "A repeated sequence used to expose induction-like routing.",
    },
    {
        "id": "induction-random",
        "text": (COPY_LETTERS * 2)[:COPY_LENGTH],
        "description": (
            "Thirteen random capitals repeated; the repeat length was never trained on, "
            "so the stripe at offset 12 is induction, not a memorised look-back."
        ),
    },
    {
        "id": "pronoun",
        "text": "The animal crossed because it was tired.",
        "description": "A short dependency with a pronoun.",
    },
    {
        "id": "line-pattern",
        "text": "ROMEO: hello\nJULIET: hello\nROMEO:",
        "description": "Repeated speaker and line structure from the training domain.",
    },
]


def copy_pair(
    letters: str = COPY_LETTERS,
    length: int = COPY_LENGTH,
    foil: str = COPY_FOIL,
) -> dict[str, str]:
    """A repeated letter string and the same string with the copied letter's source changed.

    The clean run can predict the next letter by induction: the last token's earlier
    occurrence is followed by ``target``. In the corrupt run that follower (the source) is
    replaced by ``foil``, so the same circuit has nothing correct to copy.
    """
    period = len(letters)
    clean = (letters * 2)[:length]
    source = length - period
    return {
        "id": "copy",
        "clean": clean,
        "corrupt": clean[:source] + foil + clean[source + 1 :],
        "target": letters[length % period],
    }


PATCHING_PAIRS = [
    {
        "id": "speaker",
        "clean": "ROMEO: hello\nJULIET: hello\nROMEO:",
        "corrupt": "ROMEO: hello\nJULIET: hello\nJULIET:",
        "target": " ",
    },
    copy_pair(),
]


def load_model(checkpoint_path: Path, device: torch.device) -> TinyTransformer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TransformerConfig(**checkpoint["config"])
    model = TinyTransformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def encode(text: str, stoi: dict[str, int], unknown: int, limit: int) -> Tensor:
    values = [stoi.get(character, unknown) for character in text[:limit]]
    return torch.tensor(values, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def forward_with_patch(
    model: TinyTransformer,
    input_ids: Tensor,
    *,
    patch_layer: int | None = None,
    patch_position: int | None = None,
    source: Tensor | None = None,
) -> Tensor:
    _, sequence = input_ids.shape
    positions = torch.arange(sequence, device=input_ids.device)
    hidden = model.token_embedding(input_ids) + model.position_embedding(positions)
    for layer, block in enumerate(model.blocks):
        hidden, *_ = block(hidden)
        if (
            layer == patch_layer
            and patch_position is not None
            and source is not None
        ):
            hidden = hidden.clone()
            hidden[:, patch_position, :] = source[:, patch_position, :]
    return model.lm_head(model.final_norm(hidden))


@torch.no_grad()
def attention_cache(
    model: TinyTransformer,
    prompts: list[dict[str, str]],
    stoi: dict[str, int],
    unknown: int,
    device: torch.device,
) -> list[dict[str, object]]:
    records = []
    for prompt in prompts:
        input_ids = encode(
            prompt["text"], stoi, unknown, model.config.block_size
        ).to(device)
        *_, attention, _ = model(input_ids)
        records.append(
            {
                **prompt,
                "tokens": list(prompt["text"][: model.config.block_size]),
                "shape": list(attention.shape),
                "attention": attention.squeeze(1).cpu().numpy().round(5).tolist(),
            }
        )
    return records


@torch.no_grad()
def patching_cache(
    model: TinyTransformer,
    pairs: list[dict[str, str]],
    stoi: dict[str, int],
    unknown: int,
    device: torch.device,
) -> list[dict[str, object]]:
    records = []
    for pair in pairs:
        length = min(
            len(pair["clean"]),
            len(pair["corrupt"]),
            model.config.block_size,
        )
        clean_ids = encode(pair["clean"][:length], stoi, unknown, length).to(device)
        corrupt_ids = encode(pair["corrupt"][:length], stoi, unknown, length).to(device)
        target = stoi.get(pair["target"], unknown)
        *_, clean_residuals = model(clean_ids)
        clean_logits = forward_with_patch(model, clean_ids)
        corrupt_logits = forward_with_patch(model, corrupt_ids)
        clean_score = float(clean_logits[0, -1, target].item())
        corrupt_score = float(corrupt_logits[0, -1, target].item())
        denominator = clean_score - corrupt_score
        effects = []

        for layer in range(model.config.layers):
            row = []
            for position in range(length):
                patched = forward_with_patch(
                    model,
                    corrupt_ids,
                    patch_layer=layer,
                    patch_position=position,
                    source=clean_residuals[layer],
                )
                score = float(patched[0, -1, target].item())
                recovery = (
                    (score - corrupt_score) / denominator
                    if abs(denominator) > 1e-8
                    else 0.0
                )
                row.append(round(recovery, 5))
            effects.append(row)

        gap = clean_score - corrupt_score
        record = {
            **pair,
            "tokens": list(pair["clean"][:length]),
            "clean_logit": clean_score,
            "corrupt_logit": corrupt_score,
            "logit_gap": gap,
            "metric": "logit of the target character at the final position",
            "recovery_by_layer_and_position": effects,
        }
        if pair["id"] == "copy":
            record["robustness"] = copy_pair_robustness(model, stoi, unknown, device)
        records.append(record)
    return records


@torch.no_grad()
def copy_pair_robustness(
    model: TinyTransformer,
    stoi: dict[str, int],
    unknown: int,
    device: torch.device,
    trials: int = 64,
    seed: int = 7,
) -> dict[str, object]:
    """The copy pair's logit gap over many random period-13 capital strings.

    Shows the cached gap is typical of the pair's construction, not a lucky string.
    """
    rng = np.random.default_rng(seed)
    capitals = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    gaps = []
    for _ in range(trials):
        order = list(rng.permutation(capitals))
        letters = "".join(order[:COPY_PERIOD])
        foil = order[COPY_PERIOD]  # a capital that is not in the string
        pair = copy_pair(letters, COPY_LENGTH, foil)
        target = stoi.get(pair["target"], unknown)
        scores = []
        for text in (pair["clean"], pair["corrupt"]):
            ids = encode(text, stoi, unknown, COPY_LENGTH).to(device)
            scores.append(float(forward_with_patch(model, ids)[0, -1, target].item()))
        gaps.append(scores[0] - scores[1])
    return {
        "trials": trials,
        "mean_gap": float(np.mean(gaps)),
        "min_gap": float(np.min(gaps)),
        "fraction_above_one_logit": float(np.mean(np.array(gaps) > 1.0)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "checkpoints" / "tiny-transformer.pt",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=root / "exports" / "transformer-vocab.json",
    )
    parser.add_argument("--output", type=Path, default=root / "exports")
    parser.add_argument(
        "--publish",
        type=Path,
        default=root.parent
        / "src"
        / "modules"
        / "interpretability-circuits"
        / "assets",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    stoi = vocabulary["stoi"]
    unknown = vocabulary["unk"]
    payload = {
        "format": 1,
        "model": "tiny-transformer",
        "attention": attention_cache(
            model, CANNED_PROMPTS, stoi, unknown, device
        ),
        "activation_patching": patching_cache(
            model, PATCHING_PAIRS, stoi, unknown, device
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "interpretability-cache.json"
    output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    for record in payload["activation_patching"]:
        print(
            f"{record['id']}: clean {record['clean_logit']:.3f}, corrupt "
            f"{record['corrupt_logit']:.3f}, gap {record['logit_gap']:.3f}"
            + (f", robustness {record['robustness']}" if "robustness" in record else "")
        )
    if not args.no_publish:
        args.publish.mkdir(parents=True, exist_ok=True)
        (args.publish / output.name).write_bytes(output.read_bytes())


if __name__ == "__main__":
    main()
