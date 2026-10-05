"""Train a sparse autoencoder on the transformer's residual stream."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import trange

from train_language_models import (
    TinyTransformer,
    TransformerConfig,
    choose_device,
    load_corpus,
    split_corpus,
)


# Windows start with almost no context: the residual at position p depends on only p + 1
# characters. Activations there are large and unlike running text, so top-activation
# examples drawn from the first few positions describe the window start, not language.
EARLY_POSITIONS = 4


class SparseAutoencoder(nn.Module):
    def __init__(
        self, input_width: int, features: int, active_features: int
    ) -> None:
        super().__init__()
        self.active_features = active_features
        self.encoder = nn.Linear(input_width, features)
        self.decoder = nn.Linear(features, input_width, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

    def forward(self, activations: Tensor) -> tuple[Tensor, Tensor]:
        dense = F.relu(self.encoder(activations))
        top_values, top_indices = dense.topk(self.active_features, dim=-1)
        features = torch.zeros_like(dense).scatter(-1, top_indices, top_values)
        reconstruction = self.decoder(features)
        return features, reconstruction


def load_transformer(checkpoint_path: Path, device: torch.device) -> TinyTransformer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TransformerConfig(**checkpoint["config"])
    model = TinyTransformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def collect_residuals(
    model: TinyTransformer,
    encoded: Tensor,
    *,
    layer: int,
    starts: list[int],
    device: torch.device,
) -> Tensor:
    """Residual-stream rows for the 64-character windows beginning at ``starts``.

    Row ``window * block_size + position`` is the stream after block ``layer + 1`` at that
    position of that window.
    """
    batches = []
    for offset in range(0, len(starts), 16):
        chunk = starts[offset : offset + 16]
        inputs = torch.stack(
            [
                encoded[start : start + model.config.block_size]
                for start in chunk
            ]
        ).to(device)
        *_, residual = model(inputs)
        batches.append(residual[layer].detach().cpu())
    return torch.cat(batches).reshape(-1, model.config.model_width)


def sample_starts(low: int, high: int, count: int, block_size: int) -> list[int]:
    """Random window starts whose whole window lies in ``[low, high)``."""
    return [random.randrange(low, high - block_size - 1) for _ in range(count)]


def export_sae(model: SparseAutoencoder, output: Path, input_width: int) -> None:
    model = model.cpu().eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, input_width),),
        output,
        input_names=["residual"],
        output_names=["features", "reconstruction"],
        dynamic_axes={
            "residual": {0: "tokens"},
            "features": {0: "tokens"},
            "reconstruction": {0: "tokens"},
        },
        opset_version=18,
        dynamo=False,
    )


@torch.no_grad()
def run_in_chunks(
    model: SparseAutoencoder, values: Tensor, device: torch.device, chunk: int = 8192
) -> tuple[Tensor, Tensor]:
    model.eval()
    features, reconstructions = [], []
    for start in range(0, len(values), chunk):
        encoded, reconstruction = model(values[start : start + chunk].to(device))
        features.append(encoded.cpu())
        reconstructions.append(reconstruction.cpu())
    return torch.cat(features), torch.cat(reconstructions)


def position_statistics(
    feature_values: Tensor, block_size: int
) -> dict[str, Tensor]:
    """Where in its 64-character window does each feature fire?

    ``feature_values`` is [windows * block_size, features]. At window position 0 the
    transformer's residual depends only on the single character there (causal attention
    has nothing earlier to read), so features that fire only there are first-character
    detectors, not features of language in context.
    """
    windows = feature_values.shape[0] // block_size
    by_position = feature_values.view(windows, block_size, -1)
    fired = by_position > 0
    at_zero = fired[:, 0, :].sum(dim=0)
    total = fired.sum(dim=(0, 1))
    maximum_after_zero = by_position[:, 1:, :].amax(dim=(0, 1))
    maximum_at_zero = by_position[:, 0, :].amax(dim=0)
    return {
        "fires": total,
        "fires_at_position_zero": at_zero,
        "maximum_at_position_zero": maximum_at_zero,
        "maximum_after_position_zero": maximum_after_zero,
        # Fires at position 0 and never anywhere else in the scanned windows.
        "position_zero_only": (at_zero > 0) & (total == at_zero),
        "dead": total == 0,
    }


def example_records(
    rows: list[int],
    values: list[float],
    starts: list[int],
    corpus: str,
    block_size: int,
) -> list[dict[str, object]]:
    examples = []
    for value, row in zip(values, rows):
        window_index = row // block_size
        token_offset = row % block_size
        corpus_index = starts[window_index] + token_offset
        left = max(0, corpus_index - 32)
        right = min(len(corpus), corpus_index + 33)
        examples.append(
            {
                "activation": value,
                "text": corpus[left:right].replace("\n", " "),
                "focus_offset": corpus_index - left,
                # Characters of left context the model had: 0 means the first
                # character of its 64-character window.
                "window_position": token_offset,
            }
        )
    return examples


def top_activating_texts(
    feature_values: Tensor,
    stats: dict[str, Tensor],
    starts: list[int],
    corpus: str,
    block_size: int,
    top_k: int,
    selected: int,
) -> list[dict[str, object]]:
    maxima = feature_values.max(dim=0).values
    selected_features = maxima.topk(min(selected, maxima.numel())).indices
    chance_share = 1 / block_size
    positions = torch.arange(len(feature_values)) % block_size
    in_context = positions >= EARLY_POSITIONS
    records = []

    for feature in selected_features.tolist():
        values, rows = feature_values[:, feature].topk(top_k)
        examples = example_records(
            rows.tolist(), values.tolist(), starts, corpus, block_size
        )
        at_zero_examples = sum(1 for example in examples if example["window_position"] == 0)
        early_examples = sum(
            1 for example in examples if example["window_position"] < EARLY_POSITIONS
        )
        fires = int(stats["fires"][feature])
        at_zero = int(stats["fires_at_position_zero"][feature])
        record: dict[str, object] = {
            "feature": feature,
            "maximum": maxima[feature].item(),
            "maximum_after_position_zero": stats["maximum_after_position_zero"][feature].item(),
            "fires": fires,
            "fires_at_position_zero": at_zero,
            "position_zero_share": at_zero / fires if fires else 0.0,
            "position_zero_only": bool(stats["position_zero_only"][feature]),
            "position_zero_dominated": bool(
                fires and at_zero / fires > 20 * chance_share
            ),
            "examples_at_position_zero": at_zero_examples,
            # True when at least half of the cached top examples are the first character
            # of a window: the maximum then describes the window start, not language.
            "position_zero_artefact": at_zero_examples * 2 >= len(examples),
            "examples_at_early_positions": early_examples,
            # The same test over the first EARLY_POSITIONS positions (0 to 3).
            "window_start_artefact": early_examples * 2 >= len(examples),
            "examples": examples,
        }
        if early_examples:
            masked = feature_values[:, feature].masked_fill(~in_context, float("-inf"))
            later_values, later_rows = masked.topk(top_k)
            record["examples_in_context"] = example_records(
                later_rows.tolist(),
                later_values.tolist(),
                starts,
                corpus,
                block_size,
            )
        records.append(record)
    return records


@torch.no_grad()
def reconstruction_metrics(
    model: SparseAutoencoder, values: Tensor, block_size: int, device: torch.device
) -> dict[str, float]:
    """Held-out reconstruction quality on normalised activations."""
    features, reconstruction = run_in_chunks(model, values, device)
    values = values.cpu()
    residual = (values - reconstruction).square().sum(dim=-1)
    energy = values.square().sum(dim=-1)
    positions = torch.arange(len(values)) % block_size
    later = positions > 0
    return {
        "mean_squared_error_per_dimension": float((values - reconstruction).square().mean()),
        "fraction_of_variance_unexplained": float(residual.sum() / energy.sum()),
        "fraction_unexplained_position_zero": float(
            residual[~later].sum() / energy[~later].sum()
        ),
        "fraction_unexplained_after_position_zero": float(
            residual[later].sum() / energy[later].sum()
        ),
        "fraction_unexplained_from_position_4": float(
            residual[positions >= EARLY_POSITIONS].sum()
            / energy[positions >= EARLY_POSITIONS].sum()
        ),
        "active_features_per_token": float((features > 0).sum(dim=-1).float().mean()),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=Path, default=root / "data" / "tiny-shakespeare.txt"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "checkpoints" / "tiny-transformer.pt",
    )
    parser.add_argument("--output", type=Path, default=root / "exports")
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "checkpoints")
    parser.add_argument(
        "--publish",
        type=Path,
        default=root.parent
        / "src"
        / "modules"
        / "interpretability-features"
        / "assets",
    )
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--active-features", type=int, default=32)
    parser.add_argument("--windows", type=int, default=1600)
    parser.add_argument("--heldout-windows", type=int, default=200)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--selected-features", type=int, default=128)
    parser.add_argument(
        "--skip-window-start",
        type=int,
        default=0,
        help=(
            "Experiment: leave the first N window positions out of SAE training. "
            "The shipped SAE uses 0, so it covers every position the lab can feed it."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=221)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    corpus, _, _, encoded = load_corpus(args.corpus)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = TransformerConfig(**checkpoint["config"])
    block = config.block_size
    transformer = load_transformer(args.checkpoint, device)

    # Training windows come from the first 90% of the corpus (the transformer's own
    # training split); held-out windows come from the final 10%.
    boundary = len(split_corpus(encoded)[0])
    train_starts = sample_starts(0, boundary, args.windows, block)
    heldout_starts = sample_starts(boundary, len(encoded), args.heldout_windows, block)
    residuals = collect_residuals(
        transformer, encoded, layer=args.layer, starts=train_starts, device=device
    )
    heldout_residuals = collect_residuals(
        transformer, encoded, layer=args.layer, starts=heldout_starts, device=device
    )
    kept = (torch.arange(len(residuals)) % block) >= args.skip_window_start
    mean = residuals[kept].mean(dim=0, keepdim=True)
    centered = residuals - mean
    scale = centered[kept].square().mean().sqrt().clamp_min(1e-6)
    normalized = centered / scale
    heldout_normalized = (heldout_residuals - mean) / scale

    sae = SparseAutoencoder(
        config.model_width, args.features, args.active_features
    ).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)
    normalized_device = normalized[kept].to(device)
    progress = trange(args.steps, desc="Sparse autoencoder")
    history = []

    for step in progress:
        indices = torch.randint(0, len(normalized_device), (512,), device=device)
        batch = normalized_device[indices]
        features, reconstruction = sae(batch)
        reconstruction_loss = F.mse_loss(reconstruction, batch)
        sparsity_loss = features.abs().mean()
        loss = reconstruction_loss + 2e-3 * sparsity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)

        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            record = {
                "step": step + 1,
                "loss": loss.item(),
                "reconstruction": reconstruction_loss.item(),
                "mean_activation": sparsity_loss.item(),
                "active_fraction": (features > 0).float().mean().item(),
            }
            history.append(record)
            progress.set_postfix(
                reconstruction=f"{reconstruction_loss.item():.3f}",
                active=f"{record['active_fraction']:.2%}",
            )

    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "input_mean": mean[0],
            "input_scale": scale,
            "layer": args.layer,
            "active_features": args.active_features,
        },
        args.checkpoint_dir / "residual-sae.pt",
    )
    metrics = {
        "training_windows": reconstruction_metrics(sae, normalized, block, device),
        "heldout_windows": reconstruction_metrics(sae, heldout_normalized, block, device),
    }
    sae = sae.cpu()
    sae_path = args.output / "residual-sae.onnx"
    export_sae(sae, sae_path, config.model_width)

    # Feature statistics and top examples over every window gathered (training + held-out).
    scan_values, _ = run_in_chunks(
        sae, torch.cat([normalized, heldout_normalized]), torch.device("cpu")
    )
    scan_starts = train_starts + heldout_starts
    stats = position_statistics(scan_values, block)
    examples = top_activating_texts(
        scan_values,
        stats,
        scan_starts,
        corpus,
        block,
        top_k=8,
        selected=args.selected_features,
    )
    zero_only = [int(i) for i in stats["position_zero_only"].nonzero().flatten()]
    shares = stats["fires_at_position_zero"].float() / stats["fires"].clamp_min(1).float()
    dominated = [
        int(i)
        for i in ((shares > 20 / block) & (stats["fires"] > 0)).nonzero().flatten()
        if int(i) not in zero_only
    ]
    dead = [int(i) for i in stats["dead"].nonzero().flatten()]
    summary = {
        "scanned_windows": len(scan_starts),
        "scanned_tokens": len(scan_values),
        "dead_features": len(dead),
        "position_zero_only_features": zero_only,
        "position_zero_dominated_features": dominated,
        "selected_position_zero_only": [
            record["feature"] for record in examples if record["position_zero_only"]
        ],
        "selected_position_zero_artefacts": [
            record["feature"] for record in examples if record["position_zero_artefact"]
        ],
        "selected_window_start_artefacts": [
            record["feature"] for record in examples if record["window_start_artefact"]
        ],
        "selected_identical_activation_features": [
            record["feature"]
            for record in examples
            if len({round(example["activation"], 4) for example in record["examples"]}) == 1  # type: ignore[union-attr]
        ],
        "early_positions": EARLY_POSITIONS,
        "note": (
            "A feature is position_zero_only if, over the scanned windows, it is active at "
            "window position 0 and nowhere else. It is position_zero_dominated if more than "
            "20 times its chance share (1/64) of its activations are at position 0. It is a "
            "position_zero_artefact if at least half of its cached top examples are at "
            "position 0, and a window_start_artefact if at least half are within the first "
            f"{EARLY_POSITIONS} positions; its maximum then describes the window start, and "
            "examples_in_context gives its top examples from later positions. At position 0 "
            "the residual depends only on the character itself."
        ),
    }
    examples_path = args.output / "sae-top-activations.json"
    examples_path.write_text(
        json.dumps(
            {
                "layer": args.layer,
                "features": args.features,
                "active_features": args.active_features,
                "training_windows": args.windows,
                "training_rows": int(kept.sum()),
                "skipped_window_start_positions": args.skip_window_start,
                "normalization_mean": mean[0].tolist(),
                "normalization_scale": scale.item(),
                "training": history,
                "evaluation": metrics,
                "position_summary": summary,
                "top_activations": examples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"evaluation": metrics, "position_summary": summary}, indent=2))
    if not args.no_publish:
        args.publish.mkdir(parents=True, exist_ok=True)
        for path in [sae_path, examples_path]:
            (args.publish / path.name).write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
