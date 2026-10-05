"""Load every ONNX export and run a PyTorch parity check for the transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

import induction_diagnostic as induction
from train_language_models import (
    TinyTransformer,
    TransformerConfig,
    load_corpus,
    split_corpus,
)
from train_mnist_models import MnistDenoiser, MnistVae, VaeDecoder, VaeEncoder


def validate_models(directory: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(directory.glob("*.onnx")):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        records.append(
            {
                "file": path.name,
                "opset": model.opset_import[0].version,
                "inputs": [value.name for value in session.get_inputs()],
                "outputs": [value.name for value in session.get_outputs()],
            }
        )
    return records


def transformer_parity(
    checkpoint_path: Path,
    onnx_path: Path,
    seed: int,
) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TransformerConfig(**checkpoint["config"])
    model = TinyTransformer(config).eval()
    model.load_state_dict(checkpoint["state_dict"])
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, config.vocab_size, (2, 37), generator=generator
    )
    with torch.no_grad():
        torch_outputs = model(input_ids)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_outputs = session.run(None, {"input_ids": input_ids.numpy()})
    names = ["logits", "q", "k", "v", "attention", "residual"]
    differences = {
        name: float(
            np.max(
                np.abs(
                    expected.detach().numpy()
                    - np.asarray(actual)
                )
            )
        )
        for name, expected, actual in zip(names, torch_outputs, ort_outputs)
    }
    # Absolute tolerance 2e-4 plus a float32 scale term: the residual stream carries a few
    # large "massive activation" entries (magnitude in the hundreds), where one float32
    # rounding step is already ~3e-5, so a purely absolute bound would test the
    # arithmetic's rounding rather than the export. 5e-6 of the tensor's largest entry is
    # about 40 float32 steps; a wrong weight or a missing op misses by orders of magnitude.
    for name, expected in zip(names, torch_outputs):
        scale = float(np.max(np.abs(expected.detach().numpy())))
        if differences[name] > 2e-4 + 5e-6 * scale:
            raise RuntimeError(
                f"Transformer ONNX parity failed for {name}: {differences} "
                f"(largest magnitude {scale:.1f})"
            )
    return differences


def mnist_parity(
    checkpoints: Path,
    exports: Path,
    seed: int,
) -> dict[str, float] | None:
    """Compare the exported MNIST VAE and denoiser with their saved PyTorch weights."""
    vae_path = checkpoints / "mnist-vae.pt"
    denoiser_path = checkpoints / "mnist-denoiser.pt"
    names = [
        "mnist-vae-encoder.onnx",
        "mnist-vae-decoder.onnx",
        "mnist-diffusion-denoiser.onnx",
    ]
    if not (
        vae_path.exists()
        and denoiser_path.exists()
        and all((exports / name).exists() for name in names)
    ):
        return None
    vae = MnistVae().eval()
    vae.load_state_dict(torch.load(vae_path, map_location="cpu", weights_only=True))
    saved = torch.load(denoiser_path, map_location="cpu", weights_only=True)
    denoiser = MnistDenoiser(**saved["config"]).eval()
    denoiser.load_state_dict(saved["state_dict"])
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(4, 1, 28, 28, generator=generator)
    latents = torch.randn(4, 2, generator=generator)
    noisy = torch.randn(4, 1, 28, 28, generator=generator)
    timesteps = torch.tensor([0, 33, 66, 99])
    cases = {
        "encoder": (VaeEncoder(vae), names[0], {"image": images}),
        "decoder": (VaeDecoder(vae), names[1], {"latent": latents}),
        "denoiser": (denoiser, names[2], {"image": noisy, "timestep": timesteps}),
    }
    differences = {}
    for label, (module, name, inputs) in cases.items():
        with torch.no_grad():
            expected = module(*inputs.values())
        expected = expected if isinstance(expected, tuple) else (expected,)
        session = ort.InferenceSession(
            str(exports / name), providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {key: value.numpy() for key, value in inputs.items()})
        differences[label] = float(
            max(
                np.max(np.abs(want.detach().numpy() - got))
                for want, got in zip(expected, actual)
            )
        )
    if max(differences.values()) > 2e-4:
        raise RuntimeError(f"MNIST ONNX parity failed: {differences}")
    return differences


def induction_gate(
    onnx_path: Path, metadata_path: Path, corpus_path: Path
) -> dict[str, object] | None:
    """Re-run the induction diagnostic on the exported ONNX file itself.

    The recorded verdict in the metadata must match what the exported graph does on
    held-out periods, so a stale or hand-edited ``passes_shipping_threshold`` is caught.
    """
    if not (onnx_path.exists() and corpus_path.exists()):
        return None
    _, _, _, encoded = load_corpus(corpus_path)
    _, heldout = split_corpus(encoded)
    report = induction.run_diagnostic(
        induction.onnx_forward(str(onnx_path)), heldout.numpy(), 66
    )
    recorded = None
    if metadata_path.exists():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))[
            "transformer"
        ]["induction_diagnostic"]["passes_shipping_threshold"]
    summary = {
        "passes_shipping_threshold": report["passes_shipping_threshold"],
        "recorded_in_metadata": recorded,
        "best_layer": report["best_layer"],
        "best_head": report["best_head"],
        "best_score": report["best_score"],
        "failures": report["failures"],
    }
    if recorded is not None and recorded != report["passes_shipping_threshold"]:
        raise RuntimeError(f"Recorded induction verdict disagrees with the ONNX file: {summary}")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", type=Path, default=root / "exports")
    parser.add_argument(
        "--corpus", type=Path, default=root / "data" / "tiny-shakespeare.txt"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "checkpoints" / "tiny-transformer.pt",
    )
    parser.add_argument("--checkpoints", type=Path, default=root / "checkpoints")
    args = parser.parse_args()

    records = validate_models(args.exports)
    transformer_path = args.exports / "tiny-transformer.onnx"
    parity = (
        transformer_parity(args.checkpoint, transformer_path, seed=7331)
        if transformer_path.exists() and args.checkpoint.exists()
        else None
    )
    mnist = mnist_parity(args.checkpoints, args.exports, seed=7331)
    gate = induction_gate(
        transformer_path, args.exports / "language-models.metadata.json", args.corpus
    )
    report = {
        "models": records,
        "transformer_max_absolute_error": parity,
        "transformer_induction_gate": gate,
        "mnist_max_absolute_error": mnist,
    }
    (args.exports / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
