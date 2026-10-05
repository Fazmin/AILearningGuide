"""Train the tiny MNIST diffusion denoiser and two-dimensional VAE.

Both models train on the full 60,000-image MNIST training split. The denoiser is
exported with an exponential moving average of its weights and must pass a sample
gate before anything is published: images drawn from *pure noise* with the same
DDPM sampler the Diffusion & VAEs lab runs (src/modules/diffusion-vaes/diffusion.ts)
have to be classified as digits, with every class represented, by an independent
classifier. A run that fails the gate exports for inspection and exits
unsuccessfully.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import datasets
from tqdm import tqdm

from train_language_models import choose_device, parameter_count

DIFFUSION_STEPS = 100
# Nichol & Dhariwal's cosine schedule, offset s = 0.008, evaluated over 98% of the
# curve so the last step keeps sqrt(alpha-bar) of about 0.03 instead of exactly
# zero. The lab divides by sqrt(alpha-bar) when it shows the predicted clean image.
COSINE_OFFSET = 0.008
COSINE_END = 0.98
VAE_KL_WEIGHT = 0.35  # mirrored by KL_WEIGHT in src/modules/diffusion-vaes/vae.ts

# Shipping gate for samples generated from pure noise (see evaluate_samples).
# Novelty is judged against real held-out digits, not an absolute distance: thin
# "1"s sit within 1.5 pixel-space units of a training digit even for real test images
# (3% of the 10,000 test digits do), so a fixed floor would reject honest samples.
GATE_MEAN_CONFIDENCE = 0.90
GATE_CLASSES_PRESENT = 10
GATE_NOVELTY_RATIO = 0.8  # mean nearest-training distance, samples / real test digits
NEAR_COPY_DISTANCE = 1.5


class MnistVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.SiLU(),
            nn.Linear(256, 96),
            nn.SiLU(),
        )
        self.mean = nn.Linear(96, 2)
        self.log_variance = nn.Linear(96, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, 96),
            nn.SiLU(),
            nn.Linear(96, 256),
            nn.SiLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(images.flatten(1))
        return self.mean(hidden), self.log_variance(hidden)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent).view(-1, 1, 28, 28)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encode(images)
        noise = torch.randn_like(mean)
        latent = mean + torch.exp(0.5 * log_variance) * noise
        return self.decode(latent), mean, log_variance


class VaeEncoder(nn.Module):
    def __init__(self, vae: MnistVae) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.vae.encode(images)


class VaeDecoder(nn.Module):
    def __init__(self, vae: MnistVae) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latent: Tensor) -> Tensor:
        return self.vae.decode(latent)


def timestep_features(timestep: Tensor, width: int) -> Tensor:
    """Sinusoidal features of the integer step, the usual diffusion time embedding."""
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=timestep.device) / half
    )
    angles = timestep.float().view(-1, 1) * frequencies.view(1, -1)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class ResidualBlock(nn.Module):
    """Two 3 x 3 convolutions with the time embedding added between them."""

    def __init__(self, inputs: int, outputs: int, time_width: int) -> None:
        super().__init__()
        self.norm_in = nn.GroupNorm(8, inputs)
        self.conv_in = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.time = nn.Linear(time_width, outputs)
        self.norm_out = nn.GroupNorm(8, outputs)
        self.conv_out = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.skip = nn.Conv2d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, features: Tensor, time: Tensor) -> Tensor:
        hidden = self.conv_in(F.silu(self.norm_in(features)))
        hidden = hidden + self.time(time)[:, :, None, None]
        hidden = self.conv_out(F.silu(self.norm_out(hidden)))
        return hidden + self.skip(features)


class MnistDenoiser(nn.Module):
    """A miniature U-Net that predicts the noise in a 28 x 28 image from the image
    and its integer step.

    Three resolutions (28, 14, 7 pixels) with `base`, 2 x `base` and 4 x `base`
    channels; the two coarser levels are reached by stride-2 convolutions and
    returned to by transposed convolutions, with skip connections between matching
    levels. Every residual block receives the sinusoidal time embedding.

    ONNX contract (unchanged since the first release): inputs `image`
    [batch, 1, 28, 28] float32 and `timestep` [batch] int64, output `noise`
    [batch, 1, 28, 28] float32. It is unconditional: no class label is an input.
    """

    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.config = {"base": base}
        self.time_width = 128
        wide, wider = base * 2, base * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(self.time_width, self.time_width),
            nn.SiLU(),
            nn.Linear(self.time_width, self.time_width),
        )
        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.down_block_1 = ResidualBlock(base, base, self.time_width)
        self.downsample_1 = nn.Conv2d(base, wide, 3, stride=2, padding=1)
        self.down_block_2 = ResidualBlock(wide, wide, self.time_width)
        self.downsample_2 = nn.Conv2d(wide, wider, 3, stride=2, padding=1)
        self.middle_1 = ResidualBlock(wider, wider, self.time_width)
        self.middle_2 = ResidualBlock(wider, wider, self.time_width)
        self.upsample_2 = nn.ConvTranspose2d(wider, wide, 4, stride=2, padding=1)
        self.up_block_2 = ResidualBlock(wide * 2, wide, self.time_width)
        self.upsample_1 = nn.ConvTranspose2d(wide, base, 4, stride=2, padding=1)
        self.up_block_1 = ResidualBlock(base * 2, base, self.time_width)
        self.output_norm = nn.GroupNorm(8, base)
        self.output = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, images: Tensor, timestep: Tensor) -> Tensor:
        time = self.time_embedding(timestep_features(timestep, self.time_width))
        full = self.down_block_1(self.stem(images), time)
        half = self.down_block_2(self.downsample_1(full), time)
        quarter = self.middle_2(self.middle_1(self.downsample_2(half), time), time)
        half = self.up_block_2(torch.cat([self.upsample_2(quarter), half], 1), time)
        full = self.up_block_1(torch.cat([self.upsample_1(half), full], 1), time)
        return self.output(F.silu(self.output_norm(full)))


class Classifier(nn.Module):
    """Independent digit classifier used only to judge generated samples."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)


def load_mnist(root: Path, train: bool, samples: int | None = None, seed: int = 0):
    dataset = datasets.MNIST(root, train=train, download=True)
    images = dataset.data.float().div(255).unsqueeze(1)
    labels = dataset.targets
    if samples is not None and samples < len(images):
        generator = torch.Generator().manual_seed(seed)
        keep = torch.randperm(len(images), generator=generator)[:samples]
        images, labels = images[keep], labels[keep]
    return images, labels


def batches(count: int, batch_size: int, device: torch.device):
    order = torch.randperm(count, device=device)
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def train_vae(
    model: MnistVae,
    images: Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int = 256,
) -> list[dict[str, float]]:
    model.to(device).train()
    images = images.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-5)
    history = []
    for epoch in range(epochs):
        totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
        seen = 0
        for index in tqdm(
            list(batches(len(images), batch_size, device)),
            desc=f"VAE {epoch + 1}/{epochs}",
            leave=False,
        ):
            batch = images[index]
            reconstruction, mean, log_variance = model(batch)
            reconstruction_loss = F.binary_cross_entropy(
                reconstruction, batch, reduction="sum"
            ) / len(batch)
            kl = -0.5 * torch.sum(
                1 + log_variance - mean.square() - log_variance.exp()
            ) / len(batch)
            loss = reconstruction_loss + VAE_KL_WEIGHT * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen += len(batch)
            totals["loss"] += loss.item() * len(batch)
            totals["reconstruction"] += reconstruction_loss.item() * len(batch)
            totals["kl"] += kl.item() * len(batch)
        scheduler.step()
        history.append(
            {"epoch": epoch + 1, **{key: value / seen for key, value in totals.items()}}
        )
    return history


def diffusion_schedule(steps: int = DIFFUSION_STEPS) -> tuple[Tensor, Tensor, Tensor]:
    """Cosine schedule: returns beta, alpha and alpha-bar as float32 tensors.

    The shipped JSON stores alpha-bar, so the lab and this script see the same
    float32 values; beta and alpha are recovered from it the same way the lab does.
    """

    def cosine(position: Tensor) -> Tensor:
        angle = (position + COSINE_OFFSET) / (1 + COSINE_OFFSET) * math.pi / 2
        return torch.cos(angle) ** 2

    positions = torch.arange(1, steps + 1, dtype=torch.float64) / steps * COSINE_END
    cumulative = (cosine(positions) / cosine(torch.zeros((), dtype=torch.float64))).float()
    alpha = torch.empty_like(cumulative)
    alpha[0] = cumulative[0]
    alpha[1:] = cumulative[1:] / cumulative[:-1]
    return 1 - alpha, alpha, cumulative


def noise_image(images: Tensor, timestep: Tensor, noise: Tensor, cumulative: Tensor):
    signal = cumulative[timestep].sqrt().view(-1, 1, 1, 1)
    noise_scale = (1 - cumulative[timestep]).sqrt().view(-1, 1, 1, 1)
    return signal * images + noise_scale * noise


def train_diffusion(
    model: MnistDenoiser,
    images: Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ema_decay: float,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Train with the epsilon objective; returns the EMA copy and the loss history."""
    model.to(device).train()
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    images = images.to(device) * 2 - 1
    _, _, cumulative = diffusion_schedule()
    cumulative = cumulative.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    steps_per_epoch = math.ceil(len(images) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup = min(500, total_steps // 10)

    def learning_rate_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_at)
    history = []
    step = 0
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for index in batches(len(images), batch_size, device):
            batch = images[index]
            timestep = torch.randint(0, len(cumulative), (len(batch),), device=device)
            noise = torch.randn_like(batch)
            predicted = model(noise_image(batch, timestep, noise, cumulative), timestep)
            loss = F.mse_loss(predicted, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            decay = min(ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - decay)
            total += loss.item() * len(batch)
            seen += len(batch)
        history.append({"epoch": epoch + 1, "noise_mse": total / seen})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"diffusion epoch {epoch + 1}/{epochs} noise_mse {total / seen:.4f}", flush=True)
    return ema, history


@torch.no_grad()
def held_out_noise_mse(
    model: nn.Module, images: Tensor, device: torch.device, seed: int
) -> dict[str, float]:
    """Noise-prediction error of a model on test digits, overall and at three steps."""
    model.to(device).eval()
    _, _, cumulative = diffusion_schedule()
    cumulative = cumulative.to(device)
    images = images.to(device) * 2 - 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: dict[str, float] = {}
    overall = 0.0
    for name, timestep_value in [("t0", 0), ("t49", 49), ("t99", 99)]:
        noise = torch.randn(images.shape, generator=generator).to(device)
        timestep = torch.full((len(images),), timestep_value, device=device)
        prediction = model(noise_image(images, timestep, noise, cumulative), timestep)
        result[name] = F.mse_loss(prediction, noise).item()
    for _ in range(4):
        noise = torch.randn(images.shape, generator=generator).to(device)
        timestep = torch.randint(
            0, len(cumulative), (len(images),), device=device, generator=None
        )
        prediction = model(noise_image(images, timestep, noise, cumulative), timestep)
        overall += F.mse_loss(prediction, noise).item() / 4
    result["uniform_t"] = overall
    return result


def export_onnx(
    module: nn.Module,
    inputs: tuple[Tensor, ...],
    output: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    module = module.cpu().eval()
    dynamic_axes = {
        name: {0: "batch"} for name in [*input_names, *output_names]
    }
    torch.onnx.export(
        module,
        inputs,
        output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        dynamo=False,
    )


def onnx_parity(
    module: nn.Module, path: Path, inputs: dict[str, np.ndarray], tolerance: float
) -> float:
    """Largest absolute difference between PyTorch and onnxruntime (CPU) outputs."""
    module = module.cpu().eval()
    with torch.no_grad():
        expected = module(*[torch.from_numpy(value) for value in inputs.values()])
    expected = expected if isinstance(expected, tuple) else (expected,)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, inputs)
    worst = max(
        float(np.max(np.abs(want.numpy() - got))) for want, got in zip(expected, actual)
    )
    if worst > tolerance:
        raise RuntimeError(f"{path.name} ONNX parity failed: {worst} > {tolerance}")
    return worst


def sample_from_noise(
    denoiser_path: Path, cumulative: np.ndarray, seeds: list[int]
) -> np.ndarray:
    """Reverse process from pure noise, mirroring the lab's TypeScript sampler.

    DDPM ancestral update (Ho et al. 2020, Algorithm 2) with sigma_t^2 = beta_t,
    beta_t = 1 - alpha-bar_t / alpha-bar_(t-1), no fresh noise on the last step,
    one onnxruntime call per step on the whole batch. Returns images in [-1, 1].
    """
    alpha_bar = cumulative.astype(np.float64)
    alpha = np.empty_like(alpha_bar)
    alpha[0] = alpha_bar[0]
    alpha[1:] = alpha_bar[1:] / alpha_bar[:-1]
    beta = 1 - alpha
    session = ort.InferenceSession(str(denoiser_path), providers=["CPUExecutionProvider"])
    generators = [np.random.default_rng(seed) for seed in seeds]
    x = np.stack([g.standard_normal((1, 28, 28)) for g in generators]).astype(np.float32)
    for call in range(len(alpha_bar)):
        t = len(alpha_bar) - 1 - call
        epsilon = session.run(
            None, {"image": x, "timestep": np.full((len(seeds),), t, dtype=np.int64)}
        )[0]
        mean = (x - beta[t] / np.sqrt(1 - alpha_bar[t]) * epsilon) / np.sqrt(alpha[t])
        if t > 0:
            fresh = np.stack(
                [g.standard_normal((1, 28, 28)) for g in generators]
            ).astype(np.float32)
            mean = mean + np.sqrt(beta[t]) * fresh
        x = mean.astype(np.float32)
    return x


def train_classifier(
    train_images: Tensor, train_labels: Tensor, test_images: Tensor, test_labels: Tensor,
    device: torch.device,
) -> tuple[Classifier, float]:
    model = Classifier().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    images, labels = train_images.to(device), train_labels.to(device)
    for _ in range(4):
        model.train()
        for index in batches(len(images), 128, device):
            loss = F.cross_entropy(model(images[index]), labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        accuracy = (
            model(test_images.to(device)).argmax(1).cpu() == test_labels
        ).float().mean().item()
    return model, accuracy


@torch.no_grad()
def evaluate_samples(
    samples: np.ndarray,
    classifier: Classifier,
    train_images: Tensor,
    train_labels: Tensor,
    device: torch.device,
) -> dict[str, object]:
    """Judge generated images: classifier confidence, class coverage, novelty."""
    pixels = torch.from_numpy(np.clip((samples + 1) / 2, 0, 1)).float()
    probabilities = F.softmax(classifier(pixels.to(device)), dim=1).cpu()
    confidence, predicted = probabilities.max(dim=1)
    distances, neighbours = torch.cdist(
        pixels.flatten(1), train_images.flatten(1)
    ).min(dim=1)
    closest = distances.argsort()[:8]
    return {
        "count": len(pixels),
        "mean_confidence": confidence.mean().item(),
        "fraction_above_0_9": (confidence > 0.9).float().mean().item(),
        "class_counts": torch.bincount(predicted, minlength=10).tolist(),
        "mean_nearest_training_distance": distances.mean().item(),
        "min_nearest_training_distance": distances.min().item(),
        "fraction_within_near_copy_distance": (distances < NEAR_COPY_DISTANCE)
        .float()
        .mean()
        .item(),
        "closest_eight_nearest_training_labels": train_labels[neighbours[closest]].tolist(),
    }


def save_sample_grid(samples: np.ndarray, path: Path, columns: int = 16) -> None:
    from PIL import Image

    pixels = np.clip((samples[:, 0] + 1) / 2, 0, 1)
    rows = math.ceil(len(pixels) / columns)
    canvas = np.zeros((rows * 28, columns * 28), dtype=np.float32)
    for index, image in enumerate(pixels):
        row, column = divmod(index, columns)
        canvas[row * 28 : (row + 1) * 28, column * 28 : (column + 1) * 28] = image
    Image.fromarray((canvas * 255).astype(np.uint8)).resize(
        (columns * 28 * 3, rows * 28 * 3), Image.NEAREST
    ).save(path)


MAP_GRID = 16
MAP_LIMIT = 4.0


def latent_grid(count: int, limit: float) -> Tensor:
    """Latent points of the lab's mosaic: count x count tile centres over
    [-limit, limit], row-major from the top (largest z2), as `latentGrid` in
    src/modules/diffusion-vaes/vae.ts."""
    span = 2 * limit
    points = []
    for row in range(count):
        z2 = limit - (row + 0.5) / count * span
        for column in range(count):
            points.append([-limit + (column + 0.5) / count * span, z2])
    return torch.tensor(points)


@torch.no_grad()
def vae_latent_report(
    vae: MnistVae,
    test_images: Tensor,
    test_labels: Tensor,
    classifier: Classifier,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    """Where the 10,000 test digits land on the latent map, which digit the
    independent classifier reads in each mosaic tile, and how well the decoder
    reconstructs and generates. The lab quotes these."""

    def read(images: Tensor) -> tuple[Tensor, Tensor]:
        probabilities = F.softmax(classifier(images.to(device)), dim=1).cpu()
        confidence, label = probabilities.max(dim=1)
        return confidence, label

    vae.cpu().eval()
    mean, log_variance = vae.encode(test_images)
    sigma = (0.5 * log_variance).exp()
    radius = mean.norm(dim=1)
    confidence, label = read(vae.decode(latent_grid(MAP_GRID, MAP_LIMIT)))
    reconstruction = vae.decode(mean)
    _, reconstructed_label = read(reconstruction)
    prior = torch.randn(10_000, 2, generator=torch.Generator().manual_seed(seed))
    prior_confidence, _ = read(vae.decode(prior))
    centroids = [
        mean[test_labels == digit].mean(dim=0).tolist() for digit in range(10)
    ]
    return {
        "test_images": len(test_images),
        "encoded_mean_average": mean.mean(dim=0).tolist(),
        "encoded_mean_std": mean.std(dim=0).tolist(),
        "posterior_sigma_average": sigma.mean(dim=0).tolist(),
        "fraction_within_radius_2": (radius < 2).float().mean().item(),
        "fraction_beyond_radius_3": (radius > 3).float().mean().item(),
        "class_centroids": centroids,
        "reconstruction": {
            "bce": F.binary_cross_entropy(
                reconstruction, test_images, reduction="sum"
            ).item()
            / len(test_images),
            "kl": (
                -0.5 * torch.sum(1 + log_variance - mean.square() - log_variance.exp())
            ).item()
            / len(test_images),
            "label_agreement": (reconstructed_label == test_labels).float().mean().item(),
        },
        "prior_samples": {
            "count": len(prior),
            "mean_confidence": prior_confidence.mean().item(),
            "fraction_above_0_9": (prior_confidence > 0.9).float().mean().item(),
        },
        "map": {
            "grid": MAP_GRID,
            "limit": MAP_LIMIT,
            "labels": label.tolist(),
            "confidence": [round(value, 3) for value in confidence.tolist()],
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data" / "mnist")
    parser.add_argument("--output", type=Path, default=root / "exports")
    parser.add_argument(
        "--publish",
        type=Path,
        default=root.parent / "src" / "modules" / "diffusion-vaes" / "assets",
    )
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--vae-epochs", type=int, default=40)
    parser.add_argument("--diffusion-epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=404)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    train_images, train_labels = load_mnist(args.data, True, args.samples, args.seed)
    test_images, test_labels = load_mnist(args.data, False)
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + 2)
    classifier, classifier_accuracy = train_classifier(
        train_images, train_labels, test_images, test_labels, device
    )

    torch.manual_seed(args.seed)
    vae = MnistVae()
    vae_history = train_vae(vae, train_images, device, args.vae_epochs)
    vae.to("cpu")
    vae_report = vae_latent_report(
        vae, test_images, test_labels, classifier, device, args.seed + 3
    )
    encoder_path = args.output / "mnist-vae-encoder.onnx"
    decoder_path = args.output / "mnist-vae-decoder.onnx"
    export_onnx(
        VaeEncoder(vae),
        (torch.zeros(1, 1, 28, 28),),
        encoder_path,
        ["image"],
        ["mean", "log_variance"],
    )
    export_onnx(
        VaeDecoder(vae),
        (torch.zeros(1, 2),),
        decoder_path,
        ["latent"],
        ["image"],
    )
    vae_parity = {
        "encoder": onnx_parity(
            VaeEncoder(vae), encoder_path,
            {"image": test_images[:8].numpy()}, 1e-4,
        ),
        "decoder": onnx_parity(
            VaeDecoder(vae), decoder_path,
            {"latent": np.random.default_rng(1).normal(size=(8, 2)).astype(np.float32)}, 1e-4,
        ),
    }

    torch.manual_seed(args.seed + 1)
    denoiser = MnistDenoiser(args.base_channels)
    ema, diffusion_history = train_diffusion(
        denoiser,
        train_images,
        device,
        args.diffusion_epochs,
        args.batch_size,
        args.learning_rate,
        args.ema_decay,
    )
    ema.to("cpu")
    denoiser_path = args.output / "mnist-diffusion-denoiser.onnx"
    export_onnx(
        ema,
        (torch.zeros(1, 1, 28, 28), torch.zeros(1, dtype=torch.long)),
        denoiser_path,
        ["image", "timestep"],
        ["noise"],
    )
    probe = np.random.default_rng(2)
    denoiser_parity = onnx_parity(
        ema,
        denoiser_path,
        {
            "image": probe.normal(size=(8, 1, 28, 28)).astype(np.float32),
            "timestep": probe.integers(0, DIFFUSION_STEPS, size=8).astype(np.int64),
        },
        1e-4,
    )
    checkpoint_directory = root / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    torch.save(vae.state_dict(), checkpoint_directory / "mnist-vae.pt")
    torch.save(
        {"config": ema.config, "state_dict": ema.state_dict()},
        checkpoint_directory / "mnist-denoiser.pt",
    )

    _, _, cumulative = diffusion_schedule()
    schedule_path = args.output / "mnist-diffusion-schedule.json"
    schedule_path.write_text(
        json.dumps({"steps": DIFFUSION_STEPS, "alpha_cumulative": cumulative.tolist()})
        + "\n",
        encoding="utf-8",
    )

    held_out = held_out_noise_mse(ema, test_images, device, args.seed)
    samples = sample_from_noise(
        denoiser_path, cumulative.numpy(), list(range(1, args.eval_samples + 1))
    )
    sample_check = {
        "start": "pure noise",
        "sampler": "DDPM ancestral, sigma_t^2 = beta_t, 100 steps",
        "classifier_test_accuracy": classifier_accuracy,
        **evaluate_samples(samples, classifier, train_images, train_labels, device),
    }
    reference_check = evaluate_samples(
        test_images[: args.eval_samples].numpy() * 2 - 1,
        classifier,
        train_images,
        train_labels,
        device,
    )
    sample_check["real_test_digits_mean_confidence"] = reference_check["mean_confidence"]
    sample_check["real_test_digits_mean_nearest_training_distance"] = reference_check[
        "mean_nearest_training_distance"
    ]
    sample_check["real_test_digits_fraction_within_near_copy_distance"] = reference_check[
        "fraction_within_near_copy_distance"
    ]
    sample_check["near_copy_distance"] = NEAR_COPY_DISTANCE
    save_sample_grid(samples, args.output / "mnist-diffusion-samples.png")

    final_signal = float(cumulative[-1].sqrt())
    metadata = {
        "dataset": "MNIST training split",
        "dataset_url": "https://yann.lecun.com/exdb/mnist/",
        "dataset_licence": (
            "Not established from a primary source: the upstream page only says MNIST "
            "is derived from NIST data, and mirrors state different terms. "
            "See the upstream project at https://yann.lecun.com/exdb/mnist/."
        ),
        "seed": args.seed,
        "samples": len(train_images),
        "vae": {
            "latent_dimensions": 2,
            "kl_weight": VAE_KL_WEIGHT,
            "parameters": parameter_count(vae),
            "epochs": args.vae_epochs,
            "onnx_parity_max_absolute_error": vae_parity,
            "test_digits": vae_report,
            "history": vae_history,
        },
        "diffusion": {
            "steps": DIFFUSION_STEPS,
            "schedule": {
                "kind": "cosine",
                "offset": COSINE_OFFSET,
                "curve_fraction": COSINE_END,
                "final_signal": final_signal,
                "final_noise": float((1 - cumulative[-1]).sqrt()),
            },
            "architecture": {
                "kind": "convolutional U-Net",
                "resolutions": [28, 14, 7],
                "base_channels": ema.config["base"],
                "residual_blocks": sum(
                    isinstance(module, ResidualBlock) for module in ema.modules()
                ),
                "time_embedding": "sinusoidal features, added inside every residual block",
                "conditioning": "none (unconditional)",
            },
            "parameters": parameter_count(ema),
            "epochs": args.diffusion_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "ema_decay": args.ema_decay,
            "onnx_parity_max_absolute_error": denoiser_parity,
            "held_out_noise_mse": held_out,
            "sample_check": sample_check,
            "history": diffusion_history,
        },
    }
    metadata_path = args.output / "mnist-models.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"held_out": held_out, "sample_check": sample_check}, indent=2))

    passes = (
        sample_check["mean_confidence"] >= GATE_MEAN_CONFIDENCE
        and sum(count > 0 for count in sample_check["class_counts"]) >= GATE_CLASSES_PRESENT
        and sample_check["mean_nearest_training_distance"]
        >= GATE_NOVELTY_RATIO * sample_check["real_test_digits_mean_nearest_training_distance"]
    )
    if not passes:
        print(
            "Pure-noise samples did not clear the shipping gate; nothing was published. "
            "Inspect mnist-diffusion-samples.png, then increase --diffusion-epochs "
            "or the denoiser size. Do not lower the gate.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    args.publish.mkdir(parents=True, exist_ok=True)
    for path in [
        encoder_path,
        decoder_path,
        denoiser_path,
        schedule_path,
        metadata_path,
    ]:
        (args.publish / path.name).write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
