# Teaching model pipeline

The teaching models are reproducible PyTorch models exported at ONNX opset 18. Generated
checkpoints, source datasets, and scratch exports stay ignored; only reviewed runtime
assets copied into a module's `assets/` folder ship with the application.

## Environment

```sh
python3 -m venv models/.venv
models/.venv/bin/pip install -r models/requirements.txt
python3 models/download_corpus.py
```

The corpus command downloads a pinned revision of Tiny Shakespeare, verifies its
SHA-256, and writes provenance beside the ignored text file. Shakespeare's works are
public domain. The source revision is pinned so a changed transcription cannot enter
training unnoticed.

## Build order

```sh
# Bigram table and the 1–2M parameter two-layer transformer (8,000 steps, about 10
# minutes on Apple MPS). The shipped GRU is reused, not retrained (--rnn retrain to
# change that), so its ONNX file and quoted figures stay reproducible. This publishes
# the transformer and vocabulary used by the Attention module.
# To try a recipe without touching shipped assets, add
#   --no-publish --output <dir>/exports --checkpoint-dir <dir>/checkpoints
# and install a passing run later with --promote <dir>.
models/.venv/bin/python models/train_language_models.py

# Prove the induction gate can fail (the first release) and can pass (idealised models).
# The first release's ONNX file is read from models/checkpoints/legacy-period8/ (ignored
# by git); recreate it with
#   git show 6bb37de:src/modules/attention/assets/tiny-transformer.onnx > \
#     models/checkpoints/legacy-period8/tiny-transformer.onnx
# (SHA-256 0029cae9f83f...5596b1, checked by the script). Without it that control is skipped.
models/.venv/bin/python models/check_induction_gate.py

# Residual-stream sparse autoencoder and top-activating text cache (about 3 minutes).
models/.venv/bin/python models/train_sparse_autoencoder.py

# Canned attention maps and activation-patching baselines.
models/.venv/bin/python models/precompute_interpretability.py

# Small MNIST U-Net denoiser (cosine schedule, EMA weights) plus a VAE with a
# two-dimensional latent, both trained on all 60,000 training digits. The run exports
# the ONNX files, then samples from pure noise with the lab's DDPM sampler and publishes
# only if an independent classifier reads those samples as digits.
models/.venv/bin/python models/train_mnist_models.py

# Download the pinned 50d GloVe release; only 10k ASCII words are retained.
gh release download glove-wiki-gigaword-50 \
  --repo RaRe-Technologies/gensim-data \
  --pattern glove-wiki-gigaword-50.gz --dir models/data
models/.venv/bin/python models/prepare_language_assets.py

# Validate ONNX graphs and numerical parity, then refresh checksums.
npm run models:verify
npm run models:manifest
```

The transformer trains on the first 90% of the corpus and reports loss on the last 10%.
About 30% of each batch is synthetic repeated strings with random periods; periods 11,
13, 17 and 19 are held out of training. The induction gate (`induction_diagnostic.py`)
then measures, on those unseen periods, each head's attention from the second copy of a
repeated string to the token after the earlier match, and the loss on the repeated half
against the first half. It runs on random strings and on held-out corpus text. A fixed
"attend N back" habit scores near zero on it, which the first release's did. A run that
does not pass `shipping_gate` in the metadata exports for inspection but exits
unsuccessfully and is not published unless `--allow-failing-gate` is given. Change the
curriculum or `--steps`; do not lower the thresholds in `induction_diagnostic.GATE`.
`npm run models:verify` re-runs the diagnostic on the exported ONNX file and fails if the
recorded verdict disagrees. The SAE cache records the window position of every example
and flags features that describe the start of a window (see `MODEL_CARD.md`).

The MNIST command fails closed as well. It publishes the three ONNX files, the schedule,
and the metadata to the Diffusion & VAEs module's `assets/` folder only when 256 images
generated from pure noise, with the same DDPM sampler the lab runs, clear the gate in
`train_mnist_models.py`: mean confidence of an independent digit classifier, all ten
classes present, and no near-copies of training digits. A failed run leaves its files and
`mnist-diffusion-samples.png` in `models/exports/` for inspection. Increase
`--diffusion-epochs` or `--base-channels`; do not lower the gate. `npm run models:verify`
also compares the exported MNIST models with their saved PyTorch weights.

## Outputs

- `tiny-transformer.onnx`: logits, Q/K/V tensors, attention patterns, and residual
  streams. The Attention module runs this in a Web Worker with WebGPU and WASM fallback.
- `character-rnn.onnx` and `bigram.json`: same-corpus comparison models for the
  Next-token prediction module.
- `residual-sae.onnx` and `sae-top-activations.json`: feature inspection for the
  Interpretability I: features module.
- `interpretability-cache.json`: fast canned attention and patching views for the
  Interpretability II: circuits module.
- `mnist-diffusion-denoiser.onnx`, `mnist-vae-encoder.onnx`, and
  `mnist-vae-decoder.onnx`: generative assets for the Diffusion & VAEs module. Their
  input and output names and shapes are a contract the module's TypeScript depends on:
  the denoiser takes `image` [batch, 1, 28, 28] and `timestep` [batch] (int64, 0 to 99)
  and returns `noise`; the encoder maps `image` to `mean` and `log_variance`
  [batch, 2]; the decoder maps `latent` [batch, 2] to `image`.
- `mnist-diffusion-schedule.json`: the 100 values of the cumulative signal product
  (alpha-bar) the lab reads. `mnist-models.metadata.json` records how the schedule, the
  denoiser, and the VAE were trained, plus offline measurements the lessons quote;
  it is the source of truth for those numbers.
- `embedding-10k.f32`, projection JSON, and `bpe-tokenizer.json`: assets for the
  Tokens & embeddings module.

`assets/manifest.json` records shipped byte size and SHA-256 for every module
asset. ONNX entries also record their input/output contract and minimum runtime
version. `npm run check:assets` fails on stale checksums or a total over 150 MiB.

## Reproducibility boundaries

Seeds, architecture, losses, the training recipe, training curves, and induction
diagnostics are written to metadata files. Device kernels can still cause small
run-to-run differences, so install the exact run that was reviewed with `--promote`
rather than retraining. ONNX parity is checked on CPU against the saved PyTorch
checkpoint, with a tolerance of `2e-4` plus `5e-6` of each tensor's largest value (the
transformer's residual stream has entries in the hundreds), before an artifact is
considered reviewable.
