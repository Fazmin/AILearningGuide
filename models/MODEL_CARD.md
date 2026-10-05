# Teaching asset model card

These models exist to expose mechanisms in an offline learning application. They are
not suitable for open-ended language generation, handwriting recognition, or
production decisions.

## Character transformer

- Architecture: two decoder-only transformer blocks, width 256, four heads, context
  64, GELU MLP width 768; 66 symbols (65 characters and `<unk>`); 1,367,552 parameters.
  Identical to the first release, so the ONNX contract did not change.
- Data: Tiny Shakespeare. The first 90% of the corpus (1,003,854 characters) trains the
  model; the last 10% (111,540 characters) is a held-out tail used for every loss figure
  below and for the corpus-text half of the induction gate.
- Recipe: 8,000 steps, batches of 64 windows of 64 characters (45 corpus windows and 19
  synthetic windows), AdamW at 1e-3 with a 300-step warm-up and cosine decay, seed 1337,
  Apple MPS. The recipe is recorded in `language-models.metadata.json`.
- Synthetic repeated-string windows: a random string of a random period (4 to 31
  characters, 24 values) repeated to fill the window, after an optional unrelated
  prefix. Content is random symbols (35%), random distinct symbols (25%) or a snippet of
  the training text (40%). Periods 11, 13, 17 and 19 are never used in training.
- Language-model loss, per character on the held-out tail: 1.543 nats (2.23 bits). For
  comparison, a uniform guess is 6.04 bits, an add-one bigram fitted on the training
  split scores 3.58 bits, and the GRU scores 3.04 bits. The earlier release scored 2.54
  nats (3.67 bits) on the same text. The transformer fits its training text better than
  the tail (1.24 against 1.54 nats), so it has memorised some of it: on the lab's default
  prompt, `O Romeo, Romeo! wherefore art thou R`, it predicts `o` with 99.97% confidence
  because that line is in its training data.
- Intended views: next-character logits, per-layer Q/K/V, attention maps, residual
  activations, and controlled patching.

### What the induction diagnostic measures

The first release trained on strings that all repeat every 8 characters and tested
strings that repeat every 16 (a multiple of 8). The model had learned a fixed "attend
7 positions back" look-up, which the test scored as induction (best score 0.39 against a
0.20 gate), so the gate could not fail. The diagnostic in `induction_diagnostic.py` now
tests the standard way, on repeat periods the model never trained on.

Take a random string of length P and repeat it once. A genuine induction head, reading a
token in the second copy, finds that token's earlier copy and attends to the token after
it: an attention stripe at offset P - 1. A fixed look-back head draws its stripe at the
same offset whatever P is. The diagnostic scores:

1. Per head, the mean attention from the second copy to offset P - 1, for P in 11, 13,
   17 and 19, on random strings of distinct symbols and on snippets of held-out corpus
   text.
2. The loss on the repeated half against the first half of the same sequences.

Shipping gate, fixed before the retrained model was measured and never loosened: at least
one head with a stripe score of 0.40 or more on random strings and 0.25 or more on corpus
text for every held-out period; and, for every held-out period on both kinds of text, a
repeated-half loss of at most 1.0 nats and a first-to-second-half loss drop of at least
2.0 nats (random strings) or 1.0 nats (corpus text). `check_induction_gate.py` asserts
that the gate passes an idealised induction pattern and fails both an idealised fixed
7-back pattern and the first release's ONNX file. `npm run models:verify` re-runs the
diagnostic on the exported file and compares its verdict with the metadata.

### What the model does and does not do

- It passes the gate. Layer 1 heads 3 and 4 attend to the previous character (0.99 or
  more on every diagnostic input), and layer 2 heads 2 and 3 attend to the character
  after the earlier match: 0.81 to 0.86 of their attention on random strings at every
  held-out period, 0.40 to 0.55 on corpus text. On random strings the loss falls from
  about 4.3 nats in the first copy to 0.13 to 0.25 in the second. Zero-ablating layer 2
  head 3 raises the repeated-half loss on random strings from 0.19 to 2.6 nats.
- The behaviour is induction, not a positional habit: the stripe moves with the period
  (offset 7 at period 8, offset 8 at period 9), and the copy also works for two strings
  separated by unrelated filler, a layout no training window had (loss 0.04 to 0.23 nats
  on the second string against about 4.3 on the first, random strings). The layer 1 previous-token heads are the likely partners of the layer 2 induction
  heads, but this was not tested by path patching. Zero-ablating layer 1 head 2, which
  attends two characters back (at least 0.98 of its attention at every tested period) and
  is not a previous-token head, hurts copying as much as ablating head 4, so ablation alone
  does not isolate the circuit.
- Only those heads are characterised. Nothing here says what layer 1 head 1, layer 1
  head 2, layer 2 head 1 or layer 2 head 4 compute; their patterns are cached and
  displayed, not explained.
- It is a character model of one author's plays. It does not follow instructions, hold
  facts or generalise beyond Shakespeare-shaped spelling, and its completions are not
  representative of modern language models. Its residual stream has a few very large
  entries (up to about 150 on corpus text and about 250 on the verification input of
  random characters, against about 13 for the first release), which is why ONNX parity is
  checked with a scale term.
- Shipping gate (all of): 1 to 2 million trainable parameters, a finite held-out loss
  below the bigram's, the induction gate above (the metadata records each check under
  `shipping_gate`), and ONNX parity against the PyTorch checkpoint within `2e-4` plus
  `5e-6` of each tensor's largest value. On the verification input the largest absolute
  error is 2.1e-4 on the residual stream (largest value about 250, relative error about
  1e-6) and 1.1e-5 or less on the other five outputs. The first release's gate used a
  bare `2e-4` bound; with residual entries in the hundreds, one float32 rounding step is
  already about 3e-5, so a bare bound would test arithmetic noise, not the export.

## Character RNN and bigram table

Both use the same character vocabulary and corpus as the transformer. They support
controlled architectural comparisons, not quality claims. The add-one-smoothed
bigram probabilities contain no context beyond one character.

The two-layer GRU (470,082 parameters, 240 steps at batch 24) and the bigram table are
unchanged from the first release. `train_language_models.py` reuses the GRU's saved
checkpoint by default (`--rnn reuse`) so its ONNX file and the figures quoted for it stay
reproducible; `--rnn retrain` trains a new one. Re-scored on every 64-character window of
the corpus, the GRU averages 2.12 nats (3.06 bits), the bigram 2.45 nats (3.54 bits) and
the transformer 1.27 nats (1.83 bits, mostly training text). On the held-out tail they are
2.11, 2.48 (bigram fitted on the training split) and 1.54 nats.

## Sparse autoencoder

The SAE trains on normalized residual activations from transformer layer two (the stream
after the second block), 1,024 features with the 32 largest kept per token, on 1,600
random windows from the training split (102,400 token rows) for 10,000 steps. On 200
held-out windows it leaves 5.1% of the variance unexplained (4.6% on its training
windows). Every feature is active somewhere in the 1,800 scanned windows. Feature labels are hypotheses based on cached top
activations, never ground truth. Split and polysemantic features are expected in a model
this small.

Window start artefact. At window position 0 the residual depends only on the single
character there, and at positions 1 to 3 on a handful, and the transformer's residual is
unusually large there. Features whose largest activations come from those positions
describe the start of a window, not language. Every cached example records its
`window_position`. No feature fires only at position 0, but 18 of the 128 cached features
have most of their top examples at position 0 and 21 within the first four positions;
they carry `position_zero_artefact` and `window_start_artefact` flags and an
`examples_in_context` list drawn from later positions. The strongest feature by maximum
(number 876, 56.9) is such an artefact, firing at position 1. Leaving the first four
positions out of training (`--skip-window-start 4`) made this worse (27 position 0
artefacts among the 128 against 16 at the same training length), because those rows still
reach the encoder when the lab feeds it text.

Cache excerpts replace line breaks with spaces (`example_records` in
`train_sparse_autoencoder.py`), so a feature that fires on line breaks, such as numbers 53
and 264, looks like a space feature when read from the cache. Test such labels on live text,
where a line break can be typed. The cache was not regenerated for this, because a rerun
would change feature numbers and every figure pinned to them. Decoder directions are not
all independent: the most similar pair (numbers 108 and 499, cosine -0.992) is one signed
direction represented by two ReLU features pointing opposite ways, not a duplicate.

The earlier release's feature 624 was this artefact: its eight cached examples were all
the first character of a window, with an identical activation. That SAE belonged to the
earlier transformer; features are renumbered by retraining.

## Interpretability cache

`interpretability-cache.json` holds attention maps for four canned prompts and two
activation-patching pairs, each recorded with its clean and corrupted target logits and
their gap (`logit_gap`). The `speaker` pair has a gap of 8.43 logits. The `copy` pair is
rebuilt: a period-13 string of capital letters, with the letter that induction should
copy replaced in the corrupted run; its gap is 11.99 logits, and over 64 random
period-13 strings it averages 11.3 and never falls below 3.3. The earlier `copy` pair
changed a token whose successor was a space either way, so its gap was 3e-5 logits and
its recovery figures meant nothing.

## MNIST VAE and diffusion denoiser

Both models train on all 60,000 MNIST training digits (28 × 28, grey-scale). The data
comes from `https://yann.lecun.com/exdb/mnist/`, which says only that MNIST is derived
from NIST data; mirrors state different terms, so the dataset's licence is not
established from a primary source here. See the upstream project before redistributing
anything built from it. The shipped weights contain no images, but the metadata and the
lessons quote offline measurements made on the MNIST test split.

**VAE.** A 452,628-parameter MLP with a deliberately constrained two-dimensional
latent, trained 40 epochs on binary cross-entropy plus 0.35 times KL. The weight below 1
makes it a β-VAE objective, not the exact negative ELBO, and the test digits' codes
spread wider than the prior (standard deviation about 1.4 and 1.6 per axis; 15% beyond
radius 3, where N(0, I) puts 1.1%). Sampling from the prior is therefore poor: an
independent classifier reads only 48% of 10,000 prior draws with confidence above 0.9,
and decoder(mean) is read as the original digit for 75% of test images. These figures
are in `mnist-models.metadata.json` under `vae.test_digits`, together with the
classifier's label for each of the 16 × 16 latent-map tiles that the lab draws.

**Denoiser.** A 1,181,217-parameter convolutional U-Net (28, 14 and 7 pixel
resolutions, 32, 64 and 128 channels, six residual blocks, sinusoidal step embedding)
trained with the epsilon objective for 45 epochs, batch 128, AdamW at 1e-3 with cosine
decay, and an exponential moving average of the weights (decay 0.999), which is what
ships. It is unconditional: the ONNX contract has no class input, so it cannot be asked
for a particular digit and there is no guidance. The forward process is a 100-step cosine
schedule (offset 0.008, 98% of the curve), so the last step keeps √ᾱ = 0.031 of the
signal, nearly pure noise. This replaced the first release's linear schedule, which
ended at √ᾱ = 0.60, and its 772,192-parameter MLP trained for 3 epochs on 20,000 digits,
whose samples from pure noise stayed noise (an independent classifier read them with a
mean confidence of 0.63, the same as for Gaussian noise itself, 0.59).

- Shipping gate (`train_mnist_models.py`): 256 images generated from pure noise with the
  lab's DDPM sampler (100 ancestral steps, sigma squared equal to beta) must be read by an
  independent CNN classifier (98.8% on the test split) with a mean confidence of at least
  0.90, with all ten digits present, and with a mean distance to the nearest training
  digit of at least 0.8 times that of real test digits. The shipped model reads 0.95, all
  ten digits (17 to 31 each), 86% of samples above 0.9 (real test digits: mean 0.99),
  and a mean nearest-training distance of 4.45 against 4.11 for real digits. Only 2% of
  samples sit within 1.5 of a training digit, against 4% of real digits; the closest are
  thin 1s. The first version of the gate used a fixed 1.5 floor on the closest sample,
  which the real test digits themselves fail (3% of them are closer), so novelty is
  judged relative to real digits.
- ONNX parity against the saved PyTorch weights is checked in the training script and
  again by `npm run models:verify` (largest difference 7e-6 for the denoiser).
- Limitations: a small unconditional model on one 28 × 28 dataset. About one in seven
  samples is read with confidence below 0.9 (malformed or ambiguous digits). Sample
  quality was judged by a classifier and by eye, not by FID or likelihood. The schedule
  does not reach exactly zero signal, so `Noised x₀` and `Pure noise` starts give nearly
  the same sample. The lab's reference set for the ideal denoiser is 64 VAE-decoded
  digits, so the "ideal" run reproduces VAE-blurred images, not MNIST.

## Embeddings and tokenizer

The embedding subset retains 10,000 ASCII words from a user-supplied GloVe source,
with both PCA and UMAP projections. Two-dimensional distance is a lossy view of the
original vectors. The BPE tokenizer trains only on the teaching corpus and should not
be compared with production tokenizer coverage.
