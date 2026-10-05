import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Forward process": {
    title: "Noise added in closed form",
    summary:
      "One clean digit x₀ shown at six noise levels of the shipped 100-step cosine schedule, computed exactly from q(x_t | x₀) with one seeded noise image. Beside it: the schedule's signal and noise coefficients, and the shipped network's error at predicting that noise at every level.",
    whatYouSee: [
      "A strip: x₀, then the same digit at t = 0, 19, 39, 59, 79 and 99, each labelled with √ᾱ_t. The outlined tile is the level nearest the reverse process's current step.",
      "`Noise schedule`: √ᾱ_t as a solid line and √(1−ᾱ_t) as a dashed line against step t, with a marker at the reverse process's current level. The legend prints the values at t = 99: 0.031 and 1.000.",
      "`Noise-prediction error by step`: the trained network's mean squared error at predicting ε (solid) against always predicting zero (dotted), measured on this x₀ with one seeded ε.",
      "The status chip reads `ONNX · WASM` (or `WEBGPU`) once the error curve has come back from the model worker.",
    ],
    howItWorks: [
      "Every image is x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε, with pixels mapped from [0, 1] to [−1, 1] by 2x − 1, which is how `train_mnist_models.py` scales them.",
      "ᾱ_t is read from `mnist-diffusion-schedule.json`. Recovering β_t = 1 − ᾱ_t / ᾱ_(t−1) gives a cosine schedule (Nichol and Dhariwal, 2021): tiny steps at first, β = 0.0006 at t = 0, and large ones at the end, β = 0.55 at t = 99. The strip's labels run from √ᾱ = 1.00 at t = 0 to 0.03 at t = 99.",
      "The error curve runs the shipped denoiser once on a batch of 100 images, one per t, each built from the same x₀ and ε, and scores ε̂ against ε by mean squared error over 784 pixels. The dotted baseline is the mean of ε², close to 1.",
      "x₀ is the VAE decoder's output at the latent point set in the VAE view, so the two halves of the lab share one digit. If the decoder fails, a hand-drawn 0 stands in and the caption says so.",
    ],
    controls: [
      "`Noise seed` in the Reverse diffusion card changes ε here too; moving the point in the VAE view changes x₀.",
      "Worth running: read the error curve at both ends. On the default digit and seed the network scores 0.60 at t = 0, its worst but still well below guessing zero (1.03), and under 0.001 at t = 99.",
    ],
    notice: [
      "Adding noise needs no learning at all: every tile is one line of arithmetic. Only the reverse direction needs a network.",
      "The schedule ends at almost pure noise. At t = 99 the image keeps only 0.031 of the digit's amplitude, so the last tile looks like static, and that is what lets sampling start from noise alone.",
      "Predicting ε is hardest when there is little of it. At low t the noise is a faint perturbation of an unknown digit, so the network does worst there; at high t the noise dominates x_t and is read off almost exactly.",
    ],
    limits: [
      "In this lab: x₀ is a VAE decoder output rather than an MNIST image, and the error curve comes from one image and one noise draw, not the training average over digits, steps and draws. Measured offline on 10,000 test digits, the network's error is 0.13 at t = 0, 0.028 at t = 49 and under 0.001 at t = 99. The t = 0 value is higher here because the decoder's output has a soft grey background; on the hand-drawn presets, whose background is flat, it is 0.07 to 0.10.",
      "In this lab: the schedule runs 100 steps, not the 1,000 usual in DDPM, and stops at 98% of the cosine curve, so ᾱ at the last step is 0.00097 rather than exactly 0. A faint trace of the digit is left.",
      "In general: production schedules (linear over 1,000 steps, cosine, or flow-matching paths) are designed so the last step is indistinguishable from pure Gaussian noise, and some force it to exactly zero signal, which needs care at the very first sampling step.",
      "In general: the training loss averages this error over random steps, images and draws, so a single number hides how uneven it is across noise levels.",
    ],
  },

  "Reverse diffusion": {
    title: "One denoiser call per step",
    summary:
      "The DDPM update applied 100 times from a seeded start, calling a denoiser once per step. Choose the shipped ONNX network or the exact posterior-mean denoiser for 64 VAE-drawn reference digits, then scrub through the stored trajectory from pure noise to a sample.",
    whatYouSee: [
      "Three large images for the current position: x after k calls, and the next call's predicted clean image x̂₀ and predicted noise ε̂ (mid-grey is zero; black and white are ∓2.5).",
      "A `Denoiser calls` slider from 0 to 100 and a filmstrip of x after 0, 10, …, 100 calls. Each thumbnail is a button that jumps there.",
      "The next update with its numbers filled in, then four readouts: calls made, the noise level t of the current x, √ᾱ and √(1−ᾱ) at that level, and the share of background pixels in x next to the clean digit's.",
      "With the Ideal denoiser, the 64 reference digits with outlines whose opacity is each reference's posterior weight at the next call; the heaviest is filled. With the Trained network, a note on how it was trained.",
    ],
    howItWorks: [
      "Each call k works at t = 99 − k and applies x_(t−1) = (x_t − β_t / √(1−ᾱ_t) · ε̂) / √α_t + √β_t · z, with seeded z and no fresh noise on the last call. This is DDPM's ancestral sampler (Algorithm 2 with σ_t² = β_t); the training script ships no sampler of its own, and repeats this one to judge samples before publishing.",
      "The predicted clean image is x̂₀ = (x_t − √(1−ᾱ_t) · ε̂) / √ᾱ_t, the forward formula solved for x₀.",
      "`Trained network` makes 100 sequential calls to `mnist-diffusion-denoiser.onnx` in the model worker, each with the current x_t and the integer t.",
      "`Ideal denoiser` computes weights w_i = softmax(−‖x_t − √ᾱ_t r_i‖² / (2(1−ᾱ_t))) over references r_i, sets x̂₀ = Σ w_i r_i and derives ε̂ from it. This is the exact minimum-error denoiser for that finite set: what a perfectly trained network would converge to on it.",
      "`Pure noise` starts from ε; `Noised x₀` starts from √ᾱ₉₉ · x₀ + √(1−ᾱ₉₉) · ε, which at t = 99 is 0.03 · x₀ + 1.00 · ε. One seed fixes ε and every z, so both denoisers see identical randomness.",
      "Background pixels is the share of the 784 pixels at or below −0.9: about 0.17 for the seed-7 starting noise and about 0.82 for the network's sample. Pixel RMS would not do here, because noise and a digit both sit near 1.",
    ],
    controls: [
      "`Denoiser` switches between `Ideal denoiser` and `Trained network`; `Start from` between `Noised x₀` and `Pure noise`; `Noise seed` redraws ε and z (1 to 24).",
      "`Denoiser calls` and the filmstrip move through the finished trajectory without recomputing it.",
      "Worth running: with `Trained network`, `Pure noise` and seed 7, drag to 100 calls: static becomes a digit and the Background pixels readout climbs from 0.17 to 0.82. Switch to `Ideal denoiser`: it lands exactly on reference #64, while the network's sample is nearest to reference #28 at a squared distance of 0.26.",
    ],
    notice: [
      "Every step is one full network call, and step k needs the output of step k − 1. That sequential chain is the cost diffusion pays that a VAE's single decoder pass does not.",
      "With the ideal denoiser, x̂₀ is a weighted average of references: blurry while the weight is spread, one exact reference once a single one takes it. At seed 7 the top reference passes half the weight after 15 calls and 99% after 32.",
      "The same seed gives two different images: the ideal denoiser returns one of the 64 references (distance 0), the network a new digit that is none of them (its nearest reference is never closer than 0.07 across seeds 1 to 24).",
      "The ideal denoiser's choice depends on the seed: from pure noise, seeds 1 to 24 end on 20 different references. Changing the seed changes the digit the network draws too, which is the only control you have over it.",
    ],
    limits: [
      "In this lab: the ideal denoiser is exact arithmetic over 64 references, not a trained model. It can only ever return one of them, so its samples are memorized by construction.",
      "In this lab: the shipped network is small (1,181,217 parameters, 45 epochs, 28 × 28 MNIST only) and unconditional, so you cannot ask it for a digit. Offline, an independent classifier read 256 of its pure-noise samples with a mean confidence of 0.95 and over 0.9 for 86% of them, against 0.99 for real test digits; the rest are malformed or ambiguous.",
      "In this lab: the schedule is not exactly pure noise (√ᾱ = 0.031 at t = 99), so `Noised x₀` and `Pure noise` give almost the same sample. There is no guidance, no text prompt and no faster sampler: every sample takes all 100 calls.",
      "In general: good diffusion models generalize because a real network does not reach this empirical optimum; its architecture smooths between training images. Duplicated training images can still be reproduced (Carlini et al., 2023).",
      "In general: faster samplers such as DDIM and DPM-Solver take 10 to 50 larger steps, and distilled models take 1 to 4, but each step is still a network call.",
    ],
  },

  "Latent map": {
    title: "Two numbers in, one digit out",
    summary:
      "The shipped VAE decoder evaluated on a 16 × 16 grid of latent points from −4 to 4, with a draggable point whose digit is decoded at full size. Dashed rings mark the N(0, I) prior at 1σ and 2σ, so you can see which parts of the prior the decoder draws well.",
    whatYouSee: [
      "A mosaic of 256 decoder outputs; each tile is decoded at its own centre, z₁ running left to right and z₂ bottom to top, with ticks at −4, −2, 0 and 2.",
      "Dashed circles at radius 1 and 2 around the origin, the pink point for z, and a blue diamond with a 2σ ellipse for the latest encoded drawing.",
      "The digit decoded at z, drawn large, with `Latent z₁` and `Latent z₂` sliders and two readouts: ‖z‖ in prior standard deviations, and log p(z) under N(0, I) in nats.",
    ],
    howItWorks: [
      "The decoder is an MLP, 2 → 96 → 256 → 784 with SiLU activations and a sigmoid output, so each pixel is a value in [0, 1]. The mosaic is one batched call with 256 latents; the large digit is one call with one.",
      "log p(z) = −½‖z‖² − ln 2π for the two-dimensional standard normal prior; at the default z = (−2.00, −0.80) it is −4.16 nats.",
      "The VAE was trained with binary cross-entropy summed over pixels plus 0.35 × KL, for 40 epochs on all 60,000 MNIST training digits (see `mnist-models.metadata.json`).",
    ],
    controls: [
      "Drag on the map, or focus it and use the arrow keys (0.1 per press, 0.5 with Shift). `Latent z₁` and `Latent z₂` do the same one axis at a time, in steps of 0.05.",
      "Worth running: start on the 0s along the bottom, go up the right side through the 2s to the 1s at the top right, then along the top to the 7s at the upper left. The digit morphs smoothly and blurs where two regions meet; the default point sits among the 5s on the left.",
    ],
    notice: [
      "Nearby points decode to similar digits: the map is continuous, which is what makes a two-number code navigable.",
      "Each digit owns a region, and the regions meet in the middle. Several classes crowd there: eight of the ten digits appear among the 12 tiles within radius 1. An independent classifier reads those tiles with a mean confidence of 0.70, and the tiles beyond radius 3 at 0.96.",
      "Because the KL weight is 0.35, the codes spread wider than the prior: the 10,000 MNIST test digits have standard deviations of 1.41 and 1.55 on the two axes, and 15% land beyond radius 3, where N(0, I) puts 1.1%.",
      "So a draw from the prior mostly lands in the crowded middle. Offline, 48% of 10,000 prior draws decode to a digit the classifier reads with confidence above 0.9, against 86% of the diffusion network's pure-noise samples.",
    ],
    limits: [
      "In this lab: the latent space is deliberately two-dimensional so it can be drawn. That forces heavy overlap between classes and blurry digits.",
      "In this lab: the offline measurements (where test digits land, what the classifier reads) are quoted from `mnist-models.metadata.json`, not recomputed, because the lab ships no MNIST images and no classifier.",
      "In general: a sample from the prior only decodes well if the encoder used that region. A KL weight below 1, as here, lets the codes drift from the prior; practical VAEs use tens to thousands of latent dimensions.",
    ],
  },

  "Encode a drawing": {
    title: "Compress, sample, reconstruct",
    summary:
      "A 28 × 28 drawing or preset goes through the shipped encoder to a posterior mean and variance, then back through the decoder, once from the mean and once from a sampled latent. The card prints both terms of the training objective for this one image.",
    whatYouSee: [
      "A drawing pad with preset buttons `0`, `1`, `3` and `7` and a `Clear` button; the pressed preset is highlighted.",
      "Two reconstructions: `decoder(μ)` and `decoder(μ + σ·ε)`, the second using a seeded ε.",
      "Five rows: the posterior mean μ, the posterior σ, the reconstruction BCE, the KL to the prior, and the training loss BCE + 0.35 × KL, all for this one image.",
      "The encoded mean appears on the Latent map as a blue diamond with a 2σ ellipse.",
    ],
    howItWorks: [
      "Strokes are rasterized with a soft round brush of radius 1.5 pixels into white-on-black values in [0, 1], the MNIST convention. Presets are hand-authored strokes, not dataset images.",
      "The encoder, 784 → 256 → 96 with SiLU, outputs μ and log σ² for each of the two dimensions; σ = exp(½ log σ²). The sampled latent is z = μ + σ ⊙ ε, the reparameterization used in training.",
      "BCE = −Σ[x ln x̂ + (1−x) ln(1−x̂)] over 784 pixels, with PyTorch's clamp of each log at −100. KL = −½ Σ(1 + log σ² − μ² − σ²). The loss weights KL by 0.35, as `train_mnist_models.py` does.",
      "The BCE row scores decoder(μ); the note below it scores the sampled reconstruction, which is what training actually used.",
    ],
    controls: [
      "Draw with a mouse, pen or touch; each stroke re-encodes when you lift. The preset buttons are the keyboard route.",
      "Worth running: load `7` and then `3`. The 7 encodes to μ = (1.55, 0.02) and decodes as a 3, with BCE 211.6 nats and KL 6.26. The 3 encodes to μ = (1.46, −0.32), decodes as a blurry 3, and scores BCE 121.1 and KL 7.12.",
    ],
    notice: [
      "σ is about 0.03 to 0.05 while the prior's is 1, so the encoder is confident about where each drawing goes; the KL term is what charges it for that.",
      "Reconstructions are blurry and sometimes the wrong digit. Two numbers cannot hold stroke detail, and a pixel-wise reconstruction loss rewards hedging between plausible digits. The 7 preset, with its heavy top bar, lands among the 3s.",
      "decoder(μ) and decoder(μ + σ·ε) look nearly identical here, because σ is small next to the spacing between digits on the map.",
    ],
    limits: [
      "In this lab: drawings are not preprocessed the way MNIST was (size-normalized into a 20 × 20 box and centred by mass), so unusual placement or thickness encodes worse than a dataset digit would.",
      "In this lab: the numbers describe one image, not the dataset average that training minimized. Training's last epoch averaged 133.3 nats of BCE and 8.03 of KL; on the 10,000 test digits, 75% of reconstructions are read by the classifier as the original digit.",
      "In general: with a KL weight of 0.35 this objective is a β-VAE loss rather than the exact negative ELBO, which needs weight 1. Lower weights buy sharper reconstructions at the cost of a less prior-shaped latent space.",
      "In general: image generators that use a VAE, such as latent diffusion, keep a spatial latent (for example 4 channels at 1/8 the resolution), not two numbers.",
    ],
  },
};

export default cardInfo;
