import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

// No "mode" here: until the learner picks one, the view follows the lesson step.
const initialState: ModuleState = {
  step: 62,
  latentX: -2,
  latentY: -0.8,
  seed: 7,
  denoiser: "network",
  start: "noise",
  sample: "three",
};

const clampNumber = (value: unknown, fallback: number, low: number, high: number, round = false) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  const bounded = Math.min(high, Math.max(low, numeric));
  return round ? Math.round(bounded) : bounded;
};

const oneOf = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;

/**
 * Version 1 stored the same keys with latents on [-1, 1] and "step" as a
 * denoising percentage; both still read correctly as a latent point and a
 * number of calls out of 100, so they are clamped rather than rescaled.
 */
export function hydrateDiffusionState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    const next: ModuleState = {
      ...initialState,
      step: clampNumber(parsed.step, 62, 0, 100, true),
      latentX: clampNumber(parsed.latentX, -2, -4, 4),
      latentY: clampNumber(parsed.latentY, -0.8, -4, 4),
      seed: clampNumber(parsed.seed, 7, 1, 24, true),
      denoiser: oneOf(parsed.denoiser, ["ideal", "network"], "network"),
      start: oneOf(parsed.start, ["digit", "noise"], "noise"),
      sample: oneOf(parsed.sample, ["zero", "one", "three", "seven", "custom"], "three"),
    };
    if (parsed.mode === "diffusion" || parsed.mode === "vae") next.mode = parsed.mode;
    return next;
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-14-diffusion-vaes",
  slug: "diffusion-vaes",
  title: "Diffusion & VAEs",
  group: "frontiers",
  order: 31,
  icon: "WandSparkles",
  accent: "#c05b91",
  prerequisites: ["module-03-stacking-neurons"],
  estimatedMinutes: 16,
  steps: ["Add noise", "Denoise a sample", "Decode a digit", "Encode a drawing"],
  stepInstructions: [
    "In Forward process, read the strip from x₀ to t = 99 and the Noise schedule chart, and note that √ᾱ falls to 0.03 at the last step, so the last tile is noise.",
    "In Reverse diffusion, keep Denoiser on Trained network and Start from on Pure noise, drag Denoiser calls from 0 to 100, then switch Denoiser to Ideal denoiser with the same Noise seed and compare the two final samples.",
    "Switch Model to VAE and drag the point across the Latent map, or use Latent z₁ and Latent z₂, and watch the digit change and blur where two digit regions meet.",
    "In Encode a drawing, load the 3 preset or draw a digit, then compare decoder(μ) with your drawing and read the BCE and KL rows.",
  ],
  stateVersion: 2,
  tagline: "Compare two generative models on real MNIST assets: a diffusion sampler that calls its denoiser once per step for 100 steps, and a VAE that turns two latent numbers into a digit in one decoder pass.",
  objectives: ["Contrast iterative denoising with latent reconstruction","Interpret a continuous latent space"],
  glossary: [
    {
      term: "Latent",
      definition:
        "A compressed code for a data point. This lab's VAE uses two numbers per image, so every code is a point on the map, and nearby codes decode to similar images.",
    },
    {
      term: "Diffusion",
      definition:
        "A generative method that learns to reverse gradual Gaussian noising. Training teaches a network to predict the noise added at a random step; generation applies that network once per step, starting from noise.",
    },
    {
      term: "Noise schedule",
      definition:
        "The fixed variances beta_t added at each forward step. Because every step is Gaussian, x_t has a closed form with signal sqrt(alpha-bar_t); this lab's 100-step cosine schedule keeps only 0.03 of the signal at its last step, so the last step is almost pure noise.",
    },
    {
      term: "Denoiser",
      definition:
        "The network that takes a noisy image and its step index and predicts the noise in it. One set of weights serves every noise level; the shipped one is a 1,181,217-parameter convolutional U-Net trained for 45 epochs on MNIST.",
    },
    {
      term: "Reverse process",
      definition:
        "Generation: start from noise and call the denoiser once per step, each time removing part of the predicted noise and, except on the last step, adding a little fresh noise. The calls run in sequence, which is why sampling is slow.",
    },
    {
      term: "VAE",
      definition:
        "A variational autoencoder: an encoder and decoder trained together on a reconstruction term plus a KL term that pulls each code's distribution toward a standard normal prior.",
    },
    {
      term: "Encoder",
      definition:
        "The network mapping an input to a latent distribution: a mean and a log-variance for each latent dimension, rather than a single point.",
    },
    {
      term: "Decoder",
      definition:
        "The network turning a latent code into an image in one forward pass. It only learns to draw for codes the encoder actually produced in training, so unused regions of the prior decode to junk.",
    },
    {
      term: "Reparameterization",
      definition:
        "Writing a latent sample as z = mu + sigma times epsilon, with epsilon drawn separately, so the randomness becomes an input and gradients can flow through mu and sigma.",
    },
    {
      term: "KL divergence",
      definition:
        "Here, how far the encoder's Gaussian sits from the standard normal prior, in nats. It pulls codes toward the prior; this model weights it by only 0.35, so its codes still spread wider than the prior.",
    },
    {
      term: "Latent interpolation",
      definition:
        "Walking a straight line between two latent codes and decoding along the way. Gradual change suggests a well-organized space; a blurry blend partway marks where one region hands over to another, or a hole.",
    },
    {
      term: "Classifier-free guidance",
      definition:
        "Steering a conditional diffusion model by extrapolating from its unconditional noise prediction toward its conditional one. Large guidance scales produce oversaturated, over-contrasted images.",
    },
    {
      term: "Latent diffusion",
      definition:
        "Running diffusion inside a VAE's compressed latent space instead of on pixels, then decoding once at the end. Stable Diffusion's autoencoder shrinks each image side by a factor of 8.",
    },
    {
      term: "Flow matching",
      definition:
        "Training a network to predict a velocity field that carries noise to data, often along nearly straight paths. It can be sampled in fewer steps and underlies several recent image models.",
    },
  ],
  references: [
    {
      authors: "Simon J. D. Prince",
      title: "Understanding Deep Learning",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://udlbook.github.io/udlbook/",
      note: "Chapter 17 builds the VAE step by step, from latent variables through the ELBO to the reparameterization trick. Chapter 18 covers diffusion's forward process, its reverse process, the noise-prediction loss, the U-Net denoiser, and classifier-free guidance, the same pieces this lesson walks through.",
    },
    {
      authors: "Diederik P. Kingma and Max Welling",
      title: "Auto-Encoding Variational Bayes",
      source: "2nd International Conference on Learning Representations (ICLR 2014)",
      year: 2014,
      url: "https://arxiv.org/abs/1312.6114",
      note: "The paper that introduced the VAE. It trains an encoder and a decoder together on a lower bound (the ELBO) and uses the reparameterization z = mu + sigma times epsilon so gradients can reach the encoder, as in the Encode a drawing card.",
    },
    {
      authors: "Irina Higgins, Loic Matthey, Arka Pal, et al.",
      title: "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework",
      source: "5th International Conference on Learning Representations (ICLR 2017)",
      year: 2017,
      url: "https://www.semanticscholar.org/paper/a90226c41b79f8b06007609f39f82757073641e2",
      note: "Puts a weight beta on the VAE's KL term; beta = 1 is the plain VAE. The paper raises beta above 1 for tidier codes and notes that reconstructions get worse. This lab goes the other way, with a KL weight of 0.35 for sharper copies and a looser latent map.",
    },
    {
      authors: "Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli",
      title: "Deep Unsupervised Learning using Nonequilibrium Thermodynamics",
      source: "Proceedings of the 32nd International Conference on Machine Learning (ICML 2015), PMLR 37, 2256–2265",
      year: 2015,
      url: "https://proceedings.mlr.press/v37/sohl-dickstein15.html",
      note: "The paper that first proposed diffusion models. It slowly destroys the structure in data with a fixed forward process and learns a reverse process that restores it, the destroy-then-undo idea behind the Forward process and Reverse diffusion cards.",
    },
    {
      authors: "Jonathan Ho, Ajay Jain, and Pieter Abbeel",
      title: "Denoising Diffusion Probabilistic Models",
      source: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 6840–6851",
      year: 2020,
      url: "https://arxiv.org/abs/2006.11239",
      note: "The DDPM paper. It gives the closed form for a noised image at any step, trains the network to predict the added noise, and lists the step-by-step sampling update this lab's Reverse diffusion card runs once per Denoiser call.",
    },
    {
      authors: "Alexander Quinn Nichol and Prafulla Dhariwal",
      title: "Improved Denoising Diffusion Probabilistic Models",
      source: "Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139, 8162–8171",
      year: 2021,
      url: "https://proceedings.mlr.press/v139/nichol21a.html",
      note: "Introduces the cosine noise schedule that this lab's Noise schedule chart uses. Its section on the schedule shows that the older linear schedule destroys small images too fast and why a gentler drop in the signal works better.",
    },
    {
      authors: "Olaf Ronneberger, Philipp Fischer, and Thomas Brox",
      title: "U-Net: Convolutional Networks for Biomedical Image Segmentation",
      source: "Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015), LNCS 9351, 234–241",
      year: 2015,
      url: "https://arxiv.org/abs/1505.04597",
      note: "The original U-Net, built for medical images. Its shrinking path that gathers context and matching growing path that restores detail is the design of this lab's denoiser, which sees the digit at several sizes, from fine strokes up to its overall shape.",
    },
    {
      authors: "Zahra Kadkhodaie, Florentin Guth, Eero P. Simoncelli, and Stephane Mallat",
      title: "Generalization in diffusion models arises from geometry-adaptive harmonic representations",
      source: "12th International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2310.02557",
      note: "Trains U-Net denoisers on training sets of growing size. With few images they reproduce their training images. With enough, two networks trained on separate images turn the same starting noise into nearly the same new image, one unlike their training images. It backs the lesson's point that the Ideal denoiser copies while a trained network draws new digits.",
    },
    {
      authors: "Nicholas Carlini, Jamie Hayes, Milad Nasr, et al.",
      title: "Extracting Training Data from Diffusion Models",
      source: "32nd USENIX Security Symposium (USENIX Security 23), 5253–5270",
      year: 2023,
      url: "https://www.usenix.org/conference/usenixsecurity23/presentation/carlini",
      note: "The study the lesson cites in Where it breaks. The authors pull over a thousand training images back out of large image diffusion models and find that images repeated many times in the training data are the ones most often copied.",
    },
    {
      authors: "Robin Rombach, Andreas Blattmann, Dominik Lorenz, et al.",
      title: "High-Resolution Image Synthesis With Latent Diffusion Models",
      source: "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022), 10684–10695",
      year: 2022,
      url: "https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html",
      note: "The latent diffusion paper. It runs diffusion inside a pretrained autoencoder's smaller latent grid, compares how much to shrink each side (shrinking by 4 to 8 works well), and adds cross-attention so a text prompt can steer the denoiser.",
    },
    {
      authors: "Jonathan Ho and Tim Salimans",
      title: "Classifier-Free Diffusion Guidance",
      source: "arXiv preprint arXiv:2207.12598; short version at the NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications",
      year: 2022,
      url: "https://arxiv.org/abs/2207.12598",
      note: "Trains one model both with and without the condition and pushes its prediction past the conditional one, the trick behind the lesson's guidance strength example. Its figures show strongly guided samples with saturated colors, the scorched look the lesson warns about.",
    },
    {
      authors: "Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, et al.",
      title: "Flow Matching for Generative Modeling",
      source: "11th International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2210.02747",
      note: "Introduces flow matching: train a network to predict a velocity that carries noise to data. With its optimal-transport paths each sample travels along a straight line, which the paper finds trains faster and samples in fewer steps than diffusion paths.",
    },
    {
      authors: "Patrick Esser, Sumith Kulal, Andreas Blattmann, et al.",
      title: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 12606–12633",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/esser24a.html",
      note: "Stability AI's paper on rectified flow, which joins noise and data along straight lines, scaled up to an 8-billion-parameter text-to-image model. It finds rectified flow holds up better than standard diffusion setups when sampling with fewer steps.",
    },
    {
      authors: "Tim Salimans and Jonathan Ho",
      title: "Progressive Distillation for Fast Sampling of Diffusion Models",
      source: "10th International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2202.00512",
      note: "Teaches a new model to do two sampler steps in one, then repeats, halving the step count each round. Samplers that took thousands of steps come down to as few as 4, the distillation the lesson mentions for closing the speed gap.",
    },
    {
      authors: "Shen Nie, Fengqi Zhu, Zebin You, et al.",
      title: "Large Language Diffusion Models",
      source: "Advances in Neural Information Processing Systems 38 (NeurIPS 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2502.09992",
      note: "LLaDA, an 8-billion-parameter text model that masks tokens going forward and, to generate, predicts all masked positions at once over several steps instead of left to right. It reports results close to autoregressive models of similar size, so the quality gap the lesson describes is still being tested.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "In Reverse diffusion you drag Denoiser calls from the start to the end. What does generating one image with the denoiser require?",
      options: [
        "One call that maps pure noise straight to a finished digit, the way the VAE decoder maps a latent point to one",
        "A clean digit to compare against, because every reverse step needs x₀ to compute its update",
        "One call per noise level, in sequence, each using the last output and removing part of the predicted noise",
        "A separate network for each noise level, run side by side so that all of them finish at the same time",
      ],
      answer: 2,
      explanation:
        "The network predicts noise for one noise level at a time, with one set of weights for every level. The sampler calls it, uses that prediction in the DDPM update, and moves down one level, and the next call needs that result, so the calls cannot run side by side. The update never needs a clean image, although this lab can optionally start from a noised one.",
      objective: 0,
    },
    {
      prompt:
        "On the Latent map you drag the point from a clear 0 toward a clear 2, and the digit turns into a blurry blend partway across. What does the blend tell you?",
      options: [
        "Nothing taught the decoder a digit in the gap, so it blends the two digits on either side of it",
        "Neighbouring points decode to unrelated images, so the map is not continuous anywhere on the grid",
        "The decoder adds random noise to its output in that stretch, which scrambles the pixels it draws",
        "The encoder maps that stretch to a different digit class, so the decoder swaps in the wrong label",
      ],
      answer: 0,
      explanation:
        "A decoder only learns to draw for the codes the encoder actually produced in training. Between two digit regions it still returns something, because it is a smooth function, but few or no training digits sit in the gap, so it blends the two. Nearby points elsewhere on the map decode to similar digits, and the decoder itself is deterministic: the sampling noise lives on the encoder side.",
      objective: 1,
    },
    {
      prompt:
        "A VAE's training loss adds a KL term to its reconstruction term. What does the KL term do to the latent map?",
      options: [
        "It sharpens each reconstruction by penalizing every pixel the decoder gets wrong",
        "It makes the encoder deterministic, so each image maps to exactly one point on the map",
        "It forces every digit class into its own separate cluster, so that classes never overlap at all",
        "It pulls each encoder Gaussian toward the standard normal prior, so codes share one region",
      ],
      answer: 3,
      explanation:
        "The reconstruction term rewards decoding each image back accurately; the KL term charges the encoder for every code distribution that sits far from the standard normal prior. That pressure keeps the codes in one connected region where a decoder can learn to draw, at the cost of classes overlapping. With a KL weight below 1, as here, the codes still spread wider than the prior.",
      objective: 1,
    },
    {
      prompt:
        "The Ideal denoiser is the exact minimum-error denoiser for a finite set of reference digits. What does that imply about the images it generates?",
      options: [
        "They can be any new digit, because a perfect denoiser has learned what digits look like in general",
        "They are one of its references, so it memorizes by construction rather than generating anything new at all",
        "They stay blurry averages of every reference to the end, because each reference keeps a similar weight",
        "They never resemble the clean digit, because it only ever sees noise and never the references",
      ],
      answer: 1,
      explanation:
        "The exact optimum of the denoising objective on a finite set is a weighted average of that set's members. Once one reference takes most of the weight it returns that reference, so it can only reproduce what it was given; in the lab its sample sits at distance 0 from one of them. A trained network generalizes because it does not reach this optimum: its architecture and limited capacity smooth between training images, and its sample is not any reference.",
      objective: 0,
    },
    {
      prompt:
        "The Noise schedule chart shows √ᾱ falling to nearly zero by t = 99. Why does the forward process need to end at almost pure noise?",
      options: [
        "Sampling starts from a fresh noise draw, so the last forward step must look like one",
        "The denoiser can only learn if the last forward step holds no trace of any digit at all",
        "Each reverse step removes a fixed share of the noise, so more noise at the start means fewer calls",
        "The VAE decoder needs a pure-noise input to turn into a digit, as the diffusion sampler does",
      ],
      answer: 0,
      explanation:
        "Generation begins with a Gaussian draw, but the denoiser was trained on images made by the forward process. The draw is only the kind of input it knows if the forward process ends at about the same thing. If the schedule stopped with most of a digit left, sampling would feed the denoiser inputs unlike its training inputs and the samples would stay noisy. The step count, not the amount of noise, sets how many calls generation takes, and the VAE decoder takes a latent code rather than noise.",
      objective: 0,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateDiffusionState,
};

export default definition;
