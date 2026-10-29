import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. The `step` setting is the Denoiser calls slider, from
 * 0 to 100, not the shell step. The Diffusion or VAE view follows the shell step (steps 0 and 1 show
 * Diffusion, 2 and 3 show VAE) and is not a setting a scene can change, so every scene names a step.
 * The defaults are seed 7 from pure noise with the trained network, the latent point (-2, -0.8), and
 * the 3 preset in Encode a drawing, which are the setups the lesson quotes.
 */
const scenes: LabScene[] = [
  {
    id: "packed-codes",
    sections: ["Keeping the codes packed together"],
    title: "The centre of the cloud",
    look: "The point now sits at the centre of the two dashed circles, where random picks land most often and several digits’ regions crowd together.",
    step: 2,
    state: { latentX: 0, latentY: 0 },
    focus: "Latent map",
  },
  {
    id: "best-guess-spot",
    sections: ["A best-guess spot with a small wobble", "The VAE: compress, then decode"],
    title: "A spot and its wobble",
    look: "Read Posterior mean μ and Posterior σ, then compare decoder(μ) with decoder(μ + σ·ε): the wobble is too small to see.",
    step: 3,
    state: { sample: "three" },
    focus: "Encode a drawing",
  },
  {
    id: "reverse-update",
    sections: ["Cleaning up: the reverse process", "Diffusion: destroy, then learn to undo"],
    title: "One call, worked out",
    look: "The update line under the filmstrip fills in the next call’s real numbers, and they change as you move Denoiser calls.",
    step: 1,
    focus: "Reverse diffusion",
  },
  {
    id: "perfect-denoiser",
    sections: ["A perfect denoiser to compare against"],
    title: "The ideal denoiser’s copy",
    look: "After the last call, the summary line under the filmstrip names the reference the ideal denoiser landed on, with no distance left between them.",
    step: 1,
    state: { denoiser: "ideal", step: 100 },
    focus: "Reverse diffusion",
  },
  {
    id: "watch-static",
    sections: ["Watch static being added", "Diffusion"],
    title: "Static, step by step",
    look: "Read the √ᾱ number under each tile as the digit fades, then compare the solid and dashed lines on the Noise schedule chart below.",
    step: 0,
    focus: "Forward process",
  },
  {
    id: "digit-from-static",
    sections: ["Watch a digit come out of static"],
    title: "A digit from static",
    look: "Drag Denoiser calls slowly to the end and watch Background pixels climb as the black background comes back.",
    step: 1,
    state: { step: 0 },
    focus: "Reverse diffusion",
  },
  {
    id: "ideal-from-zero",
    sections: ["Compare with the perfect denoiser"],
    title: "Weight gathers on one reference",
    look: "Drag Denoiser calls up from 0 and watch the outlines in the reference grid gather onto one highlighted reference.",
    step: 1,
    state: { denoiser: "ideal", step: 0 },
    focus: "Reverse diffusion",
  },
  {
    id: "latent-walk",
    sections: ["Walk around the latent map", "VAE"],
    title: "Walk the latent map",
    look: "Drag the point or use Latent z₁ and Latent z₂, and watch the big digit, ‖z‖ and log p(z) change as you go.",
    step: 2,
    focus: "Latent map",
  },
  {
    id: "squeeze-drawing",
    sections: ["Squeeze your own drawing"],
    title: "Squeeze a drawing",
    look: "Read the BCE, KL and Training loss rows for the 3, then click 7 or draw your own digit and read them again.",
    step: 3,
    state: { sample: "three" },
    focus: "Encode a drawing",
  },
  {
    id: "seed-seven",
    sections: ["Seed 7 gives one new digit and one copy", "What to notice"],
    title: "A new digit, not a copy",
    look: "After the last call, the summary line names the network’s nearest reference and a distance above zero, so switch Denoiser to Ideal denoiser and compare.",
    step: 1,
    state: { step: 100 },
    focus: "Reverse diffusion",
  },
  {
    id: "error-by-step",
    sections: ["Guessing static is hardest when there is little of it"],
    title: "Error across the steps",
    look: "On the Noise-prediction error by step chart, the solid line starts highest at step 0 and sinks toward zero on the right, always under the dotted line.",
    step: 0,
    focus: "Forward process",
  },
  {
    id: "seven-as-three",
    sections: ["Two numbers is a tight squeeze"],
    title: "A 7 that comes back as a 3",
    look: "This 7 comes back as a 3 in decoder(μ), and its Reconstruction BCE is far higher than the 3 preset’s.",
    step: 3,
    state: { sample: "seven" },
    focus: "Encode a drawing",
  },
];

export default scenes;
