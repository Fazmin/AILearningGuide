import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const DEFAULT_TEXT = "hath the king a good horse? KING HENRY: aye, he hath.";
const HYPOTHESES = ["char", "upper", "space", "newline", "start"];
const EXAMPLE_VIEWS = ["top", "context"];
/** Features with a steering label. Mirrors LABELLED_FEATURES in steering.ts (a test checks that). */
const STEERABLE = [160, 683, 305, 121, 122];
/**
 * The marker of the retrained SAE (retrained on the new transformer, features renumbered).
 * A payload without it came from the first release, where a feature id meant something else.
 */
const SAE_RELEASE = 2;

const initialState: ModuleState = {
  sae: SAE_RELEASE,
  text: DEFAULT_TEXT,
  token: 4,
  feature: 876,
  threshold: 5,
  hypothesis: "start",
  examples: "top",
  keep: 8,
  steerFeature: 160,
  steerAt: 43,
  strength: 10,
  seed: 1,
};

const clampNumber = (value: unknown, fallback: number, low: number, high: number, round = false) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  const bounded = Math.min(high, Math.max(low, numeric));
  return round ? Math.round(bounded) : bounded;
};

/**
 * Version 1 was a synthetic viewer: `feature` (1–32) seeded a hash, `threshold` was a percentage of a
 * made-up 0–1 activation, and `overlap` rotated a diagram, so a version-1 payload (recognized by its
 * `overlap` key) falls back to the defaults.
 *
 * Version 2 ran the first SAE. Retraining renumbered every feature, so the saved `feature`, `threshold`
 * and `hypothesis` (all tied to what an old feature id showed) are reset, while `text`, `token` and
 * `keep`, which mean the same thing for any SAE, carry over. Version 2 payloads have no `sae` marker.
 */
export function hydrateFeatureState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState | null;
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
    if ("overlap" in parsed) return { ...initialState };
    const current = parsed.sae === SAE_RELEASE;
    return {
      sae: SAE_RELEASE,
      text: typeof parsed.text === "string" ? parsed.text.slice(0, 64) : DEFAULT_TEXT,
      token: clampNumber(parsed.token, 4, 0, 63, true),
      feature: current ? clampNumber(parsed.feature, 876, 0, 1023, true) : 876,
      threshold: current ? clampNumber(parsed.threshold, 5, 0, 10) : 5,
      hypothesis: current && typeof parsed.hypothesis === "string" && HYPOTHESES.includes(parsed.hypothesis) ? parsed.hypothesis : "start",
      examples: current && typeof parsed.examples === "string" && EXAMPLE_VIEWS.includes(parsed.examples) ? parsed.examples : "top",
      keep: clampNumber(parsed.keep, 8, 1, 32, true),
      steerFeature: current && STEERABLE.includes(Number(parsed.steerFeature)) ? Number(parsed.steerFeature) : 160,
      steerAt: clampNumber(parsed.steerAt, 43, 0, 63, true),
      strength: clampNumber(parsed.strength, 10, 0, 15, true),
      seed: clampNumber(parsed.seed, 1, 1, 20, true),
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-15-interpretability-features",
  slug: "interpretability-features",
  title: "Interpretability I: features",
  group: "frontiers",
  order: 34,
  icon: "Microscope",
  accent: "#2c8a8e",
  prerequisites: ["module-09-transformer-block", "module-14-diffusion-vaes"],
  estimatedMinutes: 22,
  steps: ["Inspect activations", "Trade sparsity for fidelity", "Catch a window-start artefact", "Test a label", "Steer with a feature"],
  stepInstructions: [
    "In Residual to sparse code, click positions 0, 1 and 4 of the probe text and compare the 256 dense residual bars with the SAE code's 32 spikes; read the unexplained share and which of the largest features carry a window-start flag.",
    "In Sparsity and superposition, move Keep top k features from 1 to 32 and watch the unexplained share fall, then read Most similar pair and its signed cosine.",
    "In Feature evidence, keep feature #876, the first in the menu, and read the window position on each of its eight examples; then switch Examples shown to Top 8 from position 4 on and compare the characters and activations.",
    "In Feature evidence, choose #121 and set Label to test to Space; move Activation threshold from 5 down to 2 and up to 8 and watch precision and recall, then choose #511 with Same character.",
    "In Steer with a feature, keep #160 at position 43 and compare the change in log probability with the random-direction range; press Continue, then choose #683 and compare what rises with what the label predicts.",
  ],
  stateVersion: 3,
  tagline: "Run a real sparse autoencoder on a transformer's residual stream, watch a dense vector become 32 active features, test whether a feature's label survives new text and its window position, then steer the model along a feature's direction against a random control.",
  objectives: [
    "Explain superposition and sparse feature extraction",
    "Treat feature labels as hypotheses: check where their examples come from and score them as precision and recall",
    "Use steering as a causal test of a feature, judged against a random direction of the same length",
  ],
  glossary: [
    {
      term: "Superposition",
      definition:
        "Representing more features than there are dimensions by giving them almost-orthogonal, overlapping directions. It works because most features are absent from any single input.",
    },
    {
      term: "Sparse autoencoder",
      definition:
        "A separate model trained on recorded activations that rewrites each one as a few active features out of many more than the activation has dimensions, then reconstructs it. This lab's has 1,024 features for a 256-wide residual, trained on 1,600 text windows.",
    },
    {
      term: "Polysemantic neuron",
      definition:
        "A unit that activates on several apparently unrelated things. This is the normal case, and it is the direct consequence of superposition.",
    },
    {
      term: "Monosemantic feature",
      definition:
        "A direction that corresponds to a single recognizable concept. It is the goal of sparse decomposition rather than a property of raw neurons, and many SAE features fall short of it.",
    },
    {
      term: "Dictionary learning",
      definition:
        "Finding a set of directions such that each input is a sparse combination of a few of them. Sparse autoencoders apply this to a model's internals.",
    },
    {
      term: "Sparsity penalty",
      definition:
        "The training pressure toward few active features: an L1 penalty, a TopK cutoff, or a learned threshold. This lab's SAE keeps exactly the 32 largest of 1,024 per token and adds a small L1 term.",
    },
    {
      term: "Reconstruction loss",
      definition:
        "The term requiring the decoded output to match the original activation. It trades directly against sparsity, and choosing a point on that curve is a judgment call.",
    },
    {
      term: "Feature splitting",
      definition:
        "The way one feature separates into several more specific ones when a wider dictionary is trained, so there is no canonical number of features. This lab trains one width, so it cannot show splitting; its largest overlap is a different effect.",
    },
    {
      term: "Dead feature",
      definition:
        "A dictionary entry that never activates on any input, wasting capacity. Their prevalence is one practical measure of a poorly trained autoencoder; this lab's SAE has none among 1,800 scanned windows.",
    },
    {
      term: "Auto-interpretability",
      definition:
        "Using a language model to read top-activating examples and propose a label. It makes labeling scalable and inherits that model's bias toward tidy explanations.",
    },
    {
      term: "Activation threshold",
      definition:
        "The cutoff above which a feature is treated as firing. It is not a neutral choice: a high threshold hides the counterexamples that would disconfirm a label, and a low one lets false alarms in.",
    },
    {
      term: "Steering vector",
      definition:
        "A direction added to a model's activations while it runs, to push its output. With a feature's decoder direction it turns a label from a correlation into a testable causal claim, but only if the effect is compared with a random direction of the same length.",
    },
    {
      term: "Window-start artefact",
      definition:
        "A feature whose largest activations come from the first few positions of the model's window, where the residual depends on almost no context. Its top examples describe the start of a window rather than language, which the cache shows by recording each example's window position.",
    },
    {
      term: "Null control",
      definition:
        "A comparison that should show no real effect, here a random direction of the same length as the steering vector. It sets what a real effect means by showing how much any added vector changes the output.",
    },
  ],
  references: [
    {
      authors: "Nelson Elhage, Tristan Hume, Catherine Olsson, et al.",
      title: "Toy Models of Superposition",
      source: "Transformer Circuits Thread (Anthropic)",
      year: 2022,
      url: "https://transformer-circuits.pub/2022/toy_model/index.html",
      note: "Small networks that store more features than they have dimensions when the features are sparse, at the cost of some interference. This is the superposition idea behind the lesson's polysemantic neurons and the Sparsity and superposition card.",
    },
    {
      authors: "Trenton Bricken, Adly Templeton, Joshua Batson, et al.",
      title: "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning",
      source: "Transformer Circuits Thread (Anthropic)",
      year: 2023,
      url: "https://transformer-circuits.pub/2023/monosemantic-features/index.html",
      note: "Trains a sparse autoencoder on a small transformer layer and finds features cleaner than its neurons. It is the source for dictionary learning on model internals, dead features, and feature splitting as the dictionary grows.",
    },
    {
      authors: "Hoagy Cunningham, Aidan Ewart, Logan Riggs, et al.",
      title: "Sparse Autoencoders Find Highly Interpretable Features in Language Models",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2309.08600",
      note: "Uses sparse autoencoders to rebuild a language model's activations from a few active features out of many. It backs the Residual to sparse code card: the features it finds are more interpretable and monosemantic than directions found by other methods.",
    },
    {
      authors: "Leo Gao, Tom Dupré la Tour, Henk Tillman, et al.",
      title: "Scaling and evaluating sparse autoencoders",
      source: "International Conference on Learning Representations (ICLR 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2406.04093",
      note: "Introduces TopK sparse autoencoders that keep a fixed number of active features, the rule this lab's SAE uses. It studies the trade between sparsity and reconstruction that Keep top k features shows, and how to avoid dead latents.",
    },
    {
      authors: "Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, et al.",
      title: "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders",
      source: "arXiv preprint arXiv:2407.14435",
      year: 2024,
      url: "https://arxiv.org/abs/2407.14435",
      note: "The JumpReLU variant, where each feature learns its own threshold, compared with TopK and other SAEs. It backs the Going deeper point that each sparsity choice gives a different dictionary.",
    },
    {
      authors: "Steven Bills, Nick Cammarata, Dan Mossing, et al.",
      title: "Language models can explain neurons in language models",
      source: "OpenAI research paper",
      year: 2023,
      url: "https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html",
      note: "GPT-4 writes a label for each neuron from its top examples, then the label is scored by how well it predicts the neuron's activations. This is the auto-interpretability method the lesson describes.",
    },
    {
      authors: "Gonçalo Paulo, Alex Mallen, Caden Juang, et al.",
      title: "Automatically Interpreting Millions of Features in Large Language Models",
      source: "Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2410.13928",
      note: "An open pipeline that has a language model label sparse autoencoder features and scores each label on other text. It shows how labels get tested at scale, the same idea as scoring a label in Feature evidence.",
    },
    {
      authors: "Adly Templeton, Tom Conerly, Jonathan Marcus, et al.",
      title: "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet",
      source: "Transformer Circuits Thread (Anthropic)",
      year: 2024,
      url: "https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html",
      note: "Sparse autoencoders on a production model, including the Golden Gate Bridge feature. Its feature steering section clamps features to high values and checks that behavior changes as the label predicts, the test the Steer with a feature card runs.",
    },
    {
      authors: "David Chanin, James Wilken-Smith, Tomáš Dulka, et al.",
      title: "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders",
      source: "Advances in Neural Information Processing Systems 38 (NeurIPS 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2409.14507",
      note: "Shows that a feature that looks clean can fail to fire where it should because a more specific feature takes over. This is the absorption problem the lesson lists under real systems.",
    },
    {
      authors: "Gonçalo Paulo and Nora Belrose",
      title: "Sparse Autoencoders Trained on the Same Data Learn Different Features",
      source: "International Conference on Learning Representations (ICLR 2026)",
      year: 2026,
      url: "https://arxiv.org/abs/2501.16615",
      note: "Retrains SAEs with only the random seed changed and finds different features; in one Llama 3 8B setting only 30% were shared. It backs the lesson's point that an SAE's features are one useful split, not the model's own parts.",
    },
    {
      authors: "Zhengxuan Wu, Aryaman Arora, Atticus Geiger, et al.",
      title: "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders",
      source: "Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2501.17148",
      note: "A benchmark that compares ways to steer and detect concepts in Gemma 2 models. Prompting does best at steering and SAE features are not competitive, the result the lesson cites when it says steering with features is not settled.",
    },
    {
      authors: "Mikkel Godsk Jørgensen and Lars Kai Hansen",
      title: "Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines",
      source: "arXiv preprint arXiv:2605.31183",
      year: 2026,
      url: "https://arxiv.org/abs/2605.31183",
      note: "A reply to AxBench: when SAE features are chosen and labeled with a supervised pipeline, steering comes close to a fine-tuned LoRA reference. It is the later paper the lesson cites to show the picture is still changing.",
    },
    {
      authors: "Jacob Dunefsky, Philippe Chlenski, and Neel Nanda",
      title: "Transcoders Find Interpretable LLM Feature Circuits",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2406.11944",
      note: "Transcoders imitate a dense MLP layer with a wider layer where only a few units are active. It backs the Beyond one layer section and leads into the circuits lab.",
    },
    {
      authors: "Jack Lindsey, Adly Templeton, Jonathan Marcus, et al.",
      title: "Sparse Crosscoders for Cross-Layer Features and Model Diffing",
      source: "Transformer Circuits Thread (Anthropic), research update",
      year: 2024,
      url: "https://transformer-circuits.pub/2024/crosscoders/index.html",
      note: "A research update that introduces crosscoders, which learn features shared across several layers or across two models. It is the Anthropic work the lesson cites in Beyond one layer.",
    },
    {
      authors: "Lee Sharkey, Bilal Chughtai, Joshua Batson, et al.",
      title: "Open Problems in Mechanistic Interpretability",
      source: "Transactions on Machine Learning Research (TMLR)",
      year: 2025,
      url: "https://arxiv.org/abs/2501.16496",
      note: "A review by many interpretability researchers. Its section on sparse dictionary learning lists the method's limits, including reconstruction errors that leave doubt about whether features are faithful to the model, which backs the lesson's Where it breaks section.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "In Sparsity and superposition you drag Keep top k features from its smallest value to its largest. What happens to the unexplained share, and why?",
      options: [
        "It rises, because each extra feature adds interference from the features already kept",
        "It falls steeply at first and then flattens, because the largest features carry most of the vector",
        "It stays put, because the autoencoder's weights are fixed and keeping more features changes nothing",
        "It reaches exactly zero at the largest k, because the sparse code is a lossless re-encoding of the residual",
      ],
      answer: 1,
      explanation:
        "The reconstruction is a sum of decoder directions weighted by the active features, so each added feature typically moves the sum closer to the residual. The biggest few features carry most of the vector and the last ones add little, so the curve drops and flattens. It does not reach zero: the code keeps only a few of the dictionary's many features, so some of the activation stays unexplained.",
      objective: 0,
    },
    {
      prompt:
        "Why can one neuron respond to several unrelated things, while a feature in the sparse autoencoder's dictionary can be cleaner?",
      options: [
        "The model holds fewer concepts than neurons, so the spare neurons are given arbitrary meanings",
        "The autoencoder retrains the transformer so that each neuron ends up holding one concept",
        "It packs more features than it has dimensions onto overlapping, nearly orthogonal directions in each layer",
        "Neurons respond to everything until training noise has been averaged out of their weights",
      ],
      answer: 2,
      explanation:
        "That packing is superposition: it works because most features are absent from any one input, so overlapping directions rarely interfere. A single neuron then mixes several features. The sparse autoencoder is a separate model that rewrites an activation over a wider dictionary of directions; the transformer is untouched, and the dictionary's directions overlap only slightly rather than being exactly orthogonal.",
      objective: 0,
    },
    {
      prompt:
        "Two decoder directions overlap more than any other pair: they point almost opposite ways, and the two features are never active on the same character. What is the most likely reason?",
      options: [
        "A ReLU feature cannot go negative, so one signed direction in the residual needs a feature for each side of it",
        "They are duplicates that training failed to merge, so one of them could be deleted without changing any reconstruction",
        "The autoencoder is trained to make every pair of directions orthogonal, and this pair is where that objective failed",
        "They label the same concept, and the TopK rule keeps whichever of the two has the smaller activation",
      ],
      answer: 0,
      explanation:
        "The encoder passes each value through a ReLU, so an activation is never negative. When a direction in the residual takes both signs, the dictionary can cover it with one feature for the positive side and another, pointing the opposite way, for the negative side, and only one of them is active at a time. Duplicates would tend to be active together, and these two never are. Nothing in training pushes directions toward orthogonality, and TopK keeps the largest activations, not the smaller of a pair.",
      objective: 0,
    },
    {
      prompt:
        "A feature's top examples all show one character. You slide Activation threshold down from a high value. What should you expect for a label that looked perfect?",
      options: [
        "Other characters start to cross the line, so precision can fall and reveal what was hidden",
        "Recall falls first, because a lower cutoff misses characters that the label should include",
        "Nothing changes, because a label is a property of the feature's direction and not of the cutoff",
        "The decoder direction rotates toward the new characters, so the label is updated automatically",
      ],
      answer: 0,
      explanation:
        "A feature fires when its activation exceeds the threshold. A high threshold counts only the strongest firings, which are the ones the top examples already showed, so the label looks flawless. Lowering it lets weaker firings on other characters in, and those false alarms lower precision while recall rises. The threshold is not a neutral choice: a high one hides the counterexamples that would disconfirm a label.",
      objective: 1,
    },
    {
      prompt:
        "The sparse autoencoder rebuilds the residual well, with only a small unexplained share. What does that establish about its features?",
      options: [
        "That each feature is monosemantic, since a faithful rebuild needs clean single-concept directions",
        "That the transformer computes its answers using exactly these dictionary directions in its layers",
        "That the code compresses this activation well, not that its features are the model's concepts",
        "That no dictionary entry is dead, since every one of them took part in the rebuild",
      ],
      answer: 2,
      explanation:
        "Reconstruction measures how well a few directions rebuild the vector, and a sparse dictionary can do that without matching how the model itself represents anything. Only a few of the dictionary's many features are active at each character, so some may never fire. Meaning has to come from separate evidence: held-out tests of a label and, ideally, interventions such as steering along the direction.",
      objective: 1,
    },
    {
      prompt:
        "A feature's eight cached top examples all sit at window position 1 or 2, and it still fires on many later characters. What should you conclude about the label you would write from those snippets?",
      options: [
        "It is unreliable, since those examples describe the window start; check examples from later positions",
        "It is reliable, because the largest activations are always the best evidence for what a feature means in text",
        "It is wrong only at positions 1 and 2, and every later firing already matches it",
        "It cannot be tested, because a window position is not something a probe text can reproduce",
      ],
      answer: 0,
      explanation:
        "At window position 1 the residual depends on just two characters and its norm is unusually large, so a feature can win the top-activation ranking there for reasons unrelated to language. The cache records each example's window position so you can see this, and keeps a separate list of examples from later positions. A probe text is itself a window starting at position 0, so you can reproduce the effect by typing text and watching the first characters.",
      objective: 1,
    },
    {
      prompt:
        "You add a feature's decoder direction at one position and the label character's log probability rises by about as much as it does when you add random directions of the same length. What follows?",
      options: [
        "Nothing about this feature yet: a random direction does as well, so the change cannot be attributed to it",
        "The label is confirmed, because the character became more likely after the feature was added to the residual",
        "The label is refuted, because a real feature would have raised the character by far more than that does",
        "The strength was too low, and any strength would show the label once the change turns positive",
      ],
      answer: 0,
      explanation:
        "Adding any vector of a given length changes the output a little, in either direction, because it shifts the residual before the final normalization. The random directions measure that background. A change inside it is what chance directions produce, so it neither confirms nor refutes the label. Only a change well outside the random spread is evidence about this direction in particular.",
      objective: 2,
    },
    {
      prompt:
        "You raise the feature that fires on the letter y, and the next character's probabilities move well outside the random-direction range. What did the lab find, and what does it show?",
      options: [
        "y became less likely and o more likely: a label says where a feature fires, not what adding it writes",
        "y became the top prediction, which confirms that the feature stands for the letter y and nothing else",
        "Nothing changed, because the transformer ignores any vector it did not produce during its own run",
        "y and o rose together, because the feature groups every character that tends to follow a vowel",
      ],
      answer: 0,
      explanation:
        "The feature fires on y, but adding its direction made y less likely and o the largest riser; the lab shows that, not why. The label described where the feature fires and did not predict what adding it writes. That is why a label is a hypothesis about where a feature fires, and steering is a separate test of what it does.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateFeatureState,
};

export default definition;
