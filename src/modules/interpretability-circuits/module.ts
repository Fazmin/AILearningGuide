import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { DEFAULT_PERIOD, PERIODS } from "./experiments";

const HEAD_IDS = ["L1H1", "L1H2", "L1H3", "L1H4", "L2H1", "L2H2", "L2H3", "L2H4"];

const initialState: ModuleState = {
  period: DEFAULT_PERIOD,
  // The first head shown is L2 H3, one of the two induction heads.
  attnLayer: 1,
  attnHead: 2,
  ablated: [],
  ablation: "zero",
  sweep: "live",
  corruption: "source",
  layer: 0,
  patch: 16,
};

const clampInt = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};

/**
 * Version 1 stored a synthetic lab: `layer` 0–5 and `patch` 0–7 for a
 * decorative grid, a `steering` percentage, and `ablated` as a boolean for one
 * unnamed head. `layer` and `patch` still mean the patched layer and position
 * and are clamped to the real model; `steering` has no counterpart and is
 * dropped; a boolean `ablated` becomes no ablated heads.
 *
 * Version 2 had no `ablation` or `corruption` (zero-ablation and the copied
 * letter were the only options), offered periods 6, 7, 8, 9, 10 and 12, and
 * opened on head L1 H1; every v2 value is still valid in v3, so the missing
 * keys take their defaults and nothing else needs translating. Every key is
 * validated and clamped, because state can arrive from a share link.
 */
export function hydrateCircuitState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    const ablated = Array.isArray(parsed.ablated) ? parsed.ablated.filter((id) => HEAD_IDS.includes(id)) : [];
    const lowest = PERIODS[0];
    const highest = PERIODS[PERIODS.length - 1];
    const period = clampInt(parsed.period, DEFAULT_PERIOD, lowest, highest);
    return {
      period: (PERIODS as readonly number[]).includes(period) ? period : DEFAULT_PERIOD,
      attnLayer: clampInt(parsed.attnLayer, 1, 0, 1),
      attnHead: clampInt(parsed.attnHead, 2, 0, 3),
      ablated: [...new Set(ablated)],
      ablation: parsed.ablation === "mean" ? "mean" : "zero",
      sweep: parsed.sweep === "speaker" || parsed.sweep === "copy" ? parsed.sweep : "live",
      corruption: parsed.corruption === "control" ? "control" : "source",
      layer: clampInt(parsed.layer, 0, 0, 1),
      patch: clampInt(parsed.patch, 16, 0, 63),
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-16-interpretability-circuits",
  slug: "interpretability-circuits",
  title: "Interpretability II: circuits",
  group: "frontiers",
  order: 35,
  icon: "CircuitBoard",
  accent: "#a65d38",
  prerequisites: ["module-15-interpretability-features"],
  estimatedMinutes: 18,
  steps: ["Find an induction pattern", "Ablate a head", "Patch activations", "Try an unseen period"],
  stepInstructions: [
    "In Copy task and attention, choose L2 H3 in the head picker and read the solid green outlines (the key after the previous occurrence), then choose L1 H4 and read the dashed outlines (the previous letter). Compare the two columns of the table for all eight heads.",
    "In Head ablation, switch off L1 H4, then L2 H3, then L2 H1, and read Change and the induction-key attention for each. Then set Ablation to Mean and compare what L2 H3 costs.",
    "In Activation patching, click After block 1 at the amber position and then at the last position, and compare recovery. Then set Corrupt to An unrelated letter and read the Gap and the grid.",
    "Set Repeat period to 11 or 13, which the model's repeated-string training never used, and compare copy accuracy, the green outlines, and the L2 H3 row of the table with period 8.",
  ],
  stateVersion: 3,
  tagline: "Test a real two-layer transformer that copies repeating letters: read its attention for an induction circuit, switch heads off with zero and mean ablation, and patch activations from a clean run into a corrupted one, then ask what those tests still cannot show.",
  objectives: [
    "Read the two halves of an induction circuit in attention and test them on a repeat period the model never saw",
    "Use ablation and patching to test causality",
    "State the limits of circuit claims",
  ],
  glossary: [
    {
      term: "Circuit",
      definition:
        "A claim that specific components, connected in a specific way, implement a specific behavior. Supporting one requires intervention, not observation.",
    },
    {
      term: "Activation patching",
      definition:
        "Copying one activation from a clean run into a corrupted run and re-running the rest of the model, to test whether that activation carried the information the behavior depended on.",
    },
    {
      term: "Induction head",
      definition:
        "A head that attends from the current token to the token after its previous occurrence and copies it, so A B … A predicts B. It relies on an earlier head that writes each token's predecessor into its position.",
    },
    {
      term: "Previous-token head",
      definition:
        "A head that attends from each position to the token just before it, so the position carries a note of what came before. An induction head in a later layer reads those notes to find the token after an earlier match.",
    },
    {
      term: "Ablation",
      definition:
        "Removing or zeroing a component and measuring the damage. It answers whether the component is necessary in that setting, not what it computes or where information is stored.",
    },
    {
      term: "Mean ablation",
      definition:
        "Replacing an activation with its average over a reference set rather than zero. It reduces the distribution shift that makes zero-ablation overstate damage, though the answer still depends on which reference set is averaged.",
    },
    {
      term: "Path patching",
      definition:
        "Patching a single connection between two components rather than a location, which distinguishes a component mattering from it mattering through a particular route.",
    },
    {
      term: "Attribution patching",
      definition:
        "Estimating a whole patching sweep with a first-order gradient approximation: two forward passes and one backward pass instead of one forward pass per patched location, at some cost in accuracy.",
    },
    {
      term: "Logit difference",
      definition:
        "The gap between the logits of the right answer and one specific wrong answer. A useful metric moves a lot between clean and corrupted runs; one that barely moves makes every recovery ratio meaningless.",
    },
    {
      term: "Clean and corrupted run",
      definition:
        "A pair of minimally different prompts, one producing the behavior and one not. Their contrast is what makes a patching result interpretable.",
    },
    {
      term: "Steering",
      definition:
        "Adding a direction to the residual stream during generation to shift behavior. Testing both directions is stronger evidence than a one-sided effect.",
    },
    {
      term: "Causal scrubbing",
      definition:
        "Restating a hypothesis as which computations may be swapped without changing behavior, then performing every permitted swap. It makes circuit claims falsifiable.",
    },
    {
      term: "Self-repair",
      definition:
        "Other components compensating when one is ablated, sometimes called backup behavior. It means a small measured effect does not prove a small role.",
    },
    {
      term: "Faithfulness",
      definition:
        "Whether an identified subgraph reproduces the behavior on its own. Evaluated alongside completeness and minimality when judging any circuit claim. Not the same as reasoning-trace faithfulness in ICL & reasoning, which asks whether a written explanation reflects the computation behind the answer.",
    },
  ],
  references: [
    {
      authors: "Nelson Elhage, Neel Nanda, Catherine Olsson, et al.",
      title: "A Mathematical Framework for Transformer Circuits",
      source: "Transformer Circuits Thread, Anthropic",
      year: 2021,
      url: "https://transformer-circuits.pub/2021/framework/index.html",
      note: "Takes apart small attention-only transformers and shows that a two-layer model can build an induction head out of a previous-token head in the layer below. That is the two-part circuit the Copy task and attention card reads in L1 H4 and L2 H3.",
    },
    {
      authors: "Catherine Olsson, Nelson Elhage, Neel Nanda, et al.",
      title: "In-context Learning and Induction Heads",
      source: "Transformer Circuits Thread, Anthropic",
      year: 2022,
      url: "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html",
      note: "Defines induction heads by the pattern [A][B] ... [A] -> [B] and argues they may be the source of much in-context learning. It calls its evidence strong and causal for small attention-only models but only correlational for larger ones, which echoes this lesson's point about one small model.",
    },
    {
      authors: "Kevin Meng, David Bau, Alex Andonian, et al.",
      title: "Locating and Editing Factual Associations in GPT",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2202.05262",
      note: "Its causal tracing runs a clean and a corrupted prompt, then restores one clean hidden state at a time to see which ones bring the answer back. The Activation patching card does the same thing, one block output and position per grid cell.",
    },
    {
      authors: "Kevin Wang, Alexandre Variengien, Arthur Conmy, et al.",
      title: "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small",
      source: "International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2211.00593",
      note: "The worked example the lesson cites: 26 heads in 7 classes, judged on faithfulness, completeness and minimality. It explains why it uses mean ablation instead of zero, and it finds backup name mover heads that take over when the main ones are knocked out, the self-repair the lesson warns about.",
    },
    {
      authors: "Fred Zhang and Neel Nanda",
      title: "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2309.16042",
      note: "Shows that patching results can change with the choice of corruption and metric. It compares adding noise with swapping in a different token, the kind of corruption the Corrupt control uses, and compares logit difference with probability as the metric.",
    },
    {
      authors: "Stefan Heimersheim and Neel Nanda",
      title: "How to use and interpret activation patching",
      source: "arXiv preprint arXiv:2404.15255",
      year: 2024,
      url: "https://arxiv.org/abs/2404.15255",
      note: "A short practical guide. It separates patching clean into corrupted (what the lab does) from the reverse direction, describes path patching, and says why the authors trust logit difference most among the metrics, the measure behind the live sweep's Gap.",
    },
    {
      authors: "Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, et al.",
      title: "Localizing Model Behavior with Path Patching",
      source: "arXiv preprint arXiv:2304.05969",
      year: 2023,
      url: "https://arxiv.org/abs/2304.05969",
      note: "Introduces path patching, which tests a claim that a behavior runs along particular connections between components. It applies it to a previous-token head feeding an induction head, the layer 1 to layer 2 route this lab could only test with ablation.",
    },
    {
      authors: "János Kramár, Tom Lieberum, Rohin Shah, et al.",
      title: "AtP*: An efficient and scalable method for localizing LLM behaviour to components",
      source: "arXiv preprint arXiv:2403.00745 (Google DeepMind)",
      year: 2024,
      url: "https://arxiv.org/abs/2403.00745",
      note: "Studies attribution patching, which estimates every patching result at once from two forward passes and one backward pass. It finds cases where the shortcut misses important components and proposes fixes, the accuracy cost the glossary mentions.",
    },
    {
      authors: "Lawrence Chan, Adrià Garriga-Alonso, Nicholas Goldowsky-Dill, et al.",
      title: "Causal Scrubbing: a method for rigorously testing interpretability hypotheses",
      source: "Redwood Research, AI Alignment Forum",
      year: 2022,
      url: "https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing",
      note: "The original write-up of causal scrubbing by its authors. It tests a hypothesis by resampling every activation the hypothesis says should not matter, and it uses the method to refine an account of induction in a small language model.",
    },
    {
      authors: "Thomas McGrath, Matthew Rahtz, Janos Kramar, et al.",
      title: "The Hydra Effect: Emergent Self-repair in Language Model Computations",
      source: "arXiv preprint arXiv:2307.15771 (Google DeepMind)",
      year: 2023,
      url: "https://arxiv.org/abs/2307.15771",
      note: "Finds that when one attention layer is ablated, another layer can compensate, and names this the Hydra effect. It is the lesson's source for self-repair and for why a small ablation effect does not prove a small role.",
    },
    {
      authors: "Emmanuel Ameisen, Jack Lindsey, Adam Pearce, et al.",
      title: "Circuit Tracing: Revealing Computational Graphs in Language Models",
      source: "Transformer Circuits Thread, Anthropic",
      year: 2025,
      url: "https://transformer-circuits.pub/2025/attribution-graphs/methods.html",
      note: "The circuit tracing work the lesson names. It builds attribution graphs over interpretable features in a replacement model, then checks them with interventions and steering in the original model, rather than mapping raw heads as this lab does.",
    },
    {
      authors: "Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, et al.",
      title: "On the Biology of a Large Language Model",
      source: "Transformer Circuits Thread, Anthropic",
      year: 2025,
      url: "https://transformer-circuits.pub/2025/attribution-graphs/biology.html",
      note: "The companion paper that applies circuit tracing to Claude 3.5 Haiku. In one case it turns down the model's internal \"Texas\" features and turns up \"California\" ones, and the answer changes from Austin to Sacramento. This is the kind of test the lesson asks of a steering claim.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "At period 8 the key an induction head would read is exactly 7 letters back, so a head that always looked 7 back would draw the same map. What tells the two stories apart?",
      options: [
        "The weight on the green key at period 8, since a fixed look-back would not put any weight there",
        "The copy accuracy at period 8, since only a content-based head could produce an accuracy that high",
        "The stripe at an unseen period such as 13, which moves to the new key, not 7 back",
        "The previous-letter column at period 8, since a fixed look-back would leave that column at zero",
      ],
      answer: 2,
      explanation:
        "At period 8 the key after the previous occurrence is also 7 back, so both stories draw the same map and both copy well; neither the weight on that key nor the accuracy can separate them. Changing the period moves the previous occurrence but not a fixed distance, so the stories now disagree. At period 13, a length the model's repeated-string training never used, the outlines and the weights both move to 12 back, which is what a content-based head predicts. The previous-letter column measures a different head.",
      objective: 0,
    },
    {
      prompt:
        "L1 H4 puts nearly all its attention on the previous letter, and L2 H3 puts its on the key after the previous occurrence. What do you expect to happen to L2 H3's attention when L1 H4 is switched off?",
      options: [
        "It falls sharply, because L2 H3 reads the notes about each letter's predecessor that layer 1 writes",
        "It stays the same, because a head's attention is fixed by its own weights and not by what earlier layers write",
        "It moves to the previous letter, because the nearest letter is the next best match for the query",
        "It stays the same, because heads in different layers work on separate copies of the sequence",
      ],
      answer: 0,
      explanation:
        "Attention in layer 2 is computed from the residual stream that layer 1 leaves behind, not from the raw letters. If L1 H4 no longer writes each letter's predecessor into its position, a layer-2 head has fewer notes to match against and its weight on the green key drops. The lab shows this: at period 8 it falls from about 0.67 to 0.24. L1 H3 is a second previous-letter head, and with both off almost none is left, so no single layer-1 head carries the whole job.",
      objective: 0,
    },
    {
      prompt:
        "With the letter to copy corrupted, restoring the clean vector at that position after block 1 brings back almost all of the gap, while restoring the last position after block 1 brings back almost none. What does that show?",
      options: [
        "Block 1 plays no part in copying, since restoring its output at the last position changes nothing",
        "After block 1 the letter is carried at the earlier position, and block 2 reads it from there",
        "The earlier position is the only one that matters at any layer, so the recoveries must add to 100%",
        "The last position is unimportant, because the model could produce the answer without reading it",
      ],
      answer: 1,
      explanation:
        "Patching pastes one clean vector into the corrupted run and measures how much of the metric returns, so a large recovery at one cell says the information the metric needs sits there. Here it sits at the earlier position, where layer 2's induction heads read it; the last position does not hold it yet after block 1. That is a statement about this cell and this prompt pair: it does not say block 1 is idle, and recoveries from different cells need not add to 100%. After block 2 the last position recovers everything by construction, because nothing later reads other positions.",
      objective: 1,
    },
    {
      prompt:
        "Switching off L1 H2 costs copy accuracy as much as switching off L1 H4, yet L1 H2 attends two letters back, not one. What is the careful conclusion?",
      options: [
        "L1 H2 is the induction head, because whichever head's removal costs the most is the one doing the copying itself",
        "L1 H2 must be a second previous-token head after all, because only those heads can hurt copying this much",
        "L1 H2 is unrelated to copying, because it does not attend to the previous letter, so the drop must be an artefact",
        "L1 H2 is needed for copying here, but this does not show what it computes or that it joins the same route",
      ],
      answer: 3,
      explanation:
        "Ablation measures necessity: on these prompts the model copies much worse without L1 H2, and the drop survives mean ablation, so it is not just a zeroing artefact. It does not say what L1 H2 computes, or whether it feeds the induction heads through the same route as L1 H4; the lab even finds that L1 H2's own attention is not on the previous letter. Telling a head that is needed from a head that implements the story takes path patching or a causal test of the whole hypothesis.",
      objective: 1,
    },
    {
      prompt:
        "With L2 H3 switched off, copy accuracy is much lower under zero-ablation than under mean ablation. Why would the two disagree?",
      options: [
        "The mean is taken over the prompt under test itself, so it restores the head's real output at every position",
        "Mean ablation leaves the head partly in place, so only zero-ablation really removes it from the model",
        "Zero is a value the next layers never saw, so part of the damage is the model reacting to unfamiliar input",
        "One of the two is a bug, because removing a head should give the same answer whatever replaces its output",
      ],
      answer: 2,
      explanation:
        "Both are interventions with a free choice of what replaces the head's output, and neither is ground truth. Zero is arbitrary and later components may rely on typical values, so zero-ablation can overstate the damage; the mean comes from other random repeats, not the prompt under test, and keeps the head's average contribution. Here L2 H3 costs less under the mean, while L1 H2, L1 H3 and L1 H4 stay costly under both. When the two agree on which heads matter the claim is stronger, and the sizes are not reliable.",
      objective: 2,
    },
    {
      prompt:
        "In Activation patching, set Corrupt to An unrelated letter. A few grid cells show large recoveries, though the final answer barely changed. Why should you not read them?",
      options: [
        "Recoveries above 100% mean the patch made the model better than its clean run, which is real evidence of a second route",
        "Recovery divides by the gap between the two runs, which is tiny here, so small shifts become huge percentages",
        "The grid colour is clipped at 100%, so the large cells are drawing errors, not computed values",
        "Patching only works when the corrupted run gets the answer wrong, and with this corruption it still did",
      ],
      answer: 1,
      explanation:
        "Recovery is (patched − corrupted) ÷ (clean − corrupted). An unrelated letter moves the metric by well under a logit, so rounding-level changes in the numerator become large ratios, and the cells describe noise rather than a mechanism. The colour does saturate at 100%, but the exact values are printed. Check the gap before reading any cell, and choose a corruption that changes the answer, as the letter to copy does: its gap is several logits.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateCircuitState,
};

export default definition;
