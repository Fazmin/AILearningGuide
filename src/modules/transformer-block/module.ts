import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { ORDER_IDS } from "./block";

const initialState: ModuleState = {
  attention: true,
  mlp: true,
  norm: true,
  placement: "pre",
  inputScale: 1,
  position: 5,
  paramPreset: "gpt2",
  paramWidth: 768,
  paramLayers: 12,
  paramMlp: "gelu",
  paramNorm: "layernorm",
  order: "swap",
  positionInfo: true,
};

/**
 * Version 1 stored a decorative `strength` slider (a CSS width). Version 2
 * replaces it with `inputScale`, which really multiplies the block's input.
 * The three sublayer booleans keep their meaning. Version 3 adds the Position
 * card's `order` and `positionInfo`; older payloads simply take the defaults.
 */
export function hydrateTransformerBlockState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
    const { strength: _dropped, ...rest } = parsed;
    const merged: ModuleState = { ...initialState, ...rest };
    for (const key of ["attention", "mlp", "norm"] as const) {
      if (typeof merged[key] !== "boolean") merged[key] = initialState[key];
    }
    if (merged.placement !== "pre" && merged.placement !== "post") merged.placement = "pre";
    const scale = Number(merged.inputScale);
    merged.inputScale = Number.isFinite(scale) ? Math.min(8, Math.max(0.25, scale)) : 1;
    const position = Number(merged.position);
    merged.position = Number.isFinite(position) ? Math.min(5, Math.max(0, Math.round(position))) : 5;
    if (!(ORDER_IDS as readonly unknown[]).includes(merged.order)) merged.order = initialState.order;
    if (typeof merged.positionInfo !== "boolean") merged.positionInfo = initialState.positionInfo;
    return merged;
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-09-transformer-block",
  slug: "transformer-block",
  title: "The transformer block",
  group: "inside-models",
  order: 17,
  icon: "Blocks",
  accent: "#387f8d",
  prerequisites: ["module-08-attention"],
  estimatedMinutes: 15,
  steps: ["Normalize","Route with attention","Add the residual","Transform with the MLP","Test word order"],
  stepInstructions: [
    "Turn Layer norm off, drag Input scale from 1× to 8×, and watch the Self-attention weights collapse onto one token; turn it back on and the sublayer writes stop growing.",
    "Turn Self-attention off and read Output shift if other words change: it drops to exactly 0.000, because nothing else in the block reads another position.",
    "In the Residual stream card, follow each dimension's three lanes to the output diamond, then switch Placement to Post-norm and see the diamond leave the end of the lanes.",
    "Turn the MLP off and on and compare its write with attention's, then read the Parameter count card: the MLP holds two-thirds of a classic block.",
    "In the Position card, set Word order to Swap cat ↔ mat and Position information to Off: Words whose output changed reads 0 of 6. Switch Position information to On and compare the Output shift column.",
  ],
  stateVersion: 3,
  tagline: "Run a tiny transformer block and trace one token through normalization, attention, residual additions, and an MLP, the building block a language model repeats.",
  objectives: ["Trace information through a transformer block","Explain the role of the residual stream","Explain why attention cannot tell word order by itself and what position information adds"],
  glossary: [
  {
    "term": "Residual stream",
    "definition": "The shared vector channel running the full depth of the model, one per position. In a pre-norm block every sublayer reads it and adds an increment; nothing overwrites, so contributions accumulate."
  },
  {
    "term": "MLP",
    "definition": "The position-wise network that transforms each token's features independently: widen, apply a nonlinearity, project back. It holds about two-thirds of a block's parameters."
  },
  {
    "term": "Layer normalization",
    "definition": "Rescaling a vector by subtracting its mean and dividing by its standard deviation across features, then applying a learned scale and shift."
  },
  {
    "term": "RMSNorm",
    "definition": "A cheaper normalization that divides by the root-mean-square without subtracting the mean, with a learned scale but no shift. It performs comparably and is standard in recent models such as Llama."
  },
  {
    "term": "Pre-norm",
    "definition": "Normalizing a copy of each sublayer's input rather than the stream after the addition, leaving the residual path unnormalized. It is the standard placement because deep stacks train stably with it."
  },
  {
    "term": "Sublayer",
    "definition": "One component inside a block, either attention or the MLP. Each is wrapped in a normalization and a residual addition."
  },
  {
    "term": "Skip connection",
    "definition": "The addition that carries the input around a sublayer. It preserves information and gives gradients a direct path to earlier layers."
  },
  {
    "term": "Feed-forward expansion",
    "definition": "The ratio by which the MLP widens the representation before projecting back: four times the model width in a classic MLP, about 8/3 in SwiGLU."
  },
  {
    "term": "SwiGLU",
    "definition": "A gated MLP that multiplies one up-projection by the Swish of another, then projects down. Its hidden width near 8/3 of the model width keeps three matrices at the cost of a classic MLP's two."
  },
  {
    "term": "Position-wise",
    "definition": "Applied to each position independently with shared weights. The MLP is position-wise, which is why attention is the block's only cross-position step."
  },
  {
    "term": "Ablation",
    "definition": "Removing or zeroing a component to observe the effect. In a trained model it produces a distribution-shifted state, so damage overstates the component's isolated role."
  },
  {
    "term": "Mixture of experts",
    "definition": "Replacing one MLP with many and routing each token to a few of them, decoupling total parameter count from the compute spent per token."
  },
  {
    "term": "Positional encoding",
    "definition": "Information about each token's slot, added to its input vector or applied inside attention, because attention alone cannot tell order. Learned, sinusoidal, relative and rotary schemes exist; this lab adds a sinusoidal vector of amplitude 0.5."
  },
  {
    "term": "Permutation equivariance",
    "definition": "The property that reordering the input words only reorders the outputs the same way. Attention without position information has it: a word's output depends on which words are present, not where they sit. The causal mask breaks it slightly, because a word can still count how many came before."
  }
],
  references: [
    {
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.",
      title: "Attention Is All You Need",
      source: "Advances in Neural Information Processing Systems 30 (NIPS 2017)",
      year: 2017,
      url: "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
      note: "The original Transformer. It wraps each sublayer as LayerNorm(x + Sublayer(x)), the post-norm placement, uses an MLP four times wider than the model, and adds sine and cosine position vectors like the lab's input.",
    },
    {
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun",
      title: "Deep Residual Learning for Image Recognition",
      source: "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)",
      year: 2016,
      url: "https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
      note: "The paper behind skip connections. Each layer learns a correction that is added to its input, which is the idea behind the lesson's claim that residual additions made very deep stacks trainable.",
    },
    {
      authors: "Nelson Elhage, Neel Nanda, Catherine Olsson, et al.",
      title: "A Mathematical Framework for Transformer Circuits",
      source: "Transformer Circuits Thread (Anthropic)",
      year: 2021,
      url: "https://transformer-circuits.pub/2021/framework/index.html",
      note: "Describes the residual stream as a shared channel that every attention head and MLP reads from and writes to. It backs the Residual stream card's running sum, the reading and writing of subspaces, and the logit lens in Going deeper.",
    },
    {
      authors: "Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton",
      title: "Layer Normalization",
      source: "arXiv preprint arXiv:1607.06450",
      year: 2016,
      url: "https://arxiv.org/abs/1607.06450",
      note: "Introduces layer normalization: compute the mean and spread across one example's features, rescale, then apply a learned gain and bias. That is the Layer norm switch in the lab, and why it works on one token at a time.",
    },
    {
      authors: "Biao Zhang and Rico Sennrich",
      title: "Root Mean Square Layer Normalization",
      source: "Advances in Neural Information Processing Systems 32 (NeurIPS 2019)",
      year: 2019,
      url: "https://papers.nips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html",
      note: "Introduces RMSNorm, which drops the mean subtraction and divides by the root mean square. It shows the cheaper version works about as well, which is the RMSNorm option on the Parameter count card.",
    },
    {
      authors: "Ruibin Xiong, Yunchang Yang, Di He, et al.",
      title: "On Layer Normalization in the Transformer Architecture",
      source: "Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119, 10524–10533",
      year: 2020,
      url: "https://proceedings.mlr.press/v119/xiong20b.html",
      note: "Compares post-norm and pre-norm. It explains why post-norm needs a learning-rate warm-up and shows pre-norm trains well without one, the reason behind the lesson's Placement section.",
    },
    {
      authors: "Alec Radford, Jeffrey Wu, Rewon Child, et al.",
      title: "Language Models are Unsupervised Multitask Learners",
      source: "OpenAI technical report",
      year: 2019,
      url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
      note: "The GPT-2 paper. It moves layer normalization to the input of each sub-block (pre-norm), and its table lists the smallest model with 12 layers and width 768, the sizes behind the GPT-2 small preset.",
    },
    {
      authors: "Dan Hendrycks and Kevin Gimpel",
      title: "Gaussian Error Linear Units (GELUs)",
      source: "arXiv preprint arXiv:1606.08415",
      year: 2016,
      url: "https://arxiv.org/abs/1606.08415",
      note: "Defines GELU, the smooth nonlinearity in the lab's MLP. Instead of cutting off negative inputs, it weights each input by how likely it is to be kept.",
    },
    {
      authors: "Mor Geva, Roei Schuster, Jonathan Berant, et al.",
      title: "Transformer Feed-Forward Layers Are Key-Value Memories",
      source: "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021), 5484–5495",
      year: 2021,
      url: "https://aclanthology.org/2021.emnlp-main.446/",
      note: "Looks inside trained MLP layers and finds they act like lookup tables: input patterns trigger stored outputs. It shows what the position-wise MLP in this block can learn once the weights are trained.",
    },
    {
      authors: "Noam Shazeer",
      title: "GLU Variants Improve Transformer",
      source: "arXiv preprint arXiv:2002.05202",
      year: 2020,
      url: "https://arxiv.org/abs/2002.05202",
      note: "Introduces SwiGLU and other gated MLPs. It shrinks the hidden width so three matrices cost the same as two, the 8/3 rule behind the SwiGLU option and the Llama 2 7B preset.",
    },
    {
      authors: "William Fedus, Barret Zoph, and Noam Shazeer",
      title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
      source: "Journal of Machine Learning Research 23(120), 1–39",
      year: 2022,
      url: "https://jmlr.org/papers/v23/21-0998.html",
      note: "A mixture-of-experts transformer that sends each token to one of many MLPs. It shows how total parameters can grow while the compute spent per token stays the same, as the glossary entry says.",
    },
    {
      authors: "Adi Haviv, Ori Ram, Ofir Press, et al.",
      title: "Transformer Language Models without Positional Encodings Still Learn Positional Information",
      source: "Findings of the Association for Computational Linguistics: EMNLP 2022, 1382–1390",
      year: 2022,
      url: "https://aclanthology.org/2022.findings-emnlp.99/",
      note: "Shows that a causal model with no position vector can still work out order by counting how many tokens it can see. It backs the lesson's point that the Position card lifts the mask because a masked block already leaks some order.",
    },
    {
      authors: "Jianlin Su, Murtadha Ahmed, Yu Lu, et al.",
      title: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
      source: "Neurocomputing 568, 127063",
      year: 2024,
      url: "https://arxiv.org/abs/2104.09864",
      note: "Introduces rotary position embeddings, which put position inside attention by rotating queries and keys. It is the other way to supply order that the lesson mentions next to the lab's added position vector.",
    },
    {
      authors: "Hugo Touvron, Louis Martin, Kevin Stone, et al.",
      title: "Llama 2: Open Foundation and Fine-Tuned Chat Models",
      source: "arXiv preprint arXiv:2307.09288",
      year: 2023,
      url: "https://arxiv.org/abs/2307.09288",
      note: "Describes a modern block: pre-norm with RMSNorm, a SwiGLU MLP and rotary position embeddings. It is the real model behind the Llama 2 7B preset on the Parameter count card.",
    },
    {
      authors: "Fred Zhang and Neel Nanda",
      title: "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods",
      source: "The Twelfth International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2309.16042",
      note: "Compares ways of knocking out parts of a trained model and finds that methods that push it off its training data can mislead. It backs the Where it breaks warning that ablation in a trained model is not a clean subtraction.",
    },
  ],
  checkpoint: [
    {
      prompt: "In a pre-norm block you switch Self-attention off. What happens to one token's output when the other words in the sentence change?",
      options: [
        "It still changes, because the MLP mixes information across positions",
        "It still changes, because Layer norm averages over all the words in the sentence",
        "Nothing, because the norms and the MLP each work on a single position at a time",
      ],
      answer: 2,
      explanation: "Attention is the block's only cross-position step. Layer norm normalizes across one token's features, and the MLP applies the same weights to each position separately, so with attention off the lab's output shift is exactly 0.000 while the residual stream still carries the token's own vector through.",
      objective: 0,
    },
    {
      prompt: "In a pre-norm block you switch the MLP off. What does the block output for that token equal in the Residual stream card?",
      options: [
        "Only the attention write, because each sublayer replaces the stream with its own result",
        "The input plus the attention write, because the MLP now adds zeros to the stream",
        "Zeros, because the MLP is the last stage and the output is its write alone",
      ],
      answer: 1,
      explanation: "The stream is a running sum: output = input + attention write + MLP write. A sublayer adds its increment and never overwrites, so switching one off removes only its own lane and the diamond lands at the end of the remaining ones. That additive path is also why gradients reach early layers.",
      objective: 1,
    },
    {
      prompt: "In the Position card you swap cat and mat and set Position information to Off. What do the block's six output vectors do?",
      options: [
        "They are the same six vectors, just sitting in the other words' slots, so the block cannot tell the sentences apart",
        "They change, because each word's attention weights depend on which of the other words came before and after it",
        "Only cat and mat change, because the other four words kept their slots and so have the same inputs as before",
      ],
      answer: 0,
      explanation: "Attention scores depend only on the vectors being compared, and without a position vector a word's vector is the same wherever it sits. Each word therefore gets the same output in both orders. Position information makes the input depend on the slot, which is what lets later layers use order.",
      objective: 2,
    },
    {
      prompt: "Ablating a sublayer in this untrained block is exact. What goes wrong when you ablate one in a trained model and read the drop in quality as that sublayer's own effect?",
      options: [
        "The skip connection replaces the removed write, so the measured drop is exactly its own contribution",
        "Later layers expected its writes, so removing it also hands them inputs they never saw in training",
        "The other layers retrain their weights around the gap, so the measured drop is too small",
      ],
      answer: 1,
      explanation: "In this untrained block nothing downstream expected the sublayer, so ablation is exact. In a trained model every later layer learned to read the sublayer's contribution. Removing it hands them an input they never saw, so the damage mixes the sublayer's role with that distribution shift, and no weights change.",
      objective: 0,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateTransformerBlockState
};

export default definition;
