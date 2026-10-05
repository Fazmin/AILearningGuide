import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { DEFAULT_PIXELS, FILTERS, MAX_EXPERTS, SIZE, WIRINGS } from "./zoo";

const initialState: ModuleState = {
  pixels: DEFAULT_PIXELS,
  filter: "edge",
  rnnStep: 0,
  cell: 0,
  recurrentWeight: 0.9,
  wiring: "conv",
  convLayers: 1,
  experts: 8,
  expertsPerToken: 2,
};

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const intIn = (value: unknown, fallback: number, low: number, high: number) =>
  finite(value) ? Math.min(high, Math.max(low, Math.round(value))) : fallback;
const oneOf = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;

/**
 * Version 1 stored { pixels, filter, rnnStep, cell }. Those keys keep their meaning;
 * version 2 adds the recurrent weight, the wiring comparison, and the expert router,
 * which a version 1 payload simply takes from the defaults.
 */
export function hydrateArchitectureZoo(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
    const pixels =
      typeof parsed.pixels === "string" && parsed.pixels.length >= SIZE
        ? parsed.pixels.replace(/[^01]/g, "0").slice(0, SIZE * SIZE).padEnd(SIZE * SIZE, "0")
        : DEFAULT_PIXELS;
    const experts = intIn(parsed.experts, 8, 2, MAX_EXPERTS);
    return {
      pixels,
      filter: oneOf(parsed.filter, FILTERS.map((item) => item.id), "edge"),
      rnnStep: intIn(parsed.rnnStep, 0, 0, SIZE - 1),
      cell: intIn(parsed.cell, 0, 0, SIZE * SIZE - 1),
      recurrentWeight: finite(parsed.recurrentWeight)
        ? Math.min(2, Math.max(0, Math.round(parsed.recurrentWeight * 100) / 100))
        : 0.9,
      wiring: oneOf(parsed.wiring, WIRINGS.map((item) => item.id), "conv"),
      convLayers: intIn(parsed.convLayers, 1, 1, SIZE - 1),
      experts,
      expertsPerToken: intIn(parsed.expertsPerToken, 2, 1, Math.min(4, experts)),
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-35-architecture-zoo",
  slug: "architecture-zoo",
  title: "The architecture zoo",
  group: "foundations",
  order: 14,
  icon: "Shapes",
  accent: "#5a7a3a",
  prerequisites: ["module-34-why-gpus"],
  estimatedMinutes: 14,
  steps: ["Draw the digit", "Pick a filter", "Walk the RNN", "Compare wirings", "Route to experts"],
  stepInstructions: [
    "Click cells on the 8×8 pad, or move Selected cell and press Flip selected cell. The outlined 3×3 window is all the selected filter output can see; flip a pixel inside it and watch that output's number change.",
    "Set Filter to horizontal edge, then to center blob. The same ink and the same 9 weights, reused at all 64 positions, give a different response map.",
    "Move RNN step from t=0 to t=7 and compare the ∂h7/∂x bars: at Recurrent weight 0.90 column 1 reaches the final state with about 0.009, column 8 with about 0.6.",
    "In Who sees whom, switch Wiring among Dense MLP, Per-position MLP, Convolution, Recurrent, Attention and State-space, and with Convolution raise Conv layers from 1 to 7 to watch the band widen.",
    "In Sparse experts, set Experts per token to 1, then raise Experts from 8 to 16. Total expert weights double while Active per token stays fixed.",
  ],
  stateVersion: 2,
  tagline:
    "Compare how dense layers, convolutions, recurrent nets, attention, state-space models and mixtures of experts wire their weights, so a transformer looks like one wiring, not the definition of a neural net.",
  objectives: [
    "Match convolution, recurrence, and attention to the reuse pattern each one assumes",
    "Treat a transformer as one architecture among several that map sequences to predictions",
    "Separate the weights a model stores from the weights each input actually uses",
  ],
  glossary: [
    {
      term: "Architecture",
      definition:
        "The wiring that decides which computations share weights and which activations can see which others. It is chosen before training and shapes what is easy to learn.",
    },
    {
      term: "Inductive bias",
      definition:
        "The assumptions an architecture builds in, such as locality in a convolution or order in a recurrent net. Bias here means a prior, not a social prejudice.",
    },
    {
      term: "Dense layer",
      definition:
        "A layer where every output has its own weight to every input, as in a multilayer perceptron. Applied to a whole flattened sequence it shares nothing across positions, so its weights grow with the square of the input size.",
    },
    {
      term: "Weight sharing",
      definition:
        "Reusing the same weights at every position or time step. It is why a convolution needs 9 weights where a dense 64-to-64 layer needs 4,096, and why a pattern learned in one place works in another.",
    },
    {
      term: "Convolution",
      definition:
        "A small filter slid over every position, taking a weighted sum of each neighbourhood. Deep-learning libraries compute it as cross-correlation, without flipping the kernel.",
    },
    {
      term: "Receptive field",
      definition:
        "The set of inputs one output can depend on. A kernel-3 convolution sees 3×3 after one layer and grows by one pixel in each direction per extra layer.",
    },
    {
      term: "Recurrent network",
      definition:
        "A block that reads a sequence one step at a time and carries a fixed-size hidden state forward. Steps must run in order, and long dependencies must survive that state.",
    },
    {
      term: "Vanishing gradient",
      definition:
        "Gradients that shrink as they flow back through many steps, because each step multiplies them by a factor below one. Early inputs then barely affect learning; factors above one explode instead.",
    },
    {
      term: "LSTM",
      definition:
        "Long short-term memory (Hochreiter and Schmidhuber, 1997): a recurrent cell with gates and an additive cell state, which lets gradients survive far more steps than a plain tanh recurrence.",
    },
    {
      term: "Transformer",
      definition:
        "A stack of blocks, each an attention layer that mixes positions and a per-position MLP (Vaswani et al., 2017). With a causal mask each position reads all earlier ones in one hop, up to the context length.",
    },
    {
      term: "Causal",
      definition:
        "A constraint that lets each position read itself and earlier positions but never later ones. It is what lets a language model train on next-token prediction and then generate left to right. Here Attention, Recurrent and State-space are causal; the Convolution is centred and sees the future.",
    },
    {
      term: "KV cache",
      definition:
        "The stored keys and values of every earlier position, kept so each new token reuses them instead of recomputing them. It grows with the context length n: 2·n·d numbers per layer here, 1,024 at n = 8 and 524,288 at n = 4,096.",
    },
    {
      term: "State-space model",
      definition:
        "A sequence layer with a linear recurrent state update, so it can train with a parallel scan and generate with a fixed-size state. Mamba (Gu and Dao, 2023) makes the update depend on the input.",
    },
    {
      term: "Mixture of experts",
      definition:
        "A layer with many expert MLPs and a router that sends each token to only the top k of them. Total parameters grow with the number of experts, while compute per token grows only with k.",
    },
  ],
  references: [
    {
      authors: "Michael M. Bronstein, Joan Bruna, Taco Cohen, et al.",
      title: "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges",
      source: "arXiv preprint arXiv:2104.13478",
      year: 2021,
      url: "https://arxiv.org/abs/2104.13478",
      note: "A book-length survey that explains architectures by the assumptions they build in. Section 2.1 covers inductive bias, and Chapter 5 walks through convolutional nets, transformers, recurrent nets, and LSTMs, the same zoo this lesson compares.",
    },
    {
      authors: "Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola",
      title: "Dive into Deep Learning",
      source: "Cambridge University Press, free to read online, Chapter 7",
      year: 2023,
      url: "https://d2l.ai/chapter_convolutional-neural-networks/index.html",
      note: "Section 7.1 starts from a dense layer and shows how translation invariance and locality shrink it into a convolution. Section 7.2 explains why libraries compute cross-correlation, uses a small kernel to detect edges, and defines the receptive field, as on the Live filter card.",
    },
    {
      authors: "Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner",
      title: "Gradient-Based Learning Applied to Document Recognition",
      source: "Proceedings of the IEEE 86(11), 2278–2324",
      year: 1998,
      url: "http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf",
      note: "The LeNet-5 paper, which reads handwritten digits like the one on Drawn digit. Section II explains local receptive fields and shared weights, and notes that shifting the input shifts the feature map by the same amount, which the first checkpoint question tests.",
    },
    {
      authors: "Shaojie Bai, J. Zico Kolter, and Vladlen Koltun",
      title: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling",
      source: "arXiv preprint arXiv:1803.01271",
      year: 2018,
      url: "https://arxiv.org/abs/1803.01271",
      note: "Compares convolutional and recurrent networks on the same sequence tasks. It defines causal convolutions, the kind text models use, and explains that a plain causal convolution looks back only as far as its depth allows, which is what raising Conv layers shows.",
    },
    {
      authors: "Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio",
      title: "On the difficulty of training recurrent neural networks",
      source: "Proceedings of the 30th International Conference on Machine Learning (ICML 2013), PMLR 28(3), 1310–1318",
      year: 2013,
      url: "https://proceedings.mlr.press/v28/pascanu13.html",
      note: "Shows that a recurrent net's gradient is a product of one factor per time step, so it can shrink to zero or blow up. That is the math behind the ∂h7/∂x bars on Unrolled RNN, which fade below a Recurrent weight of 1 and grow above it.",
    },
    {
      authors: "Sepp Hochreiter and Jürgen Schmidhuber",
      title: "Long Short-Term Memory",
      source: "Neural Computation 9(8), 1735–1780",
      year: 1997,
      url: "https://www.bioinf.jku.at/publications/older/2604.pdf",
      note: "The paper that introduced the LSTM. It starts from the decaying error flow in plain recurrent nets and adds gates and a cell state that keep the gradient alive over long time gaps, the fix the lesson names for the fading RNN chain.",
    },
    {
      authors: "Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, et al.",
      title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
      source: "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), 1724–1734",
      year: 2014,
      url: "https://aclanthology.org/D14-1179/",
      note: "Section 2.3 proposes a recurrent unit with a reset gate and an update gate, the design now called the GRU. The lesson names it next to the LSTM as a gated recurrent net that carries gradients much farther than a plain chain.",
    },
    {
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.",
      title: "Attention Is All You Need",
      source: "Advances in Neural Information Processing Systems 30 (NIPS 2017)",
      year: 2017,
      url: "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
      note: "Introduces the transformer as attention plus a per-position feed-forward network, with a mask that hides later positions. Its Why Self-Attention section compares path lengths for attention, recurrence, and convolution, the same hop counts drawn in Who sees whom.",
    },
    {
      authors: "Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, et al.",
      title: "Efficiently Scaling Transformer Inference",
      source: "Proceedings of Machine Learning and Systems 5 (MLSys 2023)",
      year: 2023,
      url: "https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html",
      note: "Explains that the attention keys and values of every layer, the KV cache, must stay in memory while a transformer generates text, and that this cost grows with context length. It backs the KV cache column in the cost table.",
    },
    {
      authors: "Albert Gu, Karan Goel, and Christopher Ré",
      title: "Efficiently Modeling Long Sequences with Structured State Spaces",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2111.00396",
      note: "Introduces S4, a state-space sequence layer. It shows the same layer can run as a recurrence, one step at a time with a fixed state, or as a convolution over the whole sequence for training, the trade-off the State-space wiring stands for.",
    },
    {
      authors: "Albert Gu and Tri Dao",
      title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
      source: "Conference on Language Modeling (COLM 2024); arXiv:2312.00752",
      year: 2024,
      url: "https://arxiv.org/abs/2312.00752",
      note: "Makes the state-space update depend on the input, so the model can choose what to keep or forget, and trains it with a parallel scan. It contrasts attention, which must store the whole KV cache, with a recurrent state of fixed size, as the cost table does.",
    },
    {
      authors: "Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, et al.",
      title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer",
      source: "International Conference on Learning Representations (ICLR 2017)",
      year: 2017,
      url: "https://arxiv.org/abs/1701.06538",
      note: "Introduces the sparsely gated mixture-of-experts layer, where a gating network sends each example to only a few expert networks. It adds an extra loss to stop the router from favoring the same few experts, the fix the lesson mentions for idle experts.",
    },
    {
      authors: "William Fedus, Barret Zoph, and Noam Shazeer",
      title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
      source: "Journal of Machine Learning Research 23(120), 1–39",
      year: 2022,
      url: "https://jmlr.org/papers/v23/21-0998.html",
      note: "Routes each token to a single expert, like setting Experts per token to 1, and shows that the parameter count can grow while compute per token stays about the same. It also describes the load-balancing loss that trained routers use.",
    },
    {
      authors: "Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, et al.",
      title: "Mixtral of Experts",
      source: "arXiv preprint arXiv:2401.04088",
      year: 2024,
      url: "https://arxiv.org/abs/2401.04088",
      note: "Describes Mixtral 8x7B, where a router picks 2 of 8 experts for each token at each layer. It separates the total parameter count, which grows with the number of experts, from the active count, which grows with how many each token uses, as Sparse experts does.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "On Drawn digit you ink the same small stroke in a different place, with the same Filter selected. What happens to the response map on Live filter?",
      options: [
        "The strong responses disappear, because the kernel was tuned to the first location of the stroke",
        "The same pattern of responses appears at the new location, because one kernel is reused everywhere",
        "The map is recomputed with different weights for that region, because each position has its own filter",
        "The responses stay where the stroke used to be, because the filter remembers the earlier ink",
      ],
      answer: 1,
      explanation:
        "A convolution applies one 3×3 kernel at all 64 positions, so each response depends only on the ink inside its window, not on where the window is. Shifting the ink shifts the response map by the same amount. That sharing is why 9 weights replace a dense 64-to-64 map of 4,096, at the price of assuming the same local pattern matters everywhere.",
      objective: 0,
    },
    {
      prompt:
        "The Unrolled RNN reuses one recurrent weight at every step. At Recurrent weight 0.90, how do the ∂h7/∂x bars for column 1 and column 8 compare?",
      options: [
        "They are equal, because the same weight is reused at every step, so every input influences the final state equally",
        "Column 1 is larger, because the earliest input has had the most steps to build up its influence",
        "Column 1 is far smaller, because its influence is multiplied by a factor below one at each step on the way to h7",
        "Column 8 is zero, because the last input has no recurrent steps left to carry its effect",
      ],
      answer: 2,
      explanation:
        "The gradient of the final state with respect to an input is a product with one factor per step, w·(1 − h²). With w = 0.9 each factor is below one, so the product shrinks with distance: column 1 reaches h7 with about 0.009 and column 8 with about 0.6. Reusing the weight shares parameters, not influence. Above 1, the same product grows instead of fading.",
      objective: 0,
    },
    {
      prompt:
        "In Who sees whom, which wiring lets output position 8 use input position 1 in a single hop, while never seeing a later position?",
      options: [
        "Causal self-attention, which scores every earlier position directly and masks the later ones",
        "A recurrent network, which carries input 1 forward through its hidden state",
        "A one-layer convolution with kernel 3, which reaches farther as it slides along the sequence",
      ],
      answer: 0,
      explanation:
        "Causal attention scores every earlier position directly, so input 1 reaches output 8 in one hop and the mask hides later positions. A recurrent net also reaches input 1, but through 8 steps that run in order, and a one-layer kernel-3 convolution sees only its immediate neighbours.",
      objective: 1,
    },
    {
      prompt:
        "In Who sees whom, Per-position MLP fills only the diagonal. In a transformer block, which part moves information between positions?",
      options: [
        "The per-position MLP, because it runs at every position and so connects all of them together",
        "The residual connection, which copies each position's vector into every other position as well",
        "The shared weights, because every position uses them and so ends up holding the same information",
        "Attention, which lets positions read other allowed positions; the MLP rewrites each one alone",
      ],
      answer: 3,
      explanation:
        "A transformer block pairs attention, which mixes across positions, with a per-position MLP, which transforms each position independently and so shows only a diagonal. Sharing weights does not mix values between positions, and a residual connection adds a position's own input back to its own output. So a transformer is one particular wiring, attention plus a per-position MLP, not a synonym for neural network.",
      objective: 1,
    },
    {
      prompt:
        "In Sparse experts, with Experts per token at 2, you raise Experts from 8 to 16. What happens to the two weight counts?",
      options: [
        "Both double, because every extra expert also adds work for each token that gets routed to it",
        "Total expert weights double while Active per token stays put, as each token still uses two experts",
        "Active per token doubles while Total expert weights stays put, because tokens are spread over more experts",
      ],
      answer: 1,
      explanation:
        "Each expert is its own MLP, so storing twice as many experts doubles the stored weights. A token is still routed to only the top k experts, so the weights it actually uses depend on k, not on how many experts exist. That is the point of a mixture of experts: capacity grows with the number of experts while compute per token does not.",
      objective: 2,
    },
    {
      prompt:
        "In Sparse experts some experts get no tokens, and the load row shows 0 for them. What can you conclude from this lab?",
      options: [
        "Only that this untrained, seeded router leaves some stored experts unused; trained routers add a load-balancing loss",
        "That those experts have learned nothing useful and could be deleted without changing what the model does",
        "That the other experts have specialized on particular strokes of the digit while these have not yet",
        "That the router has learned to skip the tokens it judges unimportant",
      ],
      answer: 0,
      explanation:
        "The router here is an untrained linear layer with fixed seeded weights, so which experts receive tokens is arbitrary rather than learned. It shows the mechanism: weights can be stored and never used. In real mixtures a trained router learns which tokens suit which experts, and a load-balancing loss keeps any expert from sitting idle or being overloaded. The lab says nothing about what an expert specializes in.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateArchitectureZoo,
};

export default definition;
