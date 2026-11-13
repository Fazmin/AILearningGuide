import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateMoeState, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-55-moe-training",
  slug: "moe-training",
  title: "Training a mixture of experts",
  group: "pretraining",
  order: 48,
  icon: "Route",
  accent: "#2f7a45",
  prerequisites: ["module-20-train-tiny-lm", "module-35-architecture-zoo", "module-52-parallel-training"],
  estimatedMinutes: 16,
  steps: ["Watch it collapse", "Add a balancing loss", "Nudge with a bias", "Cap each expert", "Read the specialists", "Count what runs"],
  stepInstructions: [
    "With Balancing on None, Experts at 4 and Top-k at 1, guess how the four experts will share the text after one epoch. Then read Expert load over training: one line climbs to 100%, Effective experts reads 1.00, and Loss sits beside Dense baseline.",
    "Switch Balancing to Auxiliary loss. Predict which Aux weight balances the experts and which one goes too far, then step from 0.01 to 0.1 to 1, comparing Effective experts, Loss and Mean gate against Dense baseline each time.",
    "Switch Balancing to Bias update and step Bias step from 0.001 to 0.03. Watch the per-expert bias bars, Effective experts and Loss, and notice that Mean gate does not move.",
    "Set Capacity factor to 1.25 and guess how many choices a collapsed router will lose. Then switch Balancing between None and Auxiliary loss at Aux weight 0.1, comparing Dropped tokens and Loss.",
    "With Balancing on Auxiliary loss at 0.1, read the Specialisation grid with Grid on By previous character. Before you switch Grid to By corpus, ask what the router can actually see, then read Largest gap from the overall split.",
    "Raise Experts from 1 to 8 and Top-k from 1 to 2, comparing Total parameters, Active per token and Loss, then read the Mixtral 8x7B row.",
  ],
  stateVersion: 1,
  tagline:
    "Train a small mixture of experts on real text, watch its router hand every token to one expert, then rebalance it with an auxiliary loss or a bias, cap each expert, and count the weights each token actually runs.",
  objectives: [
    "Explain top-k routing: how router scores pick k experts and how gate values weight their outputs",
    "Recognise expert collapse as a rich-get-richer loop, and spot it in the load readouts",
    "Compare an auxiliary balancing loss with bias-update balancing, including what each costs in loss",
    "Predict the share of tokens a capacity factor drops, and why a cap does not balance the router",
    "Read what the experts specialise in, and what the router's input lets it tell apart",
    "Count total against active parameters, and connect them to memory and expert parallelism",
  ],
  glossary: [
    {
      term: "Mixture of experts",
      definition:
        "A layer that holds several separate sub-networks, the experts, and a router that sends each token to only a few of them. The layer stores every expert but runs only the chosen ones for each token.",
    },
    {
      term: "Router",
      definition:
        "The small learned layer that scores every expert for each token, here z = h times R, one score per expert. It is trained with the rest of the model, so where it sends tokens changes as training goes.",
    },
    {
      term: "Top-k routing",
      definition:
        "Running only the k experts with the highest scores for each token and skipping the rest. Switch Transformer uses k = 1 and Mixtral k = 2; compute per token grows with k, not with the number of experts.",
    },
    {
      term: "Gate value",
      definition:
        "The weight that scales a chosen expert's output, here the softmax of the router scores, as in Switch Transformer. Mixtral renormalises the gates over the chosen experts, and DeepSeek-V3 uses sigmoid scores instead.",
    },
    {
      term: "Expert load",
      definition:
        "An expert's share of the router's choices: how many tokens picked it, divided by tokens times k. An even split gives every expert 1/N.",
    },
    {
      term: "Expert collapse",
      definition:
        "Training ending with the router sending nearly every token to one expert, or a few. It feeds itself: the favoured expert trains on more tokens, improves, and is chosen even more, while the others never learn.",
    },
    {
      term: "Effective experts",
      definition:
        "The exponential of the entropy of the load shares, exp(−Σ s ln s). It reads 1.00 when one expert takes everything and N when all N share evenly, so it counts how many experts are really in use.",
    },
    {
      term: "Auxiliary load-balancing loss",
      definition:
        "An extra loss term, α times N times Σ f times P, that is smallest when tokens spread evenly; f is each expert's share of first choices and P its mean gate. Its gradient pulls the router toward balance but competes with the language-model loss.",
    },
    {
      term: "Auxiliary-loss-free balancing",
      definition:
        "Balancing with a per-expert bias added to the router scores only when picking experts: after each batch the bias of an overloaded expert goes down by a step and that of an underloaded one goes up. The gates still come from the scores alone, so no extra gradient reaches the model.",
    },
    {
      term: "Capacity factor",
      definition:
        "How many seats each expert gets per batch, relative to an even share: capacity = factor times tokens times k divided by experts. Above 1 it leaves slack for uneven routing; it caps the work but does not change where the router sends tokens.",
    },
    {
      term: "Dropped token",
      definition:
        "A token whose chosen expert was already full. Real models pass it on unchanged through the residual connection; this toy has none, so a token that loses every expert predicts all 30 characters equally.",
    },
    {
      term: "Active parameters",
      definition:
        "The weights one token actually uses: the shared parts plus its k experts. Compute per token follows the active count, while memory has to hold the total, every expert included.",
    },
    {
      term: "Expert parallelism",
      definition:
        "Spreading the experts of a layer across devices, so each GPU holds a few of them and tokens are sent to the device that holds their chosen experts and back. It is how models with many experts fit in memory, at the cost of that communication and of uneven load.",
    },
    {
      term: "Specialisation",
      definition:
        "Different experts ending up handling different kinds of tokens. What a router can specialise on is limited by what its input can tell apart, so a grouping is a finding only if the router could have grouped otherwise.",
    },
  ],
  references: [
    {
      authors: "Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton",
      title: "Adaptive Mixtures of Local Experts",
      source: "Neural Computation 3(1), 79–87",
      year: 1991,
      url: "https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf",
      note: "The original mixture of experts: several separate networks plus a gating network that learns which one handles each training case. It is the idea the lesson starts from, before sparse top-k routing made it a way to grow models cheaply.",
    },
    {
      authors: "Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, et al.",
      title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer",
      source: "International Conference on Learning Representations (ICLR 2017)",
      year: 2017,
      url: "https://arxiv.org/abs/1701.06538",
      note: "Introduces sparse top-k gating between LSTM layers. Section 4 reports that the gating network converges to the same few experts and calls the imbalance self-reinforcing, the collapse in Watch it collapse; Section 3.1 names the shrinking batch each expert sees.",
    },
    {
      authors: "Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, et al.",
      title: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding",
      source: "International Conference on Learning Representations (ICLR 2021)",
      year: 2021,
      url: "https://arxiv.org/abs/2006.16668",
      note: "Its top-2 gating seats every first choice before any second choice, passes overflowed tokens on through the residual connection, and builds the auxiliary loss from first-choice counts times mean gates, all as this lab does. It also shards the experts across devices.",
    },
    {
      authors: "William Fedus, Barret Zoph, and Noam Shazeer",
      title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity",
      source: "Journal of Machine Learning Research 23(120), 1–39",
      year: 2022,
      url: "https://jmlr.org/papers/v23/21-0998.html",
      note: "Routes each token to one expert scaled by its softmax gate, defines expert capacity as tokens per batch over experts times the capacity factor, and gives the auxiliary loss α · N · Σ f·P with α = 0.01, the weight that still collapses in this lab.",
    },
    {
      authors: "Barret Zoph, Irwan Bello, Sameer Kumar, et al.",
      title: "ST-MoE: Designing Stable and Transferable Sparse Expert Models",
      source: "arXiv preprint arXiv:2202.08906",
      year: 2022,
      url: "https://arxiv.org/abs/2202.08906",
      note: "A design guide for sparse models: it adds the router z-loss for stability, finds that higher capacity factors improve quality but cost compute and keeps 1.25 for training, and shows encoder experts specialising in punctuation, verbs and names while decoder experts do not.",
    },
    {
      authors: "Aidan Clark, Diego de las Casas, Aurelia Guy, et al.",
      title: "Unified Scaling Laws for Routed Language Models",
      source: "Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR 162, 4057–4086",
      year: 2022,
      url: "https://proceedings.mlr.press/v162/clark22a.html",
      note: "Treats parameter count and compute per token as two separate axes and fits scaling laws over both for routed models with up to hundreds of experts. It backs the lesson's point that total and active parameters are different dials.",
    },
    {
      authors: "Samyam Rajbhandari, Conglong Li, Zhewei Yao, et al.",
      title: "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale",
      source: "Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR 162",
      year: 2022,
      url: "https://arxiv.org/abs/2201.05596",
      note: "Points out that a mixture of experts needs far more memory than a dense model of similar quality: Switch-Base has 7.4B parameters against T5-Large's 0.74B and needs 8 to 10 GPUs just to hold them. It backs the lesson's point that memory follows the total, not the active count.",
    },
    {
      authors: "Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia",
      title: "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts",
      source: "Proceedings of Machine Learning and Systems 5 (MLSys 2023)",
      year: 2023,
      url: "https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html",
      note: "Notes that capacity forces a choice between dropping tokens and wasting compute on padding, and removes it with block-sparse kernels that never drop a token. It is the dropless alternative to the Capacity factor control.",
    },
    {
      authors: "Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, et al.",
      title: "Mixtral of Experts",
      source: "arXiv preprint arXiv:2401.04088",
      year: 2024,
      url: "https://arxiv.org/abs/2401.04088",
      note: "Mixtral 8x7B picks 2 of 8 experts per token in every layer, renormalising the gates over the two, and separates the sparse parameter count from the active one. Its routing analysis finds no obvious specialisation by topic, and it describes expert parallelism and its load problem.",
    },
    {
      authors: "Mistral AI",
      title: "Mixtral of experts",
      source: "Mistral AI news, 11 December 2023",
      year: 2023,
      url: "https://mistral.ai/news/mixtral-of-experts",
      note: "The release announcement gives the figures in the Mixtral 8x7B row: 46.7B parameters in total, 12.9B used per token. The paper rounds them to 47B and 13B.",
    },
    {
      authors: "Damai Dai, Chengqi Deng, Chenggang Zhao, et al.",
      title: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models",
      source: "arXiv preprint arXiv:2401.06066",
      year: 2024,
      url: "https://arxiv.org/abs/2401.06066",
      note: "Splits experts into many smaller ones and sets some aside as shared experts that every token uses, both aimed at making experts specialise. It is the design the lesson names when it says real models now route among many small experts.",
    },
    {
      authors: "Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai",
      title: "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts",
      source: "arXiv preprint arXiv:2408.15664",
      year: 2024,
      url: "https://arxiv.org/abs/2408.15664",
      note: "The Bias update method: a per-expert bias added before the top-k choice, moved by a fixed step against each expert's recent load, never added to the gate. At 1B and 3B parameters it reports lower perplexity and far better balance than an auxiliary loss, a result this toy does not reproduce.",
    },
    {
      authors: "DeepSeek-AI",
      title: "DeepSeek-V3 Technical Report",
      source: "arXiv preprint arXiv:2412.19437",
      year: 2024,
      url: "https://arxiv.org/abs/2412.19437",
      note: "Uses bias balancing at scale: the bias is only for routing, the gate comes from the original sigmoid score, and the bias step is 0.001 for the first 14.3T tokens. Each layer has 256 routed experts plus 1 shared, 8 run per token, 671B parameters in total and 37B active, and no token is dropped.",
    },
    {
      authors: "Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, et al.",
      title: "OLMoE: Open Mixture-of-Experts Language Models",
      source: "arXiv preprint arXiv:2409.02060",
      year: 2024,
      url: "https://arxiv.org/abs/2409.02060",
      note: "A fully open model with 64 experts per layer and 8 active, 7B parameters in total and 1B per token, trained dropless with an auxiliary loss weight of 0.01. Unlike Mixtral it finds strong domain specialisation, such as an expert that takes nearly all arXiv tokens.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "With Balancing on Bias update, one character's router scores rank E1 first, but after the biases are added E3 has the highest z + b. What happens to that token?",
      options: [
        "E3 runs, scaled by its own gate from the softmax of z, which the bias never touches",
        "E1 runs, because the bias only nudges the gate values and never changes which expert is picked",
        "E3 runs with a gate of exactly 1, because the bias overrides the router's probabilities",
        "E1 and E3 both run, because a disagreement between z and z + b adds a second expert",
      ],
      answer: 0,
      explanation:
        "The bias is added to the scores only to choose experts. The gate that scales the chosen expert's output is still p = softmax(z), computed without the bias, as in DeepSeek-V3. So the bias can move a token to E3, but E3's output is weighted by E3's own gate, which may be small until the router learns to agree. Top-k never adds experts: exactly k run.",
      objective: 0,
    },
    {
      prompt:
        "With Balancing on None, the four lines in Expert load over training start near an even split, and within one epoch a single line reaches 100%. What drives that?",
      options: [
        "That expert began with the biggest weights, so it was bound to win every token from the first step",
        "An expert that wins a few more tokens trains on more of them, gets better, and is picked even more",
        "The router copies the most frequent character, space, and sends everything to the expert space chose",
        "The other three experts were never wired to the output, so the router could not have chosen them",
      ],
      answer: 1,
      explanation:
        "Collapse is a feedback loop. The router learns from the gate gradient of the experts it picked, so an expert that is slightly ahead gets more tokens, trains its next-character bias on them, and lowers the loss more when chosen, which raises its router score for everyone. The starting split here is fairly even, 3.66 effective experts at seed 1, and the router's bias starts at zero, so the winner is made by training, not by its starting weights.",
      objective: 1,
    },
    {
      prompt:
        "On Auxiliary loss you raise Aux weight from 0.1 to 1. Effective experts stays near 4, yet Loss rises above Dense baseline and Mean gate falls. Why?",
      options: [
        "At weight 1 the router is switched off, so every token is sent to an expert chosen at random",
        "A heavier weight drops more tokens, and every dropped token predicts all 30 characters equally",
        "The balancing pull now outweighs the language-model loss and flattens the gates",
        "The experts stop training once the balance term reaches 1, so the run effectively ends early",
      ],
      answer: 2,
      explanation:
        "The auxiliary term and the cross-entropy pull on the same router. At weight 1 the balancing pull dominates and keeps the gates spread out, so each chosen expert's output is scaled down and the prediction gets worse: in all six seeds weight 1 ends above weight 0.1. Too small a weight fails the other way, as 0.01, Switch's value, still collapses here. Bias balancing sidesteps the trade by never touching the gates.",
      objective: 2,
    },
    {
      prompt:
        "Experts 4, Top-k 1, batches of 16, Capacity factor 1.25. If the router has collapsed so that one expert takes every token, about what share of tokens is dropped?",
      options: [
        "About 20%, because the cap lets each expert take 25% more than its even share",
        "About 69%, because that expert has 5 seats for every 16 tokens",
        "None, because a full expert passes its overflow to the router's next choice",
        "About 75%, because three of the four experts receive no tokens at all",
      ],
      answer: 1,
      explanation:
        "Seats per expert are ⌈1.25 × 16 × 1 ÷ 4⌉ = 5, so an expert that every token chooses can serve 5 of each 16 and drops 11, about 69%. Seed 4 of the lab collapses that far and drops 69%; seed 1 keeps two experts busy and drops 40%. A cap never reroutes a token, it only drops the overflow, so it limits the damage of imbalance without teaching the router to balance.",
      objective: 3,
    },
    {
      prompt:
        "With Grid on By corpus, each corpus splits across the experts within a few hundredths of the overall split. What does that show about the experts?",
      options: [
        "That they are generalists, which is what a well-trained mixture of experts should end up being",
        "That the balancing loss made every corpus use every expert by the same amount",
        "Nothing: the router sees only the previous character, so it cannot tell the corpora apart",
        "That these three corpora are too alike for any router, however much context it saw, to separate",
      ],
      answer: 2,
      explanation:
        "The router's only input is the previous character, so a corpus can only inherit the mix of its characters, and the three corpora share most characters. The close match is forced by the design, not found. To ask whether experts specialise by topic you need a router that could tell topics apart: Mixtral's analysis finds little topic specialisation, while OLMoE finds experts that take nearly all of one domain.",
      objective: 4,
    },
    {
      prompt: "You raise Experts from 4 to 8 with Top-k at 1. What happens to Total parameters and Active per token?",
      options: [
        "Total nearly doubles, while Active per token grows only by the router's extra weights",
        "Both double, because each token now consults twice as many experts",
        "Active per token doubles while Total stays put, because the experts share their weights",
        "Neither changes, because the eight experts split the same weights between them",
      ],
      answer: 0,
      explanation:
        "Each expert here is its own 5 × 30 map, 150 weights, so four more experts add 600 stored weights, plus 20 router weights: 740 becomes 1,360. A token still runs one expert, so Active per token moves only from 290 to 310, the router. Memory has to hold the total, which is why large models spread experts across GPUs: expert parallelism. Mixtral 8x7B stores 46.7B parameters and uses 12.9B per token.",
      objective: 5,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateMoeState,
};

export default definition;
