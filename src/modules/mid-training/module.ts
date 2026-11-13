import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateMidTrainingState, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-56-mid-training",
  slug: "mid-training",
  title: "Mid-training & data mixtures",
  group: "pretraining",
  order: 49,
  icon: "Blend",
  accent: "#7d7426",
  prerequisites: ["module-20-train-tiny-lm", "module-18-dataset-building", "module-44-synthetic-data"],
  estimatedMinutes: 16,
  steps: [
    "Set the mixture",
    "Upweight the scarce domain",
    "Repeat or find more",
    "Anneal on the target mix",
    "Mid-train or fine-tune?",
  ],
  stepInstructions: [
    "Under Mixture preset press Uniform, then General-heavy, and compare the Held-out loss by domain bars with the Epochs column of the Data mixture card.",
    "Drag Arithmetic weight from 0 to 8, then read the Gain per extra epoch and General held-out columns of the Arithmetic weight sweep.",
    "On Repeated data, keep Budget at 8,000 pairs and lower Unique sentences from 120 to 2, comparing Held-out loss, Epochs and Lowest along the run.",
    "Set Late phase to Target mix and compare Arithmetic held-out and General held-out in the Switched at decay and Spread evenly columns.",
    "Switch Late phase between Target mix and Target only, and compare General held-out and Forgetting in the Switched at decay column.",
  ],
  stateVersion: 1,
  tagline:
    "Split one training budget across four domains, repeat the scarce one, and switch to a target-heavy mix while the learning rate decays.",
  objectives: [
    "Read a data mixture as shares of a fixed token budget, and turn a weight into effective epochs",
    "Explain why repeating scarce data helps at first, then stops helping, and with very little data starts to hurt",
    "Explain why labs switch to a target-heavy mixture while the learning rate decays",
    "Tell mid-training on a target mix from training on the target alone by what each costs the general domains",
    "Say what per-domain held-out loss can and cannot tell you about a capability",
  ],
  glossary: [
    {
      term: "Data mixture",
      definition:
        "The share of the training budget each domain receives. Filtering decides which text is allowed in; the mixture decides how often each domain is seen, including deliberate repeats of a small one.",
    },
    {
      term: "Mixture weight",
      definition:
        "A number per domain whose ratio to the others sets that domain's share of every batch. Only the ratios matter, so weights of 3, 3, 3 and 1 and weights of 6, 6, 6 and 2 are the same mixture.",
    },
    {
      term: "Domain",
      definition:
        "A source of text with its own style and content, such as web pages, code or math, and here Harbor, Recipes, Proverbs and Arithmetic. Mixtures are set per domain because domains differ in size, quality and what they teach.",
    },
    {
      term: "Per-domain held-out loss",
      definition:
        "Cross-entropy on held-out text from one domain alone. It shows which domains a mixture helps or hurts, but it scores how well the model predicts that text, not whether it has a skill.",
    },
    {
      term: "Token budget",
      definition:
        "The number of tokens a run trains on, fixed before it starts. Here it is 600 steps of 16 character pairs, 9,600 pairs, so a larger share for one domain is a smaller share for the others.",
    },
    {
      term: "Effective epochs",
      definition:
        "Pairs drawn from a domain divided by the pairs it holds: how many passes over that domain the budget bought. Equal weights give unequal epochs, because domains differ in size.",
    },
    {
      term: "Data-constrained training",
      definition:
        "Training with a larger budget than there is unique data, so some text is seen more than once. The first few passes are worth nearly as much as new text; later passes are worth less each time, and on very little data they make held-out loss worse.",
    },
    {
      term: "Upsampling",
      definition:
        "Giving a domain a larger share of the budget than its share of the available text, so it is repeated more often. It lets a small, valuable domain matter, at the cost of repetition.",
    },
    {
      term: "Warmup-stable-decay",
      definition:
        "A learning-rate schedule that ramps up, holds one rate for most of the run, then falls to zero over a final stretch, here the last 20% of steps. A run can be extended or branched before the decay without restarting.",
    },
    {
      term: "Annealing",
      definition:
        "The final stretch of training in which the learning rate decays toward zero. Labs often change the data mixture at this point, toward the data they most want the model to end on.",
    },
    {
      term: "Mid-training",
      definition:
        "A stage between pretraining and fine-tuning that keeps training on a mixture, usually during the decay, with much more of the target data, such as math, code or high-quality text, mixed with general text. Keeping general text in the mix limits forgetting.",
    },
    {
      term: "Curriculum",
      definition:
        "An order for presenting data instead of one fixed mixture throughout. A late switch to a target mix is the simplest curriculum: one mixture for most of the run and another at the end.",
    },
    {
      term: "Forgetting",
      definition:
        "Held-out loss on earlier domains rising after training shifts toward new data. Here it is General held-out at the end minus General held-out at the switch, two checkpoints of the same run.",
    },
  ],
  references: [
    {
      authors: "Niklas Muennighoff, Alexander M. Rush, Boaz Barak, et al.",
      title: "Scaling Data-Constrained Language Models",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2305.16264",
      note: "Finds that up to 4 epochs of repeated data change loss very little compared with unique data, and that with more repetition the value of extra compute decays to zero. It is the result the Repeated data card tests at toy scale.",
    },
    {
      authors: "Danny Hernandez, Tom Brown, Tom Conerly, et al.",
      title: "Scaling Laws and Interpretability of Learning from Repeated Data",
      source: "arXiv preprint arXiv:2205.10487",
      year: 2022,
      url: "https://arxiv.org/abs/2205.10487",
      note: "Shows that repeating 0.1% of the data 100 times can bring an 800-million-parameter model down to the level of one half its size. It backs the lesson's point that real models are hurt by repetition through memorization, which a bigram cannot do.",
    },
    {
      authors: "Sang Michael Xie, Hieu Pham, Xuanyi Dong, et al.",
      title: "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2305.10429",
      note: "Sets domain weights with a small proxy model trained by group distributionally robust optimization, and reports better perplexity across all domains of The Pile, 6.5 points higher average few-shot accuracy, and baseline accuracy in 2.6 times fewer steps. It is the automatic weight search in Going deeper.",
    },
    {
      authors: "Jiasheng Ye, Peiju Liu, Tianxiang Sun, et al.",
      title: "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance",
      source: "International Conference on Learning Representations (ICLR 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2403.16952",
      note: "Fits functions that predict loss from mixture proportions on small runs and uses them to choose a mixture before the large run. It is the scaled-up version of reading the Arithmetic weight sweep and picking a weight from it.",
    },
    {
      authors: "Team OLMo, Pete Walsh, Luca Soldaini, et al.",
      title: "2 OLMo 2 Furious",
      source: "arXiv preprint arXiv:2501.00656",
      year: 2024,
      url: "https://arxiv.org/abs/2501.00656",
      note: "Names its shorter second stage mid-training, 5 to 10 percent of training compute, on Dolmino Mix 1124 (filtered web, academic and encyclopedic text plus math) while the learning rate decays linearly to zero, and averages several such runs. The 7B model pretrained on about 4 trillion tokens, and its GSM8K score rose from 24.1 to 67.5 through mid-training, the real version of the Target mix late phase.",
    },
    {
      authors: "Shengding Hu, Yuge Tu, Xu Han, et al.",
      title: "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies",
      source: "arXiv preprint arXiv:2404.06395",
      year: 2024,
      url: "https://arxiv.org/abs/2404.06395",
      note: "Introduces the Warmup-Stable-Decay schedule, finds a decay over about 10 percent of tokens enough, and reports a steep loss drop once the decay starts. It also mixes high-quality data into the decay and finds that better than adding it only at fine-tuning.",
    },
    {
      authors: "Alexander Hägele, Elie Bakouch, Atli Kosson, et al.",
      title: "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2405.18392",
      note: "Finds that a constant learning rate followed by a cooldown scales as reliably as cosine decay. That is why a warmup-stable-decay run can be extended, or branched into a decay with a new mixture, without fixing its length in advance.",
    },
    {
      authors: "Cody Blakeney, Mansheej Paul, Brett W. Larsen, et al.",
      title: "Does your data spark joy? Performance gains from domain upsampling at the end of training",
      source: "arXiv preprint arXiv:2406.03476",
      year: 2024,
      url: "https://arxiv.org/abs/2406.03476",
      note: "Upsamples domain data in the final stretch of a 7B model trained on 1 trillion tokens and gains up to 6.90 points on MMLU, 8.26 on GSM8K and 6.17 on HumanEval, finding 10 to 20 percent of training the best trade-off with general ability. It backs the Target mix comparison and its cost to General.",
    },
    {
      authors: "Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al.",
      title: "The Llama 3 Herd of Models",
      source: "arXiv preprint arXiv:2407.21783",
      year: 2024,
      url: "https://arxiv.org/abs/2407.21783",
      note: "Reports a pretraining mix of roughly 50 percent general knowledge, 25 percent math and reasoning, 17 percent code and 8 percent multilingual, and anneals the rate to zero over the last 40 million tokens while upsampling very high-quality sources. Annealing on high-quality code and math raised the 8B model's GSM8k by 24.0 percent and MATH by 6.4 percent but did little for the 405B model, and annealing a half-trained 8B model over 40 billion tokens, 30 percent of them from a candidate dataset, is how it judged new data.",
    },
    {
      authors: "Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, et al.",
      title: "SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model",
      source: "arXiv preprint arXiv:2502.02737",
      year: 2025,
      url: "https://arxiv.org/abs/2502.02737",
      note: "Trains in stages that mix web text with math, code and instruction data, and refines each stage's mixing rates by hand from the previous stage's results. It is the multi-stage curriculum the lesson describes beyond a single late switch.",
    },
    {
      authors: "Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, et al.",
      title: "Simple and Scalable Strategies to Continually Pre-train Large Language Models",
      source: "arXiv preprint arXiv:2403.08763",
      year: 2024,
      url: "https://arxiv.org/abs/2403.08763",
      note: "Finds that new data degrades performance on earlier data, and that re-warming and re-decaying the learning rate with replay of earlier data matches retraining from scratch. It backs the lesson's point that keeping general text in a late mix is what limits forgetting.",
    },
    {
      authors: "Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, et al.",
      title: "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks",
      source: "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)",
      year: 2020,
      url: "https://arxiv.org/abs/2004.10964",
      note: "Shows that a second phase of pretraining on in-domain text improves tasks in that domain. It is the earlier, target-only form of continued pretraining that the Target only late phase imitates.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "On Data mixture you press Uniform, then General-heavy. Every general domain gets more epochs, yet Proverbs held-out barely moves while Harbor improves. Why?",
      options: [
        "Proverbs' weight is the one that stayed the same, so its share of every batch did not change",
        "Proverbs is the smallest domain, so it already had the most passes and more add little",
        "Raising the general weights lowers the learning rate on each general domain's batches",
        "Proverbs' held-out sentences joined its training set once its weight rose from 1 to 3",
      ],
      answer: 1,
      explanation:
        "A weight becomes a share of every batch, a share of the fixed 9,600-pair budget becomes pairs drawn, and pairs drawn divided by a domain's size become its epochs. Proverbs holds only 195 training pairs, so Uniform already gives it 12.48 epochs; General-heavy pushes that to 14.72 and its held-out loss goes from 2.400 to 2.408. Harbor, with 378 pairs, still gains, from 2.197 to 2.097. Equal weights never mean equal epochs.",
      objective: 0,
    },
    {
      prompt:
        "On Repeated data, with Budget at 8,000 pairs, you lower Unique sentences from 120 to 2. What does the held-out loss do along the run?",
      options: [
        "It ends lowest of all, because two sentences seen many times are learned perfectly",
        "It stays the same, because the budget alone sets how much the model can learn",
        "It falls at first, bottoms out after about 27 epochs, then climbs to the end",
        "It never leaves 3.401, because two sentences are too few to learn anything from",
      ],
      answer: 2,
      explanation:
        "The same 8,000 pairs are spent on 59 pairs of text, about 136 passes over each sentence. The early passes teach letter pairs the held-out facts share, so held-out loss falls to 2.217. After that the table keeps fitting those two sentences and drifts from everything else, ending at 2.542 against 1.388 with all 120 sentences. Repetition helps at first, stops helping, and on very little data hurts.",
      objective: 1,
    },
    {
      prompt:
        "You set Late phase to Target mix. Spread evenly draws the same Arithmetic examples on average, in one constant mixture. How do the two runs end?",
      options: [
        "The same, because both runs see the same examples and only their order differs",
        "Switched ends lower on Arithmetic and a little higher on General, since the last batches count most",
        "Switched ends higher on Arithmetic, because its examples arrive when the rate is near zero",
        "Switched ends lower on General, because the decay shields the general domains from the extra Arithmetic",
      ],
      answer: 1,
      explanation:
        "The switched run draws 1,721 Arithmetic pairs and the even one 1,707, so the comparison isolates timing. Ending on a mix that is half Arithmetic leaves Arithmetic held-out at 1.911 against 2.110, and General at 2.398 against 2.339. Every step moves the same shared weights, so the batches near the end decide where they settle. That is the bet behind annealing on target data.",
      objective: 2,
    },
    {
      prompt:
        "You switch Late phase from Target mix to Target only. Why does General held-out rise so much further?",
      options: [
        "Target only runs fewer steps in all, so the general domains end the run undertrained",
        "Target only raises the learning rate for the late phase, so every step moves further",
        "No general text is in the late batches, so nothing in the loss holds those domains",
        "Target only skips the decay, so the general domains never get the chance to settle down",
      ],
      answer: 2,
      explanation:
        "Both runs take 600 steps on the same schedule; only the late batches differ. Forgetting is +0.353 with Target only against +0.060 with Target mix, because one table serves every domain and a loss that mentions only Arithmetic pulls every shared letter pair toward arithmetic text. Keeping general text in the mix keeps it in the loss. That is mid-training on a mixture versus training on the target alone, the second of which the next lab, Fine-tuning & transfer, studies in depth.",
      objective: 3,
    },
    {
      prompt:
        "As Arithmetic weight goes from 0 to 8, Arithmetic held-out falls from 2.832 to 1.706, yet at weight 8 the Answer probe gets 0 of 40 facts right. What did the loss improvement measure?",
      options: [
        "Better prediction of how arithmetic sentences are spelled and laid out, which needs no sums",
        "Real skill at adding, hidden because the probe asks only about facts the model never saw",
        "Nothing real, because lower loss always means the right answer ranks higher, so the probe must be wrong",
        "Memorization of the 120 training facts, which is why only the new facts come out wrong",
      ],
      answer: 0,
      explanation:
        "A bigram predicts each character from the one before it. Most of an arithmetic sentence is number words, plus, minus and is, and learning those spellings is what lowers the loss. The character before every answer is the same space, so the table ranks the 21 answers identically for every question and gives ten each time. Per-domain loss measures prediction of a domain's text; a capability needs its own test on held-out problems.",
      objective: 4,
    },
    {
      prompt:
        "With Target mix on, you change Schedule from Warmup-stable-decay to Constant rate. Arithmetic ends lower still and General ends worse. What does that say about this toy?",
      options: [
        "The decay is what makes late data count, so a constant rate should have helped Arithmetic less",
        "A constant rate has no warmup, so the general domains never trained properly in the first place",
        "Here late data helps mostly by coming last, and the decay buys keeping more of General",
        "Constant rate doubles the Arithmetic examples in the late phase, so Arithmetic gains twice as much",
      ],
      answer: 2,
      explanation:
        "At a constant rate Target mix ends at 1.785 on Arithmetic against 1.911 with the decay, and at 2.462 on General against 2.398, while forgetting doubles from +0.060 to +0.127. Full-size steps at the end pull the weights toward the latest batches, which is recency, and a decaying rate holds them back. Published annealing results report gains from target data during the decay; this toy reproduces the switch's advantage over spreading the data evenly, but not an extra boost from the decay itself.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateMidTrainingState,
};

export default definition;
