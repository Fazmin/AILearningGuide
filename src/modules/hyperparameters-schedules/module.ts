import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateHyperState, initialState } from "./state";

/**
 * Re-exported so the state migration stays importable from the module, as it always was. The bounds and
 * the version history live in state.ts.
 */
export { hydrateHyperState };

const definition: ModuleDefinition = {
  id: "module-19-hyperparameters-schedules",
  slug: "hyperparameters-schedules",
  title: "Hyperparameters & schedules",
  group: "training-adapting",
  order: 39,
  icon: "SlidersHorizontal",
  accent: "#b8703a",
  prerequisites: ["module-17-training-run"],
  estimatedMinutes: 16,
  steps: [
    "Shape the schedule",
    "Compare four runs",
    "Break it",
    "Clip the gradients",
    "Read the seed band",
    "Explode a deep gradient",
  ],
  stepInstructions: [
    "Switch between the four schedules and watch the step-size curve and its shaded phases change shape, then drag Inspect epoch through the warmup before looking at any loss.",
    "Compare the four loss curves at the default peak rate, then at a peak rate of about 6, and note that the winner changes.",
    "Raise Peak learning rate to about 40, where the constant curve crosses the dotted baseline and the run does worse than knowing nothing.",
    "Leave the peak rate high, set Batch size to 2, then set Gradient clip to 0.10 and compare with clipping off. Raise it toward 1.4 and watch the rescue fade as the cap stops binding.",
    "Switch Seeds to Five seeds. At the default peak, compare the shaded bands and the Spread column with the gaps between the three decaying schedules, then set Peak learning rate to 30 and read the Constant band.",
    "In Gradient through depth, leave Layers at 24 and Gain per layer at 1.20 and read Input ÷ output gradient. Set Depth clip to 1.0 and compare the lengths before and after, then lower the gain below 1 and see what the clip can still do.",
  ],
  stateVersion: 3,
  tagline:
    "Four learning-rate schedules trained side by side on the same data and the same seed, up to the peak rate where a constant schedule stops working.",
  objectives: [
    "Read a learning-rate schedule as a function of training progress",
    "Explain why decay lets you use a larger peak learning rate",
    "Tell an overshooting run from a diverging one, and see what a gradient clip changes",
    "Judge whether a difference between two runs is larger than the spread that the shuffle seed alone produces",
    "Explain how a gradient can explode or vanish with depth, and what a norm clip can and cannot fix",
  ],
  glossary: [
    {
      term: "Learning rate",
      definition:
        "The multiplier on each gradient step. It is the hyperparameter that most often decides whether a run converges at all.",
    },
    {
      term: "Schedule",
      definition:
        "A rule setting the learning rate from training progress rather than keeping it fixed. Reported here as a multiplier on the peak rate.",
    },
    {
      term: "Warmup",
      definition:
        "Starting the rate small and ramping it to the peak over the first part of the run. It keeps early steps short while the loss surface is steep and, with Adam, while the running gradient statistics are still built from a handful of batches.",
    },
    {
      term: "Cosine decay",
      definition:
        "Easing the rate down along half a cosine period, from the peak to a small floor. Small final steps let the model settle rather than bounce around a minimum.",
    },
    {
      term: "One cycle",
      definition:
        "Warm up to a peak, then decay below the starting rate. A single rise and fall for the whole run.",
    },
    {
      term: "Gradient clipping",
      definition:
        "Rescaling a batch gradient whose norm exceeds a threshold down to that threshold. The direction is preserved and only the length is capped, so it limits how far one step can move without changing which way it moves.",
    },
    {
      term: "Gradient norm",
      definition:
        "The length of the gradient vector for one batch. A sudden spike is the signature of an update about to destroy the weights.",
    },
    {
      term: "Weight decay",
      definition:
        "A pull toward zero, subtracting a small fraction of each weight at every step. For plain SGD this equals an L2 penalty; with Adam the two differ, which is why AdamW applies the decay separately from the adaptive gradient step.",
    },
    {
      term: "Overshooting",
      definition:
        "Taking a step longer than the local slope justified, so the loss rises instead of falling. Repeated overshooting is what a rising loss curve is showing.",
    },
    {
      term: "Divergence",
      definition:
        "Loss growing without bound, usually ending in non-finite weights. Once weights are not finite, no later step can recover them.",
    },
    {
      term: "Seed",
      definition:
        "The number fixing every random choice. Here that is only the shuffle order, because the weights start at zero. Holding it constant makes two runs comparable, but one seed is still one sample.",
    },
    {
      term: "Seed band",
      definition:
        "The lowest-to-highest range of a result across several seeds of the same recipe. A difference between two runs that is smaller than the band cannot be told from the noise of the shuffle.",
    },
    {
      term: "Exploding gradient",
      definition:
        "A gradient whose length grows layer by layer on the way back because each layer multiplies it by more than one. A norm clip rescales it to a threshold. The opposite case, a vanishing gradient, is not cured by a clip, which can only shorten.",
    },
  ],
  references: [
    {
      authors: "Varun Godbole, George E. Dahl, Justin Gilmer, et al.",
      title: "Deep Learning Tuning Playbook",
      source: "Google Research, GitHub repository",
      year: 2023,
      url: "https://github.com/google-research/tuning_playbook",
      note: "A practical guide from Google researchers. It says some decaying schedule beats a constant rate, with linear or cosine decay as good defaults, and lists warmup and gradient clipping as fixes for unstable runs. It also warns that runs differing only in seed or data shuffle can disagree, the point of the Read the seed band step.",
    },
    {
      authors: "Ilya Loshchilov and Frank Hutter",
      title: "SGDR: Stochastic Gradient Descent with Warm Restarts",
      source: "International Conference on Learning Representations (ICLR 2017)",
      year: 2017,
      url: "https://arxiv.org/abs/1608.03983",
      note: "Gives the cosine annealing formula that eases the rate from a maximum down to a minimum along half a cosine wave. That is the Cosine decay shape on Learning-rate schedule shapes, which here bottoms out at 3 percent of the peak.",
    },
    {
      authors: "Priya Goyal, Piotr Dollár, Ross Girshick, et al.",
      title: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
      source: "arXiv preprint arXiv:1706.02677",
      year: 2017,
      url: "https://arxiv.org/abs/1706.02677",
      note: "Introduces gradual warmup, a ramp from a small rate up to the full rate over the first epochs, like the climb that Warmup fraction controls. It also gives the linear scaling rule from Going deeper: when the batch grows by some factor, grow the learning rate by the same factor.",
    },
    {
      authors: "Leslie N. Smith and Nicholay Topin",
      title: "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates",
      source: "Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications (SPIE 2019)",
      year: 2019,
      url: "https://arxiv.org/abs/1708.07120",
      note: "Names the 1cycle policy: one rise to a large maximum rate and one fall, ending well below the starting rate. This is the One cycle schedule in the lab, and the paper's case that one cycle makes large rates usable matches what the Compare four runs step shows.",
    },
    {
      authors: "Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio",
      title: "On the difficulty of training recurrent neural networks",
      source: "Proceedings of the 30th International Conference on Machine Learning (ICML 2013), PMLR 28, 1310–1318",
      year: 2013,
      url: "https://proceedings.mlr.press/v28/pascanu13.html",
      note: "Explains exploding and vanishing gradients as a product of many factors on the way back, the compounding the Gradient through depth card draws. Its fix is the norm clip used here: when the gradient's length is above a threshold, rescale it to that length and keep its direction.",
    },
    {
      authors: "PyTorch Contributors",
      title: "torch.nn.utils.clip_grad_norm_",
      source: "PyTorch documentation",
      year: 2026,
      url: "https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html",
      note: "The standard clipping function. It measures one total norm across the gradients of all parameters, as if they were a single vector, and scales them all down together. That is why Depth clip scales every layer by the same factor, and what GPT-3 and Llama 2 mean by a clip of 1.0 on the global norm.",
    },
    {
      authors: "Sam McCandlish, Jared Kaplan, Dario Amodei, et al.",
      title: "An Empirical Model of Large-Batch Training",
      source: "arXiv preprint arXiv:1812.06162",
      year: 2018,
      url: "https://arxiv.org/abs/1812.06162",
      note: "Shows that the noise in a batch gradient shrinks as the batch grows, and that the best learning rate rises with batch size, at first linearly for plain SGD. This backs the lesson's point that learning rate and batch size move together, and why Batch size 2 is where the Constant run does worst.",
    },
    {
      authors: "Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, et al.",
      title: "Accounting for Variance in Machine Learning Benchmarks",
      source: "Proceedings of Machine Learning and Systems 3 (MLSys 2021), 747–769",
      year: 2021,
      url: "https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html",
      note: "Measures how much results move with random choices such as data order and starting weights, and warns that a comparison can end up reflecting data order rather than a real improvement. It recommends varying these random sources across repeated runs before saying one method beats another, the idea behind Five seeds and the seed band.",
    },
    {
      authors: "Diederik P. Kingma and Jimmy Ba",
      title: "Adam: A Method for Stochastic Optimization",
      source: "International Conference on Learning Representations (ICLR 2015)",
      year: 2015,
      url: "https://arxiv.org/abs/1412.6980",
      note: "Defines Adam, which keeps running averages of each weight's gradient and squared gradient and sizes each step from them. This is the update method the lesson contrasts with the lab's plain gradient descent, and the reason its peak rates are not comparable with Adam's.",
    },
    {
      authors: "Liyuan Liu, Haoming Jiang, Pengcheng He, et al.",
      title: "On the Variance of the Adaptive Learning Rate and Beyond",
      source: "International Conference on Learning Representations (ICLR 2020)",
      year: 2020,
      url: "https://arxiv.org/abs/1908.03265",
      note: "Studies why warmup helps Adam. Early in training, Adam's step sizes are built from very few gradients and vary too much, and warmup works by holding those early steps back. This backs the lesson's claim that warmup matters far more with Adam than in this lab.",
    },
    {
      authors: "Ilya Loshchilov and Frank Hutter",
      title: "Decoupled Weight Decay Regularization",
      source: "International Conference on Learning Representations (ICLR 2019)",
      year: 2019,
      url: "https://arxiv.org/abs/1711.05101",
      note: "Shows that weight decay and an L2 penalty give the same step for plain SGD but not for Adam, and proposes AdamW, which applies the decay separately. This is the Weight decay glossary entry and the AdamW note in Going deeper.",
    },
    {
      authors: "Alexander Hägele, Elie Bakouch, Atli Kosson, et al.",
      title: "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2405.18392",
      note: "Points out that cosine decay needs the run length fixed in advance, and tests the alternative of a constant rate followed by a short final cooldown, also called warmup-stable-decay. It finds this matches cosine, which backs the lesson's section on the shape large runs report.",
    },
    {
      authors: "Greg Yang, Edward J. Hu, Igor Babuschkin, et al.",
      title: "Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer",
      source: "Advances in Neural Information Processing Systems 34 (NeurIPS 2021), 17084–17097",
      year: 2021,
      url: "https://proceedings.neurips.cc/paper/2021/hash/8df7c2e3c3c3be098ef7b382bd2c37ba-Abstract.html",
      note: "Shows how to tune settings such as the learning rate on a small model and carry them over to a much larger one without tuning it again. This is the one-shot problem from Going deeper: frontier settings come from smaller stand-in models plus a transfer rule.",
    },
    {
      authors: "Tom B. Brown, Benjamin Mann, Nick Ryder, et al.",
      title: "Language Models are Few-Shot Learners",
      source: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 1877–1901",
      year: 2020,
      url: "https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html",
      note: "The GPT-3 paper. Its training details give the real figures in Toy versus real: Adam with warmup, cosine decay to 10 percent of the peak, weight decay of 0.1, a clip of 1.0 on the global gradient norm, and peak rates by model size.",
    },
    {
      authors: "Hugo Touvron, Louis Martin, Kevin Stone, et al.",
      title: "Llama 2: Open Foundation and Fine-Tuned Chat Models",
      source: "arXiv preprint arXiv:2307.09288",
      year: 2023,
      url: "https://arxiv.org/abs/2307.09288",
      note: "The Llama 2 report. Its pretraining section lists a cosine schedule with 2,000 warmup steps, decay to 10 percent of the peak, weight decay of 0.1 and a gradient clip of 1.0, and Table 1 gives the peak rates the lesson quotes.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "At a peak rate of 0.6 the constant schedule wins. At a peak rate of 30 it is the worst of the four. What changed?",
      options: [
        "At a large peak, decay can still settle because its steps shrink, while constant steps keep overshooting",
        "Decaying schedules always beat constant ones, so the low-rate result was only a measurement error that a repeat run would remove",
        "The higher rate changed which schedule the shuffle seed favours, so the winner is a matter of chance",
      ],
      answer: 0,
      explanation:
        "At a small peak rate, decay just wastes budget: the model is still underfit when the steps get small. At a large peak rate, constant steps overshoot for the whole run while decay converts the same aggressive start into a settled finish. Schedules are what make large peak rates usable.",
      objective: 1,
    },
    {
      prompt: "You drag Warmup fraction from small to large. Which of the four loss curves can change?",
      options: [
        "All four, because warmup is applied to the rate before any schedule gets to shape it",
        "Only One cycle, because it is the only one of the four schedules that has a rising phase",
        "Only Warmup + decay and One cycle, because Constant and Cosine decay ignore the setting",
        "Only Constant, because it has no decay phase that could hide the effect of the warmup",
      ],
      answer: 2,
      explanation:
        "Each schedule is a function of progress through the run. Constant and Cosine decay use only the peak rate and progress, so Warmup fraction never enters them. Warmup + decay and One cycle both ramp up over that fraction, so moving it reshapes their rate curves and therefore their losses.",
      objective: 0,
    },
    {
      prompt:
        "At a very high peak rate and Batch size 2, the Constant run ends far worse than knowing nothing. You turn on a small Gradient clip. What happens?",
      options: [
        "Nothing changes, because clipping only helps runs whose gradients explode, and the gradients in this model never explode",
        "The final loss falls, because the cap limits how far one step can travel even when no gradient was unusual",
        "The loss rises, because clipping throws away the gradient's direction along with its length",
      ],
      answer: 1,
      explanation:
        "Clipping rescales a batch gradient whose norm exceeds the threshold, keeping its direction and capping its length. That limits how far a single step can travel, which is the damage overshooting does, even though this model's gradients are bounded to begin with. It helps most at Batch size 2, where each batch gives the noisiest estimate of the true gradient.",
      objective: 2,
    },
    {
      prompt:
        "The lesson says this model cannot produce a gradient explosion. Why, then, does the Constant run at a very high peak rate end worse than chance?",
      options: [
        "Its gradient norm spikes by orders of magnitude within a single step, just as it does in real transformer training",
        "Its weights become non-finite, and no later step can recover them, so the loss stays above chance",
        "The weight decay pulls the weights toward zero faster than the gradient can fit them",
        "Each step is longer than the slope justifies, so it overshoots the minimum while gradients stay bounded",
      ],
      answer: 3,
      explanation:
        "With a single linear layer and softmax cross-entropy, each row's gradient is a probability vector minus a one, so its length is bounded at every slider setting. High rates therefore produce repeated overshooting rather than divergence: the weights stay finite and the loss is merely terrible. Real deep networks can multiply many derivatives and genuinely spike, which is what clipping was built for.",
      objective: 2,
    },
    {
      prompt:
        "With Five seeds on at the default peak, the table puts One cycle ahead of Cosine decay on average, by far less than the spread across seeds. What should you conclude?",
      options: [
        "One cycle is the better schedule, because its mean is the lower of the two",
        "Neither can be ranked, because the gap is smaller than the band the seed alone produces",
        "Cosine decay is better, because the simpler schedule wins when the difference is small",
        "The ranking is exact, because the seed is fixed and the loss is a precise cross-entropy",
      ],
      answer: 1,
      explanation:
        "The two schedules finish about two ten-thousandths of a nat apart on average, while each moves by several thousandths between shuffle orders. A gap that small can flip with the seed, so one run, or five, cannot rank them. The gap between Constant and the decaying schedules at this peak is about fifty times the band, so that ordering is real.",
      objective: 3,
    },
    {
      prompt:
        "In Gradient through depth you set 40 layers and a gain of 0.8, so the first layers receive almost no gradient. What does raising Depth clip do about that?",
      options: [
        "It lifts the early layers' gradients up to the clip value, so that they can start learning again",
        "It scales every layer's gradient up equally, until the total reaches the clip value",
        "Nothing useful: a clip only shortens gradients, so the early layers still get almost none",
        "It makes the shrinking worse, because the clip applies its own gain at every layer",
      ],
      answer: 2,
      explanation:
        "A norm clip multiplies the whole gradient by min(1, threshold ÷ length). That factor is never above 1, so it can shorten an exploding gradient but cannot lengthen a vanishing one. At a gain of 0.8 the first layer sees under a thousandth of the output layer's gradient, and the clip leaves that ratio exactly as it was.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateHyperState,
};

export default definition;
