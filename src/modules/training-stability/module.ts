import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateStabilityState, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-54-training-stability",
  slug: "training-stability",
  title: "Keeping a run stable",
  group: "pretraining",
  order: 47,
  icon: "Siren",
  accent: "#1f7a8c",
  prerequisites: ["module-53-optimizers", "module-20-train-tiny-lm"],
  estimatedMinutes: 16,
  steps: [
    "Read a healthy dashboard",
    "Push the learning rate",
    "Overflow in fp16",
    "Survive a bad batch",
    "Tame logit growth",
  ],
  stepInstructions: [
    "With Fault on None, read the shaded normal range on each of the four panels: Loss, Gradient norm, Max logit and Update ÷ weight.",
    "Set Fault to Learning rate too high and watch Gradient norm climb until the run halts, then set Gradient clip to 0.50, then set Schedule to Warmup + cosine.",
    "Set Fault to fp16 overflow with Loss scale at 2²⁰, find the overflow triangle and the halt, then switch Scaling to Dynamic and read Skipped steps.",
    "Set Fault to Garbage batch and compare the Spike readout with Gradient clip off and at 1.00, then set the clip back to off and press Roll back and skip.",
    "Set Fault to Logit growth, read Max logit, raise z-loss to 0.01, then lower Learning rate to 0.1.",
  ],
  stateVersion: 1,
  tagline:
    "One tiny model broken four ways, read on a dashboard of loss, gradient norm, max logit and update size, and repaired with the fixes large runs use: clipping, warmup, loss scaling, rollback and z-loss.",
  objectives: [
    "Read a training dashboard's four signals, loss, gradient norm, max logit and update-to-weight ratio, against their normal range",
    "Explain why the two-matrix model diverges at a high learning rate, and what gradient clipping and warmup each change",
    "Explain fp16's narrow range, why gradients are multiplied by a loss scale, and how dynamic scaling survives an overflow",
    "Recognize a loss spike and recover from it by rolling back to a checkpoint and skipping the batches that caused it",
    "Explain how logits grow without bound, and what z-loss can and cannot do about it",
  ],
  glossary: [
    {
      term: "Loss spike",
      definition:
        "A sudden jump in the training loss, which the run may or may not recover from. In large runs a spike often arrives with a jump in gradient norm, and it can come from particular batches meeting a particular state of the weights rather than from bad data alone.",
    },
    {
      term: "Divergence",
      definition:
        "Training whose loss grows without bound until a weight or the loss stops being a finite number, after which no later step can recover. This lab halts at that step and prints the step instead of the number.",
    },
    {
      term: "Gradient norm",
      definition:
        "The length of the whole batch gradient: every parameter's gradient squared, summed, then square-rooted. It is what a clip reads, and on a log scale an exponential climb shows as a straight line.",
    },
    {
      term: "Gradient clipping",
      definition:
        "Rescaling any batch gradient whose norm exceeds a threshold to exactly that norm, keeping its direction. It bounds how far one step can move the weights, but it does not make a too-large learning rate a good one, and under Adam its effect is weaker because each weight's step is divided by its own recent gradient size.",
    },
    {
      term: "Update-to-weight ratio",
      definition:
        "How far one step moved the weights as a share of their size: the length of the change divided by the length of the weights. A rule of thumb puts it near 0.001 for large networks; this toy's healthy runs sit near 0.02, so the useful signal is a jump away from a run's own normal range.",
    },
    {
      term: "Max logit",
      definition:
        "The largest raw score any example in the batch gives any next character, before the softmax. Once it reaches the tens or hundreds the softmax is nearly certain of one character, and every other answer costs an enormous loss.",
    },
    {
      term: "Log-partition",
      definition:
        "The log of the softmax normalizer, log Z: the log of the sum of the exponentials of all the logits. Adding the same amount to every logit leaves the softmax unchanged but shifts log Z, so nothing in the ordinary loss stops it drifting.",
    },
    {
      term: "z-loss",
      definition:
        "An extra loss term, a small coefficient times log Z squared for every example, that pulls the log-partition toward zero. PaLM used a coefficient of 0.0001. It limits how far the logits drift but cannot make a too-large learning rate work.",
    },
    {
      term: "Mixed precision",
      definition:
        "Running most of the arithmetic in 16-bit numbers while keeping a 32-bit master copy of the weights for the update. It is faster and saves memory, at the cost of a narrower range or coarser steps in the 16-bit parts.",
    },
    {
      term: "fp16 and bf16",
      definition:
        "Two 16-bit floating-point formats. fp16 has 5 exponent bits and 10 fraction bits, so its largest value is 65,504; bf16 keeps fp32's 8 exponent bits and only 7 fraction bits, so it has fp32's range with much coarser precision.",
    },
    {
      term: "Loss scaling",
      definition:
        "Multiplying the loss, and with it every gradient, by a large constant before a 16-bit backward pass, then dividing it back out before the update. It lifts tiny gradients above fp16's rounding-to-zero point, but too large a scale pushes big ones past 65,504.",
    },
    {
      term: "Dynamic loss scaling",
      definition:
        "Loss scaling that adjusts itself: when a gradient overflows, the step is skipped and the scale halved, and after a stretch of clean steps the scale doubles again. PyTorch's GradScaler starts at 65,536 and waits 2,000 clean steps; this lab waits 200.",
    },
    {
      term: "Checkpoint rollback",
      definition:
        "Restarting training from weights saved before a problem instead of training on through it. When the data order and the arithmetic are deterministic, the replay up to the problem reproduces the original run exactly.",
    },
    {
      term: "Batch skipping",
      definition:
        "Leaving out the batches seen around a spike when training resumes after a rollback. PaLM skipped roughly 200 to 500 batches per spike; this lab skips the five garbage batches, reading them but never applying them, so the data order after them is unchanged.",
    },
  ],
  references: [
    {
      authors: "Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al.",
      title: "PaLM: Scaling Language Modeling with Pathways",
      source: "Journal of Machine Learning Research 24(240), 1–113",
      year: 2023,
      url: "https://jmlr.org/papers/v24/22-1144.html",
      note: "Reports about 20 loss spikes in its largest run despite gradient clipping at 1.0, fixed by restarting from a checkpoint about 100 steps earlier and skipping roughly 200 to 500 batches. It found the spikes were not caused by bad data alone, used a z-loss of 0.0001, and was bitwise reproducible from any checkpoint: the facts behind Roll back and skip and the z-loss control.",
    },
    {
      authors: "Mitchell Wortsman, Peter J. Liu, Lechao Xiao, et al.",
      title: "Small-scale proxies for large-scale Transformer training instabilities",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2309.14322",
      note: "Reproduces two instabilities of large Transformers in small models trained at high learning rates: attention logits that grow, fixed by qk-layernorm, and output logits that drift from the log probabilities, fixed by z-loss with a coefficient of 0.0001. It backs the Logit growth step, and notes that it studies slow divergence, not loss spikes.",
    },
    {
      authors: "Barret Zoph, Irwan Bello, Sameer Kumar, et al.",
      title: "ST-MoE: Designing Stable and Transferable Sparse Expert Models",
      source: "arXiv preprint arXiv:2202.08906",
      year: 2022,
      url: "https://arxiv.org/abs/2202.08906",
      note: "Introduces the router z-loss, which penalizes large logits entering a mixture-of-experts router, with a coefficient of 0.001, and reports that it stabilized training without hurting quality. It is the second place the lesson's z-loss shows up in practice.",
    },
    {
      authors: "Paulius Micikevicius, Sharan Narang, Jonah Alben, et al.",
      title: "Mixed Precision Training",
      source: "International Conference on Learning Representations (ICLR 2018)",
      year: 2018,
      url: "https://arxiv.org/abs/1710.03740",
      note: "Keeps an fp32 master copy of the weights and scales the loss so that small gradients survive in fp16, where 65,504 is the largest value and anything below 2 to the minus 24 becomes zero. It offers skipping the update when an overflow is detected, the move the Dynamic setting makes.",
    },
    {
      authors: "PyTorch Contributors",
      title: "Automatic Mixed Precision package - torch.amp",
      source: "PyTorch documentation",
      year: 2026,
      url: "https://docs.pytorch.org/docs/stable/amp.html",
      note: "Explains that small float16 gradients flush to zero and their updates are lost, which is why gradients are scaled. Its GradScaler defaults, a starting scale of 65,536, a growth factor of 2, a backoff factor of 0.5 and a growth interval of 2,000 steps, are the real settings the lab's Dynamic scaling is compared with.",
    },
    {
      authors: "PyTorch Contributors",
      title: "Automatic Mixed Precision examples",
      source: "PyTorch documentation",
      year: 2026,
      url: "https://docs.pytorch.org/docs/stable/notes/amp_examples.html",
      note: "Shows the training loop with a gradient scaler and says that the optimizer step is skipped whenever the gradients contain infs or NaNs, which it describes as rare. That is what Skipped steps counts when Scaling is on Dynamic.",
    },
    {
      authors: "Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, et al.",
      title: "A Study of BFLOAT16 for Deep Learning Training",
      source: "arXiv preprint arXiv:1905.12322",
      year: 2019,
      url: "https://arxiv.org/abs/1905.12322",
      note: "Lays out the bit budgets in the Number formats card, 8 exponent and 7 fraction bits for bf16 against 5 and 10 for fp16, and finds that bf16 has fp32's range, so the need for loss scaling can be avoided.",
    },
    {
      authors: "Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio",
      title: "On the difficulty of training recurrent neural networks",
      source: "Proceedings of the 30th International Conference on Machine Learning (ICML 2013), PMLR 28(3), 1310–1318",
      year: 2013,
      url: "https://proceedings.mlr.press/v28/pascanu13.html",
      note: "Proposes gradient norm clipping: when a gradient's norm goes over a threshold, rescale it to that norm. That is exactly what Gradient clip does to the factored model's exploding gradients.",
    },
    {
      authors: "Susan Zhang, Stephen Roller, Naman Goyal, et al.",
      title: "OPT: Open Pre-trained Transformer Language Models",
      source: "arXiv preprint arXiv:2205.01068",
      year: 2022,
      url: "https://arxiv.org/abs/2205.01068",
      note: "Its training log recovers from loss divergences by lowering the learning rate and restarting from an earlier checkpoint, ties divergence to its dynamic loss scaler crashing to 0, and reports that lowering the gradient clip from 1.0 to 0.3 helped. It is the lesson's second real example of rolling back.",
    },
    {
      authors: "Team OLMo: Pete Walsh, Luca Soldaini, Dirk Groeneveld, et al.",
      title: "2 OLMo 2 Furious",
      source: "arXiv preprint arXiv:2501.00656",
      year: 2025,
      url: "https://arxiv.org/abs/2501.00656",
      note: "Describes the stability changes of a recent open model: QK-norm, a z-loss term, no weight decay on embeddings, a smaller AdamW epsilon and careful initialization, and finds that data can cause both gradient-norm and loss spikes, so it removes documents with long repeated n-grams. The paper gives two different z-loss weights, so the lesson quotes neither.",
    },
    {
      authors: "Mostafa Dehghani, Josip Djolonga, Basil Mustafa, et al.",
      title: "Scaling Vision Transformers to 22 Billion Parameters",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 7480–7512",
      year: 2023,
      url: "https://arxiv.org/abs/2302.05442",
      note: "Saw the training loss diverge after a few thousand steps because attention logits grew extremely large, making attention almost one-hot, and fixed it by normalizing queries and keys. It backs the lesson's account of attention-logit growth, which this attention-free lab cannot show.",
    },
    {
      authors: "Igor Molybog, Peter Albert, Moya Chen, et al.",
      title: "A Theory on Adam Instability in Large-Scale Machine Learning",
      source: "arXiv preprint arXiv:2304.09871",
      year: 2023,
      url: "https://arxiv.org/abs/2304.09871",
      note: "Argues that divergence in large language model training is an artifact of Adam, which can reach a state where its update has a large norm and is essentially unrelated to the direction that lowers the loss. It backs the lesson's point that Adam reacts to the garbage batch far more than plain SGD does.",
    },
    {
      authors: "Sho Takase, Shun Kiyono, Sosuke Kobayashi, and Jun Suzuki",
      title: "Spike No More: Stabilizing the Pre-training of Large Language Models",
      source: "Conference on Language Modeling (COLM 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2312.16903",
      note: "Traces loss spikes to sudden growth of the gradient norm and derives conditions that prevent it, small sub-layers and a large shortcut, checked with scaled or normalized embeddings. It backs the Going deeper point that architecture choices, not only fixes during the run, keep a run stable.",
    },
    {
      authors: "Loubna Ben Allal, Lewis Tunstall, Nouamane Tazi, et al.",
      title: "The Smol Training Playbook: The Secrets to Building World-Class LLMs",
      source: "Hugging Face",
      year: 2025,
      url: "https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook",
      note: "A practitioner's guide that calls rewinding to before a spike and skipping the problematic batches the most common fix, and lists tighter gradient clipping, z-loss and QK-norm among the others. It connects every fault in this lab to what a training team does about it.",
    },
    {
      authors: "Stanford CS231n course staff",
      title: "Neural Networks Part 3: Learning and Evaluation",
      source: "CS231n: Deep Learning for Computer Vision, Stanford University course notes",
      year: 2026,
      url: "https://cs231n.github.io/neural-networks-3/",
      note: "Gives the rule of thumb drawn as the dashed line on Update ÷ weight: the ratio of update size to weight size should be somewhere around 0.001, with lower suggesting too small a learning rate and higher too large a one.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "With Fault on None, Update ÷ weight sits near 0.026 all run, about 26 times the 10⁻³ rule of thumb, and the loss still falls to 2.039. How should you read that?",
      options: [
        "The learning rate is far too high, because any ratio above the rule of thumb means the run is close to diverging",
        "The rule of thumb is for large networks; the warning sign is a jump away from this run's own normal range",
        "The ratio is miscalculated, because a healthy step never moves the weights by more than a thousandth of their size",
        "The weights have stopped changing, because a ratio this large means each step is tiny next to the weights",
      ],
      answer: 1,
      explanation:
        "The update-to-weight ratio depends on the model, the optimizer and the learning rate, so 10⁻³ is a starting point for large networks, not a law. This healthy run keeps its ratio between about 0.021 and 0.040 and finishes at 2.039 nats. Trouble shows as a departure from that band: at a learning rate of 8 the ratio is 0.310 on the very first step, about fifteen times the healthy run's first step.",
      objective: 0,
    },
    {
      prompt:
        "You set Fault to Learning rate too high with Gradient clip off. Which readout already differs from the healthy run on step 1?",
      options: [
        "Update ÷ weight, which measures the step itself and so grows with the learning rate",
        "Gradient norm, because the trainer computes it from weights that the oversized step has already moved",
        "Max logit, which is the first quantity that a learning rate this large pushes out of range",
        "Loss, which reflects the larger learning rate before a single weight has changed",
      ],
      answer: 0,
      explanation:
        "On step 1 the loss, the gradient norm and the max logit are all measured from the same starting weights on the same first batch, before any update, so they match the healthy run exactly. Only the step differs: it is sixteen times longer, and Update ÷ weight reads 0.310 against 0.021. The gradient norm does not rise above the healthy band until step 6 and passes 10 only at step 36.",
      objective: 1,
    },
    {
      prompt: "With Fault on Learning rate too high, which setting finishes below the healthy run's 2.039 nats?",
      options: [
        "Warmup + cosine with Gradient clip off, since a gentle start prevents the blow-up",
        "Gradient clip at 0.50 on a Constant schedule, since capping each step is enough",
        "Gradient clip at 0.50 plus Warmup + cosine, which caps each step and then shrinks it",
        "Gradient clip at 1.00 on a Constant schedule, the clip value most large runs report",
      ],
      answer: 2,
      explanation:
        "Without a clip, warmup and cosine decay only slow the blow-up: the weights stay finite but huge, max logit reaches about 10²⁸, and the run ends at 19.152 nats. The clip alone survives at 3.172, barely below chance, and a clip of 1.00 ends at 4.050, worse than chance. Together the clip caps every step's length and the decay shortens the late steps, so the run reaches 1.904, below the healthy run at a sixteenth of the rate.",
      objective: 1,
    },
    {
      prompt:
        "At Loss scale 2²⁰ with Static scaling the run halts at step 33. You switch Scaling to Dynamic. What do you expect?",
      options: [
        "It skips the overflowing steps, halves the scale each time, and ends almost level with fp32",
        "It lowers the scale once, before the first step, so no step ever overflows and none is skipped",
        "It caps every overflowing gradient entry at 65,504, the largest fp16 value, and applies every step",
        "It still halts at step 33, because dynamic scaling only protects against gradients that underflow",
      ],
      answer: 0,
      explanation:
        "Dynamic scaling checks every step for an overflow. When it finds one it skips that step, so the infinite gradient never reaches a weight, and halves the scale; after 200 clean steps it doubles the scale again. From 2²⁰ it skips steps 33, 62, 184 and 430, ends at 2¹⁷, and finishes at 2.040 nats against 2.039 without fp16. A static scale applies the first overflow, which sends infinity into the weights.",
      objective: 2,
    },
    {
      prompt:
        "Across the whole healthy run, none of the 153,008 nonzero gradient entries is small enough to round to zero in fp16, even with no loss scale. What does that tell you about this lab?",
      options: [
        "It shows fp16's overflow side only; the underflow that loss scaling targets never occurs here",
        "Loss scaling is unnecessary in real fp16 training too, because gradients are never that small",
        "fp16 has the same range as fp32 for gradients, so only the forward pass needs any care",
        "Underflow cannot happen in any model, because a gradient clip keeps every entry above 2⁻²⁴",
      ],
      answer: 0,
      explanation:
        "Loss scaling exists because small gradient values flush to zero in fp16 and their updates are lost; Micikevicius and colleagues used fixed scale factors from 8 to 32K. In this toy the smallest gradient entry, 3.6 × 10⁻⁸, sits just above fp16's rounding-to-zero point of 3.0 × 10⁻⁸, so none flush even at a scale of 1, though 949 land in fp16's coarser subnormal range. The lab can show a scale that is too large, but not one that is too small.",
      objective: 2,
    },
    {
      prompt:
        "The garbage batches spike the whole-corpus loss to 3.447 at step 404. You press Roll back and skip. What does the replay do?",
      options: [
        "It matches the original bit for bit from step 351 to 399, then never applies steps 400 to 404",
        "It reshuffles the data after the checkpoint, so every step from 351 on differs from the original",
        "It applies the garbage batches again, at a smaller learning rate so that they do less damage",
        "It applies the garbage batches again, with a tighter gradient clip that removes their effect",
      ],
      answer: 0,
      explanation:
        "The rollback reloads the weights saved after step 350 and trains forward again. The seed fixes the data order and the weights are the same, so steps 351 to 399 reproduce the original exactly; PaLM relied on the same property, being bitwise reproducible from any checkpoint. Steps 400 to 404 are read but never applied, and the run finishes at 1.993 nats, against 2.022 without the rollback and 1.991 with no garbage at all.",
      objective: 3,
    },
    {
      prompt:
        "In this lab the spike has an obvious cause, garbage text at steps 400 to 404. What did PaLM find about the cause of its own loss spikes?",
      options: [
        "Not bad data alone: the same batches, replayed from an earlier checkpoint, did not spike",
        "They happened only because gradient clipping was switched off for its largest model",
        "They came from fp16 overflows and stopped once the run moved to dynamic loss scaling",
        "Each was traced to a junk document, and deleting those documents ended the spikes",
      ],
      answer: 0,
      explanation:
        "PaLM saw roughly 20 spikes while training its largest model, with gradient clipping on. Feeding the batches from around a spike to a different, earlier checkpoint did not spike, so the authors concluded that spikes come from particular batches meeting a particular state of the weights. Their fix is the one this lab imitates: restart from a checkpoint about 100 steps earlier and skip roughly 200 to 500 batches. The lab's garbage text is a simpler, known cause.",
      objective: 3,
    },
    {
      prompt:
        "At Logit growth, raising z-loss to 0.01 pulls the median log-partition from 19.7 to 2.2, yet the loss still ends at 3.277, near the 3.401 of knowing nothing. What brings it to a healthy level?",
      options: [
        "Lowering Learning rate to 0.1, because the steps themselves are far too large",
        "Raising z-loss to 0.1, ten times more, so that the logits shrink even further",
        "Switching to bf16, because these logits are overflowing their number format",
        "Nothing in this lab, because a loss near chance means this text cannot be fitted",
      ],
      answer: 0,
      explanation:
        "z-loss penalizes the squared log-partition, so it reins in how far the logits drift together, and the peak max logit falls from 395 to 179. It does nothing about the size of each step: Adam still moves each weight by roughly the learning rate on every step. At 0.1 the run ends at 1.985 nats. More z-loss is no rescue either: a coefficient of 0.1, ten times the control's top, ends at 13.418 at a rate of 1, worse than none at 11.515.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateStabilityState,
};

export default definition;
