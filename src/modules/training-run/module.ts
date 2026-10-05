import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateTrainingRunState, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-17-training-run",
  slug: "training-run",
  title: "Anatomy of a training run",
  group: "training-adapting",
  order: 36,
  icon: "Repeat",
  accent: "#4a6fd0",
  prerequisites: ["module-06-training-dynamics", "module-07-next-token-prediction"],
  estimatedMinutes: 16,
  steps: [
    "Walk the loop",
    "Read the curve",
    "Resize the batch",
    "Reload a checkpoint",
    "Step one batch",
    "Count the memory",
  ],
  stepInstructions: [
    "Click through all seven stages of the loop and read what each one reports about this run; at Backward, read the gradient strip for the pair o → g.",
    "Compare the thin batch-loss line with the thick whole-corpus line and the dotted “knows nothing” line, and note how noisy a few batches are.",
    "Set Batch size to 2, then to 64, and compare total steps and final loss at the same number of epochs.",
    "Scrub Checkpoint from step 0 to the end and read the sample text at each stop.",
    "Set Checkpoint to step 0 and press Step one batch three times. Read the table of rows that moved, then compare Batch loss with Whole-corpus loss after each press.",
    "In Where the memory goes, keep Model size at 7.5 billion, switch Optimizer from Adam to SGD, then switch Precision, and note which table row and which total change each time.",
  ],
  stateVersion: 2,
  tagline:
    "Dataset, batch, forward, loss, backward, step, checkpoint — running for real on a character model small enough to retrain every time you move a control.",
  objectives: [
    "Name each stage of a training loop and what it produces",
    "Distinguish an epoch, a step, and a batch",
    "Use a checkpoint to inspect a model partway through training",
    "Read what one optimizer step changes: which rows move, and how the loss on its own batch differs from the whole-corpus loss",
    "Estimate training memory from parameters, precision, and optimizer, and say why large runs split it across devices",
  ],
  glossary: [
    {
      term: "Epoch",
      definition:
        "One complete pass over the training data. It is a bookkeeping unit, not a unit of learning: the same epoch count means very different amounts of learning at different batch sizes.",
    },
    {
      term: "Step",
      definition:
        "One optimizer update. Steps, not epochs, are what actually change the weights, which is why the step count is the number to compare across runs.",
    },
    {
      term: "Minibatch",
      definition:
        "The group of examples averaged together before one update. Larger batches give a less noisy gradient estimate and fewer updates per epoch.",
    },
    {
      term: "Dataloader",
      definition:
        "The component that shuffles, batches, and feeds examples to the model. Shuffling makes each batch a fair sample of the whole corpus, so consecutive updates do not all pull toward one passage.",
    },
    {
      term: "Forward pass",
      definition:
        "Running inputs through the model to produce logits, then probabilities. No parameter changes during it.",
    },
    {
      term: "Cross-entropy",
      definition:
        "The negative log-probability the model assigned to the token that actually came next, averaged over the batch. Reported here in nats per token.",
    },
    {
      term: "Perplexity",
      definition:
        "The exponential of cross-entropy. Read it as the effective number of options the model is choosing between, so lower is better.",
    },
    {
      term: "Backward pass",
      definition:
        "The sweep that computes one gradient per parameter from the loss. For softmax cross-entropy the gradient of a row is the predicted distribution minus one at the true token. The length of the whole batch gradient is its gradient norm.",
    },
    {
      term: "Optimizer step",
      definition:
        "Subtracting the scaled gradient from each parameter. Everything else in the loop exists to produce the numbers this step consumes.",
    },
    {
      term: "Checkpoint",
      definition:
        "A saved copy of the weights at a point in the run. Checkpoints make training inspectable and recoverable; a checkpoint meant for resuming also stores optimizer state, the data position, and random state.",
    },
    {
      term: "Nats per token",
      definition:
        "Cross-entropy measured with natural logarithms. A drop of ln 2, about 0.69 nats, means the probability given to the correct token doubled, as a geometric mean over the tokens scored.",
    },
    {
      term: "Optimizer state",
      definition:
        "What the optimizer remembers between steps. Plain SGD keeps nothing; momentum keeps one value per parameter and Adam keeps two, a running mean and a running variance of the gradient, usually in fp32.",
    },
    {
      term: "Mixed precision",
      definition:
        "Running the forward and backward passes in 16-bit numbers while the optimizer keeps an fp32 master copy of the weights. It speeds up the arithmetic but does not shrink the model state, because the master copy and the optimizer state stay at 4 bytes.",
    },
    {
      term: "Sharding",
      definition:
        "Splitting weights, gradients, or optimizer state across devices so none holds a full copy. ZeRO partitions all three across data-parallel workers, and PyTorch's FSDP, motivated by ZeRO, shards the parameters. Data, tensor, and pipeline parallelism split the work instead.",
    },
  ],
  references: [
    {
      authors: "Ian Goodfellow, Yoshua Bengio, and Aaron Courville",
      title: "Deep Learning",
      source: "MIT Press, free to read online",
      year: 2016,
      url: "https://www.deeplearningbook.org/",
      note: "Section 8.1.3, Batch and Minibatch Algorithms, explains why the gradient from a small random batch is a noisy but useful estimate of the full gradient, and why using more examples gives less than linear returns. It also says batches must be drawn at random and that shuffling the data once is usually enough, the dataloader's job at the Batch stage.",
    },
    {
      authors: "Varun Godbole, George E. Dahl, Justin Gilmer, et al.",
      title: "Deep Learning Tuning Playbook",
      source: "Google Research, GitHub repository (version 1.0)",
      year: 2023,
      url: "https://github.com/google-research/tuning_playbook",
      note: "The section on choosing the batch size warns that comparing batch sizes at a fixed number of epochs is not the same as comparing them at a fixed number of steps. That is the trap the Resize the batch step shows. A later section explains why runs save checkpoints along the way.",
    },
    {
      authors: "Christopher J. Shallue, Jaehoon Lee, Joseph Antognini, et al.",
      title: "Measuring the Effects of Data Parallelism on Neural Network Training",
      source: "Journal of Machine Learning Research 20(112), 1–49",
      year: 2019,
      url: "https://jmlr.org/papers/v20/18-789.html",
      note: "Measures how many training steps it takes to reach a target error as the batch size grows, across many models and datasets. It finds that some disagreements about batch size come from comparing runs with different budgets, which is why this lesson compares Total steps rather than epochs.",
    },
    {
      authors: "Diederik P. Kingma and Jimmy Ba",
      title: "Adam: A Method for Stochastic Optimization",
      source: "3rd International Conference on Learning Representations (ICLR 2015)",
      year: 2015,
      url: "https://arxiv.org/abs/1412.6980",
      note: "The original Adam paper. For every weight, Adam keeps a running average of the gradient and of the squared gradient and scales each step by them. Those are the two extra numbers per weight that the Optimizer setting adds in Where the memory goes.",
    },
    {
      authors: "Matthew Inkawhich",
      title: "Saving and Loading Models",
      source: "PyTorch tutorials",
      year: 2025,
      url: "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
      note: "The part on saving a general checkpoint says that to resume training you must save the optimizer's state as well as the model's weights, plus details such as the epoch you stopped at. It backs the lesson's point that a checkpoint for resuming holds much more than the weights.",
    },
    {
      authors: "Paulius Micikevicius, Sharan Narang, Jonah Alben, et al.",
      title: "Mixed Precision Training",
      source: "International Conference on Learning Representations (ICLR 2018)",
      year: 2018,
      url: "https://arxiv.org/abs/1710.03740",
      note: "Shows how to train with 16-bit numbers without losing accuracy by keeping an fp32 master copy of the weights and scaling the loss. That master copy is why switching Precision to Mixed 16-bit does not shrink the total under Adam.",
    },
    {
      authors: "Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, et al.",
      title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
      source: "SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, 1–16",
      year: 2020,
      url: "https://arxiv.org/abs/1910.02054",
      note: "Section 3, Where Did All the Memory Go?, counts 16 bytes per parameter for mixed-precision Adam and adds activations, temporary buffers, and fragmentation. The paper gives the 120 GB figure for 7.5 billion parameters that Where the memory goes reproduces, then splits optimizer state, gradients, and parameters across devices.",
    },
    {
      authors: "Deepak Narayanan, Mohammad Shoeybi, Jared Casper, et al.",
      title: "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM",
      source: "Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC 2021), 1–15",
      year: 2021,
      url: "https://arxiv.org/abs/2104.04473",
      note: "Shows how data, tensor, and pipeline parallelism combine to train a model with a trillion parameters on 3,072 GPUs. It backs the lesson's Memory and parallelism section: splitting the batch, each layer's matrices, or the stack of layers.",
    },
    {
      authors: "Yanli Zhao, Andrew Gu, Rohan Varma, et al.",
      title: "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel",
      source: "Proceedings of the VLDB Endowment 16(12), 3848–3860",
      year: 2023,
      url: "https://arxiv.org/abs/2304.11277",
      note: "Describes PyTorch's Fully Sharded Data Parallel, which shards model parameters across devices and gathers each part only when it is needed. The paper says FSDP was motivated by ZeRO, as the Sharding glossary entry states.",
    },
    {
      authors: "Nouamane Tazi, Ferdinand Mom, Haojun Zhao, et al.",
      title: "The Ultra-Scale Playbook: Training LLMs on GPU Clusters",
      source: "Hugging Face, free to read online",
      year: 2025,
      url: "https://huggingface.co/spaces/nanotron/ultrascale-playbook",
      note: "A step-by-step guide that starts on one GPU, counting the memory of weights, gradients, optimizer state, and activations, then adds activation recomputation, gradient accumulation, ZeRO, and tensor and pipeline parallelism. It covers the Going deeper topics of this lesson in one place.",
    },
    {
      authors: "Tom B. Brown, Benjamin Mann, Nick Ryder, et al.",
      title: "Language Models are Few-Shot Learners",
      source: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 1877–1901",
      year: 2020,
      url: "https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html",
      note: "The GPT-3 paper. Table 2.1 lists the batch size and learning rate for each model size, and Appendix B adds Adam, gradient clipping at 1.0, warmup and decay of the learning rate, and weight decay of 0.1. These are the figures in the lesson's Toy versus real list.",
    },
    {
      authors: "Hugo Touvron, Louis Martin, Kevin Stone, et al.",
      title: "Llama 2: Open Foundation and Fine-Tuned Chat Models",
      source: "arXiv preprint arXiv:2307.09288",
      year: 2023,
      url: "https://arxiv.org/abs/2307.09288",
      note: "Table 1 gives 2.0 trillion pretraining tokens, a global batch of 4 million tokens, and the learning rate for each size. Those numbers give the lesson's estimate of about 500,000 updates, and the same section lists the warmup, clipping, and weight decay.",
    },
    {
      authors: "Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al.",
      title: "PaLM: Scaling Language Modeling with Pathways",
      source: "Journal of Machine Learning Research 24(240), 1–113",
      year: 2023,
      url: "https://jmlr.org/papers/v24/22-1144.html",
      note: "Reports about 20 loss spikes while training its largest model and fixes them by restarting from a checkpoint about 100 steps before each spike and skipping a few hundred batches. That is the roll-back-and-skip response the lesson describes, and the paper also explains how a run can be replayed exactly from any checkpoint.",
    },
    {
      authors: "Edward J. Hu, Yelong Shen, Phillip Wallis, et al.",
      title: "LoRA: Low-Rank Adaptation of Large Language Models",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2106.09685",
      note: "Freezes the pretrained weights and trains small added matrices instead, so only those need gradients and optimizer state. Against fine-tuning GPT-3 175B with Adam, it reports 10,000 times fewer trainable parameters and 3 times less GPU memory, the figures in Why this justifies LoRA.",
    },
    {
      authors: "Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, et al.",
      title: "QLoRA: Efficient Finetuning of Quantized LLMs",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 10088–10115",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html",
      note: "Stores the frozen weights in 4 bits and trains LoRA adapters on top, which lets a 65 billion parameter model be fine-tuned on a single 48 GB GPU. It is the lesson's last example of cutting the memory that training needs.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "You keep the epoch count fixed and raise the batch size from 4 to 64. What happens to the number of optimizer steps?",
      options: [
        "It stays the same, because the epoch count alone decides how many updates the run takes",
        "It rises, because a larger batch means more work to be done in every single epoch",
        "It falls about sixteen times, because each step consumes sixteen times more pairs",
      ],
      answer: 2,
      explanation:
        "Steps per epoch equal the number of pairs divided by the batch size. Epochs count passes over the data; steps count updates. That gap is why two runs with identical epoch counts can end at very different losses.",
      objective: 1,
    },
    {
      prompt: "Walking the loop from Dataset to Checkpoint, which stage is the first one to change a weight?",
      options: [
        "Optimizer step, which subtracts the learning-rate-scaled gradient from each weight",
        "Backward, which writes a finished gradient into every weight that it reaches in the table",
        "Loss, which pulls the predicted probabilities toward the character that really came next",
        "Forward, which turns the row for the current character into probabilities",
      ],
      answer: 0,
      explanation:
        "Forward turns a row of weights into probabilities, Loss scores them, and Backward computes one gradient per weight; all three only read the weights. The optimizer step is where the gradient, scaled by the learning rate, is subtracted, so every earlier stage exists to produce the numbers this one consumes.",
      objective: 0,
    },
    {
      prompt: "The lab prints sample text from the middle of the run after training has finished. What makes that possible?",
      options: [
        "Training reruns from step 0 up to the chosen stop every time the Checkpoint slider moves",
        "A checkpoint is a saved copy of the weights, so that earlier model can be loaded and sampled",
        "Only the final weights exist, and the earlier text is blended toward them just for display",
        "The sampler stored every string it printed during training and replays the one nearest the stop",
      ],
      answer: 1,
      explanation:
        "The run saves nine genuine copies of the weights, including the untrained one. Scrubbing loads one of those copies, so its loss, its sample text and its next-character ranking all come from exactly that model; nothing is replayed, blended or recomputed.",
      objective: 2,
    },
    {
      prompt:
        "At 60 epochs the final loss is lower than at 12 epochs. Which question about that lower number can this lab not answer?",
      options: [
        "Whether the extra epochs really took more optimizer steps, since the lab never counts them",
        "Whether the saved checkpoints are real copies of the weights or only a drawing of them",
        "Whether it learned the language or memorised the text, since loss is scored on the training text",
        "Whether a larger batch would have taken the same number of steps, since batch size has no control",
      ],
      answer: 2,
      explanation:
        "Every loss in this lab is cross-entropy on the same text the weights were fitted to, so a lower number can mean better prediction of the language or tighter memorisation of those sentences, and nothing on screen separates the two. A held-out set, as in Building a dataset, is what does.",
      objective: 0,
    },
    {
      prompt:
        "From the untrained checkpoint you press Step one batch once. The loss on that batch falls by about 0.04 nats, but the whole-corpus loss falls by about 0.005. Why the gap?",
      options: [
        "The step is fitted to the pairs its batch averaged and moves only their rows, while the corpus loss counts every pair",
        "The corpus loss is computed on different text, so a step cannot change it and the small drop is only rounding",
        "The step moves all 900 weights by the same amount, and the smaller batch simply makes that shared move look bigger",
        "The batch loss was measured after the weights moved and the corpus loss before, so the two are not comparable",
      ],
      answer: 0,
      explanation:
        "The batch gradient has entries only in the rows of the characters that began a pair in that batch, so exactly those rows move. The batch is scored on the pairs the step just fitted; the corpus loss spreads the same small improvement across all 498 pairs, most of which sit in rows that did not move. Both losses are computed on the same text with the same weights.",
      objective: 3,
    },
    {
      prompt:
        "At Batch size 2 you pick the checkpoint at step 1492 and press Step one batch again and again. What do the two losses do?",
      options: [
        "The batch loss falls on every press, but the whole-corpus loss rises on many of them",
        "Both losses fall on every press, since each step follows the gradient downhill",
        "The batch loss rises on many presses, because two pairs are too few to give a gradient",
        "Both losses wander up and down, since a batch of two has no reliable direction",
      ],
      answer: 0,
      explanation:
        "A step is computed from the pairs in its batch, so it lowers the loss on those pairs every time at this learning rate. With two pairs, a step can fit them while nudging shared structure the other pairs relied on, so the whole-corpus loss rises on most presses from that checkpoint. It still falls over the run on average; a larger batch makes each step a fairer sample and removes most of the rises.",
      objective: 3,
    },
    {
      prompt:
        "With Adam selected at 7.5 billion parameters you switch Precision from fp32 to Mixed 16-bit. What happens to Total, and why?",
      options: [
        "Total halves, because every number in the model is stored in two bytes where fp32 uses four, whatever the optimizer is",
        "Total stays the same, because the fp32 master copy and the optimizer state take back what the 16-bit parts save",
        "Total rises, because mixed precision stores every weight twice, once in each format, and the copies add up",
        "Total falls by a quarter, because only the gradients switch to the smaller format and everything else stays the same",
      ],
      answer: 1,
      explanation:
        "In fp32, Adam costs 4 bytes of weights, 4 of gradients and 8 of momentum and variance. In mixed precision the weights and gradients take 2 bytes each, but the optimizer keeps a 4-byte master copy next to the 8 bytes of state. Both come to 16 bytes per parameter. Mixed precision speeds up the arithmetic and shrinks activations; it does not shrink model state.",
      objective: 4,
    },
    {
      prompt:
        "Weights, gradients and Adam's state for a 7.5 billion parameter model need 120 GB. What does sharding such as ZeRO or FSDP do about that?",
      options: [
        "It compresses the optimizer state so that one device can hold all of it",
        "It splits the training text across devices, so each one works through its own share of the data",
        "It keeps only a slice of the state on each device, so no device has to hold a full copy",
        "It skips the gradient for most parameters, so the optimizer has less to store",
      ],
      answer: 2,
      explanation:
        "The 16 bytes per parameter are needed somewhere, so the fix is to divide them rather than shrink them. ZeRO partitions optimizer state, gradients and parameters across data-parallel workers, and each worker keeps one slice and fetches the rest when it needs it. Splitting the text is data parallelism, which on its own leaves a full copy of the state on every device.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateTrainingRunState,
};

export default definition;
