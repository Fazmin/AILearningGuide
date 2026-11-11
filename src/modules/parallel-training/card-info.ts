import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Parallel layout": {
    title: "Four ways to split one training step",
    summary:
      "Pick a published model shape and how many ways to split it. Tensor parallel splits every layer's matrices, Pipeline splits the stack of layers into stages, Data parallel copies that arrangement onto more GPUs that each read their own share of the batch, and Expert parallel spreads Mixtral's experts. GPUs = tensor × pipeline × data.",
    whatYouSee: [
      "`Model`: `Llama 3.1 8B`, `Llama 3.1 70B`, `Llama 3.1 405B`, or `Mixtral 8x7B`, and the `16,384-GPU layout` button, which loads Llama 3's 405B layout.",
      "Sliders on power-of-two ladders: `Tensor parallel` 1 to 32, `Pipeline` 1 to 16, `Data parallel` 1 to 2,048, `Expert parallel` 1 to 8 (Mixtral only, never above Data parallel), `Micro-batch size` 1 to 8 sequences of 8,192 tokens, and `Microbatches` 1 to 128 per step.",
      "A diagram in two parts. First the model as one slab, cut into pipeline stages across and tensor slices down, with the data-parallel copies stacked behind it and the piece one GPU holds filled in. Then the first two 8-GPU nodes, with a line through the GPUs of each group that GPU 0 belongs to: solid where the group stays inside a node on NVLink, dotted where it has to leave it for InfiniBand.",
      "Metrics `GPUs`, `Nodes of 8`, `Tokens per step`, and `Parameters per GPU`, then a list saying what each degree splits, what it sends, and over which link.",
    ],
    howItWorks: [
      "Parameters come from the shape: Ψ = L·(2h² + 2h·kv·h/a + 3h·f·E) + 2Vh, which gives 8.03 B, 70.55 B, and 405.85 B for the Llama shapes and 46.70 B for Mixtral, 12.88 B of them active because each token uses 2 of 8 experts.",
      "Parameters per GPU = dense parameters ÷ (tensor × pipeline) + expert parameters ÷ (tensor × pipeline × expert). Tokens per step = data × microbatches × micro-batch size × 8,192.",
      "Ranks are numbered tensor-first and packed 8 to a node, so a group of n GPUs spaced `stride` apart stays on NVLink when stride × n ≤ 8: a tensor group up to 8, a data-parallel group only when tensor × data ≤ 8, and pipeline neighbours only when the whole job fits in one node.",
      "`16,384-GPU layout` sets Llama 3.1 405B with tensor 8, pipeline 16, data 128, one sequence per microbatch, 16 microbatches, ZeRO stage 2, Selective recompute, and overlap on. That is the layout in Table 4 of the Llama 3 paper, with ZeRO stage 2 standing for its FSDP setting.",
    ],
    controls: [
      "Every slider redraws all six cards. `Expert parallel` appears only for Mixtral and is lowered with `Data parallel` whenever it would exceed it.",
      "Comparison worth running: hold the 70B model at Pipeline 4 and move Tensor parallel from 8 to 16. Parameters per GPU halves, but the TP all-reduce row in Communication per step grows almost tenfold.",
    ],
    notice: [
      "Tensor and pipeline parallelism divide the model; data parallelism multiplies the batch. Only ZeRO, on the Memory per GPU card, lets data parallelism divide memory too.",
      "The list under the metrics switches a group from NVLink to InfiniBand the moment it spans two nodes.",
    ],
    limits: [
      "In this lab: the shapes are published, and everything else is closed-form arithmetic on stated assumptions: an H100 at 989.5 TFLOP/s and 80 GB, NVLink at 450 GB/s each way, InfiniBand at 50 GB/s per GPU, and 8,192-token sequences. There is no context parallelism and no interleaved schedule, and every stage gets an equal share of the layers.",
      "In general: real clusters choose the rank order, add context parallelism for long sequences, balance stages by hand (Llama 3 removes a layer from its first and last stages), and run hierarchical collectives that this lab does not model.",
    ],
  },

  "Memory per GPU": {
    title: "Weights, gradients, optimizer state, and activations",
    summary:
      "Mixed-precision Adam keeps 16 bytes of model state per parameter: 2 for the weights, 2 for the gradients, and 12 for the optimizer. Each GPU also keeps the activations of the microbatches it is working on. Tensor and pipeline parallelism divide every part, ZeRO lets data parallelism divide the model state, and recomputation trades activations for arithmetic.",
    whatYouSee: [
      "`ZeRO stage` 0 to 3 and `Recompute`: `None`, `Selective`, or `Full`.",
      "A stacked bar on a GB axis with the four parts against the 80 GB line and a dashed outline on any overflow, a paler bar under it for the same layout at ZeRO stage 0, then a bar list with each part's size and what divides it.",
      "Metrics `Model state`, `Activations`, `Total per GPU`, and `Fits in 80 GB`, and a note on the chosen stage and recompute mode.",
    ],
    howItWorks: [
      "Parameters on one GPU Ψg = Ψdense ÷ (t·p) + Ψexperts ÷ (t·p·e). Bytes per parameter come from the shared training-memory arithmetic: 2 for weights, 2 for gradients, and 12 for the fp32 master copy, momentum, and variance.",
      "ZeRO stage 1 divides the optimizer state by d, stage 2 also the gradients, and stage 3 also the weights: 16Ψg, 4Ψg + 12Ψg/d, 2Ψg + 14Ψg/d, and 16Ψg/d. For 7.5 B parameters on 64 GPUs that is 120, 31.4, 16.6, and 1.9 GB, the ZeRO paper's Figure 1. Mixtral's experts are sharded over the d ÷ e GPUs that hold the same expert.",
      "Activations per layer per microbatch, from Korthikanti et al.'s Table 2 with 2-byte activations and sequence parallelism when t > 1: None sbh(34 + 5as/h)/t, Selective 34sbh/t, and Full 2sbh. They are multiplied by the first stage's ⌈L/p⌉ layers and its min(m, p) microbatches in flight.",
      "`Fits in 80 GB` compares the total with 80 × 10⁹ bytes.",
    ],
    controls: [
      "`ZeRO stage` and `Recompute` here. `Micro-batch size`, `Microbatches`, and the parallel degrees on the Parallel layout card move the bars too.",
      "Comparison worth running: at the defaults, move ZeRO stage from 0 to 3. Model state goes 128.5 GB, 33.6 GB, 17.8 GB, then 2.0 GB, against 36.5 GB of activations, so Fits in 80 GB turns to yes at stage 1.",
    ],
    notice: [
      "No ZeRO stage touches the Activations bar, and Data parallel shrinks nothing at all at stage 0.",
      "At 8,192 tokens the attention-score term 5as/h is 320 against 34 for the 8B model, which is why None is ten times Selective.",
    ],
    limits: [
      "In this lab: closed-form bytes, with temporary and communication buffers, the output logits, fragmentation, and the layer ZeRO stage 3 gathers all left out, so a total just under 80 GB would not fit in practice. Korthikanti et al.'s coefficients were derived for GPT-style layers with a 4h MLP and are applied unchanged to SwiGLU, grouped-query, and mixture-of-experts layers.",
      "In general: frameworks also offload state to CPU memory, keep activations in other precisions, and recompute only some layers. Real memory use is read from the framework's profiler, not predicted from a formula.",
    ],
  },

  "Pipeline schedule": {
    title: "The one-forward-one-backward schedule and its bubble",
    summary:
      "Each stage runs a forward pass of one unit and a backward pass of two units for every microbatch. A forward waits for the stage before it and a backward for the stage after it, so while the first microbatches fill the pipeline and the last ones drain it, some stages have nothing to do: the bubble, (p − 1)/(m + p − 1) of the pipeline's time.",
    whatYouSee: [
      "One row per stage with time in seconds from left to right. Blocks F1, F2, and so on are forward passes; the wider B1, B2, and so on are backward passes; hatched gaps are idle time; a dashed block after the last backward is the data-parallel sync left exposed.",
      "A badge and metrics `Bubble`, `Microbatch time`, `Pipeline time`, and `In flight on stage 1`, and a note with the bubble formula filled in.",
    ],
    howItWorks: [
      "The schedule is simulated, not drawn by hand: stage s runs min(p − s − 1, m) forwards, then alternates one forward and one backward, then drains its remaining backwards. It always takes 3(m + p − 1) units.",
      "One unit is a third of the microbatch time: compute plus tensor, pipeline, and expert traffic for one microbatch on one stage. Pipeline time = (m + p − 1) × microbatch time.",
      "`In flight on stage 1` = min(m, p): the microbatches whose activations stage 1 must keep at once, the count the Memory per GPU card multiplies by.",
    ],
    controls: [
      "`Pipeline` sets the rows and `Microbatches` the blocks, both on the Parallel layout card. Every other control moves only the time axis.",
      "Comparison worth running: Pipeline 8 at 1 microbatch, where 87.5% of the pipeline's time is idle, against 32 microbatches, where 17.9% is.",
    ],
    notice: [
      "The idle slots stay at p − 1 however many microbatches you add; only the useful work around them grows.",
      "Narayanan et al. write the same bubble as (p − 1)/m, measured against the ideal time rather than the total.",
    ],
    limits: [
      "In this lab: a backward always takes twice a forward, every stage gets the same number of layers, and the pipeline sends sit inside each microbatch's time instead of overlapping with compute.",
      "In general: production runs use interleaved schedules with several virtual stages per GPU, which divide the bubble at the cost of more traffic, and balance their uneven first and last stages, as Llama 3 does by removing a layer from each. The lab models neither.",
    ],
  },

  "Communication per step": {
    title: "Four kinds of traffic and two kinds of link",
    summary:
      "Tensor parallelism all-reduces activations inside every layer, pipeline parallelism sends activations between stages, expert parallelism exchanges tokens by all-to-all, and data parallelism all-reduces gradients once per step. Each is priced as bytes over the slowest link its group uses, from bandwidth alone.",
    whatYouSee: [
      "The `Overlap gradient sync` switch, and under it two bars on one time axis: the last backward pass and the gradient sync. The part of the sync that runs under the backward pass is hatched and the part left exposed is solid.",
      "A bar list in seconds per step: `TP all-reduce`, `PP send/receive`, `EP all-to-all` (Mixtral only), and `DP gradient sync`. Each detail line gives the time per microbatch, the bytes, the group size, and the link.",
      "Metrics `Exposed DP time`, `TP group`, and `DP group`, and a badge with the exposed time.",
    ],
    howItWorks: [
      "All-reduce time = 2(n − 1)/n × bytes ÷ bandwidth; reduce-scatter, all-gather, and all-to-all = (n − 1)/n × bytes ÷ bandwidth; a pipeline send = bytes ÷ bandwidth. The bandwidth is NVLink's 450 GB/s when the whole group sits in one node and InfiniBand's 50 GB/s per GPU when it does not.",
      "TP: 4 all-reduces per layer per microbatch, 6 with Full recompute, of 2 × b × s × h bytes, times the layers on a stage. PP: 2 sends of 2 × b × s × h ÷ t bytes per microbatch. EP: 4 all-to-alls of 2 × b × s × 2 × h bytes per MoE layer.",
      "DP: an all-reduce of 2 bytes per parameter on the GPU over the d replicas (Mixtral's experts over d ÷ e), times 1.5 at ZeRO stage 3. With overlap on, the exposed part is max(0, sync − ⅔ × one microbatch's compute); with overlap off, it is the whole sync.",
    ],
    controls: [
      "`Overlap gradient sync` here; the parallel degrees, micro-batch size, and microbatches on the Parallel layout card; `ZeRO stage` and `Recompute` on the Memory per GPU card.",
      "Comparison worth running: the 70B model at Pipeline 4 with one sequence per microbatch, Tensor parallel 8 against 16. The TP all-reduce goes from 0.042 s to 0.403 s per microbatch.",
    ],
    notice: [
      "A ring all-reduce's 2(n − 1)/n barely moves with n, so more data-parallel replicas add almost no sync time per GPU, while crossing a node boundary costs a factor of nine.",
      "The DP sync is paid once per step; tensor, pipeline, and expert traffic is paid on every microbatch.",
    ],
    limits: [
      "In this lab: bandwidth terms only, with no latency per message, no congestion, no stragglers, and no hierarchical or tree collectives. Tensor, pipeline, and expert traffic is never overlapped with compute, and the overlap rule for the gradient sync is a heuristic.",
      "In general: real collectives pay a latency on every message, share network links with other traffic, wait for the slowest GPU in the group, and are tuned per cluster; libraries choose ring or tree algorithms by message size and topology.",
    ],
  },

  "Step time and MFU": {
    title: "Where a step's seconds go, and how much of the hardware does model work",
    summary:
      "A step is the pipeline's m + p − 1 microbatch slots plus whatever data-parallel sync is left exposed. Splitting it into compute, recomputation, traffic, bubble, and exposed sync shows where the time goes; MFU is the share of the GPUs' peak arithmetic spent on the model's own work.",
    whatYouSee: [
      "A stacked strip and a bar list of `Compute`, `Recompute`, `TP, PP and EP traffic`, `Pipeline bubble`, and `Exposed DP sync`, each with its share of the step.",
      "A bar of the GPUs' peak arithmetic from 0 to 100%: MFU, then any recomputation, then the share lost to traffic, bubble, and exposed sync up to a line at the 75% kernel efficiency, and the quarter beyond it that is out of reach.",
      "Metrics `Step time`, `Tokens/s`, `MFU`, and `HFU`, and a badge with MFU.",
    ],
    howItWorks: [
      "Compute per microbatch t_c = (6 × active parameters + 12 × L × h × s) × b × s × r ÷ (t × p × 0.75 × 989.5 TFLOP/s), with r = 4/3 for Full recompute and 1 otherwise. The 0.75 is the stated kernel efficiency.",
      "Step = (m + p − 1) × (t_c + TP, PP, and EP time per microbatch) + exposed DP sync. Tokens/s = d × m × b × s ÷ step.",
      "MFU = (6 × active parameters + 12 × L × h × s) × tokens per second ÷ (GPUs × 989.5 TFLOP/s), PaLM's definition. HFU = MFU × r, so it counts the rerun forward pass.",
    ],
    controls: [
      "Every control except the three on the Failures and checkpoints card changes this card.",
      "Comparison worth running: the 8B model at ZeRO stage 3 with one sequence per microbatch. Selective gives 68.6% MFU; Full gives 53.6% MFU and 71.5% HFU.",
    ],
    notice: [
      "MFU can never pass 75% here, the kernel efficiency; every point below that is traffic, bubble, or exposed sync.",
      "Tokens/s grows almost in proportion to Data parallel while MFU barely moves, because each data-parallel copy brings its own work and pays one sync per step.",
    ],
    limits: [
      "In this lab: one fixed kernel efficiency of 0.75 stands for everything that slows arithmetic below peak; tensor, pipeline, and expert traffic is charged in full; the optimizer step, data loading, and stragglers are not timed. The 16,384-GPU layout gives 33.2% MFU here against the 41% Llama 3 reported, mostly because the lab charges plain 1F1B's bubble.",
      "In general: MFU is measured from real step times, and published values depend on how FLOPs are counted, whether attention is included, and which peak is used, so MFU, HFU, and TFLOP/s per GPU are not interchangeable.",
    ],
  },

  "Failures and checkpoints": {
    title: "How often a job is interrupted, and how often it should save",
    summary:
      "Any one GPU failing stops a synchronous job, so the job's mean time between interruptions shrinks in proportion to its GPU count. Saving costs time, and every failure throws away the work since the last save plus a restart; Young's and Daly's formulas give the interval that loses the least.",
    whatYouSee: [
      "`Checkpoint every` 5 to 120 min, `Save time` 5 to 600 s, and `Restart time` 1 to 60 min.",
      "A curve of the share of time lost against the interval, a dot at the current setting, and dashed lines at the Young and Daly optima, labelled with an arrow when they fall off the axis. Two thinner lines split the curve into its two terms, saving and redone work, and a line under the plot gives both at the current setting.",
      "Metrics `Job MTBF`, `Young optimum`, `Daly optimum`, and `Time lost`, and a kicker with the checkpoint size.",
    ],
    howItWorks: [
      "Job MTBF = 50,677 GPU-hours ÷ GPUs. The 50,677 is 16,384 GPUs × 54 days × 24 hours ÷ 419 unexpected interruptions, from the Llama 3 paper's snapshot, assuming every GPU ran throughout.",
      "Time lost = save ÷ interval + (interval ÷ 2 + restart) ÷ MTBF. Young's optimum √(2 × save × MTBF) minimizes it; Daly's is √(2δM)[1 + ⅓√(δ/2M) + (δ/2M)/9] − δ for a save time δ under 2M, and M beyond that.",
      "Checkpoint size = 14 bytes per parameter: bf16 weights and the fp32 master copy, momentum, and variance. Gradients are rebuilt by the next step, so they are not saved.",
    ],
    controls: [
      "The three sliders here. The GPU count from the Parallel layout card sets the MTBF, and the model sets the checkpoint size.",
      "Comparison worth running: press 16,384-GPU layout and sweep Checkpoint every from 5 to 120 min. Time lost is 26.7% at 5, lowest near 19 min at 15.8%, and 38.6% at 120.",
    ],
    notice: [
      "The MTBF falls in proportion to the GPU count, but the best interval only with its square root: 109 min at 512 GPUs and 19.3 min at 16,384.",
      "The curve is flat near its minimum, so a few minutes either way cost little, while far too short or far too long costs a lot.",
    ],
    limits: [
      "In this lab: failures are independent and memoryless at a rate taken from one 54-day snapshot of one cluster; the save time is an input, not derived from the checkpoint size; and Time lost reads 100%+ once failures come faster than the job can save and restart, where the first-order formula no longer holds.",
      "In general: failures cluster and recur, asynchronous and in-memory checkpoints hide much of the save time, restarts include rescheduling and warm-up, and Llama 3 reports keeping effective training time above 90% with automated recovery.",
    ],
  },
};

export default cardInfo;
