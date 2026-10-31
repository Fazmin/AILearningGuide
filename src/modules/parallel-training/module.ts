import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateParallelState, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-52-parallel-training",
  slug: "parallel-training",
  title: "Splitting training across GPUs",
  group: "pretraining",
  order: 37,
  icon: "Waypoints",
  accent: "#2f7d95",
  prerequisites: ["module-17-training-run", "module-34-why-gpus"],
  estimatedMinutes: 16,
  steps: ["Shard the state", "Count the activations", "Split each layer", "Fill the pipeline", "Pay for the sync", "Plan for failure"],
  stepInstructions: [
    "With Model at Llama 3.1 8B and Data parallel at 64 replicas, move ZeRO stage from 0 to 3 and watch which Memory per GPU bars shrink and when Fits in 80 GB turns to yes.",
    "Raise Micro-batch size from 1 to 4, then switch Recompute to None, Selective, and Full in turn, and compare the Activations bar in Memory per GPU with Compute and Recompute in Step time and MFU.",
    "Switch Model to Llama 3.1 70B and set Pipeline to 4, then raise Tensor parallel from 1 to 8 to 16 and watch the TP all-reduce row in Communication per step jump once the group leaves the node.",
    "Set Pipeline to 8 and Microbatches to 1, read Bubble in Pipeline schedule, then raise Microbatches to 32 and watch the hatched idle time shrink.",
    "Raise Data parallel from 16 to 256, then switch Overlap gradient sync off and ZeRO stage to 3, reading Exposed DP time, MFU, and Tokens/s after each change.",
    "Press 16,384-GPU layout, then move Checkpoint every from 5 to 120 min and find the lowest Time lost against the marked Young and Daly optima.",
  ],
  stateVersion: 1,
  tagline:
    "Price a training step on thousands of H100s from closed-form arithmetic: what each GPU holds under ZeRO and recomputation, what tensor, pipeline, data, and expert parallelism send, how much time the pipeline bubble and the gradient sync cost, and how often a job that size has to save.",
  objectives: [
    "Split the memory one GPU holds into weights, gradients, optimizer state, and activations, and predict what each ZeRO stage and each parallel degree divides",
    "Say what data, tensor, pipeline, and expert parallelism each split and send, and why tensor parallelism stays inside a node",
    "Compute the pipeline bubble from stages and microbatches, and explain how more microbatches shrink it",
    "Read step time, MFU, and tokens per second as compute plus exposed communication plus idle time, and explain why MFU leaves recomputation out",
    "Choose a checkpoint interval with Young's and Daly's formulas, and explain why failures become routine at scale",
  ],
  glossary: [
    {
      term: "Data parallelism",
      definition:
        "Running a full copy of the model, or of its tensor-and-pipeline slice, on each of d groups of GPUs, each on its own share of the batch, and averaging the gradients once per step. It multiplies throughput but on its own divides no memory: every copy holds the whole model state.",
    },
    {
      term: "Tensor parallelism",
      definition:
        "Splitting each weight matrix of a layer across t GPUs so that each computes part of every matrix multiply, then combining the parts with all-reduces, four per layer per microbatch. That traffic is so frequent that the group is kept inside one node's NVLink.",
    },
    {
      term: "Pipeline parallelism",
      definition:
        "Splitting the stack of layers into p stages on different GPUs and passing activations forward and gradients back between neighbouring stages. The traffic is light, but stages sit idle while the pipeline fills and drains.",
    },
    {
      term: "Expert parallelism",
      definition:
        "Placing the experts of a mixture-of-experts layer on different GPUs and sending each token to the GPUs that hold its chosen experts with an all-to-all exchange. It divides expert memory, and costs four all-to-alls per MoE layer per microbatch.",
    },
    {
      term: "ZeRO stage",
      definition:
        "How much of the model state ZeRO shards across the data-parallel GPUs: stage 1 the optimizer state, stage 2 also the gradients, stage 3 also the weights. Stages 1 and 2 move the same bytes as plain data parallelism; stage 3 moves 1.5 times as many.",
    },
    {
      term: "All-reduce",
      definition:
        "A collective that leaves every GPU in a group holding the sum of everyone's buffers. A ring all-reduce sends and receives 2(n − 1)/n of the buffer per GPU, so its cost per GPU barely grows with the group size.",
    },
    {
      term: "Reduce-scatter and all-gather",
      definition:
        "The two halves of an all-reduce: reduce-scatter leaves each of n GPUs with the sum of one 1/n slice, and all-gather gives every GPU every slice. ZeRO uses the pair so that each GPU updates only the slice it owns.",
    },
    {
      term: "Interconnect",
      definition:
        "The links between GPUs: NVLink inside a node, 450 GB/s each way on an H100, and a network such as InfiniBand between nodes, 50 GB/s per GPU in this lab. The ninefold gap decides where each kind of parallelism can go.",
    },
    {
      term: "Gradient accumulation",
      definition:
        "Running several microbatches forward and backward and summing their gradients before one optimizer step. It grows the batch without more memory, and in a pipeline the microbatches are what keep the stages busy.",
    },
    {
      term: "Pipeline bubble",
      definition:
        "The time pipeline stages spend idle while the first microbatches fill the stages and the last ones drain: (p − 1)/(m + p − 1) of the pipeline's time for p stages and m microbatches under the 1F1B schedule.",
    },
    {
      term: "Activation recomputation",
      definition:
        "Discarding intermediate results during the forward pass and recomputing them in the backward pass. Full recomputation keeps only each layer's input and costs a third more arithmetic; selective recomputation redoes only the attention scores, which are large but cheap to recompute.",
    },
    {
      term: "MFU",
      definition:
        "Model FLOPs utilization: the model's own FLOPs per token times tokens per second, over the combined peak of the GPUs. Recomputed work does not count, so it measures useful throughput; hardware FLOPs utilization (HFU) counts it.",
    },
    {
      term: "Mean time between failures",
      definition:
        "The average time between interruptions. For a synchronous job it shrinks in proportion to the number of GPUs, because any one failure stops the whole job: about 3 hours on 16,384 H100s by Llama 3's numbers.",
    },
    {
      term: "Checkpoint interval",
      definition:
        "The time between saves of the full training state. Saving often wastes time writing; saving rarely redoes more work after each failure. Young's interval, the square root of 2 × save time × MTBF, balances the two to first order.",
    },
  ],
  references: [
    {
      authors: "Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He",
      title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
      source: "SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, 1–16",
      year: 2020,
      url: "https://arxiv.org/abs/1910.02054",
      note: "Section 3.1 counts 16 bytes per parameter for mixed-precision Adam, and Figure 1 gives the 120, 31.4, 16.6, and 1.9 GB per GPU for 7.5 billion parameters on 64 GPUs that this lab's ZeRO stages reproduce. Section 7 shows stages 1 and 2 move the same bytes as plain data parallelism and stage 3 moves 1.5 times as many.",
    },
    {
      authors: "Mohammad Shoeybi, Mostofa Patwary, Raul Puri, et al.",
      title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
      source: "arXiv preprint arXiv:1909.08053",
      year: 2019,
      url: "https://arxiv.org/abs/1909.08053",
      note: "Introduces the tensor parallelism this lab prices: each layer's matrices are split so that a transformer layer needs two all-reduces in the forward pass and two in the backward pass, the four per layer the TP all-reduce row charges.",
    },
    {
      authors: "Deepak Narayanan, Mohammad Shoeybi, Jared Casper, et al.",
      title: "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM",
      source: "Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC21), 1–15",
      year: 2021,
      url: "https://arxiv.org/abs/2104.04473",
      note: "Combines tensor, pipeline, and data parallelism to reach 502 petaFLOP/s on 3,072 GPUs. It gives the pipeline bubble of (p − 1)/m against the ideal time, the 1F1B schedule that keeps at most p microbatches in flight, the interleaved schedule that divides the bubble, and the takeaway to keep tensor parallelism within a server.",
    },
    {
      authors: "Vijay Korthikanti, Jared Casper, Sangkug Lym, et al.",
      title: "Reducing Activation Recomputation in Large Transformer Models",
      source: "Proceedings of Machine Learning and Systems 5 (MLSys 2023), 341–353",
      year: 2023,
      url: "https://arxiv.org/abs/2205.05198",
      note: "Table 2 gives the activation bytes per layer this lab uses: sbh(34 + 5as/h) with no recomputation, divided by t with sequence parallelism, 34sbh/t with selective recomputation, and 2sbh with full recomputation. It also counts the full rerun at about a third more arithmetic and measures 30 to 40% extra time.",
    },
    {
      authors: "Yanping Huang, Youlong Cheng, Ankur Bapna, et al.",
      title: "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism",
      source: "Advances in Neural Information Processing Systems 32 (NeurIPS 2019)",
      year: 2019,
      url: "https://arxiv.org/abs/1811.06965",
      note: "Splits a model into pipeline stages fed with micro-batches and gives the bubble overhead as (K − 1)/(M + K − 1) for K stages and M micro-batches, the formula on the Pipeline schedule card. It reports the overhead is negligible once there are at least four micro-batches per stage.",
    },
    {
      authors: "Shen Li, Yanli Zhao, Rohan Varma, et al.",
      title: "PyTorch Distributed: Experiences on Accelerating Data Parallel Training",
      source: "Proceedings of the VLDB Endowment 13(12), 3005–3018",
      year: 2020,
      url: "https://arxiv.org/abs/2006.15704",
      note: "Describes how PyTorch's DistributedDataParallel groups gradients into buckets, 25 MB by default, and starts all-reducing each bucket before the backward pass has finished. That overlap is what the Overlap gradient sync switch models.",
    },
    {
      authors: "Yanli Zhao, Andrew Gu, Rohan Varma, et al.",
      title: "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel",
      source: "Proceedings of the VLDB Endowment 16(12), 3848–3860",
      year: 2023,
      url: "https://arxiv.org/abs/2304.11277",
      note: "Describes PyTorch's Fully Sharded Data Parallel, motivated by ZeRO, which keeps parameters, gradients, and optimizer state sharded and gathers each unit only when needed. It reports that full sharding costs 1.5 times the communication of plain data parallelism, as ZeRO stage 3 does here.",
    },
    {
      authors: "Nouamane Tazi, Ferdinand Mom, Haojun Zhao, et al.",
      title: "The Ultra-Scale Playbook: Training LLMs on GPU Clusters",
      source: "Hugging Face, free to read online",
      year: 2025,
      url: "https://huggingface.co/spaces/nanotron/ultrascale-playbook",
      note: "A step-by-step guide to data, tensor, sequence, context, pipeline, and expert parallelism and the three ZeRO strategies, with measurements. It states the rule the TP all-reduce row shows, that tensor parallelism is usually kept within a node at a degree of 8 or less, and that FlashAttention already recomputes the attention scores, which is why Selective is the lab's default.",
    },
    {
      authors: "Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, et al.",
      title: "PaLM: Scaling Language Modeling with Pathways",
      source: "Journal of Machine Learning Research 24(240), 1–113",
      year: 2023,
      url: "https://jmlr.org/papers/v24/22-1144.html",
      note: "Defines model FLOPs utilization as observed tokens per second over the throughput at peak FLOPs, counting only the forward and backward passes and not rematerialization. Appendix B gives the 6N + 12LHQT FLOPs per token the MFU readout uses, and PaLM 540B reports 46.2% MFU against 57.8% hardware FLOPs utilization.",
    },
    {
      authors: "Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. (Llama Team, AI @ Meta)",
      title: "The Llama 3 Herd of Models",
      source: "arXiv preprint arXiv:2407.21783",
      year: 2024,
      url: "https://arxiv.org/abs/2407.21783",
      note: "Table 3 gives the 8B, 70B, and 405B shapes, and Table 4 the 16,384-GPU layout behind the preset (tensor 8, pipeline 16, data 128, 41% MFU). Section 3.3 describes eight NVLink-connected H100s per server, 400 Gbps between GPUs, and FSDP sharding, and reports 419 unexpected interruptions in a 54-day snapshot, from which this lab derives its failure rate.",
    },
    {
      authors: "Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, et al.",
      title: "Mixtral of Experts",
      source: "arXiv preprint arXiv:2401.04088",
      year: 2024,
      url: "https://arxiv.org/abs/2401.04088",
      note: "Table 1 gives the Mixtral 8x7B shape the lab prices: 32 layers of width 4,096 with 8 experts per layer and 2 chosen per token, so each token uses about 13 billion of 47 billion parameters. That gap is why its memory and its compute are counted differently.",
    },
    {
      authors: "Ziheng Jiang, Haibin Lin, Yinmin Zhong, et al.",
      title: "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs",
      source: "21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24), 745–760",
      year: 2024,
      url: "https://arxiv.org/abs/2402.15627",
      note: "Reports 55.2% MFU training a 175 billion parameter model on 12,288 GPUs, and states that failures and stragglers are the norm rather than the exception at that scale, which is why it detects and recovers from them automatically.",
    },
    {
      authors: "John W. Young",
      title: "A first order approximation to the optimum checkpoint interval",
      source: "Communications of the ACM 17(9), 530–531",
      year: 1974,
      url: "https://www.semanticscholar.org/paper/fb35b5bc1e02de4d9b31176c39247ee9ad6c3290",
      note: "The origin of the square-root rule: the best interval between checkpoints is about the square root of 2 × the time to save × the mean time between failures, the Young optimum marked on the Failures and checkpoints card.",
    },
    {
      authors: "J. T. Daly",
      title: "A higher order estimate of the optimum checkpoint interval for restart dumps",
      source: "Future Generation Computer Systems 22(3), 303–312",
      year: 2006,
      url: "https://laro.lanl.gov/esploro/outputs/journalArticle/A-higher-order-estimate-of-the/9916364420003761",
      note: "Refines Young's estimate with higher-order terms and a cap when the save time approaches twice the mean time between failures. The Daly optimum on the card uses that formula, a little shorter than Young's at these settings.",
    },
    {
      authors: "NVIDIA",
      title: "NVIDIA H100 Tensor Core GPU",
      source: "NVIDIA data-center product page",
      year: 2026,
      url: "https://www.nvidia.com/en-us/data-center/h100/",
      note: "Lists the H100 SXM at 80 GB of memory, 1,979 teraFLOPS of BF16 with sparsity, and 900 GB/s of NVLink. The lab takes half the teraFLOPS figure, 989.5 dense, and half the NVLink figure, 450 GB/s each way, as stated assumptions.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "At Llama 3.1 8B on 64 data-parallel GPUs, you move ZeRO stage from 1 to 2. Which Memory per GPU bar shrinks?",
      options: [
        "Weights, because stage 2 is the first stage that keeps each weight on only one GPU",
        "Gradients, because each GPU now reduce-scatters them and keeps only its 1/64 slice",
        "Activations, because sharding the gradients also drops the layer outputs each GPU stored",
        "Optimizer state, which stage 1 left whole on every GPU until stage 2 split it",
      ],
      answer: 1,
      explanation:
        "Stage 1 shards the optimizer state, stage 2 adds the gradients, and stage 3 the weights. At stage 2 each GPU reduce-scatters the gradients and keeps the 1/64 it will update, so Gradients falls from 16.1 GB to 250.9 MB while Weights stays whole until stage 3. Activations belong to the microbatch each GPU processes, so no ZeRO stage touches them.",
      objective: 0,
    },
    {
      prompt:
        "On the 8B model at Data parallel 64 and ZeRO stage 0, Model state reads 128.5 GB on every GPU. Why do more replicas not shrink it?",
      options: [
        "Each replica holds the whole model state and differs only in the data it reads",
        "Replicas split the layers, but each still stores every weight for its backward pass",
        "The optimizer state grows with the number of replicas and cancels the saving",
        "Replicas share one copy of the weights, but each keeps its own optimizer state",
      ],
      answer: 0,
      explanation:
        "Data parallelism splits the batch, not the model: every replica runs the same full model on its own sequences and all-reduces the gradients once per step, so each needs all 16 bytes per parameter, 128.5 GB for 8.03 billion parameters. Tensor and pipeline parallelism split the model itself, and ZeRO shards the replicated state across the data-parallel GPUs.",
      objective: 1,
    },
    {
      prompt:
        "On the 8B model, raising Micro-batch size from 1 to 4 pushes Total per GPU far past 80 GB at Selective recompute. Which change brings it back under, and at what price?",
      options: [
        "ZeRO stage 3, which shards the stored activations across the 64 replicas as well",
        "More data parallelism, which gives each replica a smaller part of every stored layer output",
        "Recompute None, which frees the space the recompute buffers were holding",
        "Recompute Full, which keeps only each layer's input and adds a third more arithmetic",
      ],
      answer: 3,
      explanation:
        "Activations belong to the microbatch each GPU processes, so ZeRO and data parallelism leave them alone: at 4 sequences of 8,192 tokens, Selective needs 146 GB for the 32 layers. Full recomputation keeps only each layer's input, 8.6 GB, and reruns the forward pass during the backward pass, which Step time shows as a Recompute row a third the size of Compute. None is the largest of the three, 1.5 TB, because it also keeps the attention scores.",
      objective: 0,
    },
    {
      prompt:
        "On Llama 3.1 70B with Pipeline at 4 and one sequence per microbatch, raising Tensor parallel from 8 to 16 makes the TP all-reduce row about ten times longer. Why?",
      options: [
        "Each all-reduce now carries twice the bytes, because every GPU holds half as much of each matrix",
        "Sixteen GPUs span two 8-GPU nodes, so the all-reduces leave NVLink for the slower network",
        "Tensor parallelism adds four more all-reduces per layer each time the group doubles in size",
        "The stages now wait for each other twice as long, and the lab counts that wait as tensor traffic",
      ],
      answer: 1,
      explanation:
        "A tensor-parallel layer all-reduces a 2 × b × s × h byte activation four times per microbatch, and that buffer does not shrink as t grows. At 8 the group fills one node and uses NVLink at 450 GB/s; at 16 it spans two nodes and is limited by InfiniBand at 50 GB/s, so at Selective recompute 0.042 s per microbatch becomes 0.403 s. That is why tensor parallelism is kept inside a node.",
      objective: 1,
    },
    {
      prompt:
        "A pipeline has 8 stages. You raise Microbatches from 1 to 32 at the same Micro-batch size. What happens to Bubble, and what else changes?",
      options: [
        "It stays at 87.5%, because the bubble depends only on how many stages there are",
        "It falls from 87.5% to 17.9%, and every step now finishes sooner than before",
        "It falls from 87.5% to 17.9%, and each step processes 32 times as many tokens",
        "It falls to 0%, because a pipeline with enough microbatches never leaves a stage idle",
      ],
      answer: 2,
      explanation:
        "Bubble = (p − 1)/(m + p − 1): 7/8 at one microbatch and 7/39 at 32. The 7 idle slots stay; the useful work around them grows, so the step takes longer in seconds but wastes a smaller share. Each microbatch is a gradient-accumulation step, so 32 of them also make the batch 32 times larger, which is why the bubble cannot be removed for free.",
      objective: 2,
    },
    {
      prompt:
        "Llama 3 reported 41% MFU on the 16,384-GPU layout, and this lab gives 33.2%. Which technique that Llama 3 used to cut idle time does the lab leave out?",
      options: [
        "An interleaved schedule, with several smaller stages on each GPU to shrink the bubble",
        "A faster network between nodes, which would shorten the gradient sync to nothing",
        "Storing activations in lower precision, which would let more microbatches fit in memory at once",
        "Skipping the optimizer step on most iterations, which removes most of its cost",
      ],
      answer: 0,
      explanation:
        "With 16 stages and 16 microbatches, plain 1F1B idles 15 of 31 slots, a 48.4% bubble. Llama 3 used an interleaved schedule with several virtual stages per GPU, which divides the bubble by the number of virtual stages at the cost of more pipeline traffic. The lab also fixes kernel efficiency at 0.75, so its MFU is an estimate, not a forecast.",
      objective: 2,
    },
    {
      prompt:
        "You switch Recompute from Selective to Full on a layout that already fits. What happens to MFU and HFU?",
      options: [
        "Both rise, because the GPUs now do more arithmetic in every step they run",
        "Both fall, because the recomputed pass counts as idle time in either measure",
        "MFU rises and HFU falls, because storing less leaves the GPUs more time for arithmetic",
        "MFU falls and HFU rises, because the rerun is real arithmetic but not model work",
      ],
      answer: 3,
      explanation:
        "MFU counts only the model's own FLOPs per token, 6 per active parameter plus the attention term, so the rerun forward pass makes the step longer without adding model work. HFU counts every FLOP the hardware runs, including the rerun, a third more. On the 8B model at ZeRO 3 and one sequence per microbatch, MFU falls from 68.6% to 53.6% while HFU reads 71.5%.",
      objective: 3,
    },
    {
      prompt:
        "On Llama 3.1 70B at Tensor parallel 8, Pipeline 8, Data parallel 256, and 32 microbatches, switching Overlap gradient sync off raises Exposed DP time from 0.032 s to 0.088 s. Why did overlap hide most of it?",
      options: [
        "Overlap compresses the gradients to half their size before they are sent",
        "The sync runs under the last microbatch's backward pass, so only the excess shows",
        "Overlap sends the gradients over NVLink instead of over the slower network between nodes",
        "Overlap skips the sync on most steps and catches up on the steps that follow",
      ],
      answer: 1,
      explanation:
        "The gradients of the last layers are final while earlier layers are still running backward, so their all-reduce can start then, as PyTorch's DistributedDataParallel does with gradient buckets. The lab lets the sync hide under two thirds of one microbatch's compute and charges only what is left. The bytes and the link do not change, and skipping syncs would let the replicas drift apart.",
      objective: 3,
    },
    {
      prompt:
        "The 16,384-GPU layout shows one interruption every 3.09 hours, while 512 GPUs see one every 99 hours. Where does the difference come from?",
      options: [
        "Big jobs run on older GPUs, which the lab assumes fail more often than new ones",
        "Interruptions grow with the square of the GPU count because of network traffic",
        "Each GPU fails no more often, but 32 times as many GPUs fail 32 times as often",
        "The larger model takes longer to save, and most failures happen while saving",
      ],
      answer: 2,
      explanation:
        "The lab divides 50,677 GPU-hours per interruption, from Llama 3's 419 unexpected interruptions in 54 days, by the GPU count: any one failure stops the whole synchronous job, so failure rates add up. 16,384 is 32 times 512, so the mean time falls from 99.0 hours to 3.09. Young's interval falls with its square root, from 109 to 19.3 minutes at a 60-second save.",
      objective: 4,
    },
    {
      prompt:
        "At 16,384 GPUs with a 60-second save, Time lost is lowest near 19 minutes. Which assumption behind that curve fails first in a real cluster?",
      options: [
        "Failures arrive independently at a steady rate, while real ones cluster and recur",
        "Saving takes time, while real clusters save every checkpoint instantly in the background",
        "Every failure costs a restart, while real restarts take no time at all",
        "The interval is fixed, while real runs never save their state on any regular schedule",
      ],
      answer: 0,
      explanation:
        "Young's and Daly's formulas assume failures are independent and memoryless with a known mean, which the lab takes from one 54-day snapshot. Real failures cluster, since a faulty node fails again and one network event can stop many jobs, and the rate changes over a run. Saving does take time, which asynchronous checkpointing hides only in part, and restarts are never free.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateParallelState,
};

export default definition;
