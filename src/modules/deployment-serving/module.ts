import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { DEFAULT_CONFIG, normalizeConfig } from "./operate";

const initialState: ModuleState = { ...DEFAULT_CONFIG };

/**
 * Version 1 stored only the model, format, users, context, and scheduler; the hardware, lever, and cost keys that
 * version 2 added take their defaults. Every key is validated and clamped to its control.
 */
export function hydrateServingState(value: string): ModuleState {
  try {
    const parsed: unknown = JSON.parse(value);
    return { ...normalizeConfig(typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : {}) };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-28-deployment-serving",
  slug: "deployment-serving",
  title: "Deployment & serving",
  group: "training-adapting",
  order: 50,
  icon: "Server",
  accent: "#1f8f86",
  prerequisites: ["module-26-quantization", "module-11-inference-kv-cache"],
  estimatedMinutes: 16,
  steps: ["Add users", "Extend the context", "Balance the budget", "Change the hardware", "Pull the levers", "Export it"],
  stepInstructions: [
    "Raise Concurrent users from 1 toward 64 and compare Total with Per user, then find where the shaded region begins: past that count the KV cache no longer fits.",
    "Set Concurrent users to 1 and raise Context length to 32,768, and watch the KV cache segment approach the weights while Streams that fit drops.",
    "Pick the 2B · this app model and read Per token, per stream and Fixed state per stream, then find a Model, Weight format, Context length, and Concurrent users that fit with headroom to spare, using Streams that fit as the cap.",
    "In Hardware, levers, and cost, switch Hardware between RTX 4090 and H100 SXM at the same load and compare Streams that fit, Per user, and Cost per million tokens, then edit Price and watch the cost move.",
    "Switch on Paged attention and Speculative decoding, move the KV cache type from fp16 to q8_0, and compare the Streams that fit and Per user in each lever's row of the bar list.",
    "Read the export checklist for the Weight format you selected and compare its block bits per weight with the file figure in the Serving controls note.",
  ],
  stateVersion: 2,
  tagline:
    "Budget memory, decode speed, and cost per million tokens from stated assumptions and real model shapes, including the model this app runs, then walk the export path and what it takes to operate a served model.",
  objectives: [
    "Separate the fixed cost of weights from the per-stream cost of the KV cache",
    "Explain why batching raises throughput and lowers per-user speed",
    "List what an export has to produce before a model can be served",
    "Predict how hardware, paged attention, KV-cache quantization, and speculative decoding change streams that fit, speed, and cost per million tokens",
    "Name what operating a served model adds beyond speed: monitoring and drift, versioned rollouts, guardrails, and where prompts go",
  ],
  glossary: [
    {
      term: "KV cache",
      definition:
        "Stored keys and values for every token already processed, in each layer that attends over the whole context. It is charged per token per stream, which makes it the memory term that scales with load.",
    },
    {
      term: "Memory bandwidth bound",
      definition:
        "Limited by how fast bytes can be read rather than by arithmetic. Decoding reads every weight to produce one token per stream, so it sits here at any batch size this lab offers on its stated hardware.",
    },
    {
      term: "Continuous batching",
      definition:
        "Scheduling one decode step at a time: finished requests leave and waiting ones join at every step, so the batch stays full. Every stream in the batch shares one read of the weights.",
    },
    {
      term: "Throughput and latency",
      definition:
        "Total tokens per second across all users, against tokens per second for one user, whose inverse is the time per output token. Batching improves the first and worsens the second.",
    },
    {
      term: "Time to first token",
      definition:
        "Queueing plus prefill, the pass over the whole prompt before generation starts. Prefill is compute bound and grows with prompt length.",
    },
    {
      term: "Paged attention",
      definition:
        "Storing the KV cache in small fixed-size blocks allocated as a stream grows, as vLLM's PagedAttention does with 16 tokens by default. A stream holds only what it has filled instead of a full-context reservation, so the waste per stream is under one block.",
    },
    {
      term: "Speculative decoding",
      definition:
        "A small draft model proposes γ tokens and the large model checks them in one pass, accepting a prefix by rejection sampling. The output distribution is unchanged; with acceptance rate α each pass yields (1 − α^(γ+1)) / (1 − α) tokens, and the wall-time gain divides that by 1 + γc, where c is the draft's cost against the target's.",
    },
    {
      term: "Hybrid attention",
      definition:
        "Mixing full-attention layers, whose cache grows with context, with recurrent or linear-attention layers that keep a fixed-size state per stream. Qwen3.5 2B has 6 full-attention layers among its 24, so only those are charged per token.",
    },
    {
      term: "KV-cache quantization",
      definition:
        "Storing cached keys and values in fewer bits, such as llama.cpp's q8_0 at 8.5 bits per value or q4_0 at 4.5. It shrinks the per-token cost and the bytes read each step, at some cost in accuracy. A recurrent state is a separate store and stays float32 in llama.cpp.",
    },
    {
      term: "Cost per million tokens",
      definition:
        "The hourly price of the hardware over the output tokens it generates in an hour, scaled to a million. It falls as batching raises total throughput and rises with idle time. The price is an input, not a property of the model.",
    },
    {
      term: "Monitoring and drift",
      definition:
        "Watching latency, throughput, queueing, cache occupancy, and error rates in service, and replaying a fixed evaluation to catch drift: a shift in traffic or behaviour that makes yesterday's measurements stop describing today's use.",
    },
    {
      term: "Versioned rollout",
      definition:
        "Serving a pinned model file, engine, and configuration by exact version, sending a small share of traffic to a new one first, and keeping the old one ready so that a rollback is a switch rather than a rebuild.",
    },
    {
      term: "Guardrails",
      definition:
        "Checks and limits around the model rather than inside it: input and output filters, tool permissions, rate and length limits, and refusal policies. They are ordinary software and need their own tests.",
    },
    {
      term: "On-device inference",
      definition:
        "Running the model on the user's own hardware, as this app does with llama.cpp on a loopback port. Prompts stay on the machine and nothing is billed per token, at the price of smaller models and the speed of the local memory system.",
    },
  ],
  references: [
    {
      authors: "Samuel Williams, Andrew Waterman, and David Patterson",
      title: "Roofline: An Insightful Visual Performance Model for Multicore Architectures",
      source: "Communications of the ACM 52(4), 65–76",
      year: 2009,
      url: "https://www.semanticscholar.org/paper/092217c2267f6e0673590aa151d811e579ff7760",
      note: "Introduces the roofline model, which caps speed by either memory bandwidth or arithmetic, whichever runs out first. This lab is a roofline of that kind, and its badge reports which ceiling binds.",
    },
    {
      authors: "Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, et al.",
      title: "Efficiently Scaling Transformer Inference",
      source: "Proceedings of Machine Learning and Systems 5 (MLSys 2023), 606–624",
      year: 2023,
      url: "https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html",
      note: "A Google study of serving large models. It shows that loading the weights dominates each decode step at small batches, while the KV cache, unique to each stream, takes over at large batches and long contexts, which is the Memory budget's main lesson.",
    },
    {
      authors: "Joshua Ainslie, James Lee-Thorp, Michiel de Jong, et al.",
      title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
      source: "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), 4895–4901",
      year: 2023,
      url: "https://aclanthology.org/2023.emnlp-main.298/",
      note: "Introduces grouped-query attention, where several query heads share one key-value head. Fewer KV heads is why the 3B shape's cache per token is so much smaller than the 8B shape's.",
    },
    {
      authors: "Songlin Yang, Jan Kautz, and Ali Hatamizadeh",
      title: "Gated Delta Networks: Improving Mamba2 with Delta Rule",
      source: "International Conference on Learning Representations (ICLR 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2412.06464",
      note: "Introduces the Gated DeltaNet layer, a linear-attention layer with a memory of fixed size, and hybrid models that mix it with attention. These are the layers that give 2B · this app its Fixed state per stream.",
    },
    {
      authors: "Qwen Team",
      title: "Qwen3.5-2B model card",
      source: "Hugging Face model card",
      year: 2026,
      url: "https://huggingface.co/Qwen/Qwen3.5-2B",
      note: "The model card for the model this app runs. Its layer layout shows 24 layers, with three Gated DeltaNet layers before each Gated Attention layer, which is why the lab charges a KV cache for only 6 layers.",
    },
    {
      authors: "Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, et al.",
      title: "Orca: A Distributed Serving System for Transformer-Based Generative Models",
      source: "16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), 521–538",
      year: 2022,
      url: "https://www.usenix.org/conference/osdi22/presentation/yu",
      note: "Introduces iteration-level scheduling, where the server runs one step at a time so requests can join and leave the batch between steps. This is the idea behind the Continuous batching scheduler.",
    },
    {
      authors: "Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al.",
      title: "Efficient Memory Management for Large Language Model Serving with PagedAttention",
      source: "Proceedings of the 29th Symposium on Operating Systems Principles (SOSP 2023), 611–626",
      year: 2023,
      url: "https://arxiv.org/abs/2309.06180",
      note: "The paper behind vLLM and the Paged attention lever. It stores the cache in small blocks (16 tokens by default) and reports the lesson's figure that earlier systems used only 20.4% to 38.2% of KV cache memory for token states.",
    },
    {
      authors: "Amey Agrawal, Nitin Kedia, Ashish Panwar, et al.",
      title: "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve",
      source: "18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), 117–134",
      year: 2024,
      url: "https://www.usenix.org/conference/osdi24/presentation/agrawal",
      note: "Introduces chunked prefill, which splits a long prompt into pieces so new requests join without pausing other streams' decoding. It backs the lesson's point that one long prompt need not stall everyone else.",
    },
    {
      authors: "Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al.",
      title: "SGLang: Efficient Execution of Structured Language Model Programs",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024)",
      year: 2024,
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html",
      note: "Describes RadixAttention, which keeps the KV cache of shared prompt prefixes and reuses it across requests. It is an example of the prefix sharing that makes a long shared system prompt nearly free after the first request.",
    },
    {
      authors: "Yaniv Leviathan, Matan Kalman, and Yossi Matias",
      title: "Fast Inference from Transformers via Speculative Decoding",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 19274–19286",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/leviathan23a.html",
      note: "The source of the Speculative decoding lever. It shows the output distribution is unchanged and gives Theorem 3.8, the wall-time gain with acceptance rate α, draft length γ, and draft cost c that the lab uses.",
    },
    {
      authors: "llama.cpp contributors",
      title: "llama.cpp HTTP server (llama-server) README",
      source: "llama.cpp documentation on GitHub",
      year: 2026,
      url: "https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md",
      note: "Lists the server options the lesson names: --cache-type-k and --cache-type-v with q8_0 and q4_0 for the KV cache type, continuous batching, parallel slots, and a default listening address of 127.0.0.1, as this app's local guide uses.",
    },
    {
      authors: "ggml contributors",
      title: "GGUF file format specification",
      source: "ggml documentation on GitHub",
      year: 2026,
      url: "https://github.com/ggml-org/ggml/blob/master/docs/gguf.md",
      note: "Specifies the single-file format the export checklist writes: a header with a magic number, key-value metadata including the tokenizer, and the tensor data. It backs the Write GGUF step of the export path.",
    },
    {
      authors: "Rob Ewaschuk (Betsy Beyer, ed.)",
      title: "Monitoring Distributed Systems, chapter 6 of Site Reliability Engineering",
      source: "O'Reilly Media, free to read online",
      year: 2016,
      url: "https://sre.google/sre-book/monitoring-distributed-systems/",
      note: "Google's guide to watching a live service, built on four golden signals: latency, traffic, errors, and saturation. It also explains why the slow tail of requests matters more than the average, as the lesson says of tail latency.",
    },
    {
      authors: "D. Sculley, Gary Holt, Daniel Golovin, et al.",
      title: "Hidden Technical Debt in Machine Learning Systems",
      source: "Advances in Neural Information Processing Systems 28 (NeurIPS 2015)",
      year: 2015,
      url: "https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html",
      note: "Explains why machine learning systems need ongoing upkeep. Its section on changes in the external world argues for monitoring a live model, which backs the lesson's warning about drift.",
    },
    {
      authors: "Alec Warner and Štěpán Davidovič",
      title: "Canarying Releases, chapter 16 of The Site Reliability Workbook",
      source: "O'Reilly Media, free to read online",
      year: 2018,
      url: "https://sre.google/workbook/canarying-releases/",
      note: "Defines canarying as sending a change to a small, time-limited share of traffic and checking it before going further. That is the versioned rollout the lesson describes, with rollback kept cheap.",
    },
  ],
  checkpoint: [
    {
      prompt: "You drag Concurrent users from 8 up toward 64 at a fixed context length. Which part of the Memory budget changes?",
      options: [
        "Both the weights and the cache grow, because each new user needs its own copy of the model",
        "Only the KV cache grows; the weights are a fixed cost paid once, however many users connect",
        "Only the activations grow, because the runtime allocates more scratch space for each new user",
        "Neither changes, because the budget is set by the model and the weight format alone",
      ],
      answer: 1,
      explanation:
        "Weights are loaded once and shared by every stream, so they never move with load. The KV cache is charged for every token of every active conversation, so context length times concurrency decides whether a configuration survives real traffic. That is how a model that loads with room to spare can still run out of memory under load.",
      objective: 0,
    },
    {
      prompt:
        "The lab says 36 streams fit at 4,096 tokens on the 8B shape. Why might a real server using paged attention hold more?",
      options: [
        "Paged attention quantizes the cache to four bits, which divides the memory cost of every token by four or more",
        "Real servers skip the activation term, so the flat 1 GiB the lab charges is what limits how many streams fit",
        "Paged attention shares one copy of the weights across all streams, so each extra stream adds no memory at all",
        "The lab reserves each stream's full context up front; paged attention allocates blocks only as a stream fills them",
      ],
      answer: 3,
      explanation:
        "The lab charges every stream for its whole context at once, which is the worst case. Paged attention allocates the cache in small blocks as a stream grows, so streams that have not filled their context hold less and more of them fit. The weights are already shared in the lab, and quantizing the cache is a separate technique.",
      objective: 0,
    },
    {
      prompt: "You raise Concurrent users from 1 to 8 with Continuous batching on. Which readouts move, and in which directions?",
      options: [
        "Total rises and Per user falls: all streams share one read of the weights, while each step also reads more cache",
        "Both rise, because a bigger batch keeps the hardware busier and so every user's tokens arrive faster than before",
        "Total stays the same and Per user falls, because the available bandwidth is simply split evenly between the users",
        "Both fall, because each extra user adds another full copy of the weights that every decode step has to read",
      ],
      answer: 0,
      explanation:
        "A decode step reads every weight once and each stream's cache once, then emits one token per stream, so extra streams ride on the same weight read and total throughput rises. The step also grows because it reads more cache, so each user's own tokens per second fall. The even split in the third option is what the One at a time scheduler does.",
      objective: 1,
    },
    {
      prompt: "At 64 streams the badge still says memory bandwidth bound, not compute bound. Why?",
      options: [
        "The lab ignores compute altogether, so the badge has no other limit that it could ever report",
        "Batching moves the arithmetic off the accelerator, which leaves bandwidth as the only limit left",
        "Each token costs only about two FLOPs per parameter, while every weight byte must be read at every step",
        "More users add more weights to read, so memory traffic always grows faster than the arithmetic",
      ],
      answer: 2,
      explanation:
        "Decoding does about two FLOPs per parameter per token but has to read every weight byte for each step, so the arithmetic finishes long before the bytes arrive. The lab does compute a ceiling for it, far above the throughput the streams reach, and batching helps precisely because more tokens share one weight read.",
      objective: 1,
    },
    {
      prompt: "Which set of artifacts does an export have to produce before llama.cpp can serve the model?",
      options: [
        "The training checkpoint exactly as it was saved, because the engine converts any format when the first request arrives",
        "Only the quantized weights, because the engine rebuilds the tokenizer and metadata from the weights themselves",
        "One GGUF file with the merged, quantized tensors, their metadata, and the tokenizer, plus a header a loader can verify",
        "A separate KV cache file for each user, because the cache has to be loaded before the first token appears",
      ],
      answer: 2,
      explanation:
        "The checklist merges any adapter so inference costs what the base cost, quantizes to the chosen block type, and writes one GGUF file holding tensors, quantization metadata, and tokenizer behind a header the loader checks. This lab names those steps and prices the toy table at the block type; it does not write a file. The KV cache is built at run time, never exported.",
      objective: 2,
    },
    {
      prompt:
        "The 2B · this app preset charges a KV cache for only 6 of its 24 layers. What do the other 18 layers cost per stream?",
      options: [
        "Nothing, because linear-attention layers keep no information between tokens and so need no stored state at all",
        "A fixed recurrent state, shown as Fixed state per stream, that does not grow with the context length",
        "The same cache per token as the other six, which the lab leaves out of the bar to keep the chart readable",
        "A cache that grows with context but is stored at 4 bits, which is why the total looks so small here",
      ],
      answer: 1,
      explanation:
        "Gated DeltaNet layers keep a fixed-size state per stream, about 19 MiB for all 18 layers in float32, so they add a constant per stream and nothing per token. Only the six full-attention layers hold a cache that grows with context: 12 KiB a token, against 48 KiB if all 24 layers cached.",
      objective: 0,
    },
    {
      prompt:
        "Moving one stream from the RTX 4090 to the H100 SXM raises its decode speed by about the ratio of their memory bandwidths, not of their compute. Why?",
      options: [
        "The H100's larger capacity lets each stream read its weights from several places at once, so capacity rather than bandwidth sets speed",
        "Decode does about two FLOPs per parameter but reads every weight byte, so it waits on memory, far below the compute ceiling",
        "Compute decides decode speed, but the lab rounds both cards' compute figures to the same value before dividing",
        "A dearer card runs at higher clocks, and its memory bandwidth simply tracks that clock and the hourly price",
      ],
      answer: 1,
      explanation:
        "One decode step reads every weight byte for about two FLOPs per parameter, so the time is the bytes over the bandwidth. The H100's bandwidth is 3.3 times the 4090's and its dense FP16 compute is 6.0 times, yet one stream speeds up by 3.3. Capacity sets how many streams fit, not how fast one runs.",
      objective: 3,
    },
    {
      prompt:
        "You switch on Speculative decoding with acceptance 0.8 and draft length 4. Per-user speed rises, but by less than the 3.36 tokens a pass. Why?",
      options: [
        "Each pass also pays for the draft model's steps, so the gain is the tokens per pass over one plus the draft length times the draft cost",
        "The target model has to verify every accepted token a second time, so each accepted token costs two passes",
        "Speculative decoding only speeds up the first token of a reply, so the gain is one token per request",
        "The acceptance rate caps the gain at the rate itself, however long the draft is made",
      ],
      answer: 0,
      explanation:
        "A pass yields (1 − α^(γ+1)) / (1 − α) = 3.36 tokens, but it also runs γ draft steps at a cost c of a target step, so the speed-up is 3.36 divided by 1 + γc: 2.4 at c = 0.1. Verification does not repeat work, and the gain applies to every token, not just the first.",
      objective: 3,
    },
    {
      prompt:
        "Paged attention and a q8_0 KV cache both raise Streams that fit on the same hardware. What does each one change?",
      options: [
        "Paged attention charges only the filled pages of each stream, and q8_0 shrinks the bytes per cached value",
        "Both shrink the weights, because the cache is stored inside the model file the engine loads",
        "Paged attention stores the cache at fewer bits and q8_0 splits it into pages, so they are two names for one saving",
        "Paged attention only speeds up prefill and q8_0 only changes accuracy, so neither adds any capacity",
      ],
      answer: 0,
      explanation:
        "Paged attention changes how many tokens each stream is charged for: the filled pages, not the whole reserved context. A q8_0 cache changes the bytes per value, 8.5 bits instead of 16. The two multiply, and neither touches the weights.",
      objective: 3,
    },
    {
      prompt:
        "On the RTX 4090 the Cost per million tokens falls as you raise Concurrent users from 1 toward its Streams that fit. Why?",
      options: [
        "The hourly price stays the same while every extra stream shares one read of the weights, so more tokens come out of each hour",
        "The provider charges less per hour as more streams connect, so each hour of rental costs less once the machine is busy",
        "Each extra stream makes the cache cheaper to store, so the memory part of the bill shrinks as load grows",
        "The cost is computed from the model's parameter count alone, so users only change how it is displayed",
      ],
      answer: 0,
      explanation:
        "Cost per million tokens is the price of an hour over the tokens produced in it. A batched step reads the weights once for all streams, so total tokens per second rises with users while the price does not. It stops improving once memory runs out, where Streams that fit caps the batch.",
      objective: 3,
    },
    {
      prompt:
        "A new model file passes your tests and you roll it out. Which practice makes a bad release cheap to undo?",
      options: [
        "Overwrite the old file in place, so there is only ever one version to maintain",
        "Trust the benchmark score alone, since a model's behaviour cannot drift once it has been exported",
        "Switch off monitoring during the rollout, so alerts do not hide the real problem",
        "Keep the previous pinned file ready and send a small share of traffic to the new one while watching the metrics",
      ],
      answer: 3,
      explanation:
        "A pinned artifact with a hash and a small first slice of traffic turns a rollback into a switch. Overwriting in place destroys the thing you would roll back to, a benchmark says nothing about live traffic that shifts later, and monitoring is how a bad release is noticed in the first place.",
      objective: 4,
    },
    {
      prompt:
        "In this app, where does a prompt go when you chat with the local guide, and what changes with a cloud provider?",
      options: [
        "To a llama-server on 127.0.0.1 on this machine; with a cloud provider the text leaves for that provider's servers",
        "To Hugging Face, which hosts the model file, and a cloud provider only adds a second copy",
        "Into the GGUF file, so later runs can reuse earlier prompts, and a cloud provider reads that file too",
        "Nowhere in either case, because the app is local-first and never sends a prompt off the machine, whichever provider you pick",
      ],
      answer: 0,
      explanation:
        "The local guide runs llama.cpp bound to a loopback address on a random port, started and stopped by the app, so prompts stay on the machine. Choosing a cloud provider sends the text to that provider, under its own terms. Hugging Face only supplies the model file once, at download.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateServingState,
};

export default definition;
