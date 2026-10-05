import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { CONTEXT_RANGE, GENERATED_RANGE, PROMPT_RANGE, SEQUENCE_RANGE } from "./ranges";
import { clampQuality, GAMMA_RANGE, SEED_RANGE } from "./speculative";

const initialState: ModuleState = {
  context: 7,
  generated: 4,
  useCache: true,
  realContext: 8192,
  sequences: 16,
  specGamma: 4,
  specQuality: 0.7,
  specSeed: 7,
};

const intIn = (value: unknown, low: number, high: number, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, Math.round(value))) : fallback;

const definition: ModuleDefinition = {
  id: "module-11-inference-kv-cache",
  slug: "inference-kv-cache",
  title: "Inference & KV cache",
  group: "inside-models",
  order: 18,
  icon: "DatabaseZap",
  accent: "#1b8f72",
  prerequisites: ["module-09-transformer-block"],
  estimatedMinutes: 16,
  steps: ["Prefill the prompt", "Decode one token", "Store keys and values", "Compare recomputation", "Speculate and verify"],
  stepInstructions: [
    "Set Generated tokens to 0 and change Prompt tokens: the whole lower triangle is computed in one pass, because every prompt position is available at once.",
    "Raise Generated tokens one at a time. With Use cache, only the newest row lights up and New Q, K, V stays at 1.",
    "Keep KV cache on Use cache and follow the K and V rows under the triangle: each step writes one slot and only reads the earlier ones.",
    "Switch KV cache to Recompute and compare Q·K this pass and Q·K whole run, then read Prefill versus decode for the same choice at 8,192 tokens.",
    "Under Speculative decoding, set Draft quality to 1.00 and read Closed form, tokens per pass, then lower Draft quality, raise Draft length, and compare Simulated, tokens per pass with the closed form and find a pass where the target refused a proposal.",
  ],
  stateVersion: 2,
  tagline:
    "Learn how generation moves from a parallel prefill to one-token-at-a-time decoding, what caching keys and values saves, and what it costs in memory on a real model.",
  objectives: [
    "Separate prefill from decode",
    "Explain the KV cache memory-speed trade-off",
    "Predict how draft length and draft quality set the tokens one verification pass yields in speculative decoding",
  ],
  glossary: [
    {
      term: "Prefill",
      definition:
        "The forward pass that processes every prompt token in parallel and fills the cache. For prompts longer than a few hundred tokens it is compute bound, and its attention cost grows with the square of prompt length.",
    },
    {
      term: "Decode",
      definition:
        "The phase producing one token per forward pass. At small batch sizes it is limited by memory bandwidth, because every weight and every cached key and value is read to produce one new position per sequence.",
    },
    {
      term: "KV cache",
      definition:
        "Stored attention keys and values for every earlier position, in every layer and key/value head. The causal mask means later tokens never change them, so they are written once and then only read.",
    },
    {
      term: "Grouped-query attention",
      definition:
        "Several query heads sharing one key/value head. Llama 3 8B has 32 query heads and 8 key/value heads, which makes its cache four times smaller than one key/value head per query head.",
    },
    {
      term: "Time to first token",
      definition:
        "Latency before the first output token appears. It is dominated by prefill, and the cache does not shorten it unless the same prompt prefix was already cached.",
    },
    {
      term: "Inter-token latency",
      definition:
        "The delay between successive output tokens during decode. It sets the perceived streaming speed and is governed mostly by memory bandwidth.",
    },
    {
      term: "Memory bandwidth bound",
      definition:
        "Limited by how fast weights and cache entries can be read from memory rather than by arithmetic throughput. This is the normal state of decode.",
    },
    {
      term: "Compute bound",
      definition:
        "Limited by the hardware's arithmetic units. Long prefills are compute bound because many positions share one read of the weights.",
    },
    {
      term: "Arithmetic intensity",
      definition:
        "Operations performed per byte of memory traffic. Below the hardware's ridge point, about 295 FLOP per byte for an H100 in bf16, a kernel waits on memory.",
    },
    {
      term: "Roofline model",
      definition:
        "An estimate of a kernel's time as the larger of its FLOPs divided by peak FLOP/s and its bytes divided by peak bandwidth. It gives a lower bound, not a measurement.",
    },
    {
      term: "Batching",
      definition:
        "Serving many sequences in one pass so a single read of the weights does more work. It raises throughput until cache memory runs out, but each sequence still reads its own cache.",
    },
    {
      term: "Continuous batching",
      definition:
        "Admitting and retiring individual sequences at each decode step instead of processing fixed batches, which raises utilization under real traffic.",
    },
    {
      term: "PagedAttention",
      definition:
        "Managing the cache in fixed-size blocks with a per-sequence block table, which nearly eliminates fragmentation and lets requests share blocks.",
    },
    {
      term: "Speculative decoding",
      definition:
        "A small draft model proposes gamma tokens that the large model checks in one pass. A proposal is kept with probability min(1, p over q); the first refused one is replaced by a sample from the leftover difference of the two distributions, so the output distribution is exactly the large model's. With acceptance rate alpha, a pass yields (1 − alpha^(gamma+1)) ÷ (1 − alpha) tokens on average.",
    },
    {
      term: "Prefix caching",
      definition:
        "Reusing cached blocks for a shared prompt prefix across requests, so a long repeated system prompt is prefilled once instead of on every request.",
    },
    {
      term: "Quantization",
      definition:
        "Storing weights or cache entries at reduced precision. It saves proportional memory and can measurably change output, making it a quality trade rather than free.",
    },
  ],
  references: [
    {
      authors: "Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, et al.",
      title: "Efficiently Scaling Transformer Inference",
      source: "Proceedings of Machine Learning and Systems 5 (MLSys 2023)",
      year: 2023,
      url: "https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html",
      note: "Splits inference into prefill and decode and weighs the time to read weights and the KV cache against the time to do the arithmetic. It explains why weight reads dominate at small batches and why each sequence's cache adds its own memory traffic, as the Prefill versus decode card shows.",
    },
    {
      authors: "Samuel Williams, Andrew Waterman, and David Patterson",
      title: "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures",
      source: "Technical Report UCB/EECS-2008-134, University of California, Berkeley",
      year: 2008,
      url: "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html",
      note: "The original roofline model: a kernel's speed is capped either by peak arithmetic or by memory bandwidth times its arithmetic intensity. The Prefill versus decode card uses this idea to estimate pass times as lower bounds.",
    },
    {
      authors: "NVIDIA",
      title: "NVIDIA H100 GPU",
      source: "NVIDIA data center product page and specifications",
      year: 2026,
      url: "https://www.nvidia.com/en-us/data-center/h100/",
      note: "The spec table for the H100 SXM lists 80 GB of memory, 3.35 TB/s of bandwidth, and BF16 tensor throughput of 1,979 teraFLOPS with sparsity, which is twice the 989 dense figure the lab uses. These are the peaks behind the Memory budget and Prefill versus decode cards.",
    },
    {
      authors: "Tri Dao, Daniel Y. Fu, Stefano Ermon, et al.",
      title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2205.14135",
      note: "Computes exact attention in tiles that fit in fast on-chip memory, so the full score matrix is never written out. This is the real kernel the Causal attention work card mentions next to its toy triangle.",
    },
    {
      authors: "Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al.",
      title: "The Llama 3 Herd of Models",
      source: "arXiv preprint arXiv:2407.21783",
      year: 2024,
      url: "https://arxiv.org/abs/2407.21783",
      note: "Table 3 gives the published shape of Llama 3 8B: 32 layers, 32 attention heads, and 8 key/value heads. The text says grouped-query attention was chosen to shrink the key-value cache during decoding. These are the numbers the Memory budget card turns into 128 KiB per token.",
    },
    {
      authors: "Noam Shazeer",
      title: "Fast Transformer Decoding: One Write-Head is All You Need",
      source: "arXiv preprint arXiv:1911.02150",
      year: 2019,
      url: "https://arxiv.org/abs/1911.02150",
      note: "Argues that one-token-at-a-time decoding is limited by the memory bandwidth needed to reload stored keys and values. It proposes multi-query attention, where all query heads share one key/value head, the extreme case of the sharing in Going deeper.",
    },
    {
      authors: "Joshua Ainslie, James Lee-Thorp, Michiel de Jong, et al.",
      title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
      source: "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), 4895–4901",
      year: 2023,
      url: "https://aclanthology.org/2023.emnlp-main.298/",
      note: "Introduces grouped-query attention, where groups of query heads share a key/value head, between one shared head and one per query head. It reports quality close to full multi-head attention with speed close to multi-query, the trade behind Llama 3 8B's fourfold smaller cache.",
    },
    {
      authors: "DeepSeek-AI",
      title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
      source: "arXiv preprint arXiv:2405.04434",
      year: 2024,
      url: "https://arxiv.org/abs/2405.04434",
      note: "Introduces Multi-head Latent Attention, which compresses keys and values into a smaller latent vector before caching them. It is an example of the latent-attention schemes the lesson says store a compressed form of the cache.",
    },
    {
      authors: "Yaniv Leviathan, Matan Kalman, and Yossi Matias",
      title: "Fast Inference from Transformers via Speculative Decoding",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 19274–19286",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/leviathan23a.html",
      note: "The source the lesson cites for speculative decoding. Algorithm 1 is the accept-or-replace rule that keeps the target's output distribution, the paper derives the expected tokens per run that Closed form, tokens per pass shows, and it reports a 2X to 3X speed-up on T5-XXL.",
    },
    {
      authors: "Tianle Cai, Yuhong Li, Zhengyang Geng, et al.",
      title: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 5209–5235",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/cai24b.html",
      note: "Adds extra decoding heads to the large model so it can guess several later tokens and check them in one step, without a separate draft model. This is the extra-prediction-heads kind of draft the lesson mentions.",
    },
    {
      authors: "Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, et al.",
      title: "Orca: A Distributed Serving System for Transformer-Based Generative Models",
      source: "Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 2022)",
      year: 2022,
      url: "https://www.usenix.org/system/files/osdi22-yu.pdf",
      note: "Proposes iteration-level scheduling: the server picks which requests run at every decode step instead of waiting for a whole batch to finish. That is the continuous batching described in Going deeper.",
    },
    {
      authors: "Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al.",
      title: "Efficient Memory Management for Large Language Model Serving with PagedAttention",
      source: "Proceedings of the 29th Symposium on Operating Systems Principles (SOSP 2023), 611–626",
      year: 2023,
      url: "https://arxiv.org/abs/2309.06180",
      note: "Shows that naive cache allocation wastes memory through fragmentation, then stores the cache in fixed-size blocks like pages of virtual memory. This is the PagedAttention design behind vLLM, with near-zero waste and cache blocks shared between requests.",
    },
    {
      authors: "Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al.",
      title: "SGLang: Efficient Execution of Structured Language Model Programs",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2312.07104",
      note: "Introduces RadixAttention, which keeps cached keys and values in a tree so requests that start with the same prompt prefix reuse them. It is a worked example of the prefix caching that lets a long system prompt be prefilled once.",
    },
    {
      authors: "Amey Agrawal, Nitin Kedia, Ashish Panwar, et al.",
      title: "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve",
      source: "Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2403.02310",
      note: "Explains why prefill saturates compute while decode uses it poorly, then splits long prompts into chunks that run alongside other users' decode steps. This is the chunked prefill the lesson says keeps one long prompt from stalling other streams.",
    },
    {
      authors: "Guangxuan Xiao, Yuandong Tian, Beidi Chen, et al.",
      title: "Efficient Streaming Language Models with Attention Sinks",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2309.17453",
      note: "Shows that keeping only recent keys and values fails once the first tokens are dropped, because models park strong attention on them as attention sinks. It backs the warning in Where it breaks that evicting old cache entries is riskier than it looks.",
    },
  ],
  checkpoint: [
    {
      prompt: "With Generated tokens at 0 the whole lower triangle is computed in one pass, but each later step lights one row. Why can prefill do the whole prompt at once?",
      options: [
        "Prefill drops the causal mask, so each prompt position can read the whole prompt at once",
        "Every prompt token is already known, so all positions can go through each layer together",
        "The cache already holds the prompt's results, so prefill only looks them up",
      ],
      answer: 1,
      explanation: "Prefill and decode use the same model and the same causal mask. The difference is availability: all prompt tokens exist up front, so each layer does one large matrix multiply over every position, while a decode step must wait for the token the previous step produced. The cache is filled by prefill, not used to skip it.",
      objective: 0,
    },
    {
      prompt: "A user sends a very long prompt. Does turning the KV cache on shorten the wait for the first token?",
      options: [
        "Yes, because the cache stores the whole prompt's results, so the first token needs almost no computation",
        "Yes, because cached keys let prefill process the prompt one token at a time instead of all at once",
        "No: prefill still processes every prompt token, and the cache saves work only on the later decode steps",
      ],
      answer: 2,
      explanation: "Time to first token is dominated by prefill, which has to compute every prompt position's keys and values and fill the cache. Its attention cost also grows with the square of prompt length. The cache pays off afterwards: each decode step reads stored keys and values instead of recomputing the prefix.",
      objective: 0,
    },
    {
      prompt: "With the KV cache on, what does each decode step still have to read from memory?",
      options: [
        "Every model weight plus every stored key and value for that sequence",
        "Nothing but the newest token, because everything else is cached",
        "Only the keys and values, because the weights stay in the arithmetic units",
      ],
      answer: 0,
      explanation: "The cache removes recomputation of past keys and values, not memory traffic. Each step multiplies the new position through every weight matrix and attends over every cached entry, so it reads both. That is why decode is bandwidth bound and why batching helps: one weight read serves many sequences.",
      objective: 1,
    },
    {
      prompt: "You switch KV cache from Use cache to Recompute during a decode step. What changes in the work for that step?",
      options: [
        "The attention output changes, because recomputed keys differ slightly from the ones that were stored in the cache",
        "Nothing changes in the work, because the causal mask already limits every step to a single row of scores",
        "Every earlier key and value is projected and scored again, so the pass grows from one row to the full triangle",
      ],
      answer: 2,
      explanation: "Keys and values depend only on earlier tokens, so recomputing them gives exactly the numbers the cache would have stored and the output is identical. What changes is cost: each step reruns the whole prefix, so per-token work grows with the square of length, which Q·K this pass and Q·K whole run both show.",
      objective: 1,
    },
    {
      prompt: "On Memory budget you double Concurrent sequences. What happens to KV per token and to the KV cache total?",
      options: [
        "KV per token doubles and the total quadruples, because sequences also store each other's keys",
        "KV per token stays the same and the total doubles, since each sequence keeps its own cache of keys and values",
        "KV per token stays the same and the total stays the same, because the cache is shared across sequences",
      ],
      answer: 1,
      explanation: "Bytes per token depend only on the model's shape: 2 times layers times key/value heads times head dimension times bytes per value. Each sequence keeps a private cache of its own tokens, so the total scales with sequences times context. Sequences that fit is unchanged, and it is the limit batching runs into.",
      objective: 1,
    },
    {
      prompt: "In Prefill versus decode you raise Concurrent sequences. Why does total decode throughput rise while each sequence's own tokens per second fall?",
      options: [
        "One read of the weights serves every sequence, but each sequence still reads its own cache",
        "Larger batches make the arithmetic units faster, so every sequence's step finishes sooner",
        "The sequences share one cache, so each step reads less memory per sequence as the batch grows",
      ],
      answer: 0,
      explanation: "Decode is memory-bound: the step time is set by bytes read. Weights are read once however many sequences run, so tokens per second in total rise. But every sequence adds its own cache to the bytes, so the step takes longer and each sequence's rate falls. Cache memory is what finally caps the batch.",
      objective: 1,
    },
    {
      prompt:
        "On Speculative decoding you set Draft quality to 1.00 and Draft length to 4. How many tokens does each target pass yield?",
      options: [
        "Five, because all four proposals are kept and the target adds one more token of its own",
        "Four, because once every proposal is kept the target has nothing left to add to the text",
        "One, because the target must still generate each token itself before any is shown to a reader",
        "Two, because a perfect draft is still checked one proposal at a time and kept in pairs",
      ],
      answer: 0,
      explanation:
        "At quality 1.00 the draft is the target's own distribution, so every proposal passes the accept test. A pass that keeps all four proposals still ends with the token the target samples at the position after them, so it yields draft length plus one. That is the ceiling: a pass can never yield more.",
      objective: 2,
    },
    {
      prompt:
        "At a middling Draft quality, raising Draft length from 1 to 4 adds far more tokens per pass than raising it from 5 to 8. Why?",
      options: [
        "A later proposal counts only if every earlier one was kept, so each extra token is reached less often",
        "The target model verifies long drafts more slowly, so every pass with more proposals costs more time to run",
        "The draft gets worse the further it writes, so every proposal after the fourth one is refused outright",
        "The simulation runs out of passes at long drafts, so the extra tokens are simply lost in sampling noise",
      ],
      answer: 0,
      explanation:
        "The expected tokens per pass are 1 plus the chance that the first proposal is kept, plus the chance that the first two are, and so on. Each term is smaller than the last by roughly a factor of the acceptance rate, so the curve flattens toward 1 ÷ (1 − α). The lab counts passes, not time, and every point is a full run.",
      objective: 2,
    },
    {
      prompt:
        "Speculative decoding keeps some tokens that a weak draft proposed. How does the text it writes compare with the target model's own?",
      options: [
        "The same distribution, because a refused proposal is replaced from the leftover target probability",
        "It is a little worse, because the draft's mistakes sometimes get through the accept test unchanged",
        "It is a little better, because two models now effectively vote on every token that is written",
        "It matches only at Draft quality 1.00, because any other draft changes what gets written to the text",
      ],
      answer: 0,
      explanation:
        "The accept test and the replacement sample are built so that each written token is distributed exactly as the target alone would have written it, whatever the draft. A weaker draft is refused more often and so yields fewer tokens per pass, but the text's distribution does not move. The lab checks this by counting character pairs in a long simulated text.",
      objective: 2,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      // Version 2 added the speculative-decoding controls; an older payload gets their defaults.
      return {
        ...initialState,
        ...parsed,
        context: intIn(parsed.context, PROMPT_RANGE.min, PROMPT_RANGE.max, 7),
        generated: intIn(parsed.generated, GENERATED_RANGE.min, GENERATED_RANGE.max, 4),
        useCache: typeof parsed.useCache === "boolean" ? parsed.useCache : true,
        realContext: intIn(parsed.realContext, CONTEXT_RANGE.min, CONTEXT_RANGE.max, 8192),
        sequences: intIn(parsed.sequences, SEQUENCE_RANGE.min, SEQUENCE_RANGE.max, 16),
        specGamma: intIn(parsed.specGamma, GAMMA_RANGE.min, GAMMA_RANGE.max, 4),
        specQuality: typeof parsed.specQuality === "number" ? clampQuality(parsed.specQuality) : 0.7,
        specSeed: intIn(parsed.specSeed, SEED_RANGE.min, SEED_RANGE.max, 7),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
