import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import {
  BANDWIDTH_MAX,
  BANDWIDTH_MIN,
  BATCH_MAX,
  BATCH_MIN,
  PRECISIONS,
  WIDTH_MAX,
  WIDTH_MIN,
} from "./roofline";

const OPERATIONS = ["matmul", "add"] as const;

const initialState: ModuleState = {
  operation: "matmul",
  batch: 1,
  layerWidth: 4096,
  precision: "fp16",
  bandwidth: 2000,
};

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const clampNumber = (value: unknown, fallback: number, low: number, high: number) =>
  Math.min(high, Math.max(low, finite(value) ? value : fallback));
const powerOfTwo = (value: unknown, fallback: number, low: number, high: number) =>
  2 ** Math.round(Math.log2(clampNumber(value, fallback, low, high)));

/**
 * Version 1 stored { matrixSize, precision, bandwidth } for an n×n by n×n multiply.
 * That is the same operation as a batch of n rows through an n-wide layer, so it
 * translates to { batch: n, layerWidth: n } without changing the arithmetic.
 */
export function migrateWhyGpusState(parsed: ModuleState): ModuleState {
  if (!("matrixSize" in parsed) || "layerWidth" in parsed || "batch" in parsed) return parsed;
  const { matrixSize, ...rest } = parsed;
  return { ...rest, operation: "matmul", batch: matrixSize, layerWidth: matrixSize };
}

const definition: ModuleDefinition = {
  id: "module-34-why-gpus",
  slug: "why-gpus",
  title: "Why GPUs",
  group: "foundations",
  order: 13,
  icon: "Cpu",
  accent: "#c43d5a",
  prerequisites: ["module-03-stacking-neurons"],
  estimatedMinutes: 12,
  steps: ["Grow the batch", "Change precision", "Starve the bandwidth", "Try an elementwise op"],
  stepInstructions: [
    "With Operation on Matmul X·W, move Batch rows from 1 to 1024. Watch the dot climb the slanted roof, cross the ridge near 162 rows, and Bound flip from memory to compute.",
    "Set Batch rows back to 1, then switch Precision from fp16 to fp32 and to int8. Compare Intensity and the Memory traffic time for each.",
    "Set Batch rows to 1024, then drag Memory bandwidth down to 100 GB/s. Watch the ridge slide right past the dot, and read the 2× memory bandwidth row in Where the time goes.",
    "Switch Operation to Elementwise add and move Batch rows to 4096. Intensity stays at 0.167 FLOP/B, so Bound stays memory at every batch.",
  ],
  stateVersion: 2,
  tagline:
    "See a layer as matrix multiplies a GPU can run in parallel, and use a roofline to find when memory movement, not arithmetic, sets the time.",
  objectives: [
    "Explain why a GPU helps a neural net without claiming that the chip understands the task",
    "Separate arithmetic throughput from memory bandwidth as two different limits",
    "Predict from arithmetic intensity whether an operation is memory-bound or compute-bound",
  ],
  glossary: [
    {
      term: "Matrix multiply",
      definition:
        "The dominant arithmetic in a dense layer: each output is a dot product of one input row with one weight column, so B rows through a d-by-d layer cost 2·B·d² operations.",
    },
    {
      term: "FLOP",
      definition:
        "One floating-point operation; a multiply-add counts as two. A FLOP count measures arithmetic work and ignores the memory traffic that often decides wall time.",
    },
    {
      term: "Throughput",
      definition:
        "How many operations a chip can finish per second at best, its peak. The lab's illustrative device peaks at 20, 300 and 600 trillion per second for fp32, fp16 and int8.",
    },
    {
      term: "Memory bandwidth",
      definition:
        "How many bytes per second can move between the GPU's main memory and its compute units. It caps speed whenever each byte feeds only a few operations.",
    },
    {
      term: "Arithmetic intensity",
      definition:
        "Operations performed per byte moved. A batch-1 fp16 matmul does about one FLOP per byte; an fp16 elementwise add does one sixth.",
    },
    {
      term: "Roofline",
      definition:
        "A plot of attainable speed against arithmetic intensity, min(peak, bandwidth times intensity). The slanted part is the memory limit and the flat part is the compute limit.",
    },
    {
      term: "Ridge point",
      definition:
        "The intensity where the two roofs meet, peak divided by bandwidth. Work below it is memory-bound and work above it is compute-bound.",
    },
    {
      term: "Memory-bound",
      definition:
        "Limited by bytes, not arithmetic. Only more bandwidth, fewer bytes, or more reuse per byte shortens the time; extra peak FLOP/s changes nothing.",
    },
    {
      term: "Compute-bound",
      definition:
        "Limited by arithmetic: data arrive faster than the multiply units can use them, so only more peak throughput or fewer operations help.",
    },
    {
      term: "Batch",
      definition:
        "The rows processed together against the same weights, such as examples in a training step or tokens in a prompt. Each weight fetched from memory is reused once per row.",
    },
    {
      term: "Parallelism",
      definition:
        "Doing many independent pieces of work at once. Every output of a matmul is an independent dot product, the shape a GPU's thousands of lanes need; a long serial dependency is not.",
    },
    {
      term: "Kernel",
      definition:
        "A GPU program that runs one operation, such as a matmul or an add, over a large grid of threads. Each launch costs a few microseconds, which this lab ignores.",
    },
    {
      term: "Precision",
      definition:
        "The number format and its bytes per value: fp32 is 4, fp16 is 2, int8 is 1. Fewer bytes cut traffic, and matrix units run narrower formats at higher peak rates, at some cost in accuracy.",
    },
  ],
  references: [
    {
      authors: "Rajat Raina, Anand Madhavan, and Andrew Y. Ng",
      title: "Large-scale Deep Unsupervised Learning using Graphics Processors",
      source: "Proceedings of the 26th International Conference on Machine Learning (ICML 2009), 873–880",
      year: 2009,
      url: "https://icml.cc/Conferences/2009/papers/218.pdf",
      note: "An early paper that moved deep network training onto graphics chips and ran it many times faster than on a CPU. It also notes that the main slowdown is often moving data into GPU memory, not the arithmetic, which is the two-speeds idea in this lesson.",
    },
    {
      authors: "Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton",
      title: "ImageNet Classification with Deep Convolutional Neural Networks",
      source: "Advances in Neural Information Processing Systems 25 (NeurIPS 2012)",
      year: 2012,
      url: "https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
      note: "The image network that showed a large neural net could be trained in days on two GPUs, using fast GPU code for its convolutions. It is a concrete case of the lesson's claim that neural nets became practical at scale on GPUs.",
    },
    {
      authors: "Sara Hooker",
      title: "The Hardware Lottery",
      source: "Communications of the ACM 64(12), 58–65",
      year: 2021,
      url: "https://arxiv.org/abs/2009.06489",
      note: "An essay on how ideas succeed when they fit the hardware of the day. It tells how GPUs, built for video games and graphics, turned out to suit the simple, splittable matrix multiplies of neural nets, which is why a GPU helps without understanding the task.",
    },
    {
      authors: "NVIDIA",
      title: "CUDA Programming Guide",
      source: "NVIDIA CUDA documentation",
      year: 2026,
      url: "https://docs.nvidia.com/cuda/cuda-programming-guide/",
      note: "The official guide to programming NVIDIA GPUs. Its programming model section defines a kernel as a function launched on the GPU across many threads, grouped into blocks and a grid, which backs the glossary's Kernel and Parallelism entries.",
    },
    {
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.",
      title: "Attention Is All You Need",
      source: "Advances in Neural Information Processing Systems 30 (NeurIPS 2017)",
      year: 2017,
      url: "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
      note: "The Transformer paper. Its introduction explains that a recurrent network must work through a sequence one step at a time, which blocks parallel work, the serial-chain point this lesson makes before The architecture zoo.",
    },
    {
      authors: "NVIDIA",
      title: "Matrix Multiplication Background User's Guide",
      source: "NVIDIA Deep Learning Performance documentation",
      year: 2026,
      url: "https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html",
      note: "Counts the work of a matrix multiply as 2 operations per multiply-add and divides it by the bytes read and written to get arithmetic intensity, the same sums as the Work versus traffic row. It shows that small or matrix-vector shapes are memory-limited while large ones are math-limited.",
    },
    {
      authors: "Samuel Williams, Andrew Waterman, and David Patterson",
      title: "Roofline: An Insightful Visual Performance Model for Multicore Architectures",
      source: "Communications of the ACM 52(4), 65–76",
      year: 2009,
      url: "https://www.semanticscholar.org/paper/092217c2267f6e0673590aa151d811e579ff7760",
      note: "The paper that introduced the roofline plot. It defines attainable speed as the smaller of peak arithmetic and memory bandwidth times operations per byte, and names the ridge point where the slanted and flat roofs meet, as on the Roofline card.",
    },
    {
      authors: "NVIDIA",
      title: "GPU Performance Background User's Guide",
      source: "NVIDIA Deep Learning Performance documentation",
      year: 2026,
      url: "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html",
      note: "Splits run time into a math time and a memory time and takes the larger, like the two clocks on the Where the time goes card. Its table shows a batch-1 linear layer at about 1 FLOP per byte and memory-limited, and it says elementwise adds are memory-limited too.",
    },
    {
      authors: "NVIDIA",
      title: "NVIDIA A100 Tensor Core GPU Architecture",
      source: "NVIDIA whitepaper",
      year: 2020,
      url: "https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf",
      note: "The design document for the A100, the chip the lab's made-up device is modelled on. Its spec table lists 19.5 TFLOP/s for fp32, 312 for fp16 on Tensor Cores, and 624 TOPS for int8, close to the lab's round 20, 300 and 600.",
    },
    {
      authors: "Paulius Micikevicius, Sharan Narang, Jonah Alben, et al.",
      title: "Mixed Precision Training",
      source: "International Conference on Learning Representations (ICLR 2018)",
      year: 2018,
      url: "https://arxiv.org/abs/1710.03740",
      note: "Shows how to train in fp16 without losing accuracy. It explains that half precision halves the memory and bandwidth needed, and that GPUs do half-precision math several times faster than fp32, using Tensor Cores for matrix multiplies, the trade the Change precision step explores.",
    },
    {
      authors: "Benoit Jacob, Skirmantas Kligys, Bo Chen, et al.",
      title: "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference",
      source: "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018)",
      year: 2018,
      url: "https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html",
      note: "Runs a network with 8-bit integer weights and inputs. It shows why the running totals need a 32-bit integer and are then scaled back down, the int8 detail the lab simplifies in Where it breaks.",
    },
    {
      authors: "Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, et al.",
      title: "Data Movement Is All You Need: A Case Study on Optimizing Transformers",
      source: "Proceedings of Machine Learning and Systems 3 (MLSys 2021)",
      year: 2021,
      url: "https://proceedings.mlsys.org/paper_files/paper/2021/hash/bc86e95606a6392f51f95a8de106728d-Abstract.html",
      note: "Finds that in BERT training, matrix multiplies do over 99% of the arithmetic but only about 61% of the run time, because elementwise and normalization steps are memory-bound. Fusing those steps cuts the data moved, as the Try an elementwise op step suggests.",
    },
    {
      authors: "Tri Dao, Daniel Y. Fu, Stefano Ermon, et al.",
      title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html",
      note: "The FlashAttention paper named in Going deeper. It speeds up attention by cutting reads and writes to GPU main memory, and runs faster even though it does more FLOPs; its background section also explains memory-bound and compute-bound work and kernel fusion.",
    },
    {
      authors: "Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, et al.",
      title: "Efficiently Scaling Transformer Inference",
      source: "Proceedings of Machine Learning and Systems 5 (MLSys 2023)",
      year: 2023,
      url: "https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html",
      note: "A study of serving very large language models. It finds that at small batch sizes the time to load the weights dominates each generated token, and it uses int8 weights to cut that time, the serving point in Where it breaks.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "Parallel outputs reads 4,096 dots of 4096 at Batch rows 1. What makes this the kind of work a GPU's thousands of lanes suit?",
      options: [
        "The chip works out which weights matter for the task, so it multiplies only the useful ones and skips the rest",
        "The dot products must run one after another, and a GPU runs that single sequence far faster",
        "Dot products need almost no data, so thousands of lanes never have to wait on memory",
        "Each output is its own dot product that needs no other output's result, so lanes run side by side",
      ],
      answer: 3,
      explanation:
        "Each of the B·d outputs of Y = X·W is an independent dot product: regular work with no dependencies, which is the shape thousands of arithmetic lanes need. The chip does not know which weights matter or what the task is; the data and the loss carry the meaning and the GPU multiplies. Parallel work is necessary for speed but not sufficient, because the lanes still need their numbers delivered.",
      objective: 0,
    },
    {
      prompt:
        "At Batch rows 1 the lab's fp16 matmul reaches under 1% of peak arithmetic. Which change raises the useful work done per second?",
      options: [
        "A chip with twice the peak FLOP/s and the same memory bandwidth, so the arithmetic finishes in half the time",
        "Reusing each fetched weight for more rows, or storing the weights in fewer bytes",
        "Nothing: a GPU always runs a matrix multiply at its peak rate, whatever the batch size",
      ],
      answer: 1,
      explanation:
        "The operation is memory-bound: each weight byte crosses the bus to feed about one FLOP. More rows per fetch raise intensity (about 2·B / bytes) and fewer bytes cut traffic; both move the dot up the slanted roof. Extra peak only raises the flat roof the dot never reaches.",
      objective: 1,
    },
    {
      prompt:
        "At Batch rows 1024 the fp16 matmul reads compute-bound. You drag Memory bandwidth down to 100 GB/s. What does Bound read now, and why?",
      options: [
        "Still compute: bandwidth only matters at Batch rows 1, where each weight is used once",
        "Memory, because the 1024 rows of data no longer fit in the slower memory at this bandwidth",
        "Memory: the ridge moves right past the dot, so the work now sits on the slanted bandwidth roof",
        "Still compute: the operation's arithmetic is unchanged, so the same limit still applies",
      ],
      answer: 2,
      explanation:
        "The ridge is peak divided by bandwidth, so cutting bandwidth pushes it to higher intensity. The matmul's intensity did not change, but it now falls left of the ridge, where bytes rather than arithmetic set the time. Bandwidth is a rate, not a capacity, so the rows do not stop fitting; they simply arrive more slowly than the multipliers could use them.",
      objective: 2,
    },
    {
      prompt:
        "Switch Operation to Elementwise add and raise Batch rows to 4096. Bound stays memory at every batch. Why do extra rows not help here, as they do for the matmul?",
      options: [
        "An add has fewer total operations than a matmul, so it is always too small to reach the ridge",
        "Elementwise adds run on a different unit that has no peak arithmetic rate, so no ridge can be measured against it",
        "Each value is read, added once, and written, so extra rows bring bytes in proportion and nothing is reused",
      ],
      answer: 2,
      explanation:
        "A matmul reads each weight once and uses it for every row, so operations grow with the batch much faster than bytes. An elementwise add does one operation per three memory touches at any batch, so its intensity stays at 1 / (3·bytes), about 0.167 FLOP/B at fp16, far left of the ridge. Real frameworks fuse such operations into the matmul before them to skip the traffic.",
      objective: 2,
    },
    {
      prompt:
        "The Roofline gives the dot's Attainable speed. How should you read that number against a real GPU running a real kernel?",
      options: [
        "As a ceiling: real kernels land at or below it, because the lab assumes perfect overlap of work and traffic",
        "As the measured speed of a real kernel, because the lab times it on a real chip and reports the result",
        "As a floor: real GPUs can only be faster, because caches and tiling add speed the lab leaves out",
        "As exact whenever Bound reads compute, because the memory system stops mattering there",
      ],
      answer: 0,
      explanation:
        "Every readout is closed-form arithmetic on shapes: the peaks are round illustrative numbers, arithmetic and traffic overlap perfectly, each operand moves exactly once, and launch costs, caches, and tile shapes are ignored. The roofline is therefore the best any kernel could do on that device. Measured kernels land below it, even compute-bound ones.",
      objective: 1,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const raw = JSON.parse(value) as ModuleState;
      if (!raw || typeof raw !== "object" || Array.isArray(raw)) return { ...initialState };
      const parsed = migrateWhyGpusState(raw);
      const precision =
        typeof parsed.precision === "string" && PRECISIONS.some((item) => item.id === parsed.precision)
          ? parsed.precision
          : initialState.precision;
      const operation =
        typeof parsed.operation === "string" && (OPERATIONS as readonly string[]).includes(parsed.operation)
          ? parsed.operation
          : initialState.operation;
      return {
        operation,
        precision,
        batch: powerOfTwo(parsed.batch, 1, BATCH_MIN, BATCH_MAX),
        layerWidth: powerOfTwo(parsed.layerWidth, 4096, WIDTH_MIN, WIDTH_MAX),
        bandwidth: Math.round(clampNumber(parsed.bandwidth, 2000, BANDWIDTH_MIN, BANDWIDTH_MAX)),
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
