import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Workload knobs": {
    title: "Shape, number format, and memory speed",
    summary:
      "One layer operation is counted from its shapes: how many arithmetic operations it needs and how many bytes must cross the memory bus. Operation, Batch rows and Layer width set the shapes, Precision sets bytes per value and the peak rate, and Memory bandwidth sets how fast bytes arrive.",
    whatYouSee: [
      "A kicker with the tensor shapes, the precision, and the bandwidth.",
      "Operation (Matmul X·W or Elementwise add), Precision (fp32, fp16, int8: 4, 2 and 1 bytes per value), Batch rows from 1 to 4096 in doublings, Layer width d from 512 to 16384 in doublings, and Memory bandwidth from 100 to 5000 GB/s.",
      "A formula row with the operation and byte counts at the current values, and a note naming the illustrative device.",
    ],
    howItWorks: [
      "Matmul Y = X·W with X of B×d and W of d×d: ops = 2·B·d², bytes = (d² + 2·B·d) × bytes per value, which reads W and X once and writes Y once.",
      "Elementwise add Y = X + R: ops = B·d, bytes = 3·B·d × bytes per value (two reads, one write).",
      "Peaks are round illustrative numbers of the same order as an A100-class GPU: fp32 20 TFLOP/s on ordinary lanes, fp16 300 TFLOP/s and int8 600 TOP/s on matrix units. The A100 datasheet lists 19.5, 312 and 624 (dense).",
    ],
    controls: [
      "Operation, Precision, Batch rows, Layer width, and Memory bandwidth.",
      "Comparison worth running: Batch rows 1 against 1024 at d = 4096 and fp16. Operations grow 1024×, bytes grow only 1.5×.",
    ],
    notice: [
      "For a matmul the weights dominate the bytes at small batch, so adding rows is nearly free traffic. That reuse is the whole story of intensity.",
      "The add has no reuse at all: every value is touched three times for one operation.",
    ],
    limits: [
      "In this lab: closed-form counts with every operand stored at the chosen precision and moved exactly once. Real int8 matmuls accumulate in int32 and rescale, and real kernels move extra bytes when tiles do not fit on chip.",
      "In general: a real layer also has attention, normalization and activation kernels, launch overheads of a few microseconds each, and communication between GPUs. The roofline is a first estimate, not a profiler.",
    ],
  },

  "Roofline": {
    title: "Attainable speed against arithmetic intensity",
    summary:
      "The roofline says the fastest an operation can run is min(peak, bandwidth × intensity). Left of the ridge point the slanted memory roof is lower; right of it the flat compute roof is lower. The dot is the current operation, placed by its computed intensity.",
    whatYouSee: [
      "Log–log axes: arithmetic intensity in operations per byte from 0.01 to 10k, and attainable TFLOP/s (TOP/s for int8) from 0.001 to 1000.",
      "The roof as one bent line, a dashed ridge line with its value, and a tinted memory-bound zone to its left.",
      "A filled dot for the current operation, hollow circles for the same matmul at 1 to 4096 rows (every other one labelled), and a diamond for the other operation at the same shape.",
      "Metrics Intensity, Ridge, Attainable, and Bound.",
    ],
    howItWorks: [
      "Intensity = ops / bytes from the counts on the knobs card. Ridge = peak / bandwidth.",
      "Attainable = min(peak, bandwidth × intensity). Bound is memory when bytes / bandwidth exceeds ops / peak.",
      "Every point sits on the roof by construction, because the model assumes arithmetic and traffic overlap perfectly.",
    ],
    controls: [
      "This card has no controls. The knobs above move the dot, the ridge, and both roofs.",
      "Comparison worth running: walk Batch rows from 1 to 4096 and watch the dot slide along the hollow circles, then change Precision and see the whole trail shift.",
    ],
    notice: [
      "On the slanted roof, doubling intensity doubles speed; on the flat roof it does nothing.",
      "Lower bandwidth drags the slanted roof down and pushes the ridge right, so work that was compute-bound can become memory-bound.",
      "fp32 has a ridge of only 10 FLOP/B here because its peak is low, so even 32 rows are compute-bound in fp32.",
    ],
    limits: [
      "In this lab: two ideal roofs and nothing else. There is no cache hierarchy, no occupancy or tile-shape effect, and no launch cost, so real kernels land below the roof.",
      "In general: measured kernels add more roofs (on-chip cache bandwidth, special-function units) and land below the ideal one, often far below for small or oddly shaped matrices. The roofline tells you which resource to buy or save, not the exact time.",
    ],
  },

  "Where the time goes": {
    title: "Two clocks, and what would move them",
    summary:
      "Arithmetic time is ops divided by peak; memory time is bytes divided by bandwidth. The roofline wall-clock is the larger of the two. The table re-runs the same arithmetic after one change so you can see which change helps.",
    whatYouSee: [
      "Two bars, Arithmetic at peak and Memory traffic, with their times and the division that produced them. The one that sets the time is labelled so.",
      "A table of wall-clock after doubling peak, doubling bandwidth, or dropping to the next smaller precision, each with its speed-up.",
      "Metrics Parallel outputs and Rows to reach ridge, plus a note that changes with the bound.",
    ],
    howItWorks: [
      "Wall-clock = max(ops / peak, bytes / bandwidth). Utilization in the kicker = (ops / wall-clock) / peak.",
      "Rows to reach ridge solves 2·B·d² / ((d² + 2·B·d)·bytes) = ridge for B, which gives B = ridge·bytes·d / (2(d − ridge·bytes)). It reads never when d / bytes is below the ridge.",
      "Parallel outputs = B·d, each an independent dot product of length d (or one add).",
    ],
    controls: [
      "This card has no controls. Use the knobs above.",
      "Comparison worth running: at Batch rows 1, compare the 2× peak arithmetic row with the 2× memory bandwidth row. Then repeat at 1024 rows.",
    ],
    notice: [
      "At Batch rows 1 there are still 4,096 independent dot products, plenty of parallel work, yet utilization is under 1%. Parallelism is necessary but not sufficient.",
      "The fp16 → int8 row halves memory time and doubles peak, so it helps in both regimes on this device.",
    ],
    limits: [
      "In this lab: perfect overlap of arithmetic and traffic, and peak rates that any shape can reach. Small matrices cannot actually fill a GPU, and quantized formats cost accuracy the table does not show.",
      "In general: generating text one token at a time runs every weight matrix at a batch of one per sequence, which is why serving is usually memory-bound and why batching many users together and quantizing weights both raise throughput.",
    ],
  },
};

export default cardInfo;
