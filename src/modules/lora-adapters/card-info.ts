import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "The low-rank update": {
    title: "One frozen table plus two thin trainable ones",
    summary:
      "LoRA drawn from the trained numbers: B (30 × r) times A (r × 30), scaled by α / r, is the 30 × 30 update added to the frozen table W. Rank and Alpha sit beside the picture, so you can watch B widen, A deepen, and the update's number of directions follow r.",
    whatYouSee: [
      "A kicker `rank r · alpha α · scale α/r` and a badge `N of 900 weights train`.",
      "`Rank` (1 to 12, printed with r · 60 weights), `Alpha` (1 to 24, printed with the scale), a warning whenever alpha ≠ rank with a button to set them equal, and a bar list comparing `Full fine-tuning` (d · k = 900) with `LoRA rank r` (r · (d + k)), each with its target loss.",
      "A matrix-product diagram: A across the top, B down the left, and ΔW = (α/r)·B·A where they meet. Blue cells raise a logit, red lower it, and each matrix is shaded against its own largest entry.",
      "A chart of how much of each update's squared size its first k directions hold, for the LoRA ΔW (solid) and a full fine-tune's ΔW at the same epochs (dashed), with a guide at k = r. The full fine-tune's ΔW is drawn beside it.",
      "A formula row `W + (α / r) · B · A` and a detail line on how many of the full run's 900 weights actually move.",
    ],
    howItWorks: [
      "The adapter is `trainTinyLora` on Recipe steps: B starts at zero, A at small random values, batch 16, learning rate 0.6, seed 5, from a base trained 60 epochs on Harbor weather notes.",
      "Both ΔW maps have each row's mean subtracted. A softmax row ignores a constant added to all of its logits, so that part changes no prediction; subtracting it cannot raise the rank.",
      "The chart uses singular values from a Jacobi eigendecomposition of ΔWᵀΔW. The share at k is (σ₁² + … + σₖ²) / Σσ², the best any rank-k matrix can keep of that update.",
      "Trainable count is `r · (d + k)` with d = k = 30, so 60r: 240 at rank 4, 720 at rank 12. At rank 15 it would equal the full table.",
    ],
    controls: [
      "`Rank` resizes B and A and retrains. `Alpha` changes the scale and retrains. `Adapter epochs`, on the training card, retrains too.",
      "Comparison worth running: at 24 epochs, move Rank from 1 to 12. The solid curve always reaches 100% at k = r, while the dashed one needs 11 directions to pass 95%.",
    ],
    notice: [
      "B is blank at 0 epochs, so ΔW is blank too, whatever A holds. That is zero initialization in one picture.",
      "The full fine-tune's update is spread out: its first 4 directions hold 63%, its first 8 hold 86%. The rank-4 adapter still matches its recipe loss, 2.077 against 2.090.",
      "The scale α / r is printed in the kicker. Raising rank at a fixed alpha shrinks every update, which is why the warning appears.",
    ],
    limits: [
      "In this lab: W, A and B are the whole model, one 30 × 30 table. Real LoRA attaches to chosen projection matrices inside attention and the MLP, usually thousands of rows wide. The Adapter size calculator below does the same arithmetic for real architectures.",
      "In this lab: the full-fine-tune bar and map are one run at one seed, there to give the parameter count and the spectrum a reference, not to settle which method wins.",
      "In general: LoRA bets that the update a task needs is close to low rank. When it is not, as for large shifts in language or knowledge, more rank or full fine-tuning wins, and only held-out evaluation tells you which case you are in.",
    ],
  },

  "Adapter training and behaviour": {
    title: "Only B and A receive gradients",
    summary:
      "How long the adapter trains, its loss while training against the frozen base's recipe loss, and what it does to the next-character distribution after a chosen character.",
    whatYouSee: [
      "`Adapter epochs`, from `untrained` at 0 to 48 passes.",
      "A chart of optimizer step against batch loss on recipes. Solid `LoRA rank r, per-batch`, dotted `frozen base, full corpus` at the base model's recipe cross-entropy, and a footnote with the adapted model's full-corpus recipe loss.",
      "`Next character after`, choosing the context `␣`, `e`, `s`, `l` or `a`.",
      "The seven most likely next characters after that context, each with a hatched bar for the base model above a solid bar for base plus adapter, and both probabilities printed.",
    ],
    howItWorks: [
      "The solid curve averages minibatch cross-entropy between history points while A and B train. At 0 epochs the trainer has no history, so the card draws a flat segment at the base loss.",
      "The dotted line is `tinyCrossEntropy(base, recipes)`, recomputed rather than trained.",
      "Probe bars are the softmax of one row: the frozen table's row, then the same row of `W + (α/r)·B·A`.",
    ],
    controls: [
      "`Adapter epochs` retrains the adapter and the full-fine-tune comparison. `Next character after` only changes which row is probed.",
      "Comparison worth running: at 24 epochs pick `s`. The base puts 5% on `a`; the adapter puts 29%, because Recipe steps says `sauce`, `salt` and `salted`.",
    ],
    notice: [
      "At 0 epochs every pair of bars matches exactly: an untrained adapter is a no-op.",
      "The space row barely moves (`t` 22% → 18%). Of the five probe rows, `l` moves most, then `s` and `a`; `e` and the space row move least.",
      "The solid training curve is noisy because it is batch loss. The footnote's full-corpus number is the one to compare with the dotted line.",
    ],
    limits: [
      "In this lab: the loss is training loss on eight recipe sentences, and the probe shows one row of a bigram model. There is no held-out recipe set.",
      "In this lab: five probe contexts are offered, chosen because recipes change them visibly; the other rows are trained too.",
      "In general: adapter training loss can look excellent while the adapter fails on prompts unlike its training set. The serving card's source loss is this page's cheap check for side effects.",
    ],
  },

  "Hot-swap and merge": {
    title: "Attach it, detach it, or fold it in",
    summary:
      "The serving choice LoRA exists to give you: run the base, run the base plus adapter, or add (α/r)·B·A into W once. The merge is checked against the unmerged path, logit by logit.",
    whatYouSee: [
      "A `Serving` control: `Base only` and `Base + adapter`, or a single option `Merged — nothing left to swap` after you merge.",
      "Three metrics: `Target loss` and `Source loss` of whatever is being served, and `Serving weights` as 900, 900 + 60r, or 900 once merged.",
      "A sample from the served table: prompt `the `, temperature 0.7, seed 12, 76 characters.",
      "A button `Merge the adapter into W`, and a callout that reports the largest logit gap between the merged table and the unmerged path, plus the recipe loss through the unmerged path.",
    ],
    howItWorks: [
      "The unmerged path computes `W[x] + (α/r)·B[x]·A` in float64 for every context row x, never forming B·A. The merged table is `W + ΔW` stored in float32.",
      "The gap is float32 rounding, about 4e-7 at rank 4, and the recipe loss agrees to four decimals. At 0 epochs the gap is exactly 0.",
      "`Source loss` turns the loss tone when the adapter raises harbor cross-entropy by more than 0.05 nats, a display threshold rather than a standard.",
    ],
    controls: [
      "`Serving` attaches or detaches the adapter. It does nothing once merged.",
      "`Merge the adapter into W` holds until you change Rank, Alpha or Adapter epochs, which clears the merge.",
      "Comparison worth running: with a trained adapter, flip Serving and watch Target loss, Source loss and the sample move together. Then merge and confirm Target loss stays at the adapted value while Serving weights drops to 900.",
    ],
    notice: [
      "Detaching restores the base exactly: Source loss returns to 1.921. The base array was never written.",
      "Serving weights reads 1140 at rank 4 unmerged. That is the memory you pay to keep the adapter swappable.",
      "Merging an untrained adapter changes nothing: W + 0 is W, and the gap reads 0.",
    ],
    limits: [
      "In this lab: merge is a flag plus a real numerical check, not a file write, and there is no second adapter to swap in. `Detach` is the only swap.",
      "In this lab: while attached, the sample and losses are read from the merged numbers, so you do not pay the extra multiply as you click; the serving-weights metric still counts A and B.",
      "In general: servers keep many adapters in memory and apply one per request with batched kernels, or merge one default adapter for a dedicated endpoint. The trade is extra parameters and a matmul against a one-time add you cannot undo.",
    ],
  },

  "Adapter size calculator": {
    title: "What the same idea costs on a real model",
    summary:
      "A closed-form calculator for LoRA's size. Pick an architecture, the matrices to adapt, a rank, and a precision, and it reports the trainable parameters, their share of the base model, and the adapter's size in megabytes. Nothing on it is trained: it is arithmetic on published shapes, set beside the toy's 27%.",
    whatYouSee: [
      "A kicker naming the architecture with its model dimension d and layer count, and a badge with the base model's parameter count.",
      "Architecture, a choice of Llama 3.2 1B, Llama 3 8B, Llama 3 70B, or Custom. Target matrices, five toggle buttons: q, k, v, o, and MLP. Model dimension d from 256 to 16,384, Layers from 1 to 126, Adapter rank r from 1 to 256, and Adapter precision of 32, 16, 8, or 4 bits.",
      "Three metrics: Trainable parameters, Share of the base model, and Adapter size at the chosen precision in megabytes of 10^6 bytes.",
      "A bar list of trainable parameters for each targeted matrix across every layer, a formula row with the live totals, and a callout comparing the toy at its current Rank with this architecture at the same rank and targets.",
    ],
    howItWorks: [
      "A LoRA pair on a matrix with `inputs` columns and `outputs` rows holds rank × (inputs + outputs) numbers: B is outputs by rank and A is rank by inputs. The query projection is d by heads × head dimension, key and value are d by key-value heads × head dimension, the output projection reverses the query's shape, and the MLP is three matrices between d and the feed-forward width.",
      "Trainable parameters sum that over the targeted matrices and every layer. The base count is the embedding table, a separate output table unless it is tied, each layer's four attention matrices, three MLP matrices and two norm vectors, and a final norm vector.",
      "Size is parameters times bits divided by 8, in megabytes of 10^6 bytes. It counts weights only, not file metadata.",
      "The 8B and 70B presets take layers, d, feed-forward width and head counts from Table 3 of the Llama 3 paper (arXiv 2407.21783). The 1B shape and the 128,256-entry vocabulary come from the released config.json files. Custom keeps Llama 3's ratios: key and value width a quarter of d, feed-forward width 3.5 times d, untied output table.",
      "Cross-check: Llama 3.2 1B with q only at rank 8 gives 524,288 trainable parameters, which is the count the Hugging Face PEFT quick tour prints for that configuration. PEFT's own percentage, 0.0424%, divides by the base plus the adapter; this card divides by the base alone, and both round to the same digits.",
    ],
    controls: [
      "`Architecture` loads a preset's d and layers; moving `Model dimension d` or `Layers` switches to Custom. `Target matrices` toggles q, k, v, o, and the MLP. `Adapter rank r` and `Adapter precision` scale the count and the size.",
      "Comparison worth running: on Llama 3 8B at rank 16 with every matrix targeted, switch the MLP off and then back on. Then set Adapter precision to 32-bit and back to 16-bit.",
      "Comparison worth running: switch Architecture through 1B, 8B, and 70B at the same rank and targets and read Share of the base model.",
    ],
    notice: [
      "At rank 16 on every matrix, Llama 3 8B trains 41,943,040 parameters, which is 0.522% of its 8.03 billion, and the adapter is 83.9 MB at 16-bit and 167.8 MB at 32-bit. The MLP holds 28,311,552 of them, 67.5%.",
      "With only q and v at rank 8 the count is 3,407,872, 0.0424% of the base. Adding the MLP to q and v at rank 16 multiplies the count by 5.15.",
      "The share falls as the model grows: at rank 16 on every matrix it is 0.912% for the 1B model, 0.522% for 8B, and 0.294% for 70B, whose adapter is 207,093,760 parameters or 414.2 MB at 16-bit.",
      "Size is exactly linear in rank, in layers, and in bits. Doubling rank doubles the file.",
      "The callout compares like with like: at rank 4 the toy trains 26.7% of its table and the 8B preset with every matrix targeted trains 0.131%, about 200 times less.",
    ],
    limits: [
      "In this lab: every number is closed-form arithmetic. No adapter is trained or served here, so the card cannot say whether a given rank is enough for a task. A custom model follows Llama 3's ratios and is not a published architecture, and the counts leave out anything beyond the LoRA pairs, such as extra saved layers.",
      "In this lab: the three presets are one family, with grouped-query attention and a gated MLP. Models with other shapes, or with biases, would change the per-matrix counts.",
      "In general: the file on disk depends on the dtype the adapter is stored in, which can differ from the base model's. PEFT, for example, casts 16-bit adapter weights to float32 by default for stable training, so the 32-bit figure is a realistic training-time size. A small share also says nothing about quality.",
    ],
  },
};

export default cardInfo;
