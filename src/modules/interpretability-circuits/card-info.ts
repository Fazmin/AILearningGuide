import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Copy task and attention": {
    title: "An attention map that matches the induction recipe",
    summary:
      "The shipped two-layer, four-head character transformer, run in your browser on 24 letters that repeat with a period you choose. The grid is one head's real attention; solid green outlines mark the key an induction head would read, dashed amber outlines mark the previous letter, and the table scores all eight heads on both keys.",
    whatYouSee: [
      "`Repeat period`, from 6 to 13 letters. 11 and 13 are marked `unseen`: the model's synthetic training repeats used every period from 4 to 31 except 11, 13, 17 and 19.",
      "The prompt as letter chips, with alternate periods shaded and a green outline on the previous occurrence of the next letter, then a `next` box with the letter the repeat calls for and the model's top guess.",
      "Three readouts: `Copy accuracy`, the probability of the right letter at the end, and the top three next letters.",
      "Eight head buttons and a query × key grid for the chosen head: rows are the letter doing the looking, columns the letters it can see, and darker purple means more attention weight. Solid green outlines mark key i − period + 1 and dashed amber outlines mark key i − 1, for each query in the second repeat. A line under the grid names the one offset that takes the most of the chosen head's attention.",
      "A table with each head's mean attention on the previous letter and on the key after the previous occurrence, and a status chip comparing this forward pass with ONNX Runtime.",
    ],
    howItWorks: [
      "The weights are read straight out of `tiny-transformer.onnx`, the attention lab's model file, and run in TypeScript: token plus learned position embedding, two pre-LayerNorm blocks with four heads of width 64 and causal softmax(QKᵀ / 8), an exact-erf GELU MLP from 256 to 768 to 256, then a final LayerNorm and unembedding.",
      "The status chip reruns the prompt through ONNX Runtime in the model worker and reports the largest logit difference it finds. A unit test also matches every attention weight in `interpretability-cache.json`, computed in PyTorch, to within 5 × 10⁻⁵.",
      "`Copy accuracy` is the mean softmax probability of the correct next letter over positions period to 23: the second repeat onward, where the current letter has already appeared once. Position period − 1 is left out, because nothing before it can say that the string repeats.",
      "The table's columns are mean attention from every query i ≥ period to key i − 1 (the previous letter) and to key i − period + 1 (the token after the previous occurrence). With three or four repeats in the prompt the nearest earlier copy is only one of several, so an induction head spreads its attention over all of them.",
    ],
    controls: [
      "`Repeat period` rebuilds the prompt from the same pool of letters, `QXZRKWMPJVBH…`.",
      "The head buttons and the table rows both choose which head the grid shows.",
      "Worth running at period 8: choose L2 H3, then L1 H4, then L1 H2, and read the line under the grid. L2 H3's largest share is 7 back (0.67), the key after the previous occurrence; L1 H4's is 1 back (1.00); L1 H2's is 2 back (0.99).",
      "Then set the period to 11 or 13, which the training repeats never used: the green outlines move to 10 or 12 back, and L2 H2 and L2 H3 follow them.",
    ],
    notice: [
      "Copy accuracy is 94.5% at period 8 and 88.0% to 97.2% at every period from 6 to 13, including the unseen 11 and 13 (88.5% and 88.0%). The first scored letter, the first of the second repeat, is the weak one; from the second letter on the mean is at least 95% at every period.",
      "Only L2 H2 and L2 H3 put much attention on the induction key (0.55 to 0.84 across the periods); no other head puts more than 0.10 there. L1 H3 and L1 H4 put 0.98 or more of their attention on the previous letter at every period. Those are the two halves of the textbook induction circuit.",
      "The stripe follows the period: the strongest offset for L2 H2 and L2 H3 is period − 1 at every period from 6 to 13. A head that always looked 7 back would draw the same stripe at period 8 and a different one elsewhere; these heads do not.",
    ],
    limits: [
      "In this lab: the prompts are synthetic capitals from one fixed pool, 24 letters long, one prompt per period. The model was trained mostly on Shakespeare, with about 30% of each batch synthetic repeated strings of period 4 to 31 (minus 11, 13, 17 and 19), so a repeated string of capitals is in-family, not a far-out test.",
      "In this lab: the table names two keys, and a head can do something else. L1 H1 puts about 0.5 to 0.6 of its attention exactly 3 back, and L1 H2 about 0.99 exactly 2 back; neither column shows that, and the line under the grid does.",
      "In general: attention shows where information could flow, not what it is used for. A previous-token head in layer 1 feeding an induction head in layer 2 is the standard account of induction (Elhage et al., 2021; Olsson et al., 2022); the attention map makes it a hypothesis, and the next two cards test it.",
      "In general: passing the period test on unseen periods is evidence that the heads follow content, not a proof that no positional signal also contributes, and one small model says little about what a large model does.",
    ],
  },

  "Head ablation": {
    title: "Necessity, one head at a time, two ways",
    summary:
      "Switch any of the eight heads off, either by zeroing its output or by replacing it with its average over random repeats, then read copy accuracy and the induction heads' attention on the current prompt. Each button also says what removing that head alone costs, so you can compare single and joint ablations and compare the two ways of ablating.",
    whatYouSee: [
      "An `Ablation` control with `Zero` and `Mean`, which sets what replaces a switched-off head's output.",
      "Eight toggles, L1 H1 to L2 H4. Each shows `on` or `off` (struck through, dashed red border when off) and `alone ±x pts`, the change in copy accuracy when only that head is removed, under the chosen ablation.",
      "`Ablate layer 1`, `Ablate layer 2` and `Restore all` shortcuts.",
      "Readouts: copy accuracy intact, with the chosen heads off, the change in percentage points, and the model's top next letter with those heads off; then `Attention on the induction key, L2 H2 · L2 H3`, the intact values, an arrow, and the values with the chosen heads off.",
      "Paired bars for every scored position: outlined blue for the intact model, solid red with the chosen heads off, with the letter and position under each pair.",
    ],
    howItWorks: [
      "Zero-ablation multiplies that head's attention-weighted values by 0 before the output projection W_O. W_O's bias and every other head are untouched, and the whole forward pass reruns.",
      "Mean ablation replaces the head's pre-projection output with one constant vector: its average over the second repeat (positions period to 23) of eight seeded random-letter strings with the same period, not the prompt on screen. The SDK runner can scale a head but not substitute a value, so the lab scales the head to zero and adds mean · W_O to that block's output bias, which is the same forward pass. A unit test checks the construction: replacing a layer-2 head by its own output at the last position leaves that position's logits unchanged.",
      "`alone` comes from eight extra forward passes (sixteen for mean, which first averages the reference strings), run just after the prompt or the ablation changes; it is copy accuracy with only that head off minus copy accuracy intact.",
      "The induction-key readout is the table's mean attention on key i − period + 1, recomputed on the ablated run. A layer-2 head's attention depends on layer 1's output but not on other layer-2 heads, so switching off layer-2 heads never changes it.",
      "The bars are the same per-position probabilities that `Copy accuracy` averages.",
    ],
    controls: [
      "Click a head to toggle it; the shortcuts set whole layers; `Ablation` switches between Zero and Mean.",
      "Worth running at period 8 (intact 94.5%): L1 H4 alone leaves 9.9% under zero and 24.3% under mean, and the induction-key attention of L2 H2 · L2 H3 falls from 0.66 · 0.67 to 0.19 · 0.24 under zero. Switch L1 H3 and L1 H4 off together and it falls to 0.01 · 0.03.",
      "Then switch off L2 H3 alone, under Zero and then Mean, and compare: 6.0% against 47.9%.",
    ],
    notice: [
      "Single heads are necessary here. Zero-ablating one head at period 8 leaves L1 H2 5.1%, L2 H3 6.0% and L1 H4 9.9%; L2 H2 50.2% and L1 H3 56.6%; and L1 H1, L2 H1 and L2 H4 89.2%, 95.8% and 99.6%.",
      "Zero overstates. Under mean ablation L2 H3 leaves 47.9% and L2 H2 75.9%, but L1 H2, L1 H4 and L1 H3 still leave 27.4%, 24.3% and 48.8%. The same layer-1 heads matter either way; the induction heads' solo roles shrink, and they partly overlap (both off: 1.3% under zero, 2.4% under mean).",
      "L2 H4 off raises accuracy under zero (99.6%) and barely moves it under mean (95.3%). The head is not needed for this task; an ablation reports an effect on one metric on one input, not what the head is for.",
      "L1 H2 attends two letters back, not to the previous one, yet removing it costs as much as removing L1 H4 and also cuts the induction heads' attention (0.30 · 0.21 under zero). It is needed; what it computes is not established here.",
    ],
    limits: [
      "In this lab: one synthetic prompt per period. Offline tests repeat the main contrasts on three seeded random strings at each of periods 9 and 13, and the same layer-1 heads stand out, but the lab does not show those strings.",
      "In this lab: the mean is one constant vector per head, averaged over eight random repeats. A position-by-position mean, or a resample ablation that swaps in another prompt's value, is a different replacement and could give different sizes. Zero and Mean disagree on size and agree on which layer-1 heads matter; neither is ground truth.",
      "In general: ablation tests necessity, not mechanism, and trained models can repair themselves when a component is removed: McGrath et al. (2023) call this the Hydra effect, and Wang et al. (2022) found backup name mover heads that take over when the main ones are knocked out.",
      "In general: no path patching was run, so \"layer-1 previous-token heads feed layer-2 induction heads\" rests on the induction heads' attention falling when they are off, not on a patched route; L1 H2 shows that other layer-1 heads matter too.",
    ],
  },

  "Activation patching": {
    title: "Transplant one vector, measure what returns",
    summary:
      "Run the model on a clean prompt and on a corrupted copy, then paste one residual-stream vector from the clean run into the corrupted run and measure how much of the metric comes back. Every cell is one such experiment, run live; two sweeps precomputed in the shipped cache are included for comparison.",
    whatYouSee: [
      "`Sweep`: the live copy task, or the cached `speaker` and `letters` sweeps from `interpretability-cache.json`. For the live sweep, `Corrupt` chooses which letter is replaced: the letter to copy, or an unrelated one.",
      "The clean and corrupted prompts as chips, with the replaced position outlined in amber.",
      "The metric, then readouts: its value on the clean run, on the corrupted run, the gap between them and, for live sweeps, each run's top next letter. A warning appears when the gap is under 1 logit.",
      "A grid with rows `After block 1` and `After block 2` and a column per position. Green cells recover the metric, red cells push it the other way, and the colour saturates at ±100%.",
      "A readout for the selected cell with its exact recovery and, for live sweeps, the metric after patching.",
    ],
    howItWorks: [
      "Live sweep: the corrupted prompt replaces one letter with `Y`, a letter the prompt never uses. With `The letter to copy` it is the previous occurrence of the next letter (position 24 − period); with `An unrelated letter` it is a letter in the first repeat, half a period from the letter being copied. The metric is logit(right letter) − logit(Y) at the last position.",
      "Each cell overwrites the corrupted run's residual vector after block 1 or block 2, at one position, with the clean run's vector, and reruns the blocks after it. Recovery = (patched − corrupted) / (clean − corrupted).",
      "Cached sweeps were computed by `precompute_interpretability.py` with the target logit alone as the metric. `letters` is the same period-13 prompt pair the live sweep builds at period 13, so the two compare directly: gaps of 11.99 and 18.43 under the two metrics, and recoveries within 0.006 of each other in every cell. A unit test reproduces the speaker sweep to two decimals.",
      "After the last block nothing mixes positions, so in the `After block 2` row every cell but the last is exactly 0% and the last is exactly 100%.",
    ],
    controls: [
      "Click any cell, or Tab to it and press Enter, to read its value.",
      "Worth running at period 8: `After block 1` at position 16, the letter to copy, recovers 98.8%, and at the last position −0.5%. Then set `Corrupt` to `An unrelated letter`: the gap falls from 11.70 to 0.16 and the warning appears.",
    ],
    notice: [
      "One route, clearly. After block 1 only the corrupted letter's own position recovers (98.8% at period 8) and every other cell in the row stays within 0.6 points of zero. The letter's identity is carried at that earlier position, where layer 2's heads can read it; the last position does not hold it yet.",
      "The model copies what it finds there. With `Y` in place of the letter to copy, the corrupted run's top prediction is `Y`: 48% at period 8 and 99.5% at period 13.",
      "The gap is large at every period: 6.05 to 22.35 logits across periods 6 to 13, and on 16 random period-13 strings it averages 20.7 and never falls below 15.9.",
      "The control shows what a bad measurement looks like. With an unrelated letter replaced, the gap is under 0.35 logits at every period, yet at period 8 a cell after block 1 still reads 43%, because the denominator is tiny.",
      "`Cached · speaker` recovers only at the last position (96%): its corrupted prompt is cut to the clean one's length and ends at JULIET with no colon, so the last character is where the runs differ. `Cached · letters` recovers 99.5% at position 11.",
    ],
    limits: [
      "In this lab: patching is at block outputs in the residual stream only, not per head or per path, and each sweep uses one prompt pair. The metric is a logit difference against one foil letter, `Y`, chosen before the sweep.",
      "In this lab: the corruption changes one letter. The cached `speaker` pair differs in its final character, which makes the last position trivially the place the difference enters; it shows a pattern, not a discovery.",
      "In general: a patching result supports a claim about this prompt pair and this metric only. A circuit found on one template may not transfer to paraphrases, other models or other seeds, and it says where information passes, not which head wrote it.",
      "In general: a full sweep costs one forward pass per cell. Attribution patching estimates the whole grid from two forward passes and one backward pass, at some cost in accuracy.",
    ],
  },
};

export default cardInfo;
