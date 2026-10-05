import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Student controls": {
    title: "Size, target, temperature, and how long the student trains",
    summary:
      "The knobs that define the student: how wide its factorization is, whether it imitates the teacher's distribution or the one-hot labels, how much both distributions are softened in the loss, and how many passes it gets. Both students retrain on every change, so the comparison is always live.",
    whatYouSee: [
      "`Training target`: `Teacher distribution` (soft) or `One-hot labels` (hard).",
      "`Student size` from rank 1 to 12, printed as `rank r · 60r weights`.",
      "`Distillation temperature` from 0.5 to 6, and `Student epochs` from 1 to 40.",
      "Metrics `Student loss` (hard-label corpus cross-entropy of the selected student), `Teacher loss`, and `KL to teacher`, plus a callout comparing the two students' KL.",
    ],
    howItWorks: [
      "Soft labels minimize T² · KL(softmax(teacher/T) ‖ softmax(student/T)) at every corpus position. Its gradient with respect to the student's logits is T · (student_T − teacher_T), which keeps step sizes comparable across temperatures.",
      "Hard labels minimize ordinary cross-entropy against the next character. The teacher is never read during that update.",
      "`KL to teacher` is KL(teacher ‖ student) at T = 1, averaged over the corpus's own positions, so each context counts as often as the data uses it.",
    ],
    controls: [
      "All four controls retrain both students; `Distillation temperature` only affects the soft one.",
      "Comparison worth running: at rank 6, sweep T through 0.5, 1, 1.5, 2, 4, and 6. KL reads 0.4037, 0.1348, 0.1340, 0.1407, 0.2193, and 0.2737 against the hard student's 0.2577.",
    ],
    notice: [
      "Soft labels win at every rank here: 0.4200 against 0.4668 at rank 2, 0.1348 against 0.2577 at rank 6, 0.0722 against 0.1295 at rank 12 (all at T = 1).",
      "Temperature has a shallow optimum near 1 to 1.5. Very high T spends the student's limited rank on matching tiny probabilities.",
    ],
    limits: [
      "In this lab: the loss is either pure soft or pure hard, with no mixing weight, and every run uses seed 3.",
      "In general: production distillation usually blends the tempered KL with a hard-label term, and uses a teacher far more confident than a bigram.",
    ],
  },

  "Distillation": {
    title: "Two students of the same size, two targets",
    summary:
      "Training curves for the soft-label and hard-label students, both measured with the same hard-label cross-entropy, and a bar pair for how far each ended up from the teacher's distribution.",
    whatYouSee: [
      "A kicker `Teacher 900 weights · student N` and a badge with the student's share of the teacher's parameters.",
      "A chart of minibatch hard-label loss against optimizer step: solid soft, dashed hard, dotted teacher.",
      "Bars for KL(teacher ‖ student) over the corpus's own contexts, lower is closer; the selected target is emphasized.",
    ],
    howItWorks: [
      "The teacher is a 30 × 30 character bigram trained 80 epochs on Harbor weather notes at seed 1: loss 1.887, perplexity 6.600.",
      "Each student's logits are U · V with inner dimension `rank`, so it has 2 · rank · 30 weights. Both use batch 16, learning rate 0.5, and seed 3.",
      "The KL average skips the four context rows the corpus never visits (j, x, <, >): neither model was trained there.",
    ],
    controls: [
      "No controls of its own; everything on Student controls redraws it.",
      "Comparison worth running: flip Training target and read the emphasis move. Both bars are real runs whichever is selected.",
    ],
    notice: [
      "The two loss curves nearly overlap even when the KL bars differ by a factor of two. Hard-label loss is not the same question as closeness to the teacher.",
      "The dotted line is a finished 80-epoch model with no rank limit; the students are not expected to reach it.",
    ],
    limits: [
      "In this lab: a 900-weight bigram teacher and low-rank students on one short corpus.",
      "In general: a student inherits the teacher's errors along with its skills, and matching next-token distributions does not by itself transfer multi-step behaviour.",
    ],
  },

  "Soft targets": {
    title: "What one context's soft label looks like",
    summary:
      "For one context character, three next-character distributions side by side: the one-hot labels averaged over the corpus (the frequencies a hard-label student chases), the teacher at the current temperature (the soft target), and the selected student at T = 1.",
    whatYouSee: [
      "`Context character`: ␣, h, t, e, o, or s, and a kicker counting how many different characters follow it in the corpus.",
      "For the teacher's ten most likely next characters plus a `rest` row: grey bar for label frequency, accent bar for the teacher at T, blue bar for the student, and the three percentages.",
      "Metrics for the teacher's probability mass on next characters that never follow this context in the corpus, at T = 1 and at the current T.",
    ],
    howItWorks: [
      "Label frequency counts corpus bigrams starting with the context. Teacher and student bars are row softmaxes of their logit tables.",
      "The unseen mass sums the teacher's probability over characters with zero label frequency, excluding the two reserved symbols.",
    ],
    controls: [
      "`Context character` here; `Distillation temperature` and `Training target` on Student controls.",
      "Comparison worth running: after “h”, the teacher puts 4.8% on unseen characters at T = 1, 39.3% at T = 2, and 64.6% at T = 4. Then pick ␣ with One-hot labels: the hard student gives “a” 46.2% where the labels say 14.1%.",
    ],
    notice: [
      "At T = 1 the teacher is close to the label frequencies (76.6% against 76.0% for “e” after “h”). A bigram teacher knows little beyond the counts, so its dark knowledge is a thin tail.",
      "Temperature reshapes the target the soft student sees; the student bars stay at T = 1 because that is how it will be used.",
    ],
    limits: [
      "In this lab: six context characters and a top-ten view of a 30-way distribution.",
      "In general: a large teacher's soft labels carry similarity structure (which wrong tokens are plausible) that label counts cannot, which is where distillation earns most of its value.",
    ],
  },

  "Pruning controls": {
    title: "How much to delete, and in what shape",
    summary:
      "The two decisions magnitude pruning needs: the share of weights to zero, and the pattern of the zeros. The two samples use the same prompt and seed, so any difference is the missing weights.",
    whatYouSee: [
      "`Pruned share` from 0% to 90%, reading `fixed at 50% by 2:4` in 2:4 mode.",
      "`Pruning mode`: `Unstructured`, `2:4 semi-structured`, or `Structured rows`.",
      "Samples from the teacher and the pruned table: prompt “the ”, temperature 0.7, 68 characters, seed 4, with a count of differing characters.",
    ],
    howItWorks: [
      "Unstructured zeroes the round(share × 900) smallest-magnitude weights anywhere.",
      "2:4 keeps the two largest of every four consecutive weights down each column (along the context dimension); the leftover pair of rows keeps one of two. Exactly 450 zeros.",
      "Structured zeroes the round(share × 30) context rows with the smallest L2 norm.",
    ],
    controls: [
      "`Pruned share` and `Pruning mode`.",
      "Comparison worth running: at 50%, Unstructured gives perplexity 6.65, 2:4 gives 7.15, and Structured rows give 8.22.",
    ],
    notice: [
      "Identical samples at a non-zero share are possible: the removed weights may be ones this draw never needed.",
      "The note under the samples is the serving distinction: scattered zeros do not skip work, a 2:4 pattern can on supporting hardware, and removed rows shrink the matrix.",
    ],
    limits: [
      "In this lab: nothing is timed, so the speed claims are stated from how dense and sparse kernels work, not measured.",
      "In general: pruning is usually followed by fine-tuning to recover quality; here the zeros are one-shot.",
    ],
  },

  "Pruning": {
    title: "Quality against the share deleted, and the shape of the mask",
    summary:
      "Perplexity of the teacher with a rising share of weights set to zero, for unstructured and whole-row pruning, with the fixed 2:4 point at 50%, and a 30 × 30 map of which weights the current setting kept.",
    whatYouSee: [
      "A chart of share removed (0–90%) against perplexity: solid unstructured, dashed structured, a diamond for 2:4, and a marker at the current share.",
      "A mask grid: rows are context characters, columns next characters. Filled cells are kept and shaded by magnitude; empty outlined cells are zero.",
      "Metrics `Unpruned`, `Pruned`, and `Rows removed`.",
    ],
    howItWorks: [
      "Each point re-prunes the frozen teacher and re-measures exp of corpus cross-entropy. Nothing is retrained.",
      "The mask is the current setting's zero pattern, from the same pruning call that produced `Pruned`.",
    ],
    controls: [
      "No controls of its own; `Pruned share` and `Pruning mode` move the marker, the mask, and the metrics.",
      "Comparison worth running: at 30%, unstructured costs almost nothing (6.61) because it removes the four never-used rows and the smallest logits first; structured already costs 6.95.",
    ],
    notice: [
      "The 2:4 diamond sits between the two curves: a constrained pattern costs more than free choice at the same share and less than deleting rows.",
      "Unstructured is nearly flat to 50% and then bends: 6.65 at 50%, 6.84 at 70%, 8.07 at 90%.",
    ],
    limits: [
      "In this lab: a row of a bigram table stands in for a head or neuron, and magnitude is the only importance score on this card; the next card adds two more.",
      "In general: unstructured sparsity saves little time on dense hardware, 2:4 gives up to about 2× on the matmul with supporting tensor cores, and structured pruning shrinks the model itself.",
    ],
  },

  "Compose the three steps": {
    title: "Distil, prune, quantize: size and perplexity after each stage",
    summary:
      "The student from the cards above is stored as its two factors, pruned by an importance score, then quantized, and its stored size and corpus perplexity are re-measured after each stage. A second bar list holds the pruned share fixed and compares three importance scores: magnitude, activation-weighted, and measured ablation.",
    whatYouSee: [
      "`Importance score` (`Magnitude`, `Activation-weighted`, `Measured ablation`), `Share pruned from the student` (0 to 90%), and `Final bit width` (2 to 8). The student itself comes from Student controls.",
      "A bar list with one row per stage, `Teacher`, `Distilled student`, `Pruned`, and `Quantized`: bar length is stored bytes, and the detail line gives the stored values, bits each, and perplexity with its change on the teacher's 6.600. The badge is the teacher's size over the final size.",
      "A second bar list with the three scores' perplexity after the prune stage at the current share, drawn as the rise over the unpruned student, with the passes each score needs to compute.",
      "Metrics `Final perplexity`, `Final size`, `Passes to score`, and `Values kept`.",
    ],
    howItWorks: [
      "Distil: the student's table is exactly rank-limited, so a one-sided Jacobi factorization splits it into U (30 × r) and W (r × 30) with U·W equal to the table to float precision. The student stores those 60r values.",
      "Prune scores every factor entry and zeroes the lowest-scored round(share × 30r) of each factor. Magnitude is |w|, the rule of the Pruning card. Activation-weighted is |w| times the norm of the input over the corpus: the square root of the context's count for U, and the square root of the count-weighted squares of U's component for W, a Wanda-style score. Measured ablation is the rise in corpus cross-entropy when that one entry is set to zero, computed exactly from the corpus's bigram counts.",
      "Sizes use a 32-bit baseline. Teacher: 900 × 32 bits = 3,600 B. Student: 360 × 32 = 1,440 B at rank 6. Pruned: kept values × 32 bits plus one mask bit per position when anything was removed. Quantized: the kept values in storage order through the SDK's group quantizer, blocks of 32 with two 16-bit numbers each, so kept × bits + blocks × 32 + the mask. Pruned positions stay zero.",
      "Passes to score: 0 for magnitude, 1 for activation-weighted, and one per factor entry (60 × rank) for ablation. Every perplexity is the exponent of the corpus cross-entropy on the harbor text, which is also the calibration text for the last two scores.",
    ],
    controls: [
      "`Importance score`, `Share pruned from the student`, and `Final bit width` redraw both bar lists. The student comes from the Student controls card: its rank, target, temperature, and epochs.",
      "Comparison worth running at rank 6, T = 1, 18 epochs, 50% and 4 bits: Teacher 3,600 B at 6.600, Distilled student 1,440 B at 8.326, Pruned 765 B at 8.560, Quantized 159 B at 8.619. That is 22.6 times smaller for +30.6% perplexity.",
      "Then raise the share to 70% and compare the scores after pruning: Magnitude 9.682, Activation-weighted 9.999, and Measured ablation 9.480, for 0, 1, and 360 passes.",
    ],
    notice: [
      "The stages buy different things: distilling saves 2,160 B for +1.73 perplexity, pruning 675 B for +0.23, and quantizing 606 B for +0.06 at 4 bits. At 2 bits the quantize stage costs +1.34.",
      "Activation-weighted does not beat magnitude here, the opposite of what Wanda's authors report on LLaMA models. In a one-hot bigram an input's activation is only how often its context occurs: there are no outlier channels for magnitude to miss, and the score ignores how much the softmax cares about each logit.",
      "Pruning only saves bytes here because of the mask, a sparse format: 180 zeros out of 360 values cost 360 mask bits, and a dense matrix multiply would still touch every position.",
    ],
    limits: [
      "In this lab: there is no fine-tuning after pruning or quantizing, both factors lose the same share, the scores and the perplexity share one corpus and one seed, and the sizes are idealized bit counts rather than a file format.",
      "In general: real pipelines recover quality by fine-tuning between stages, prune in shapes the hardware can skip, calibrate on text held apart from the evaluation, and measure speed on the target hardware as well as size.",
    ],
  },
};

export default cardInfo;
