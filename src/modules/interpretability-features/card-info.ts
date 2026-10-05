import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Residual to sparse code": {
    title: "A dense vector, rewritten sparsely",
    summary:
      "Your text runs through the shipped transformer, and the residual stream after its last block, at the character you pick, goes through the shipped sparse autoencoder. The panels show the 256 dense numbers, the 1,024-slot code with its 32 active features, and how well those 32 rebuild the original.",
    whatYouSee: [
      "A `Probe text` box (up to 64 characters; Enter adds a line break) and one button per character; the pressed one is inspected. The line under it says how many characters that position has read.",
      "The residual panel: 256 bars for the normalized residual x, with a red dash on each bar for the SAE's reconstruction x̂.",
      "The code panel: 1,024 slots with a spike for each active feature, its height the activation; the selected feature's spike is red.",
      "Three readouts: active features (L0), the unexplained share ‖x − x̂‖² / ‖x‖², and the cosine between x and x̂.",
      "The eight largest active features with their values. `window start` marks a feature whose cached top examples come from the first four positions of a window; `no cached examples` means the feature has no training-text evidence.",
    ],
    howItWorks: [
      "`tiny-transformer.onnx` runs in the model worker and returns the residual stream after each block. The lab takes the one after block 2, the layer the SAE was trained on.",
      "x = (r − μ) / s, with the per-dimension corpus mean μ and one global scale s = 3.58 stored in `sae-top-activations.json`, exactly as training normalized its inputs.",
      "`residual-sae.onnx` then computes f = TopK₃₂(ReLU(W_enc x + b_enc)) and x̂ = W_dec f: every value but the 32 largest is set to zero, and the decoder has no bias.",
      "The whole probe text is one window starting at position 0, so the character at position p has read p + 1 characters. That is also how the SAE's training windows were cut.",
    ],
    controls: [
      "Edit the text or click a character to inspect another position; choosing a feature row selects it for the cards below.",
      "Worth running: at the default position 4, the first space, 32 of 1,024 features are active, 1.0% of ‖x‖² is left unexplained, and #121 is the largest at 14.04. Second is #499 at 12.82, which carries a window-start mark. Then click position 0: all six of its largest features are marked.",
    ],
    notice: [
      "The residual is dense: nearly all 256 numbers are nonzero. The code is sparse by construction: exactly 32 of 1,024, at every character, because TopK fixes the count.",
      "Reconstruction is good but uneven. On the default text the unexplained share runs from 0.0% at position 0 to 22.3% at the K of KING, and averages 7.0%. On 200 held-out windows the SAE leaves 5.1% of the variance unexplained.",
      "The largest feature at positions 0, 1 and 2 is a window-start one (#959, #876, #876). At the K of KING, position 28, none of the six largest is marked.",
      "The code is wider than the vector it describes, four features per dimension. That only works if most features are absent most of the time.",
    ],
    limits: [
      "In this lab: one SAE on one layer of a 1.4-million-parameter character model, trained once (one seed) for 10,000 steps on 1,600 windows of Tiny Shakespeare. Nothing here shows that a different seed would find the same features.",
      "In this lab: the SAE reads characters in the context you type; text unlike the training corpus can land in regions it reconstructs poorly, and the first characters of any text sit at positions where the residual is unlike running text.",
      "In general: Anthropic reported extracting millions of features from the middle layer of Claude 3 Sonnet. A separate study (Paulo and Belrose, 2025) found that SAEs trained on the same model and data with different random seeds shared only about 30% of their features in one large Llama 3 8B setting, so a feature list is one run's answer.",
    ],
  },

  "Feature evidence": {
    title: "A label is a hypothesis",
    summary:
      "Pick one of the 128 features whose top examples were cached at training time, read where each example sat in the model's window and which flags the cache set, then turn a label into a rule and score it on your own text, at a threshold you choose, as precision and recall.",
    whatYouSee: [
      "A `Feature` menu of the 128 cached features in cache order (largest maximum first), each listed with the character its top examples fired on most; `window start` marks the flagged ones.",
      "Flag chips for the selected feature, and a line giving how many of the 115,200 scanned characters it is active on and how many of those were at window position 0.",
      "`Examples shown`: the top eight overall, or, for a feature with early-window examples, the top eight from position 4 on.",
      "Eight training-text snippets with the character the feature fired on highlighted, a badge giving its window position, the activation, and a bar relative to the largest shown. Dimmed text is outside what the model had read: before its 64-character window, or after the marked character.",
      "`Label to test` (`Same character`, `Capital letter`, `Space`, `Line break`, `First 4 characters`) and `Activation threshold` from 0 to 10 in activation units.",
      "Your text as chips shaded by the feature's activation, with the value underneath. A heavy border means the feature fires, a pink bar means the character matches the label, and a dotted top edge marks window positions 0 to 3.",
      "Precision, recall and how many characters the feature fires on, the counts of hits, misses and false alarms, and how many of the shown examples the label matches.",
    ],
    howItWorks: [
      "The examples come from `sae-top-activations.json`, written by `train_sparse_autoencoder.py`: over 1,800 scanned windows (1,600 training and 200 held-out), the 128 features with the largest maximum activation, each with its eight highest-activating positions and up to 32 corpus characters either side.",
      "Every example records `window_position`: how many characters the transformer had read before it (0 is the first). `focus_offset` is something else, the index of the character inside the excerpt, always 32 here. Position labels use the window position.",
      "Flags: a feature is a position-0 artefact if at least half its top examples are at position 0, a window-start artefact if at least half are within positions 0 to 3, and position-0-dominated if more than 20 times the chance share (1 in 64) of its activations are at position 0. A feature with any early example also gets `examples_in_context`, its top eight from position 4 on.",
      "`Same character` means the most common focus character among the examples shown. A character fires when its activation exceeds the threshold. Precision = hits ÷ (hits + false alarms); recall = hits ÷ (hits + misses). `First 4 characters` matches window positions 0 to 3: a probe text's index, a cached example's `window_position`.",
    ],
    controls: [
      "Choose a feature from the menu or from the list in the first card; pick a label; move the threshold; switch which examples are shown.",
      "Worth running: #876, the first in the menu (largest maximum, 56.9), has all eight top examples at window position 1 or 2, so `First 4 characters` matches 8 of 8. Switch to the top eight from position 4 on: its largest is 36.3, at position 19, on an ampersand.",
      "Worth running: #121 with `Space` at 5.0 scores 100% precision and 70% recall on the default text; at 2.0 it scores 90% and 90%. On 768 held-out characters the same label scores 96% and 91% at 5.0, 63% and 97% at 2.0, and 100% and 56% at 8.0.",
      "Worth running: #511's top eight are all commas, at about 14. Test `Same character` at 3.0: on the default text it scores 6% precision and 100% recall, on the held-out characters 9% and 100%.",
    ],
    notice: [
      "The cache flags 21 of its 128 features as window-start artefacts, 18 of them position-0 artefacts. None of the 1,024 features fires only at position 0, but 24 are dominated by it. The top-ranked feature, #876, is one of the flagged: its maximum describes position 1, not language.",
      "Flagging is not the whole story. #876 is active on 31.6% of all scanned characters. On the held-out passages it averages 13.3 at position 1 and 0.64 from position 8 on, so it is large early and small later, and `First 4 characters` is a poor label for it: 43% precision and 40% recall at 5.0.",
      "Top examples are the extreme end of a feature, not its typical behavior. #511's eight examples are commas; at 5.0 on held-out text it fires on 18 of 19 commas and on 102 other characters.",
      "An excerpt shows every line break as a space. #53's eight examples all show a space, yet `Space` scores 0% precision and 0% recall at 5.0 on held-out text, while `Line break` scores 41% precision and 62% recall. Type a line break into the probe text to see it.",
    ],
    limits: [
      "In this lab: labels are simple character and position rules so they can be scored automatically. A human label such as `speaker names` carries meaning no such rule captures, and one short probe text is a very small test set; the 768-character check behind the quoted figures is also small.",
      "In this lab: only 128 of 1,024 features have cached examples, chosen for their largest activations rather than at random. The flags only catch the start-of-window case: #511 is unflagged and still a poor comma label, and nothing here guarantees that an unflagged feature's examples are representative.",
      "In general: labels are usually proposed by a language model reading top examples, then scored by how well they predict activations on held-out text. Neither step shows the feature causes anything; steering along its direction is the causal test, and the next card tries it.",
    ],
  },

  "Sparsity and superposition": {
    title: "Fewer features, worse reconstruction",
    summary:
      "Rebuild the chosen character's residual from only its k largest features, using decoder directions read from `residual-sae.onnx`, and watch the sparsity–fidelity trade. Then measure how much the directions overlap, find the most similar pair, and read the SAE's own training curve.",
    whatYouSee: [
      "`Keep top k features`, from 1 to 32.",
      "Left chart: the unexplained share against k, largest features first, with a marker at the slider. Right chart: the SAE's reconstruction error over its 10,000 training steps.",
      "Three readouts: the unexplained share keeping k, keeping 1, and keeping all 32.",
      "Overlap readouts: mean and largest |cosine| between the decoder directions of the features active here, and across all 523,776 pairs of the 1,024 directions.",
      "The most similar pair, its signed cosine, the number of pairs above 0.9, and how often the pair is active together on your text.",
    ],
    howItWorks: [
      "The decoder matrix is parsed from the ONNX file in the browser. Each row is one feature's direction, of unit length because training renormalized it after every step.",
      "x̂_k = Σ f_i d_i over the k largest active features; the unexplained share is ‖x − x̂_k‖² / ‖x‖². With all 32 kept it matches the ONNX reconstruction in the first card.",
      "Overlap is |d_i · d_j| for unit rows; the most similar pair is the largest, reported with the sign of d_i · d_j. The training curve and the 3.125% active fraction are read from `sae-top-activations.json`.",
    ],
    controls: [
      "Drag `Keep top k features`; pick a different character in the first card to redraw the curve.",
      "Worth running: at the default first space, one feature leaves 53.5% unexplained, eight leave 4.6%, and all 32 leave 1.0%.",
    ],
    notice: [
      "The curve drops steeply, then flattens: a handful of features carry most of the vector, and the last 16 add little. It is not strictly falling: at the first space the 29th feature nudges the share from 1.00% to 1.03%, and at the K of KING it ticks up four times. The code was fitted as a whole, not to be rebuilt in order.",
      "Sparsity here is set by TopK, not by the penalty. 3.125% of features, 32 of 1,024, were active at every logged step, and the L1 term added only 9.7 × 10⁻⁵ to a final loss of 0.0471.",
      "1,024 unit directions in 256 dimensions average |cos| 0.069: nearly orthogonal, not orthogonal. That overlap is superposition, measured in the dictionary itself.",
      "The maximum is 0.992, between #108 and #499, and its sign is negative: they point almost exactly opposite ways. Seven pairs exceed 0.9 and all seven are negative. On 768 held-out characters, #108 and #499 are never active together, and one of the two is active on 734. From position 4 on, #499 is active on 101 of the 111 spaces and #108 on 6; on the 181 lowercase vowels it reverses, #108 on 132 and #499 on 41.",
      "That reads as a sign split, not as two copies of one concept: the encoder passes values through a ReLU, so a direction in the residual that takes both signs needs one feature for each side. It is not feature splitting, which needs a wider dictionary this lab does not train.",
    ],
    limits: [
      "In this lab: cutting a trained 32-feature code down to k is not the same as training with k. An SAE trained at k = 8 would learn different, broader features.",
      "In this lab: the overlap numbers describe the SAE's dictionary, which is a model of the transformer's representation, not the transformer's own basis. The sign-split reading rests on the cosine and on never being co-active on 768 characters; we did not test, for example, whether removing one of the pair changes the reconstruction.",
      "In general: an L1 penalty shrinks active values and biases reconstructions, TopK fixes the count directly, and JumpReLU learns a threshold; each yields a different dictionary from identical activations. Feature absorption, where a feature fails to fire where it should because a more specific feature has taken over (Chanin et al., 2024), makes labels harder to trust.",
    ],
  },

  "Steer with a feature": {
    title: "Does turning a feature up do what its label says?",
    summary:
      "Pick one of five features whose labels were measured on held-out text, add its decoder direction, times a strength, to the transformer's layer-2 residual at one position, and read how the next-character distribution changes. The same push along random directions of the same length shows what a real effect has to beat.",
    whatYouSee: [
      "Four controls: `Feature to steer`, `Steering position` (a character of the probe text, default 43, the comma after aye), `Steering strength` from 0 to 15, and `Random control direction` (a seed from 1 to 20).",
      "A sentence giving what the model has read, the character your text continues with, how long the push is next to the residual, and how the label scores on your text.",
      "Three lists of the six most likely next characters, with their probabilities as text: before, after adding the feature, and after adding random direction `seed`. The label character is marked `(label)` and added if it is not in the top six.",
      "Readouts: the label character's probability before and after, its change in log probability (nats), the range of that change over 20 random directions, and how many standard deviations the feature's change is from their mean.",
      "A sentence stating the verdict, the three largest rises and falls, a chart of the change against strength with the random spread as dotted lines, a table of the same numbers, and a `Continue` button that writes 12 characters greedily three ways.",
    ],
    howItWorks: [
      "The SAE sees x = (r − μ) / s and each decoder direction d has unit length in x-space, so strength k adds k × s × d to the raw residual r (s = 3.58): it raises the feature's activation by about k. At position 43 the residual is 90.1 long, so strength 10 adds a vector about 40% of its length.",
      "The patch is the SDK's `finalLogitsWithPatch`, run in TypeScript on the weights read from `tiny-transformer.onnx`; a test checks it equals a full forward pass with the residual overwritten. Layer 2 is the last block, so the push acts only through the final normalization and the output matrix, on that position's prediction.",
      "The null control is 20 fixed unit-length Gaussian directions (seeds 1 to 20) scaled the same. The readout `SDs` is (feature change − mean random change) ÷ their standard deviation; beyond about 3 the change is unlikely to be a random draw, under 2 it cannot be told apart.",
      "`Continue` takes the most likely real character at each step with the push added at the last position each time. ONNX Runtime supplies the residual and the patch is computed in TypeScript; the tests pin these continuations and check that every choice is wider than the difference between the two.",
    ],
    controls: [
      "Pick a feature, move the strength, and compare the `After` list with `Before` and with the control. Move `Random control direction` to see that the control is one draw among 20.",
      "Worth running: #160 at position 43, strength 10. A hyphen goes from 1.3% to 43.0% and becomes the top choice; its log probability rises by 3.53, the 20 random directions move it by −0.53 to +1.36 (6.9 SDs), and the continuation is a row of hyphens. The random control writes ` where is no`.",
      "Worth running: #683, which fires on y. At strength 5 the probability of y falls from 0.55% to 0.14% (−1.38, −3.8 SDs) and the largest rise is an o (+1.36). The y feature does not write y.",
    ],
    notice: [
      "A label can predict the effect: at strength 5, #160's direction raises a hyphen's log probability more than any of the 1,024 decoder directions (+2.09; the next, #563, gives +1.68).",
      "A label can predict the wrong one: #683 ranks 1,011th of 1,024 for raising a y, and the directions that raise y most are #167, #902 and #737, whose cached examples are the letters b, m and t. A label says where a feature fires, not what adding it writes.",
      "An effect can vanish into the control: #122, which fires on m, changes log p(m) by +0.39 at strength 5, against a random maximum of +0.37 (1.6 SDs). That cannot be told apart from adding any vector of that length.",
      "The space feature #121 changes the log probability of a space by −0.10, yet it lowers a line break by 1.87 and raises c, p and v: its effect is elsewhere. And random directions rewrite the continuation too (` what we wil` at strength 5); the difference is a systematic pattern, not whether the text changes.",
    ],
    limits: [
      "In this lab: the SAE reads the stream after the last block, so the push can only act through the final normalization and the output matrix. It cannot change what a later layer computes, because there is none, and it touches one position at a time.",
      "In this lab: the five features were picked because their labels measured cleanly (precision of 96% or better and recall of 90% or better on 768 held-out characters), not at random; most of the 1,024 have no such label. Twenty random directions are a rough, easy null: a harder one would use other features' directions.",
      "In general: production steering adds a vector at a middle layer, where every later layer processes it, and is judged on held-out prompts. Anthropic amplified a Golden Gate Bridge feature in Claude 3 Sonnet and its replies began to mention the bridge even when irrelevant, yet AxBench (Wu et al., 2025) found SAE steering not competitive with prompting, while a later paper (Jørgensen and Hansen, arXiv 2605.31183) reports a supervised feature-selection pipeline that gets close to its LoRA reference. This card is not a validation of steering.",
    ],
  },
};

export default cardInfo;
