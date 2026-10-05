import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Model attention routing diagram": {
    title: "Where one token retrieves context",
    summary:
      "An arc diagram of one attention head: the selected query word is joined to every word it reads from, and arc weight is the softmax attention it assigns. The weights come from a real two-layer character transformer running in a worker, summed from characters up to words.",
    whatYouSee: [
      "The sentence laid out left to right as word chips; each chip is one word the head can read from.",
      "Arcs from the highlighted query chip to every other chip. A thicker, more opaque arc means a larger share of that query's attention. The bar under each chip repeats the share, including the query's weight on itself.",
      "The kicker reads `Layer N · head N`, so the picture is one head of one layer, not the whole model.",
      "A status badge: `ONNX model` with the execution provider once weights load, `Loading teaching model…` while they run, or `Model unavailable · inspectable fallback` if they fail.",
      "Example buttons that load prepared inputs: the pronoun sentence, a name repeated later in a sentence, the same sentence with the name moved earlier, and two repeating strings for the Characters view.",
      "A readout strip naming the query, the inspected key, the strongest route, its weight, and the share of the row that lands on space characters. With Mask set to Bidirectional it adds the share on later characters and on the very next character.",
    ],
    howItWorks: [
      "Your sentence is split into characters and looked up in the module's vocabulary file, capped at 64 characters. Unknown characters map to the vocabulary's unknown id.",
      "Those ids run through the tiny transformer, which returns query, key, and value tensors plus the attention probabilities for every layer and head.",
      "The model attends over characters. A word's row is read at its last character, the position that has seen the whole word under causal masking.",
      "Each word owns the whitespace in front of it. With that rule every character a word's row can see belongs to that word or an earlier one, so summing the row per word keeps its total at exactly 100% with nothing dropped or renormalized.",
      "Weights are a softmax over scaled dot products, so every row sums to 100% and routes compete: one arc can only thicken if others thin.",
    ],
    controls: [
      "Edit the sentence to change the token set. Doing so resets the query, the inspected key, and the stage to the start.",
      "Press an Examples button to load a prepared input and its query. `Repeated name` and `Name moved earlier` select the second `Mark`; the two `Repeats every` buttons switch the matrix to Characters. An example also puts Mask, Score scaling, and Value vectors back to the model's own settings, so the figures in the lesson apply.",
      "Click any word chip to make it the query and redraw all arcs from it.",
      "Step through Query → Keys → Score → Softmax → Mix to highlight each part of the calculation in the inspector; the module's step navigation moves this too.",
    ],
    notice: [
      "Attention in this decoder is causal: a query reads earlier characters and itself, so arcs never point to later words. Set Mask to Bidirectional in the matrix card and they can.",
      "No head's strongest route from `it` is `animal`. Layer 1 heads 3 and 4 send 99.9% of `it` to the word `it` itself, because they read the character just before the query and that character, the `i`, sits inside `it`.",
      "The On spaces share needs reading. For `it`, Layer 1 head 2 puts 99.1% on spaces, but only because it reads two characters back and `it` has two letters. For `because` the same head puts 1.3% there. Layer 2 head 4 puts 61.7% on spaces for `it`.",
      "Layer 2 head 3 sends the second `Mark` 92.7% of its weight to `Jones`, the word after the earlier `Mark`. Press `Name moved earlier` and it is 97.2% on `Jones` again, although the earlier name is now 19 characters back instead of 34: the route follows the content, not the distance.",
    ],
    limits: [
      "In this lab: the model is a tiny teaching transformer trained for 8,000 steps on the first 90% of Tiny Shakespeare, about 23 passes over it, with a validation loss of 1.54 nats per character on the held-out last 10%. Its routes are far noisier than a production model's. While the model loads or if it fails, every weight comes from a deterministic sine pattern with no learned meaning; the badge and a note in the matrix card say so.",
      "In this lab: word totals are sums over characters, not a real tokenizer's tokens, and input is truncated at 64 characters.",
      "In this lab: which head does what is read from patterns and from switching heads off. Layer 2 heads 2 and 3 behave as induction heads and Layer 1 heads 3 and 4 as previous-token heads, but the wiring between them was not tested by path patching, and Layer 1 head 1 and Layer 2 heads 1 and 4 are not explained at all.",
      "In general: a large attention weight shows where information could flow, not that the model used it. Attribution needs interventions such as the activation patching in Interpretability II, not attention maps alone.",
    ],
  },

  "Attention matrix": {
    title: "Every query–key route at once",
    summary:
      "The full attention matrix for the selected layer and head, as words or as the characters the model actually computes over, with a Mask control that lifts the causal mask for that head. Each row is one query's distribution over keys and sums to 100%.",
    whatYouSee: [
      "Words view: a grid whose rows are query words and columns are key words. Cell shade encodes the weight; hover, focus, or select a cell to print it.",
      "Characters view: the model's own character-by-character grid. Darker cells carry more weight on a linear scale, hatched cells are masked, the query character's row is outlined, and the inspected word's columns are shaded with a dashed outline. Axis ticks mark every 8th position. With Mask set to Bidirectional nothing is hatched.",
      "A dashed note when the grid shows stand-in values instead of model output, and another when the mask is lifted, saying what that view is and is not.",
      "A `View weights as text` disclosure listing the current query's row, by word or by character, so the matrix never depends on colour alone.",
    ],
    howItWorks: [
      "Causal mask: the Characters view draws the model's attention tensor for this layer and head directly: row i is softmax(q_i · k_j ÷ √64) over j ≤ i.",
      "Bidirectional mask: the lab recomputes the grid from the same query and key tensors with softmax(q_i · k_j ÷ √64) over every j, so each row now also reads the characters after position i. Every earlier weight is multiplied by one minus the weight that moved to later characters, so the ratios among earlier keys do not change.",
      "Words view sums each character row, read at a word's last character, over the character spans of each word, where a word owns the space before it. A causal row sums to exactly 100%; a lifted row can lose weight that falls on trailing spaces or past the first 12 words, and the note under the grid says how much.",
      "Rows are normalized independently. Compare across a row, not down a column.",
      "Selecting a word cell sets both the query and the inspected key and jumps the stage to Score, so the inspector shows that pair's arithmetic.",
    ],
    controls: [
      "`Matrix view` switches between Words and Characters.",
      "`Mask` switches between Causal (the model as trained) and Bidirectional (this head only, from its own query and key vectors). Compare the same row both ways: weight appears above the diagonal and the cells below it shrink together.",
      "Click a word cell to inspect that query–key pair.",
      "Change Layer or Attention head in the inspector to redraw the whole matrix. `Score scaling` in the inspector redraws the grid too.",
    ],
    notice: [
      "The empty upper triangle is the causal mask: a query cannot read later positions. Lift it and the share a head puts on later characters depends on the head. For `it`, Layer 1 heads 3 and 4 move 0.1% and 0.2% there, Layer 2 head 4 moves 90.2%.",
      "With the mask lifted, Layer 2 heads 2 and 3 put 33.8% and 30.6% of `it` on the very next character. In next-token training that character is the answer, which is why a decoder trains with the mask on.",
      "With `Repeats every 8`, Layer 2 heads 2 and 3 show stripes 7, 15 and 23 positions below the diagonal, where the next character of the pattern sits. With `Repeats every 9` those stripes move to 8, 17 and 26, so they follow the period. The Layer 1 stripes stay at 3, 2, 1 and 1 positions back, so they track distance.",
      "Compare Layer 1 and Layer 2 on the same sentence: Layer 1 heads 2, 3 and 4 draw thin lines at a fixed distance, while Layer 2's patterns differ because layer 2 reads vectors that layer 1 already changed.",
    ],
    limits: [
      "In this lab: the shipped model was trained with the causal mask on. Bidirectional shows what this head would score if it were allowed to read later characters, a pair it was never trained to score, so it is what a bidirectional layer would be allowed to see here, not a trained bidirectional model. Layer 1 is exact for that head; in Layer 2 the queries and keys were still built from a Layer 1 that stayed causal. Only the selected head changes, never the rest of the network.",
      "In this lab: after you edit Q or K in the inspector, only the selected query's word row is recomputed from your vectors; every other row keeps the model's weights, and the character grid shows the model's weights, or the same head recomputed with the mask lifted or the scaling off, never your edits.",
      "In this lab: two layers and four heads is far smaller than a production model's dozens of each, and the stripes you see come from a character model trained on one author's plays.",
      "In general: reading a matrix as the model's reasoning over-reads it. A stripe that moves with the period shows a head follows content, not that the output uses it. Heads compose across layers, and a single head's map is one term in a much larger computation.",
    ],
  },

  "Query key value inspector": {
    title: "The arithmetic behind one weight",
    summary:
      "The scaled dot-product attention calculation for the selected query and key characters, from raw score through scaling and softmax to the mixed value vector, recomputed from the model's own Q and K and checked against the weight the model reports. Two switches test its claims: scaling off, and zeroed values.",
    whatYouSee: [
      "Layer and Attention head controls that pick one of the model's eight heads (two layers of four).",
      "Two editable vector chips: Q for the last character of the query word and K for the last character of the inspected word, showing the first 4 of their 64 dimensions.",
      "A strip with one bar per character: the scores softmax sees above a zero line, softmax weights below, the inspected word's characters outlined, and later positions hatched as masked while the mask is on.",
      "A `Score scaling` control with `÷ √64` and `No scaling`, and formula rows: `q · k`, the same score divided by √64 (or not), the softmax share of that one character with the model's own value beside it, and the word total that the heatmap cell shows.",
      "A `Value vectors (V)` control with `Model`, `Zero word` and `Zero all`, a read-only V chip for the inspected key, and a `Mixed value` chip: the first 4 of 64 dimensions of the weighted sum of value vectors, which this head passes to the output projection.",
      "Two metric rows: the pair's score, the word's weight and the share of the row on spaces; then the largest weight in the row, how many rows of the whole grid put 99% or more on one character, and the length of the mixed value. The current stage is ringed.",
    ],
    howItWorks: [
      "Score is the 64-dimensional dot product of the two vectors. Scale divides it by √64 = 8, unless Score scaling is `No scaling`, in which case softmax sees the raw score.",
      "Softmax exponentiates every visible scaled score, subtracting the row maximum for stability, and divides by their sum. Without edits and with the model's settings, the recomputed row matches the model's attention tensor to within about 1e-6.",
      "Word total sums that softmax row over every character of the inspected word, including its leading space.",
      "Mixed value is Σ weight_j · v_j over every visible character j, using the head's real value vectors, or zero vectors where Value vectors says so. The weights are computed from Q and K before any value is read, so zeroing V cannot change them.",
      "Editing a chip replaces the first four components with your numbers and recomputes the row, the word weights, and the mix from them.",
    ],
    controls: [
      "Layer and Attention head choose the head and reset the stage walkthrough.",
      "Type into the Q or K chips to edit the first four dimensions; the remaining 60 learned dimensions stay fixed.",
      "`Match the model vectors` shows the model's own values again. Your edits are kept, and they come back if you type in a chip.",
      "`Score scaling`: switch to No scaling and compare Largest weight and Saturated rows. For Layer 2 head 2 on the pronoun sentence, no row reaches 99% on one character with the division and 26 of 55 rows do without it.",
      "`Value vectors (V)`: set Zero all and watch every weight stay exactly where it was while the Mixed value falls to zero. Zero word removes only the inspected word's values, including its leading space, and the mix loses exactly that word's share.",
    ],
    notice: [
      "Raising one dimension of Q where the same dimension of K is positive increases the score; opposite signs decrease it. Alignment, not magnitude alone, drives attention.",
      "A single character often gets a tiny share while its word total is large, because the weight is spread across the word's characters.",
      "Push a score up and the other weights shrink, but the strip draws each weight relative to its tallest bar, so read the arcs or the percentages rather than bar heights. Softmax is relative, so the other keys' scores matter as much as this one.",
      "With the division off, the raw scores are 8 times larger and the softmax saturates: the strip's weight bars collapse toward one character and Saturated rows rises. The weights moved, the values did not.",
      "With the values zeroed, the strip does not change at all. Q and K say who is read; V is what is read.",
    ],
    limits: [
      "In this lab: only the first four of 64 dimensions are editable, so you perturb a slice of the real vector rather than steering the head.",
      "In this lab: edited values are your own, not the model's, so the arcs and the edited word row show a hypothetical head until you reset.",
      "In this lab: No scaling multiplies this trained head's scores by 8 at inference. It shows saturation in a model trained with the division, not what training without it would do; the original Transformer paper suspected that large unscaled dot products push softmax into regions with extremely small gradients, which this lab does not measure.",
      "In this lab: Zero word and Zero all change the lab's mix for this one head. They do not run the zeroed head through the output projection and later layers, so they show what V carries, not what the model would predict.",
      "In this lab: the formulas stop before the output projection, residual addition, and layer normalization, which sit between this mix and the block's output. See The transformer block.",
      "In general: production heads are wider and often share keys and values across query heads with grouped-query or multi-query attention, so one K per head, as drawn here, simplifies how keys are stored at inference.",
    ],
  },
};

export default cardInfo;
