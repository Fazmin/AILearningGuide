import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Generation loop": {
    title: "Predict, draw, append, repeat",
    summary:
      "The autoregressive loop run by real character models. The selected model predicts the next character from the text so far, the sampler draws one with a seeded random number, the character is appended, and the longer text is fed back in for the next prediction.",
    whatYouSee: [
      "A Prompt box and a tape of every character so far. Spaces show as ␣ and newlines as ↵; sampled characters are boxed and bold.",
      "Two buttons under the prompt, Unseen line (default) and Line from the training text, and a Seen in training? line. The default prompt is not in Tiny Shakespeare; the other button loads a line that is.",
      "Under each character, a bar whose height is its surprisal under the selected model, −log2 p from 0 to 10 bits, measured teacher-forced on the text as it stands. Characters the model cannot see are faded (all but the last for the bigram, all but the last 64 for the transformer).",
      "A dashed slot at the end showing where the next draw would land with the current seed and settings.",
      "Buttons Sample next token, Sample 10 more, Undo, and Clear continuation; metrics for the last draw; and a log of the five most recent draws.",
    ],
    howItWorks: [
      "One token is one character from a 66-symbol vocabulary: the 65 characters of Tiny Shakespeare plus <unk>. Characters outside it become <unk>.",
      "Each draw uses u = a deterministic hash of (Seed, draw number) in [0, 1) and picks the first kept character whose running sum of q passes u. The same prompt, seed and settings always reproduce the same text.",
      "Last draw records p, the model's probability at T = 1 with nothing cut, and q, the probability the sampler actually used. Surprisal is −log2 p; Mean over draws averages it and ppl = 2 to that power.",
      "Seen in training? is not a search. The corpus is not shipped with the app, so the lab recognises only its two preset lines (both checked against the corpus offline) and says Unknown for anything else you type.",
    ],
    controls: [
      "Prompt (up to 120 characters; editing it clears the continuation), the two preset buttons, Sample next token, Sample 10 more, Undo, Clear continuation.",
      "Comparison worth running: load each preset line and compare the transformer's Mean surprisal on this text. Then, with the RNN, set Top-k to 1 on the default prompt and sample 30 characters: it repeats itself, and the lab does not stop it.",
    ],
    notice: [
      "Low temperature produces text the model itself finds unsurprising; high temperature produces text it finds surprising. Neither is a measure of truth.",
      "A drawn character is never revisited. An unlikely draw stays in the tape and conditions everything after it.",
      "The text ends only at the 120-character cap or when you stop. The model never signals that it is finished.",
    ],
    limits: [
      "In this lab: character-level models trained on Tiny Shakespeare, about 1.1 million characters, so continuations are Shakespeare-flavoured spelling, not sense. The vocabulary has no end-of-sequence token, so nothing but the 120-character cap ends a continuation.",
      "In general: production models predict subword tokens from vocabularies of tens of thousands to a few hundred thousand entries, condition on thousands of tokens, and are trained on trillions of tokens. The loop is the same, but it ends when the model draws an end-of-sequence token, reaches a maximum length, or the output contains a stop sequence the caller chose.",
    ],
  },

  "Next token distribution": {
    title: "The model's odds, before and after the sampler",
    summary:
      "The selected model's real next-character distribution for the text on the tape, next to the distribution the sampler actually draws from after temperature, top-k, top-p and min-p. The strip below lays the kept characters end to end so you can see where the seeded number u lands.",
    whatYouSee: [
      "A status line that reports loading, running, or failure for the selected model, and which runtime ran it.",
      "The 10 most likely next characters. The outlined bar is p, the model's probability at T = 1; the filled bar is q, what the sampler uses. Cut characters read cut · top-k, cut · top-p or cut · min-p. A final row sums the other 56.",
      "An inverse-CDF strip of the kept characters with a marker at u; the highlighted segment is the next draw.",
      "Metrics Model entropy, Sampler entropy, Kept, and Kept mass.",
    ],
    howItWorks: [
      "Bigram: logits are the logarithms of the row of bigram.json for the last character. RNN: character-rnn.onnx, a two-layer GRU, runs on the whole text. Transformer: the attention lab's tiny-transformer.onnx runs on the last 64 characters, its trained block size. Both ONNX models run in a background worker.",
      "p = softmax(z). q = softmax(z / T) restricted to the kept set and renormalised. Entropy = −Σ p log2 p in bits. Kept mass is the tempered probability of the kept set before renormalising.",
      "Bars are scaled so the longest shown value fills the track; the percentages are exact.",
    ],
    controls: [
      "This card has no controls of its own. Model, Temperature, Top-k, Top-p, Min-p and Seed live on Sampler controls.",
      "Comparison worth running: on the default prompt, Bigram puts U first (33%) because it only sees J, and 125 of the corpus's 320 capital Js are followed by U, the start of the speaker name JULIET. The RNN, which sees the whole line, puts o first (26%). The Transformer puts u first (99%).",
    ],
    notice: [
      "The order of the outlined p bars never changes with temperature, top-k, top-p or min-p. Only the filled q bars move.",
      "A confident step has low model entropy and top-p keeps few characters; an uncertain step, such as right after a space, keeps many.",
    ],
    limits: [
      "In this lab: the full 66-entry distribution is computed, and the top 10 are drawn. Models are tiny: the GRU trained for 240 steps and the transformer for 8,000. If the worker cannot run ONNX, the RNN and transformer report an error and the bigram still works.",
      "In general: probabilities are over tokens, not meanings, and a model's probability is not a calibrated statement that a continuation is true.",
    ],
  },

  "Sampler controls": {
    title: "Reshaping real logits before the draw",
    summary:
      "The model produces logits; decoding decides how to draw from them. Temperature rescales the logits, top-k, top-p and min-p cut the tail, and the seed fixes the random numbers. None of these changes what the model computed or the order of its candidates.",
    whatYouSee: [
      "Model: Bigram, RNN, Transformer.",
      "Temperature 0.10 to 2.00, Top-k 1 to 66 (66 reads off, 1 reads greedy), Top-p 0.05 to 1.00 (1.00 reads off), Min-p 0.00 to 0.50 (0.00 reads off) and Seed 1 to 64.",
      "A formula row with the top candidate's logit z, z / T, and its final q. When Min-p is on, a second row shows its cut-off: p_min times the top character's probability, and how many characters clear it.",
    ],
    howItWorks: [
      "Order of operations in this lab: divide logits by T; keep the k highest; of those, keep the smallest most-likely prefix whose renormalised mass reaches p; drop any survivor whose probability is below p_min times the top probability; renormalise; draw by inverse CDF.",
      "Min-p keeps a character when its tempered probability is at least p_min × p_max (Nguyen et al., ICLR 2025, section 3.1). The test is a ratio to the top character, so renormalising does not change it, and the top character always passes.",
      "Temperature divides every logit, so it scales the gaps between them: below 1 the distribution sharpens, above 1 it flattens toward uniform. Rescaling cannot reorder.",
      "Seed feeds the hash that makes u for each draw number; changing it changes future draws, not past ones.",
    ],
    controls: [
      "Model, Temperature, Top-k, Top-p, Min-p, Seed.",
      "Comparison worth running: Min-p 0.10 with the RNN keeps about seven characters at Temperature 1.00 and about nineteen at 2.00, because a flatter distribution leaves more characters above a tenth of the top one. With the Transformer on the default prompt it keeps one at every temperature.",
      "Temperature 0.10 against 2.00 with nothing cut. Then set Top-k to 1: every draw is the top character, the pick a very low temperature makes almost every time.",
    ],
    notice: [
      "Top-p and min-p both adapt to confidence. At Top-p 0.50 the RNN keeps one or two characters in the middle of a word and about six right after a space.",
      "Lower temperature is lower variance, not higher accuracy. If the top candidate is wrong, it becomes reliably wrong.",
    ],
    limits: [
      "In this lab: there is no repetition penalty, stop sequence or beam search; those are described in the lesson only. Min-p is applied last, after top-k and top-p.",
      "In general: temperature 0 is not perfectly reproducible on real hardware, because floating-point reduction order can flip near-ties. Truncation settings also interact with temperature, as the Min-p comparison above shows.",
    ],
  },

  "Three models, one context": {
    title: "What each wiring sees, and how well it was trained",
    summary:
      "All three models read the same 66-symbol vocabulary and come from the same Tiny Shakespeare text. The bigram sees one character, the GRU sees everything through a fixed-size state, and the transformer attends directly to the last 64. The columns compare their next-character guesses and their measured loss, and the training behind each number.",
    whatYouSee: [
      "One column per model, the selected one outlined: what it sees, its top three next characters, the entropy of its next-character distribution, and its mean surprisal on the tape's text.",
      "Held-out loss and Whole-corpus loss, in bits per character, from language-models.metadata.json: consecutive non-overlapping 64-character windows over the last 10% of the corpus and over all of it. The note under each says what that model trained on.",
      "Size and training: 66 × 66 table entries, the parameter counts from the metadata file, and the number of training steps.",
    ],
    howItWorks: [
      "Mean surprisal on this text = the average of −log2 p(character | preceding characters) over every character the model can score, computed from one forward pass (teacher forcing).",
      "Held-out loss is the same average over the last 10% of the corpus. The transformer trained only on the first 90%. The GRU trained on random windows of the whole corpus for a third of an epoch, so its tail figure is nearly, not strictly, held out. The bigram figure comes from a table fitted on the first 90%; the table in this lab was fitted on all of it, so its whole-corpus figure is the one that describes it.",
      "Nats convert to bits by multiplying by log2 e ≈ 1.443.",
    ],
    controls: [
      "This card has no controls. Edit Prompt or generate text to change the context; Model on Sampler controls chooses the outlined column.",
      "Comparison worth running: load Line from the training text and back to the default, and compare each model's Mean surprisal. Then read the Held-out and Whole-corpus rows for the transformer.",
    ],
    notice: [
      "On held-out text the transformer scores lowest, the GRU next and the bigram highest. On the whole corpus the order is the same.",
      "The transformer's whole-corpus figure is lower than its held-out figure because 90% of the corpus is text it trained on. The GRU and bigram figures barely differ between the two.",
    ],
    limits: [
      "In this lab: the transformer trained for 8,000 steps and the GRU for 240, so the ordering is not a comparison of architectures. The lab cannot say how much of the gap is wiring and how much is training. The transformer's training also replaced about 30% of each batch with synthetic repeated strings, for the attention and circuits labs.",
      "In general: at scale, transformers beat recurrent nets and n-gram tables by a wide margin, largely because attention trains in parallel over long contexts. Comparing architectures fairly needs matched training budgets and text none of the models trained on.",
    ],
  },

  "Layer by layer": {
    title: "The whole model, read one stage at a time",
    summary:
      "The path from characters to one sampled character: ids, embeddings plus position, two transformer layers, a final norm and unembedding, softmax, a draw. The logit lens applies the model's own last two steps to the stream after the embeddings, after layer 1 and after layer 2, so you can see what each stage would predict.",
    whatYouSee: [
      "A numbered strip of the seven steps with their sizes. Three carry a lens reads here tag: after the embeddings, after layer 1 and after layer 2.",
      "Three panels, one per tap. Each lists the five most likely next characters with their probabilities as text and as bars (all panels share one scale, so a full bar is 100%), the entropy of that readout in bits, and the rank and probability it gives the character the finished model ranks first.",
      "A sentence and four metrics: the finished model's first choice, the first stage at which it heads the list, the entropy at each stage, and how many characters the model read.",
    ],
    howItWorks: [
      "The page runs the transformer in plain TypeScript on the weights read out of tiny-transformer.onnx, on the last 64 characters of the tape. The residual stream at the last position is taken after the embeddings (token embedding plus position embedding) and after each layer.",
      "Each tap goes through the same readout as the model's own output: final LayerNorm, then multiplication by the 256 × 66 unembedding matrix, which has no bias. Softmax turns the 66 logits into probabilities and entropy = −Σ p log2 p.",
      "The last tap equals the model's output logits to rounding error, so the third panel is the answer the Transformer column of Three models, one context would give.",
    ],
    controls: [
      "This card has no controls of its own. Edit Prompt, press Sample next token, or use the preset buttons; the lens reads the last character of whatever is on the tape.",
      "Comparison worth running: on the default prompt, note where u ranks in each panel. Then change both Juliets to Jason and see which character layer 2 puts on top.",
    ],
    notice: [
      "On the default prompt the entropy is about 4.9, 4.5 and 0.1 bits: nearly all of the drop is in layer 2, and the answer first heads the list there. The pattern is not always so sharp. For the prompt “To be, or not to be, that is the question” the readout after layer 1 is less certain than after the embeddings.",
      "The after-embeddings panel depends on the last character and its position alone, so it cannot depend on the earlier words. Anything the finished model does with those words happens in the layers.",
    ],
    limits: [
      "In this lab: only three taps, not each attention head or MLP step, and only the last position. The weights are the shipped 2-layer, 256-wide model. The finished model's choice is one character from one prompt, so a pattern across stages here does not show how the model works in general.",
      "In general: the logit lens applies a readout that was trained for the last layer to earlier states, which were never trained to be read that way, and it shows what is linearly readable, not what caused it. Its author (nostalgebraist, 2020) found early layers can look like nonsense, and Belrose et al. (2023) call their tuned lens a refinement of it because the plain lens is often brittle. Real models have dozens of layers.",
    ],
  },
};

export default cardInfo;
