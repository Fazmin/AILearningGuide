import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { DEFAULT_PROMPT, TRAINING_LINE } from "./prompts";
import { isModelId, MAX_GENERATED, MAX_PROMPT, MIN_P_MAX, VOCAB_SIZE } from "./sampling";

const initialState: ModuleState = {
  prompt: DEFAULT_PROMPT,
  temperature: 0.8,
  model: "rnn",
  sample: 2,
  topK: VOCAB_SIZE,
  topP: 1,
  minP: 0,
  generated: "",
  draws: [],
};

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const clampNumber = (value: unknown, fallback: number, low: number, high: number) =>
  finite(value) ? Math.min(high, Math.max(low, value)) : fallback;

/**
 * Version 1 stored { prompt, temperature, model, sample } where `sample` was an
 * ever-increasing draw counter over six authored logits. Version 2 keeps the
 * prompt, temperature and model choice, reads `sample` as the seed (folded into
 * 1–64), and adds top-k, top-p and the generated continuation. Version 3 adds min-p
 * and changes the default prompt: the earlier default was a line from the training
 * text, which made the models look more certain than a fair comparison allows. A
 * payload with no min-p that still holds that unedited line was never the learner's
 * choice, so it moves to the new default and drops the continuation drawn from it.
 */
export function hydrateNextToken(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
    const older = !("minP" in parsed);
    const stored = typeof parsed.prompt === "string" ? parsed.prompt.slice(0, MAX_PROMPT) : DEFAULT_PROMPT;
    const retired = older && stored === TRAINING_LINE;
    const prompt = retired ? DEFAULT_PROMPT : stored;
    const generated =
      !retired && typeof parsed.generated === "string"
        ? Array.from(parsed.generated).slice(0, MAX_GENERATED).join("")
        : "";
    const draws =
      generated && Array.isArray(parsed.draws)
        ? parsed.draws.filter((item): item is string => typeof item === "string").slice(-Array.from(generated).length)
        : [];
    const rawSeed = finite(parsed.sample) ? Math.round(parsed.sample) : 2;
    return {
      prompt,
      temperature: Math.round(clampNumber(parsed.temperature, 0.8, 0.1, 2) * 100) / 100,
      model: isModelId(parsed.model) ? parsed.model : initialState.model,
      sample: rawSeed >= 1 ? ((rawSeed - 1) % 64) + 1 : 2,
      topK: Math.round(clampNumber(parsed.topK, VOCAB_SIZE, 1, VOCAB_SIZE)),
      topP: Math.round(clampNumber(parsed.topP, 1, 0.05, 1) * 100) / 100,
      minP: Math.round(clampNumber(parsed.minP, 0, 0, MIN_P_MAX) * 100) / 100,
      generated,
      draws,
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-07-next-token-prediction",
  slug: "next-token-prediction",
  title: "Next-token prediction",
  group: "inside-models",
  order: 15,
  icon: "TextCursorInput",
  accent: "#4974d1",
  prerequisites: ["module-01-tokens-embeddings", "module-06-training-dynamics"],
  estimatedMinutes: 20,
  steps: ["Read the context", "Compare models", "Shape probabilities", "Sample a token", "Read the layers"],
  stepInstructions: [
    "With Model on Bigram, edit the start of Prompt but keep the final J: change both Juliets to Jason. The Next token distribution does not move, because the bigram sees only the J. Switch Model to RNN: it does not move either, because its memory of text this far back has faded. Switch to Transformer: its top character changes from u to a.",
    "On the default prompt, switch Model among Bigram, RNN, and Transformer. Their top next characters differ (U, o, u); compare Entropy, model and the columns in Three models, one context. Read the Held-out loss row: the transformer scores lowest, but it trained for far more steps than the GRU.",
    "Move Temperature from 0.1 to 2 and watch q sharpen and flatten while the order stays fixed. Then lower Top-k to 3, Top-p to 0.50 and raise Min-p to 0.10: cut rows read cut · top-k, cut · top-p or cut · min-p, and the kept q values renormalise.",
    "Press Sample next token a few times, then Sample 10 more; each drawn character is appended and conditions the next distribution. Then clear the continuation, set Top-k to 1 and sample about 30 characters with the RNN: greedy decoding falls into a loop, and nothing in the lab ends the text before 120 characters.",
    "In Layer by layer, read the three panels for the last character: compare each panel's entropy and where the finished model's choice ranks. It first heads the list after layer 2. Then change both Juliets in Prompt to Jason and watch which character layer 2 puts on top.",
  ],
  stateVersion: 3,
  tagline:
    "Run three real character models, read what each layer of the transformer would predict, reshape the next-character odds with temperature, top-k, top-p and min-p, and sample one character at a time, feeding each draw back in as context.",
  objectives: [
    "Describe autoregressive generation, including what ends it: an end-of-sequence token, a length limit or a stop sequence",
    "Explain temperature, top-k, top-p and min-p as reshaping the model's odds, without calling it creativity",
    "Read surprisal and cross-entropy as the model's own measure of how expected a token was, and compare models only on text they did not train on",
    "Trace one prediction from characters to probabilities, and read what each stage of the transformer would predict with a logit lens",
  ],
  glossary: [
    {
      term: "Logit",
      definition:
        "An unnormalized score the model produces for one vocabulary entry. There is one logit per token, and only their differences matter after softmax.",
    },
    {
      term: "Softmax",
      definition:
        "The function turning logits into positive probabilities that sum to one: the next-token distribution, one probability for every vocabulary entry at the next position. Because they sum to one, raising one candidate necessarily lowers others.",
    },
    {
      term: "Temperature",
      definition:
        "A divisor applied to logits before softmax. Below one it sharpens the distribution, above one it flattens it, and it never changes the ranking. At the limit it becomes greedy decoding, always taking the top token: repeatable, and reliably wrong when the top candidate is wrong.",
    },
    {
      term: "Autoregressive",
      definition:
        "Generating a sequence one element at a time, appending each choice to the context before predicting the next. Earlier choices become fixed premises, and the loop needs a rule for when to stop.",
    },
    {
      term: "Top-k sampling",
      definition:
        "Restricting the draw to the k highest-probability tokens and renormalizing, which cuts off the long tail with a fixed candidate count.",
    },
    {
      term: "Top-p sampling",
      definition:
        "Also called nucleus sampling (Holtzman et al., 2020): keep the smallest set of top tokens whose probability reaches p, so confident steps keep few candidates and uncertain steps keep many.",
    },
    {
      term: "Min-p sampling",
      definition:
        "Keeping the tokens whose probability is at least p_min times the top token's probability (Nguyen et al., ICLR 2025), then renormalizing. The cut-off rises when the model is confident and falls when it is not, because each token is compared only with the top one.",
    },
    {
      term: "Repetition penalty",
      definition:
        "A penalty on the logits of tokens already in the context, applied before the draw. It works against the loops that greedy and low-temperature decoding fall into, at the cost of also discouraging repetition that is correct, such as a name that has to recur.",
    },
    {
      term: "End-of-sequence token",
      definition:
        "A special vocabulary entry that marks where a text ends; drawing it stops generation. Hosts also stop at a maximum length and at stop sequences, strings the caller chooses. This lab's vocabulary has no such token, so only the length cap ends a continuation.",
    },
    {
      term: "Surprisal and cross-entropy",
      definition:
        "Surprisal is the negative base-2 logarithm of the probability the model gave the token that actually came, in bits: 1 bit is a coin flip, 6 bits about 1 in 64. Cross-entropy is its average over real text, the training loss, and 2 to that power is perplexity, the effective number of equally likely choices. Lower means more predictable, not true, and the comparison is only fair on text neither model trained on.",
    },
    {
      term: "Bigram model",
      definition:
        "A model conditioning only on the single previous token: its entire context is one row of a count table. The simplest baseline, and the reference for what a longer context adds, since nothing before the last token can change its answer.",
    },
    {
      term: "Residual stream",
      definition:
        "The vector each position carries through the network. It starts as the token embedding plus a position embedding, every attention and MLP step adds its output to it, and the last state is what the model reads out as logits.",
    },
    {
      term: "Unembedding",
      definition:
        "The model's last step: a final normalization, then a matrix that turns one position's residual stream into one logit per vocabulary entry. In this transformer it is a 256 by 66 matrix with no bias.",
    },
    {
      term: "Logit lens",
      definition:
        "Applying the final normalization and unembedding to the residual stream after an earlier layer, to see what the network would predict if it stopped there. It shows what is linearly readable, not why, and early layers were never trained to be read this way.",
    },
  ],
  references: [
    {
      authors: "C. E. Shannon",
      title: "Prediction and Entropy of Printed English",
      source: "Bell System Technical Journal 30(1), 50–64",
      year: 1951,
      url: "https://www.semanticscholar.org/paper/c1e3f2d537e50e0d5263e4731ab6c7983acd6687",
      note: "People guess the next letter of a text while the earlier letters are known, and their guesses are used to measure the entropy of English. It is the same next-character game the lab's models play, scored in bits like the Surprisal bars.",
    },
    {
      authors: "Daniel Jurafsky and James H. Martin",
      title: "Speech and Language Processing, 3rd edition draft",
      source: "Stanford University, free to read online",
      year: 2026,
      url: "https://web.stanford.edu/~jurafsky/slp3/",
      note: "Three chapters follow this lab. N-gram Language Models builds a bigram from a table of counts, like the lab's Bigram, and ties perplexity to the cross-entropy behind Held-out loss. Transformers and Pretraining covers the residual stream, the unembedding, greedy decoding, temperature, top-k, top-p, and teacher forcing. Interpretability explains the logit lens and warns the network was not trained to be read that way, the caveat on the Layer by layer card.",
    },
    {
      authors: "Alec Radford, Jeffrey Wu, Rewon Child, et al.",
      title: "Language Models are Unsupervised Multitask Learners",
      source: "OpenAI technical report (GPT-2)",
      year: 2019,
      url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
      note: "The GPT-2 report. Its Equation 1 writes the probability of a text as a product of next-token probabilities, which the lesson calls the factorization, and notes that this lets one model both sample text and score it.",
    },
    {
      authors: "Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, et al.",
      title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
      source: "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), 1724–1734",
      year: 2014,
      url: "https://aclanthology.org/D14-1179/",
      note: "Introduces the recurrent unit with a reset gate and an update gate that GRU layers use. The lab's RNN is a two-layer GRU built from this unit.",
    },
    {
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.",
      title: "Attention Is All You Need",
      source: "Advances in Neural Information Processing Systems 30 (NIPS 2017)",
      year: 2017,
      url: "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
      note: "The paper that introduced the transformer: token embeddings plus position information, then stacked blocks of attention and feed-forward layers. The lab's Transformer is a small two-layer model of this kind.",
    },
    {
      authors: "Angela Fan, Mike Lewis, and Yann Dauphin",
      title: "Hierarchical Neural Story Generation",
      source: "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), 889–898",
      year: 2018,
      url: "https://aclanthology.org/P18-1082/",
      note: "Generates stories by sampling only from the 10 most likely next words at each step. This is the top-k cut that the Top-k control applies, with a fixed number of candidates.",
    },
    {
      authors: "Ari Holtzman, Jan Buys, Li Du, et al.",
      title: "The Curious Case of Neural Text Degeneration",
      source: "International Conference on Learning Representations (ICLR 2020)",
      year: 2020,
      url: "https://arxiv.org/abs/1904.09751",
      note: "Introduces nucleus (top-p) sampling, the Top-p control. It also shows that greedy decoding and beam search repeat far more than human text, and that each repeat of a phrase makes the next one more likely, the loop the lesson's greedy run falls into.",
    },
    {
      authors: "Minh Nhat Nguyen, Andrew Baker, Clement Neo, et al.",
      title: "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs",
      source: "International Conference on Learning Representations (ICLR 2025)",
      year: 2025,
      url: "https://arxiv.org/abs/2407.01082",
      note: "Defines min-p: the cut-off is p_base times the top token's probability, so it rises when the model is confident and falls when it is not. This is the rule the Min-p control and the cut · min-p rows follow.",
    },
    {
      authors: "Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, et al.",
      title: "CTRL: A Conditional Transformer Language Model for Controllable Generation",
      source: "arXiv preprint arXiv:1909.05858",
      year: 2019,
      url: "https://arxiv.org/abs/1909.05858",
      note: "Proposes penalized sampling, which lowers the scores of tokens already generated so that greedy decoding repeats itself less. Hugging Face's repetition_penalty, described in Going deeper, points to this paper.",
    },
    {
      authors: "Hugging Face",
      title: "Generation",
      source: "Hugging Face Transformers documentation (API reference, GenerationConfig)",
      year: 2026,
      url: "https://huggingface.co/docs/transformers/main_classes/text_generation",
      note: "Lists the real settings the lesson names: eos_token_id, max_new_tokens and stop_strings for ending the loop, and temperature, top_k, top_p, min_p and repetition_penalty for reshaping each draw.",
    },
    {
      authors: "Anthropic",
      title: "Stop reasons and fallback",
      source: "Claude Platform documentation",
      year: 2026,
      url: "https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons",
      note: "Explains the stop reasons a reply can end with: end_turn when the model finished on its own, max_tokens when it hit the length limit, and stop_sequence when it produced one of the caller's stop strings. These are the three ways to end the loop in Where generation stops.",
    },
    {
      authors: "Gregor Bachmann and Vaishnavh Nagarajan",
      title: "The Pitfalls of Next-Token Prediction",
      source: "Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235, 2296–2318",
      year: 2024,
      url: "https://proceedings.mlr.press/v235/bachmann24a.html",
      note: "Separates teacher-forced training from generation that feeds the model its own output, and examines the claim that errors compound during generation. It backs the lesson's point that training and generation differ.",
    },
    {
      authors: "Nora Belrose, Igor Ostrovsky, Lev McKinney, et al.",
      title: "Eliciting Latent Predictions from Transformers with the Tuned Lens",
      source: "arXiv preprint arXiv:2303.08112",
      year: 2023,
      url: "https://arxiv.org/abs/2303.08112",
      note: "Presents the tuned lens as a refinement of the logit lens and calls the logit lens often brittle, as the lesson's Going deeper says. Its figures compare what each layer would predict under both lenses.",
    },
  ],
  checkpoint: [
    {
      prompt: "You press Sample next token and the model draws the character a. What does it compute next?",
      options: [
        "The same distribution again, because the prompt itself has not changed",
        "A revised distribution for the character it just drew, so it can correct the a if it was unlikely",
        "A new distribution for the following character, conditioned on the text that now ends in a",
      ],
      answer: 2,
      explanation: "Generation is autoregressive: the drawn character is appended and becomes context for the next prediction. Nothing in the loop revisits an earlier draw, so an unlikely character stays and everything after it is conditioned on it.",
      objective: 0,
    },
    {
      prompt: "This lab keeps accepting characters until you stop or it reaches 120. Why can the model not end the text itself, the way a chat reply ends?",
      options: [
        "Models never choose when to stop: in real systems too the host alone decides, by cutting every text off at a fixed length limit",
        "The vocabulary has no end-of-sequence token for the model to draw, so only the length cap ever ends the text",
        "A character model cannot score a newline, so it has no way to mark where a line or a text finishes",
      ],
      answer: 1,
      explanation: "A real model can draw a special end-of-sequence token, and the host stops when it does. It also enforces a maximum length and watches for stop sequences, which the caller chooses. This vocabulary has 66 symbols and none of them means end, and a newline is an ordinary character, so only the length cap stops the lab.",
      objective: 0,
    },
    {
      prompt: "You drag Temperature from 0.1 up to 2 with nothing cut. What happens to the order of the candidates and to their bars?",
      options: [
        "The model grows more creative, so it adds candidates it never scored at low temperature",
        "The order never changes: the top bar shrinks while the long shots grow toward it",
        "The order changes, so the second choice can overtake the first at high temperature",
      ],
      answer: 1,
      explanation: "Temperature divides every logit by the same number before softmax, so it only stretches or squeezes the gaps between them. A larger gap sharpens the distribution and a smaller one flattens it, but the ranking and the set of scored candidates stay fixed. It reshapes odds the model already produced.",
      objective: 1,
    },
    {
      prompt: "With Top-p at 0.50, the sampler keeps one character at a confident step and several at an uncertain one. Why?",
      options: [
        "Top-p keeps the smallest top set whose probability reaches 0.50, so a confident step needs very few",
        "Top-p keeps a fixed number of characters, and the confident step has fewer that the model scored at all",
        "Top-p lowers the temperature at confident steps, which sharpens the odds until only one is left",
      ],
      answer: 0,
      explanation: "Top-p walks down the ranked characters and stops once their combined probability reaches p. When one character already holds most of the mass the set has one member; when the mass is spread out it takes many. Top-k, by contrast, keeps a fixed count, and neither one changes the temperature.",
      objective: 1,
    },
    {
      prompt: "With Min-p at 0.10, one step keeps a single character and another keeps seven. What decides how many survive?",
      options: [
        "Min-p keeps the same share of the ranked list at every step, and the share only looks different because the bars are rescaled",
        "Min-p turns the temperature down until the characters it wants to drop have no probability left",
        "The cut-off is a tenth of the top character's probability, so a confident step sets a high bar and an uncertain one a low bar",
      ],
      answer: 2,
      explanation: "Min-p keeps a character when its probability is at least p_min times the top character's. A confident step has a large top probability, so the bar is high and few characters clear it; an uncertain step has a small one, so many do. It compares each character with the top one and changes neither the temperature nor the order.",
      objective: 1,
    },
    {
      prompt: "On the tape one character has a very tall surprisal bar. What does that tell you?",
      options: [
        "The character is wrong, because high surprisal marks text that is false",
        "The selected model gave that character a low probability before it appeared",
        "The selected model was confident about that character, because a tall bar means a high probability",
      ],
      answer: 1,
      explanation: "Surprisal is the negative log of the probability the model gave to the character that actually came, so a tall bar means it was unexpected under that model. It measures expectation, not correctness: a rare but correct character is high surprisal, and a likely wrong one is low.",
      objective: 2,
    },
    {
      prompt: "In Three models, one context, the transformer's held-out loss is lowest and the GRU's is next. What does that show?",
      options: [
        "As trained here it predicts held-out text best, but the lab cannot separate its wiring from its longer training",
        "A transformer beats a GRU at any training budget, because attention reads every earlier character directly and a GRU cannot",
        "It will write more truthful text, because lower loss means the model has checked its claims against the corpus text",
      ],
      answer: 0,
      explanation: "Held-out loss is the cross-entropy on text the model did not train on, so lower means that text was more predictable to it. The transformer also trained for far more steps than the shipped GRU, so wiring and training are tangled together here. And loss ranks distribution quality, not truth.",
      objective: 2,
    },
    {
      prompt: "A continuation is fluent and the model was confident at every step, yet its claim is false. What does the loop's mechanism say about that?",
      options: [
        "Probability measures how expected a token is given the text so far, not whether the claim it makes is true",
        "This cannot happen, because a model confident at every step has checked the claim against its training text",
        "This only happens at high temperature, because low temperature always follows the true continuation",
      ],
      answer: 0,
      explanation: "Each step gives probabilities for the next token from the text so far. A false but common-sounding continuation can carry a high probability, and the sampler and the loop contain no step that checks facts. Low temperature makes the model's top guess reliable, including when that guess is wrong.",
      objective: 0,
    },
    {
      prompt: "On the default prompt, the readout after layer 1 in Layer by layer puts a different character on top from the finished model's. What does it mean?",
      options: [
        "It is the decision layer 1 made for this position, which layer 2 can only confirm or leave exactly as it was, since layer 2 reads the same stream",
        "It is what the final norm and unembedding would predict from the stream at that point, though layer 1 was never trained for that",
        "It is the distribution the sampler draws from once layer 1 has finished, before layer 2 has made any adjustment to it",
      ],
      answer: 1,
      explanation: "The logit lens applies the readout that normally follows layer 2 to the stream after layer 1. It shows what is linearly readable there, which is not a decision or a cause. Here layer 2 changes the top character, so it does more than confirm what layer 1 held.",
      objective: 3,
    },
    {
      prompt: "Two prompts end in the same character, at the same position, but have different earlier words. Why is their After embeddings readout identical?",
      options: [
        "The lens averages the stream over every position in the text, which hides any differences between the two earlier sets of words",
        "The model drops everything before the last character at every stage, so the later readouts match for the two prompts as well",
        "At that point the stream holds only that character's embedding plus its position, before attention mixes in earlier characters",
      ],
      answer: 2,
      explanation: "The stream starts as the token embedding plus the position embedding, so it depends on the last character and its position alone. Attention in layer 1 is what mixes earlier characters in, which is why the readouts after layers 1 and 2 can differ between the two prompts while the first one cannot.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateNextToken,
};

export default definition;
