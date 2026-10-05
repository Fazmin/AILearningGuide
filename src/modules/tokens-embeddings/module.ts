import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateTokensState, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-01-tokens-embeddings",
  slug: "tokens-embeddings",
  title: "Tokens & embeddings",
  group: "foundations",
  order: 7,
  icon: "Binary",
  accent: "#5a67d8",
  prerequisites: ["module-31-data-features-representation"],
  estimatedMinutes: 15,
  steps: ["Split text", "Replay the merges", "Inspect token IDs", "Map meaning", "Try vector arithmetic"],
  stepInstructions: [
    "Keep the default sentence and drag Vocabulary size from 512 down to 69. Compare the Tokens count: the text is unchanged, only the merge budget shrank.",
    "Select the ␣king chip and read Merge trace from bytes to the final token. Then set Vocabulary size below 345 and watch the last rung fade and the chip split.",
    "Compare the IDs on The, ␣the and ␣king, and note that the two ␣the chips in the default sentence share ID 82: one row, wherever the word sits. Then pick the sample 2024 was 33 years and find the <unk> chips with ID 0.",
    "Type a word into Query word and read its ten cosine neighbours. Then switch Projection between UMAP and PCA and read how many neighbours the map keeps.",
    "With king − man + woman, switch Input words between Excluded and Allowed to win. Then try bigger − big + small and read the honest top answer.",
  ],
  stateVersion: 2,
  tagline:
    "Watch a real byte-pair tokenizer turn text into IDs one merge at a time, then query real word vectors for neighbours, analogies, and a flat map that distorts them.",
  objectives: [
    "Explain why models process tokens rather than words",
    "Trace how byte-pair merges turn bytes into token IDs",
    "Interpret distance in an embedding space",
    "Explain why an embedding is one vector per token and where word order has to come from",
  ],
  glossary: [
    {
      term: "Token",
      definition:
        "A chunk of text drawn from a fixed vocabulary. It may be a whole word with its leading space, a word fragment, punctuation, a single byte, or a reserved special token.",
    },
    {
      term: "Token ID",
      definition:
        "The integer address of a token in the vocabulary. It only selects one row of the embedding matrix and is never used in arithmetic. In this BPE vocabulary IDs follow merge order, but the model attaches no meaning to that order.",
    },
    {
      term: "Vocabulary",
      definition:
        "The fixed set of tokens a tokenizer can emit: 512 in this lab, usually 32,000 to about 256,000 in production. It is set before training; adding entries later means adding and training new embedding rows.",
    },
    {
      term: "Subword",
      definition:
        "A token smaller than a word, used so a fixed vocabulary can still spell rare words, names, and typos by combining pieces.",
    },
    {
      term: "Byte-pair encoding",
      definition:
        "A tokenizer training method that starts from single bytes or characters and repeatedly adds the most frequent adjacent pair as a new token, so frequent words become one token and rare words decompose.",
    },
    {
      term: "Merge rule",
      definition:
        "One learned pair, such as i + n becoming in, stored with its rank. Encoding replays the rules in rank order, and a smaller vocabulary is the same list cut short.",
    },
    {
      term: "Pre-tokenization",
      definition:
        "The split into word-like chunks before any merge runs. Merges never cross these boundaries, which is why a leading space stays attached to the word after it.",
    },
    {
      term: "Special token",
      definition:
        "A reserved ID with a structural meaning, such as end-of-text, padding, or a chat role marker. Whether the literal string in user text becomes one is an encoder setting; this lab's tokenizer allows it.",
    },
    {
      term: "Embedding",
      definition:
        "The learned vector a token ID is replaced with. It is the model's entire representation of that token before any layer runs, and it is the same in every context.",
    },
    {
      term: "Embedding matrix",
      definition:
        "The lookup table of shape vocabulary-by-width holding one trainable row per token. It is often the largest single tensor in a model. Selecting a row equals multiplying a one-hot vector by the matrix.",
    },
    {
      term: "Embedding dimension",
      definition:
        "The width of each vector, written d: 50 for the GloVe vectors here, several hundred to over 16,000 in production models. More dimensions give more directions but raise the cost of every later operation.",
    },
    {
      term: "Cosine similarity",
      definition:
        "The cosine of the angle between two vectors, from −1 to 1, ignoring their lengths. High similarity means used in similar contexts, which includes opposites: the two nearest words to hot here are cool and cold.",
    },
    {
      term: "Analogy arithmetic",
      definition:
        "Adding and subtracting word vectors, as in king − man + woman, then taking the nearest remaining word. It works for some relations and fails for others, and the famous answers depend on excluding the input words.",
    },
    {
      term: "Projection",
      definition:
        "A reduction of high-dimensional vectors to two or three dimensions for display. It keeps some neighbourhoods and distorts most distances, and how much depends on the method.",
    },
    {
      term: "Positional encoding",
      definition:
        "A vector that depends on a token's slot, added to its embedding or applied inside attention, because the embedding itself is the same wherever the token appears. Learned, sinusoidal, relative and rotary schemes exist.",
    },
  ],
  references: [
    {
      authors: "Daniel Jurafsky and James H. Martin",
      title: "Speech and Language Processing, 3rd edition (draft)",
      source: "Online manuscript, August 2026 release, free to read online",
      year: 2026,
      url: "https://web.stanford.edu/~jurafsky/slp3/",
      note: "Chapter 2, Words and Tokens, walks through byte-pair encoding on UTF-8 bytes and the pre-tokenization step. Chapter 5, Embeddings, explains why words used in similar contexts get similar vectors, how cosine similarity compares them, and why analogy answers must exclude the input words.",
    },
    {
      authors: "Rico Sennrich, Barry Haddow, and Alexandra Birch",
      title: "Neural Machine Translation of Rare Words with Subword Units",
      source: "Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016), 1715–1725",
      year: 2016,
      url: "https://aclanthology.org/P16-1162/",
      note: "The paper that brought byte-pair encoding to language models. It merges the most frequent adjacent pair again and again so rare words are spelled from subword pieces, the recipe the Tokenizer workbench runs.",
    },
    {
      authors: "Alec Radford, Jeffrey Wu, Rewon Child, et al.",
      title: "Language Models are Unsupervised Multitask Learners",
      source: "OpenAI technical report",
      year: 2019,
      url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
      note: "The GPT-2 report. Its section on input representation explains byte-level BPE: a base vocabulary of 256 bytes, so any text can be encoded, and rules that stop merges from crossing character types. This lab's split uses the GPT-2 pattern.",
    },
    {
      authors: "Hugging Face",
      title: "Normalization and pre-tokenization",
      source: "Hugging Face LLM Course, Chapter 6",
      year: 2026,
      url: "https://huggingface.co/learn/llm-course/chapter6/4",
      note: "Shows the steps that run before any merge: Unicode normalization such as NFKC, then splitting into word-like chunks. It prints the GPT-2 pre-tokenizer's output, which keeps each space and marks it as Ġ, the symbol this lab draws as ␣.",
    },
    {
      authors: "Hugging Face",
      title: "Byte-Pair Encoding tokenization",
      source: "Hugging Face LLM Course, Chapter 6",
      year: 2026,
      url: "https://huggingface.co/learn/llm-course/chapter6/5",
      note: "Trains a tiny BPE tokenizer by hand, learning one merge rule at a time, then encodes new words by applying the rules in the order they were learned. That is what Merge trace replays rung by rung.",
    },
    {
      authors: "OpenAI",
      title: "tiktoken",
      source: "GitHub repository (openai/tiktoken), file tiktoken/core.py",
      year: 2026,
      url: "https://github.com/openai/tiktoken/blob/main/tiktoken/core.py",
      note: "The source of OpenAI's BPE tokenizer. The encode method raises an error by default when user text contains a special-token string, so it cannot be used to forge structure. The lesson contrasts this with this lab's tokenizer, which accepts such strings.",
    },
    {
      authors: "Aaditya K. Singh and DJ Strouse",
      title: "Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs",
      source: "arXiv preprint arXiv:2402.14903",
      year: 2024,
      url: "https://arxiv.org/abs/2402.14903",
      note: "Tests how the way digits are grouped into tokens changes a model's arithmetic. Adding commas so numbers split from the right made answers much more accurate, backing the lesson's point that digit grouping affects arithmetic.",
    },
    {
      authors: "Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi",
      title: "Language Model Tokenizers Introduce Unfairness Between Languages",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html",
      note: "Measures how many tokens the same translated text needs in different languages and finds large gaps. That changes cost and how much fits in the context window, the lesson's point about the same sentence costing different amounts.",
    },
    {
      authors: "Jeffrey Pennington, Richard Socher, and Christopher Manning",
      title: "GloVe: Global Vectors for Word Representation",
      source: "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), 1532–1543",
      year: 2014,
      url: "https://aclanthology.org/D14-1162/",
      note: "The method behind the word vectors in Nearest neighbours, Vector arithmetic and Embedding projection. It learns vectors from how often words appear near each other, and it trains on the 6-billion-token Wikipedia and Gigaword corpus the lab's vectors come from.",
    },
    {
      authors: "Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig",
      title: "Linguistic Regularities in Continuous Space Word Representations",
      source: "Proceedings of the 2013 Conference of the North American Chapter of the ACL: Human Language Technologies (NAACL-HLT 2013), 746–751",
      year: 2013,
      url: "https://aclanthology.org/N13-1090/",
      note: "The paper that showed king − man + woman landing near queen. It is where the famous preset in Vector arithmetic comes from, and it tests offsets for both meaning and grammar relations.",
    },
    {
      authors: "Omer Levy and Yoav Goldberg",
      title: "Linguistic Regularities in Sparse and Explicit Word Representations",
      source: "Proceedings of the Eighteenth Conference on Computational Natural Language Learning (CoNLL 2014), 171–180",
      year: 2014,
      url: "https://aclanthology.org/W14-1618/",
      note: "Names the add-then-compare method 3CosAdd, the term in the Vector arithmetic heading, and shows how one large similarity can outweigh the others in the sum. It also proposes a multiplicative version.",
    },
    {
      authors: "Malvina Nissim, Rik van Noord, and Rob van der Goot",
      title: "Fair Is Better than Sensational: Man Is to Doctor as Woman Is to Doctor",
      source: "Computational Linguistics 46(2), 487–497",
      year: 2020,
      url: "https://aclanthology.org/2020.cl-2.7/",
      note: "Shows that standard analogy code never lets an input word be the answer, and that accuracy drops sharply when inputs are allowed, often because king itself comes back. This is what the Input words switch shows.",
    },
    {
      authors: "Leland McInnes, John Healy, and James Melville",
      title: "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction",
      source: "arXiv preprint arXiv:1802.03426",
      year: 2018,
      url: "https://arxiv.org/abs/1802.03426",
      note: "The method behind the UMAP option in Projection. It builds a weighted graph of each point's nearest neighbours and lays out a low-dimensional map that tries to match it, which is why it favours keeping local neighbours.",
    },
    {
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.",
      title: "Attention Is All You Need",
      source: "Advances in Neural Information Processing Systems 30 (NeurIPS 2017)",
      year: 2017,
      url: "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
      note: "The transformer paper. Section 3.5 adds positional encodings to the embeddings because nothing else tells the model token order, and Section 3.4 shares one weight matrix between the embedding layers and the output layer.",
    },
    {
      authors: "Ofir Press and Lior Wolf",
      title: "Using the Output Embedding to Improve Language Models",
      source: "Proceedings of the 15th Conference of the European Chapter of the ACL (EACL 2017), Volume 2: Short Papers, 157–163",
      year: 2017,
      url: "https://aclanthology.org/E17-2025/",
      note: "Recommends tying the input embedding matrix to the output layer that scores every vocabulary entry, and shows this can shrink models. It backs the lesson's note that many models tie the unembedding to the embedding.",
    },
  ],
  checkpoint: [
    {
      prompt: "The sample 2024 was 33 years shows <unk> chips in this lab, but a production byte-level tokenizer would never produce one. Why not?",
      options: [
        "Its vocabulary holds every word that could ever appear, so no text arrives that it lacks an entry for",
        "It starts from all 256 byte values, so any text can be spelled out, at the cost of more tokens",
        "It skips characters it has not seen, so they cost no tokens and never reach the model at all",
      ],
      answer: 1,
      explanation: "This lab's alphabet is the 65 byte symbols Tiny Shakespeare happens to use, with no byte fallback, so a digit it never saw becomes <unk>. A production byte-level tokenizer begins from every possible byte, so rare text just splits into many small tokens. Vocabularies hold tens of thousands of entries, far from every word.",
      objective: 0,
    },
    {
      prompt: "In this lab ␣king is one token but ␣queen is three (␣qu, e, en). What explains the difference?",
      options: [
        "Queen is less important to the model, so the tokenizer spends fewer IDs on it than on king",
        "Longer words are always cut into more pieces, and queen is longer than king in letters",
        "The merge table learned a chain of merges that completes ␣king but never completes ␣queen",
      ],
      answer: 2,
      explanation: "Merges follow pair frequency in the tokenizer's training text, Tiny Shakespeare. There, merges #8, #44, #91 and #275 build ␣king; ␣queen only reaches ␣qu, e, en. Length and meaning play no direct role: ␣speak is five letters and one token.",
      objective: 1,
    },
    {
      prompt: "You drag Vocabulary size from 512 down to 69 and leave the text alone. What happens to the Tokens count?",
      options: [
        "It rises, because with fewer merges words fall back to the smaller pieces they are built from",
        "It stays the same, because the text is unchanged and each word is still one token",
        "It falls, because a smaller vocabulary has fewer entries to emit for the sentence",
      ],
      answer: 0,
      explanation: "A smaller vocabulary is the same merge list cut short, so the later merges that glued letters into words are missing. Encoding replays only the merges that remain, and each word stays in more pieces. Nothing about the text changed, only how many glue steps were allowed.",
      objective: 1,
    },
    {
      prompt: "In Nearest neighbours, the two closest words to hot are cool and cold. What does a high cosine similarity between two word vectors tell you?",
      options: [
        "The words mean the same thing, so the model treats them as interchangeable",
        "The words have nearby token IDs, because similar words are numbered together",
        "The words are used in similar contexts, which includes opposites such as hot and cold",
      ],
      answer: 2,
      explanation: "Training pushes tokens that appear in similar sentence frames toward similar vectors, and hot and cold fill the same frames. Cosine similarity measures that overlap, not agreement in meaning. Token IDs are only shelf addresses assigned in merge order, so they carry no similarity at all.",
      objective: 2,
    },
    {
      prompt: "On the Embedding projection map, some of a word's true 50-dimensional neighbours sit far from it. What does that tell you?",
      options: [
        "Those words were never really neighbours, so the cosine list in Nearest neighbours is wrong",
        "The 2-D map had to discard distance information, so the real cosines are the better guide to use",
        "The map is correct and the vectors are noisy, so far-away dots mean the vectors were poorly trained",
      ],
      answer: 1,
      explanation: "Two directions cannot hold the neighbourhoods of ten thousand words in fifty dimensions, so any projection keeps some neighbours and pulls others apart. The badge counts how many survive: it differs between UMAP and PCA on the same word. Use the map to find a region, then check it against the cosines.",
      objective: 2,
    },
    {
      prompt: "In the default sentence both ␣the chips show ID 82. What does the embedding lookup return for them, and what follows from that?",
      options: [
        "A different vector each time, because the lookup reads the words around the token",
        "The same vector both times, which is enough because the IDs themselves are numbered in word order",
        "The same vector both times, so something else must tell the model which came first",
      ],
      answer: 2,
      explanation: "An embedding is one row per token ID, chosen without looking at the context, so identical tokens get identical vectors. IDs here follow merge order, not position in the text. Order has to be added separately, as a positional encoding added to the embedding or applied inside attention.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateTokensState,
};

export default definition;
