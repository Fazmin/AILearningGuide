import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { CONTEXT_MAX, CONTEXT_MIN, DEFAULT_CONTEXT } from "./recite";

const initialState: ModuleState = {
  dropBoilerplate: true,
  dropExact: true,
  dropNear: false,
  decontaminate: false,
  leak: false,
  minLength: 0,
  view: "raw",
  context: DEFAULT_CONTEXT,
};

/**
 * Version 1 had no leak or decontamination switches; both default to off, so
 * an old payload keeps its meaning once the new keys are merged in. Version 2 had no
 * context length for the regurgitation probe; it takes its default the same way.
 */
export function hydrateDatasetState(value: string): ModuleState {
  try {
    const parsed = JSON.parse(value) as ModuleState;
    const flag = (key: string) =>
      typeof parsed[key] === "boolean" ? (parsed[key] as boolean) : (initialState[key] as boolean);
    const minLength =
      typeof parsed.minLength === "number" && Number.isFinite(parsed.minLength)
        ? Math.max(0, Math.min(60, Math.round(parsed.minLength)))
        : 0;
    return {
      dropBoilerplate: flag("dropBoilerplate"),
      dropExact: flag("dropExact"),
      dropNear: flag("dropNear"),
      decontaminate: flag("decontaminate"),
      leak: flag("leak"),
      minLength,
      view: parsed.view === "clean" ? "clean" : "raw",
      context:
        typeof parsed.context === "number" && Number.isFinite(parsed.context)
          ? Math.max(CONTEXT_MIN, Math.min(CONTEXT_MAX, Math.round(parsed.context)))
          : DEFAULT_CONTEXT,
    };
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-18-dataset-building",
  slug: "dataset-building",
  title: "Building a dataset",
  group: "training-adapting",
  order: 37,
  icon: "Database",
  accent: "#2f8b6a",
  prerequisites: ["module-17-training-run"],
  estimatedMinutes: 15,
  steps: [
    "Read the raw scrape",
    "Run the filters",
    "Check the counts",
    "Score both datasets",
    "Leak and decontaminate",
    "Probe for recitation",
  ],
  stepInstructions: [
    "With Show on All lines, read every scraped line with its verdict, and find the four kinds of junk before turning any filter on.",
    "Toggle each switch in Cleaning stages and watch which lines change verdict, how many each removes, and how the Survival funnel narrows.",
    "Compare Distinct trigrams, Repetition, and Raw repetition in Dataset statistics, then read the train, val, and test split bar.",
    "Compare the two held-out loss curves and the Difference metric: the cleaned dataset is smaller, so decide whether it still wins.",
    "Turn on Leak a held-out sentence into the scrape and read Cleaned held-out and the contamination check, then turn on Decontaminate.",
    "In Regurgitation probe, drag Context length from 1 to 8 and compare Copied verbatim for the raw and cleaned sets, then read Footer text in the raw sample.",
  ],
  stateVersion: 3,
  tagline:
    "Filter a small scrape line by line, then train the same model on the raw and the cleaned version and score both on text neither has seen.",
  objectives: [
    "Name the failure modes a cleaning pipeline removes",
    "Explain why held-out text must be separate from every training set",
    "Judge a dataset change by its validation loss rather than its size",
    "Explain when a model recites its training text, and why personal data, licences, and consent need checks that deduplication does not give",
  ],
  glossary: [
    {
      term: "Boilerplate",
      definition:
        "Text that belongs to the website rather than the document: navigation, cookie notices, copyright lines. It is highly repetitive, so a model spends real capacity on it.",
    },
    {
      term: "Exact deduplication",
      definition:
        "Removing byte-identical documents, usually by hashing each one and keeping the first of every hash. Cheap, and it stops a copied page from being seen once per copy every epoch.",
    },
    {
      term: "Near-duplicate",
      definition:
        "A document that differs only trivially from another, such as one spelling variant or one changed sentence. Detected by comparing shingles (short overlapping pieces) rather than exact strings.",
    },
    {
      term: "MinHash",
      definition:
        "A signature of k minimum hash values over a document's shingles. Two signatures agree in each position with probability equal to the documents' Jaccard similarity, so near-duplicates can be found without comparing every pair of sets.",
    },
    {
      term: "Trigram diversity",
      definition:
        "Distinct three-character sequences divided by total sequences. A blunt but real measure of how much new signal a corpus carries per character.",
    },
    {
      term: "Train/validation/test split",
      definition:
        "Three disjoint slices: one to fit on, one to tune decisions against, one to touch once. Splitting by document, not by sentence, is what keeps them disjoint.",
    },
    {
      term: "Held-out loss",
      definition:
        "Cross-entropy on text that never entered training. It reports whether a dataset change helped the model predict text like the held-out sample, and only while that sample stays out of training.",
    },
    {
      term: "Leakage",
      definition:
        "Evaluation text appearing in training data, usually through copies in a crawl. It makes a model look better on that evaluation without making it better anywhere else.",
    },
    {
      term: "Decontamination",
      definition:
        "Removing training documents that overlap an evaluation set, typically any document sharing a long word n-gram with a test item. The threshold trades missed leaks against deleting innocent text.",
    },
    {
      term: "Data quality",
      definition:
        "Not a single property. In practice it means low duplication, on-target style, correct formatting, and no evaluation contamination.",
    },
    {
      term: "Data mixture",
      definition:
        "The share of the training budget each source receives. Filtering decides what is admissible; the mixture decides how often each admissible source is seen, including deliberate repeats of small, valuable sources.",
    },
    {
      term: "Memorization",
      definition:
        "A model reproducing a stretch of its training text verbatim when given a prefix of it. It grows with the model's capacity, with how often the text was duplicated, and with how much context the prompt supplies; the probe here varies the last.",
    },
    {
      term: "Personal data",
      definition:
        "Text that identifies a person, such as a name, phone number, or email address. A model can memorize it from a single document, so deduplication does not remove the risk.",
    },
    {
      term: "Data governance",
      definition:
        "The records and rules around a dataset: where each document came from, under what licence, with what consent or opt-out, and what personal data it holds. These are legal and policy questions that no cleaning filter can answer by reading the text.",
    },
  ],
  references: [
    {
      authors: "Colin Raffel, Noam Shazeer, Adam Roberts, et al.",
      title: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
      source: "Journal of Machine Learning Research 21(140), 1–67",
      year: 2020,
      url: "https://jmlr.org/papers/v21/20-074.html",
      note: "Section 2.2 builds C4 from Common Crawl with simple rules: drop lines without end punctuation or under three words, drop Javascript warnings and placeholder text, and keep one copy of any repeated three-sentence span. These are full-size versions of Strip boilerplate, Minimum line length, and Drop exact duplicates.",
    },
    {
      authors: "Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, et al.",
      title: "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track",
      year: 2024,
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/370df50ccfdf8bde18f8f9c2d9151bda-Abstract-Datasets_and_Benchmarks_Track.html",
      note: "A modern web pipeline in which every step is judged by training models on the data and scoring them, as Score both datasets does. It covers boilerplate-removing text extraction, MinHash deduplication, and a quality classifier, and it found that deduplicating harder across all crawls did not help, so more removal is not automatically better.",
    },
    {
      authors: "Luca Soldaini, Rodney Kinney, Akshita Bhagia, et al.",
      title: "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research",
      source: "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 15725–15788",
      year: 2024,
      url: "https://aclanthology.org/2024.acl-long.840/",
      note: "Documents a full open pipeline with the stages this lab leaves out: language identification, quality and toxicity filters, and masking of personal data, plus deduplication. It also removes benchmark text before training, the real version of Decontaminate.",
    },
    {
      authors: "Jure Leskovec, Anand Rajaraman, and Jeffrey D. Ullman",
      title: "Mining of Massive Datasets",
      source: "Cambridge University Press, 3rd edition, free to read online",
      year: 2020,
      url: "http://www.mmds.org/",
      note: "Chapter 3, Finding Similar Items, explains shingles, Jaccard similarity, MinHash signatures, and locality-sensitive hashing step by step. It shows why two signatures agree with probability equal to the Jaccard similarity, which is what the MinHash note on Cleaning stages estimates.",
    },
    {
      authors: "Katherine Lee, Daphne Ippolito, Andrew Nystrom, et al.",
      title: "Deduplicating Training Data Makes Language Models Better",
      source: "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022), 8424–8445",
      year: 2022,
      url: "https://aclanthology.org/2022.acl-long.577/",
      note: "The study the lesson quotes: one 61-word sentence repeated over 60,000 times in C4, and over 1 percent of model output copied from training data. Deduplicated models emitted memorized text ten times less often, and the paper also measures train-test overlap in standard datasets.",
    },
    {
      authors: "Shayne Longpre, Gregory Yauney, Emily Reif, et al.",
      title: "A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity",
      source: "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024), 3245–3276",
      year: 2024,
      url: "https://aclanthology.org/2024.naacl-long.179/",
      note: "Trains 28 models on data with different quality and toxicity filters and finds trade-offs, with effects that are hard to predict in advance. It backs the lesson's point that a filter removing text a human would call junk can still cost accuracy, so each filter needs a measured verdict.",
    },
    {
      authors: "Jesse Dodge, Maarten Sap, Ana Marasović, et al.",
      title: "Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus",
      source: "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021), 1286–1305",
      year: 2021,
      url: "https://aclanthology.org/2021.emnlp-main.98/",
      note: "An audit of C4 that found examples from benchmark datasets inside the crawl, the leakage this lab's Leak switch imitates. It also found that C4's word blocklist removed text from and about minority groups more often, an example of a filter carrying its own bias.",
    },
    {
      authors: "Tom B. Brown, Benjamin Mann, Nick Ryder, et al.",
      title: "Language Models are Few-Shot Learners",
      source: "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)",
      year: 2020,
      url: "https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html",
      note: "The GPT-3 report. Section 4 treats a benchmark example as possibly leaked if it shares a 13-gram with the training data, then scores a clean version of each benchmark. This is the full-size version of the 8-word check that Decontaminate runs.",
    },
    {
      authors: "Niklas Muennighoff, Alexander M. Rush, Boaz Barak, et al.",
      title: "Scaling Data-Constrained Language Models",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html",
      note: "Finds that repeating data for up to about 4 epochs changes loss very little compared with fresh data, while much more repetition stops helping. It backs the Going deeper point that a data mixture can show a small, valuable source a few times on purpose.",
    },
    {
      authors: "Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, et al.",
      title: "Quantifying Memorization Across Neural Language Models",
      source: "International Conference on Learning Representations (ICLR 2023)",
      year: 2023,
      url: "https://arxiv.org/abs/2202.07646",
      note: "Shows that memorization grows with model size, with how often an example was duplicated, and with how much context the prompt gives. The Regurgitation probe varies the last of these with its Context length slider.",
    },
    {
      authors: "Nicholas Carlini, Florian Tramèr, Eric Wallace, et al.",
      title: "Extracting Training Data from Large Language Models",
      source: "30th USENIX Security Symposium (USENIX Security 21), 2633–2650",
      year: 2021,
      url: "https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting",
      note: "Recovers hundreds of verbatim passages from GPT-2, including names, phone numbers, and email addresses, each found in just one training document. It is the lesson's evidence that deduplication alone does not protect personal data.",
    },
    {
      authors: "Shayne Longpre, Robert Mahari, Anthony Chen, et al.",
      title: "A large-scale audit of dataset licensing and attribution in AI",
      source: "Nature Machine Intelligence 6(8), 975–987",
      year: 2024,
      url: "https://www.nature.com/articles/s42256-024-00878-8",
      note: "The Data Provenance Initiative traced more than 1,800 text datasets and found licences often missing or wrong on popular hosting sites. It backs the Licences point under Data governance: whether text may be used cannot be read from the text itself.",
    },
    {
      authors: "Shayne Longpre, Robert Mahari, Ariel Lee, et al.",
      title: "Consent in Crisis: The Rapid Decline of the AI Data Commons",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track",
      year: 2024,
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3738949a80306cc48a8ea8ba0560f9d-Abstract-Datasets_and_Benchmarks_Track.html",
      note: "An audit of 14,000 web domains and their robots.txt files and terms of service. It is the source of the lesson's Consent figures: in one year, restrictions grew to cover over 5 percent of C4's tokens through robots.txt and 45 percent through terms of service.",
    },
    {
      authors: "Timnit Gebru, Jamie Morgenstern, Briana Vecchione, et al.",
      title: "Datasheets for Datasets",
      source: "Communications of the ACM 64(12), 86–92",
      year: 2021,
      url: "https://arxiv.org/abs/1803.09010",
      note: "Proposes a standard question list to publish with every dataset, covering where the data came from, whether people can be identified, whether they consented, and what licence applies. It is a practical form of the record the lesson says must be kept with the data.",
    },
    {
      authors: "European Parliament and Council of the European Union",
      title: "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)",
      source: "Official Journal of the European Union, L series",
      year: 2024,
      url: "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
      note: "Article 53(1) asks providers of general-purpose AI models to keep a policy for following EU copyright law, including rights reservations, and to publish a sufficiently detailed summary of the content used for training. It is the rule the lesson cites under Data governance.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "Cleaning removes a third of your corpus and the held-out loss improves. What is the most defensible reading?",
      options: [
        "Smaller datasets always train better models, so any cut of this size would have helped as much",
        "Fewer training tokens means fewer tokens are scored, so the average loss falls without any real gain in the model",
        "The held-out text must have leaked into training, since less data cannot lower the loss",
        "Repeated text kept pulling the weights toward itself without new signal, so removing it freed capacity",
      ],
      answer: 3,
      explanation:
        "Repeated boilerplate is still gradient signal: a line seen four times pulls the weights four times as hard, so the model spends its limited parameters fitting it. Both models are scored on the same held-out text, so the comparison is fair, and size only helps when the added text carries new signal. That is why held-out loss is the arbiter rather than the token count.",
      objective: 2,
    },
    {
      prompt:
        "One scraped line writes harbour where an earlier line writes harbor and is otherwise identical. Which filter catches it, and why do the other filters miss it?",
      options: [
        "Exact deduplication, because the two lines mean the same thing and the hash compares meaning",
        "Near-duplicate removal: exact matching misses it because the bytes differ, but the trigrams overlap",
        "Boilerplate stripping, because a repeated sentence is the same kind of junk as a copyright line or a subscribe banner",
        "Minimum line length, because a one-letter spelling change turns the line into a fragment",
      ],
      answer: 1,
      explanation:
        "Exact deduplication hashes bytes, so one changed letter gives a different hash. The near-duplicate test compares sets of character trigrams and flags a pair when their overlap passes a threshold, which catches spelling variants like this one. Boilerplate and length filters look at what a line says or how long it is, not at whether another line already said it.",
      objective: 0,
    },
    {
      prompt:
        "You switch on Leak a held-out sentence into the scrape, and Cleaned held-out loss improves. What does that improvement mean?",
      options: [
        "The model has seen that sentence, so the score now rewards recall of copied text, not generalisation",
        "The pipeline recovered more useful training text, so the cleaned model now generalises better, and the loss shows it",
        "The held-out passage became more representative of the scrape, so the same model is scored more fairly",
      ],
      answer: 0,
      explanation:
        "Once evaluation text also sits in the training data, part of the score measures recall of a memorised string, so the loss falls without the model getting better at unseen text. The tell in the lab is that loss on the held-out sentences that were not copied does not improve. Decontamination removes the overlap and the score returns to its honest value.",
      objective: 1,
    },
    {
      prompt:
        "In this lab the best-scoring filter setting beats the default by less than a hundredth of a nat, and the winner changes with the shuffle seed. How should you read that gap?",
      options: [
        "As proof that boilerplate is valuable training text that real pipelines should keep in the corpus",
        "As an error, since removing less text can never score better than removing more",
        "As unresolved, because it is smaller than the spread a different shuffle order would produce",
        "As the true ranking, since the seed is fixed and the loss is an exact cross-entropy",
      ],
      answer: 2,
      explanation:
        "The lab fixes one shuffle seed, so each score is a single draw without an error bar. Across seeds the raw-versus-cleaned gap stays clearly positive, but the difference between the two best filter settings changes sign, so it is smaller than the noise a different shuffle adds. A gap is only evidence when it is larger than that spread.",
      objective: 2,
    },
    {
      prompt:
        "In Regurgitation probe you drag Context length from 1 to 8 on the same training text. What happens to the share of the sample copied verbatim?",
      options: [
        "It stays near zero, because a count-based model only averages the text it saw and never repeats it",
        "It rises from near zero to almost everything, because long contexts leave one possible continuation",
        "It falls, because a longer context gives the model many more ways to recombine the text it saw",
        "It rises only for the raw scrape, because cleaning removes most of the text that a model could copy",
      ],
      answer: 1,
      explanation:
        "With one character of context, many different characters follow each context, so sampling mixes the text and copies nothing. As the context grows, nearly every context occurs once, so the table has a single continuation and sampling follows the training text word for word. Cleaning changes how much of the output is footer text, but the cleaned set is also copied almost completely.",
      objective: 3,
    },
    {
      prompt:
        "A scraped page names one person and their phone number, once, in a single document. Why does deduplication not make a model trained on it safe?",
      options: [
        "A repeated page is the only kind a model can memorize, so a page that occurs once is safe from that",
        "The phone number is a short string, and short strings are always removed by a length filter",
        "A model can memorize text it saw once, so removing copies does not remove the risk of recitation",
        "Deduplication also deletes the original, so the remaining model has seen the text too many times",
      ],
      answer: 2,
      explanation:
        "Duplication raises the chance that text is memorized, and removing it lowers the rate: one study found deduplicated training cut emitted memorized text about tenfold. But an extraction study recovered names, phone numbers and email addresses from GPT-2 that each appeared in just one document. Personal data needs its own detection and policy, and a licence or consent check is a separate question again.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateDatasetState,
};

export default definition;
