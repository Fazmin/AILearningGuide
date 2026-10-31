import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { hydrateEmbeddingTrainingState, initialState } from "./state";

const definition: ModuleDefinition = {
  id: "module-57-embedding-training",
  slug: "embedding-training",
  title: "Training an embedding model",
  group: "training-adapting",
  order: 50,
  icon: "Magnet",
  accent: "#5b4fa8",
  prerequisites: ["module-38-embeddings-search", "module-21-fine-tuning-transfer"],
  estimatedMinutes: 16,
  steps: [
    "Read the untrained space",
    "Train with in-batch negatives",
    "Turn the temperature",
    "Mine hard negatives",
    "Spot a false negative",
    "Rerank the shortlist",
  ],
  stepInstructions: [
    "Set Training step to 0 and read Recall@1 and MRR in Held-out retrieval and Mean cosine between passages in Embedding map, then compare them with the BM25 row of the table.",
    "Drag Training step to 300, then switch Batch size between 2 and 32 and compare Starting loss (ln B), MRR in Held-out retrieval and Duplicate-pair cosine in Embedding map.",
    "Move Temperature τ from 0.10 to 0.03, then to 1.00, watching Mean cosine between passages and how far the points spread in Embedding map.",
    "Set Batch size to 2 and switch Negatives from In-batch only to Nearest wrong, comparing Closest wrong cosine, run average with Recall@1, then repeat the switch at Batch size 8.",
    "Keep Negatives on Nearest wrong at Batch size 8, switch Skip known duplicates on and off, and compare Duplicate-pair cosine with the rank of Sunday notice in Query inspector.",
    "Switch Rerank top 5 on and off and compare Recall@1 with Scoring passes per query in Held-out retrieval.",
  ],
  stateVersion: 1,
  tagline:
    "Train a small bi-encoder in the page with a contrastive loss, then watch batch size, temperature, mined negatives and duplicate labels change what it retrieves for questions it never saw.",
  objectives: [
    "Describe how a bi-encoder is trained with InfoNCE and in-batch negatives, where every query is scored against every passage in its batch",
    "Predict what batch size and temperature do to the starting loss, to held-out recall and to how spread out the vectors end up",
    "Explain why mined hard negatives help with confusable passages, and how a false negative pushes two true answers apart",
    "Evaluate a trained retriever with held-out Recall@k and MRR against BM25 and against its untrained start",
    "Compare a bi-encoder with a cross-encoder reranker on accuracy and on the work each query costs",
  ],
  glossary: [
    {
      term: "Bi-encoder",
      definition:
        "A retriever that encodes the query and each passage separately into vectors and scores a pair by their cosine or dot product. Passage vectors can be computed once and stored, so a search costs one query encoding plus fast comparisons, but the two texts never see each other.",
    },
    {
      term: "InfoNCE",
      definition:
        "A contrastive loss: score the query against its answer and a set of wrong candidates, divide by the temperature, take a softmax, and pay −log of the answer's share. With B equally scored candidates it starts at ln B, and it falls only as the answer pulls ahead.",
    },
    {
      term: "In-batch negatives",
      definition:
        "Using the other queries' answers in the same batch as the wrong candidates for each query, so a batch of B pairs gives every query B − 1 negatives at no extra encoding cost. Most of them come from unrelated topics, so they are easy.",
    },
    {
      term: "Temperature",
      definition:
        "The number τ that divides every cosine before the softmax. A small τ magnifies small differences, so the loss nears zero once the answer leads by a small margin; a large τ flattens the softmax, so every candidate keeps a share. It is unrelated to sampling temperature in text generation.",
    },
    {
      term: "Hard negative",
      definition:
        "A wrong passage the current model scores close to the query, often from the same topic, such as another dock's opening hours. Mining them, with BM25 or with the model's own nearest neighbours, puts the confusable cases into every batch.",
    },
    {
      term: "False negative",
      definition:
        "A candidate treated as wrong that actually answers the query, usually because the labels name only one relevant passage. Training then pushes a true answer away from the query; skipping known duplicates or filtering negatives with a stronger model reduces it.",
    },
    {
      term: "Character trigrams",
      definition:
        "Three-letter pieces of a word padded with boundary marks, so dock becomes #do, doc, ock and ck#. A typo or a plural still shares some pieces with the clean word, so it can land near it, but a word whose pieces training never connected to a meaning carries none.",
    },
    {
      term: "Mean pooling",
      definition:
        "Averaging the vectors of all the pieces of a text into one vector. Word order is lost, and every piece counts equally unless training shrinks the rows of the pieces that do not help.",
    },
    {
      term: "L2 normalisation",
      definition:
        "Dividing a vector by its length so it sits on the unit sphere. After it, cosine similarity is a plain dot product and only direction matters, which is why training can only rotate vectors, not stretch them.",
    },
    {
      term: "Recall@k",
      definition:
        "In this lab, the hit rate: the share of held-out queries with at least one relevant passage in the top k, which DPR calls top-k retrieval accuracy. Another common definition counts the share of all relevant passages found, which differs when a query has more than one.",
    },
    {
      term: "MRR",
      definition:
        "Mean reciprocal rank: for each query take 1 divided by the rank of the first relevant passage, then average. A first-place hit scores 1, second place 0.5, and a miss 0, so it rewards near misses that Recall@1 ignores.",
    },
    {
      term: "Anisotropy",
      definition:
        "Embeddings crowding into a narrow cone, so even unrelated texts have a high cosine. It was measured in pretrained language models; in this lab it is constructed by giving every weight row the same offset before training.",
    },
    {
      term: "Cross-encoder reranker",
      definition:
        "A model that reads the query and one passage together and outputs a relevance score. It sees both texts at once, so it can compare words directly, but every candidate costs a full pass, so it reranks a short list from a cheaper first stage. Here a five-feature logistic model stands in for it.",
    },
    {
      term: "Matryoshka embedding",
      definition:
        "An embedding trained with the loss applied to several leading slices of the vector, so its first few numbers already work as a smaller embedding. A vector not trained that way, like the one here, loses about half of its Recall@1 when cut to its first 4 numbers.",
    },
  ],
  references: [
    {
      authors: "Aaron van den Oord, Yazhe Li, and Oriol Vinyals",
      title: "Representation Learning with Contrastive Predictive Coding",
      source: "arXiv preprint arXiv:1807.03748",
      year: 2018,
      url: "https://arxiv.org/abs/1807.03748",
      note: "Names and defines InfoNCE: a cross-entropy for picking the one positive sample out of N − 1 negatives. It is the loss Batch similarity computes, with the batch's other answers and any mined passages as the negatives.",
    },
    {
      authors: "Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton",
      title: "A Simple Framework for Contrastive Learning of Visual Representations",
      source: "Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119, 1597–1607",
      year: 2020,
      url: "https://proceedings.mlr.press/v119/chen20j.html",
      note: "SimCLR. Its loss divides cosine similarities by a temperature before the softmax, it finds that larger batches help because they bring more negatives, and its temperature table peaks at 0.1, the lab's default. It backs the lesson's batch size and temperature points.",
    },
    {
      authors: "Vladimir Karpukhin, Barlas Oğuz, Sewon Min, et al.",
      title: "Dense Passage Retrieval for Open-Domain Question Answering",
      source:
        "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), 6769–6781",
      year: 2020,
      url: "https://aclanthology.org/2020.emnlp-main.550/",
      note: "DPR trains a question encoder and a passage encoder with in-batch negatives, and finds that one BM25 negative per question, shared across the batch as the lab's mined negatives are, helps substantially. Its top-k retrieval accuracy is the hit rate the lab calls Recall@k, and it reports BM25 winning on SQuAD, where questions copy the passage's words.",
    },
    {
      authors: "Nils Reimers and Iryna Gurevych",
      title: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
      source:
        "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP 2019), 3982–3992",
      year: 2019,
      url: "https://aclanthology.org/D19-1410/",
      note: "Measures the cost gap the reranking step shows: finding the most similar pair among 10,000 sentences takes about 65 hours with BERT reading every pair as a cross-encoder and about 5 seconds with separately encoded sentence vectors. Its default pooling is the mean, as here.",
    },
    {
      authors: "Liang Wang, Nan Yang, Xiaolong Huang, et al.",
      title: "Text Embeddings by Weakly-Supervised Contrastive Pre-training",
      source: "arXiv preprint arXiv:2212.03533",
      year: 2022,
      url: "https://arxiv.org/abs/2212.03533",
      note: "E5, a production recipe for the same loss: InfoNCE with in-batch negatives, a temperature of 0.01, batches of 32,768, and query: and passage: prefixes, then fine-tuning with mined hard negatives. It gives the real-scale numbers the lesson sets against the lab's batch of 2 to 32.",
    },
    {
      authors: "Lee Xiong, Chenyan Xiong, Ye Li, et al.",
      title: "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval",
      source: "International Conference on Learning Representations (ICLR 2021)",
      year: 2021,
      url: "https://arxiv.org/abs/2007.00808",
      note: "ANCE takes hard negatives from a nearest-neighbour index of the model being trained, refreshed asynchronously during training, which is what Nearest wrong does every 10 steps. It argues that negatives sampled locally in the batch are uninformative and give diminishing gradients.",
    },
    {
      authors: "Yingqi Qu, Yuchen Ding, Jing Liu, et al.",
      title:
        "RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering",
      source:
        "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2021), 5835–5847",
      year: 2021,
      url: "https://aclanthology.org/2021.naacl-main.466/",
      note: "Finds that about 70% of the top-retrieved passages not labelled as answers, checked for 100 questions, were actually answers or highly relevant, and filters such false negatives with a cross-encoder. It also shares passages across GPUs for more in-batch negatives. It backs the false-negative step.",
    },
    {
      authors: "Tongzhou Wang and Phillip Isola",
      title: "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere",
      source: "Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119, 9929–9939",
      year: 2020,
      url: "https://proceedings.mlr.press/v119/wang20k.html",
      note: "Defines the two readouts in Embedding map: alignment, the mean squared distance between positive pairs, and uniformity, the log of the mean of exp(−2 × squared distance) over pairs. It proves that the contrastive loss optimizes both as the number of negatives grows.",
    },
    {
      authors: "Feng Wang and Huaping Liu",
      title: "Understanding the Behaviour of Contrastive Loss",
      source:
        "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021), 2495–2504",
      year: 2021,
      url: "https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Understanding_the_Behaviour_of_Contrastive_Loss_CVPR_2021_paper.html",
      note: "Shows that a small temperature concentrates the penalty on the hardest negatives and makes embeddings more uniform, while a large one penalizes all negatives alike. The lesson sets this published result against the lab, where a small τ leaves the passages bunched.",
    },
    {
      authors: "Kawin Ethayarajh",
      title:
        "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings",
      source:
        "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP 2019), 55–65",
      year: 2019,
      url: "https://aclanthology.org/D19-1006/",
      note: "Finds that contextual word representations are anisotropic, occupying a narrow cone, so that in GPT-2's last layer two random words have almost perfect cosine similarity on average. The lab's shared starting offset is a constructed stand-in for this.",
    },
    {
      authors: "Tianyu Gao, Xingcheng Yao, and Danqi Chen",
      title: "SimCSE: Simple Contrastive Learning of Sentence Embeddings",
      source:
        "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021), 6894–6910",
      year: 2021,
      url: "https://aclanthology.org/2021.emnlp-main.552/",
      note: "Shows that the contrastive objective regularizes pretrained embeddings' anisotropic space to be more uniform, the change Mean cosine between passages tracks as training proceeds.",
    },
    {
      authors: "Po-Sen Huang, Xiaodong He, Jianfeng Gao, et al.",
      title: "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data",
      source:
        "Proceedings of the 22nd ACM International Conference on Information & Knowledge Management (CIKM 2013), 2333–2338",
      year: 2013,
      url: "https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/",
      note: "DSSM, an early two-tower retriever. Its word hashing pads words with # and cuts them into letter trigrams (#good# becomes #go, goo, ood, od#), the lab's encoder input, and it trains with a softmax over the clicked document and four unclicked ones.",
    },
    {
      authors: "Rodrigo Nogueira and Kyunghyun Cho",
      title: "Passage Re-ranking with BERT",
      source: "arXiv preprint arXiv:1901.04085",
      year: 2019,
      url: "https://arxiv.org/abs/1901.04085",
      note: "Feeds the query and a passage to BERT together and reads a relevance score, reranking BM25's top 1,000 on MS MARCO. This is the cross-encoder the lab's five-feature reranker stands in for.",
    },
    {
      authors: "Aditya Kusupati, Gantavya Bhatt, Aniket Rege, et al.",
      title: "Matryoshka Representation Learning",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 30233–30249",
      year: 2022,
      url: "https://arxiv.org/abs/2205.13147",
      note: "Adds the loss at several nested prefix sizes so the first m numbers of one embedding work as a smaller embedding. The lab's First 4 of 16 numbers readout shows what happens to a vector not trained that way.",
    },
    {
      authors: "Nandan Thakur, Nils Reimers, Andreas Rücklé, et al.",
      title: "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
      source:
        "Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1 (NeurIPS Datasets and Benchmarks 2021)",
      year: 2021,
      url: "https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html",
      note: "Tests retrievers on 18 datasets they were not trained for: BM25 is a robust baseline, dense retrievers often fall behind it under large domain shift, and rerankers do best at a high computational cost. It backs the lesson's warning that recall on one small, single-domain set says little about another.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "At Training step 0 every vector points almost the same way. With Batch size 8 and In-batch only, what should Loss at step 0 sit close to?",
      options: [
        "About ln 8, since equal scores give each of the 8 answers an even share",
        "About zero, since each answer already shares more letters with its own query",
        "About one, since every cosine between the vectors is close to one",
      ],
      answer: 0,
      explanation:
        "InfoNCE pays −log of the answer's share of its row. If all 8 candidates score the same, each gets one eighth and the loss is ln 8 = 2.08, which is what Starting loss (ln B) shows. The first batch measures 1.72, a little lower, because the untrained encoder already gives each answer slightly more than an even share. Only the differences between cosines in a row matter, so cosines near one do not make the loss near one.",
      objective: 0,
    },
    {
      prompt:
        "At step 300 with Skip known duplicates off, you switch Batch size from 2 to 32. Which readout moves the most?",
      options: [
        "Held-out MRR roughly doubles, since every query must now beat 31 negatives",
        "Duplicate-pair cosine falls, as twins share batches more often",
        "Starting loss falls, since more candidates make the first guess easier",
      ],
      answer: 1,
      explanation:
        "A bigger batch gives every query more negatives and raises Starting loss (ln B) from 0.69 to 3.47, but held-out MRR barely moves, 0.69 against 0.71. What changes is how often a query meets its answer's twin as another query's answer: in the 31 batches the card can show, no row did at batch 2 and 89 rows did at batch 32. Each time, the loss pushes a true answer away, so Duplicate-pair cosine falls from 0.73 to 0.27.",
      objective: 1,
    },
    {
      prompt: "At step 300 you lower Temperature τ from 0.10 to 0.03. What happens to Mean cosine between passages in this lab?",
      options: [
        "It rises, because small leads already win and the loss stops pushing",
        "It falls further below zero, because a small τ pushes harder on every negative",
        "It stays the same, because τ only rescales the softmax and never moves a vector",
      ],
      answer: 0,
      explanation:
        "Mean cosine rises from −0.01 to 0.52. Dividing by 0.03 turns a cosine lead of 0.2 into a logit lead of almost 7, so each answer wins nearly all of its row early, the training loss falls to 0.001, and the push that spreads passages apart stops. τ does move the vectors: it scales every gradient. Published large-scale work (Wang and Liu, 2021) finds the opposite trend, a small τ spreading embeddings, because there the hardest negatives never stop arriving; 90 training pairs are fitted within a few dozen steps.",
      objective: 1,
    },
    {
      prompt:
        "At Batch size 2 you switch Negatives from In-batch only to Nearest wrong, and held-out Recall@1 rises from 63% to 77%. Why?",
      options: [
        "Each row now holds a wrong passage the model confuses, so the loss keeps teaching",
        "Mined passages count as extra answers, so each step trains on twice as many pairs as before",
        "The index refresh copies some held-out queries into training every 10 steps",
      ],
      answer: 0,
      explanation:
        "At batch 2 a row's only in-batch negative is the other query's answer, usually from another topic: the closest wrong cosine averages 0.39 over the run and the answer already takes 89% of the share. Nearest wrong adds the wrong passage the current encoder finds closest, which raises that cosine to 0.68 and lowers the answer's share to 74%, so the gradient keeps working on confusable passages. Miners never return the labelled answer, and held-out queries never enter training. At batch 8 the batch already holds near misses, and Recall@1 stays at 63%.",
      objective: 2,
    },
    {
      prompt:
        "With Nearest wrong at Batch size 8, the Sunday question ranks North dock hours first and Sunday notice third; Skip known duplicates brings Sunday notice back to first. What went wrong?",
      options: [
        "The miner kept handing each Sunday passage the other as its wrong answer",
        "North dock hours was labelled as the answer to some of the Sunday training queries",
        "Rerank top 5 reordered the shortlist and pushed Sunday notice below the weekday hours",
      ],
      answer: 0,
      explanation:
        "Sunday hours and Sunday notice state the same fact, but each training query labels only one. In 81 of 124 index refreshes for the four Sunday training queries, the nearest wrong passage was the other Sunday passage, so training pushed two true answers apart and Duplicate-pair cosine fell to 0.33. Skip known duplicates leaves the twin out of the softmax and out of mining, and the pair returns to 0.59. Nothing was relabelled, and the inspector's After training row is the bi-encoder's own ranking.",
      objective: 2,
    },
    {
      prompt:
        "After 300 steps, held-out Recall@1 is 63% against BM25's 43%. What does that result support?",
      options: [
        "That training helped on these harbor queries, not that it beats BM25 elsewhere",
        "That the trained encoder will beat BM25 on any collection of questions and documents",
        "That BM25 is broken, since it cannot find answers that share no word with the query",
      ],
      answer: 0,
      explanation:
        "The held-out queries were never trained on, so the gain is real for this collection. But all 120 queries come from one author and one small harbor, with the same habits, such as jetty for dock. BEIR tested retrievers on 18 datasets they were not trained for and found BM25 a robust baseline, with dense retrievers often behind it under large domain shift. BM25 is not broken either: it finds 12 of the 20 paraphrases at rank 1, and at the defaults no ranker here finds any of the five queries built on unseen words first.",
      objective: 3,
    },
    {
      prompt:
        "A held-out query's labelled answer ranks fourth, but its duplicate ranks first. What does this query add to Recall@1 and to MRR in this lab?",
      options: [
        "A hit at rank 1 and a reciprocal rank of 1, since either twin counts",
        "A miss at rank 1 and a reciprocal rank of 0.25, since only the labelled answer counts",
        "Half a hit and a reciprocal rank of 0.5, since one of two relevant passages was found",
      ],
      answer: 0,
      explanation:
        "Both members of a duplicate pair are relevant, and Recall@k here is the hit rate: a query scores 1 when any relevant passage is in the top k. MRR uses the rank of the first relevant passage, so first place gives 1. The other common definition of recall, the share of all relevant passages found, would give 0.5 at k = 1, which is why the lab names the hit rate in its card notes.",
      objective: 3,
    },
    {
      prompt:
        "Rerank top 5 lifts Recall@1 from 63% to 67% and Scoring passes per query from 1 to 6. Why not run the cross-encoder on all 41 passages instead?",
      options: [
        "Each passage would cost a joint pass, 41 per query, and the top 5 hold most answers",
        "A cross-encoder can only score passages that a bi-encoder has already ranked for it",
        "Reading the query and the passage together makes a cross-encoder less accurate overall",
      ],
      answer: 0,
      explanation:
        "Passage vectors are stored in advance, so the bi-encoder costs one query encoding plus 41 dot products. A cross-encoder must read every pair: 41 passes here, and Sentence-BERT reports about 65 hours against 5 seconds for the most similar pair among 10,000 sentences. Reranking the top 5 spends 5 passes where they matter, since a relevant passage is in the top 5 for 83% of queries. It cannot fix a miss below rank 5, and here it moves three queries to first place while knocking two off it.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateEmbeddingTrainingState,
};

export default definition;
