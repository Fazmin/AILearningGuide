import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { initialState, normalizeState } from "./state";

const definition: ModuleDefinition = {
  id: "module-08-attention",
  slug: "attention",
  title: "Attention",
  group: "inside-models",
  order: 16,
  icon: "ScanEye",
  accent: "#8a5bc2",
  prerequisites: ["module-07-next-token-prediction"],
  estimatedMinutes: 20,
  steps: [
    "Choose a query",
    "Compare keys",
    "Scale and normalize",
    "Mix values",
    "Lift the mask",
    "Test a route",
  ],
  stepInstructions: [
    "Click a word chip in the arc diagram to make it the query, then read its strongest route and its On spaces share in the readout strip.",
    "Click a heatmap cell, then compare the Q and K chips and find the inspected key's outlined characters in the score strip.",
    "Step through Score and Softmax: watch scaled scores become weights in the strip, and check that the Softmax row matches the model's own value. Then set Score scaling to No scaling and compare Largest weight and Saturated rows with and without the division.",
    "Step to Mix and read the Mixed value chip. Set Value vectors (V) to Zero all and watch the weights stay put while the Mixed value falls to zero, then try Zero word on a key with a large weight.",
    "In the Attention matrix card set Mask to Bidirectional, then read on later characters in the readout strip and watch the earlier cells shrink. Compare Layer 1 head 3 with Layer 2 head 4, then set Mask back to Causal.",
    "Press Pronoun sentence and move the Attention head slider through both layers with it as the query: no head's strongest route is animal. Then press Repeated name, set Layer 2 and head 3, and press Name moved earlier to see whether the route follows the content. Finish with Repeats every 8 and Repeats every 9 in the Characters view and compare which heads' stripes move.",
  ],
  stateVersion: 3,
  tagline: "Explore how each token compares its query with other tokens' keys, then mixes their values to bring relevant context into its representation.",
  objectives: ["Distinguish queries, keys, and values","Read an attention pattern without over-interpreting it","Tell a head that follows content from one that reads a fixed distance","Explain what the causal mask changes and how encoder and decoder-only models differ","Say why scores are divided by the square root of the head dimension"],
  glossary: [
  {
    "term": "Query",
    "definition": "The projected vector describing what the current position is looking for. It is compared against keys and never carries the retrieved payload itself."
  },
  {
    "term": "Key",
    "definition": "The projected vector advertising what a position can offer. Query-key alignment sets the score, so keys decide who gets read, not what is read."
  },
  {
    "term": "Value",
    "definition": "The projected vector actually mixed into the output. A high attention weight moves this content; the score alone says nothing about what it contains."
  },
  {
    "term": "Attention weight",
    "definition": "The normalized share controlling how much one value contributes to one query's output. Every query's weights sum to one, so routes compete."
  },
  {
    "term": "Attention head",
    "definition": "One independent query, key, and value projection set. Heads specialize during training, but rarely into cleanly describable roles."
  },
  {
    "term": "Scaled dot product",
    "definition": "The query-key dot product divided by the square root of the head dimension. Without the division the scores grow with width, softmax saturates toward a one-hot row, and its gradients shrink; the lab can switch the division off at inference to show the saturation, which is not the same as training without it."
  },
  {
    "term": "Causal mask",
    "definition": "Adding minus infinity to the scores of later positions before softmax, so their weights are exactly zero and the matrix is triangular. It lets every position train in parallel on next-token prediction without seeing its own answer; the lab can lift it for one head, but this model was trained with it."
  },
  {
    "term": "Multi-head attention",
    "definition": "Several heads run in parallel on the same input, their outputs concatenated and passed through the output projection. Splitting one wide head into several narrow ones costs about the same arithmetic."
  },
  {
    "term": "Output projection",
    "definition": "The learned matrix applied to the concatenated head outputs before they are added back to the residual stream. It decides how retrieved content is written."
  },
  {
    "term": "Head dimension",
    "definition": "The width of one head's query, key, and value vectors, usually the model width divided by the head count (256 over 4 gives 64 here). It is the d in the scaling factor."
  },
  {
    "term": "Attention pattern",
    "definition": "The matrix of weights for one head and layer. It is evidence about routing, not an explanation of behavior: a route can come from position alone, and patterns can sometimes be altered without changing output."
  },
  {
    "term": "Induction head",
    "definition": "A head that attends from a repeated token to the token that followed its earlier occurrence, so it can copy what came next. In the usual two-layer circuit it is fed by a previous-token head in an earlier layer, and in larger models its appearance coincides with better in-context learning."
  },
  {
    "term": "Previous-token head",
    "definition": "A head whose weight sits almost entirely on the position just before the query, copying that character's information one step forward. In a two-layer model it can hand a layer-2 induction head the identity of each position's predecessor."
  },
  {
    "term": "Attention sink",
    "definition": "A position, often the first token, that absorbs large attention weight when a head has nothing useful to retrieve. Evicting it from a cache degrades output."
  },
  {
    "term": "Grouped-query attention",
    "definition": "Sharing one key and value pair across several query heads to shrink the KV cache, trading a small amount of quality for a large memory saving."
  },
  {
    "term": "Encoder",
    "definition": "A transformer stack with bidirectional attention: no causal mask, so the weight matrix is a full square and every position reads both earlier and later ones. It builds a representation of a whole input, for classification or search, but cannot be trained on next-token prediction because each position could see its own answer."
  },
  {
    "term": "Decoder-only (causal) language model",
    "definition": "A stack with causal attention, so each position reads only itself and earlier positions (a triangular matrix), trained to predict the next token at every position. This lab's model and most current chat models are of this kind, and they generate by feeding each new token back in."
  },
  {
    "term": "Masked language model",
    "definition": "A bidirectional model trained by hiding some tokens and predicting them from the visible tokens on both sides, as BERT does. A causal language model learns to continue text left to right; a masked one learns to fill gaps, which suits understanding better than free-form generation."
  },
  {
    "term": "Positional encoding",
    "definition": "A signal about each token's slot, because attention alone cannot tell order. Absolute schemes add a learned or sinusoidal vector to the input; relative schemes such as ALiBi bias the scores by distance; rotary embeddings rotate queries and keys. This lab's model uses learned absolute vectors."
  }
],
  references: [
    {
      authors: "Simon J. D. Prince",
      title: "Understanding Deep Learning",
      source: "MIT Press, free to read online",
      year: 2023,
      url: "https://udlbook.github.io/udlbook/",
      note: "Chapter 12, Transformers, builds dot-product self-attention step by step, then adds position encoding, scaled scores, and multiple heads. It contrasts an encoder (BERT) with a decoder (GPT3) that uses masked self-attention, the same split this lesson draws.",
    },
    {
      authors: "Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio",
      title: "Neural Machine Translation by Jointly Learning to Align and Translate",
      source: "International Conference on Learning Representations (ICLR 2015)",
      year: 2015,
      url: "https://arxiv.org/abs/1409.0473",
      note: "The paper that brought attention to neural translation. It argues that squeezing a whole sentence into one fixed-length vector is a bottleneck and lets the model softly search the input instead, the problem described in Why it's here.",
    },
    {
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.",
      title: "Attention Is All You Need",
      source: "Advances in Neural Information Processing Systems 30 (NIPS 2017)",
      year: 2017,
      url: "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
      note: "The original Transformer paper. Section 3.2.1 gives softmax(QK^T / sqrt(d_k)) V and the reason for the division behind the Score scaling control; later sections cover multi-head attention, masking later positions with minus infinity, and the sinusoidal position signal.",
    },
    {
      authors: "Ofir Press, Noah A. Smith, and Mike Lewis",
      title: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2108.12409",
      note: "Introduces ALiBi. It adds no position vectors at all and instead subtracts a penalty proportional to the distance between query and key from each score, one of the position schemes listed in Where position comes from.",
    },
    {
      authors: "Jianlin Su, Murtadha Ahmed, Yu Lu, et al.",
      title: "RoFormer: Enhanced Transformer with Rotary Position Embedding",
      source: "Neurocomputing 568, 127063",
      year: 2024,
      url: "https://arxiv.org/abs/2104.09864",
      note: "Introduces rotary position embedding (RoPE), used by Llama and many recent models. Queries and keys are rotated by their position, so the score depends on the distance between the two, as the lesson's position section describes.",
    },
    {
      authors: "Jacob Devlin, Ming-Wei Chang, Kenton Lee, et al.",
      title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      source: "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2019), 4171–4186",
      year: 2019,
      url: "https://aclanthology.org/N19-1423/",
      note: "The BERT paper, the lesson's example of an encoder and a masked language model. It explains why a bidirectional model cannot be trained left to right, since each word could see itself, and so trains it to fill hidden tokens instead.",
    },
    {
      authors: "Sarthak Jain and Byron C. Wallace",
      title: "Attention is not Explanation",
      source: "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2019), 3543–3556",
      year: 2019,
      url: "https://aclanthology.org/N19-1357/",
      note: "Finds that attention weights often fail to line up with gradient-based measures of which inputs mattered, and that very different weights can give the same prediction. It backs the lesson's warning that an attention pattern is not an explanation.",
    },
    {
      authors: "Sarah Wiegreffe and Yuval Pinter",
      title: "Attention is not not Explanation",
      source: "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP 2019), 11–20",
      year: 2019,
      url: "https://aclanthology.org/D19-1002/",
      note: "A reply to Jain and Wallace. It argues the answer depends on what counts as an explanation and proposes careful tests, which is why the lesson treats a pattern as evidence to test rather than as proof either way.",
    },
    {
      authors: "Kevin Clark, Urvashi Khandelwal, Omer Levy, et al.",
      title: "What Does BERT Look at? An Analysis of BERT's Attention",
      source: "Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, 276–286",
      year: 2019,
      url: "https://aclanthology.org/W19-4828/",
      note: "Studies the heads of a real model. Some put most of their weight on the previous or next token, a fixed offset much like this lab's Layer 1 heads, and many park weight on a separator token as a kind of no-op, an early look at what the lesson calls an attention sink.",
    },
    {
      authors: "Nelson Elhage, Neel Nanda, Catherine Olsson, et al.",
      title: "A Mathematical Framework for Transformer Circuits",
      source: "Transformer Circuits Thread, Anthropic",
      year: 2021,
      url: "https://transformer-circuits.pub/2021/framework/index.html",
      note: "Takes apart small attention-only transformers. Its two-layer section shows how an induction head is built from a previous-token head in the layer below, the circuit the Repeats every 8 and Repeats every 9 tests probe.",
    },
    {
      authors: "Catherine Olsson, Nelson Elhage, Neel Nanda, et al.",
      title: "In-context Learning and Induction Heads",
      source: "Transformer Circuits Thread, Anthropic",
      year: 2022,
      url: "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html",
      note: "The paper the lesson cites for induction heads. It defines them by prefix matching and copying, reports a phase change early in training when they form as in-context learning improves, and calls its evidence that they drive most in-context learning preliminary and indirect.",
    },
    {
      authors: "Guangxuan Xiao, Yuandong Tian, Beidi Chen, et al.",
      title: "Efficient Streaming Language Models with Attention Sinks",
      source: "International Conference on Learning Representations (ICLR 2024)",
      year: 2024,
      url: "https://arxiv.org/abs/2309.17453",
      note: "Names the attention sink: models put large weight on the first tokens even when they carry little meaning. Dropping those tokens' keys and values from the cache hurts output, and keeping a few of them fixes it, as the Going deeper section says.",
    },
    {
      authors: "Joshua Ainslie, James Lee-Thorp, Michiel de Jong, et al.",
      title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
      source: "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), 4895–4901",
      year: 2023,
      url: "https://aclanthology.org/2023.emnlp-main.298/",
      note: "Introduces grouped-query attention, where several query heads share one key and value head. It reports quality close to full multi-head attention with speed close to a single shared head, the trade the glossary describes.",
    },
    {
      authors: "Tri Dao, Daniel Y. Fu, Stefano Ermon, et al.",
      title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 16344–16359",
      year: 2022,
      url: "https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html",
      note: "Starts from the cost the lesson names, time and memory that grow with the square of sequence length. It computes the same exact softmax in tiles held in fast on-chip memory, so the full score grid is never written out.",
    },
  ],
  checkpoint: [
    {
      prompt: "You edit the K chip of the inspected word and leave every V vector alone. What can change in the Mixed value for the current query?",
      options: [
        "That position's own value vector, because the key and the value are two views of the same projection",
        "Only the weights, and so how much of each value is mixed in, because keys decide who gets read",
        "Nothing, because a key only changes the score readout and the mix is built from values alone",
      ],
      answer: 1,
      explanation: "Query-key alignment sets the scores, softmax turns them into weights, and the mix is a weighted sum of the unchanged value vectors. Editing a key therefore moves the mix by changing who is read and how much, never what each value contains. Keys and values are separate learned projections of the same input.",
      objective: 0,
    },
    {
      prompt: "You set Value vectors (V) to Zero all and look at the strip and the Mixed value. What do you see?",
      options: [
        "The weights stay exactly as before and the Mixed value is zero, since Q and K set the weights and V is the payload",
        "Every weight falls to zero as well, because a zero value leaves nothing for the attention to route to anyone at all",
        "The weights spread out evenly, because identical values make every key equally useful to the query mix",
      ],
      answer: 0,
      explanation: "The weights are computed from queries and keys before any value is read, so zeroing V cannot move them. The mix is the weighted sum of the values, so with every value zero it is the zero vector whatever the weights are. A heavy weight on a key whose value carries nothing moves nothing, which is why a weight alone never says what was retrieved.",
      objective: 0,
    },
    {
      prompt: "With Pronoun sentence loaded, Layer 1 and head 3 selected, the query it sends nearly all its weight to the word it. What is the sound reading?",
      options: [
        "The head has decided the pronoun refers to itself, because the heaviest arc marks what a word means to the model",
        "A stripe one place below the diagonal in the Characters view shows the head reads the previous character, inside it",
        "The head is broken, because a working head would send its weight to a different word than the query does",
      ],
      answer: 1,
      explanation: "A weight is routing evidence, not a reason. This head puts its weight on the character just before the query, and for the word it that character is the i, so the word view credits the word it. The same sentence gives no head a strongest route from it to animal, so a pronoun is not being resolved by a visible arc.",
      objective: 1,
    },
    {
      prompt: "On Repeated name, Layer 2 and head 3, the second Mark sends most of its weight to Jones. What does that pattern establish?",
      options: [
        "That the model knows who Mark is, because the heaviest arc lands on the surname that belongs to him in the text it read",
        "That the output depends on this head, because a heavy weight means the model used that head's value for its answer",
        "That this route is worth testing, but showing the head matters takes an intervention such as switching it off",
      ],
      answer: 2,
      explanation: "The pattern is a hypothesis about routing: the head sends weight to the word after the earlier Mark. It does not show what the value carried, whether later layers kept it, or whether the output used it. Switching the head off and measuring the loss is the kind of intervention that can show it mattered, and even that has limits.",
      objective: 1,
    },
    {
      prompt: "A head's stripes sit 7, 15 and 23 characters back on Repeats every 8. You press Repeats every 9. How do you tell a copying head from a fixed-distance one?",
      options: [
        "A copying head's stripes move to 8, 17 and 26 back, since it seeks the character after the earlier match; a fixed-distance head's stay put",
        "Compare how dark the stripes are at one period, because a copying head always puts more weight on the right character than a distance habit does",
        "Check that the head lands on the correct next character at period 8, because only a head that copies can do that, whatever its offset is",
      ],
      answer: 0,
      explanation: "A head that copies attends to the successor of the earlier occurrence, so its stripes follow the period. A head with a fixed offset keeps its stripes at that distance and only looks right when the period happens to match it. Landing on the correct next character at one period cannot separate the two, which is why the method is to change the period.",
      objective: 2,
    },
    {
      prompt: "Layer 1 head 3 reads the previous character whatever the text says. What makes a rule that depends on distance possible at all?",
      options: [
        "The causal mask only allows keys at fixed offsets from each query, so distance is built into it",
        "The value vectors store how far back each character sits and the mix reads that number out",
        "Position information is added to the token vectors, so queries and keys can depend on the slot",
      ],
      answer: 2,
      explanation: "The dot product sees only vectors, so a head can use distance only if the vectors carry the slot. This model adds a learned vector for each of its 64 positions to the token vector, which lets query and key encode where they are; with those vectors zeroed the stripes of its four Layer 1 heads collapse. The causal mask only blocks later positions, and values carry content.",
      objective: 2,
    },
    {
      prompt: "You set Mask to Bidirectional. What happens to the weights a query had already given to earlier characters?",
      options: [
        "They stay exactly as they were, because lifting the mask only adds new cells to the grid and leaves the old ones alone",
        "They all shrink by the same factor, because the row still sums to 100% and some weight moves to later characters",
        "They grow, because the head can now use more context to confirm which of the earlier characters really matter here",
      ],
      answer: 1,
      explanation: "Softmax divides by the sum over every readable key. Lifting the mask adds keys to that sum, so the denominator grows and every earlier weight is multiplied by the same factor, one minus the weight now on later characters. Ratios among the earlier keys do not change; only their share of the row does.",
      objective: 3,
    },
    {
      prompt: "You set Mask to Bidirectional and a Layer 2 head puts much of its weight on later characters. What does that show?",
      options: [
        "How a BERT-style encoder would read the sentence, because lifting the mask turns this whole model into one of those",
        "That the model peeks at later text when it generates, because generation reads the whole sentence at once anyway",
        "What this head scores for pairs it never trained on: what a bidirectional layer could read, not a trained one",
      ],
      answer: 2,
      explanation: "The model was trained with the causal mask, so nothing ever asked this head to score a later key. Lifting the mask at inference shows its raw scores for those pairs, and Layer 2's queries and keys still come from a causal Layer 1. A trained encoder learns different scores because it is trained with every position visible.",
      objective: 3,
    },
    {
      prompt: "Why is a decoder-only model trained on next-token prediction with the causal mask on?",
      options: [
        "Each position could otherwise attend to the token it must predict, so the loss could fall by copying the answer",
        "Later tokens do not exist yet during training, so there would be nothing for any query to attend to in its own row",
        "Without the mask each row of weights would no longer sum to one, so softmax would be undefined for those whole rows",
      ],
      answer: 0,
      explanation: "Training sees the whole text at once, so the later tokens do exist. Without the mask the position before an answer could read the answer itself, and next-token prediction would be solved by copying it. An encoder avoids this by training on a different task, filling hidden tokens from both sides, which is what a masked language model does.",
      objective: 3,
    },
    {
      prompt: "You switch Score scaling to No scaling on a trained head. What happens to its rows?",
      options: [
        "Nothing changes, because softmax gives the same weights whatever the overall size of all the scores is",
        "The scores are 8 times larger, so softmax pushes many rows onto one character and Saturated rows rises",
        "The scores shrink, so softmax flattens each row toward equal weights and the Largest weight metric falls",
      ],
      answer: 1,
      explanation: "Softmax is not scale-free: multiplying every score by 8 makes the largest score dominate far more. Removing the division leaves the raw dot product, which is the square root of the head dimension times larger. This shows saturation in a model trained with the division; it does not show training without it, where the original authors suspect small gradients would be the problem.",
      objective: 4,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      return normalizeState(JSON.parse(value));
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
