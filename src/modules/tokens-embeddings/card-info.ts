import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Tokenizer workbench": {
    title: "Your text through a real byte-level BPE tokenizer",
    summary:
      "The text is split, turned into bytes, and merged by the 443 learned rules in `bpe-tokenizer.json`, giving the same IDs Hugging Face `tokenizers` produces. `Vocabulary size` keeps only the first merges, which is the tokenizer a shorter training run would have learned.",
    whatYouSee: [
      "`Text to tokenize` and a row of sample texts. Each sample is chosen to show one behaviour: whole-word tokens, spacing and case, missing digits, accented letters, and a literal special token.",
      "`Vocabulary size`, printed as entries and merges. 69 entries means 4 special tokens plus 65 byte symbols and no merges; 512 means all 443 merges.",
      "One chip per token. The small number is the token ID; `␣` marks a leading space that belongs to the token. A coloured bracket under a group of chips marks one pre-token, a chunk merges never cross; the selected chip's bracket is thicker.",
      "Dashed red chips are `<unk>` (ID 0): a byte this vocabulary never saw. Double-bordered chips are special tokens such as `<eos>`.",
      "`Tokens`, `Characters`, `Characters per token` and `Unknown bytes`, then a line explaining where the selected chip's ID comes from.",
    ],
    howItWorks: [
      "The text is NFKC-normalized and split by the GPT-2 pattern, which keeps a leading space with the word after it. Each chunk becomes its UTF-8 bytes, shown as printable symbols (a space byte is `Ġ` in the file, `␣` here).",
      "Within each chunk, the adjacent pair with the lowest merge rank joins, and this repeats until no ranked pair remains. A byte missing from the 65-symbol alphabet becomes `<unk>`, since this tokenizer has no byte fallback.",
      "IDs are addresses: 0 to 3 are specials, 4 to 68 are byte symbols, and a merged token's ID is 69 plus its merge rank. `␣king` is merge 275, so its ID is 344.",
      "A literal `<eos>`, `<pad>`, `<bos>` or `<unk>` in the text is matched before normalization and becomes that special token, as in the Hugging Face encoder.",
      "The implementation was checked against Hugging Face `tokenizers` 0.23.2 on 316 strings with no differences; a subset is a unit test.",
    ],
    controls: [
      "`Text to tokenize`: edit freely, up to 200 characters. The chips, counts, and merge trace update on every keystroke.",
      "Sample buttons replace the text. `the The  the` shows `The` (167), `␣the` (82), and a leading `the` that splits into `t` + `he`.",
      "`Vocabulary size` (69 to 512): drag from 512 to 69 on the default sentence and the count rises from 18 tokens to 47, with the text unchanged.",
      "Click a chip to select it. The explanation line and the Merge trace card follow the selection.",
    ],
    notice: [
      "Frequent Shakespeare words are one token (`␣king`, `␣and`, `␣the`), while `␣queen` needs three and `␣attention` four. Frequency in the tokenizer's training text decides this, not length or meaning.",
      "`The` is ID 167 and `␣the` is ID 82. At the start of the text, with no leading space, `the` becomes `t` + `he`. Case and spacing change the tokens.",
      "In `2024 was 33 years`, the digits 2, 0 and 4 are `<unk>` but 3 is kept (ID 11). Tiny Shakespeare contains the digit 3 and not the others.",
      "Shrinking the vocabulary never changes a surviving token's ID. It only removes the latest merges, so words fall back to shorter pieces.",
      "The default sentence has `␣the` twice, after `and` and after `read`, and both chips show ID 82. A model would read the same embedding row at both places, so the row carries no information about position: order has to be added separately, as a positional encoding.",
    ],
    limits: [
      "In this lab: the tokenizer is tiny (512 entries) and trained only on Tiny Shakespeare, so modern words fragment heavily and most digits, accents, and emoji are `<unk>`. Its coverage should not be compared with a production tokenizer.",
      "In this lab: no model reads these IDs. The line about an embedding row describes what a model would do; the vectors in the cards below come from a separate word-level table.",
      "In general: production tokenizers have 32,000 to about 256,000 entries, and most modern ones cover all 256 bytes, directly or as a fallback, so no text is `<unk>`; rare text just costs more tokens.",
      "In general: segmentation explains visible model behaviour. Digit grouping affects arithmetic, letter counting fails when a word is one opaque token, and languages underrepresented in tokenizer training spend more tokens on the same content.",
      "In general: encoders differ on special tokens. This one, like the Hugging Face default, turns a literal `<eos>` in user text into the real end-of-text token; tiktoken refuses such strings by default. Accepting them lets user text forge structure.",
    ],
  },

  "Merge trace": {
    title: "The merge sequence for one word, rung by rung",
    summary:
      "The selected chip's pre-token replayed through the merge table one rule at a time: start from bytes, join the lowest-ranked adjacent pair, repeat. Faded rungs are merges the current `Vocabulary size` does not contain.",
    whatYouSee: [
      "A heading naming the pre-token (for example `␣king`) and a badge counting how many of its merges fit in the current vocabulary.",
      "The first rung, `bytes`: one box per UTF-8 byte of the pre-token, before any merge.",
      "One rung per merge. Left to right: the merge rank (`#275`), the rule (`␣k + ing → ␣king`) with the new token's ID, and the word's pieces after that merge, with the newly joined piece outlined.",
      "Faded, dashed rungs with a note such as `needs vocabulary ≥ 345` when the slider has cut that merge. `×2` marks a rule that joined two places in one word.",
    ],
    howItWorks: [
      "At each step every adjacent pair is looked up in the merge table, and the pair with the lowest rank joins everywhere it occurs, left to right. The process stops when no adjacent pair has a rank.",
      "A merge can only use pieces that earlier merges created, so the ranks along a word always increase. That is why a smaller vocabulary simply cuts the list from the bottom.",
      "The trace always shows every merge in the full 512-entry table. The chips on the workbench use only the merges below the slider's budget, so the two agree on every rung that is not faded.",
      "Byte-pair encoding was trained greedily: each merge was the most frequent adjacent pair in the training text at that moment. Rank order is therefore the order in which the merges were learned.",
    ],
    controls: [
      "Select a chip on Tokenizer workbench to trace its pre-token. Selecting any chip of `␣attention` traces the whole chunk.",
      "`Vocabulary size` on the workbench: set it below 345 with `␣king` selected and the last rung fades, and the chip splits into `␣k` and `ing`.",
    ],
    notice: [
      "`␣king` is built in four steps: `i + n` (#8), `in + g` (#44), `␣ + k` (#91), `␣k + ing` (#275). Common fragments such as `in` and `ing` were learned long before the whole word.",
      "`␣queen` stops at `␣qu`, `e`, `en`: no merge in the table joins those three, so the word stays in pieces at every vocabulary size.",
      "A word with no rungs, such as a run of unknown bytes, received no merges at all.",
    ],
    limits: [
      "In this lab: the table has only 443 merges learned from one small corpus, so the traces are short and the word pieces are unlike those in production tokenizers.",
      "In this lab: the trace follows one pre-token. Merges never cross pre-token boundaries, so the rest of the sentence cannot change it.",
      "In general: production tokenizers use the same rank-ordered replay, but with tens of thousands of merges and optimized data structures. Some, such as SentencePiece unigram models, choose segmentations by likelihood instead of merge order.",
      "In general: merge order reflects pair frequency during tokenizer training, not meaning. A word that sits in one token carries no extra meaning because of it.",
    ],
  },

  "Nearest neighbours": {
    title: "Cosine similarity over real word vectors",
    summary:
      "The query word's 50-number GloVe vector compared with all 10,000 rows by cosine similarity, listing the ten highest. Every value is computed live from the 2 MB vector file, loaded on first use.",
    whatYouSee: [
      "A status pill: `Loading…`, `Loaded`, or `Unavailable` with a retry button. The embedding cards wait for it; the tokenizer cards do not.",
      "`Query word`, which accepts any of the 10,000 lowercase words. An unknown word is flagged and the last valid word stays in use.",
      "Ten bars, one per neighbour, with the cosine printed to three decimals. The bar length runs from 0 to 1. Click a bar to make that word the query.",
      "The first 6 of the query's 50 raw numbers, then `Frequency rank` (its position in GloVe's frequency-ordered list) and `Vector length ‖v‖`.",
    ],
    howItWorks: [
      "Each row is divided by its length once, so the cosine of two words is the dot product of their unit vectors: `cos(u, v) = u·v / (‖u‖‖v‖)`.",
      "The query is compared with every other row and the ten largest cosines are kept. The query itself is skipped.",
      "The vectors are GloVe 6B 50-dimensional vectors, trained on Wikipedia and Gigaword news text. The first 10,000 purely alphabetic ASCII words in GloVe's frequency order were kept.",
    ],
    controls: [
      "`Query word`: type a word. The list updates as soon as the typed text is one of the 10,000 words.",
      "Click a neighbour to walk the space. Try `queen`, then `hot`, then `apple`.",
    ],
    notice: [
      "`queen` → princess 0.852, lady 0.805, elizabeth 0.787, king 0.784: royalty, women, and specific queens mixed together.",
      "`hot` → cool 0.861, cold 0.801. Words used in the same sentence frames end up close even when they mean opposite things.",
      "`apple` → chips, microsoft, pc. In news text the company sense dominates, and a word gets one vector for all its senses.",
    ],
    limits: [
      "In this lab: these are word-level GloVe vectors looked up by whole lowercase word, not rows for the BPE token IDs above. A real model keeps one row per token ID; no model here was trained with this tokenizer.",
      "In this lab: only 10,000 words and 50 dimensions ship, so rare words are missing and neighbourhoods are coarser than in larger tables.",
      "In general: cosine similarity measures shared contexts, not shared meaning. Antonyms, co-hyponyms, and topical associations all score high.",
      "In general: a transformer's input embedding is static like this one, but every later layer rewrites the vector using context, so `bank` in two sentences ends up different after attention.",
    ],
  },

  "Vector arithmetic": {
    title: "Analogies as vector offsets, with the honest answer",
    summary:
      "The 3CosAdd analogy test: normalize A, B and C to unit length, form `A − B + C`, and rank all words by cosine to that target. The result is reported as computed, including when it is not the textbook answer.",
    whatYouSee: [
      "Three word fields laid out as `A − B + C`. Each accepts any of the 10,000 words.",
      "Preset analogies, and `Input words` with `Excluded` or `Allowed to win`.",
      "Five result bars with their cosines. The top result is bold. When inputs are allowed, an input word is greyed and tagged `input word`.",
      "A sentence stating the top answer and, when inputs are excluded, what would have won otherwise.",
    ],
    howItWorks: [
      "`target = unit(A) − unit(B) + unit(C)`, then every row is scored by `cos(target, row)` and the five best are kept.",
      "With `Excluded`, A, B and C are skipped when ranking, which is how most published analogy benchmarks score. With `Allowed to win`, nothing is skipped.",
      "Nothing is tuned toward an expected answer. Whatever word ranks first is shown.",
    ],
    controls: [
      "Edit A, B or C, or pick a preset.",
      "`Input words`: switch on `king − man + woman` and watch the winner change from queen (0.852) to king (0.885).",
    ],
    notice: [
      "`king − man + woman` → queen at 0.852 only once the inputs are excluded; allowed to win, `king` itself scores 0.885.",
      "`paris − france + italy` → rome at 0.847, one of the relations this space encodes well.",
      "`bigger − big + small` → larger (0.924), with smaller second (0.885), and `tokyo − japan + germany` → frankfurt (0.889) ahead of berlin (0.868). The offset lands near the right region, not always on the right word.",
    ],
    limits: [
      "In this lab: the space is the same 10,000-word, 50-dimensional GloVe subset. Larger vocabularies and dimensions change which analogies succeed.",
      "In general: analogy arithmetic is fragile. Results depend on normalization, word frequency, and excluding the inputs, and many relations fail. It shows that some directions are consistent, not that the space stores facts as clean offsets.",
      "In general: a transformer's own embeddings are rarely queried this way. Its useful structure appears after attention and MLP layers mix in context.",
    ],
  },

  "Embedding projection": {
    title: "A 2-D map of 50-D vectors, and what it loses",
    summary:
      "All 10,000 words placed by one of the two precomputed projections in the asset file, UMAP or PCA, with the query's true 50-D neighbours or the analogy words highlighted. A readout counts how many 50-D neighbours survive as 2-D neighbours.",
    whatYouSee: [
      "A grey dot for every word inside the frame, drawn at its projected position with the same scale on both axes. The axes carry no units.",
      "In `queen + neighbours` mode: the query in the accent colour, its ten nearest words by 50-D cosine in violet, and dashed spokes from the query to each one.",
      "In `Analogy words` mode: solid arrows from B to A and from C to the top answer, and a dashed arrow that copies the B→A offset onto C.",
      "A `keeps k of 10 neighbours` badge, a caption counting the words in view, and leader lines where a label had to move off its dot.",
    ],
    howItWorks: [
      "UMAP was fit with cosine distance, 18 neighbours, and minimum distance 0.12. PCA takes the two directions of greatest variance after standardizing each of the 50 dimensions. Both were computed once when the asset was built, not in the browser.",
      "The badge compares two lists: the query's ten nearest words by 50-D cosine, and its ten nearest dots by 2-D Euclidean distance. `keeps` is the size of their overlap.",
      "`Fit highlights` frames the highlighted words with padding; `All words` shows the whole projection.",
    ],
    controls: [
      "`Projection`: switch UMAP and PCA with `queen` selected. UMAP keeps 5 of 10 neighbours; PCA keeps 0.",
      "`Highlight`: switch between the query's neighbours and the analogy words from Vector arithmetic.",
      "`Frame`: zoom to the highlighted words or show all 10,000.",
    ],
    notice: [
      "Under UMAP most of queen's neighbours cluster, but elizabeth and victoria land far away: their spokes are long even though their cosines are 0.787 and 0.729.",
      "Under PCA the same neighbours scatter across the map. Two directions of variance cannot hold ten thousand words' neighbourhoods.",
      "In analogy mode the dashed arrow rarely ends on the answer. An offset that works in 50 dimensions need not survive a 2-D projection.",
    ],
    limits: [
      "In this lab: the projections were computed when the asset was built and cannot place a new vector such as the analogy target. The map shows words only, never `A − B + C` itself.",
      "In this lab: the same scale on both axes keeps shapes honest, but the absolute positions, cluster sizes, and gaps in UMAP carry no meaning.",
      "In general: every 2-D view of high-dimensional vectors discards most distance information. Use a projection to find neighbourhoods to inspect, then check them with the real cosines.",
    ],
  },
};

export default cardInfo;
