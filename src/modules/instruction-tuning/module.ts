import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";
import { isServeMode, sanitizeDataset } from "./sft";

const initialState: ModuleState = {
  useSystem: true,
  useRoles: true,
  useEnd: true,
  lossMask: "assistant",
  epochs: 30,
  serveWith: "template",
  dataset: [
    "when does the fog arrive | the fog settles over the harbor in the morning.",
    "how do i start the sauce | warm the pan and add a spoon of oil.",
    "what happens by noon | the clouds break apart and the sun warms the streets.",
    "how long should it simmer | simmer the sauce until it thickens.",
  ].join("\n"),
};

const bool = (value: unknown, fallback: boolean) => (typeof value === "boolean" ? value : fallback);
const intIn = (value: unknown, low: number, high: number, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, Math.round(value))) : fallback;

const definition: ModuleDefinition = {
  id: "module-23-instruction-tuning",
  slug: "instruction-tuning",
  title: "Instruction tuning & templates",
  group: "training-adapting",
  order: 44,
  icon: "MessageSquareCode",
  accent: "#c0574f",
  prerequisites: ["module-21-fine-tuning-transfer", "module-12-post-training"],
  estimatedMinutes: 16,
  steps: ["Edit the template", "Read the loss mask", "Fine-tune on your pairs", "Break the boundary", "Serve a different format"],
  stepInstructions: [
    "Turn System block, Role markers and End-of-turn marker on and off and watch the first example's strip gain and lose structure.",
    "Find where the graded cells begin in the strip, then switch Loss on between Assistant reply only and Every character and compare Mean −log p, graded with Mean −log p, context.",
    "Write your own instruction | response line, then raise SFT epochs and watch Reply log-prob per char, templated respond.",
    "Turn End-of-turn marker off and compare the two lists under What follows the end of a turn.",
    "Under Serve the trained models, switch Serve with between Chat template, No template and Wrong role marker, and compare the First reply character readout for the templated and plain models.",
  ],
  stateVersion: 3,
  tagline:
    "Completion data becomes a conversation only when the format says whose turn it is. Edit the template, see which characters are graded, and train on it.",
  objectives: [
    "Explain what a chat template adds to raw completion data",
    "Identify which part of the sequence the model is asked to produce",
    "Predict what training without an end-of-turn marker does to a model's sense of where a reply stops",
    "Predict how serving a trained model with a format it did not train on changes what it does next",
  ],
  glossary: [
    {
      term: "Supervised fine-tuning",
      definition:
        "Next-token training on curated prompt-and-response pairs. The per-token loss is the same cross-entropy as pretraining; what changes is the data, its formatting, and usually a mask that grades only the response.",
    },
    {
      term: "Chat template",
      definition:
        "The exact string layout that turns a list of messages into one sequence, including every marker. Two models with different templates need different strings for the same conversation.",
    },
    {
      term: "Special token",
      definition:
        "A token id reserved for structure, such as a turn marker. It is unforgeable only if the serving stack tokenizes user text so that typing the marker's spelling never yields that id.",
    },
    {
      term: "System prompt",
      definition:
        "A leading block of instructions outside the user turn. It is only privileged because the template and the training data agreed to treat it that way.",
    },
    {
      term: "End-of-turn token",
      definition:
        "The marker that tells a decoder to stop generating. Without it a model keeps writing, usually by inventing the next user turn itself.",
    },
    {
      term: "Loss masking",
      definition:
        "Grading only the assistant's reply and the marker that closes it, while the system and user text is read as context but never graded. Common in SFT recipes and this lab's default; Every character switches it off.",
    },
    {
      term: "Template mismatch",
      definition:
        "Serving a model with formatting that differs from its training format. It degrades quality quietly, because nothing errors — the model simply sees an unfamiliar string.",
    },
    {
      term: "Instruction following",
      definition:
        "Producing a response to a request instead of continuing the request's text. It is learned from formatted examples, not from more pretraining.",
    },
    {
      term: "Role markers",
      definition:
        "Tags such as user and assistant that wrap each turn so the model can tell whose words are whose. Here they are spelled in letters between reserved angle brackets and share rows with ordinary text; a real tokenizer gives each its own reserved id.",
    },
    {
      term: "Packing",
      definition:
        "Putting several short training examples into one sequence to fill the context window. Unless the attention mask also blocks attention across each boundary, one example can read the one before it.",
    },
    {
      term: "Nats",
      definition:
        "The unit of every loss and surprise readout in this lab: the negative natural log of the probability the model gave the actual next character. A model that spreads probability evenly over the 30 characters scores ln 30, about 3.401.",
    },
    {
      term: "Pretraining",
      definition:
        "Next-token training on a broad corpus with every position graded, which is what the Every character setting reproduces. Supervised fine-tuning keeps the same loss and changes the layout and the mask.",
    },
    {
      term: "Superficial alignment hypothesis",
      definition:
        "The proposal, from the LIMA paper, that a model's knowledge and capabilities are learned almost entirely in pretraining, and that alignment mostly teaches which format and style to use with users. It predicts that a small set of high-quality examples can be enough, and it is a hypothesis, not a measured law.",
    },
    {
      term: "Multi-turn data",
      definition:
        "Training conversations with several user and assistant turns. A model fine-tuned only on single-turn pairs can lose track of the dialogue after a few turns, so multi-turn examples teach it to read earlier turns as context.",
    },
  ],
  references: [
    {
      authors: "Jason Wei, Maarten Bosma, Vincent Y. Zhao, et al.",
      title: "Finetuned Language Models Are Zero-Shot Learners",
      source: "International Conference on Learning Representations (ICLR 2022)",
      year: 2022,
      url: "https://arxiv.org/abs/2109.01652",
      note: "The FLAN paper fine-tunes a large pretrained model on over 60 datasets, each rewritten with hand-made instruction templates, and finds it handles new kinds of tasks much better. It backs the lesson's point that instruction following is learned from formatted examples, not from more pretraining.",
    },
    {
      authors: "Long Ouyang, Jeffrey Wu, Xu Jiang, et al.",
      title: "Training language models to follow instructions with human feedback",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html",
      note: "The InstructGPT paper opens by saying that a bigger model is not by itself better at doing what a user asks. Its first step is supervised fine-tuning on replies written by people, which is the SFT stage this lesson formats and the Post-training module places first.",
    },
    {
      authors: "Hugging Face",
      title: "Chat templates",
      source: "Hugging Face Transformers documentation",
      year: 2026,
      url: "https://huggingface.co/docs/transformers/en/chat_templating",
      note: "Shows two chat models fine-tuned from the same base model that need different strings for the same conversation, and how apply_chat_template builds them. It explains the generation prompt that opens the assistant turn, warns that adding special tokens twice hurts the model, and says to apply the same template in training.",
    },
    {
      authors: "Meta",
      title: "Llama 3: Model Cards and Prompt formats",
      source: "Meta Llama documentation",
      year: 2024,
      url: "https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/",
      note: "A real chat template laid out token by token: reserved header tokens wrap the system, user and assistant roles, and <|eot_id|> ends each turn. It also notes that the model expects the assistant header at the end of the prompt, which is what Role markers, End-of-turn marker and the served strings in this lab imitate with letters.",
    },
    {
      authors: "OpenAI",
      title: "OpenAI Harmony Response Format",
      source: "OpenAI Cookbook",
      year: 2025,
      url: "https://developers.openai.com/cookbook/articles/openai-harmony",
      note: "Lists each special token with its own token id, including the markers that start and end a message and the stop tokens that tell the caller to stop generating. It also ranks the roles, system above developer above user, which is the kind of privilege the lesson says a System block holds only because training made it so.",
    },
    {
      authors: "Hugo Touvron, Louis Martin, Kevin Stone, et al.",
      title: "Llama 2: Open Foundation and Fine-Tuned Chat Models",
      source: "arXiv preprint arXiv:2307.09288",
      year: 2023,
      url: "https://arxiv.org/abs/2307.09288",
      note: "Section 3.1 gives a real SFT recipe: prompts and answers are joined into full-length sequences with a special separator token, and the loss on user prompt tokens is set to zero so only answer tokens are trained. That is the Assistant reply only setting of Loss on, plus packing.",
    },
    {
      authors: "Zhengyan Shi, Adam X. Yang, Bin Wu, et al.",
      title: "Instruction Tuning With Loss Over Instructions",
      source: "Advances in Neural Information Processing Systems 37 (NeurIPS 2024)",
      year: 2024,
      url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ffb43adf37b3eeaba559098bc084cc6-Abstract-Conference.html",
      note: "Tests grading the instruction as well as the reply, like the lab's Every character setting. It helps most when instructions are long and replies short, or when there are only a few training examples, which is the lesson's reason to treat the mask as a choice.",
    },
    {
      authors: "Mathew Huerta-Enochian and Seung Yong Ko",
      title: "Instruction Fine-Tuning: Does Prompt Loss Matter?",
      source: "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024), 22771–22795",
      year: 2024,
      url: "https://aclanthology.org/2024.emnlp-main.1267/",
      note: "Instead of all or nothing, this study gives the prompt a partial weight in the loss. On data with short replies, a small nonzero weight did best on short-answer tests, while a full weight did better on long answers, so the loss mask works like a hyperparameter.",
    },
    {
      authors: "Chunting Zhou, Pengfei Liu, Puxin Xu, et al.",
      title: "LIMA: Less Is More for Alignment",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html",
      note: "The paper the lesson cites for the superficial alignment hypothesis. It fine-tunes a 65B LLaMA model on 1,000 curated examples, and it reports the multi-turn result from the Multi-turn data section: adding 30 multi-turn chains made the model much better at holding a conversation.",
    },
    {
      authors: "Mario Michael Krell, Matej Kosec, Sergio P. Perez, et al.",
      title: "Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance",
      source: "arXiv preprint arXiv:2107.02027",
      year: 2021,
      url: "https://arxiv.org/abs/2107.02027",
      note: "The source of the lesson's padding figure: up to half the tokens in common datasets can be padding. It shows how to pack short sequences together and change the attention mask so one packed example cannot read another, the care the Packing section asks for.",
    },
    {
      authors: "Nathan Lambert, Jacob Morrison, Valentina Pyatkin, et al.",
      title: "Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
      source: "arXiv preprint arXiv:2411.15124",
      year: 2024,
      url: "https://arxiv.org/abs/2411.15124",
      note: "A fully open post-training recipe. Its SFT experiments compare several chat templates, finding that even small changes, such as removing one newline before the reply, shift the average score a little, and an appendix prints the exact template as code. It shows that the template is a design choice a model must then be served with.",
    },
    {
      authors: "Eric Wallace, Kai Xiao, Reimar Leike, et al.",
      title: "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions",
      source: "arXiv preprint arXiv:2404.13208",
      year: 2024,
      url: "https://arxiv.org/abs/2404.13208",
      note: "Argues that models are open to prompt injection because they often give a system prompt the same priority as user text, and trains them on generated examples to rank system instructions higher. It backs the lesson's Where it breaks point that a system prompt is privileged only because training made it so.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "Both models train on the same four pairs for the same number of epochs, one with the chat template and one as plain text. What does the next-character list show after a reply's last character?",
      options: [
        "Only the plain model has learned anything there, because its text is shorter and simpler",
        "Only the templated model has learned anything there: it expects the reserved end marker",
        "Both models expect a space next, since a space follows most full stops in ordinary text",
        "Neither differs, because markers carry no information the letters do not already have",
      ],
      answer: 1,
      explanation:
        "The template adds a graded end marker after every reply, so the row for the full stop is trained to point at it. In plain concatenation nothing ever follows a reply's last character within its example, so that row never receives a gradient and stays uniform, one in thirty for every character.",
      objective: 0,
    },
    {
      prompt: "You switch Loss on from Assistant reply only to Every character. What changes in the strip and its metrics?",
      options: [
        "The system and user text stop being read, because they are now graded instead",
        "The reply stops being graded, because the mask now covers it",
        "The graded count stays the same, but each cell's bar grows taller",
        "Every character becomes graded, so no cell is context-only any more",
      ],
      answer: 3,
      explanation:
        "The mask decides which characters count in the loss, not which are read. Under the mask the system and user text is still read as context for the next prediction; it just is not graded. Every character removes the mask, which is exactly pretraining's rule.",
      objective: 1,
    },
    {
      prompt:
        "You turn End-of-turn marker off and keep the mask on. What happens to the templated model's next-character list after a finished reply?",
      options: [
        "The marker's share falls to the untrained level, because only the graded end marker taught it",
        "The marker's share stays high, because the role markers around each turn still teach where turns end",
        "The marker's share rises, because with no closing marker the model is less certain where a reply stops",
        "The list is empty, because a turn with no closing marker leaves the model nothing to condition on",
      ],
      answer: 0,
      explanation:
        "Under the mask only the reply and its closing end marker are graded, so the row for the full stop is trained only by that end marker. Remove it and the row is never updated, and returns the uniform one in thirty. The role markers sit before the reply, where nothing is graded, so they cannot teach where it ends. A model with no trained stop has no reason to end.",
      objective: 2,
    },
    {
      prompt:
        "The templated model puts most of its probability on the marker after a full stop. Why is 'it knows the turn ended' too strong a description?",
      options: [
        "The marker's letters never appear in the training pairs, so the probability must have come from the plain model",
        "A probability on one character can never show what a model has learned about a whole conversation",
        "It sees one previous character, so it has only learned that the row for a full stop points at the marker",
        "The angle brackets are stripped from everything typed, so the model has never seen a marker in training",
      ],
      answer: 2,
      explanation:
        "The model conditions on exactly one preceding character through a 30 by 30 table, so knowing a turn ended means the full-stop row learned to point at the marker. A real stop is one reserved id predicted from the whole conversation. The lab shows the mechanism that teaches a stop; it cannot show understanding.",
      objective: 2,
    },
    {
      prompt:
        "Under Serve with, you pick Chat template. The plain-trained model has never seen a closing bracket in training. What do its next-character bars show?",
      options: [
        "Every character at the same probability, because the bracket's row was never trained",
        "The reply's first letters rank highest, because the markers show a model where a reply begins",
        "The same bars as the templated model, because both models trained on the same four pairs",
        "No bars at all, because a model refuses to continue a string it never saw in training",
      ],
      answer: 0,
      explanation:
        "Each model reads one character to choose the next. Plain concatenation never puts a bracket in front of anything, so that row of the plain model's table was never updated and stays uniform. The templated model trained on exactly this row, so the two sets of bars differ. Nothing errors, which is why a template mismatch is silent: the model simply answers from an untrained row.",
      objective: 3,
    },
    {
      prompt:
        "Chat template and Wrong role marker give identical bars and samples. What does that show about the lab, rather than about real models?",
      options: [
        "Both strings end in the same bracket, and a one-character model cannot see which role came first",
        "Role markers carry no information for any model, so a real model would treat the two serves the same",
        "Loss masking erased the roles from the weights, so the markers no longer matter to any model",
        "Both models were retrained to one table, so every served string now produces the same bars",
      ],
      answer: 0,
      explanation:
        "The lab's models condition on a single preceding character, so a served string matters only through its last character. Both serves end in a bracket, so they read the same row. A real model reads the whole marker and would answer as the user. Switching to No template does change the bars, which rules out a retrained shared table.",
      objective: 3,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      const parsed = JSON.parse(value) as ModuleState;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { ...initialState };
      // Version 1 trained on every character. A snapshot from then has template switches
      // but no lossMask, so keep its meaning rather than silently masking it.
      const legacy =
        !("lossMask" in parsed) &&
        ["useSystem", "useRoles", "useEnd", "epochs", "dataset"].some((key) => key in parsed);
      // Version 3 added serveWith; older payloads fall back to the chat template. Every key is
      // validated, and the dataset is cut to the editor's own limits, because training runs in the render.
      return {
        useSystem: bool(parsed.useSystem, true),
        useRoles: bool(parsed.useRoles, true),
        useEnd: bool(parsed.useEnd, true),
        lossMask: legacy ? "all" : parsed.lossMask === "all" ? "all" : "assistant",
        epochs: intIn(parsed.epochs, 1, 80, 30),
        dataset: typeof parsed.dataset === "string" ? sanitizeDataset(parsed.dataset) : initialState.dataset,
        serveWith: isServeMode(parsed.serveWith) ? parsed.serveWith : "template",
      };
    } catch {
      return { ...initialState };
    }
  },
};

export default definition;
