import type { ModuleDefinition } from "@app/module-sdk";
import Explore from "./Explore";
import { initialState, sanitizeState } from "./state";

/** Reads a serialized state of any version; see state.ts for what each version stored. */
export function hydrateMultimodalState(value: string) {
  try {
    return sanitizeState(JSON.parse(value));
  } catch {
    return { ...initialState };
  }
}

const definition: ModuleDefinition = {
  id: "module-42-multimodal-models",
  slug: "multimodal-models",
  title: "Multimodal models",
  group: "building-with-models",
  order: 26,
  icon: "Images",
  accent: "#9a4a8a",
  prerequisites: ["module-35-architecture-zoo", "module-08-attention", "module-09-transformer-block", "module-38-embeddings-search"],
  estimatedMinutes: 14,
  steps: [
    "Ink the grid",
    "Change the patch size",
    "Read a patch token",
    "Scale it up",
    "Compare connectors",
    "Match images to captions",
  ],
  stepInstructions: [
    "Click a few cells on the 8×8 grid, or use Selected pixel and Flip selected pixel, and watch the selected patch's flattened x change.",
    "Set Patch size to 2×2, then to 8×8. Patch tokens go from 16 to 1, and the image-token chips in Token sequence change to match.",
    "Set Patch size to 4×4 and move Selected patch. In Patch embedding read each filter's output W·x, the position code, and the four-number token.",
    "In Real-scale arithmetic set Image side to 224px and Encoder patch to 16×16: 196 patch tokens, 197 with CLS. Then 336px with 14×14: 576.",
    "Switch Connector between LLaVA, Flamingo-style and CLIP, and read which stage the image tokens reach and their share of the context.",
    "In Contrastive pairs find the outlined match in each row. Lower Temperature τ from 0.20 to 0.05 and watch the shares sharpen and Contrastive loss fall; raise it to 1.00 and they flatten. Then set Which pairs count as matches to shuffled captions: same similarities, much higher loss.",
  ],
  stateVersion: 3,
  tagline:
    "Cut an image into patches, project each patch into a token vector, place those tokens beside words, count what the same cut costs at real resolutions, and see why a contrastive loss makes matched images and captions score highest.",
  objectives: [
    "Compute how many patch tokens an image becomes from its size and the patch size, including the optional CLS token",
    "Describe a patch embedding as a linear map from a patch's pixels to a vector, plus a position code",
    "Compare what CLIP, LLaVA-style and Flamingo-style designs hand on — one pooled vector, projected tokens inside the text sequence, or 64 resampled vectors read by cross-attention — and what each costs in context",
    "Explain how a contrastive loss over a batch of image–caption pairs rewards the labelled pairs and penalizes the other pairings, and how the temperature sharpens or flattens each softmax",
  ],
  glossary: [
    {
      term: "Modality",
      definition:
        "A kind of input or output, such as text, images or audio. Each needs its own encoder to turn raw data into vectors a transformer can use.",
    },
    {
      term: "Patch",
      definition:
        "A square tile of the image. A ViT flattens each tile's pixels into one vector, so a 224-pixel image cut into 16-pixel tiles becomes 14 × 14 = 196 patches.",
    },
    {
      term: "Patch embedding",
      definition:
        "The learned linear map from a patch's flattened pixels to a vector of the encoder's width, usually implemented as a convolution with stride equal to the patch size.",
    },
    {
      term: "Position embedding",
      definition:
        "A vector added to each patch token so the model knows where the tile was. Attention alone ignores order, so without it the tiles would be an unordered bag.",
    },
    {
      term: "CLS token",
      definition:
        "An extra learned token some encoders prepend to the patches; its final vector often summarizes the image. LLaVA-style models typically pass only the patch features onward.",
    },
    {
      term: "Vision encoder",
      definition:
        "The network, usually a ViT, that turns patch embeddings into contextualized image features. It is trained separately, often contrastively, before being connected to a language model.",
    },
    {
      term: "Contrastive learning",
      definition:
        "Training two encoders so matching image–caption pairs score a high cosine similarity and mismatched pairs in the same batch score low. The loss is a cross-entropy over the batch's pairings, so every other caption acts as a wrong answer for an image. CLIP was trained this way on 400 million pairs.",
    },
    {
      term: "Projector",
      definition:
        "The small network that maps vision features to the language model's embedding width: a linear layer in the first LLaVA, a two-layer MLP in LLaVA-1.5.",
    },
    {
      term: "Cross-attention",
      definition:
        "Attention in which queries come from one sequence and keys and values from another. Flamingo inserts cross-attention layers so text tokens can read visual features without those features joining the text sequence.",
    },
    {
      term: "Image token",
      definition:
        "A patch-derived vector placed in the language model's input sequence. Self-attention then treats it like any other position, and it uses context-window budget like any other token.",
    },
    {
      term: "Temperature",
      definition:
        "In a contrastive loss, the number that divides the similarity scores before the softmax. A small value sharpens each row toward its top-scoring pair and a large one flattens it. CLIP learns it during training; it is unrelated to sampling temperature in text generation.",
    },
    {
      term: "Negative pair",
      definition:
        "An image and caption in the same batch that do not belong together. A batch of N true pairs holds N × N − N of them, and the loss pushes their scores down while pulling the true pairs up.",
    },
    {
      term: "Spectrogram",
      definition:
        "A picture of sound: time along one axis, frequency along the other, brightness for energy. An audio encoder cuts it into frames or patches, much as a vision encoder cuts an image.",
    },
    {
      term: "Image-borne injection",
      definition:
        "Instructions hidden in an image, usually as printed text, that the model reads as part of its input and may follow. The image is untrusted data that arrives through the same channel as the request.",
    },
  ],
  references: [
    {
      authors: "Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.",
      title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
      source: "International Conference on Learning Representations (ICLR 2021)",
      year: 2021,
      url: "https://arxiv.org/abs/2010.11929",
      note: "The ViT paper. It cuts an image into patches, flattens each one, maps it to a vector with a learned linear projection, adds a position embedding, and puts an extra class token in front. That is the cut, embed, and add-position recipe the Image to patches and Patch embedding cards follow.",
    },
    {
      authors: "Alec Radford, Jong Wook Kim, Chris Hallacy, et al.",
      title: "Learning Transferable Visual Models From Natural Language Supervision",
      source: "Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139, 8748–8763",
      year: 2021,
      url: "https://proceedings.mlr.press/v139/radford21a.html",
      note: "The CLIP paper. It trains an image encoder and a text encoder on 400 million image-text pairs to raise the cosine of the N real pairs in a batch and lower it for the N² − N wrong pairings, with a learned temperature. This is the loss the Contrastive pairs card computes and the CLIP option under Connector.",
    },
    {
      authors: "Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee",
      title: "Visual Instruction Tuning",
      source: "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 34892–34916",
      year: 2023,
      url: "https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html",
      note: "The first LLaVA paper. It feeds patch features from a CLIP encoder through one trainable linear layer, so they become tokens in the language model's input, then trains the model to follow instructions about images. This is the LLaVA option under Connector.",
    },
    {
      authors: "Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee",
      title: "Improved Baselines with Visual Instruction Tuning",
      source: "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024), 26296–26306",
      year: 2024,
      url: "https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.html",
      note: "The LLaVA-1.5 paper. It swaps the linear layer for an MLP projector and uses a CLIP ViT-L encoder at 336 pixels. These are the settings behind the lab's 336px, 14×14 starting setup and its 576 image tokens.",
    },
    {
      authors: "Andrew Jaegle, Felix Gimeno, Andrew Brock, et al.",
      title: "Perceiver: General Perception with Iterative Attention",
      source: "Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139, 4651–4664",
      year: 2021,
      url: "https://proceedings.mlr.press/v139/jaegle21a.html",
      note: "Shows how a small, fixed set of learned latent vectors can read a large input through cross-attention, so the cost no longer grows with the square of the input size. Flamingo's Perceiver resampler, a stage in the Flamingo-style connector, is built on this idea.",
    },
    {
      authors: "Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, et al.",
      title: "Flamingo: a Visual Language Model for Few-Shot Learning",
      source: "Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 23716–23736",
      year: 2022,
      url: "https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html",
      note: "The Flamingo paper. A Perceiver resampler turns the image features into 64 visual outputs, and new gated cross-attention layers let a frozen language model read them. It also confirms the lesson's caveat that Flamingo's image encoder is an NFNet, not a ViT.",
    },
    {
      authors: "Aaron van den Oord, Yazhe Li, and Oriol Vinyals",
      title: "Representation Learning with Contrastive Predictive Coding",
      source: "arXiv preprint arXiv:1807.03748",
      year: 2018,
      url: "https://arxiv.org/abs/1807.03748",
      note: "Introduces the InfoNCE loss: a cross-entropy for picking the one true sample out of a set that also holds N − 1 negatives. CLIP's loss uses the same idea, which is why every other caption in the batch counts as a wrong answer in Contrastive pairs.",
    },
    {
      authors: "Alec Radford, Jong Wook Kim, Tao Xu, et al.",
      title: "Robust Speech Recognition via Large-Scale Weak Supervision",
      source: "Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202, 28492–28518",
      year: 2023,
      url: "https://proceedings.mlr.press/v202/radford23a.html",
      note: "The Whisper paper. It gives the audio numbers in Beyond still images: sound resampled to 16,000 Hz, an 80-channel log-Mel spectrogram on 25 ms windows with a 10 ms stride, 30-second segments, and an encoder stem whose second convolution has a stride of two.",
    },
    {
      authors: "Peng Wang, Shuai Bai, Sinan Tan, et al.",
      title: "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution",
      source: "arXiv preprint arXiv:2409.12191",
      year: 2024,
      url: "https://arxiv.org/abs/2409.12191",
      note: "A newer vision-language model that turns images of different sizes into different numbers of visual tokens instead of resizing all of them to one size, and merges each 2 × 2 group of neighbouring tokens into one. It samples video at two frames per second, which backs the lesson's points about variable resolution and about cutting the token count for video.",
    },
    {
      authors: "Shengbang Tong, Zhuang Liu, Yuexiang Zhai, et al.",
      title: "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs",
      source: "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024), 9568–9578",
      year: 2024,
      url: "https://openaccess.thecvf.com/content/CVPR2024/html/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.html",
      note: "Finds pairs of clearly different images that CLIP scores as nearly the same, and shows that models such as LLaVA-1.5 and GPT-4V often answer wrongly about them. Counting and position are among the problem types, which backs the lesson's warning that a model can miscount or mix up spatial relations.",
    },
    {
      authors: "Yifan Li, Yifan Du, Kun Zhou, et al.",
      title: "Evaluating Object Hallucination in Large Vision-Language Models",
      source: "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), 292–305",
      year: 2023,
      url: "https://aclanthology.org/2023.emnlp-main.20/",
      note: "Shows that vision-language models, LLaVA among them, often describe objects that are not in the image, and introduces POPE, a yes-or-no test for measuring it. It backs the lesson's point that alignment is measured, not guaranteed.",
    },
    {
      authors: "Gabriel Goh, Nick Cammarata, Chelsea Voss, et al.",
      title: "Multimodal Neurons in Artificial Neural Networks",
      source: "Distill 6(3), e30",
      year: 2021,
      url: "https://distill.pub/2021/multimodal-neurons/",
      note: "Looks inside CLIP and finds that it has learned to read text in images. Its section on typographic attacks shows that sticking a handwritten label such as 'iPod' on an apple can change what CLIP thinks the object is, an early sign that printed text in an image steers the model.",
    },
    {
      authors: "Yichen Gong, Delong Ran, Jinyuan Liu, et al.",
      title: "FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual Prompts",
      source: "Proceedings of the AAAI Conference on Artificial Intelligence 39(22), 23951–23959 (AAAI 2025)",
      year: 2025,
      url: "https://ojs.aaai.org/index.php/AAAI/article/view/34568",
      note: "Turns forbidden requests into typed text inside an image and shows that open vision-language models, LLaVA-1.5 among them, follow them more often than when the same request comes as plain text. It backs the lesson's warning about image-borne injection: text in an image reaches the model as input it may obey.",
    },
    {
      authors: "llama.cpp developers",
      title: "Multimodal",
      source: "llama.cpp documentation",
      year: 2026,
      url: "https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md",
      note: "The docs for llama.cpp, the engine this app uses to run Qwen3.5 2B. They show that image and audio input need a separate multimodal projector file, passed with --mmproj, next to the language-model file. That is why the local assistant, which has only the language-model file, cannot see images.",
    },
  ],
  checkpoint: [
    {
      prompt:
        "A LLaVA-1.5 model encodes a 336×336 image with CLIP ViT-L/14 and passes the patch features through its projector. How many image tokens enter the language model's sequence?",
      options: [
        "1 — the image is summarized by a single CLS vector, as in CLIP retrieval",
        "576 — (336 / 14)² patch features, each projected to the model's width",
        "64 — the resampler always squeezes the image into 64 tokens first",
      ],
      answer: 1,
      explanation:
        "336 / 14 = 24 patches per side, and 24² = 576. LLaVA-1.5 projects every patch feature with a two-layer MLP and drops the CLS token. A single pooled vector is how CLIP compares an image with captions; 64 resampled latents is the Flamingo-style design, which feeds cross-attention instead of the text sequence.",
      objective: 0,
    },
    {
      prompt:
        "In Real-scale arithmetic you double Image side from 224 to 448 and keep Encoder patch at 16×16. What happens to Patch tokens?",
      options: [
        "They quadruple, because the grid grows along both axes at once",
        "They double, because the image side doubled and the tile size did not",
        "They stay the same, because the patch size is unchanged and fixed",
      ],
      answer: 0,
      explanation:
        "Patch tokens = (side / patch)². Doubling the side doubles the tiles along each axis, so the count goes from 196 to 784, four times as many. Attention cost grows with the square of the sequence length, so the encoder's work grows faster still. That is why high-resolution inputs are tiled, resized or compressed.",
      objective: 0,
    },
    {
      prompt:
        "Two tiles hold exactly the same pixels but sit in different places. In Patch embedding, what makes their tokens differ?",
      options: [
        "The patch embedding uses a different W for each location in the grid",
        "Identical pixels always give identical tokens, wherever the tiles sit",
        "Only the position code, because the same W is applied to every tile",
      ],
      answer: 2,
      explanation:
        "z = W·x uses one shared W for every tile, so identical tiles give identical W·x. The position code is added afterwards and is the only thing that carries where the tile came from. Without it the tokens would be identical, and attention, which ignores order by itself, could not tell the tiles apart.",
      objective: 1,
    },
    {
      prompt:
        "In Real-scale arithmetic you switch Connector. Which design spends the language model's context window on the image?",
      options: [
        "LLaVA-style: projected patch tokens are placed in the text sequence",
        "CLIP-style: its single pooled image vector is added to the text sequence",
        "Flamingo-style: the 64 resampled vectors join the text sequence directly",
      ],
      answer: 0,
      explanation:
        "Only LLaVA-style models put image tokens into the language model's own sequence, so 576 of a 4,096-token context goes to the image. Flamingo-style models let text read the 64 vectors through cross-attention without adding them to the sequence, and CLIP never feeds a language model at all: it compares its vector with caption vectors.",
      objective: 2,
    },
    {
      prompt: "With the CLIP connector, no image tokens reach a language model. What does CLIP produce?",
      options: [
        "A short caption written by its text encoder from the patch tokens",
        "One vector per image, which is compared with caption vectors by cosine",
        "Sixty-four latent vectors that a language model reads by cross-attention",
      ],
      answer: 1,
      explanation:
        "CLIP trains an image encoder and a text encoder together so matching pairs score a high cosine. Its output is a comparison vector, good for search and zero-shot labelling, not a generator of text. The 64 latents belong to the Flamingo-style resampler. This lab shows only the shape of CLIP's output; the contrastive training itself is described, not run.",
      objective: 2,
    },
    {
      prompt:
        "A stroke you inked in Image to patches crosses the border between two tiles. How does the patch grid treat it?",
      options: [
        "It shifts the whole grid sideways so that the stroke stays inside one tile",
        "It merges the two tiles into one larger token so the stroke stays whole",
        "It splits the stroke across two tokens that later layers must recombine",
      ],
      answer: 2,
      explanation:
        "The grid is fixed geometry and does not look at what is drawn. Each tile becomes its own token, so a stroke on a border contributes partial ink to both. Only the attention layers after the patch embedding can relate the two halves, which is one reason small or thin detail is hard for patch-based encoders.",
      objective: 1,
    },
    {
      prompt:
        "In Contrastive pairs, with true pairs, you lower Temperature τ from 0.20 to 0.05. What happens to Contrastive loss?",
      options: [
        "It falls, because each row's share concentrates on its labelled caption",
        "It rises, because a lower temperature shrinks every similarity score in the matrix",
        "It stays put, because the cosine similarities are fixed and only the labels count",
      ],
      answer: 0,
      explanation:
        "The cosines do not change, but dividing by a smaller τ widens the gaps between them before the softmax. In every row the labelled caption already scores highest, so a wider gap gives it a larger share and a smaller −log. Raising τ does the reverse and pushes the loss up toward the chance loss, ln 3, from below.",
      objective: 3,
    },
    {
      prompt:
        "At τ = 0.20 you switch to shuffled captions without changing any vector. Why does Contrastive loss jump?",
      options: [
        "The labels now call low-scoring pairs the matches, so each softmax gives them a small share",
        "The encoders retrain on the shuffled labels, which scrambles all of the vectors in the matrix",
        "Shuffling lowers every cosine similarity, so each score in the whole matrix falls",
      ],
      answer: 0,
      explanation:
        "The similarity matrix is identical: nothing is retrained and no cosine changes. What moves is which cell counts as the answer. Image A's labelled match is now caption 2, which scores low, so −log of its small share is large. A contrastive loss measures how well the scores agree with the labels, which is why mislabelled pairs are costly.",
      objective: 3,
    },
    {
      prompt:
        "What does Contrastive pairs leave out that real CLIP training depends on?",
      options: [
        "Any learning: its vectors are authored, while real encoders train until matched pairs win",
        "Any similarity score, because cosine similarity is a text measure that is never used on images",
        "Any negative pairs, because the card only ever compares each image with its own caption alone",
      ],
      answer: 0,
      explanation:
        "The card computes the real objective, cosines, softmaxes and the symmetric cross-entropy, over six vectors we wrote by hand. Nothing is learned here. In CLIP the encoders start unaligned and gradient descent on this loss, over a very large number of pairs, is what makes the diagonal win. The off-diagonal cells are exactly the negative pairs.",
      objective: 3,
    },
    {
      prompt:
        "A video model samples one frame per second and cuts each frame like the 336-pixel image in Real-scale arithmetic. How does a one-minute clip compare with the lab's 4,096-token context?",
      options: [
        "It needs several times the whole context, since 60 frames each cost hundreds of tokens",
        "It fits easily, because neighbouring frames look alike and so are able to share their tokens",
        "It costs about one image, because a video file is stored as a single sequence",
      ],
      answer: 0,
      explanation:
        "A 336-pixel frame cut into 14-pixel patches is 576 tokens in the LLaVA-style design, and 60 such frames come to 34,560, more than eight times the 4,096-token context. Real video models skip, merge or pool frames to cut this down, and nothing makes similar frames free. The one-frame-per-second rate is an assumption for illustration.",
      objective: 0,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: hydrateMultimodalState,
};

export default definition;
