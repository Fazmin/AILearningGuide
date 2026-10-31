import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Router and experts": {
    title: "A router picks k experts; their gates weight the answer",
    summary:
      "A mixture-of-experts layer trained live on character pairs. A router scores every expert for the previous character, the top k by score plus balancing bias run, and each chosen expert's 30 next-character scores are scaled by its gate, the softmax of the router scores.",
    whatYouSee: [
      "Experts (1 to 8), Top-k (1 or 2), Epochs (1 to 40) and Seed (1 to 6). The kicker repeats them, and the badge counts optimizer steps: 74 per epoch.",
      "Example character picks one input. The table shows, for every expert, the trained router score z, the balancing bias b, z + b, the gate p, and whether the expert runs; the caption prints h, the character's four learned numbers and the constant 1.",
      "A formula row sums the chosen experts' outputs, each times its gate, and names the model's top guess for the next character with its probability.",
    ],
    howItWorks: [
      "h = [E[x], 1] with E a 30 × 4 embedding; z = h · R with R a 5 × N router; p = softmax(z). The top k experts by z + b run, lower index first on a tie; b is 0 unless Balancing is Bias update.",
      "logits = Σ over the chosen experts of p_e · (h · W_e), with each W_e a 5 × 30 map; the prediction is softmax(logits). The gates are not renormalised over the chosen experts, as in Switch Transformer.",
      "Training is minibatch SGD on cross-entropy, learning rate 0.5, batches of 16, over all 1,181 pairs of the three tiny corpora, from weights drawn from the seed. The router's bias row, the one the constant 1 multiplies, starts at zero.",
    ],
    controls: [
      "Experts, Top-k, Epochs and Seed retrain every run on the page. Top-k never exceeds Experts.",
      "Example character changes only which row of the trained router the table shows.",
      "Comparison worth running: at the defaults, read “t” with Balancing on None and then on Bias update at 0.03. The bias column fills in, but the gate of the chosen expert still comes from z alone.",
    ],
    notice: [
      "The constant 1 in h gives each expert a next-character bias that all of its tokens train, and gives the router a per-expert bias. Those two are the channel the collapse runs through.",
      "With Experts at 1 there is nothing to choose: the single gate is exactly 1 and the layer is the dense baseline.",
    ],
    limits: [
      "In this lab: each expert is one 5 × 30 linear map, not a feed-forward network, in a single layer with no attention or residual stream, and the router sees only the previous character, so every occurrence of a character takes the same route.",
      "In general: real experts are feed-forward blocks inside every transformer layer, the router reads a contextual hidden state, and gating varies: Mixtral renormalises the softmax over its top 2, and DeepSeek-V3 uses sigmoid scores normalised over the chosen 8.",
    ],
  },

  "Expert load over training": {
    title: "Collapse, drawn as it happens",
    summary:
      "Each expert's share of the router's choices at every epoch boundary, measured on the saved weights over all 1,181 pairs. With no balancing one line climbs to 100% within the first epoch and stays there: the router has handed every token to one expert.",
    whatYouSee: [
      "One line per expert, E1 to E8; the first four solid in four colours, the next four dashed, and a dotted line at the even share 1/N. The legend prints each line's final share.",
      "Metrics Effective experts, Loss and Dense baseline, and a callout that names a collapsed or idle expert, or the busiest and quietest when all are in use.",
      "A badge with the effective-expert count, marked as a warning when the run has collapsed.",
    ],
    howItWorks: [
      "share_e = choices of e ÷ (pairs × k), counted with the routing the weights give at that epoch boundary, before any capacity cap; epoch 0 is the starting weights.",
      "Effective experts = exp(−Σ s ln s) over the final shares: 1.00 for one expert, N for an even split.",
      "Loss is mean cross-entropy in nats over the same 1,181 pairs the run trained on. Dense baseline is the same recipe with one expert, whose gate is always exactly 1.",
    ],
    controls: [
      "Every control on the page that changes training moves these lines: Experts, Top-k, Epochs, Seed, Balancing and its weight or step, and Capacity factor.",
      "Comparison worth running: Balancing on None, then Auxiliary loss at 0.1. At seed 1 the first collapses onto E3 (1.00 effective experts, loss 2.197 against a dense 2.199), the second ends at 3.89 and a loss of 2.131.",
    ],
    notice: [
      "Collapse costs storage, not fit: a collapsed run's loss sits within a few hundredths of the dense baseline, because it is the dense model plus three idle experts.",
      "The starting split is fairly even, 3.66 effective experts at seed 1. The winner is made by training, which is the self-reinforcing loop Shazeer et al. (2017) describe.",
      "With Top-k at 2 a full collapse still uses two experts, so it reads 2.00, not 1.00.",
    ],
    limits: [
      "In this lab: how completely the toy collapses depends on its starting scale. Start the embeddings twice as large and no balancing leaves 1.5 to 1.9 effective experts in five of six seeds.",
      "In general: real runs measure load per layer and per batch on held-out traffic, and a router can drift after it looks balanced. Load is about choices, not quality: an even split can still hold experts that learned nothing distinct.",
    ],
  },

  "Balancing": {
    title: "Two ways to push the router back to an even split",
    summary:
      "An auxiliary loss adds α · N · Σ f·P to every batch, so its gradient pulls the router's scores toward balance, at a cost to the main loss when α is large. Bias update adds a per-expert bias only when choosing experts and steps it against each expert's load after every batch, leaving the gates alone.",
    whatYouSee: [
      "Balancing: None, Auxiliary loss or Bias update. Aux weight (0.001 to 1) appears with Auxiliary loss, Bias step (0.001 to 0.1) with Bias update.",
      "A table comparing the same settings with no balancing, the current balancing and the dense baseline: Effective experts, Loss and Mean gate, the average gate of the experts that ran.",
      "Metrics Balance term N·Σ f·P and Aux loss, and bars of each expert's bias after training, positive to the right.",
    ],
    howItWorks: [
      "Auxiliary loss: each batch adds α · N · Σ_e f_e · P_e, with f_e the share of the batch whose first choice is e, held fixed, and P_e the batch's mean gate for e. It is 1 when routing is even and N when one expert takes everything; its gradient reaches the router through P only.",
      "Bias update: after each batch, b_e += γ · sign(B·k/N − c_e), where c_e counts the batch's choices of e. b is added to z only to pick experts; the gate stays softmax(z).",
      "Balance term in the metric row is N · Σ f·P computed over all 1,181 pairs with the final weights.",
    ],
    controls: [
      "Balancing, Aux weight and Bias step retrain the balanced run. The No balancing row is a separate run that shares everything else.",
      "Comparison worth running: Aux weight 0.01, 0.1 and 1 at seed 1. Effective experts reads 1.00, 3.89 and 3.84; Loss 2.196, 2.131 and 2.216; Mean gate 0.82, 0.84 and 0.57.",
      "Comparison worth running: Bias step 0.001 and 0.03. The first is still at 1.63 effective experts after 2,220 steps, with a loss of 2.275; the second reaches 4.00 and 2.131.",
    ],
    notice: [
      "Only differences between biases matter: adding the same amount to every expert changes no choice. The bars drift upward together because sign() does not cancel.",
      "Bias update keeps the gates sharp, a mean gate of 0.89 at 0.03, while Aux weight 1 flattens them to 0.57.",
      "Over seeds 1 to 6 the bias method at 0.03 averages a loss of 2.138 against 2.129 for Aux weight 0.1. The gap is smaller than the spread between seeds.",
    ],
    limits: [
      "In this lab: the toy needs an aux weight about 10 times Switch's 0.01 to stay balanced, and the bias step that works, 0.03, is 30 times DeepSeek-V3's 0.001, because this run has 2,220 steps, not hundreds of thousands. The bias method does not beat the auxiliary loss here.",
      "In general: Wang et al. (2024) report the bias method giving lower perplexity and better balance than an auxiliary loss at 1B and 3B parameters, and DeepSeek-V3 adds a tiny sequence-level auxiliary loss as a backstop. Balance is measured per batch and per layer in real runs.",
    ],
  },

  "Capacity and dropped tokens": {
    title: "Seats per expert, and what happens to the overflow",
    summary:
      "A capacity factor gives each expert a fixed number of seats in every batch of 16, ⌈factor × 16 × k ÷ N⌉. Choices beyond an expert's seats are dropped; a token that loses every expert predicts uniformly, ln 30 = 3.401 nats. A cap limits the work an expert does but never sends a token anywhere else.",
    whatYouSee: [
      "Capacity factor: off, 1.00, 1.25, 1.50 or 2.00, and a formula row with the seats per expert when a cap is on.",
      "Metrics Dropped tokens, the share of the router's choices that found their expert full, Unserved tokens, the share of tokens that lost every expert they chose, and Loss. At Top-k 1 the two shares are the same.",
      "Bars of each expert's share of the router's choices, with the share its seats can serve; an expert that wants more than its room is coloured as a loss.",
    ],
    howItWorks: [
      "Within each batch every token's first choice is seated before any second choice, each in batch order, as in GShard. The last batch of each epoch holds 13 pairs and gets ⌈factor × 13 × k ÷ N⌉ seats.",
      "A dropped choice contributes nothing to the token's output and no gradient to that expert. A token with no expert has all-zero logits.",
      "The readouts come from scoring the final weights on all 1,181 pairs in corpus order, 16 at a time, with the same cap, so dropped tokens count in Loss at ln 30.",
    ],
    controls: [
      "Capacity factor retrains every run on the page with the cap in place.",
      "Comparison worth running: factor 1.25 with Balancing on None, then on Auxiliary loss at 0.1, at seed 1. Dropped tokens reads 40% and then 11%, Loss 2.651 and then 2.306; Bias update at 0.03 drops 6%.",
    ],
    notice: [
      "At 1.25 with 4 experts and top-1 each expert has 5 seats for 16 tokens, so a fully collapsed router drops 11 of 16, 69%. Seed 4 collapses that far; seed 1 keeps two experts and drops 40%.",
      "Even a balanced router drops some tokens, because a batch of 16 is never split exactly evenly.",
    ],
    limits: [
      "In this lab: dropped tokens predict uniformly because there is no residual connection, so the loss cost is far larger than in a real model, and the cap is applied when scoring the final weights as well as in training.",
      "In general: Switch Transformer and GShard pass an overflowed token to the next layer unchanged through the residual connection. Dropless systems such as MegaBlocks avoid capacity altogether, and DeepSeek-V3 drops no tokens in training or inference.",
    ],
  },

  "Specialisation": {
    title: "Which tokens each expert gets, and what that can mean",
    summary:
      "A grid of the router's choices: each row is a previous character or a corpus, each column an expert, and each cell the share of that row's choices sent to that expert. Character rows are clean by construction; corpus rows can only mirror their characters, because the router never sees which corpus a pair came from.",
    whatYouSee: [
      "Grid: By previous character (one row per character, with its pair count) or By corpus (Harbor, Recipes, Proverbs). Each row sums to 100%; darker cells hold more, and a cell's percentage shows on hover or focus.",
      "By previous character: Top expert's share per character, the pair-weighted average of each row's largest cell, and a list of the characters each expert receives. At Top-k 2 every character appears under two experts.",
      "By corpus: Largest gap from the overall split, the biggest difference between any corpus's share for an expert and that expert's overall share.",
    ],
    howItWorks: [
      "Counts come from the final routing over all 1,181 pairs, before any capacity cap, normalised within each row. With Top-k 2 each pair counts once for each of its two experts.",
      "The router's input is [E[x], 1] for the previous character x alone, so every occurrence of a character takes the same route, and a corpus's split is the average of its characters' rows.",
    ],
    controls: [
      "Grid switches the rows between characters and corpora. Balancing, Experts, Top-k and Seed change the routing itself.",
      "Comparison worth running: Auxiliary loss at 0.1, seed 1. E2 takes a, e, o and u (with w and z) and E4 takes space alone; By corpus, every corpus is within ±0.01 of the overall split. Change Seed and the grouping changes.",
    ],
    notice: [
      "Top expert's share per character reads 100% at Top-k 1 whatever the training, because routing depends only on the character. At Top-k 2 it reads 50%, since each character's two choices always go to two different experts.",
      "What training does decide is which characters share an expert. Grouping vowels together or giving space its own expert is a pattern in one run, not a law.",
    ],
    limits: [
      "In this lab: the even corpus split is forced by the router's input, not discovered, and character purity is guaranteed. One layer, 1,181 pairs and seeded runs make any grouping fragile.",
      "In general: real routers read a contextual hidden state, so specialisation is an empirical question. Mixtral's analysis finds little specialisation by topic, ST-MoE finds encoder experts for punctuation, verbs and names, and OLMoE finds an expert that takes nearly all arXiv tokens.",
    ],
  },

  "Total vs active parameters": {
    title: "Stored weights against the weights a token runs",
    summary:
      "Total parameters count everything stored: the embedding, the router and every expert. Active per token counts what one token uses: the embedding, the router and its k experts. Adding experts grows the total; only Top-k grows the active count.",
    whatYouSee: [
      "Metrics Total parameters, Active per token and Loss for the current run.",
      "Formula rows that add up both counts from the embedding (30 × 4), router (5 × N) and experts (N or k times 5 × 30).",
      "A table with this lab, the one-expert dense baseline and Mixtral 8x7B, with each model's active share; the Mixtral row is published, not computed.",
    ],
    howItWorks: [
      "Total = 30·4 + 5·N + N·5·30 and active = 30·4 + 5·N + k·5·30: 740 and 290 at the defaults, 275 for one expert, and 1,360 with 460 active at 8 experts and top-2.",
      "Mixtral 8x7B's figures are authored constants from Mistral AI's announcement: 46.7B parameters in total and 12.9B used per token, an active share of 28%. Its attention and embeddings are shared, so it is not 8 × 7B.",
    ],
    controls: [
      "Experts and Top-k on Router and experts change both counts; Loss comes from the current run.",
      "Comparison worth running: Experts 1, 4 and 8 at Top-k 1, then Top-k 2. Total goes 275, 740, 1,360; Active per token 275, 290, 310, and 460 at top-2.",
    ],
    notice: [
      "More experts is not automatically better here. At Aux weight 0.1, 8 experts end worse than 4 in five of six seeds, and with Bias step 0.03 in all six, because each expert now sees about an eighth of the pairs.",
      "Top-2 runs twice the expert weights per token: at 8 experts and Aux weight 0.1 it lowers the loss from 2.170 to 2.123 at seed 1.",
    ],
    limits: [
      "In this lab: the counts are formulas over a 1,360-weight toy that fits in any memory, so nothing here measures the memory or communication cost of experts.",
      "In general: memory must hold every expert, so large models spread them across GPUs and send each token to the device holding its experts, expert parallelism. That adds all-to-all communication and makes uneven load a speed problem as well as a quality one.",
    ],
  },
};

export default cardInfo;
