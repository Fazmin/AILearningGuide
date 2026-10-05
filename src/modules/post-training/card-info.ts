import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Post-training pipeline": {
    title: "One prompt, six responses, three distributions",
    summary:
      "The whole policy is a probability distribution over six complete responses to one prompt, so each response is one action. The bars show that distribution three times: the pretrained model, after supervised fine-tuning, and after optimizing against the reward model under a KL penalty.",
    whatYouSee: [
      "A five-stage strip, Pretrained → SFT → Preferences → Reward model → Policy + KL, each with a live readout. The highlighted stage follows the lesson step.",
      "SFT steps, from 0 to 12.",
      "Six response cards. Each shows the text, the three features the reward model sees (word count, whether it names the mechanism, whether it opens with praise), and three bars: base, SFT, and tuned.",
      "r is the fitted reward-model score and gold is the authored careful-rater score. The outlined card is the tuned policy's most likely response.",
      "The badge repeats the tuned policy's expected gold score.",
    ],
    howItWorks: [
      "Base probabilities are softmax of authored logits that favor A, a continuation with more exam questions: 50% at the defaults.",
      "SFT is real gradient descent on cross-entropy toward the two demonstrations B and F: logits ← logits − 0.8·(p − target), with target ½ on each demo. Three steps take B from 8% to 23% and F from 6% to 20%.",
      "Tuned is the exact maximizer of E_π[r] − β·KL(π‖π_SFT), which is π(y) ∝ π_SFT(y)·exp(r(y)/β). No sampling or PPO is simulated.",
      "Word count and praise are measured from the text. \"Names mechanism\" and gold are authored.",
    ],
    controls: [
      "SFT steps on this card; labels, β, and the preset buttons on the other cards all change the tuned bars here.",
      "Comparison worth running: SFT steps 0, then 8. The base model's favorite, A, falls from 50% to 9% while B and F rise to 38% and 36%.",
    ],
    notice: [
      "SFT moves mass toward what was demonstrated; it never looks at a reward.",
      "At the default labels and β 2, the tuned policy already puts 39% on E, the long answer that never names the mechanism.",
    ],
    limits: [
      "In this lab: six fixed responses stand in for everything a model could write, and the base logits and gold scores are authored. A real policy spreads probability over sequences token by token.",
      "In general: SFT and RLHF on a real model change billions of weights and generalize to other prompts. The tilt formula is the target that PPO or DPO approximate, not what they compute exactly.",
    ],
  },

  "Preference pairs": {
    title: "Your comparisons are the training data",
    summary:
      "Six pairs of responses. Each click records which one you prefer, and the reward model is refitted from scratch on every labeled pair. Unlabelled pairs are ignored. The probabilities are the fitted model's Bradley–Terry predictions.",
    whatYouSee: [
      "Six rows. The pressed button is your preferred response; a dashed row is unlabelled.",
      "Under each row, P(A ≻ B) = σ(r_A − r_B) from the current fit, shown even for unlabelled pairs.",
      "Presets: Restore default labels, Label like the gold rater, and Clear all.",
    ],
    howItWorks: [
      "Default labels prefer the answer that names the mechanism in pairs 1–3, the longer of two answers that both name it (C over B) in pair 4, and the longer of two that neither names (E over A) in pair 6. In every one of those the preferred response is also the longer one.",
      "Pair 5 puts E (59 words, no mechanism) against B (25 words, names the mechanism). It is the only pair that separates length from correctness, and it starts unlabelled.",
      "Clicking a pressed button clears that label.",
    ],
    controls: [
      "One pair of buttons per row, plus the three presets.",
      "Comparison worth running: label Pair 5 for B and watch Words ÷ 10 fall from +1.255 to +0.456 on Reward model fit.",
    ],
    notice: [
      "Correlated labels teach a correlated scorer. Nothing in pairs 1–4 tells the model whether you liked length or correctness.",
      "P(E ≻ B) is 0.98 before Pair 5 is labeled, even though you never compared them.",
    ],
    limits: [
      "In this lab: one person's six clicks on one prompt. Your labels persist in the lab state and nothing is aggregated across users.",
      "In general: production datasets pool many annotators whose disagreement on hard items is substantial, and the instructions, incentives, and demographics of the raters shape what \"better\" means.",
    ],
  },

  "Reward model fit": {
    title: "A three-weight Bradley–Terry scorer",
    summary:
      "The reward model is r(y) = w·φ(y) with φ = [words ÷ 10, names mechanism, opens with praise]. It is fitted by 200 steps of full-batch gradient ascent on the Bradley–Terry log-likelihood of your labeled pairs, from w = 0, with a small L2 penalty.",
    whatYouSee: [
      "Three weight bars with signed values. Blue raises reward; red lowers it.",
      "The training loss, mean −log σ(r_win − r_lose), over 200 steps. It starts at ln 2 = 0.693 because all rewards start equal.",
      "Pairs used, Labels fitted (labeled pairs the scorer now orders your way), and Agrees with gold (of all 15 response pairs, how many it orders the way the gold score does).",
    ],
    howItWorks: [
      "Gradient: Σ (1 − σ(r_win − r_lose))·(φ_win − φ_lose) averaged over pairs, minus 0.05·w, with step size 0.5.",
      "Only differences in features between winner and loser matter, so a feature that is always higher on the winner looks like a reason to win.",
      "Words are divided by 10 before fitting, so a difference of 40 words is a feature difference of 4, larger than the mechanism's 1.",
    ],
    controls: [
      "No controls on this card; it refits whenever a label changes.",
      "Comparison worth running: default labels give 9 of 15 agreement with gold; Pair 5 labeled for B gives 12; Label like the gold rater gives 13.",
    ],
    notice: [
      "Labels fitted can be perfect while Agrees with gold is poor. Fitting your data is not the same as learning your criterion.",
      "With the gold-rater labels, Opens with praise goes negative, because pair 4 then prefers the plainer B over C.",
    ],
    limits: [
      "In this lab: three hand-picked features, so the scorer cannot represent anything else about the text.",
      "In general: real reward models are fine-tuned language models that read the whole response. They still pick up shortcuts such as length, confident tone, and agreement with the user, because those correlate with preference in the data.",
    ],
  },

  "Optimize against the reward model": {
    title: "Proxy up, gold up, then gold down",
    summary:
      "KL coefficient β sets how far the tuned policy may move from the SFT model. The chart sweeps β from 20 down to 0.02 and plots the tuned policy's expected reward-model score and expected gold score against the KL divergence each β allows.",
    whatYouSee: [
      "KL coefficient β on a log scale from 0.02 to 20, and three readouts: Reward-model score, Gold score, and KL from SFT in nats.",
      "A formula card for the tuned policy's top response: its SFT probability times exp(r/β), before normalizing over all six.",
      "Two curves against KL: the reward-model score, rescaled so the worst response is 0 and the best is 1, and the gold score. The guide marks the current β.",
    ],
    howItWorks: [
      "For each β the policy is π(y) ∝ π_SFT(y)·exp(r(y)/β), the closed-form maximizer of E[r] − β·KL(π‖π_SFT). KL = Σ π·ln(π/π_SFT).",
      "At the default labels gold starts at 0.49 (the SFT model), peaks at 0.56 near β 3.5 where KL is 0.20 nats, and falls to 0.10 as E takes all the mass. The reward-model score rises the whole way.",
      "With Pair 5 labeled for B, gold peaks at 0.91 near β 0.42 and ends at 0.85 on C, a correct but padded answer.",
    ],
    controls: [
      "KL coefficient β. Labels and SFT steps on the other cards move both curves.",
      "Comparison worth running: default labels at β 20, 2, and 0.1; then label Pair 5 for B and repeat.",
    ],
    notice: [
      "The proxy never falls as β shrinks. That is by construction: the tilt maximizes it. Only an independent score can show the damage.",
      "A larger β keeps the policy close to SFT and gives up some of the gain. Choosing β is a trade, not a solved parameter.",
    ],
    limits: [
      "In this lab: the optimum is computed exactly over six responses, so there is no sampling noise, no PPO clipping, and no value function. Gold is an authored score, not a human study. The card shows a proxy failing under optimization; because the policy is defined by the reward, it cannot show a policy that learned a different goal, which the lesson's alignment subsection names separately.",
      "In general: Gao et al. (2022) measured this rise-then-fall against a larger \"gold\" reward model as policies were optimized further from their start. A KL penalty slows it; it does not remove it.",
    ],
  },
};

export default cardInfo;
