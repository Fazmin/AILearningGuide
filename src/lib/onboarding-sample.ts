/**
 * The three paragraphs the welcome tour shows side by side. They are copied word for word from the
 * "What it is" section of the Next-token prediction lab (content/simple.mdx, content/plain.mdx and
 * content/standard.mdx), so the sample is true to what the learner will actually read.
 * `onboarding-sample.test.ts` fails if any paragraph stops appearing in its source file; update the
 * strings here when that happens.
 */
export const ONBOARDING_SAMPLE = {
  moduleSlug: "next-token-prediction",
  moduleTitle: "Next-token prediction",
  section: "What it is",
  simple:
    "A step called **softmax** turns the logits into odds that are all positive and add up to 100%. It follows one rule. Take two characters whose scores are one point apart. The higher one gets about 2.7 times the odds of the lower one. Softmax shares out the full 100% so that this rule holds for every pair of characters.",
  plain:
    "A step called **softmax** turns the logits into odds that are all positive and add up to 100%. It follows one rule: every extra point of score makes a character about 2.7 times as likely as a character with one point less. Softmax then shares out the full 100% so that this rule holds for every pair of characters.",
  standard:
    "The forward pass ends with a vector of **logits**, one unnormalized score per vocabulary entry. **Softmax** turns them into probabilities: `p_i = exp(z_i / T) / Σ exp(z_j / T)`. One token is drawn, appended to the context, and the calculation runs again on the longer sequence. That loop is what **autoregressive** means.",
} as const;
