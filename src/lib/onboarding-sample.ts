/**
 * The two paragraphs the welcome tour shows side by side. They are copied word for word from the
 * "What it is" section of the Next-token prediction lab (content/plain.mdx and content/standard.mdx),
 * so the sample is true to what the learner will actually read. `onboarding-sample.test.ts` fails if
 * either paragraph stops appearing in its source file; update the strings here when that happens.
 */
export const ONBOARDING_SAMPLE = {
  moduleSlug: "next-token-prediction",
  moduleTitle: "Next-token prediction",
  section: "What it is",
  plain:
    "Picture a video game that makes up its weather one hour at a time. Each hour it looks at the weather so far and sets odds for the next hour, such as 70% sun and 30% rain. Then it picks one at random, using those odds. A language model writes text the same way: it sets odds for every possible next piece of text, picks one, adds it to the end, and starts again.",
  standard:
    "The forward pass ends with a vector of **logits**, one unnormalized score per vocabulary entry. **Softmax** turns them into probabilities: `p_i = exp(z_i / T) / Σ exp(z_j / T)`. One token is drawn, appended to the context, and the calculation runs again on the longer sequence. That loop is what **autoregressive** means.",
} as const;
