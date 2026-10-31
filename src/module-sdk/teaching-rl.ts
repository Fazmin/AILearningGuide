/**
 * Reinforcement-learning helpers shared by the labs that train a policy against a reward.
 *
 * GRPO replaces a learned value function with the group itself: sample several attempts at the same
 * prompt, score them, and judge each against the group's mean. Dividing by the group's standard
 * deviation is the original recipe (Shao et al., 2024); Liu et al. (2025) argue the division biases
 * which prompts get weight, so `scale: "none"` keeps only the mean baseline.
 */
export const STD_EPSILON = 1e-6;

export type AdvantageScale = "std" | "none";

export interface AdvantageOptions {
  /** `"std"` divides by the group's population standard deviation; `"none"` subtracts the mean only. */
  scale?: AdvantageScale;
}

/** A group whose rewards are all equal carries no signal: every advantage is zero under either scale. */
export function groupAdvantages(rewards: ReadonlyArray<number>, { scale = "std" }: AdvantageOptions = {}): number[] {
  const count = rewards.length;
  const mean = rewards.reduce((sum, value) => sum + value, 0) / count;
  if (scale === "none") return rewards.map((value) => value - mean);
  const variance = rewards.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;
  const std = Math.sqrt(variance);
  return rewards.map((value) => (value - mean) / (std + STD_EPSILON));
}
