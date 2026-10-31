import { describe, expect, it } from "vitest";
import { groupAdvantages, STD_EPSILON } from "./teaching-rl";

/** The formula reasoning-models used before the helper moved here, kept to prove nothing changed. */
const before = (rewards: ReadonlyArray<number>) => {
  const count = rewards.length;
  const mean = rewards.reduce((sum, value) => sum + value, 0) / count;
  const variance = rewards.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;
  return rewards.map((value) => (value - mean) / (Math.sqrt(variance) + STD_EPSILON));
};

describe("group-relative advantages", () => {
  const groups = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0.95, 0.2, 0.75, 0.3, 0.1],
    [-0.05, -0.1, 0.85, 0.9, -0.15, 0.8],
    [2, 2, 2],
  ];

  it.each(groups.map((group) => [group.join(", "), group] as const))(
    "scales by the group std exactly as before for [%s]",
    (_label, group) => {
      expect(groupAdvantages(group)).toEqual(before(group));
    },
  );

  it("subtracts only the mean when scale is none", () => {
    expect(groupAdvantages([1, 0, 0, 1], { scale: "none" })).toEqual([0.5, -0.5, -0.5, 0.5]);
  });

  it("gives a flat group no signal under either scale", () => {
    expect(groupAdvantages([0.25, 0.25, 0.25, 0.25]).every((value) => value === 0)).toBe(true);
    expect(groupAdvantages([0.25, 0.25, 0.25, 0.25], { scale: "none" }).every((value) => value === 0)).toBe(true);
  });
});
