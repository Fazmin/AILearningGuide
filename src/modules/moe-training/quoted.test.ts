import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { TINY_VOCAB, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import cardInfo from "./card-info";
import definition from "./module";
import scenes from "./scenes";
import {
  BATCH,
  capacitySeats,
  denseSettings,
  LEARNING_RATE,
  MIXTRAL_8X7B,
  moeData,
  parameterCounts,
  routeCharacter,
  trainMoe,
  type MoeRun,
  type MoeSettings,
} from "./moe";
import { AUX_LADDER, BIAS_LADDER, CAPACITY_LADDER, defaultState, EPOCHS_MAX, SEED_MAX } from "./state";

/**
 * Pins every measured figure the three lessons, the card explanations and the scenes quote, computed
 * from the trainer, and checks that each quoted figure is printed where the lesson says it is.
 */
const here = path.dirname(fileURLToPath(import.meta.url));
const lesson = (mode: string) => readFileSync(path.join(here, "content", `${mode}.mdx`), "utf8");
const TEXT = { standard: lesson("standard"), plain: lesson("plain"), simple: lesson("simple"), card: JSON.stringify(cardInfo) };
type Place = keyof typeof TEXT;
const LESSONS: Place[] = ["standard", "plain", "simple"];
const quotedIn = (value: string, places: Place[]) => {
  for (const place of places) expect(TEXT[place], `"${value}" in ${place}`).toContain(value);
};

const BASE: MoeSettings = {
  experts: defaultState.experts,
  topK: defaultState.topK,
  balancing: "none",
  auxWeight: defaultState.auxWeight,
  biasStep: defaultState.biasStep,
  capacity: 0,
  epochs: defaultState.epochs,
  seed: defaultState.seed,
};
const cache = new Map<string, MoeRun>();
const run = (patch: Partial<MoeSettings> = {}) => {
  const settings = { ...BASE, ...patch };
  const key = JSON.stringify(settings);
  if (!cache.has(key)) cache.set(key, trainMoe(settings));
  return cache.get(key)!;
};
const SEEDS = [1, 2, 3, 4, 5, 6];
const dense = (seed = 1) => trainMoe(denseSettings({ epochs: 30, seed })).final;
const r2 = (value: number) => value.toFixed(2);
const r3 = (value: number) => value.toFixed(3);
const pct = (value: number) => `${Math.round(value * 100)}%`;
const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

describe("the setup the lessons describe", () => {
  it("trains on 1,181 pairs in batches of 16 at learning rate 0.5: 74 steps an epoch, 2,220 at the default 30 epochs", () => {
    expect(moeData().count).toBe(1181);
    expect(BATCH).toBe(16);
    expect(LEARNING_RATE).toBe(0.5);
    expect(defaultState.epochs).toBe(30);
    expect(run().steps).toBe(2220);
    expect(trainMoe({ ...BASE, epochs: 1 }).steps).toBe(74);
    expect(moeData().count - 73 * 16).toBe(13);
    expect(UNIFORM_CROSS_ENTROPY.toFixed(3)).toBe("3.401");
    expect([EPOCHS_MAX, SEED_MAX]).toEqual([40, 6]);
    expect([AUX_LADDER[0], AUX_LADDER.at(-1), BIAS_LADDER[0], BIAS_LADDER.at(-1)]).toEqual([0.001, 1, 0.001, 0.1]);
    expect(CAPACITY_LADDER).toEqual([0, 1, 1.25, 1.5, 2]);
    quotedIn("1,181", [...LESSONS, "card"]);
    quotedIn("2,220", [...LESSONS, "card"]);
    quotedIn("74", [...LESSONS, "card"]);
    quotedIn("3.401", [...LESSONS, "card"]);
    quotedIn("0.5", ["standard", "card"]);
  });

  it("routes t to E3 with a gate of 1.00 at the defaults, scoring 4.74 against others below −1.6, and guesses h at 39%", () => {
    const route = routeCharacter(run().params, TINY_VOCAB.indexOf("t"));
    expect(route.chosen).toEqual([2]);
    expect(r2(route.z[2])).toBe("4.74");
    expect(Math.max(route.z[0], route.z[1], route.z[3])).toBeLessThan(-1.6);
    expect(r2(route.gates[2])).toBe("1.00");
    const best = route.next.indexOf(Math.max(...route.next));
    expect(TINY_VOCAB[best]).toBe("h");
    expect(pct(route.next[best])).toBe("39%");
    quotedIn("4.74", ["plain", "simple"]);
    quotedIn("39%", ["plain", "simple"]);
  });
});

describe("1. Watch it collapse", () => {
  it("starts from a 12-23-42-23 split, 3.66 effective experts, and gives E3 100% after one epoch", () => {
    const history = run().history;
    expect(history[0].shares.map(pct)).toEqual(["12%", "23%", "42%", "23%"]);
    expect(r2(history[0].effective)).toBe("3.66");
    expect(history[1].shares.map(pct)).toEqual(["0%", "0%", "100%", "0%"]);
    expect(history[1].effective).toBeLessThan(1.001);
    quotedIn("3.66", [...LESSONS, "card"]);
    quotedIn("12", LESSONS);
    quotedIn("42", LESSONS);
  });

  it("ends at loss 2.197 against a dense baseline of 2.199, with a balance term of 3.75", () => {
    const final = run().final;
    expect(r3(final.loss)).toBe("2.197");
    expect(r3(dense().loss)).toBe("2.199");
    expect(r2(final.effective)).toBe("1.00");
    expect(r2(final.balance)).toBe("3.75");
    quotedIn("2.197", [...LESSONS, "card"]);
    quotedIn("2.199", [...LESSONS, "card"]);
    quotedIn("3.75", LESSONS);
  });

  it("reads 2.00, not 1.00, when a top-2 router collapses fully onto two experts", () => {
    const final = run({ experts: 8, topK: 2 }).final;
    expect(final.shares.filter((share) => share > 0.001).map(pct)).toEqual(["50%", "50%"]);
    expect(r2(final.effective)).toBe("2.00");
    quotedIn("2.00", ["card"]);
  });

  it("collapses with 8 experts within two epochs", () => {
    const history = run({ experts: 8 }).history;
    expect(history[1].effective).toBeGreaterThan(1.5);
    expect(history[2].effective).toBeLessThan(1.001);
    expect(scenes.find((scene) => scene.id === "one-expert-wins")?.look).toContain("within two epochs");
  });

  it("collapses in all six seeds with 2, 4 or 8 experts, within 0.04 of the dense baseline", () => {
    for (const experts of [2, 4, 8]) {
      for (const seed of SEEDS) {
        const final = run({ experts, seed }).final;
        expect(final.effective).toBeLessThan(1.005);
        expect(Math.abs(final.loss - dense(seed).loss)).toBeLessThan(0.04);
      }
    }
    quotedIn("0.04", LESSONS);
  }, 60_000);
});

describe("2. Add a balancing loss", () => {
  it("still collapses at 0.01 (1.00, loss 2.196, mean gate 0.82), fully in five of six seeds", () => {
    const final = run({ balancing: "aux", auxWeight: 0.01 }).final;
    expect([r2(final.effective), r3(final.loss), r2(final.meanGate)]).toEqual(["1.00", "2.196", "0.82"]);
    const collapsed = SEEDS.filter((seed) => run({ balancing: "aux", auxWeight: 0.01, seed }).final.effective < 1.01);
    expect(collapsed).toHaveLength(5);
    quotedIn("2.196", [...LESSONS, "card"]);
    quotedIn("0.82", ["card"]);
  }, 30_000);

  it("spreads to 3.89 at 0.1 with loss 2.131, mean gate 0.84 and a balance term of 1.04; 3.59 to 3.96 and below dense in all six seeds", () => {
    const final = run({ balancing: "aux", auxWeight: 0.1 }).final;
    expect([r2(final.effective), r3(final.loss), r2(final.meanGate), r2(final.balance)]).toEqual(["3.89", "2.131", "0.84", "1.04"]);
    const finals = SEEDS.map((seed) => run({ balancing: "aux", auxWeight: 0.1, seed }).final);
    expect(r2(Math.min(...finals.map((each) => each.effective)))).toBe("3.59");
    expect(r2(Math.max(...finals.map((each) => each.effective)))).toBe("3.96");
    finals.forEach((each, index) => expect(each.loss).toBeLessThan(dense(SEEDS[index]).loss));
    for (const value of ["3.89", "2.131", "0.84", "1.04", "3.59", "3.96"]) quotedIn(value, LESSONS);
  }, 30_000);

  it("stays even at 1 (3.84) but rises to 2.216 with a mean gate of 0.57, worse than 0.1 in all six seeds", () => {
    const final = run({ balancing: "aux", auxWeight: 1 }).final;
    expect([r2(final.effective), r3(final.loss), r2(final.meanGate)]).toEqual(["3.84", "2.216", "0.57"]);
    expect(final.loss).toBeGreaterThan(dense().loss);
    for (const seed of SEEDS) {
      expect(run({ balancing: "aux", auxWeight: 1, seed }).final.loss).toBeGreaterThan(run({ balancing: "aux", auxWeight: 0.1, seed }).final.loss);
    }
    for (const value of ["3.84", "2.216", "0.57"]) quotedIn(value, [...LESSONS, "card"]);
    // The window starts at 0.1, ten times Switch's 0.01; 0.3 is thirty times.
    expect(0.1 / 0.01).toBeCloseTo(10, 12);
    quotedIn("10 times", LESSONS);
  }, 30_000);
});

describe("3. Nudge with a bias", () => {
  it("is too slow at 0.001: one expert from epoch 1 to 21, then 1.63 effective experts and a loss of 2.275", () => {
    const slow = run({ balancing: "bias", biasStep: 0.001 });
    for (let epoch = 1; epoch <= 21; epoch += 1) expect(slow.history[epoch].effective).toBeLessThan(1.001);
    expect(slow.history[22].effective).toBeGreaterThan(1.2);
    expect([r2(slow.final.effective), r3(slow.final.loss)]).toEqual(["1.63", "2.275"]);
    expect(slow.final.loss).toBeGreaterThan(dense().loss);
    for (const value of ["1.63", "2.275", "21"]) quotedIn(value, LESSONS);
    quotedIn("1.63", ["card"]);
  });

  it("balances at 0.03: 4.00 effective experts, loss 2.131, mean gate 0.89, bars +3.30 +1.59 +2.52 +3.87", () => {
    const balanced = run({ balancing: "bias", biasStep: 0.03 });
    expect([r2(balanced.final.effective), r3(balanced.final.loss), r2(balanced.final.meanGate)]).toEqual(["4.00", "2.131", "0.89"]);
    expect(Array.from(balanced.params.bias, (value) => `+${r2(value)}`)).toEqual(["+3.30", "+1.59", "+2.52", "+3.87"]);
    for (const value of ["4.00", "0.89", "+3.30", "+1.59", "+2.52", "+3.87"]) quotedIn(value, LESSONS);
    quotedIn("0.89", ["card"]);
  });

  it("averages 2.138 over six seeds at 0.03 against 2.129 for aux weight 0.1, winning in two, inside the seed spread", () => {
    const bias = SEEDS.map((seed) => run({ balancing: "bias", biasStep: 0.03, seed }).final.loss);
    const aux = SEEDS.map((seed) => run({ balancing: "aux", auxWeight: 0.1, seed }).final.loss);
    expect(r3(mean(bias))).toBe("2.138");
    expect(r3(mean(aux))).toBe("2.129");
    expect(bias.filter((value, index) => value < aux[index])).toHaveLength(2);
    const gap = mean(bias) - mean(aux);
    expect(gap).toBeLessThan(Math.max(...aux) - Math.min(...aux));
    expect(gap).toBeLessThan(Math.max(...bias) - Math.min(...bias));
    quotedIn("2.138", [...LESSONS, "card"]);
    quotedIn("2.129", [...LESSONS, "card"]);
    // DeepSeek-V3's 0.001 against the step that works here, 0.03: thirty times.
    expect(0.03 / 0.001).toBeCloseTo(30, 9);
    quotedIn("30 times", ["standard", "plain", "simple", "card"]);
  }, 30_000);
});

describe("4. Cap each expert", () => {
  it("gives 5 seats per expert at 1.25, and a fully collapsed router would drop 11 of 16, 69%", () => {
    expect(capacitySeats(1.25, 16, 1, 4)).toBe(5);
    expect(pct(11 / 16)).toBe("69%");
    quotedIn("11 of", [...LESSONS, "card"]);
    quotedIn("69%", [...LESSONS, "card"]);
  });

  it("drops 40% at seed 1 with no balancing (loss 2.651), 69% at seed 4, 40% to 69% across six seeds", () => {
    const none = run({ capacity: 1.25 }).final;
    expect([pct(none.dropped), r3(none.loss)]).toEqual(["40%", "2.651"]);
    expect(run({ capacity: 1.25, seed: 4 }).final.effective).toBeLessThan(1.001);
    const dropped = SEEDS.map((seed) => run({ capacity: 1.25, seed }).final.dropped);
    expect(pct(dropped[3])).toBe("69%");
    expect([pct(Math.min(...dropped)), pct(Math.max(...dropped))]).toEqual(["40%", "69%"]);
    for (const value of ["40%", "2.651"]) quotedIn(value, [...LESSONS, "card"]);
  }, 30_000);

  it("drops 11% with aux weight 0.1 (loss 2.306) and 6% with bias step 0.03 (loss 2.210)", () => {
    const aux = run({ capacity: 1.25, balancing: "aux", auxWeight: 0.1 }).final;
    const bias = run({ capacity: 1.25, balancing: "bias", biasStep: 0.03 }).final;
    expect([pct(aux.dropped), r3(aux.loss)]).toEqual(["11%", "2.306"]);
    expect([pct(bias.dropped), r3(bias.loss)]).toEqual(["6%", "2.210"]);
    for (const value of ["11%", "2.306", "6%"]) quotedIn(value, [...LESSONS, "card"]);
    quotedIn("2.210", LESSONS);
  });
});

describe("5. Read the specialists", () => {
  it("groups the characters at aux weight 0.1, seed 1, and splits every corpus within ±0.01 of the overall split", () => {
    const final = run({ balancing: "aux", auxWeight: 0.1 }).final;
    const groups = final.shares.map((_, e) =>
      final.byCharacter
        .map((row, character) => (row[e] > 0 && row.indexOf(Math.max(...row)) === e ? TINY_VOCAB[character] : ""))
        .filter(Boolean)
        .join(""),
    );
    expect(groups).toEqual(["bcdfhmpstvy.", "aeouwz", "giklnqr", " "]);
    const gap = Math.max(
      ...final.byCorpus.flatMap((row) => {
        const total = row.reduce((sum, value) => sum + value, 0);
        return row.map((value, e) => Math.abs(value / total - final.shares[e]));
      }),
    );
    expect(r2(gap)).toBe("0.01");
    quotedIn("±0.01", [...LESSONS, "card"]);
    quotedIn("a, e, o and u", LESSONS);
    quotedIn("b, c, d, f, h, m, p, s, t, v, y and the full stop", LESSONS);
    quotedIn("g, i, k, l, n, q and r", LESSONS);
  });

  it("keeps every character's choices on one expert at top-1, so the top expert's share is 100%", () => {
    for (const patch of [{}, { balancing: "aux" as const, auxWeight: 0.1 }, { balancing: "bias" as const, biasStep: 0.03, seed: 3 }]) {
      for (const row of run(patch).final.byCharacter) {
        const total = row.reduce((sum, value) => sum + value, 0);
        if (total > 0) expect(Math.max(...row)).toBe(total);
      }
    }
    quotedIn("100%", [...LESSONS, "card"]);
  });

  it("splits every character's two choices evenly over two experts at top-2, so the top expert's share is 50%", () => {
    for (const patch of [{ experts: 8, topK: 2, balancing: "bias" as const, biasStep: 0.03 }, { experts: 4, topK: 2 }]) {
      for (const row of run(patch).final.byCharacter) {
        const used = row.filter((value) => value > 0);
        if (used.length === 0) continue;
        expect(used).toHaveLength(2);
        expect(used[0]).toBe(used[1]);
      }
    }
    quotedIn("50%", ["card"]);
  });
});

describe("6. Count what runs", () => {
  it("counts 275 for one expert, 740 and 290 at the defaults, and 1,360 with 310 or 460 active at 8 experts", () => {
    expect(parameterCounts(1, 1)).toMatchObject({ total: 275, active: 275 });
    expect(parameterCounts(4, 1)).toMatchObject({ embedding: 120, router: 20, perExpert: 150, total: 740, active: 290 });
    expect(parameterCounts(8, 1)).toMatchObject({ total: 1360, active: 310 });
    expect(parameterCounts(8, 2)).toMatchObject({ total: 1360, active: 460 });
    for (const value of ["275", "1,360", "310", "460"]) quotedIn(value, [...LESSONS, "card"]);
    for (const value of ["740", "290"]) quotedIn(value, ["plain", "simple", "card"]);
    quotedIn("120 + 20 + 600 = 740", ["plain", "simple"]);
    quotedIn("120 + 20 + 150 = 290", ["plain", "simple"]);
  });

  it("ends at 2.170 with 8 experts and 2.123 at top-2 under aux weight 0.1, against 2.131 with 4", () => {
    const eight = run({ experts: 8, balancing: "aux", auxWeight: 0.1 }).final.loss;
    const eightTop2 = run({ experts: 8, topK: 2, balancing: "aux", auxWeight: 0.1 }).final.loss;
    expect([r3(eight), r3(eightTop2)]).toEqual(["2.170", "2.123"]);
    expect(eightTop2).toBeLessThan(run({ balancing: "aux", auxWeight: 0.1 }).final.loss);
    for (const value of ["2.170", "2.123"]) quotedIn(value, [...LESSONS, "card"]);
  });

  it("does worse with 8 experts than 4 in five of six seeds at aux weight 0.1 and all six at bias step 0.03", () => {
    const worse = (patch: Partial<MoeSettings>) =>
      SEEDS.filter((seed) => run({ ...patch, experts: 8, seed }).final.loss > run({ ...patch, seed }).final.loss).length;
    expect(worse({ balancing: "aux", auxWeight: 0.1 })).toBe(5);
    expect(worse({ balancing: "bias", biasStep: 0.03 })).toBe(6);
    // Each of 8 experts sees about an eighth of the 1,181 pairs.
    expect(Math.round(1181 / 8)).toBe(148);
    quotedIn("five of six", [...LESSONS, "card"]);
  }, 60_000);

  it("prints Mixtral 8x7B's authored totals: 46.7B stored, 12.9B per token, an active share of 28%", () => {
    expect(MIXTRAL_8X7B).toMatchObject({ total: 46.7e9, active: 12.9e9, experts: 8, topK: 2 });
    expect(pct(MIXTRAL_8X7B.active / MIXTRAL_8X7B.total)).toBe("28%");
    for (const value of ["46.7", "12.9", "28%"]) quotedIn(value, [...LESSONS, "card"]);
    const reference = definition.references?.find((entry) => entry.url === "https://mistral.ai/news/mixtral-of-experts");
    expect(reference?.note).toContain("46.7B");
    expect(reference?.note).toContain("12.9B");
  });
});

describe("authored real-scale figures", () => {
  /** Figures typed in from papers, not measured: each must match the reference note that backs it. */
  const note = (url: string) => definition.references?.find((entry) => entry.url === url)?.note ?? "";
  it.each([
    ["https://arxiv.org/abs/2412.19437", ["256", "671B", "37B", "0.001", "14.3T"], ["256", "671", "37", "14.3"]],
    ["https://arxiv.org/abs/2409.02060", ["64", "7B", "1B", "0.01"], ["64", "7", "0.01"]],
    ["https://arxiv.org/abs/2201.05596", ["7.4B", "0.74B", "8 to 10"], ["7.4", "0.74", "8 to 10"]],
    ["https://jmlr.org/papers/v23/21-0998.html", ["α = 0.01"], ["0.01"]],
    ["https://arxiv.org/abs/2408.15664", ["1B and 3B", "not reproduce"], ["9.50", "9.56"]],
  ] as const)("matches the note for %s", (url, inNote, inLessons) => {
    for (const value of inNote) expect(note(url), url).toContain(value);
    for (const value of inLessons) quotedIn(value, LESSONS);
  });
});

describe("where it breaks", () => {
  it("weakens the collapse when the embeddings start twice as large: 1.5 to 1.9 effective experts in five of six seeds", () => {
    const effective = SEEDS.map((seed) => trainMoe({ ...BASE, seed }, { embed: 1, router: 0.1, expert: 0.1 }).final.effective);
    const kept = effective.filter((value) => value >= 1.5 && value <= 1.9);
    expect(kept).toHaveLength(5);
    expect(effective.filter((value) => value < 1.01)).toHaveLength(1);
    quotedIn("1.5 to 1.9", [...LESSONS, "card"]);
  }, 30_000);
});
