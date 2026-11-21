import { describe, expect, it } from "vitest";
import {
  ASI_CELL,
  BOTTLENECK_STOPS,
  CEILING,
  CLAIMS,
  DOMAINS,
  LEVELS,
  PROFILES,
  RETURN_STOPS,
  SYSTEMS,
  cellOf,
  columnOfCell,
  decodePlacement,
  decodeTags,
  encodePlacement,
  encodeTags,
  horizonAfter,
  levelOfCell,
  projectTime,
  researchEffort,
  takeoff,
  takeoffGap,
  verdict,
} from "./model";

describe("the levels table", () => {
  it("has six levels, twelve cells, and ASI in the top general cell", () => {
    expect(LEVELS.map((level) => level.name)).toEqual(["No AI", "Emerging", "Competent", "Expert", "Exceptional", "Superhuman"]);
    expect(ASI_CELL).toBe(11);
    expect(levelOfCell(ASI_CELL)).toBe(5);
    expect(columnOfCell(ASI_CELL)).toBe("general");
    expect(LEVELS[5].general).toContain("Artificial superintelligence");
    for (const level of [2, 3, 4]) expect(LEVELS[level].general).toBe("Not yet achieved");
  });

  it("places every system where Table 1 lists it", () => {
    const where = Object.fromEntries(SYSTEMS.map((system) => [system.id, [levelOfCell(system.cell), columnOfCell(system.cell)]]));
    expect(where).toEqual({
      calculator: [0, "narrow"],
      turk: [0, "general"],
      shrdlu: [1, "narrow"],
      chatgpt: [1, "general"],
      siri: [2, "narrow"],
      grammarly: [3, "narrow"],
      deepblue: [4, "narrow"],
      alphafold: [5, "narrow"],
    });
    // No system the table names is in the ASI cell.
    expect(SYSTEMS.some((system) => system.cell === ASI_CELL)).toBe(false);
  });

  it("stores placements as one letter per system and reads back anything", () => {
    const cells = SYSTEMS.map((system) => system.cell);
    expect(decodePlacement(encodePlacement(cells))).toEqual(cells);
    expect(decodePlacement("")).toEqual(SYSTEMS.map(() => -1));
    expect(decodePlacement("zz?")).toEqual(SYSTEMS.map(() => -1));
    expect(encodePlacement([cellOf(5, "general")])).toBe(`l${"-".repeat(SYSTEMS.length - 1)}`);
  });
});

describe("the jagged profile", () => {
  it("copies Table 1 of Hendrycks et al.: 27% and 57%", () => {
    expect(DOMAINS).toHaveLength(10);
    expect(verdict(PROFILES.gpt4, "sum")).toBeCloseTo(0.27, 12);
    expect(verdict(PROFILES.gpt5, "sum")).toBeCloseTo(0.57, 12);
  });

  it("gives three verdicts from one profile", () => {
    expect([verdict(PROFILES.gpt5, "sum"), verdict(PROFILES.gpt5, "weakest"), verdict(PROFILES.gpt5, "count")]).toEqual([0.57, 0, 0.2]);
    expect([verdict(PROFILES.gpt4, "weakest"), verdict(PROFILES.gpt4, "count")]).toEqual([0, 0]);
    expect(PROFILES.gpt5[DOMAINS.findIndex((domain) => domain.id === "MS")]).toBe(0);
  });
});

describe("faster, more, or better", () => {
  const base = { work: 100, speed: 1, copies: 1, inOrder: 0, world: 0 };

  it("takes one expert 100 years, a fast crowd about 3 seconds, and the world share sets a floor", () => {
    expect(projectTime(base).total).toBe(100);
    expect(projectTime({ ...base, speed: 1000 }).total * 365.25).toBeCloseTo(36.525, 9);
    const crowd = projectTime({ ...base, speed: 1000, copies: 1e6 }).total * 365.25 * 86400;
    expect(crowd).toBeCloseTo(3.156, 3);
    const floor = projectTime({ ...base, speed: 1000, copies: 1e6, world: 0.1 });
    expect(floor.world).toBe(10);
    expect(floor.total).toBeCloseTo(10, 6);
  });

  it("lets copies split only the thinking that does not have to happen in order", () => {
    const serial = projectTime({ ...base, inOrder: 1, copies: 1e6 });
    expect(serial.total).toBe(100);
    const half = projectTime({ ...base, inOrder: 0.5, copies: 1e6, speed: 10 });
    expect(half.inOrder).toBe(5);
    expect(half.parallel).toBeCloseTo(5e-6, 12);
  });
});

describe("the feedback loop", () => {
  const gap = (returns: number, bottleneck = 0) => Number(takeoffGap(takeoff(returns, bottleneck)).toFixed(1));

  it("matches people in year 6 in every run", () => {
    for (const returns of RETURN_STOPS) for (const bottleneck of BOTTLENECK_STOPS) expect(takeoff(returns, bottleneck).matchYear).toBeCloseTo(6, 1);
  });

  it("climbs from 1× to 100× in 30.7, 12.0 and 3.3 years at returns 1, 1.5 and 3, and never below 1", () => {
    expect(gap(1)).toBe(30.7);
    expect(gap(1.5)).toBe(12);
    expect(gap(3)).toBe(3.3);
    expect(takeoff(0.75, 0).beyondYear).toBe(-1);
    expect(takeoff(0.5, 0).beyondYear).toBe(-1);
    expect(Number(takeoff(1, 0).beyondYear.toFixed(1))).toBe(36.7);
    expect(Number(takeoff(1.5, 0).beyondYear.toFixed(1))).toBe(18);
    expect(Number(takeoff(3, 0).beyondYear.toFixed(1))).toBe(9.3);
  });

  it("agrees with the closed form where one exists", () => {
    // With no bottleneck and r = 2 from A = 1 at year 6: A(t) = 1 / (1 − 0.15 (t − 6)), so 100× at 6 + 0.99 / 0.15.
    expect(takeoff(2, 0).beyondYear).toBeCloseTo(6 + 0.99 / 0.15, 2);
    // r = 1: steady growth at 0.15 a year, so 100× at 6 + ln(100) / 0.15.
    expect(takeoff(1, 0).beyondYear).toBeCloseTo(6 + Math.log(100) / 0.15, 2);
  });

  it("is capped by a human-only share: 20% stops 100×, 1% only stops the runaway", () => {
    expect(researchEffort(1e9, 0.2)).toBeCloseTo(5, 6);
    expect(researchEffort(0.5, 0.2)).toBe(1);
    const capped = takeoff(1.5, 0.2);
    expect(capped.beyondYear).toBe(-1);
    expect(capped.offChartYear).toBe(-1);
    const small = takeoff(1.5, 0.01);
    expect(Number(small.beyondYear.toFixed(1))).toBe(19.8);
    expect(small.offChartYear).toBe(-1);
    expect(takeoff(1.5, 0).offChartYear).toBeGreaterThan(18);
  });

  it("never draws past the ceiling and stays bounded for any input", () => {
    for (const [returns, bottleneck] of [[3, 0], [99, -1], [Number.NaN, Number.NaN], [-5, 2]]) {
      const run = takeoff(returns, bottleneck);
      for (const value of run.ability) {
        expect(Number.isFinite(value)).toBe(true);
        expect(value).toBeLessThanOrEqual(CEILING);
      }
    }
  });
});

describe("forecasts", () => {
  it("carries the 50-minute horizon forward: 1.9 working months at 7 months, 27 hours at 12", () => {
    expect(horizonAfter(0, 7)).toBe(50);
    expect(horizonAfter(7 / 12, 7)).toBeCloseTo(100, 9);
    expect(horizonAfter(5, 7) / 60).toBeCloseTo(317, 0);
    expect(Number((horizonAfter(5, 7) / (167 * 60)).toFixed(1))).toBe(1.9);
    expect(Number((horizonAfter(5, 12) / 60).toFixed(1))).toBe(26.7);
    expect(Math.round(horizonAfter(5, 4) / 60 / 1000)).toBe(27);
  });
});

describe("the claims", () => {
  it("has two claims of each kind and stores answers as one letter each", () => {
    for (const kind of ["m", "d", "a", "f"]) expect(CLAIMS.filter((claim) => claim.kind === kind)).toHaveLength(2);
    const answers = CLAIMS.map((claim) => claim.kind);
    expect(decodeTags(encodeTags(answers))).toEqual(answers);
    expect(decodeTags("xyz")).toEqual(CLAIMS.map(() => null));
  });
});
