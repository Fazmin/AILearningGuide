import { describe, expect, it } from "vitest";
import definition, { migrateEcosystemState } from "./module";
import { CARD, LICENSES, USES, checkUse, standardError, tierOf, trainingEnergy } from "./terms";

const license = (id: string) => LICENSES.find((item) => item.id === id)!;
const use = (id: string) => USES.find((item) => item.id === id)!;

describe("license checks", () => {
  it("allows research under every license", () => {
    for (const item of LICENSES) expect(checkUse(item, use("research")).verdict).toBe("yes");
  });

  it("reports the strictest clause a use needs", () => {
    expect(checkUse(license("research"), use("product")).verdict).toBe("no");
    expect(checkUse(license("community"), use("product")).verdict).toBe("conditional");
    expect(checkUse(license("apache"), use("product")).verdict).toBe("yes");
    expect(checkUse(license("community"), use("outputs")).verdict).toBe("no");
    expect(checkUse(license("apache"), use("outputs")).verdict).toBe("yes");
    expect(checkUse(license("apache"), use("finetune")).verdict).toBe("conditional");
  });
});

describe("release tiers", () => {
  it("never calls restricted terms open source, whatever else is released", () => {
    expect(tierOf({ license: license("research"), trainingCode: true, dataInformation: true })).toBe("restricted");
    expect(tierOf({ license: license("community"), trainingCode: true, dataInformation: true })).toBe("restricted");
  });

  it("needs code and data information on top of a permissive license", () => {
    expect(tierOf({ license: license("apache"), trainingCode: false, dataInformation: false })).toBe("permissive");
    expect(tierOf({ license: license("apache"), trainingCode: true, dataInformation: false })).toBe("permissive");
    expect(tierOf({ license: license("apache"), trainingCode: true, dataInformation: true })).toBe("osaid");
  });
});

describe("computed card fields", () => {
  it("estimates training compute and energy from 6ND", () => {
    const energy = trainingEnergy(CARD.parameters, CARD.trainingTokens);
    expect(energy.flops).toBeCloseTo(8.4e22, -19);
    expect(Math.round(energy.gpuHours / 1000)).toBe(187);
    expect(Math.round(energy.megawattHours)).toBe(75);
    // Within 2% of the 184,320 GPU-hours Meta reported for Llama 2 7B on 2T tokens.
    expect(Math.abs(energy.gpuHours - 184320) / 184320).toBeLessThan(0.02);
  });

  it("gives a 40-item harness about seven points of standard error", () => {
    expect(standardError(0.7, 40)).toBeCloseTo(0.0725, 4);
  });

  it("leaves a 3-point gap inside the noise, and the error shrinks only slowly near the ceiling", () => {
    // Checkpoint question: 73% against 70% on 40 items, and "about 6 points at 80%".
    expect(standardError(0.73, 40)).toBeCloseTo(0.0702, 4);
    expect(0.73 - 0.7).toBeLessThan(standardError(0.7, 40));
    expect(standardError(0.8, 40)).toBeCloseTo(0.0632, 4);
    expect(0.73 - 0.7).toBeLessThan(standardError(0.8, 40));
  });
});

describe("the-ecosystem state", () => {
  it("keeps a version 1 use and fills the new release fields", () => {
    expect(migrateEcosystemState({ use: "finetune" })).toEqual({
      use: "finetune",
      license: "research",
      code: "closed",
      data: "undisclosed",
    });
    expect(definition.hydrateState(JSON.stringify({ use: "nope", license: "apache" }))).toMatchObject({
      use: "research",
      license: "apache",
    });
  });
});
