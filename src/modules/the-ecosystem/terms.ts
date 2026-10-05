/**
 * The fictional harbor-lm-7b release, as data the lab can reason over:
 * license clauses checked against what each use needs, a release tier computed
 * from which components ship, and two numbers computed from the card's own fields.
 */

export type Outcome = "yes" | "conditional" | "no";

export const CLAUSES = ["research", "commercial", "modify", "redistribute", "outputs"] as const;
export type Clause = (typeof CLAUSES)[number];

export const CLAUSE_NAMES: Record<Clause, string> = {
  research: "Non-commercial research use",
  commercial: "Commercial use",
  modify: "Modify or fine-tune",
  redistribute: "Redistribute the weights",
  outputs: "Use outputs to train another model",
};

export interface License {
  id: "research" | "community" | "apache";
  name: string;
  anyPurpose: boolean;
  clauses: Record<Clause, { outcome: Outcome; text: string }>;
}

/** Fictional licenses whose clauses are of the kinds found in real model licenses. */
export const LICENSES: readonly License[] = [
  {
    id: "research",
    name: "Research only",
    anyPurpose: false,
    clauses: {
      research: { outcome: "yes", text: "granted for non-commercial research and evaluation" },
      commercial: { outcome: "no", text: "no commercial use of the weights or a hosted service" },
      modify: { outcome: "conditional", text: "derivatives allowed for research only" },
      redistribute: { outcome: "no", text: "weights may not be redistributed" },
      outputs: { outcome: "no", text: "outputs may not be used outside research" },
    },
  },
  {
    id: "community",
    name: "Custom community",
    anyPurpose: false,
    clauses: {
      research: { outcome: "yes", text: "granted" },
      commercial: {
        outcome: "conditional",
        text: "allowed below a monthly-user cap, subject to an acceptable-use policy",
      },
      modify: { outcome: "yes", text: "allowed" },
      redistribute: { outcome: "conditional", text: "allowed if this license and its use policy travel with the copy" },
      outputs: { outcome: "no", text: "outputs may not be used to improve another model family" },
    },
  },
  {
    id: "apache",
    name: "Apache-2.0",
    anyPurpose: true,
    clauses: {
      research: { outcome: "yes", text: "granted" },
      commercial: { outcome: "yes", text: "granted for any purpose" },
      modify: { outcome: "yes", text: "allowed; modified files must say they were changed" },
      redistribute: { outcome: "conditional", text: "allowed with a copy of the license and the notices" },
      outputs: { outcome: "yes", text: "the license places no restriction on outputs" },
    },
  },
];

export interface Use {
  id: "research" | "product" | "finetune" | "outputs";
  name: string;
  needs: readonly Clause[];
}

export const USES: readonly Use[] = [
  { id: "research", name: "Research paper", needs: ["research"] },
  { id: "product", name: "Hosted product", needs: ["commercial"] },
  { id: "finetune", name: "Fine-tune and ship", needs: ["commercial", "modify", "redistribute"] },
  { id: "outputs", name: "Train on its outputs", needs: ["commercial", "outputs"] },
];

const RANK: Record<Outcome, number> = { yes: 0, conditional: 1, no: 2 };

export function checkUse(license: License, use: Use) {
  const rows = use.needs.map((clause) => ({ clause, ...license.clauses[clause] }));
  const worst = rows.reduce<Outcome>((acc, row) => (RANK[row.outcome] > RANK[acc] ? row.outcome : acc), "yes");
  return { rows, verdict: worst };
}

/* ------------------------------------------------------------- tiers */

export const TIERS = [
  { id: "api", name: "API only" },
  { id: "restricted", name: "Open weights, restricted license" },
  { id: "permissive", name: "Open weights, permissive license" },
  { id: "osaid", name: "Open source AI (OSAID 1.0 components)" },
] as const;
export type TierId = (typeof TIERS)[number]["id"];

export interface Release {
  license: License;
  trainingCode: boolean;
  dataInformation: boolean;
}

/**
 * The weights are downloadable, so the card is never API-only. OSAID 1.0 asks for
 * the parameters, the complete training and inference code, and data information
 * detailed enough to build a substantially equivalent system, all under terms that
 * allow use for any purpose.
 */
export function tierOf(release: Release): TierId {
  if (!release.license.anyPurpose) return "restricted";
  if (release.trainingCode && release.dataInformation) return "osaid";
  return "permissive";
}

export function missingForOsaid(release: Release) {
  const missing: string[] = [];
  if (!release.license.anyPurpose) missing.push("terms that allow any purpose");
  if (!release.trainingCode) missing.push("the complete training code");
  if (!release.dataInformation) missing.push("detailed data information");
  return missing;
}

export const SPECTRUM_ROWS: readonly { label: string; cells: Record<TierId, string> }[] = [
  { label: "Run on your own hardware", cells: { api: "no", restricted: "yes", permissive: "yes", osaid: "yes" } },
  { label: "Any purpose, including commercial", cells: { api: "per terms of service", restricted: "no", permissive: "yes", osaid: "yes" } },
  { label: "Modify and redistribute weights", cells: { api: "no", restricted: "limited", permissive: "yes", osaid: "yes" } },
  { label: "Complete training code", cells: { api: "no", restricted: "not required", permissive: "not required", osaid: "required" } },
  { label: "Data information to rebuild it", cells: { api: "no", restricted: "not required", permissive: "not required", osaid: "required" } },
  { label: "The training data itself", cells: { api: "no", restricted: "not required", permissive: "not required", osaid: "not always required" } },
];

/* ---------------------------------------------------- computed fields */

export const CARD = {
  parameters: 7e9,
  trainingTokens: 2e12,
  harnessItems: 40,
  harnessScore: 0.7,
  /** A100 dense BF16 peak, FLOP/s. */
  peakFlops: 312e12,
  utilization: 0.4,
  gpuWatts: 400,
};

/** Training compute ≈ 6·N·D (forward and backward pass, dense transformer). */
export const trainingFlops = (parameters: number, tokens: number) => 6 * parameters * tokens;

export function trainingEnergy(parameters: number, tokens: number, peak = CARD.peakFlops, utilization = CARD.utilization, watts = CARD.gpuWatts) {
  const flops = trainingFlops(parameters, tokens);
  const gpuHours = flops / (peak * utilization) / 3600;
  const megawattHours = (gpuHours * watts) / 1e6;
  return { flops, gpuHours, megawattHours };
}

/** Binomial standard error of an accuracy measured on n items. */
export const standardError = (accuracy: number, items: number) => Math.sqrt((accuracy * (1 - accuracy)) / items);
