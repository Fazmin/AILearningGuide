/// <reference types="node" />
/**
 * Test-only helpers: the shipped SAE and transformer read from the repository's assets, and the
 * transformer's TypeScript forward pass (the SDK's) turned into residuals and SAE codes, so every
 * figure the lesson quotes is measured with the code the lab runs, never copied from a report.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { teachingTransformer } from "@app/module-sdk";
import cache from "./assets/sae-top-activations.json";
import { PASSAGES } from "./fixture-passages";
import { encode, normalizeResidual, saeFromOnnx, WIDTH } from "./sae";
import { encodeText } from "./vocabulary";

const read = (path: string) => new Uint8Array(readFileSync(resolve(process.cwd(), path)));

export const sae = saeFromOnnx(read("src/modules/interpretability-features/assets/residual-sae.onnx"));
export const transformer = teachingTransformer.weightsFromOnnx(read("src/modules/attention/assets/tiny-transformer.onnx"));
export const SCALE = cache.normalization_scale;

export const DEFAULT_TEXT = "hath the king a good horse? KING HENRY: aye, he hath.";

export interface TextRun {
  text: string;
  /** Raw layer-2 residual, [characters x WIDTH] flattened. */
  residual: Float32Array;
  /** Normalized residual of each character. */
  x: Float32Array[];
  /** SAE code of each character (32 nonzero values). */
  features: Float32Array[];
}

/** The transformer's residual after block 2 for `text`, and the SAE code of every character. */
export function runText(text: string): TextRun {
  const run = teachingTransformer.runTransformer(transformer, encodeText(text));
  const residual = run.residuals[cache.layer];
  const x: Float32Array[] = [];
  const features: Float32Array[] = [];
  for (let position = 0; position < run.sequence; position += 1) {
    const row = normalizeResidual(
      teachingTransformer.rowOf(residual, position, WIDTH),
      cache.normalization_mean,
      cache.normalization_scale,
    );
    x.push(row);
    features.push(encode(sae, row));
  }
  return { text, residual, x, features };
}

let fixture: TextRun[] | null = null;
/** The twelve held-out passages (see fixture-passages.ts), run once per test file. */
export const fixtureRuns = () => (fixture ??= PASSAGES.map(runText));
