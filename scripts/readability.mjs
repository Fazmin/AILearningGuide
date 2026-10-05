// Reading-level report for the Plain lessons (src/modules/<slug>/content/plain.mdx).
//
//   node scripts/readability.mjs                 one line per module, hardest first
//   node scripts/readability.mjs <slug> [...]    detail for those modules: grade, longest
//                                                sentences, and the long words used most
//   node scripts/readability.mjs --check         exit 1 if a Plain lesson is outside GRADE_MIN to
//                                                GRADE_MAX or has a sentence over SENTENCE_MAX words
//
// The grade is Flesch-Kincaid: 0.39 × words per sentence + 11.8 × syllables per word − 15.59.
// Syllables are counted with a spelling heuristic, so treat a grade as ±0.5. Headings,
// frontmatter and fenced code are skipped; each list item counts as its own sentence; a
// number counts as one two-syllable word.
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

/** Plain lessons are written at a grade 8 to 10 reading level (CONTENT_STYLE_GUIDE.md). */
export const GRADE_MIN = 8;
export const GRADE_MAX = 10;
/** Sentences longer than this are listed in the detail report. */
export const SENTENCE_LIMIT = 30;
/** A sentence longer than this fails --check. */
export const SENTENCE_MAX = 35;

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modulesRoot = path.join(root, "src", "modules");

export function syllables(word) {
  const w = word.toLowerCase().replace(/[^a-z]/g, "");
  if (!w) return /\d/.test(word) ? 2 : 0;
  if (w.length <= 3) return 1;
  const trimmed = w
    .replace(/(?:[^laeiouy]es|[^laeiouy]ed|[^laeiouy]e)$/, (m) => m.slice(0, 1))
    .replace(/^y/, "");
  const groups = trimmed.match(/[aeiouy]{1,2}/g);
  return Math.max(1, groups ? groups.length : 1);
}

/** Lesson prose as a list of sentences, with markdown removed. */
export function sentencesOf(source) {
  const body = source
    .replace(/^---\n[\s\S]*?\n---\n/, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/^#{1,6} .*$/gm, "")
    .replace(/^\s*(?:[-*+]|\d+\.)\s+/gm, "\n\n")
    .replace(/^\s*\|.*\|\s*$/gm, (row) => (/^\s*\|[\s:|-]+\|\s*$/.test(row) ? "" : `\n\n${row.replace(/\|/g, ". ")}\n\n`))
    .replace(/<[^>]+>/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[`*_>]/g, "");
  return body
    .split(/\n\s*\n/)
    .flatMap((block) =>
      block
        .replace(/\s+/g, " ")
        .split(/(?<=[.!?])["”’)]?\s+(?=["“‘(]?[A-Z0-9])/)
        .map((s) => s.trim())
        .filter((s) => /[A-Za-z0-9]/.test(s)),
    );
}

export function wordsOf(sentence) {
  return sentence.split(/\s+/).map((w) => w.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "")).filter(Boolean);
}

export function measure(source) {
  const sentences = sentencesOf(source);
  let words = 0;
  let syl = 0;
  let complex = 0;
  const long = new Map();
  for (const sentence of sentences) {
    for (const word of wordsOf(sentence)) {
      const s = syllables(word);
      words += 1;
      syl += s;
      if (s >= 3) {
        complex += 1;
        const key = word.toLowerCase();
        long.set(key, (long.get(key) ?? 0) + 1);
      }
    }
  }
  const wps = words / Math.max(1, sentences.length);
  const spw = syl / Math.max(1, words);
  return {
    grade: 0.39 * wps + 11.8 * spw - 15.59,
    words,
    sentences: sentences.length,
    wordsPerSentence: wps,
    complexShare: complex / Math.max(1, words),
    longSentences: sentences
      .map((text) => ({ text, words: wordsOf(text).length }))
      .filter((s) => s.words > SENTENCE_LIMIT)
      .sort((a, b) => b.words - a.words),
    longWords: [...long.entries()].sort((a, b) => b[1] - a[1]),
  };
}

async function main() {
  const args = process.argv.slice(2);
  const check = args.includes("--check");
  const picked = args.filter((a) => !a.startsWith("--"));
  const folders = picked.length
    ? picked
    : (await readdir(modulesRoot, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => e.name);

  const rows = [];
  for (const folder of folders) {
    let source;
    try {
      source = await readFile(path.join(modulesRoot, folder, "content", "plain.mdx"), "utf8");
    } catch {
      continue;
    }
    rows.push({ folder, ...measure(source) });
  }
  rows.sort((a, b) => b.grade - a.grade);

  if (picked.length) {
    for (const r of rows) {
      console.log(`\n${r.folder}: grade ${r.grade.toFixed(1)} (target ${GRADE_MIN}–${GRADE_MAX}), ${r.words} words, ${r.sentences} sentences, ${r.wordsPerSentence.toFixed(1)} words/sentence, ${(r.complexShare * 100).toFixed(1)}% words of 3+ syllables`);
      console.log(`  sentences over ${SENTENCE_LIMIT} words: ${r.longSentences.length}`);
      for (const s of r.longSentences.slice(0, 15)) console.log(`   [${s.words}] ${s.text}`);
      console.log(`  most-used 3+ syllable words: ${r.longWords.slice(0, 40).map(([w, n]) => `${w}×${n}`).join(", ")}`);
    }
  } else {
    for (const r of rows) {
      console.log(`${r.grade.toFixed(1).padStart(5)}  ${r.wordsPerSentence.toFixed(1).padStart(5)} w/s  ${String(r.longSentences.length).padStart(3)} long  ${String(r.words).padStart(5)} words  ${r.folder}`);
    }
    const mean = rows.reduce((a, r) => a + r.grade, 0) / Math.max(1, rows.length);
    console.log(`mean grade ${mean.toFixed(2)} across ${rows.length} Plain lessons`);
  }

  if (check) {
    let failed = false;
    for (const r of rows) {
      if (r.grade < GRADE_MIN || r.grade > GRADE_MAX) {
        failed = true;
        console.error(`${r.folder}/plain.mdx: reading grade ${r.grade.toFixed(1)} is outside ${GRADE_MIN}–${GRADE_MAX}`);
      }
      for (const s of r.longSentences.filter((s) => s.words > SENTENCE_MAX)) {
        failed = true;
        console.error(`${r.folder}/plain.mdx: ${s.words}-word sentence: ${s.text}`);
      }
    }
    if (failed) process.exit(1);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
