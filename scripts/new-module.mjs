import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modulesRoot = path.join(root, "src", "modules");
const slug = process.argv[2];

if (!slug || !/^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/.test(slug)) {
  console.error("Usage: npm run new:module -- <kebab-case-slug>");
  process.exit(1);
}

const folders = (await readdir(modulesRoot, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name);

let order = 1;
for (const folder of folders) {
  try {
    const source = await readFile(path.join(modulesRoot, folder, "module.ts"), "utf8");
    order = Math.max(order, Number(source.match(/\border:\s*(\d+)/)?.[1] ?? 0) + 1);
  } catch {
    // Invalid existing modules are reported by check:modules.
  }
}

const title = slug
  .split("-")
  .map((part) => `${part[0].toUpperCase()}${part.slice(1)}`)
  .join(" ");
const directory = path.join(modulesRoot, slug);

try {
  await mkdir(directory);
  await mkdir(path.join(directory, "content"));
} catch {
  console.error(`Module folder already exists: src/modules/${slug}`);
  process.exit(1);
}

await writeFile(path.join(directory, "Explore.tsx"), `import { PlannedLab, type ModuleContext } from "@app/module-sdk";

export default function Explore({ currentStep }: ModuleContext) {
  return (
    <PlannedLab
      heading=${JSON.stringify(title)}
      description="Describe the experiment this lab will run and the mechanism it makes visible."
      controls={["First control", "Second control", "Readout to compare"]}
      currentStep={currentStep}
    />
  );
}
`);

await writeFile(path.join(directory, "module.ts"), `import type { ModuleDefinition, ModuleState } from "@app/module-sdk";
import Explore from "./Explore";

const initialState: ModuleState = {};

const definition: ModuleDefinition = {
  id: ${JSON.stringify(`module-${String(order).padStart(2, "0")}-${slug}`)},
  slug: ${JSON.stringify(slug)},
  title: ${JSON.stringify(title)},
  group: "frontiers",
  order: ${order},
  icon: "CircleDot",
  accent: "#5a67d8",
  prerequisites: [],
  estimatedMinutes: 10,
  steps: ["Explore", "Notice", "Test"],
  stepInstructions: [
    "Change the primary control and predict what the output will do.",
    "Compare the important readouts and explain the relationship you observe.",
    "Push one control to an extreme and identify where the explanation stops working.",
  ],
  stateVersion: 1,
  tagline: "Describe the idea this interactive makes inspectable.",
  objectives: ["Explain the core mechanism", "Identify one important limitation"],
  glossary: [
    { term: "Term one", definition: "State the mechanism in one or two sentences, plus the caveat that keeps the term from being over-read." },
    { term: "Term two", definition: "Replace every placeholder. Keep the terms frontmatter in both MDX files identical to this list." },
    { term: "Term three", definition: "Define what a learner needs to read the interactive's labels and readouts." },
    { term: "Term four", definition: "Define the terms the standard-mode explanation introduces." },
    { term: "Term five", definition: "Define anything named in the failure modes." },
    { term: "Term six", definition: "Define anything named in the production mechanism." },
  ],
  // One question per objective, at least three. Keys in this order: prompt first.
  checkpoint: [
    {
      prompt: "Add a question that makes the learner predict what a control does.",
      options: ["Correct answer", "A misconception the lesson addresses", "Another misconception"],
      answer: 0,
      explanation: "Explain why the answer is correct and what the misconception gets wrong.",
      objective: 0,
    },
    {
      prompt: "Add a question that tests the second objective.",
      options: ["A misconception", "Correct answer", "Another misconception"],
      answer: 1,
      explanation: "Explain why the answer is correct and what the misconception gets wrong.",
      objective: 1,
    },
    {
      prompt: "Add a question about where the mechanism or the lab breaks.",
      options: ["A misconception", "Another misconception", "Correct answer"],
      answer: 2,
      explanation: "Explain why the answer is correct and what the misconception gets wrong.",
      objective: 1,
    },
  ],
  initialState,
  Explore,
  serializeState: (state) => JSON.stringify(state),
  hydrateState: (value) => {
    try {
      return { ...initialState, ...(JSON.parse(value) as ModuleState) };
    } catch {
      return { ...initialState };
    }
  }
};

export default definition;
`);

await writeFile(path.join(directory, "card-info.ts"), `import type { ModuleCardInfo } from "@app/module-sdk";

/**
 * Info-icon explanations for this module's lab cards, keyed by the exact \`label\`
 * of each <LabSurface>. Add one entry per card; check:modules reports the labels
 * this module renders and any that are still unexplained.
 *
 * Each entry owes: title, summary, whatYouSee, howItWorks, limits. \`limits\` needs
 * an "In this lab: " bullet naming what is synthetic here and an "In general: "
 * bullet naming what fails in production. See CONTENT_STYLE_GUIDE.md, and
 * src/modules/attention/card-info.ts for a worked example.
 */
const cardInfo: ModuleCardInfo = {};

export default cardInfo;
`);

const content = `---
objectives: ["Explain the core mechanism", "Identify one important limitation"]
terms: ["Term one", "Term two", "Term three", "Term four", "Term five", "Term six"]
---

## The big picture

Give an overview: what this module is about, why it matters, what the learner will be able to explain, and what the lab shows.

## What it is

Write a precise description.

## Why it’s here

Explain why the system needs this idea.

## How to play with it

**Try this:** Change one control and predict the result before observing it.

## What to notice

Name the signal learners should watch.

## Where it breaks

State a concrete limitation.

**A common mix-up:** Correct a likely misconception here.

## Going deeper

Connect the idea to an advanced mechanism.
`;

await writeFile(path.join(directory, "content", "plain.mdx"), content);
await writeFile(
  path.join(directory, "content", "standard.mdx"),
  content.replace("**Try this:** ", "").replace("\n**A common mix-up:** Correct a likely misconception here.\n", ""),
);
await writeFile(path.join(directory, "assets.json"), '{\n  "assets": []\n}\n');

console.log(`Created src/modules/${slug} as module ${order}.`);
console.log("Replace the PlannedLab placeholder with a purpose-built interactive, then run npm run check:modules.");
console.log("check:modules will list each LabSurface label that still needs a card-info.ts entry.");
