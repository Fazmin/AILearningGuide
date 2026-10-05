# Module authoring

Modules are convention-based plugins. Adding one must not require edits to the shell, sidebar, router, database, snapshots, or guide.

## Scaffold

```sh
npm run new:module -- mixture-of-experts
```

This creates:

```text
src/modules/mixture-of-experts/
  module.ts
  Explore.tsx
  card-info.ts
  assets.json
  content/
    plain.mdx
    standard.mdx
```

Run `npm run check:modules` after every structural or content change. Search
indexes every module, Learn section, glossary term, checkpoint, step, and card
explanation at package time; after content edits run `npm run index:search` so
`npm run check:search` stays green.

## Definition contract

`module.ts` exports one default `ModuleDefinition`. IDs and order values must be unique; the slug must exactly match the folder. Keep the ID stable after release because progress, snapshots, and chat use it as their foreign identifier.

Required metadata covers title, group, order, icon, prerequisite IDs, time estimate,
steps, one concrete `stepInstructions` entry per step, objectives, glossary,
checkpoint, and `stateVersion`. Each instruction should tell the learner which
control to use and which output to compare; step navigation must not silently reset
their experiment state.

`prerequisites` is what the lab's "Builds on" bar shows and what the Connections map draws,
so list the labs this one genuinely draws on, not every lab that came earlier. A link you
leave out is a connection the learner cannot see; a link you add that is not real sends them
back for nothing. Every prerequisite must have a lower `order` than the lab itself.

## Checkpoint questions

`checkpoint` is either one question object (legacy) or an array of them. Every module should
declare an array of **at least three** questions, and **every objective needs at least one
question**: tag each with the zero-based index of the objective it tests.

```ts
checkpoint: [
  {
    prompt: "…",                       // the key a question object must start with
    options: ["…", "…", "…"],          // at least three; no "all of the above" or "both A and B"
    answer: 1,                         // index into options as written
    explanation: "…",                  // explains the mechanism whether or not the learner was right
    objective: 0,                      // index into `objectives`
  },
  // …
],
```

Options are shuffled when shown (seeded by module id and question index), so the order you
write them in never gives the answer away and no option may refer to another by position.
Only a learner's **first** answer to each question counts toward their score; retrying a
question just reads the explanation. `npm run check:modules` and `registry.test.ts` enforce
the structure; the writing rules are in `CONTENT_STYLE_GUIDE.md`. Read questions in code
through `checkpointQuestions(module)` from the SDK, never `module.checkpoint` directly.

State is JSON-safe. `serializeState` produces the state placed in snapshots and guide context. `hydrateState` must merge defaults and handle malformed or older state. When changing a released state shape:

1. increment `stateVersion`;
2. detect older payloads in `hydrateState`;
3. translate them before merging current defaults;
4. keep a test fixture for the old payload.

## Explore component

`Explore.tsx` imports only from `@app/module-sdk`. It receives:

- `state` — current JSON-safe module state
- `setState(patch)` — persistent partial updates
- `currentStep` — the shell's active step index
- `mode` — `plain` or `standard`
- `narrate(message)` — screen-reader narration when verbose mode is enabled

Controls need visible labels, keyboard operation, and useful output text. A drag surface must have a slider, arrow-key behavior, or numeric alternative. Do not infer meaning from colour alone.

The attention module is the production worked example. It imports its own declared
assets and composes only SDK exports:

```tsx
import modelUrl from "./assets/tiny-transformer.onnx?url";
import {
  ArcDiagram,
  Heatmap,
  useTeachingTransformer,
  type ModuleContext,
} from "@app/module-sdk";

export default function Explore({ state }: ModuleContext) {
  const runtime = useTeachingTransformer({ modelUrl, inputIds: [/* … */] });
  // Aggregate model tensors into the view, with an explicitly labelled fallback.
  return <ArcDiagram /* … */ />;
}
```

For a new production module, add its reusable visualization to the SDK or compose
existing SDK primitives. Do not import shell components or the global store. Run
`npm run visuals` to review vector chips, formulas, arc diagrams, heatmaps, stage flows,
line charts, bar lists, and the step controller in the standalone component gallery.

A module that needs to train something can import `tiny-lm` from the SDK, which trains a
character bigram model in a few milliseconds. Keep the training call inside `useMemo`
keyed on the controls, store only controls and seeds in module state, and quote measured
values in the lesson rather than expected ones. If a measurement contradicts the standard
result, say so in the content and explain why the toy differs.

## Card explanations

Every `LabSurface` carries an info icon that opens a detailed explanation of that one
card on hover, on keyboard focus, and on click. The copy lives in `card-info.ts`, keyed
by the exact `label` passed to the surface:

```ts
import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Attention matrix": {
    title: "Every query–key route at once",
    summary: "One or two sentences giving the smallest accurate model of the mechanism.",
    whatYouSee: ["Decode each axis, mark, animation, and numeric readout."],
    howItWorks: ["State the formula the code actually evaluates."],
    controls: ["Name the control as it is labelled on screen, plus a comparison to run."],
    notice: ["Name the signal that distinguishes outcomes."],
    limits: [
      "In this lab: name what is synthetic, hard-coded, or closed-form.",
      "In general: name what fails in the production mechanism.",
    ],
  },
};

export default cardInfo;
```

`title`, `summary`, `whatYouSee`, `howItWorks`, and `limits` are required. `limits` must
carry both an `In this lab: ` bullet and an `In general: ` bullet, because the limits of
the interactive never substitute for the limits of the mechanism. Registration is
automatic: `registry.ts` globs `card-info.ts` and `ModuleWorkspace` provides it, so a
module still edits nothing outside its own folder.

`npm run check:modules` fails on a card with no explanation, an explanation whose key
matches no surface, a missing required field, or `limits` missing either prefix.
`src/modules/card-info.test.ts` additionally renders every module and asserts that the
number of info triggers equals the number of lab cards. `src/modules/attention/card-info.ts`
is the worked example.

Surface labels must be string literals so both checks can read them statically. A freshly
scaffolded module renders the `PlannedLab` placeholder and inherits its single
`Planned interactive` surface until its own lab replaces it.

## Content

Both MDX files share `objectives` and `terms` frontmatter and contain exactly these seven `##`
sections, in this order:

1. The big picture
2. What it is
3. Why it’s here
4. How to play with it
5. What to notice
6. Where it breaks
7. Going deeper

Plain mode is written at a grade 8 to 10 reading level, and additionally requires an analogy, a bold `Try this:` nudge, and a bold `A common mix-up:` correction. See "Plain mode" in `CONTENT_STYLE_GUIDE.md`; `npm run check:readability` measures the reading level.

Sections may use `###` subheadings, ordered and unordered lists, inline `code`, and fenced
code blocks; `.learn-copy` styles all of them. Depth comes from more short paragraphs and
named sub-topics, never from longer paragraphs.

Because most labs are illustrative rather than model-backed, `Where it breaks` must separate
what fails in the lab from what fails in production. Name the synthetic parts explicitly —
which readouts are closed-form formulas, which strings are fixed, and which declared assets
the current view does not read from — so a learner never mistakes a demonstration for
inference.

`Going deeper` should connect the toy to the current production mechanism and to the
neighboring modules by name, so the course reads as one system rather than sixteen unrelated
labs.

## Glossary

`glossary` is the module's reference surface, rendered in a scrollable panel in the Learn
header. Ship every term the two MDX files rely on — roughly 10 to 14 per module — and keep
`terms` frontmatter in both files identical to the glossary order.

Each definition is one or two sentences that state the mechanism and, where it matters, the
caveat that keeps the term from being over-read. "A normalized score" is a label; "the
normalized share controlling how much one value contributes, where every query's weights sum
to one, so routes compete" is a definition. Definitions are plain-text strings in `module.ts`,
so write notation in words rather than symbols that need escaping.

## References

`references` lists further reading under a References heading at the bottom of the Learn panel,
after the checkpoint, in both modes. Every module needs seven to fifteen, each with
`authors`, `title`, `source`, `year`, `url`, and a one-sentence `note` saying what the source adds
to this lesson, written in plain language because both modes share it.

Prefer primary papers, standard textbooks, and official documentation, and prefer recent sources
(2015 or later) where one covers the topic as well as an older one; keep a classic when the lesson
names it or it is the primary source for a claim. Check every entry against the source itself (the
publisher page, a DOI resolved through Crossref, or the arXiv abstract), never from memory. Tie each
note to a lab stop, term, or claim in the lesson.

Every link must open. Run `npm run check:references -- <slug>` (it needs network access and Chrome,
so it is not part of `npm run check`). Some publishers (APA, ACM, IEEE, Cell/Elsevier) answer
scripted requests with a bot wall; when a DOI fails, link the arXiv abstract, the proceedings page,
PubMed Central, or the Semantic Scholar record instead, and keep citing the published venue.

## Assets

Declare lazy module assets in `assets.json`:

```json
{
  "assets": [
    {
      "path": "assets/model.onnx",
      "bytes": 1843200,
      "sha256": "..."
    }
  ]
}
```

Paths are module-relative and cannot leave the folder. Validation fails if a declared file is missing. Production model loading should occur in a worker and expose loading, ready, and failure states to assistive technology.

## Completion checklist

- The module appears automatically in its group and hash route.
- Both explanation modes can switch without state loss.
- Every control works with keyboard input and has a label.
- Every lab card has a `card-info.ts` entry that names its mechanism and its limits.
- The visualization has a text summary or data alternative.
- Reset, snapshot, share, and hydration work.
- The checkpoint has one unambiguous answer and explanatory feedback.
- 200% text and reduced motion remain usable.
- `npm run check` passes.
