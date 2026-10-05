# Architecture

## Runtime shape

The React application owns presentation and ephemeral interaction state. Privileged or durable desktop work crosses the Tauri command boundary into Rust.

```text
Module folder ─┐
Module folder ─┼─ Vite auto-discovery ─ Module workspace ─ Zustand
Module folder ─┘          │                    │              │
                       MDX modes          narration bus       │
                                              │              │
                                          Tauri invoke ─ SQLite / keychain
```

In a normal browser, Tauri calls are skipped and Zustand persistence provides a development fallback.

## Frontend layers

`src/App.tsx` resolves the current theme, text scale, accessibility attributes, and hash route. It composes the custom title bar, sidebar, and one of the four top-level screens: Home, a lab, Settings, or Connections. `#/connections?focus=<slug>` opens the Connections map with a lab selected.

`src/components/ConnectionsScreen.tsx` draws every lab's prerequisites as one graph and lights up the route into and out of a selected lab. It reads only `prerequisites` and `order` from the registry, so a new or re-linked lab appears on the map with no change here. `src/lib/module-graph.ts` holds the pure parts: the transitive before/after sets for a lab, and a deterministic layered layout (rows by longest prerequisite chain, a reserved lane for each link that skips rows, crossing reduction, then straightening). The selected lab lives in the store as `connectionsFocusId` and is not persisted.

`src/components/ModuleWorkspace.tsx` is the stable lesson frame. It owns step navigation, Learn content, glossary, snapshots, sharing, completion, narration, and the scoped guide drawer. Two parts of it live in their own components: `Checkpoint.tsx` renders a module's list of questions and `PrerequisiteBar.tsx` shows the labs a module builds on with the learner's progress in each. `src/lib/learning-path.ts` holds the pure helpers behind the prerequisite bar, the Home screen's "next in the sequence" hint, and its completion state.

Checkpoints are a list of questions per module (`checkpointQuestions` in the SDK normalises the legacy single-question form). Options are shuffled at render with a seeded generator, and only a learner's first answer to each question counts. `src/store/checkpoint-progress.ts` holds the pure progress logic; the store persists a result per question and migrates version 1 progress, which stored one boolean, on load and on import. The desktop database already stores `checkpoint_score` as a real number from 0 to 1, so a fraction needs no migration there.

`src/module-sdk/` is the only shell-facing package a module may import. `types.ts`
defines `ModuleDefinition`, `ModuleContext`, and versioned state.
`visualizations.tsx` exposes the shared vector chip, heatmap, arc diagram,
step-through controller, formula readout, stage flow, line chart, bar list, and
instrument surfaces. `teaching-math.ts` holds the closed-form derivations, and
`tiny-lm.ts` holds the trainable character bigram model the train-and-adapt group uses.
Three helpers that more than one lab needs are exposed as namespaces so generic names
stay out of the index: `teachingEmbeddings` (the toy encoder and its search), `teachingEval`
(likelihood scoring, intervals, pass@k), and `teachingServing` (the serving arithmetic).
The modules that first needed them keep their import paths through thin re-exports.
Every module composes these primitives in its own `Explore.tsx`; `PlannedLab.tsx` is the
labelled placeholder a freshly scaffolded module renders until its lab is built.

`card-info.tsx` carries the per-card explanation contract: `LabSurface` resolves the
`CardInfo` registered for its label through context and renders an info trigger that
opens the explanation on hover, on keyboard focus, and on click. The panel is portalled
to `document.body` with viewport-aware placement so it is never clipped by the scrolling
explore column.

`src/modules/registry.ts` eagerly discovers `module.ts` definitions, the two MDX files, and `card-info.ts` in every module folder. The sidebar, learning map, route lookup, content mode, progress UI, and packaged search index all derive from this registry.

## Module lifecycle

1. Vite discovers `src/modules/*/module.ts`.
2. The registry verifies content imports and sorts definitions by `order`.
3. Opening a module merges its `initialState` with persisted state.
4. The shell passes state, a patch function, step, mode, and narration callback into `Explore`.
5. Snapshots use the module state contract and carry a `stateVersion`.
6. A future state migration belongs in that module's `hydrateState`; old snapshots remain readable.

Build-time validation in `scripts/check-modules.mjs` catches structural errors before Vite runs.

## Desktop data layer

`src-tauri/migrations/001_initial.sql` creates:

- settings
- module progress
- snapshots
- chat sessions and messages
- local model metadata
- cloud provider metadata

The schema intentionally accepts arbitrary string module IDs. `src-tauri/src/lib.rs` initializes SQLite in the application data directory, enables WAL and foreign keys, and exposes typed commands. Provider credentials use the `keyring` crate and are absent from SQLite, browser storage, and exports.

A second, read-only SQLite database is built before the app is packaged. `scripts/index-search.mjs` walks every module definition, Learn section, glossary term, checkpoint, lab step, card explanation, group, and settings surface, then writes `src/search/content-index.json`. `src-tauri/build.rs` compiles that catalog into an FTS5 index at `src-tauri/resources/content-index.sqlite`. The desktop `search_content` command queries the bundled index; the web development build uses the same documents in process. Selecting a hit jumps to the lab, Learn section, glossary term, step, instrument card, or settings tab.

## Local AI runtime

`src-tauri/src/local_ai.rs` owns the optional local topic guide. One setup action:

1. selects the pinned llama.cpp `b10991` archive for the current OS and CPU architecture;
2. checks free disk space and resumes any partial download;
3. verifies the release-provided engine SHA-256 before extraction;
4. downloads the immutable Qwen3.5 2B Q4_K_M revision and verifies its 1.28 GB file against the Hugging Face LFS SHA-256;
5. validates the GGUF header, records verification markers, and starts `llama-server` on a random loopback port;
6. waits for `/health` before reporting the model ready.

Both downloads require known SHA-256 hashes before execution, and the server uses a random loopback-only port. llama.cpp is used rather than MLX because the same manager works on macOS, Windows, and Linux.

The server binds only to `127.0.0.1`, is stopped with the application window, and auto-starts on the first local chat after a restart. Topic chat uses llama.cpp’s OpenAI-compatible streaming endpoint. Rust emits progress and token events to React; the model never receives network requests after installation.

## First-run onboarding

`src/components/Onboarding.tsx` is a three-step overlay mounted by `App.tsx`: the two explanation modes (a Plain and a Standard paragraph copied from the Next-token prediction lab, kept honest by `src/lib/onboarding-sample.test.ts`), the optional topic guide (local model or cloud provider, set up later or opened in Settings after the tour), and live accessibility controls. It holds no settings of its own: every control calls the same store action as Settings and the accessibility menu.

The rules are pure and live in `src/store/onboarding.ts`. The tour shows when the persisted `onboarded` flag is false **and** the learner has no progress, snapshots, or saved lab state. `onboardingReplay` is a second, non-persisted switch that Settings > About sets to reopen it. The persist version is 3; `migrate` composes the checkpoint migration with `migrateOnboardedFlag`, so state written by any earlier build marks that learner onboarded and a version 1 store passes through both steps. A brand-new device has no stored state, so migration never runs for it. `resetLearningData` also sets `onboarded`, because a reset empties the history that exempts a learner and must not push them back through first-run onboarding. The flag describes the device: exports do not write it and imports do not read it. App renders the tour only after the first route is applied, so a shared link never flashes it.

As a dialog it is `role="dialog"` with `aria-modal`, labelled by the current step heading; it moves focus to that heading on each step, traps Tab, treats Escape as skip, marks the rest of the app `inert` (except the title bar, so the frameless window can still be moved and closed), and returns focus to where it was when it closes.

## Licences and attributions

Settings > About lists every third-party component the app bundles or downloads: libraries, fonts, runtime and models, and data. `src/lib/licenses.ts` composes two checked-in files. `licenses.generated.json` is the npm half, produced by `scripts/gen-licenses.mjs` from package.json `dependencies`, followed through each installed package's own package.json `license` and LICENSE files (type-only packages excluded). `licenses.manual.json` is everything else (Rust crates, SQLite, the llama.cpp engine, the Qwen model, the teaching models, and the datasets), each entry naming the primary source its licence was read from. Where a licence could not be verified the entry says "Check the upstream project" with the URL rather than asserting one.

Each licence type's text appears once, offline, beneath the copyright lines of the components that use it. `npm run gen:licenses` rewrites the generated file after a dependency change; `npm run check:licenses` is in `npm run check` and fails if the file is stale, a direct npm dependency has no `npm` entry, or a direct Rust crate in `src-tauri/Cargo.toml` has no `crate` entry. `src/lib/licenses.test.ts` also ties the engine release and model revision in the list to `src-tauri/src/local_ai.rs`.

## State-sharing format

Share links contain URL-safe base64-encoded UTF-8 JSON:

```json
{
  "module": "attention",
  "version": 1,
  "state": { "head": 3, "selected": 8 }
}
```

The destination module validates its slug and hydrates through its version-aware contract. Shared state is never executed as code.

## Teaching model runtime

`src/model-runtime/` owns a single lazy Web Worker. It caches ONNX sessions by asset
URL, requests a WebGPU adapter before selecting the WebGPU execution provider, and
falls back to single-threaded WASM. Explicit Vite-managed `.mjs` and `.wasm` URLs keep
runtime initialization valid in development and packaged builds. Typed arrays are
transferred back to React without copying.

The attention module tokenizes up to 64 characters, runs the shipped two-layer model,
and aggregates character attention under word labels for display. Large Q/K/V and
attention tensors remain ephemeral; only controls and vector edits enter module state.
If runtime initialization fails, the UI identifies its deterministic fallback rather
than presenting simulated values as model inference.

`models/` contains the reproducible corpus, training, export, parity, precomputation,
and checksum pipeline. Reviewed outputs are copied into each module's `assets/`
folder. `asset-registry.ts` turns every declared module asset into a Vite-managed URL,
so packaged builds contain the files without eagerly fetching them.
`assets/manifest.json` enforces checksums and the 150 MiB budget.

## In-browser training runtime

`src/module-sdk/tiny-lm.ts` is a second, separate runtime for the labs that must retrain
a model between renders. It is a 30 by 30 character bigram logit table trained
synchronously in TypeScript, with no worker, no asset, and no network dependency:
minibatch SGD with schedules, clipping and frozen rows; low-rank adaptation with exact
merging; the DPO objective against a frozen reference; affine quantization at several
granularities; magnitude pruning; and rank-limited distillation. Every run is
deterministic in its seed, which is what lets modules keep only controls in state and
recompute weights inside a memo. The full pipeline is covered by
`tiny-lm.test.ts`, including the results the lessons claim.

## Production extension points

- Route the RNN, SAE, MNIST, tokenizer, and embedding outputs through task-specific worker adapters.
- Optionally move from verified runtime engine downloads to code-signed bundled sidecars for store distribution.
- Hydrate the Zustand store from SQLite at startup and keep browser persistence only under web development.
- Add content-level MDX imports for the shared formula and data-table components.
- Add provenance and quality gates for each additional published teaching asset.
