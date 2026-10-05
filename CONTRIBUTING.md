# Contributing to Discover AI

Thanks for wanting to help. Discover AI is meant to be a place where anyone can learn how modern AI works, and it gets better every time someone spots a confusing sentence, a wrong number, or a lab that could teach something more clearly.

You don't need to be an AI expert or a React developer to contribute. Some of the most useful help is a learner saying "I didn't follow this part."

## Ways to help

- **Report a bug.** Something crashes, a control doesn't respond, or a lab looks wrong.
- **Fix a lesson.** A fact is wrong, a sentence is hard to follow, a reference link is broken, or Plain and Standard mode say different things.
- **Make it easier to use.** Keyboard support, screen readers, contrast, text sizes, and reduced motion all matter here.
- **Improve a lab.** Make an interactive clearer, add a missing control, or fix a visual.
- **Add a new lab.** Please open an issue first (see [Adding a new lab](#adding-a-new-lab)).
- **Improve the teaching models.** The small models the labs run are rebuilt with the Python scripts in `models/`.

## Reporting a bug or a lesson mistake

[Open an issue](https://github.com/Fazmin/AILearningGuide/issues) and include:

- which lab (its name or number) and which step you were on
- whether you were reading in **Plain** or **Standard** mode
- what you did, what you expected, and what you saw instead
- whether you were in the browser or the desktop app, and your operating system
- a screenshot, if it helps show the problem

For a mistake in a lesson, quoting the sentence and saying what you think it should say is perfect.

**Security issues.** If the problem involves API keys, the keychain, the local model download, or anything else that could put someone's data at risk, please don't describe it in a public issue. Use **Report a vulnerability** on the repository's Security tab instead.

## Getting set up

You'll need the tools listed under [What you need](README.md#what-you-need) in the README. Then fork the repository and clone your fork:

```sh
git clone https://github.com/<your-username>/AILearningGuide.git
cd AILearningGuide
npm install
npm run dev
```

Open [http://localhost:1420](http://localhost:1420). Use `npm run tauri dev` instead if your change touches the desktop side (the local AI guide, the keychain, or the database).

`npm run check` also needs `python3` on your path for the model file check. It only uses the Python standard library, so you don't need to set up the `models/` environment for it.

## Making a change

1. **Create a branch** from `main` with a short name that says what it does, like `fix-attention-legend` or `plain-wording-backprop`.
2. **Keep it focused.** One fix or one idea per pull request is much easier to review than several mixed together.
3. **Run the checks** before you open the pull request:

   ```sh
   npm run check
   ```

   If you changed anything in `src-tauri/`, also run:

   ```sh
   cd src-tauri && cargo test
   ```

4. **Keep generated files in sync.** Some checks compare a generated file against the source, and fail if you forget to rebuild it:

   | If you changed... | Run this, then commit the result |
   | --- | --- |
   | lesson text, glossary terms, checkpoints, steps, or card explanations | `npm run index:search` |
   | npm dependencies in `package.json` | `npm run gen:licenses` |
   | a Rust crate in `src-tauri/Cargo.toml` | add an entry to `src/lib/licenses.manual.json` |

5. **Open a pull request.** Say what you changed and why, which labs it affects, and how you tested it. For anything you can see on screen, add a before and after screenshot. If you changed a lesson, mention whether you updated both Plain and Standard.

## Working on lessons

Each lab's lessons live in `src/modules/<lab>/content/`, with one file for each reading mode: `plain.mdx` and `standard.mdx`. The full rules are in the [content style guide](docs/CONTENT_STYLE_GUIDE.md). The ones that come up most often:

- **Both modes state the same facts.** Plain can leave out a detail, but it should never say something Standard contradicts.
- **Plain is written for a curious 13 to 16 year old.** Explain a word before you use it, use everyday words, and give a concrete example for every idea. `npm run check:readability` fails if a Plain lesson falls outside a grade 8 to 10 reading level or has a sentence over 35 words.
- **Every lesson has the same seven sections, in the same order.** The checks will tell you if one is missing.
- **Be honest about what's real.** Most labs are demonstrations, not real models. "Where it breaks" must say which numbers are made up or calculated by a formula, and separately say where the real technique fails.
- **Check references against the source.** Never add a reference from memory. Run `npm run check:references -- <lab-slug>` to make sure every link opens. It needs internet access, so it isn't part of `npm run check`.

## Working on labs

Each lab is a self-contained folder in `src/modules/<lab>/`, and the app finds it on its own. The [module authoring guide](docs/MODULE_AUTHORING.md) covers how labs work in detail. A few rules to know up front:

- **A lab only changes its own folder.** Its `Explore.tsx` imports only from `@app/module-sdk`, never from the app shell or the global store. If several labs need the same visual, add it to `src/module-sdk/`.
- **Don't rename a lab's ID once it's released.** Saved progress, snapshots, and the guide all use it.
- **Bump `stateVersion` when you change the shape of a lab's saved state,** and teach `hydrateState` to read the old shape so people's existing snapshots still open.
- **Every control works with a keyboard and has a visible label.** Never use color as the only way to tell two things apart.
- **Every card on the lab needs an explanation** in `card-info.ts`. `npm run check:modules` will tell you if one is missing.

`npm run visuals` opens a gallery of the shared building blocks (charts, heatmaps, arc diagrams, and so on), which is a good place to look before building something new. The [completion checklist](docs/MODULE_AUTHORING.md#completion-checklist) lists everything a lab needs before it's done.

For how the whole app fits together, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Adding a new lab

Please [open an issue](https://github.com/Fazmin/AILearningGuide/issues) before you start, so we can agree on the idea before you put in the work. Tell me:

- what the lab would teach, and what the learner would actually do in it
- which group it belongs in, and which existing labs it builds on
- whether it needs a real model, or can work as a demonstration

Once we've agreed, create the folder with:

```sh
npm run new:module -- my-new-lab
npm run check:modules
```

## Working on the teaching models

You don't need this to work on the app, because the trained models are already included. If you want to change or retrain them, follow [models/README.md](models/README.md). Checkpoints, exports, and datasets are ignored by git, so only commit the final files a lab actually uses, and run `npm run models:manifest` so `npm run check:assets` passes. In your pull request, explain how you produced the new files.

## Ground rules

- **Keep it private and offline.** Discover AI has no accounts and no tracking, and the lessons work without the internet. Please don't add analytics, telemetry, or new network calls. If your idea needs one, open an issue to talk about it first.
- **Keep it accessible.** Check your change in dark theme, at a large text size, and with reduced motion turned on.
- **Be kind.** Everyone here is learning something. Assume good intent, and keep feedback about the work, not the person.

## License

Discover AI is released under the [MIT License](LICENSE). By contributing, you agree that your contribution is released under the same license. If you add a library, font, model, or dataset made by someone else, make sure its license allows that, and make sure it appears in the license list under **Settings > About**.
