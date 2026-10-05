import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

export const SEARCH_INDEX_VERSION = 1;
export const SEARCH_INDEX_SCHEMA = "search-index-schema:1";

const REQUIRED_LEARN_SECTIONS = [
  "The big picture",
  "What it is",
  "Why it’s here",
  "How to play with it",
  "What to notice",
  "Where it breaks",
  "Going deeper",
];

const SETTINGS_TABS = [
  ["appearance", "Appearance", "Theme, text size, and how the learning workspace looks."],
  ["accessibility", "Accessibility", "Higher contrast, reduced motion, verbose narration, and colour-vision palettes."],
  ["explanations", "Explanation mode", "Switch between plain-language and standard technical writing."],
  ["local", "Local model", "Private offline topic chat through a llama.cpp sidecar."],
  ["cloud", "Cloud providers", "Choose a provider and model, store an API key in the operating system keychain, and test the connection for optional topic chat."],
  ["data", "Data", "Export, import, or reset progress, snapshots, and saved lab states."],
  ["about", "About", "Version, architecture, and what Discover AI is. Replay the welcome tour. Licences and attributions for every library, font, model, and dataset: open-source licences (MIT, Apache, SIL Open Font), copyright notices, Tauri, ONNX Runtime, llama.cpp, Qwen, GloVe, MNIST, Tiny Shakespeare."],
];

export function slugify(value) {
  return String(value)
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[’‘ʻʼ`']/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function unescapeQuoted(value) {
  try {
    return JSON.parse(`"${value}"`);
  } catch {
    return value
      .replace(/\\n/g, "\n")
      .replace(/\\t/g, "\t")
      .replace(/\\"/g, '"')
      .replace(/\\\\/g, "\\");
  }
}

function quotedStrings(source) {
  return [...String(source ?? "").matchAll(/"((?:\\.|[^"\\])*)"/g)].map((match) => unescapeQuoted(match[1]));
}

function field(source, name) {
  return source.match(new RegExp(`\\b${name}:\\s*"((?:\\\\.|[^"\\\\])*)"`))?.[1];
}

function unquotedField(source, name) {
  const quoted = field(source, name);
  return quoted === undefined ? undefined : unescapeQuoted(quoted);
}

function numericField(source, name) {
  const value = source.match(new RegExp(`\\b${name}:\\s*(-?\\d+)`))?.[1];
  return value === undefined ? undefined : Number(value);
}

function bracketBlock(source, name, until) {
  const pattern = new RegExp(`\\b${name}:\\s*\\[([\\s\\S]*?)\\],\\s*${until}`);
  return source.match(pattern)?.[1] ?? "";
}

function objectBlock(source, name, until) {
  const pattern = new RegExp(`\\b${name}:\\s*\\{([\\s\\S]*?)\\},\\s*${until}`);
  return source.match(pattern)?.[1] ?? "";
}

function collapseWhitespace(value) {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function mdxToPlain(source) {
  return collapseWhitespace(
    String(source ?? "")
      .replace(/^---[\s\S]*?---\n/, "")
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/__([^_]+)__/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/^#{1,6}\s+/gm, "")
      .replace(/^[-*]\s+/gm, "")
      .replace(/^\d+\.\s+/gm, ""),
  );
}

function splitLearnSections(source) {
  const body = String(source ?? "").replace(/^---[\s\S]*?---\n/, "");
  const sections = [];
  const matches = [...body.matchAll(/^## (.+)$/gm)];
  matches.forEach((match, index) => {
    const title = collapseWhitespace(match[1]);
    const start = (match.index ?? 0) + match[0].length;
    const end = matches[index + 1]?.index ?? body.length;
    sections.push({
      title,
      body: mdxToPlain(body.slice(start, end)),
    });
  });
  return sections;
}

function extractStringField(block, name) {
  const sameLine = block.match(new RegExp(`^ {4}${name}:\\s*"((?:\\\\.|[^"\\\\])*)"`, "m"));
  if (sameLine) return unescapeQuoted(sameLine[1]);
  const wrapped = block.match(new RegExp(`^ {4}${name}:\\s*\\n\\s*"((?:\\\\.|[^"\\\\])*)"`, "m"));
  return wrapped ? unescapeQuoted(wrapped[1]) : "";
}

function extractStringArray(block, name) {
  const body = block.match(new RegExp(`^ {4}${name}: \\[([\\s\\S]*?)^ {4}\\]`, "m"))?.[1] ?? "";
  return quotedStrings(body);
}

function parseCardInfo(source) {
  const declarations = [...source.matchAll(/^ {2}"((?:[^"\\]|\\.)*)":\s*\{$/gm)];
  return declarations.map((declaration, position) => {
    const next = declarations[position + 1];
    const block = source.slice(declaration.index, next ? next.index : source.length);
    const label = unescapeQuoted(declaration[1]);
    const title = extractStringField(block, "title") || label;
    const parts = [
      title,
      extractStringField(block, "summary"),
      ...extractStringArray(block, "whatYouSee"),
      ...extractStringArray(block, "howItWorks"),
      ...extractStringArray(block, "controls"),
      ...extractStringArray(block, "notice"),
      ...extractStringArray(block, "limits"),
    ];
    return {
      label,
      title,
      body: collapseWhitespace(parts.filter(Boolean).join(" ")),
    };
  });
}

function parseGroups(source) {
  const groups = [];
  const block = source.match(/export const moduleGroups[^=]*=\s*\[([\s\S]*)\];/)?.[1] ?? "";
  for (const entry of block.matchAll(/\{\s*id:\s*"([^"]+)",\s*title:\s*"((?:\\.|[^"\\])*)",\s*description:\s*"((?:\\.|[^"\\])*)",?\s*\}/g)) {
    groups.push({
      id: entry[1],
      title: unescapeQuoted(entry[2]),
      description: unescapeQuoted(entry[3]),
    });
  }
  return groups;
}

function parseGlossary(source) {
  const block = bracketBlock(source, "glossary", "checkpoint:");
  const terms = [];
  for (const match of block.matchAll(
    /(?:["']?term["']?):\s*"((?:\\.|[^"\\])*)"\s*,\s*(?:["']?definition["']?):\s*"((?:\\.|[^"\\])*)"/g,
  )) {
    terms.push({
      term: unescapeQuoted(match[1]),
      definition: unescapeQuoted(match[2]),
    });
  }
  return terms;
}

function parseCheckpointBlock(block) {
  return {
    prompt: unquotedField(block, "prompt") ?? "",
    options: quotedStrings(block.match(/\boptions:\s*\[([\s\S]*?)\]/)?.[1] ?? ""),
    answer: numericField(block, "answer"),
    explanation: unquotedField(block, "explanation") ?? "",
    objective: numericField(block, "objective"),
  };
}

/**
 * A module declares `checkpoint: { … }` (one question) or `checkpoint: [{ … }, …]`. Each
 * question object must start with its `prompt` key so the list can be split textually.
 */
export function parseModuleCheckpoints(source) {
  const list = source.match(/\bcheckpoint:\s*\[([\s\S]*?)\],\s*initialState/)?.[1];
  if (list !== undefined) {
    const blocks = list.split(/(?=\{\s*prompt:)/).filter((chunk) => /\bprompt:/.test(chunk));
    return { isList: true, questions: blocks.map(parseCheckpointBlock) };
  }
  const single = objectBlock(source, "checkpoint", "initialState");
  return { isList: false, questions: single ? [parseCheckpointBlock(single)] : [] };
}

function documentRecord(partial) {
  return {
    subtitle: "",
    body: "",
    moduleId: null,
    moduleTitle: null,
    moduleSlug: null,
    groupId: null,
    mode: null,
    section: null,
    stepIndex: null,
    term: null,
    cardLabel: null,
    screen: null,
    settingsTab: null,
    rankBoost: 1,
    ...partial,
  };
}

export async function collectSearchSources(root) {
  const modulesRoot = path.join(root, "src", "modules");
  const entries = await readdir(modulesRoot, { withFileTypes: true });
  const folders = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort();
  const files = [
    path.join(root, "src", "modules", "groups.ts"),
    path.join(root, "scripts", "lib", "search-documents.mjs"),
  ];

  for (const folder of folders) {
    files.push(
      path.join(modulesRoot, folder, "module.ts"),
      path.join(modulesRoot, folder, "card-info.ts"),
      path.join(modulesRoot, folder, "content", "plain.mdx"),
      path.join(modulesRoot, folder, "content", "standard.mdx"),
    );
  }

  return files;
}

export async function fingerprintSources(files, root) {
  const hash = createHash("sha256");
  hash.update(SEARCH_INDEX_SCHEMA);
  // Hash repository-relative paths so the index stays current wherever the repo is cloned.
  for (const file of [...files].sort()) {
    hash.update(path.relative(root, file).split(path.sep).join("/"));
    hash.update("\0");
    hash.update(await readFile(file));
    hash.update("\0");
  }
  return hash.digest("hex");
}

export async function extractSearchDocuments(root) {
  const modulesRoot = path.join(root, "src", "modules");
  const entries = await readdir(modulesRoot, { withFileTypes: true });
  const folders = entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort();
  const groups = parseGroups(await readFile(path.join(root, "src", "modules", "groups.ts"), "utf8"));
  const documents = [];

  documents.push(
    documentRecord({
      id: "screen:home",
      kind: "screen",
      title: "Learning map",
      subtitle: "Home",
      body: "Learn AI. Browse every lab by group. Continue exploring. Start with the groundwork, build the foundations, look inside models, build with a trained model, explore the frontier, train and adapt a model.",
      screen: "home",
      rankBoost: 1.6,
    }),
    documentRecord({
      id: "screen:settings",
      kind: "screen",
      title: "Settings",
      subtitle: "App",
      body: "Appearance, accessibility, explanation mode, local model, cloud providers, data export, and about.",
      screen: "settings",
      settingsTab: "appearance",
      rankBoost: 1.5,
    }),
  );

  for (const [id, title, description] of SETTINGS_TABS) {
    documents.push(
      documentRecord({
        id: `screen:settings:${id}`,
        kind: "screen",
        title,
        subtitle: "Settings",
        body: description,
        screen: "settings",
        settingsTab: id,
        rankBoost: 1.35,
      }),
    );
  }

  for (const group of groups) {
    documents.push(
      documentRecord({
        id: `group:${group.id}`,
        kind: "group",
        title: group.title,
        subtitle: "Learning map",
        body: group.description,
        groupId: group.id,
        screen: "home",
        rankBoost: 1.7,
      }),
    );
  }

  for (const folder of folders) {
    const directory = path.join(modulesRoot, folder);
    const source = await readFile(path.join(directory, "module.ts"), "utf8");
    const id = unquotedField(source, "id");
    const slug = unquotedField(source, "slug") ?? folder;
    const title = unquotedField(source, "title") ?? folder;
    const groupId = unquotedField(source, "group") ?? "";
    const tagline = unquotedField(source, "tagline") ?? "";
    const objectives = quotedStrings(source.match(/\bobjectives:\s*\[([\s\S]*?)\],\s*glossary:/)?.[1] ?? "");
    const steps = quotedStrings(bracketBlock(source, "steps", "stepInstructions:"));
    const instructions = quotedStrings(bracketBlock(source, "stepInstructions", "stateVersion:"));
    const glossary = parseGlossary(source);
    const { questions: checkpoints } = parseModuleCheckpoints(source);
    const moduleMeta = {
      moduleId: id ?? `module-${slug}`,
      moduleTitle: title,
      moduleSlug: slug,
      groupId,
    };

    documents.push(
      documentRecord({
        id: `module:${moduleMeta.moduleId}`,
        kind: "module",
        title,
        subtitle: tagline,
        body: collapseWhitespace([tagline, ...objectives].join(" ")),
        ...moduleMeta,
        rankBoost: 3,
      }),
    );

    steps.forEach((step, index) => {
      documents.push(
        documentRecord({
          id: `step:${slug}:${index}`,
          kind: "step",
          title: step,
          subtitle: `Step ${index + 1} · ${title}`,
          body: instructions[index] ?? "",
          ...moduleMeta,
          stepIndex: index,
          rankBoost: 1.8,
        }),
      );
    });

    glossary.forEach((item) => {
      documents.push(
        documentRecord({
          id: `glossary:${slug}:${slugify(item.term)}`,
          kind: "glossary",
          title: item.term,
          subtitle: `Glossary · ${title}`,
          body: item.definition,
          ...moduleMeta,
          term: item.term,
          rankBoost: 2.2,
        }),
      );
    });

    checkpoints.forEach((checkpoint, index) => {
      const many = checkpoints.length > 1;
      documents.push(
        documentRecord({
          // A single legacy question keeps its original id so existing links stay stable.
          id: many ? `checkpoint:${slug}:${index}` : `checkpoint:${slug}`,
          kind: "checkpoint",
          title: checkpoint.prompt || `Checkpoint · ${title}`,
          subtitle: many ? `Checkpoint · ${title} · question ${index + 1}` : `Checkpoint · ${title}`,
          body: collapseWhitespace([checkpoint.prompt, ...checkpoint.options, checkpoint.explanation].join(" ")),
          ...moduleMeta,
          rankBoost: 1.4,
        }),
      );
    });

    for (const mode of ["plain", "standard"]) {
      const mdx = await readFile(path.join(directory, "content", `${mode}.mdx`), "utf8");
      const sections = splitLearnSections(mdx);
      const known = new Set(sections.map((section) => section.title));
      for (const required of REQUIRED_LEARN_SECTIONS) {
        if (!known.has(required) && !sections.some((section) => slugify(section.title) === slugify(required))) {
          // The module checker owns missing-section errors; search just indexes what exists.
        }
      }
      for (const section of sections) {
        documents.push(
          documentRecord({
            id: `learn:${slug}:${mode}:${slugify(section.title)}`,
            kind: "learn",
            title: section.title,
            subtitle: `${mode === "plain" ? "Plain" : "Standard"} · ${title}`,
            body: section.body,
            ...moduleMeta,
            mode,
            section: section.title,
            rankBoost: 1.15,
          }),
        );
      }
    }

    try {
      const cards = parseCardInfo(await readFile(path.join(directory, "card-info.ts"), "utf8"));
      for (const card of cards) {
        documents.push(
          documentRecord({
            id: `card:${slug}:${slugify(card.label)}`,
            kind: "card",
            title: card.title,
            subtitle: `${card.label} · ${title}`,
            body: card.body,
            ...moduleMeta,
            cardLabel: card.label,
            rankBoost: 1.55,
          }),
        );
      }
    } catch {
      // A missing card-info file is reported by module validation.
    }
  }

  return documents;
}

export async function buildSearchIndex(root) {
  const files = await collectSearchSources(root);
  const [documents, fingerprint] = await Promise.all([
    extractSearchDocuments(root),
    fingerprintSources(files, root),
  ]);
  return {
    version: SEARCH_INDEX_VERSION,
    schema: SEARCH_INDEX_SCHEMA,
    fingerprint,
    generatedAt: new Date().toISOString(),
    documentCount: documents.length,
    documents,
  };
}
