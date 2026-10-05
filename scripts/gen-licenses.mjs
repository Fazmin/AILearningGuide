/**
 * Builds src/lib/licenses.generated.json, the npm half of the About > Licences list, from what is
 * actually installed: package.json `dependencies`, then every installed package's own package.json
 * `license` and LICENSE files, following `dependencies` down the tree. Nothing is written from
 * memory. The output is deterministic (sorted, no timestamps) so it can be checked in CI.
 *
 *   node scripts/gen-licenses.mjs           write the file
 *   node scripts/gen-licenses.mjs --check   fail if the file is stale or the list is incomplete
 *
 * The hand-written half (Rust crates, the engine, models, data) lives in src/lib/licenses.manual.json.
 * The check also fails if a direct Rust crate in src-tauri/Cargo.toml has no entry there.
 */
import { existsSync } from "node:fs";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputPath = path.join(root, "src", "lib", "licenses.generated.json");
const manualPath = path.join(root, "src", "lib", "licenses.manual.json");
const checkOnly = process.argv.includes("--check");

const GROUPS = ["libraries", "fonts", "runtime", "data"];
const STATUSES = ["verified", "declared", "check-upstream"];
const DELIVERIES = ["bundled", "downloaded", "derived"];

/** Packages that only carry TypeScript declarations; nothing from them reaches the bundle. */
const isTypeOnly = (name) => name.startsWith("@types/") || name === "undici-types" || name === "csstype";

/**
 * One canonical text per licence type. `from` marks where the licence proper begins in a LICENSE
 * file (everything before it is the title and copyright line, which differ per package and are listed
 * separately). A package whose licence body differs from the canonical one keeps its own full text
 * under "additional notices" instead of being silently folded in.
 */
const CANONICAL = {
  MIT: {
    title: "MIT License",
    header: "MIT License\n\nCopyright (c) <year> <copyright holder>\n\n",
    from: /Permission is hereby granted/,
    reference: { pkg: "react", file: "LICENSE" },
  },
  ISC: {
    title: "ISC License",
    header: "ISC License\n\nCopyright (c) <year> <copyright holder>\n\n",
    from: /Permission to use, copy, modify/,
    reference: { pkg: "@ungap/structured-clone", file: "LICENSE" },
  },
  "BSD-3-Clause": {
    title: "BSD 3-Clause License",
    header: "BSD 3-Clause License\n\nCopyright (c) <year>, <copyright holder>\nAll rights reserved.\n\n",
    from: /Redistribution and use in source and binary forms/,
    reference: { pkg: "@protobufjs/aspromise", file: "LICENSE" },
  },
  "Apache-2.0": {
    title: "Apache License, Version 2.0",
    header: "",
    from: /TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION/,
    until: /END OF TERMS AND CONDITIONS/,
    whole: true,
    reference: { pkg: "@tauri-apps/api", file: "LICENSE_APACHE-2.0" },
  },
  "OFL-1.1": {
    title: "SIL Open Font License, Version 1.1",
    header: "",
    from: /^-{20,}\s*$/m,
    reference: { pkg: "@fontsource-variable/inter", file: "LICENSE" },
  },
};

const fail = (message) => {
  console.error(`check:licenses: ${message}`);
  process.exit(1);
};

const byCodepoint = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
const squash = (value) => String(value).replace(/\s+/g, " ").trim();
const normalised = (value) => String(value).toLowerCase().replace(/[^a-z0-9]+/g, "");

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

/** Node-style lookup: <from>/node_modules/<name>, then each parent directory's. */
function resolvePackageDir(name, from) {
  let dir = from;
  for (;;) {
    const candidate = path.join(dir, "node_modules", name);
    if (existsSync(path.join(candidate, "package.json"))) return candidate;
    const parent = path.dirname(dir);
    if (parent === dir || !parent.startsWith(root)) return null;
    dir = parent;
  }
}

function licenceExpression(pkg) {
  if (typeof pkg.license === "string" && pkg.license.trim()) return pkg.license.trim();
  if (pkg.license && typeof pkg.license.type === "string") return pkg.license.type;
  if (Array.isArray(pkg.licenses) && pkg.licenses.length) {
    return pkg.licenses.map((item) => (typeof item === "string" ? item : item.type)).join(" OR ");
  }
  return "";
}

const licenceIds = (expression) =>
  expression
    .split(/\s+(?:OR|AND|WITH)\s+|[()]|\//)
    .map((part) => part.trim())
    .filter(Boolean);

function repositoryUrl(pkg) {
  const raw = typeof pkg.repository === "string" ? pkg.repository : pkg.repository?.url;
  if (!raw) return "";
  let url = raw.trim();
  if (/^[\w.-]+\/[\w.-]+$/.test(url)) return `https://github.com/${url}`;
  if (url.startsWith("github:")) return `https://github.com/${url.slice(7)}`;
  url = url
    .replace(/^git\+/, "")
    .replace(/^git:\/\//, "https://")
    .replace(/^ssh:\/\/git@/, "https://")
    .replace(/^git@github\.com:/, "https://github.com/")
    .replace(/\.git$/, "");
  return url;
}

function authorName(pkg) {
  const author = typeof pkg.author === "string" ? pkg.author : pkg.author?.name;
  if (author) return author.replace(/<[^>]*>/g, "").replace(/\([^)]*\)/g, "").trim();
  const first = Array.isArray(pkg.contributors) ? pkg.contributors[0] : undefined;
  const contributor = typeof first === "string" ? first : first?.name;
  return contributor ? contributor.replace(/<[^>]*>/g, "").replace(/\([^)]*\)/g, "").trim() : "";
}

const COPYRIGHT_LINE = /^\s*(?:Copyright\s*(?:\(c\)|©)|Copyright(?=\s+(?:\d|\[|[A-Z]))|©)/;

/** Real copyright lines only: not the Apache appendix placeholder, not prose that mentions copyright. */
function extractCopyrights(text, { stopAt } = {}) {
  const limit = stopAt ? text.search(stopAt) : -1;
  const scanned = limit > 0 ? text.slice(0, limit) : text;
  const found = [];
  for (const line of scanned.split(/\r?\n/)) {
    if (!COPYRIGHT_LINE.test(line)) continue;
    if (/yyyy|name of copyright owner|<year>|<dates>|<copyright holder>/i.test(line)) continue;
    // Fontsource repeats each notice once per file: "Copyright 2016 X (url) Font[wght].ttf: Copyright 2016 X (url)".
    const pieces = squash(line)
      .split(/\s+\S+\.(?:ttf|otf|woff2?)\s*:\s*/i)
      .map((piece) => piece.replace(/\s+\S+\.(?:ttf|otf|woff2?)$/i, "").trim());
    for (const piece of pieces) if (piece) found.push(piece);
  }
  return [...new Set(found)];
}

function holderFrom(lines, fallback) {
  const holders = lines
    .map((line) =>
      line
        .replace(/^Copyright\s*(?:\(c\)|©)?\s*/i, "")
        .replace(/^\[?\d{4}(?:\s*[-–]\s*(?:\d{4}|[Pp]resent))?\]?,?\s*/, "")
        .replace(/\s*All rights reserved\.?$/i, "")
        .replace(/\s*<[^>]*@[^>]*>/g, "")
        .replace(/[.,;\s]+$/, "")
        .trim(),
    )
    .filter(Boolean);
  const unique = [...new Set(holders)];
  return unique.length ? unique.join("; ") : fallback;
}

function coreOf(rule, text) {
  const start = text.search(rule.from);
  if (start < 0) return null;
  let end = text.length;
  if (rule.until) {
    const match = rule.until.exec(text);
    if (!match) return null;
    end = match.index + match[0].length;
  }
  return text.slice(start, end);
}

async function licenceFiles(dir) {
  const names = (await readdir(dir)).filter((name) => /^(licen[sc]e|copying|notice)/i.test(name)).sort(byCodepoint);
  const files = [];
  for (const name of names) {
    const full = path.join(dir, name);
    const text = (await readFile(full, "utf8")).replace(/\r\n/g, "\n");
    files.push({ name, text, notice: /^notice/i.test(name) });
  }
  return files;
}

async function main() {
  if (!existsSync(path.join(root, "node_modules"))) {
    fail("node_modules is missing. Run `npm ci` first; the list is generated from the installed packages.");
  }
  const manual = await readJson(manualPath);
  const rootPackage = await readJson(path.join(root, "package.json"));
  const directNames = Object.keys(rootPackage.dependencies ?? {}).sort(byCodepoint);

  // --- Walk the runtime dependency tree from each direct dependency. -----------------------------
  const closure = new Map(); // dir -> { dir, pkg, via: Set<directName> }
  for (const direct of directNames) {
    const rootDir = resolvePackageDir(direct, root);
    if (!rootDir) fail(`direct dependency "${direct}" is not installed. Run \`npm ci\`.`);
    const queue = [rootDir];
    const seen = new Set();
    while (queue.length) {
      const dir = queue.shift();
      if (seen.has(dir)) continue;
      seen.add(dir);
      const pkg = await readJson(path.join(dir, "package.json"));
      const entry = closure.get(dir) ?? { dir, pkg, via: new Set() };
      entry.via.add(direct);
      closure.set(dir, entry);
      const optional = new Set(Object.keys(pkg.optionalDependencies ?? {}));
      for (const name of Object.keys({ ...(pkg.dependencies ?? {}), ...(pkg.optionalDependencies ?? {}) })) {
        const child = resolvePackageDir(name, dir);
        if (child) queue.push(child);
        else if (!optional.has(name)) fail(`"${pkg.name}" depends on "${name}", which is not installed. Run \`npm ci\`.`);
      }
    }
  }

  // --- Read each package's licence files once. ---------------------------------------------------
  const packages = [];
  for (const { dir, pkg, via } of closure.values()) {
    const typeOnly = isTypeOnly(pkg.name);
    const expression = licenceExpression(pkg);
    if (!typeOnly && !expression) fail(`${pkg.name}@${pkg.version} declares no license in its package.json.`);
    packages.push({ dir, pkg, via: [...via].sort(byCodepoint), typeOnly, expression, files: typeOnly ? [] : await licenceFiles(dir) });
  }
  packages.sort((a, b) => byCodepoint(a.pkg.name, b.pkg.name) || byCodepoint(a.pkg.version, b.pkg.version));
  const byName = (name) => packages.find((item) => item.pkg.name === name);

  // --- Canonical texts, one per licence type. -----------------------------------------------------
  const canonical = {};
  for (const [id, rule] of Object.entries(CANONICAL)) {
    const holder = byName(rule.reference.pkg);
    const file = holder?.files.find((item) => item.name === rule.reference.file);
    if (!file) {
      fail(`reference licence file ${rule.reference.pkg}/${rule.reference.file} for ${id} is not installed; choose another reference in scripts/gen-licenses.mjs.`);
    }
    const core = coreOf(rule, file.text);
    if (!core) fail(`could not find the licence body of ${id} in ${rule.reference.pkg}/${rule.reference.file}.`);
    canonical[id] = {
      id,
      title: rule.title,
      text: rule.whole ? file.text.trim() : `${rule.header}${core.trim()}`,
      core: normalised(core),
      reference: `${rule.reference.pkg}/${rule.reference.file}`,
    };
  }

  // --- Classify every licence file: standard (copyright lines only) or an additional notice. -------
  const notices = new Map(); // normalised text -> { text, packages:Set }
  const records = new Map(); // dir -> { copyright:[], textIds:Set, hasFile, hasStandard }
  for (const item of packages) {
    if (item.typeOnly) continue;
    const declared = licenceIds(item.expression);
    const record = { copyright: [], textIds: new Set(), hasFile: item.files.length > 0 };
    for (const file of item.files) {
      const owner = declared.find((id) => CANONICAL[id] && coreOf(CANONICAL[id], file.text) !== null);
      const standard =
        !file.notice &&
        owner &&
        normalised(coreOf(CANONICAL[owner], file.text)) === canonical[owner].core;
      if (standard) {
        record.textIds.add(owner);
        const stopAt = owner === "OFL-1.1" ? CANONICAL[owner].from : undefined;
        record.copyright.push(...extractCopyrights(file.text, { stopAt }));
        continue;
      }
      // Non-standard: keep this package's own words, and still pull its copyright lines.
      record.copyright.push(...extractCopyrights(file.text));
      const key = normalised(file.text);
      const existing = notices.get(key) ?? { text: file.text.trim(), packages: new Set() };
      existing.packages.add(item.pkg.name);
      notices.set(key, existing);
    }
    for (const id of declared) if (CANONICAL[id]) record.textIds.add(id);
    record.copyright = [...new Set(record.copyright)];
    records.set(item.dir, record);
  }

  // --- Entries. -----------------------------------------------------------------------------------
  const unknownIds = new Set();
  const describe = (item) => {
    const record = records.get(item.dir);
    const declared = licenceIds(item.expression);
    for (const id of declared) if (!CANONICAL[id]) unknownIds.add(`${item.pkg.name}: ${id}`);
    return { record, declared };
  };

  const direct = [];
  for (const name of directNames) {
    const item = byName(name);
    const curated = manual.npm?.[name];
    if (!curated) {
      fail(`"${name}" is a runtime dependency in package.json but has no entry in src/lib/licenses.manual.json "npm". Add what it is used for, then run \`npm run gen:licenses\`.`);
    }
    if (!GROUPS.includes(curated.group)) fail(`licenses.manual.json: "${name}" has an unknown group "${curated.group}".`);
    if (!curated.displayName || !curated.usedFor) fail(`licenses.manual.json: "${name}" needs displayName and usedFor.`);
    const { record } = describe(item);
    const copyright = curated.copyright ?? record.copyright;
    const verified = record.hasFile || Boolean(curated.copyright);
    direct.push({
      id: `npm:${name}`,
      package: name,
      name: curated.displayName,
      version: item.pkg.version,
      kind: curated.group === "fonts" ? "Font" : "npm package",
      group: curated.group,
      delivery: "bundled",
      usedFor: curated.usedFor,
      licence: item.expression,
      textIds: [...record.textIds].sort(byCodepoint),
      copyright,
      holder: curated.holder ?? holderFrom(copyright, authorName(item.pkg) || "See the package"),
      url: curated.url ?? (item.pkg.homepage || repositoryUrl(item.pkg) || `https://www.npmjs.com/package/${name}`),
      status: verified ? "verified" : "declared",
      basis: curated.holderBasis ?? (record.hasFile ? "license field and LICENSE file of the installed package" : "license field of the installed package; it ships no licence file"),
    });
  }

  const transitive = [];
  for (const item of packages) {
    if (item.typeOnly || directNames.includes(item.pkg.name)) continue;
    const override = manual.npmTransitiveOverrides?.[item.pkg.name];
    const { record } = describe(item);
    const copyright = override?.copyright ?? record.copyright;
    transitive.push({
      name: item.pkg.name,
      version: item.pkg.version,
      licence: item.expression,
      textIds: [...record.textIds].sort(byCodepoint),
      copyright,
      holder: holderFrom(copyright, authorName(item.pkg) || "See the package"),
      url: item.pkg.homepage || repositoryUrl(item.pkg) || `https://www.npmjs.com/package/${item.pkg.name}`,
      via: item.via,
      status: record.hasFile || override?.copyright ? "verified" : "declared",
    });
  }

  if (unknownIds.size) {
    fail(
      `licence types without a bundled text: ${[...unknownIds].join(", ")}. Add the type to CANONICAL in scripts/gen-licenses.mjs or review the dependency.`,
    );
  }

  // --- Validate the hand-written half. ------------------------------------------------------------
  const crateEntries = new Map();
  for (const entry of manual.entries ?? []) {
    for (const field of ["id", "name", "kind", "group", "delivery", "usedFor", "licence", "holder", "status", "basis"]) {
      if (!entry[field]) fail(`licenses.manual.json: entry "${entry.id ?? entry.name}" is missing "${field}".`);
    }
    if (!GROUPS.includes(entry.group)) fail(`licenses.manual.json: "${entry.id}" has an unknown group "${entry.group}".`);
    if (!DELIVERIES.includes(entry.delivery)) fail(`licenses.manual.json: "${entry.id}" has an unknown delivery "${entry.delivery}".`);
    if (!STATUSES.includes(entry.status)) fail(`licenses.manual.json: "${entry.id}" has an unknown status "${entry.status}".`);
    if (!Array.isArray(entry.textIds)) fail(`licenses.manual.json: "${entry.id}" needs a textIds array (may be empty).`);
    for (const id of entry.textIds) {
      if (!canonical[id]) fail(`licenses.manual.json: "${entry.id}" names licence text "${id}", which is not available.`);
    }
    if (entry.status === "check-upstream" && !entry.url) fail(`licenses.manual.json: "${entry.id}" must give the upstream URL.`);
    if (entry.crate) crateEntries.set(entry.crate, entry);
  }
  const cargo = await readFile(path.join(root, "src-tauri", "Cargo.toml"), "utf8");
  const crates = new Set();
  let inDependencies = false;
  for (const line of cargo.split(/\r?\n/)) {
    const section = line.match(/^\[(.+)\]\s*$/);
    if (section) {
      inDependencies = /(^|\.)dependencies$/.test(section[1]) && !/build-dependencies$/.test(section[1]);
      continue;
    }
    const crate = inDependencies && line.match(/^([A-Za-z0-9_-]+)\s*=/);
    if (crate) crates.add(crate[1]);
  }
  for (const crate of [...crates].sort(byCodepoint)) {
    if (!crateEntries.has(crate)) {
      fail(`Rust crate "${crate}" is a direct dependency in src-tauri/Cargo.toml but has no entry in src/lib/licenses.manual.json.`);
    }
  }
  for (const crate of crateEntries.keys()) {
    if (!crates.has(crate)) fail(`licenses.manual.json lists crate "${crate}", which is not a direct dependency in src-tauri/Cargo.toml.`);
  }

  // --- Compose and compare. -----------------------------------------------------------------------
  const usedTextIds = new Set([
    ...direct.flatMap((item) => item.textIds),
    ...transitive.flatMap((item) => item.textIds),
    ...(manual.entries ?? []).flatMap((item) => item.textIds),
  ]);
  const texts = [...usedTextIds]
    .sort(byCodepoint)
    .map((id) => ({ id, title: canonical[id].title, text: canonical[id].text, reference: canonical[id].reference }));
  const additionalNotices = [...notices.values()]
    .map((item) => ({ packages: [...item.packages].sort(byCodepoint), text: item.text }))
    .sort((a, b) => byCodepoint(a.packages[0], b.packages[0]));

  const output = {
    generator: "scripts/gen-licenses.mjs",
    source: "package.json dependencies, followed through each installed package's own package.json and LICENSE files",
    typeOnlyExcluded: [...new Set(packages.filter((item) => item.typeOnly).map((item) => item.pkg.name))].sort(byCodepoint),
    direct,
    transitive,
    texts,
    additionalNotices,
  };
  const serialized = `${JSON.stringify(output, null, 2)}\n`;

  if (checkOnly) {
    let existing = "";
    try {
      existing = await readFile(outputPath, "utf8");
    } catch {
      fail("src/lib/licenses.generated.json is missing. Run `npm run gen:licenses`.");
    }
    if (existing !== serialized) {
      fail("src/lib/licenses.generated.json is stale. Run `npm run gen:licenses` after changing dependencies or licences.");
    }
    console.log(`Licence list is current (${direct.length} direct, ${transitive.length} transitive packages, ${crates.size} Rust crates).`);
    return;
  }

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, serialized);
  console.log(
    `Wrote ${direct.length} direct and ${transitive.length} transitive packages, ${texts.length} licence texts, ${additionalNotices.length} additional notices to ${path.relative(root, outputPath)}.`,
  );
}

await main();
