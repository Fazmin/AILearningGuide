import generatedData from "./licenses.generated.json";
import manualData from "./licenses.manual.json";

/**
 * The About > Licences list. Two sources, both checked in:
 *  - `licenses.generated.json`: every runtime npm package, read from node_modules by
 *    `scripts/gen-licenses.mjs` (`npm run gen:licenses`; `npm run check:licenses` fails if stale);
 *  - `licenses.manual.json`: Rust crates, the engine, models, and data, each with the primary source
 *    its licence was read from.
 * Everything here is bundled, so the list and the licence texts work offline.
 */

export type LicenceGroupId = "libraries" | "fonts" | "runtime" | "data";
export type LicenceStatus = "verified" | "declared" | "check-upstream";
export type LicenceDelivery = "bundled" | "downloaded" | "derived";

export interface LicenceEntry {
  id: string;
  name: string;
  version?: string;
  /** "npm package", "Rust crate", "Font", "Engine", "Language model", "Image dataset", ... */
  kind: string;
  group: LicenceGroupId;
  /** Shown inside a collapsed sub-list: packages installed with a library, and the Rust crates. */
  secondary: boolean;
  delivery: LicenceDelivery;
  usedFor: string;
  /** An SPDX expression where one applies, otherwise a plain statement. */
  licence: string;
  /** Ids of the entries in `licenceTexts` that this component's licence refers to. */
  textIds: string[];
  /** Copyright lines as the component's own licence file states them. */
  copyright: string[];
  holder: string;
  url?: string;
  note?: string;
  status: LicenceStatus;
  basis: string;
}

export interface LicenceText {
  id: string;
  title: string;
  text: string;
  reference: string;
}

export interface AdditionalNotice {
  packages: string[];
  text: string;
}

export interface LicenceGroup {
  id: LicenceGroupId;
  title: string;
  blurb: string;
  secondaryTitle?: string;
}

/** The same line tauri.conf.json carries as `bundle.copyright`. The repository declares no licence for the app itself. */
export const APP_COPYRIGHT = "Copyright © 2026 Discover AI contributors";

export const LICENCE_GROUPS: LicenceGroup[] = [
  {
    id: "libraries",
    title: "Libraries",
    blurb: "Code that runs inside the interface, read from the installed packages.",
    secondaryTitle: "Packages installed with these libraries",
  },
  {
    id: "fonts",
    title: "Fonts",
    blurb: "Bundled with the app, so lessons read the same offline.",
  },
  {
    id: "runtime",
    title: "Runtime & models",
    blurb:
      "What runs on your device: the desktop shell, the model runtime, and the models. The local chat engine and language model are downloaded only if you set them up.",
    secondaryTitle: "Rust crates in the desktop shell",
  },
  {
    id: "data",
    title: "Data",
    blurb: "What the teaching models and word vectors were built from.",
  },
];

export const DELIVERY_LABEL: Record<LicenceDelivery, string> = {
  bundled: "Bundled in the app",
  downloaded: "Downloaded on request",
  derived: "Used to build the bundled models",
};

export const STATUS_LABEL: Record<LicenceStatus, string> = {
  verified: "Checked against the upstream licence",
  declared: "Declared by the package, which ships no licence file",
  "check-upstream": "Not verified here: check the upstream project",
};

interface GeneratedDirect extends Omit<LicenceEntry, "secondary"> {
  package: string;
}
interface GeneratedTransitive {
  name: string;
  version: string;
  licence: string;
  textIds: string[];
  copyright: string[];
  holder: string;
  url: string;
  via: string[];
  status: LicenceStatus;
}
interface Generated {
  direct: GeneratedDirect[];
  transitive: GeneratedTransitive[];
  texts: LicenceText[];
  additionalNotices: AdditionalNotice[];
  typeOnlyExcluded: string[];
}
interface ManualEntry extends Omit<LicenceEntry, "secondary" | "copyright"> {
  secondary?: boolean;
  /** Copyright lines, where the component's own licence file states any. */
  copyright?: string[];
  crate?: string;
}

const generated = generatedData as unknown as Generated;
const manual = manualData as unknown as { entries: ManualEntry[] };

const byName = (a: LicenceEntry, b: LicenceEntry) =>
  a.name.toLowerCase() < b.name.toLowerCase() ? -1 : a.name.toLowerCase() > b.name.toLowerCase() ? 1 : 0;

function buildEntries(): LicenceEntry[] {
  const direct: LicenceEntry[] = generated.direct.map(({ package: _package, ...entry }) => ({
    ...entry,
    secondary: false,
  }));
  const transitive: LicenceEntry[] = generated.transitive.map((item) => ({
    id: `npm:${item.name}@${item.version}`,
    name: item.name,
    version: item.version,
    kind: "npm package",
    group: "libraries",
    secondary: true,
    delivery: "bundled",
    usedFor: `Installed with ${item.via.join(", ")}.`,
    licence: item.licence,
    textIds: item.textIds,
    copyright: item.copyright,
    holder: item.holder,
    url: item.url,
    status: item.status,
    basis: "license field and LICENSE file of the installed package",
  }));
  const handwritten: LicenceEntry[] = manual.entries.map(({ crate: _crate, secondary, copyright, ...entry }) => ({
    ...entry,
    copyright: copyright ?? [],
    secondary: Boolean(secondary),
  }));
  return [...direct, ...transitive, ...handwritten].sort(byName);
}

export const licenceEntries: LicenceEntry[] = buildEntries();
export const licenceTexts: LicenceText[] = generated.texts;
export const additionalNotices: AdditionalNotice[] = generated.additionalNotices;
export const typeOnlyExcluded: string[] = generated.typeOnlyExcluded;

/** "licence" and "license" find each other, so a search in either spelling works. */
const fold = (value: string) => value.toLowerCase().replace(/licen[sc]e/g, "licence");

function haystack(entry: LicenceEntry) {
  const group = LICENCE_GROUPS.find((item) => item.id === entry.group);
  return fold(
    [
      entry.name,
      entry.version,
      entry.kind,
      group?.title,
      DELIVERY_LABEL[entry.delivery],
      entry.usedFor,
      `licence ${entry.licence}`,
      entry.holder,
      ...entry.copyright,
      entry.url,
      entry.note,
      entry.status === "check-upstream" ? "check upstream unverified" : "",
    ]
      .filter(Boolean)
      .join(" \n "),
  );
}

/** Every whitespace-separated term must appear somewhere in the entry. An empty query keeps everything. */
export function filterLicenceEntries(entries: LicenceEntry[], query: string): LicenceEntry[] {
  const terms = fold(query).split(/\s+/).filter(Boolean);
  if (!terms.length) return entries;
  return entries.filter((entry) => {
    const text = haystack(entry);
    return terms.every((term) => text.includes(term));
  });
}

export interface LicenceTextSection extends LicenceText {
  /** Each distinct copyright line under this licence, with the components that carry it. */
  notices: Array<{ line: string; names: string[] }>;
  componentCount: number;
}

/** One section per licence type: the copyright lines of every component under it, then the text once. */
export function licenceTextSections(entries: LicenceEntry[] = licenceEntries): LicenceTextSection[] {
  return licenceTexts.map((text) => {
    const users = entries.filter((entry) => entry.textIds.includes(text.id));
    const lines = new Map<string, Set<string>>();
    for (const entry of users) {
      const label = entry.version && entry.kind === "npm package" ? `${entry.name} ${entry.version}` : entry.name;
      for (const line of entry.copyright) {
        const names = lines.get(line) ?? new Set<string>();
        names.add(label);
        lines.set(line, names);
      }
    }
    return {
      ...text,
      componentCount: users.length,
      notices: [...lines.entries()]
        .map(([line, names]) => ({ line, names: [...names].sort() }))
        .sort((a, b) => (a.line.toLowerCase() < b.line.toLowerCase() ? -1 : 1)),
    };
  });
}
