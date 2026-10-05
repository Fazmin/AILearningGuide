import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  APP_COPYRIGHT,
  LICENCE_GROUPS,
  additionalNotices,
  filterLicenceEntries,
  licenceEntries,
  licenceTextSections,
  licenceTexts,
} from "./licenses";

const root = path.resolve(__dirname, "..", "..");
const readText = (...parts: string[]) => readFileSync(path.join(root, ...parts), "utf8");
const packageJson = JSON.parse(readText("package.json")) as { dependencies: Record<string, string> };
const entry = (id: string) => licenceEntries.find((item) => item.id === id);

/** Direct crates from src-tauri/Cargo.toml: every [dependencies] table, build-dependencies excluded. */
function directCrates() {
  const crates = new Set<string>();
  let inDependencies = false;
  for (const line of readText("src-tauri", "Cargo.toml").split(/\r?\n/)) {
    const section = line.match(/^\[(.+)\]\s*$/);
    if (section) {
      inDependencies = /(^|\.)dependencies$/.test(section[1]) && !/build-dependencies$/.test(section[1]);
      continue;
    }
    const crate = inDependencies && line.match(/^([A-Za-z0-9_-]+)\s*=/);
    if (crate) crates.add(crate[1]);
  }
  return [...crates];
}

describe("coverage of what the app ships", () => {
  it("lists every runtime dependency in package.json", () => {
    const names = Object.keys(packageJson.dependencies);
    expect(names.length).toBeGreaterThan(0);
    for (const name of names) {
      expect(entry(`npm:${name}`), `${name} is a runtime dependency but is missing from the licence list`).toBeDefined();
    }
  });

  it("lists every direct Rust crate in src-tauri/Cargo.toml", () => {
    const crates = directCrates();
    expect(crates).toContain("tauri");
    for (const crate of crates) {
      expect(entry(`crate:${crate}`), `${crate} is a direct crate but is missing from the licence list`).toBeDefined();
    }
  });

  it("does not list development-only tooling as something the app ships", () => {
    const names = licenceEntries.map((item) => item.name.toLowerCase());
    for (const dev of ["vite", "vitest", "typescript", "jsdom", "@tauri-apps/cli"]) {
      expect(names, dev).not.toContain(dev);
    }
  });

  it("covers the three bundled font families, each under the SIL Open Font License", () => {
    const fonts = licenceEntries.filter((item) => item.group === "fonts");
    expect(fonts.map((item) => item.name).sort()).toEqual(["Inter", "JetBrains Mono", "Newsreader"]);
    for (const font of fonts) {
      expect(font.licence).toBe("OFL-1.1");
      expect(font.copyright.join(" ")).toMatch(/Project Authors/);
    }
    const main = readText("src", "main.tsx");
    for (const name of ["inter", "jetbrains-mono", "newsreader"]) expect(main).toContain(`@fontsource-variable/${name}`);
  });

  it("names the pinned engine and model exactly as the setup code downloads them", () => {
    const rust = readText("src-tauri", "src", "local_ai.rs");
    const engine = rust.match(/const ENGINE_VERSION: &str = "([^"]+)"/)![1];
    const revision = rust.match(/Qwen3\.5-2B-GGUF\/resolve\/([0-9a-f]{40})\//)![1];
    expect(entry("llama-cpp")).toMatchObject({ version: engine, licence: "MIT", delivery: "downloaded" });
    expect(entry("qwen3-5-2b-gguf")).toMatchObject({ licence: "Apache-2.0", delivery: "downloaded" });
    expect(entry("qwen3-5-2b-gguf")!.version).toBe(`revision ${revision.slice(0, 7)}`);
  });

  it("covers the data the shipped teaching models were built from", () => {
    const data = licenceEntries.filter((item) => item.group === "data").map((item) => item.name);
    expect(data).toEqual(expect.arrayContaining(["GloVe word vectors, 50 dimensions", "MNIST handwritten digits", "Tiny Shakespeare"]));
    expect(entry("teaching-models")).toBeDefined();
    expect(entry("npm:onnxruntime-web")).toBeDefined();
  });
});

describe("honesty about what could not be verified", () => {
  it("sends the reader to the upstream project for MNIST instead of asserting a licence", () => {
    const mnist = entry("data:mnist")!;
    expect(mnist.status).toBe("check-upstream");
    expect(mnist.licence).toBe("Check the upstream project");
    expect(mnist.url).toBe("http://yann.lecun.com/exdb/mnist/");
    expect(mnist.textIds).toEqual([]);
    expect(mnist.note).toMatch(/could not verify/i);
  });

  it("never leaves an entry without a licence statement, holder, or way to check it", () => {
    for (const item of licenceEntries) {
      expect(item.licence.trim(), item.id).not.toBe("");
      expect(item.licence, item.id).not.toMatch(/^(UNLICENSED|UNKNOWN|NONE)$/i);
      expect(item.holder.trim(), item.id).not.toBe("");
      expect(item.basis.trim(), item.id).not.toBe("");
      expect(["verified", "declared", "check-upstream"], item.id).toContain(item.status);
      expect(["bundled", "downloaded", "derived"], item.id).toContain(item.delivery);
      expect(LICENCE_GROUPS.map((group) => group.id), item.id).toContain(item.group);
      if (item.status === "check-upstream") expect(item.url, item.id).toBeTruthy();
    }
  });

  it("does not claim a licence for the app or the models trained for it", () => {
    expect(entry("teaching-models")!.licence).toBe("No separate licence declared");
    expect(entry("teaching-models")!.holder).toBe("Discover AI contributors");
  });

  it("shows the same copyright line as tauri.conf.json", () => {
    const conf = JSON.parse(readText("src-tauri", "tauri.conf.json")) as { bundle: { copyright: string } };
    expect(APP_COPYRIGHT).toBe(conf.bundle.copyright);
  });
});

describe("licence texts", () => {
  it("bundles one text for every licence type an entry points at", () => {
    const known = new Set(licenceTexts.map((text) => text.id));
    for (const item of licenceEntries) {
      for (const id of item.textIds) expect(known.has(id), `${item.id} -> ${id}`).toBe(true);
    }
    for (const text of licenceTexts) expect(licenceEntries.some((item) => item.textIds.includes(text.id)), text.id).toBe(true);
  });

  it("has the real text of each licence, with no unfilled stand-ins from the generator", () => {
    const byId = Object.fromEntries(licenceTexts.map((text) => [text.id, text.text]));
    expect(byId["MIT"]).toContain("Permission is hereby granted, free of charge");
    expect(byId["ISC"]).toContain("Permission to use, copy, modify, and/or distribute");
    expect(byId["BSD-3-Clause"]).toContain("Redistribution and use in source and binary forms");
    expect(byId["Apache-2.0"]).toContain("END OF TERMS AND CONDITIONS");
    expect(byId["OFL-1.1"]).toContain("SIL OPEN FONT LICENSE Version 1.1");
    for (const text of licenceTexts) expect(text.text.length, text.id).toBeGreaterThan(500);
  });

  it("lists copyright lines under each licence, as the components' own licence files state them", () => {
    const sections = licenceTextSections();
    const mit = sections.find((section) => section.id === "MIT")!;
    const lines = mit.notices.map((notice) => notice.line).join("\n");
    expect(lines).toContain("Meta Platforms, Inc. and affiliates");
    const meta = mit.notices.find((notice) => notice.line.includes("Meta Platforms"))!.names;
    expect(meta.some((name) => name.startsWith("React "))).toBe(true);
    expect(meta.some((name) => name.startsWith("React DOM "))).toBe(true);
    expect(lines).toContain("Microsoft Corporation");
    expect(lines).toContain("Tauri Apps Contributors");
    const ofl = sections.find((section) => section.id === "OFL-1.1")!;
    expect(ofl.notices.map((notice) => notice.line).join("\n")).toContain("The Inter Project Authors");
    expect(ofl.componentCount).toBe(3);
  });

  it("keeps the packages whose licence text differs from the standard one", () => {
    const names = additionalNotices.flatMap((notice) => notice.packages);
    expect(names).toContain("lucide-react");
    for (const notice of additionalNotices) expect(notice.text.length).toBeGreaterThan(100);
  });
});

describe("filterLicenceEntries", () => {
  it("keeps everything for an empty query", () => {
    expect(filterLicenceEntries(licenceEntries, "  ")).toHaveLength(licenceEntries.length);
  });

  it("matches name, licence, holder, and purpose, case-insensitively", () => {
    expect(filterLicenceEntries(licenceEntries, "ZUSTAND").map((item) => item.name)).toContain("Zustand");
    expect(filterLicenceEntries(licenceEntries, "ofl-1.1")).toHaveLength(3);
    expect(filterLicenceEntries(licenceEntries, "microsoft").map((item) => item.name)).toContain("ONNX Runtime Web");
    expect(filterLicenceEntries(licenceEntries, "topic chat").map((item) => item.id)).toEqual(
      expect.arrayContaining(["llama-cpp", "qwen3-5-2b-gguf"]),
    );
  });

  it("treats the two spellings of licence alike", () => {
    expect(filterLicenceEntries(licenceEntries, "license apache").length).toBe(
      filterLicenceEntries(licenceEntries, "licence apache").length,
    );
    expect(filterLicenceEntries(licenceEntries, "licence apache").length).toBeGreaterThan(0);
  });

  it("requires every word to match, and finds nothing for nonsense", () => {
    const both = filterLicenceEntries(licenceEntries, "font inter");
    expect(both.map((item) => item.name)).toEqual(["Inter"]);
    expect(filterLicenceEntries(licenceEntries, "zzzz-no-such-thing")).toEqual([]);
  });

  it("surfaces the entries that need checking upstream", () => {
    const flagged = filterLicenceEntries(licenceEntries, "check upstream").map((item) => item.id);
    expect(flagged).toEqual(expect.arrayContaining(["data:mnist", "data:tiny-shakespeare"]));
  });
});
