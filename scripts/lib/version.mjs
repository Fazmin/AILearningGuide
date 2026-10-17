/**
 * The app version is recorded in five places, and they must all agree:
 *
 *   package.json, package-lock.json   what npm, the web build, and Settings > About show
 *   src-tauri/tauri.conf.json         what the desktop installers are named and stamped with
 *   src-tauri/Cargo.toml, Cargo.lock  what the Rust crate reports
 *
 * package.json is the one to read. setVersion() rewrites all of them and leaves the rest of each file
 * untouched.
 */
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const replaceVersion = (text, pattern, version, file) => {
  if (!pattern.test(text)) throw new Error(`Could not find the version in ${file}.`);
  return text.replace(pattern, `$1${version}$2`);
};

const FILES = [
  {
    file: "package.json",
    read: (text) => JSON.parse(text).version,
    write: (text, version) => replaceVersion(text, /^(  "version": ")[^"]*(")/m, version, "package.json"),
  },
  {
    // npm writes this file as JSON.stringify(lock, null, 2), so rewriting it whole changes nothing else.
    file: "package-lock.json",
    read: (text) => {
      const lock = JSON.parse(text);
      return [...new Set([lock.version, lock.packages[""].version])].join(" and ");
    },
    write: (text, version) => {
      const lock = JSON.parse(text);
      lock.version = version;
      lock.packages[""].version = version;
      return `${JSON.stringify(lock, null, 2)}\n`;
    },
  },
  {
    file: "src-tauri/tauri.conf.json",
    read: (text) => JSON.parse(text).version,
    write: (text, version) => replaceVersion(text, /^(  "version": ")[^"]*(")/m, version, "src-tauri/tauri.conf.json"),
  },
  {
    // The first `version = ` at the start of a line is the one under [package].
    file: "src-tauri/Cargo.toml",
    read: (text) => text.match(/^version = "([^"]*)"/m)?.[1],
    write: (text, version) => replaceVersion(text, /^(version = ")[^"]*(")/m, version, "src-tauri/Cargo.toml"),
  },
  {
    file: "src-tauri/Cargo.lock",
    read: (text) => text.match(/^name = "the-ai-guide"\r?\nversion = "([^"]*)"/m)?.[1],
    write: (text, version) =>
      replaceVersion(text, /^(name = "the-ai-guide"\r?\nversion = ")[^"]*(")/m, version, "src-tauri/Cargo.lock"),
  },
];

export const VERSION_FILES = FILES.map(({ file }) => file);

export const isVersion = (value) => /^\d+\.\d+\.\d+$/.test(value);

/** The version in package.json. */
export const currentVersion = (root) => JSON.parse(readFileSync(path.join(root, "package.json"), "utf8")).version;

/** Every file's version, so a check can name the ones that disagree. */
export const readVersions = (root) =>
  FILES.map(({ file, read }) => ({ file, version: read(readFileSync(path.join(root, file), "utf8")) ?? "(missing)" }));

export const setVersion = (root, version) => {
  if (!isVersion(version)) throw new Error(`"${version}" isn't a version. Use three numbers, like 0.2.0.`);
  for (const { file, write } of FILES) {
    const filePath = path.join(root, file);
    writeFileSync(filePath, write(readFileSync(filePath, "utf8"), version));
  }
};

/** 0.1.70 → 0.1.71 for "patch", 0.2.0 for "minor", 1.0.0 for "major". */
export const bumpVersion = (version, part = "patch") => {
  const [major, minor, patch] = version.split(".").map(Number);
  if (part === "major") return `${major + 1}.0.0`;
  if (part === "minor") return `${major}.${minor + 1}.0`;
  return `${major}.${minor}.${patch + 1}`;
};
