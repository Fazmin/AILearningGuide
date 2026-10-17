/**
 * Publishes a new version of the desktop app: `npm run release`.
 *
 *   1. Picks the next version by adding one to the last number (0.1.70 → 0.1.71) and writes it to
 *      every file that records it. `npm run release -- minor` gives 0.2.0, and `-- major` gives 1.0.0.
 *      If the current version has never been released (it has no tag yet), it's released as it is,
 *      so a version you set by hand with `node scripts/version.mjs --set` is kept.
 *   2. Commits that as "Release v0.1.71" and tags the commit v0.1.71.
 *   3. Pushes main and the tag together. The tag starts .github/workflows/release.yml, which builds
 *      the macOS, Windows, and Linux installers and publishes them as a GitHub release.
 *
 * Pass --no-push to stop after step 2 and look before pushing.
 */
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { VERSION_FILES, bumpVersion, currentVersion, readVersions, setVersion } from "./lib/version.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const git = (...args) => execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
const fail = (message) => {
  console.error(`release: ${message}`);
  process.exit(1);
};

const args = process.argv.slice(2);
const parts = args.filter((arg) => ["patch", "minor", "major"].includes(arg));
const push = !args.includes("--no-push");
if (parts.length > 1 || args.some((arg) => !parts.includes(arg) && arg !== "--no-push")) {
  fail("usage: npm run release [-- patch | minor | major] [--no-push]");
}

if (git("status", "--porcelain")) fail("you have uncommitted changes. Commit or stash them first.");
const branch = git("rev-parse", "--abbrev-ref", "HEAD");
if (branch !== "main") fail(`releases are made from main, and you're on ${branch}.`);

// Fetch first, so the check below sees releases tagged from another machine.
git("fetch", "--quiet", "--tags", "origin", "main");
const [behind] = git("rev-list", "--left-right", "--count", "origin/main...HEAD").split(/\s+/).map(Number);
if (behind > 0) fail("main is behind origin/main. Pull first.");

const current = currentVersion(root);
const mismatched = readVersions(root).filter(({ version }) => version !== current);
if (mismatched.length > 0) {
  fail(`${mismatched.map(({ file }) => file).join(", ")} disagree with package.json. Run \`npm run check:version\`.`);
}

const isReleased = (version) => git("tag", "--list", `v${version}`) !== "";
const version = parts.length > 0 || isReleased(current) ? bumpVersion(current, parts[0]) : current;
const tag = `v${version}`;
if (isReleased(version)) fail(`${tag} has already been released.`);

if (version !== current) {
  setVersion(root, version);
  git("add", ...VERSION_FILES);
  git("commit", "--quiet", "-m", `Release ${tag}`);
}
git("tag", "-a", tag, "-m", `Discover AI ${tag}`);
console.log(version === current ? `Tagged ${tag}.` : `Bumped ${current} → ${version} and tagged ${tag}.`);

if (!push) {
  console.log(`Not pushed. When you're ready: git push --atomic origin main ${tag}`);
  process.exit(0);
}

try {
  // --atomic: the branch and the tag land together, or neither does.
  execFileSync("git", ["push", "--atomic", "origin", "main", tag], { cwd: root, stdio: "inherit" });
} catch {
  fail(`the push failed. The commit and tag are still here. Retry with: git push --atomic origin main ${tag}`);
}

const repo = git("remote", "get-url", "origin").match(/github\.com[:/](.+?)(?:\.git)?$/)?.[1];
console.log(`Pushed ${tag}. GitHub is building the installers${repo ? `: https://github.com/${repo}/actions` : "."}`);
