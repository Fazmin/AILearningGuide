/**
 * Keeps the app version the same in every file that records it (listed in scripts/lib/version.mjs).
 *
 *   node scripts/version.mjs              print the version
 *   node scripts/version.mjs --check      fail if any file disagrees with package.json
 *   node scripts/version.mjs --set 0.2.0  write a new version to every file
 *
 * To publish a release, run `npm run release` instead. It picks the next version for you.
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { currentVersion, readVersions, setVersion } from "./lib/version.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);

if (args[0] === "--set") {
  try {
    setVersion(root, args[1]);
  } catch (error) {
    console.error(`version: ${error.message}`);
    process.exit(1);
  }
  console.log(`Set the version to ${args[1]}.`);
} else if (args[0] === "--check") {
  const expected = currentVersion(root);
  const wrong = readVersions(root).filter(({ version }) => version !== expected);
  if (wrong.length > 0) {
    console.error(`check:version: package.json says ${expected}, but:`);
    for (const { file, version } of wrong) console.error(`  ${file} says ${version}`);
    console.error(`Run \`node scripts/version.mjs --set ${expected}\` to make them match.`);
    process.exit(1);
  }
  console.log(`Version ${expected} is the same in every file.`);
} else if (args.length === 0) {
  console.log(currentVersion(root));
} else {
  console.error("Usage: node scripts/version.mjs [--check | --set <version>]");
  process.exit(1);
}
