import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { buildSearchIndex } from "./lib/search-documents.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputPath = path.join(root, "src", "search", "content-index.json");
const checkOnly = process.argv.includes("--check");

const index = await buildSearchIndex(root);
const serialized = `${JSON.stringify(index)}\n`;

if (checkOnly) {
  let existing = "";
  try {
    existing = await readFile(outputPath, "utf8");
  } catch {
    console.error("Search index is missing. Run `npm run index:search`.");
    process.exit(1);
  }

  let parsed;
  try {
    parsed = JSON.parse(existing);
  } catch {
    console.error("Search index is not valid JSON. Run `npm run index:search`.");
    process.exit(1);
  }

  if (parsed.fingerprint !== index.fingerprint || parsed.documentCount !== index.documentCount) {
    console.error(
      "Search index is stale. Run `npm run index:search` after changing module content.",
    );
    process.exit(1);
  }

  console.log(`Search index is current (${parsed.documentCount} documents).`);
  process.exit(0);
}

await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, serialized);
console.log(`Wrote ${index.documentCount} search documents to ${path.relative(root, outputPath)}.`);
