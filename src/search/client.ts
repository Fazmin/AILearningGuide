import { invoke } from "@tauri-apps/api/core";
import index from "./content-index.json";
import { searchDocuments } from "./query";
import type { SearchHit, SearchIndexFile } from "./types";

const bundled = index as SearchIndexFile;

export function bundledSearchDocuments() {
  return bundled.documents;
}

function desktopSearch() {
  return Boolean(window.__TAURI_INTERNALS__);
}

export async function querySearchIndex(query: string, limit = 32): Promise<SearchHit[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];

  if (desktopSearch()) {
    try {
      const hits = await invoke<SearchHit[]>("search_content", { query: trimmed, limit });
      if (Array.isArray(hits)) return hits;
    } catch {
      // The packaged FTS index is preferred; the bundled documents stay available in the browser.
    }
  }

  return searchDocuments(bundled.documents, trimmed, limit);
}
