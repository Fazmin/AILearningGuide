import { describe, expect, it } from "vitest";
import index from "./content-index.json";
import { ftsQuery, searchDocuments, tokenizeQuery } from "./query";
import { focusFromDocument, type SearchIndexFile } from "./types";

const documents = (index as SearchIndexFile).documents;

describe("search query helpers", () => {
  it("tokenizes and builds a prefix FTS query", () => {
    expect(tokenizeQuery("  Attention head ")).toEqual(["attention", "head"]);
    expect(ftsQuery("Attention head")).toBe("attention* AND head*");
    expect(ftsQuery("a")).toBeNull();
  });
});

describe("packaged search index", () => {
  it("indexes every shipped module and both explanation modes", () => {
    const modules = documents.filter((document) => document.kind === "module");
    expect(modules.length).toBeGreaterThanOrEqual(28);
    expect(documents.some((document) => document.kind === "learn" && document.mode === "plain")).toBe(true);
    expect(documents.some((document) => document.kind === "learn" && document.mode === "standard")).toBe(true);
    expect(documents.some((document) => document.kind === "glossary")).toBe(true);
    expect(documents.some((document) => document.kind === "card")).toBe(true);
    expect(documents.some((document) => document.id === "screen:home")).toBe(true);
  });

  it("jumps from a selected hit to a concrete destination", () => {
    const attention = documents.find((document) => document.moduleSlug === "attention" && document.kind === "module");
    const queryTerm = documents.find(
      (document) => document.moduleSlug === "attention" && document.kind === "glossary" && document.title === "Query",
    );
    // "softmax attention" now ranks the tiled-softmax lesson in modern-attention-at-scale
    // first, so aim at the query–key mechanism the attention lab owns.
    const softmax = searchDocuments(documents, "query key softmax")[0];

    expect(attention && focusFromDocument(attention)).toEqual({
      kind: "module",
      moduleId: "module-08-attention",
    });
    expect(queryTerm && focusFromDocument(queryTerm)).toMatchObject({
      kind: "glossary",
      moduleId: "module-08-attention",
      term: "Query",
    });
    expect(softmax?.moduleSlug).toBe("attention");
    expect(softmax?.snippet).toMatch(/[«»]/);
  });

  it("finds settings, glossary terms, and lab cards by their own words", () => {
    expect(searchDocuments(documents, "keychain")[0]?.settingsTab).toBe("cloud");
    expect(searchDocuments(documents, "induction head")[0]?.kind).toBe("glossary");
    expect(searchDocuments(documents, "arc diagram")[0]?.kind).toBe("card");
  });
});
