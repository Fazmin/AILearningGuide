import type { SearchDocument, SearchHit } from "./types";
import { SEARCH_MARK_END, SEARCH_MARK_START } from "./types";

const TOKEN_PATTERN = /[0-9a-zA-Z\u00c0-\u024f]+/g;

export function tokenizeQuery(query: string) {
  return [...query.toLowerCase().matchAll(TOKEN_PATTERN)]
    .map((match) => match[0])
    .filter((token) => token.length >= 2);
}

export function ftsQuery(query: string) {
  const tokens = tokenizeQuery(query)
    .map((token) => token.replace(/[^0-9a-zA-Z\u00c0-\u024f]/g, ""))
    .filter(Boolean)
    .map((token) => `${token}*`);
  return tokens.length ? tokens.join(" AND ") : null;
}

function normalize(value: string) {
  return value.toLowerCase();
}

function markSnippet(text: string, tokens: string[], width = 110) {
  const source = collapseWhitespace(text);
  if (!source) return "";
  const haystack = normalize(source);
  let found = -1;
  let matched = "";
  for (const token of tokens) {
    const index = haystack.indexOf(token);
    if (index !== -1 && (found === -1 || index < found)) {
      found = index;
      matched = source.slice(index, index + token.length);
    }
  }
  if (found === -1) {
    const excerpt = source.slice(0, width);
    return excerpt.length < source.length ? `${excerpt}…` : excerpt;
  }
  const start = Math.max(0, found - Math.floor(width / 3));
  const end = Math.min(source.length, start + width);
  const window = source.slice(start, end);
  const local = found - start;
  const highlighted =
    window.slice(0, local) +
    SEARCH_MARK_START +
    (matched || window.slice(local, local + 2)) +
    SEARCH_MARK_END +
    window.slice(local + (matched || window.slice(local, local + 2)).length);
  return `${start > 0 ? "…" : ""}${highlighted}${end < source.length ? "…" : ""}`;
}

function collapseWhitespace(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function fieldScore(haystack: string, tokens: string[], weight: number) {
  if (!haystack) return 0;
  let score = 0;
  for (const token of tokens) {
    if (haystack === token) score += weight * 3;
    else if (haystack.startsWith(token)) score += weight * 2;
    else if (haystack.includes(token)) score += weight;
  }
  return score;
}

export function scoreDocument(document: SearchDocument, tokens: string[]) {
  const title = normalize(document.title);
  const subtitle = normalize(document.subtitle);
  const body = normalize(document.body);
  const haystacks = [title, subtitle, body];
  const matchesEveryToken = tokens.every((token) => haystacks.some((field) => field.includes(token)));
  if (!matchesEveryToken) return 0;

  const raw =
    fieldScore(title, tokens, 8) +
    fieldScore(subtitle, tokens, 3) +
    fieldScore(body, tokens, 1);
  return raw * document.rankBoost;
}

export function snippetForDocument(document: SearchDocument, tokens: string[]) {
  const preferred = [document.subtitle, document.body, document.title].find((value) => {
    const haystack = normalize(value);
    return tokens.some((token) => haystack.includes(token));
  });
  return markSnippet(preferred || document.body || document.subtitle || document.title, tokens);
}

export function searchDocuments(
  documents: readonly SearchDocument[],
  query: string,
  limit = 32,
): SearchHit[] {
  const tokens = tokenizeQuery(query);
  if (!tokens.length) return [];

  return documents
    .map((document) => {
      const score = scoreDocument(document, tokens);
      if (score <= 0) return null;
      return {
        ...document,
        score,
        snippet: snippetForDocument(document, tokens),
      };
    })
    .filter((hit): hit is SearchHit => hit !== null)
    .sort((left, right) => right.score - left.score || left.title.localeCompare(right.title))
    .slice(0, limit);
}
