import type { ExplanationMode } from "@app/module-sdk";
import { sameSection } from "@app/lib/learn-sections";

/**
 * Bookmarks and reading positions for the Learn panel, kept pure so they can be tested without the store.
 *
 * A bookmark names a section by its heading. The seven lesson sections share their headings across
 * the three reading modes, but subsections do not, so a bookmark also records the mode it was made in
 * and the top-level section it sits under: opening it uses the mode only when the current one has no
 * such heading, and lands on the top-level section when the heading has gone altogether.
 */

export interface LearnBookmark {
  id: string;
  moduleId: string;
  mode: ExplanationMode;
  section: string;
  parent: string;
  createdAt: string;
}

export interface ReadingPosition {
  mode: ExplanationMode;
  section: string;
  parent: string;
  updatedAt: string;
}

export type BookmarkTarget = Pick<LearnBookmark, "moduleId" | "mode" | "section" | "parent">;

const modes: ExplanationMode[] = ["simple", "plain", "standard"];

const isMode = (value: unknown): value is ExplanationMode => modes.includes(value as ExplanationMode);
const isText = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;

export function findBookmark(bookmarks: LearnBookmark[], moduleId: string, section: string) {
  return bookmarks.find((bookmark) => bookmark.moduleId === moduleId && sameSection(bookmark.section, section));
}

/** Add a bookmark for `target`, or remove the one that already names the same section of the same lab. */
export function toggleBookmarkIn(
  bookmarks: LearnBookmark[],
  target: BookmarkTarget,
  stamp: () => Pick<LearnBookmark, "id" | "createdAt">,
): LearnBookmark[] {
  const existing = findBookmark(bookmarks, target.moduleId, target.section);
  if (existing) return bookmarks.filter((bookmark) => bookmark !== existing);
  return [{ ...target, ...stamp() }, ...bookmarks];
}

/** Keep the well-formed bookmarks from an import, one per section. */
export function sanitizeBookmarks(value: unknown): LearnBookmark[] {
  if (!Array.isArray(value)) return [];
  const kept: LearnBookmark[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") continue;
    const candidate = item as Record<string, unknown>;
    if (
      !isText(candidate.id) ||
      !isText(candidate.moduleId) ||
      !isMode(candidate.mode) ||
      !isText(candidate.section) ||
      typeof candidate.createdAt !== "string"
    ) {
      continue;
    }
    if (findBookmark(kept, candidate.moduleId, candidate.section)) continue;
    kept.push({
      id: candidate.id,
      moduleId: candidate.moduleId,
      mode: candidate.mode,
      section: candidate.section,
      parent: isText(candidate.parent) ? candidate.parent : candidate.section,
      createdAt: candidate.createdAt,
    });
  }
  return kept;
}

export function sanitizeReadingPositions(value: unknown): Record<string, ReadingPosition> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const kept: Record<string, ReadingPosition> = {};
  for (const [moduleId, item] of Object.entries(value as Record<string, unknown>)) {
    if (!item || typeof item !== "object") continue;
    const candidate = item as Record<string, unknown>;
    if (!isMode(candidate.mode) || !isText(candidate.section)) continue;
    kept[moduleId] = {
      mode: candidate.mode,
      section: candidate.section,
      parent: isText(candidate.parent) ? candidate.parent : candidate.section,
      updatedAt: typeof candidate.updatedAt === "string" ? candidate.updatedAt : new Date(0).toISOString(),
    };
  }
  return kept;
}
