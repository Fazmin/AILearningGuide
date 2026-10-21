import { isValidElement, type ReactNode } from "react";

/**
 * The Learn panel's table of contents, built from the rendered lesson.
 *
 * Every jump target in the panel carries `data-learn-section` (its title) and `data-learn-level`:
 * the lesson's own `##` and `###` headings, plus the checkpoint and references that follow the
 * lesson. Titles are compared after `normalizeHeading`, so curly and straight quotes match.
 */

export interface LearnEntry {
  /** Heading text as the learner sees it. */
  text: string;
  level: 2 | 3;
  /** The top-level section this entry sits in; equal to `text` for a top-level entry. */
  parent: string;
  /** 1-based number of a lesson section (`##`), null for subsections and for the checkpoint and references. */
  number: number | null;
}

export interface LearnGroup {
  entry: LearnEntry;
  /** Position of `entry` in the flat list. */
  index: number;
  children: Array<{ entry: LearnEntry; index: number }>;
}

export const LEARN_SECTION_SELECTOR = "[data-learn-section]";

export function normalizeHeading(value: string) {
  return value.replace(/[’‘ʻʼ`']/g, "'").replace(/\s+/g, " ").trim().toLowerCase();
}

export function sameSection(a: string, b: string) {
  return normalizeHeading(a) === normalizeHeading(b);
}

/** The text React will render for `children`, so a heading can name itself before it is in the DOM. */
export function textOfNode(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOfNode).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return textOfNode(node.props.children);
  return "";
}

export function headingText(node: ReactNode) {
  return textOfNode(node).replace(/\s+/g, " ").trim();
}

/** Read the panel's jump targets in document order. */
export function collectLearnEntries(root: ParentNode): LearnEntry[] {
  const entries: LearnEntry[] = [];
  let parent = "";
  let number = 0;
  root.querySelectorAll<HTMLElement>(LEARN_SECTION_SELECTOR).forEach((element) => {
    const text = element.dataset.learnSection?.trim();
    if (!text) return;
    const level = element.dataset.learnLevel === "3" ? 3 : 2;
    if (level === 2) parent = text;
    const inLesson = element.dataset.learnExtra === undefined;
    entries.push({
      text,
      level,
      parent: level === 2 ? text : parent || text,
      number: level === 2 && inLesson ? ++number : null,
    });
  });
  return entries;
}

export function groupLearnEntries(entries: LearnEntry[]): LearnGroup[] {
  const groups: LearnGroup[] = [];
  entries.forEach((entry, index) => {
    if (entry.level === 2 || groups.length === 0) {
      groups.push({ entry, index, children: [] });
    } else {
      groups[groups.length - 1].children.push({ entry, index });
    }
  });
  return groups;
}

/** Index of the entry for `section`, falling back to its top-level `parent`; -1 when neither is present. */
export function findLearnEntry(entries: LearnEntry[], section: string, parent?: string) {
  const exact = entries.findIndex((entry) => sameSection(entry.text, section));
  if (exact !== -1 || !parent) return exact;
  return entries.findIndex((entry) => entry.level === 2 && sameSection(entry.text, parent));
}

export function findLearnElement(root: ParentNode, section: string) {
  return (
    [...root.querySelectorAll<HTMLElement>(LEARN_SECTION_SELECTOR)].find((element) =>
      sameSection(element.dataset.learnSection ?? "", section),
    ) ?? null
  );
}

/**
 * The entry being read: the last one whose top has passed `threshold` pixels below the top of the
 * scroll area. At the very bottom, where later headings can never reach the threshold, it is the
 * last one on screen.
 */
export function activeLearnIndex(tops: number[], viewportHeight: number, threshold: number, atBottom: boolean) {
  let active = 0;
  tops.forEach((top, index) => {
    if (top <= threshold || (atBottom && top < viewportHeight)) active = index;
  });
  return active;
}
