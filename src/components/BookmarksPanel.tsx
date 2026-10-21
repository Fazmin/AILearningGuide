import { BookmarkCheck, BookmarkPlus, Trash2, X } from "lucide-react";
import { useEffect, useRef } from "react";
import type { ExplanationMode } from "@app/module-sdk";
import type { LearnEntry } from "@app/lib/learn-sections";
import type { RegisteredModule } from "@app/modules/registry";
import { modules } from "@app/modules/registry";
import { useAppStore, type LearnBookmark } from "@app/store/app-store";
import { findBookmark } from "@app/store/bookmarks";
import { IconButton } from "./IconButton";

const modeName: Record<ExplanationMode, string> = {
  simple: "Simple",
  plain: "Plain",
  standard: "Standard",
};

/**
 * Bookmarked Learn sections, this lab's first and then every other lab's in path order. Opening one
 * goes through the same focus the search palette uses, so the Learn panel scrolls to it.
 */
export function BookmarksPanel({
  module,
  section,
  onClose,
}: {
  module: RegisteredModule;
  /** The section the Learn panel is showing, or null when the panel is closed. */
  section: LearnEntry | null;
  onClose: () => void;
}) {
  const bookmarks = useAppStore((state) => state.bookmarks);
  const mode = useAppStore((state) => state.mode);
  const toggleBookmark = useAppStore((state) => state.toggleBookmark);
  const removeBookmark = useAppStore((state) => state.removeBookmark);
  const setSearchFocus = useAppStore((state) => state.setSearchFocus);
  const setLearnOpen = useAppStore((state) => state.setLearnOpen);
  const openModule = useAppStore((state) => state.openModule);
  const panelRef = useRef<HTMLDivElement>(null);
  const here = bookmarks.filter((bookmark) => bookmark.moduleId === module.id);
  const elsewhere = modules
    .filter((item) => item.id !== module.id)
    .map((item) => ({ module: item, items: bookmarks.filter((bookmark) => bookmark.moduleId === item.id) }))
    .filter((group) => group.items.length > 0);
  const current = section ? findBookmark(bookmarks, module.id, section.text) : undefined;

  useEffect(() => {
    panelRef.current?.focus();
  }, []);

  const open = (bookmark: LearnBookmark) => {
    onClose();
    setSearchFocus({
      kind: "learn",
      moduleId: bookmark.moduleId,
      mode: bookmark.mode,
      section: bookmark.section,
      parent: bookmark.parent,
    });
    setLearnOpen(true);
    if (bookmark.moduleId !== module.id) openModule(bookmark.moduleId);
  };

  const row = (bookmark: LearnBookmark) => {
    const subsection = bookmark.section !== bookmark.parent;
    const details = [
      subsection ? `in ${bookmark.parent}` : "",
      // Only subsections differ between reading modes, so only they can open in another one.
      subsection && bookmark.mode !== mode ? `${modeName[bookmark.mode]} reading` : "",
      new Date(bookmark.createdAt).toLocaleDateString([], { dateStyle: "medium" }),
    ].filter(Boolean);
    return (
      <li key={bookmark.id}>
        <button type="button" onClick={() => open(bookmark)}>
          <strong>{bookmark.section}</strong>
          <span>{details.join(" · ")}</span>
        </button>
        <IconButton label={`Remove bookmark “${bookmark.section}”`} tooltipSide="left" onClick={() => removeBookmark(bookmark.id)}>
          <Trash2 />
        </IconButton>
      </li>
    );
  };

  return (
    <div
      ref={panelRef}
      tabIndex={-1}
      className="floating-panel bookmark-panel"
      role="dialog"
      aria-label="Bookmarks"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation();
          onClose();
        }
      }}
    >
      <div className="floating-panel__heading">
        <div><strong>Bookmarks</strong><span>Sections you saved to come back to.</span></div>
        <IconButton label="Close bookmarks" onClick={onClose}><X /></IconButton>
      </div>
      {section ? (
        <button
          type="button"
          className="bookmark-panel__current"
          aria-pressed={Boolean(current)}
          onClick={() => toggleBookmark({ moduleId: module.id, mode, section: section.text, parent: section.parent })}
        >
          {current ? <BookmarkCheck /> : <BookmarkPlus />}
          <span>
            <small>{current ? "Bookmarked · select to remove" : "Bookmark where you are reading"}</small>
            <strong>{section.text}</strong>
          </span>
        </button>
      ) : (
        <p className="bookmark-panel__hint">Open the Learn panel to bookmark the section you are reading.</p>
      )}
      <div className="bookmark-list">
        {here.length === 0 && elsewhere.length === 0 && (
          <p className="bookmark-list__empty">
            No bookmarks yet. Select the bookmark beside any heading in the Learn panel, and the section will be
            listed here.
          </p>
        )}
        {here.length > 0 && (
          <section aria-labelledby="bookmarks-here">
            <h3 id="bookmarks-here">This lab</h3>
            <ul>{here.map(row)}</ul>
          </section>
        )}
        {elsewhere.map((group) => (
          <section key={group.module.id} aria-labelledby={`bookmarks-${group.module.id}`}>
            <h3 id={`bookmarks-${group.module.id}`}>
              <span>Lab {String(group.module.order).padStart(2, "0")}</span> {group.module.title}
            </h3>
            <ul>{group.items.map(row)}</ul>
          </section>
        ))}
      </div>
    </div>
  );
}
