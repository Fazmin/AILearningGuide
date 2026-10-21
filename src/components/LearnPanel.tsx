import {
  Bookmark,
  BookmarkCheck,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  History,
  ListTree,
  Square,
  Volume2,
  X,
} from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ComponentPropsWithoutRef, KeyboardEvent, ReactNode } from "react";
import type { ExplanationMode } from "@app/module-sdk";
import type { RegisteredModule } from "@app/modules/registry";
import { modules } from "@app/modules/registry";
import {
  LEARN_SECTION_SELECTOR,
  activeLearnIndex,
  collectLearnEntries,
  findLearnElement,
  findLearnEntry,
  groupLearnEntries,
  headingText,
  normalizeHeading,
  type LearnEntry,
  type LearnGroup,
} from "@app/lib/learn-sections";
import { canUseSpeech, speakLesson, stopSpeaking, textFromLearnRoot } from "@app/lib/speech";
import { useAppStore } from "@app/store/app-store";
import { Checkpoint } from "./Checkpoint";
import { IconButton } from "./IconButton";

const learnModeLabel: Record<ExplanationMode, string> = {
  simple: "simple language",
  plain: "plain language",
  standard: "standard",
};

const sectionNumber = (value: number) => String(value).padStart(2, "0");

interface LearnHeadingContextValue {
  isBookmarked: (section: string) => boolean;
  toggle: (section: string) => void;
}

const LearnHeadingContext = createContext<LearnHeadingContextValue | null>(null);

/**
 * A lesson heading with its bookmark toggle beside it, not inside it, so the heading's text (which
 * search, speech and the table of contents read) and its accessible name stay exactly the title.
 */
function LearnHeading({ level, children, ...props }: ComponentPropsWithoutRef<"h2"> & { level: 2 | 3 }) {
  const context = useContext(LearnHeadingContext);
  const text = headingText(children);
  const Heading = level === 2 ? "h2" : "h3";
  const saved = context?.isBookmarked(text) ?? false;
  return (
    <div className={`learn-heading learn-heading--h${level}`}>
      <Heading {...props} data-learn-section={text} data-learn-level={level}>{children}</Heading>
      {context && text && (
        <button
          type="button"
          className="learn-heading__bookmark"
          aria-pressed={saved}
          aria-label={`Bookmark “${text}”`}
          data-tooltip={saved ? "Remove bookmark" : "Bookmark this section"}
          data-tooltip-side="left"
          onClick={() => context.toggle(text)}
        >
          {saved ? <BookmarkCheck /> : <Bookmark />}
        </button>
      )}
    </div>
  );
}

// Defined once so the lesson keeps its DOM when the panel re-renders.
const learnComponents = {
  h2: (props: ComponentPropsWithoutRef<"h2">) => <LearnHeading level={2} {...props} />,
  h3: (props: ComponentPropsWithoutRef<"h3">) => <LearnHeading level={3} {...props} />,
};

function LearnContents({
  id,
  open,
  groups,
  activeIndex,
  bookmarked,
  resumeEntry,
  onJump,
  onResume,
  onClose,
}: {
  id: string;
  open: boolean;
  groups: LearnGroup[];
  activeIndex: number;
  bookmarked: Set<string>;
  resumeEntry: LearnEntry | null;
  onJump: (entry: LearnEntry) => void;
  onResume: () => void;
  onClose: () => void;
}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const listRef = useRef<HTMLElement>(null);
  const activeRef = useRef<HTMLButtonElement>(null);
  let activeGroup = 0;
  groups.forEach((group, index) => {
    if (group.index <= activeIndex) activeGroup = index;
  });
  const lessonSections = groups.filter((group) => group.entry.number !== null).length;

  useEffect(() => {
    if (open) activeRef.current?.focus({ preventScroll: true });
  }, [open]);

  // Keep the current entry in view inside the list without scrolling anything around it.
  useEffect(() => {
    const list = listRef.current;
    const item = activeRef.current;
    if (!list || !item || list.clientHeight === 0) return;
    const top = item.offsetTop - list.offsetTop;
    if (top < list.scrollTop + 48 || top > list.scrollTop + list.clientHeight - 48) {
      list.scrollTop = Math.max(0, top - list.clientHeight / 3);
    }
  }, [activeIndex, open, expanded]);

  const isSaved = (entry: LearnEntry) => bookmarked.has(normalizeHeading(entry.text));

  const item = (entry: LearnEntry, index: number, className: string, content: ReactNode, savedInside = false) => {
    const isActive = index === activeIndex;
    return (
      <button
        type="button"
        className={className}
        ref={isActive ? activeRef : undefined}
        aria-current={isActive ? "location" : undefined}
        onClick={() => onJump(entry)}
      >
        {content}
        {(isSaved(entry) || savedInside) && (
          <>
            <BookmarkCheck className={`learn-contents__saved${isSaved(entry) ? "" : " is-inside"}`} />
            <span className="sr-only">{isSaved(entry) ? ", bookmarked" : ", has bookmarked subsections"}</span>
          </>
        )}
      </button>
    );
  };

  return (
    <nav
      id={id}
      ref={listRef}
      className="learn-contents"
      data-open={open ? "true" : "false"}
      aria-label="Lesson contents"
      onKeyDown={(event: KeyboardEvent) => {
        if (event.key === "Escape" && open) {
          event.stopPropagation();
          onClose();
        }
      }}
    >
      <div className="learn-contents__head">
        <strong>Contents</strong>
        <span>{lessonSections} sections</span>
        <IconButton label="Close contents" tooltipSide="left" className="learn-contents__close" onClick={onClose}><X /></IconButton>
      </div>
      <i className="learn-progress" aria-hidden="true"><b /></i>
      {resumeEntry && (
        <button type="button" className="learn-contents__resume" onClick={onResume}>
          <History />
          <span><small>Continue where you left off</small><strong>{resumeEntry.text}</strong></span>
        </button>
      )}
      <ol className="learn-contents__list">
        {groups.map((group, groupIndex) => {
          const isOpen = expanded[group.entry.text] ?? groupIndex === activeGroup;
          return (
            <li
              key={group.entry.text}
              className={[groupIndex === activeGroup ? "is-active" : "", group.entry.number === null ? "is-extra" : ""].join(" ").trim() || undefined}
            >
              <div className="learn-contents__row">
                {item(
                  group.entry,
                  group.index,
                  "learn-contents__section",
                  <>
                    <span className="learn-contents__number" aria-hidden="true">
                      {group.entry.number !== null ? sectionNumber(group.entry.number) : ""}
                    </span>
                    <span className="learn-contents__label">{group.entry.text}</span>
                  </>,
                  // A collapsed section still shows that something inside it is bookmarked.
                  !isOpen && group.children.some((child) => isSaved(child.entry)),
                )}
                {group.children.length > 0 && (
                  <button
                    type="button"
                    className="learn-contents__expand"
                    aria-expanded={isOpen}
                    aria-label={`Subsections of ${group.entry.text}`}
                    onClick={() => setExpanded((current) => ({ ...current, [group.entry.text]: !isOpen }))}
                  >
                    <ChevronRight />
                  </button>
                )}
              </div>
              {isOpen && group.children.length > 0 && (
                <ol>
                  {group.children.map((child) => (
                    <li key={child.entry.text}>
                      {item(child.entry, child.index, "learn-contents__subsection", <span className="learn-contents__label">{child.entry.text}</span>)}
                    </li>
                  ))}
                </ol>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export function LearnPanel({
  module,
  onSectionChange,
}: {
  module: RegisteredModule;
  /** Reports the section being read, and null when the panel closes. */
  onSectionChange?: (entry: LearnEntry | null) => void;
}) {
  const mode = useAppStore((state) => state.mode);
  const setMode = useAppStore((state) => state.setMode);
  const searchFocus = useAppStore((state) => state.searchFocus);
  const setSearchFocus = useAppStore((state) => state.setSearchFocus);
  const setView = useAppStore((state) => state.setView);
  const reducedMotion = useAppStore((state) => state.reducedMotion);
  const bookmarks = useAppStore((state) => state.bookmarks);
  const toggleBookmark = useAppStore((state) => state.toggleBookmark);
  const setReadingPosition = useAppStore((state) => state.setReadingPosition);
  const isLastModule = modules[modules.length - 1]?.id === module.id;
  const Content = module.content[mode];
  const [showGlossary, setShowGlossary] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [entries, setEntries] = useState<LearnEntry[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [contentsOpen, setContentsOpen] = useState(false);
  // Offered once per visit: where this lab was last read, unless the visit is already headed somewhere.
  const [resume, setResume] = useState(() => {
    const { searchFocus: pending, readingPositions } = useAppStore.getState();
    if (pending && "moduleId" in pending && pending.moduleId === module.id) return null;
    return readingPositions[module.id] ?? null;
  });
  const panelRef = useRef<HTMLElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const userScrolled = useRef(false);
  const contentsId = useId();
  const speechSupported = canUseSpeech();

  const groups = useMemo(() => groupLearnEntries(entries), [entries]);
  const active = entries[activeIndex] ?? null;
  let activeGroup: LearnGroup | null = null;
  for (const group of groups) if (group.index <= activeIndex) activeGroup = group;
  const resumeIndex = resume ? findLearnEntry(entries, resume.section, resume.parent) : -1;
  // The first section is where the panel opens anyway.
  const resumeEntry = resumeIndex > 0 ? entries[resumeIndex] : null;

  const bookmarked = useMemo(
    () => new Set(bookmarks.filter((item) => item.moduleId === module.id).map((item) => normalizeHeading(item.section))),
    [bookmarks, module.id],
  );
  const activeSaved = active ? bookmarked.has(normalizeHeading(active.text)) : false;

  const toggleSection = useCallback(
    (section: string) => {
      const entry = entries[findLearnEntry(entries, section)];
      toggleBookmark({ moduleId: module.id, mode, section, parent: entry?.parent ?? section });
    },
    [entries, mode, module.id, toggleBookmark],
  );

  const headingContext = useMemo<LearnHeadingContextValue>(
    () => ({ isBookmarked: (section) => bookmarked.has(normalizeHeading(section)), toggle: toggleSection }),
    [bookmarked, toggleSection],
  );

  // Rendered once per mode: scrolling and bookmarking reach the headings through context instead.
  const lesson = useMemo(() => <Content components={learnComponents} />, [Content]);

  const reveal = useCallback(
    (target: HTMLElement) => {
      target.dataset.searchHit = "true";
      target.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start", inline: "nearest" });
      // Move focus too, so keyboard and screen reader users continue from the section they chose.
      if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
      target.focus({ preventScroll: true });
      window.setTimeout(() => {
        delete target.dataset.searchHit;
      }, 1600);
    },
    [reducedMotion],
  );

  const jumpTo = (entry: LearnEntry) => {
    const target = scrollRef.current && findLearnElement(scrollRef.current, entry.text);
    if (!target) return;
    setContentsOpen(false);
    reveal(target);
  };

  const resumeReading = () => {
    if (resumeEntry) jumpTo(resumeEntry);
    setResume(null);
  };

  const closeContents = () => {
    setContentsOpen(false);
    toggleRef.current?.focus();
  };

  useEffect(() => () => stopSpeaking(), []);

  useEffect(() => {
    stopSpeaking();
    setSpeaking(false);
  }, [mode, module.id]);

  useLayoutEffect(() => {
    if (scrollRef.current) setEntries(collectLearnEntries(scrollRef.current));
  }, [Content]);

  useEffect(() => {
    const root = scrollRef.current;
    const panel = panelRef.current;
    if (!root || !panel) return;
    let frame = 0;
    const measure = () => {
      frame = 0;
      const rootTop = root.getBoundingClientRect().top;
      const tops = [...root.querySelectorAll<HTMLElement>(LEARN_SECTION_SELECTOR)].map(
        (element) => element.getBoundingClientRect().top - rootTop,
      );
      const max = root.scrollHeight - root.clientHeight;
      const atBottom = max > 0 && root.scrollTop >= max - 2;
      setActiveIndex(activeLearnIndex(tops, root.clientHeight, Math.min(root.clientHeight * 0.25, 96), atBottom));
      panel.style.setProperty("--learn-progress", String(max > 0 ? Math.min(1, root.scrollTop / max) : 1));
    };
    const onScroll = () => {
      userScrolled.current = true;
      if (!frame) frame = window.requestAnimationFrame(measure);
    };
    measure();
    root.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      root.removeEventListener("scroll", onScroll);
      window.cancelAnimationFrame(frame);
    };
  }, [entries]);

  // Remember the section once the learner has scrolled to it, not merely because the panel opened at the top.
  useEffect(() => {
    if (!active || !userScrolled.current) return;
    const timer = window.setTimeout(() => {
      setReadingPosition(module.id, { mode, section: active.text, parent: active.parent });
      setResume(null);
    }, 600);
    return () => window.clearTimeout(timer);
  }, [active, mode, module.id, setReadingPosition]);

  useEffect(() => {
    onSectionChange?.(active);
  }, [active, onSectionChange]);

  useEffect(() => () => onSectionChange?.(null), [onSectionChange]);

  useEffect(() => {
    if (!searchFocus || !("moduleId" in searchFocus) || searchFocus.moduleId !== module.id) return;
    if (searchFocus.kind === "glossary") setShowGlossary(true);
    if (searchFocus.kind === "learn" || searchFocus.kind === "checkpoint") setShowGlossary(false);
  }, [module.id, searchFocus]);

  useEffect(() => {
    if (!searchFocus || !("moduleId" in searchFocus) || searchFocus.moduleId !== module.id) return;
    if (!["learn", "glossary", "checkpoint"].includes(searchFocus.kind)) return;
    const panel = panelRef.current;
    const root = scrollRef.current;
    if (!panel || !root) return;

    let inner = 0;
    const frame = window.requestAnimationFrame(() => {
      inner = window.requestAnimationFrame(() => {
        let target: HTMLElement | null = null;
        if (searchFocus.kind === "glossary") {
          target = panel.querySelector(`[data-glossary-term="${CSS.escape(searchFocus.term)}"]`);
        } else if (searchFocus.kind === "checkpoint") {
          target = root.querySelector("[data-search-target=checkpoint]");
        } else if (searchFocus.kind === "learn") {
          target = findLearnElement(root, searchFocus.section);
          if (!target && searchFocus.mode !== mode) {
            // A subsection saved in another reading mode: switch to that mode, then look again.
            setMode(searchFocus.mode);
            return;
          }
          if (!target && searchFocus.parent) target = findLearnElement(root, searchFocus.parent);
        }
        if (!target) return;
        setContentsOpen(false);
        reveal(target);
        setSearchFocus(null);
      });
    });
    return () => {
      window.cancelAnimationFrame(frame);
      window.cancelAnimationFrame(inner);
    };
  }, [mode, module.id, reveal, searchFocus, setMode, setSearchFocus, showGlossary]);

  return (
    <aside ref={panelRef} className="learn-panel" aria-label={`Learn about ${module.title}`}>
      <div className="learn-panel__header">
        <div>
          <span>Learn / {learnModeLabel[mode]}</span>
          <strong>Keep the idea beside the experiment.</strong>
        </div>
        <div className="learn-panel__actions">
          <button
            type="button"
            className="learn-speak"
            aria-pressed={speaking}
            aria-label={speaking ? "Stop reading" : "Read out loud"}
            disabled={!speechSupported}
            title={speechSupported ? undefined : "Speech is not available in this window."}
            onClick={() => {
              if (speaking) {
                stopSpeaking();
                setSpeaking(false);
                return;
              }
              const text = scrollRef.current ? textFromLearnRoot(scrollRef.current) : "";
              if (!text) return;
              setSpeaking(true);
              void speakLesson(text)
                .catch(() => undefined)
                .finally(() => setSpeaking(false));
            }}
          >
            {speaking ? <Square /> : <Volume2 />}
            <span>{speaking ? "Stop reading" : "Read out loud"}</span>
          </button>
          <button type="button" className="glossary-toggle" aria-expanded={showGlossary} onClick={() => setShowGlossary(!showGlossary)}>
            <span>Glossary</span>
            <small>{module.glossary.length}</small>
          </button>
        </div>
      </div>
      {showGlossary && (
        <div className="glossary-panel">
          {module.glossary.map((item) => (
            <div key={item.term} data-glossary-term={item.term}><strong>{item.term}</strong><span>{item.definition}</span></div>
          ))}
        </div>
      )}
      <div className="learn-sectionbar">
        <button
          ref={toggleRef}
          type="button"
          className="learn-sectionbar__toggle"
          aria-expanded={contentsOpen}
          aria-controls={contentsId}
          onClick={() => setContentsOpen(!contentsOpen)}
        >
          <ListTree />
          <span className="learn-sectionbar__where">
            <span className="sr-only">Contents. Now reading: </span>
            {activeGroup?.entry.number != null && (
              <small aria-hidden="true">{sectionNumber(activeGroup.entry.number)}</small>
            )}
            <strong>{activeGroup?.entry.text ?? "Contents"}</strong>
            {active?.level === 3 && <span className="learn-sectionbar__sub">{active.text}</span>}
          </span>
          <ChevronDown className="learn-sectionbar__chevron" />
        </button>
        {active && (
          <button
            type="button"
            className="learn-sectionbar__bookmark"
            aria-pressed={activeSaved}
            aria-label={`Bookmark “${active.text}”`}
            data-tooltip={activeSaved ? "Remove bookmark" : "Bookmark this section"}
            data-tooltip-side="left"
            onClick={() => toggleSection(active.text)}
          >
            {activeSaved ? <BookmarkCheck /> : <Bookmark />}
          </button>
        )}
        <i className="learn-progress" aria-hidden="true"><b /></i>
      </div>
      <div className="learn-panel__body">
        <LearnContents
          id={contentsId}
          open={contentsOpen}
          groups={groups}
          activeIndex={activeIndex}
          bookmarked={bookmarked}
          resumeEntry={resumeEntry}
          onJump={jumpTo}
          onResume={resumeReading}
          onClose={closeContents}
        />
        <div className="learn-panel__scroll" ref={scrollRef}>
          <div className="objectives">
            <span>After this lab, you can</span>
            {module.objectives.map((objective) => <p key={objective}><CheckCircle2 />{objective}</p>)}
          </div>
          <LearnHeadingContext.Provider value={headingContext}>
            <article className="learn-copy">{lesson}</article>
          </LearnHeadingContext.Provider>
          <div data-search-target="checkpoint" data-learn-section="Checkpoint" data-learn-level="2" data-learn-extra="">
            <Checkpoint key={module.id} module={module} />
          </div>
          {module.references && module.references.length > 0 && (
            <section className="references" aria-labelledby={`references-${module.id}`}>
              <h2 id={`references-${module.id}`} data-learn-section="References" data-learn-level="2" data-learn-extra="">References</h2>
              <ol>
                {module.references.map((reference) => (
                  <li key={reference.url}>
                    <a href={reference.url} target="_blank" rel="noreferrer">{reference.title}</a>
                    <span>{reference.authors} ({reference.year}). {reference.source}.</span>
                    <p>{reference.note}</p>
                  </li>
                ))}
              </ol>
            </section>
          )}
          {isLastModule && (
            <section className="end-of-path" aria-labelledby={`end-of-path-${module.id}`}>
              <strong id={`end-of-path-${module.id}`}>That was the last lab.</strong>
              <p>
                You’ve reached the end of the learning path. Open any lab from the learning map to retake its
                checkpoint or try a different experiment, or export your progress from Settings.
              </p>
              <button type="button" onClick={() => setView("home")}>Back to the learning map</button>
            </section>
          )}
        </div>
        {/* Laid over the text rather than in it, so it can go away without moving what is being read. */}
        {resumeEntry && (
          <div className="learn-resume" role="status">
            <History />
            <div>
              <span>Pick up where you left off</span>
              <strong>
                {resumeEntry.level === 3 && `${resumeEntry.parent} · `}
                {resumeEntry.text}
              </strong>
            </div>
            <button type="button" className="learn-resume__go" onClick={resumeReading}>Continue</button>
            <IconButton label="Dismiss" tooltipSide="left" onClick={() => setResume(null)}><X /></IconButton>
          </div>
        )}
      </div>
    </aside>
  );
}
