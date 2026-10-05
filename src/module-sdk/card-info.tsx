import { Info, X } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";

/**
 * Long-form explanation of one lab surface. The sections mirror the obligations in
 * CONTENT_STYLE_GUIDE.md so a card explanation states the mechanism, the visible
 * signal, and the limits rather than only naming the parts.
 */
export interface CardInfo {
  /** Plain-language name of the visualization, shown as the panel title. */
  title: string;
  /** The smallest accurate model of the mechanism, in one or two sentences. */
  summary: string;
  /** Every visual element decoded: axes, marks, colours, animation, readouts. */
  whatYouSee: string[];
  /** The computation behind the picture, including the formulas actually evaluated. */
  howItWorks: string[];
  /** What each control changes, and the comparison it is worth running. */
  controls?: string[];
  /** The signal that distinguishes outcomes once the learner starts experimenting. */
  notice?: string[];
  /** Limits of this lab and of the real mechanism, kept as separate claims. */
  limits: string[];
}

/** Explanations for one module, keyed by the `label` of each `LabSurface`. */
export type ModuleCardInfo = Record<string, CardInfo>;

const CardInfoContext = createContext<ModuleCardInfo>({});
const CardFocusContext = createContext<string | null>(null);

export function CardInfoProvider({
  value,
  focusLabel = null,
  children,
}: {
  value: ModuleCardInfo | undefined;
  focusLabel?: string | null;
  children?: React.ReactNode;
}) {
  return (
    <CardInfoContext.Provider value={value ?? {}}>
      <CardFocusContext.Provider value={focusLabel}>{children}</CardFocusContext.Provider>
    </CardInfoContext.Provider>
  );
}

/** Resolves the explanation registered for a surface label, if the module ships one. */
export function useCardInfo(label: string): CardInfo | undefined {
  return useContext(CardInfoContext)[label];
}

const PANEL_GAP = 10;
const VIEWPORT_MARGIN = 12;
const OPEN_DELAY = 110;
const CLOSE_DELAY = 160;

interface Placement {
  top: number;
  left: number;
  side: "top" | "bottom";
}

/**
 * Info affordance for a lab surface. Opens on hover and on keyboard focus, and a
 * click pins it open so the panel can be read, scrolled, and dismissed on touch
 * devices and by pointer users who do not want to hold a hover.
 */
export function CardInfoButton({ info, label }: { info: CardInfo; label: string }) {
  const focusLabel = useContext(CardFocusContext);
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [placement, setPlacement] = useState<Placement | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const openTimer = useRef<number | null>(null);
  const closeTimer = useRef<number | null>(null);
  const panelId = useId();

  const clearTimers = useCallback(() => {
    if (openTimer.current !== null) window.clearTimeout(openTimer.current);
    if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
    openTimer.current = null;
    closeTimer.current = null;
  }, []);

  useEffect(() => clearTimers, [clearTimers]);

  useEffect(() => {
    if (focusLabel !== label) return;
    clearTimers();
    setPinned(true);
    setOpen(true);
  }, [clearTimers, focusLabel, label]);

  const scheduleOpen = useCallback(() => {
    clearTimers();
    openTimer.current = window.setTimeout(() => setOpen(true), OPEN_DELAY);
  }, [clearTimers]);

  const scheduleClose = useCallback(() => {
    if (pinned) return;
    clearTimers();
    closeTimer.current = window.setTimeout(() => setOpen(false), CLOSE_DELAY);
  }, [clearTimers, pinned]);

  const dismiss = useCallback(
    (returnFocus: boolean) => {
      clearTimers();
      setPinned(false);
      setOpen(false);
      if (returnFocus) buttonRef.current?.focus();
    },
    [clearTimers],
  );

  const reposition = useCallback(() => {
    const button = buttonRef.current;
    if (!button) return;
    const anchor = button.getBoundingClientRect();
    const panel = panelRef.current;
    const width = panel?.offsetWidth ?? 340;
    const height = panel?.offsetHeight ?? 320;
    const maxTop = window.innerHeight - VIEWPORT_MARGIN - height;

    let side: Placement["side"] = "bottom";
    let top = anchor.bottom + PANEL_GAP;
    if (top > maxTop) {
      const above = anchor.top - PANEL_GAP - height;
      if (above >= VIEWPORT_MARGIN) {
        top = above;
        side = "top";
      } else {
        top = Math.max(VIEWPORT_MARGIN, maxTop);
      }
    }

    // Right-align to the anchor, then keep the whole panel inside the viewport.
    const left = Math.max(
      VIEWPORT_MARGIN,
      Math.min(anchor.right - width, window.innerWidth - VIEWPORT_MARGIN - width),
    );

    setPlacement({ top, left, side });
  }, []);

  useLayoutEffect(() => {
    if (open) reposition();
    else setPlacement(null);
  }, [open, reposition]);

  useEffect(() => {
    if (!open) return;
    const handle = () => reposition();
    window.addEventListener("resize", handle);
    // Capture phase so the panel follows the scrollable explore column too.
    window.addEventListener("scroll", handle, true);
    return () => {
      window.removeEventListener("resize", handle);
      window.removeEventListener("scroll", handle, true);
    };
  }, [open, reposition]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      dismiss(true);
    };
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!target) return;
      if (buttonRef.current?.contains(target) || panelRef.current?.contains(target)) return;
      dismiss(false);
    };
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open, dismiss]);

  const sections: Array<{ heading: string; items: string[] }> = [
    { heading: "What you see", items: info.whatYouSee },
    { heading: "How it works", items: info.howItWorks },
    { heading: "Controls", items: info.controls ?? [] },
    { heading: "What to notice", items: info.notice ?? [] },
    { heading: "Where it breaks", items: info.limits },
  ].filter((section) => section.items.length > 0);

  return (
    <>
      <button
        type="button"
        ref={buttonRef}
        className="card-info-trigger"
        aria-label={`About this visualization: ${info.title}`}
        aria-describedby={open ? panelId : undefined}
        aria-expanded={open}
        data-pinned={pinned || undefined}
        onPointerEnter={(event) => {
          if (event.pointerType === "touch") return;
          scheduleOpen();
        }}
        onPointerLeave={(event) => {
          if (event.pointerType === "touch") return;
          scheduleClose();
        }}
        onFocus={() => {
          clearTimers();
          setOpen(true);
        }}
        onBlur={() => {
          if (pinned) return;
          clearTimers();
          setOpen(false);
        }}
        onClick={() => {
          clearTimers();
          if (pinned) {
            setPinned(false);
            setOpen(false);
            return;
          }
          setPinned(true);
          setOpen(true);
        }}
      >
        <Info aria-hidden="true" />
      </button>

      {open &&
        createPortal(
          <div
            ref={panelRef}
            id={panelId}
            role="tooltip"
            className="card-info-panel"
            data-side={placement?.side ?? "bottom"}
            data-pinned={pinned || undefined}
            style={{
              top: placement?.top ?? 0,
              left: placement?.left ?? 0,
              visibility: placement ? "visible" : "hidden",
            }}
            onPointerEnter={clearTimers}
            onPointerLeave={(event) => {
              if (event.pointerType === "touch") return;
              scheduleClose();
            }}
          >
            <div className="card-info-panel__head">
              <div>
                <span>{label}</span>
                <h4>{info.title}</h4>
              </div>
              {pinned && (
                <button
                  type="button"
                  className="card-info-panel__close"
                  aria-label="Close explanation"
                  onClick={() => dismiss(true)}
                >
                  <X aria-hidden="true" />
                </button>
              )}
            </div>

            <p className="card-info-panel__summary">{info.summary}</p>

            <div className="card-info-panel__body">
              {sections.map((section) => (
                <section key={section.heading}>
                  <strong>{section.heading}</strong>
                  <ul>
                    {section.items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>
              ))}
            </div>

            {!pinned && (
              <p className="card-info-panel__hint">Click the icon to keep this open</p>
            )}
          </div>,
          document.body,
        )}
    </>
  );
}
