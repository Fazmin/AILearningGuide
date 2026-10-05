import {
  ArrowLeft,
  ArrowRight,
  Check,
  Cloud,
  Eye,
  Laptop,
  MessageSquareText,
  MousePointer2,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { localAiAvailable } from "@app/lib/local-ai";
import { ONBOARDING_SAMPLE } from "@app/lib/onboarding-sample";
import { fontScaleForStep, useAppStore } from "@app/store/app-store";
import type { ExplanationMode } from "@app/module-sdk";
import { shouldShowOnboarding } from "@app/store/onboarding";
import { palettes, Switch } from "./AccessibilityPopover";

type GuideDestination = "local" | "cloud";

const steps = [
  { id: "modes", title: "Choose how explanations read" },
  { id: "guide", title: "An optional guide, if you want one" },
  { id: "access", title: "Make it comfortable to read" },
] as const;

const lastStep = steps.length - 1;

const destinationLabel: Record<GuideDestination, string> = {
  local: "Local model",
  cloud: "Cloud providers",
};

const modeCopy: Record<ExplanationMode, { label: string; lead: string }> = {
  plain: {
    label: "Plain",
    lead: "Written at a grade 8 to 10 reading level. Starts with the big picture, then everyday words, examples and short stories, a concrete nudge to try, and misconceptions corrected in place.",
  },
  standard: {
    label: "Standard",
    lead: "Uses the technical vocabulary. Compact definitions, mechanisms, limitations, and links between ideas.",
  },
};

/** The sample text carries the lesson's own **bold**, `code`, and *italic* marks. */
function renderInline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
    if (part.startsWith("*")) return <em key={index}>{part.slice(1, -1)}</em>;
    return part;
  });
}

const focusableSelector =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function focusableIn(container: HTMLElement) {
  return Array.from(container.querySelectorAll<HTMLElement>(focusableSelector)).filter(
    (element) => !element.hasAttribute("hidden") && element.getAttribute("aria-hidden") !== "true",
  );
}

/**
 * The first-run welcome tour: the two explanation modes, the optional guide, and accessibility.
 * It reads and writes the same store state as Settings and the accessibility menu, so nothing here
 * keeps a copy of a setting. It shows for a learner with no history who has not finished it, and on
 * demand from Settings > About.
 */
export function Onboarding() {
  const visible = useAppStore(shouldShowOnboarding);
  return visible ? <OnboardingDialog /> : null;
}

function OnboardingDialog() {
  const store = useAppStore();
  const [step, setStep] = useState(0);
  const [destination, setDestination] = useState<GuideDestination | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const titleId = useId();
  const desktop = localAiAvailable();

  // The tour sits above the app, so keep the search palette and accessibility menu shut underneath it.
  const searchOpen = useAppStore((state) => state.searchOpen);
  const accessibilityOpen = useAppStore((state) => state.accessibilityOpen);
  useEffect(() => {
    const { setSearchOpen, setAccessibilityOpen } = useAppStore.getState();
    if (searchOpen) setSearchOpen(false);
    if (accessibilityOpen) setAccessibilityOpen(false);
  }, [searchOpen, accessibilityOpen]);

  // Open: hide everything behind the dialog from the tab order and assistive technology (the title
  // bar stays live so the window can still be moved and closed), then move focus in.
  // Close: undo that and return focus to where it was.
  useEffect(() => {
    const overlay = overlayRef.current;
    const dialog = dialogRef.current;
    if (!overlay || !dialog) return undefined;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const hidden: Element[] = [];
    for (const sibling of Array.from(overlay.parentElement?.children ?? [])) {
      if (sibling === overlay || sibling.classList.contains("titlebar") || sibling.hasAttribute("inert")) continue;
      sibling.setAttribute("inert", "");
      hidden.push(sibling);
    }
    const keepFocusInside = (event: FocusEvent) => {
      if (event.target instanceof Node && !dialog.contains(event.target)) headingRef.current?.focus();
    };
    document.addEventListener("focusin", keepFocusInside);
    headingRef.current?.focus();
    return () => {
      document.removeEventListener("focusin", keepFocusInside);
      hidden.forEach((element) => element.removeAttribute("inert"));
      if (previous && previous !== document.body && previous.isConnected) previous.focus();
    };
  }, []);

  // A new step starts at its heading, so a screen reader announces it and the text starts at the top.
  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
    headingRef.current?.focus();
  }, [step]);

  const finish = (target?: GuideDestination | "home") => {
    store.completeOnboarding();
    if (target === "home") {
      store.setView("home");
    } else if (target) {
      store.setSettingsTab(target);
      store.setView("settings");
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      finish();
      return;
    }
    // The title bar's search shortcut would open a palette behind this dialog.
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (event.key !== "Tab" || !dialogRef.current) return;
    const focusable = focusableIn(dialogRef.current);
    if (!focusable.length) {
      event.preventDefault();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (event.shiftKey && (active === first || active === headingRef.current || active === dialogRef.current)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const current = steps[step];

  return (
    <div className="onboarding-overlay" ref={overlayRef}>
      <div
        className="onboarding"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        ref={dialogRef}
        onKeyDown={onKeyDown}
      >
        <header className="onboarding__header">
          <p className="onboarding__brand">Welcome to Discover AI: A Learning Guide</p>
          <ol className="onboarding__progress" aria-label="Tour progress">
            {steps.map((item, index) => (
              <li
                key={item.id}
                className={index <= step ? "is-reached" : ""}
                aria-current={index === step ? "step" : undefined}
              >
                <span className="sr-only">
                  {item.title}
                  {index < step ? ", done" : ""}
                </span>
              </li>
            ))}
          </ol>
        </header>

        <div className="onboarding__body" ref={bodyRef}>
          <h2 id={titleId} className="onboarding__title" tabIndex={-1} ref={headingRef}>
            <span className="onboarding__step">
              Step {step + 1} of {steps.length}
            </span>{" "}
            {current.title}
          </h2>

          {step === 0 && <ModesStep mode={store.mode} onChoose={store.setMode} />}
          {step === 1 && (
            <GuideStep
              desktop={desktop}
              destination={destination}
              onToggle={(value) => setDestination((existing) => (existing === value ? null : value))}
              onSkip={() => {
                setDestination(null);
                setStep(2);
              }}
            />
          )}
          {step === 2 && <AccessibilityStep />}
        </div>

        <footer className="onboarding__footer">
          {step < lastStep && (
            <button type="button" className="onboarding__skip" onClick={() => finish()}>
              Skip tour
            </button>
          )}
          <div className="onboarding__nav">
            {step > 0 && (
              <button type="button" className="settings-action settings-action--secondary" onClick={() => setStep(step - 1)}>
                <ArrowLeft /> Back
              </button>
            )}
            {step < lastStep ? (
              <button type="button" className="settings-action" onClick={() => setStep(step + 1)}>
                Next <ArrowRight />
              </button>
            ) : (
              <>
                {destination && (
                  <button type="button" className="settings-action settings-action--secondary" onClick={() => finish(destination)}>
                    Open {destinationLabel[destination]} settings
                  </button>
                )}
                <button type="button" className="settings-action" onClick={() => finish("home")}>
                  Start learning <ArrowRight />
                </button>
              </>
            )}
          </div>
        </footer>
      </div>
    </div>
  );
}

function ModesStep({ mode, onChoose }: { mode: ExplanationMode; onChoose: (mode: ExplanationMode) => void }) {
  return (
    <>
      <p className="onboarding__lead">
        Every lab explains each idea two ways. Below is one idea written both ways. Pick the one you would rather
        read. You can switch at any time, and your experiments stay as they are.
      </p>
      <p className="onboarding__source">
        Both samples are from the lab {ONBOARDING_SAMPLE.moduleTitle}, section “{ONBOARDING_SAMPLE.section}”.
      </p>
      <div className="onboarding-modes" role="group" aria-label="Choose an explanation mode">
        {(["plain", "standard"] as const).map((value) => {
          const selected = mode === value;
          return (
            <div key={value} className={`onboarding-mode${selected ? " is-selected" : ""}`}>
              <div className="onboarding-mode__top">
                <h3>{modeCopy[value].label}</h3>
                <button type="button" className="onboarding-mode__choose" aria-pressed={selected} onClick={() => onChoose(value)}>
                  {selected && <Check />}
                  Use {modeCopy[value].label}
                </button>
              </div>
              <p className="onboarding-mode__lead">{modeCopy[value].lead}</p>
              <p className="onboarding-mode__sample">{renderInline(ONBOARDING_SAMPLE[value])}</p>
            </div>
          );
        })}
      </div>
      <p className="privacy-note">
        <ShieldCheck /> Both modes cover the same learning objectives.
      </p>
    </>
  );
}

function GuideStep({
  desktop,
  destination,
  onToggle,
  onSkip,
}: {
  desktop: boolean;
  destination: GuideDestination | null;
  onToggle: (destination: GuideDestination) => void;
  onSkip: () => void;
}) {
  return (
    <>
      <p className="onboarding__lead">
        Each lab has a topic guide: a chat that can answer questions about the lab you are in. It is optional. The labs,
        both explanation modes, checkpoints, and snapshots all work without it, and you can set it up at any time in
        Settings, under Local model or Cloud providers.
      </p>
      <div className="onboarding-options">
        <div className="onboarding-option">
          <h3><Laptop /> Private, on this computer</h3>
          <p>
            Downloads a small language model (Qwen3.5 2B, about 1.3 GB) and the llama.cpp engine once. After that it
            runs offline, and what you type stays on this computer.
          </p>
          {!desktop && <p className="onboarding-option__note">Local setup is available in the desktop app.</p>}
        </div>
        <div className="onboarding-option">
          <h3><Cloud /> A cloud provider</h3>
          <p>
            Connect OpenAI, Anthropic, Google, or a compatible service with your own API key. Your questions go to that
            provider. The key is kept in your operating system’s keychain, never in the app’s database or in exports.
          </p>
        </div>
      </div>
      <div className="onboarding-setup" role="group" aria-label="Set up the guide">
        <button type="button" aria-pressed={destination === "local"} onClick={() => onToggle("local")}>
          {destination === "local" && <Check />}
          Take me to Local model settings when I finish
        </button>
        <button type="button" aria-pressed={destination === "cloud"} onClick={() => onToggle("cloud")}>
          {destination === "cloud" && <Check />}
          Take me to Cloud provider settings when I finish
        </button>
        <button type="button" className="onboarding-setup__skip" onClick={onSkip}>
          Skip, set up later
        </button>
      </div>
    </>
  );
}

function AccessibilityStep() {
  const store = useAppStore();
  return (
    <>
      <p className="onboarding__lead">
        These apply straight away, and you can change them whenever you like from the accessibility button in the title
        bar or in Settings.
      </p>
      <div className="onboarding-a11y">
        <div className="onboarding-textsize">
          <div>
            <strong id="onboarding-text-size">Text size</strong>
            <span>Seven steps from 87.5% to 200%. Layouts reflow at every step.</span>
          </div>
          <div className="text-size-setting">
            <button
              type="button"
              aria-label="Decrease text size"
              disabled={store.fontStep === 0}
              onClick={() => store.setFontStep(store.fontStep - 1)}
            >
              A
            </button>
            <input
              type="range"
              min={0}
              max={6}
              step={1}
              value={store.fontStep}
              aria-labelledby="onboarding-text-size"
              onChange={(event) => store.setFontStep(Number(event.target.value))}
            />
            <button
              type="button"
              aria-label="Increase text size"
              disabled={store.fontStep === 6}
              onClick={() => store.setFontStep(store.fontStep + 1)}
            >
              A
            </button>
            <output aria-live="polite">{Math.round(fontScaleForStep(store.fontStep) * 100)}%</output>
          </div>
        </div>
        <Switch
          checked={store.highContrast}
          label="Higher contrast"
          description="Stronger edges and text separation"
          icon={<Eye />}
          onChange={store.setHighContrast}
        />
        <Switch
          checked={store.reducedMotion}
          label="Reduce motion"
          description="Remove nonessential animation"
          icon={<MousePointer2 />}
          onChange={store.setReducedMotion}
        />
        <Switch
          checked={store.verboseNarration}
          label="Verbose narration"
          description="Announce interactive changes"
          icon={<MessageSquareText />}
          onChange={store.setVerboseNarration}
        />
        <fieldset className="palette-picker">
          <legend>Colour-vision palette</legend>
          <div>
            {palettes.map((palette) => (
              <button
                type="button"
                key={palette.value}
                aria-pressed={store.colorVision === palette.value}
                onClick={() => store.setColorVision(palette.value)}
              >
                <i className={`palette-swatch palette-swatch--${palette.value}`} />
                {palette.label}
              </button>
            ))}
          </div>
        </fieldset>
      </div>
    </>
  );
}
