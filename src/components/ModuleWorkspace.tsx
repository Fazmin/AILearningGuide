import {
  ArrowLeft,
  ArrowRight,
  Bookmark,
  BookmarkCheck,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Eye,
  Maximize2,
  MessageCircle,
  PanelRightClose,
  PanelRightOpen,
  RotateCcw,
  Send,
  Sparkles,
  Waypoints,
  X,
} from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import type { LocalChatChunk } from "@app/lib/local-ai";
import type { LabScene, ModuleState } from "@app/module-sdk";
import { CardInfoProvider, checkpointQuestions } from "@app/module-sdk";
import type { RegisteredModule } from "@app/modules/registry";
import { modules } from "@app/modules/registry";
import { sceneState } from "@app/lib/lab-scenes";
import type { LearnEntry } from "@app/lib/learn-sections";
import { encodeState } from "@app/lib/math";
import { useAppStore } from "@app/store/app-store";
import { checkpointTally } from "@app/store/checkpoint-progress";
import { BookmarksPanel } from "./BookmarksPanel";
import { LabSceneBanner } from "./LabSceneBanner";
import { LearnPanel } from "./LearnPanel";
import { PrerequisiteBar } from "./PrerequisiteBar";
import { GuideMarkdown } from "./GuideMarkdown";
import { IconButton } from "./IconButton";

interface GuideMessage {
  id: string;
  role: "guide" | "user";
  text: string;
  pending?: boolean;
}

/** A lab scene on show, with the reader's own setup from before the first one. */
interface ShownScene {
  scene: LabScene;
  section: string;
  before: { state: ModuleState; step: number };
}

/** Below this width the Learn panel is laid over the lab instead of beside it (styles.css). */
const LEARN_OVERLAY_QUERY = "(max-width: 1050px)";

function builtInReply(module: RegisteredModule, message: string, currentStep: number) {
  const normalized = message.toLowerCase();
  if (normalized.includes("break") || normalized.includes("limit")) {
    return `A useful limit to test: ${module.content ? checkpointQuestions(module)[0].explanation : module.tagline} Open “Where it breaks” in the Learn panel, then push one control to an extreme.`;
  }
  if (normalized.includes("what") || normalized.includes("explain")) {
    return `${module.tagline} Right now you’re on “${module.steps[currentStep]}.” Change one control at a time and name what stayed fixed.`;
  }
  if (normalized.includes("try") || normalized.includes("next")) {
    return `Try this: change the most prominent slider to each extreme, then return it to the middle. Watch which output changes smoothly and which changes abruptly.`;
  }
  return `Connect your question to the live experiment: “${message}” relates to ${module.title}. Compare the current output before and after one small intervention, then use the checkpoint to test your explanation.`;
}

function GuidePanel({
  module,
  currentStep,
  currentState,
}: {
  module: RegisteredModule;
  currentStep: number;
  currentState: ModuleState;
}) {
  const setChatOpen = useAppStore((state) => state.setChatOpen);
  const provider = useAppStore((state) => state.provider);
  const providerModel = useAppStore((state) => state.providerModel);
  const providerBaseUrl = useAppStore((state) => state.providerBaseUrl);
  const maxReplyTokens = useAppStore((state) => state.maxReplyTokens);
  const systemPrompt = useAppStore((state) => state.systemPrompt);
  const localModelReady = useAppStore((state) => state.localModelReady);
  const desktop = Boolean(window.__TAURI_INTERNALS__);
  const mode = useAppStore((state) => state.mode);
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [messages, setMessages] = useState<GuideMessage[]>([
    {
      id: "welcome",
      role: "guide",
      text: `I’m scoped to ${module.title}. Ask what to notice, what to try, or where this idea breaks.`,
    },
  ]);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [messages]);

  useEffect(() => {
    if (!window.__TAURI_INTERNALS__) return;
    let disposed = false;
    let removeListener: (() => void) | undefined;
    void listen<LocalChatChunk>("local-chat-chunk", ({ payload }) => {
      setMessages((current) =>
        current.map((message) =>
          message.id === `${payload.requestId}-reply`
            ? {
                ...message,
                text: `${message.text}${payload.content}`,
                pending: !payload.done,
              }
            : message,
        ),
      );
    }).then((unlisten) => {
      if (disposed) unlisten();
      else removeListener = unlisten;
    });
    return () => {
      disposed = true;
      removeListener?.();
    };
  }, []);

  const persistExchange = (id: string, userMessage: string, reply: string, backend: string) => {
    if (!window.__TAURI_INTERNALS__) return;
    const sessionId = `guide-${module.id}`;
    void invoke("upsert_chat_session", {
      session: {
        id: sessionId,
        moduleId: module.id,
        backend,
        title: `${module.title} guide`,
      },
    })
      .then(() =>
        invoke("append_chat_message", {
          message: { id, sessionId, role: "user", content: userMessage },
        }),
      )
      .then(() =>
        invoke("append_chat_message", {
          message: { id: `${id}-reply`, sessionId, role: "assistant", content: reply },
        }),
      )
      .catch(() => undefined);
  };

  const finishReply = (id: string, userMessage: string, reply: string, backend: string, pending = false) => {
    setMessages((current) =>
      current.map((item) =>
        item.id === `${id}-reply` ? { ...item, text: reply, pending } : item,
      ),
    );
    if (!pending) persistExchange(id, userMessage, reply, backend);
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const message = input.trim();
    if (!message) return;
    const id = Date.now().toString();
    const history = messages
      .filter((item) => item.id !== "welcome" && !item.pending && item.text.trim())
      .slice(-8)
      .map((item) => ({
        role: item.role === "user" ? "user" : "assistant",
        content: item.text,
      }));
    const useCloud = provider !== "none" && desktop;
    const useLocalModel = localModelReady && provider === "none" && desktop;
    setMessages((current) => [
      ...current,
      { id, role: "user", text: message },
      { id: `${id}-reply`, role: "guide", text: useCloud || useLocalModel ? "" : builtInReply(module, message, currentStep), pending: useCloud || useLocalModel },
    ]);
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";

    if (useCloud) {
      void invoke<string>("cloud_chat", {
        request: {
          requestId: id,
          provider,
          model: providerModel || null,
          baseUrl: providerBaseUrl || null,
          moduleTitle: module.title,
          mode,
          objectives: module.objectives,
          currentStep: module.steps[currentStep],
          stateJson: JSON.stringify(currentState),
          message,
          history,
          maxReplyTokens,
          systemPrompt: systemPrompt || null,
        },
      })
        .then((reply) => finishReply(id, message, reply, provider))
        .catch((error) => {
          finishReply(id, message, `The cloud guide could not answer: ${String(error)}`, `${provider}-error`);
        });
      return;
    }

    if (useLocalModel) {
      void invoke<string>("local_chat", {
        request: {
          requestId: id,
          moduleTitle: module.title,
          mode,
          objectives: module.objectives,
          currentStep: module.steps[currentStep],
          stateJson: JSON.stringify(currentState),
          message,
        },
      })
        .then((reply) => finishReply(id, message, reply, "local"))
        .catch((error) => {
          finishReply(
            id,
            message,
            `${builtInReply(module, message, currentStep)} Local model error: ${String(error)}`,
            "built-in-fallback",
          );
        });
      return;
    }

    persistExchange(id, message, builtInReply(module, message, currentStep), "built-in");
  };

  return (
    <aside className="guide-panel" aria-label={`${module.title} study guide`}>
      <div className="guide-panel__header">
        <div><Sparkles /><span><strong>Topic guide</strong><small>{provider !== "none" ? `${provider}${providerModel ? ` · ${providerModel}` : ""}` : localModelReady ? "local model" : "built-in prompts"}</small></span></div>
        <IconButton label="Close topic guide" onClick={() => setChatOpen(false)}><X /></IconButton>
      </div>
      {provider !== "none" && !desktop && (
        <div className="guide-notice">
          Cloud chat runs in the desktop app. This browser preview still uses built-in prompts.
        </div>
      )}
      {provider === "none" && !localModelReady && (
        <div className="guide-notice">
          This preview uses built-in coaching prompts. Connect a model in Settings for open-ended answers.
        </div>
      )}
      <div className="guide-messages">
        {messages.map((message) => (
          <div className={`guide-message guide-message--${message.role}`} key={message.id}>
            <span>{message.role === "guide" ? "Guide" : "You"}</span>
            <div className="guide-message__body">
              {message.text ? (
                <GuideMarkdown text={message.text} />
              ) : message.pending ? (
                <i className="guide-typing"><b /><b /><b /></i>
              ) : null}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <div className="guide-dock">
        <div className="guide-suggestions">
          {["What should I notice?", "Give me something to try", "Where does this break?"].map((suggestion) => (
            <button type="button" key={suggestion} onClick={() => setInput(suggestion)}>{suggestion}</button>
          ))}
        </div>
        <form onSubmit={submit} className="guide-composer">
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            aria-label="Ask the topic guide"
            placeholder={`Ask about ${module.title.toLowerCase()}…`}
            onChange={(event) => {
              const field = event.currentTarget;
              setInput(field.value);
              field.style.height = "auto";
              field.style.height = `${Math.min(field.scrollHeight, 104)}px`;
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <button type="submit" aria-label="Send question" disabled={!input.trim() || messages.some((message) => message.pending)}>
            <Send />
          </button>
        </form>
      </div>
    </aside>
  );
}

function ShareToast({ text }: { text: string }) {
  return <div className="share-toast" role="status"><Check />{text}</div>;
}

export function ModuleWorkspace({ module }: { module: RegisteredModule }) {
  const store = useAppStore();
  const progress = store.progress[module.id];
  const currentStep = Math.min(progress?.currentStep ?? 0, module.steps.length - 1);
  const stepInstruction =
    module.stepInstructions[currentStep] ??
    `Use the interactive to ${module.steps[currentStep].toLowerCase()}, then compare what changed.`;
  const currentState = useMemo(
    () => ({ ...module.initialState, ...(store.moduleStates[module.id] ?? {}) }),
    [module, store.moduleStates],
  );
  const [bookmarksOpen, setBookmarksOpen] = useState(false);
  // The section the Learn panel is showing, so the bookmarks panel can offer to save it.
  const [learnSection, setLearnSection] = useState<LearnEntry | null>(null);
  const bookmarksButtonRef = useRef<HTMLButtonElement>(null);
  const hasBookmarks = store.bookmarks.some((bookmark) => bookmark.moduleId === module.id);
  const [toast, setToast] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const exploreScrollRef = useRef<HTMLDivElement>(null);
  const [shownScene, setShownScene] = useState<ShownScene | null>(null);
  const sceneBannerRef = useRef<HTMLDivElement>(null);
  const pendingLabFrame = useRef(0);
  const moduleIndex = modules.findIndex((item) => item.id === module.id);
  const checkpointTotal = checkpointQuestions(module).length;
  const checkpointTallyNow = checkpointTally(progress?.checkpointResults, checkpointTotal);
  const checkpointBadge =
    checkpointTallyNow.correct > 0
      ? {
          complete: checkpointTallyNow.correct === checkpointTotal,
          label: checkpointTotal > 1 ? `${checkpointTallyNow.correct}/${checkpointTotal}` : "",
        }
      : null;
  const previousModule = modules[moduleIndex - 1];
  const nextModule = modules[moduleIndex + 1];

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 2200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const root = exploreScrollRef.current;
      if (!root) return;
      const target = Array.from(root.querySelectorAll<HTMLElement>("*")).find(
        (element) =>
          window
            .getComputedStyle(element)
            .getPropertyValue("--lesson-step-target")
            .trim() === "1",
      );
      if (!target) return;
      const rootRect = root.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      const isOutsideView =
        targetRect.top < rootRect.top + 12 ||
        targetRect.bottom > rootRect.bottom - 12;
      if (isOutsideView) {
        target.scrollIntoView({
          behavior: store.reducedMotion ? "auto" : "smooth",
          block: "center",
          inline: "nearest",
        });
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [currentStep, module.id, store.reducedMotion]);

  /** Scroll a lab card into view and flash it; false when the lab is not showing that card. */
  const revealSurface = useCallback((label: string, block: ScrollLogicalPosition) => {
    const target = exploreScrollRef.current?.querySelector<HTMLElement>(`[data-lab-surface="${CSS.escape(label)}"]`);
    if (!target) return false;
    target.dataset.searchHit = "true";
    target.scrollIntoView({
      behavior: useAppStore.getState().reducedMotion ? "auto" : "smooth",
      block,
      inline: "nearest",
    });
    window.setTimeout(() => {
      delete target.dataset.searchHit;
    }, 1600);
    return true;
  }, []);

  useEffect(() => {
    if (store.searchFocus?.kind !== "card" || store.searchFocus.moduleId !== module.id) return;
    const label = store.searchFocus.label;
    const frame = window.requestAnimationFrame(() => {
      if (revealSurface(label, "center")) store.setSearchFocus(null);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [module.id, revealSurface, store.searchFocus, store.setSearchFocus]);

  /**
   * Move the lab to a step and a state. Labs that follow the step reset their own stage when it changes
   * (attention, backpropagation), so the state goes in a frame after that has happened, and the card is
   * revealed a frame after the lab has drawn it.
   */
  const applyLab = useCallback(
    (step: number | undefined, state: ModuleState, focus?: string) => {
      const { setCurrentStep, setModuleState } = useAppStore.getState();
      window.cancelAnimationFrame(pendingLabFrame.current);
      if (step !== undefined) setCurrentStep(module.id, Math.min(Math.max(0, step), module.steps.length - 1));
      pendingLabFrame.current = window.requestAnimationFrame(() => {
        pendingLabFrame.current = window.requestAnimationFrame(() => {
          setModuleState(module.id, state);
          if (!focus) return;
          pendingLabFrame.current = window.requestAnimationFrame(() => revealSurface(focus, "start"));
        });
      });
    },
    [module.id, module.steps.length, revealSurface],
  );

  useEffect(() => () => window.cancelAnimationFrame(pendingLabFrame.current), []);

  const showScene = useCallback(
    (scene: LabScene, section: string) => {
      const app = useAppStore.getState();
      const before = {
        state: { ...module.initialState, ...(app.moduleStates[module.id] ?? {}) },
        step: Math.min(app.progress[module.id]?.currentStep ?? 0, module.steps.length - 1),
      };
      // Keep the setup from before the first scene, so a chain of scenes still leads back to it.
      setShownScene((shown) => ({ scene, section, before: shown?.before ?? before }));
      // The lab has to be visible: leave the wide reading view, and over a narrow window step aside.
      const overlaid = window.matchMedia(LEARN_OVERLAY_QUERY).matches;
      if (overlaid) app.setLearnOpen(false);
      else if (app.learnExpanded) app.setLearnExpanded(false);
      applyLab(scene.step, sceneState(module.initialState, scene), scene.focus);
      if (overlaid) window.requestAnimationFrame(() => sceneBannerRef.current?.focus());
    },
    [applyLab, module],
  );

  const restoreLab = () => {
    if (shownScene) applyLab(shownScene.before.step, shownScene.before.state);
    setShownScene(null);
  };

  const returnToReading = () => {
    if (!shownScene) return;
    store.setSearchFocus({ kind: "learn", moduleId: module.id, mode: store.mode, section: shownScene.section });
    store.setLearnOpen(true);
  };

  const share = async () => {
    const encoded = encodeState({
      module: module.slug,
      version: module.stateVersion,
      state: currentState,
    });
    const url = `${window.location.origin}${window.location.pathname}#/${module.slug}?state=${encoded}`;
    try {
      await navigator.clipboard.writeText(url);
      setToast("State link copied");
    } catch {
      setToast("Copy unavailable in this window");
    }
  };

  return (
    <main className="module-workspace" style={{ "--module-accent": module.accent } as React.CSSProperties}>
      <header className="module-header">
        <div className="module-breadcrumb">
          <button type="button" onClick={() => store.setView("home")}><ArrowLeft /> Learning map</button>
          <span>/</span>
          <span>{String(module.order).padStart(2, "0")}</span>
          <button type="button" className="breadcrumb-connections" onClick={() => store.openConnections(module.id)}>
            <Waypoints /> Where this fits
          </button>
        </div>
        <IconButton
          label={!store.learnOpen ? "Show Learn panel" : store.learnExpanded ? "Hide Learn panel" : "Expand Learn panel"}
          tooltipSide="left"
          className="learn-toggle"
          onClick={store.cycleLearnPanel}
        >
          {!store.learnOpen ? <PanelRightOpen /> : store.learnExpanded ? <PanelRightClose /> : <Maximize2 />}
        </IconButton>
        <div className="module-title">
          <div>
            <span>Lab {String(module.order).padStart(2, "0")}</span>
            <h1>{module.title}</h1>
            <p>{module.tagline}</p>
          </div>
          <div className="module-meta">
            <span>{module.estimatedMinutes} min</span>
            <span>{module.steps.length} steps</span>
            {checkpointBadge && <span className={checkpointBadge.complete ? "is-passed" : ""}>{checkpointBadge.complete && <Check />} checkpoint {checkpointBadge.label}</span>}
          </div>
        </div>
        <PrerequisiteBar module={module} />
      </header>

      <div className={`workspace-body ${store.learnOpen ? (store.learnExpanded ? "learn-is-expanded" : "") : "learn-is-closed"} ${store.chatOpen ? "chat-is-open" : ""}`}>
        <section className="explore-region" aria-label={`Explore ${module.title}`}>
          <div className="region-label">
            <span>Step {currentStep + 1} of {module.steps.length}</span>
            <strong>{currentStep + 1}. {module.steps[currentStep]}</strong>
          </div>
          <div className="step-instruction" key={`${module.id}-${currentStep}`} aria-live="polite">
            <span>Try this</span>
            <p>{stepInstruction}</p>
          </div>
          <div
            ref={exploreScrollRef}
            className="explore-scroll"
            data-module={module.slug}
            data-current-step={currentStep}
          >
            <CardInfoProvider
              value={module.cardInfo}
              focusLabel={
                store.searchFocus?.kind === "card" && store.searchFocus.moduleId === module.id
                  ? store.searchFocus.label
                  : null
              }
            >
              <module.Explore
                state={currentState}
                setState={(patch) => store.setModuleState(module.id, patch)}
                currentStep={currentStep}
                mode={store.mode}
                narrate={(message) => {
                  if (store.verboseNarration) setAnnouncement(message);
                }}
              />
            </CardInfoProvider>
          </div>
          {shownScene && (
            <LabSceneBanner
              ref={sceneBannerRef}
              scene={shownScene.scene}
              section={shownScene.section}
              learnClosed={!store.learnOpen}
              onRestore={restoreLab}
              onReturnToReading={returnToReading}
              onDismiss={() => setShownScene(null)}
            />
          )}
        </section>

        {store.learnOpen && (
          <LearnPanel
            module={module}
            onSectionChange={setLearnSection}
            activeSceneId={shownScene?.scene.id ?? null}
            onShowScene={showScene}
          />
        )}
        {store.chatOpen && <GuidePanel module={module} currentStep={currentStep} currentState={currentState} />}
      </div>

      <footer className="module-dock">
        <div className="dock-steps">
          <IconButton
            label="Previous step"
            disabled={currentStep === 0}
            tooltipSide="top"
            onClick={() => store.setCurrentStep(module.id, currentStep - 1)}
          ><ChevronLeft /></IconButton>
          <div className="step-dots" aria-label={`Step ${currentStep + 1} of ${module.steps.length}`}>
            {module.steps.map((step, index) => (
              <button
                type="button"
                key={step}
                aria-label={`Go to step ${index + 1}: ${step}`}
                aria-current={index === currentStep ? "step" : undefined}
                className={index === currentStep ? "is-current" : index < currentStep ? "is-done" : ""}
                onClick={() => store.setCurrentStep(module.id, index)}
              ><i /><span>{step}</span></button>
            ))}
          </div>
          <IconButton
            label="Next step"
            disabled={currentStep === module.steps.length - 1}
            tooltipSide="top"
            onClick={() => store.setCurrentStep(module.id, currentStep + 1)}
          ><ChevronRight /></IconButton>
        </div>
        <div className="dock-actions">
          <IconButton label="Reset interactive" tooltipSide="top" onClick={() => {
            window.cancelAnimationFrame(pendingLabFrame.current);
            setShownScene(null);
            store.resetModuleState(module.id, module.initialState);
            setToast("Interactive reset");
          }}><RotateCcw /></IconButton>
          <IconButton
            ref={bookmarksButtonRef}
            label="Bookmarks"
            tooltipSide="top"
            className={bookmarksOpen ? "is-active" : ""}
            aria-expanded={bookmarksOpen}
            onClick={() => setBookmarksOpen(!bookmarksOpen)}
          >
            {hasBookmarks ? <BookmarkCheck /> : <Bookmark />}
          </IconButton>
          <IconButton label="Copy a link to this state" tooltipSide="top" onClick={() => void share()}><Copy /></IconButton>
          <button type="button" className="open-guide" onClick={() => store.setChatOpen(!store.chatOpen)}>
            <MessageCircle /><span>{store.chatOpen ? "Close guide" : "Ask the guide"}</span>
          </button>
          <button
            type="button"
            className={`complete-action ${progress?.completed ? "is-complete" : ""}`}
            onClick={() => store.toggleModuleComplete(module.id)}
          >
            {progress?.completed ? <Check /> : <Eye />}
            <span>{progress?.completed ? "Completed" : "Mark complete"}</span>
          </button>
        </div>
      </footer>

      {bookmarksOpen && (
        <BookmarksPanel
          module={module}
          section={store.learnOpen ? learnSection : null}
          onClose={() => {
            setBookmarksOpen(false);
            bookmarksButtonRef.current?.focus();
          }}
        />
      )}
      {toast && <ShareToast text={toast} />}
      <div className="sr-only" aria-live="polite">{announcement}</div>

      <div className="module-edge-nav" aria-label="Adjacent modules">
        {previousModule && <button type="button" onClick={() => store.openModule(previousModule.id)}><ArrowLeft /><span>Previous lab</span><strong>{previousModule.title}</strong></button>}
        {nextModule && <button type="button" onClick={() => store.openModule(nextModule.id)}><span>Next lab</span><strong>{nextModule.title}</strong><ArrowRight /></button>}
      </div>
    </main>
  );
}
