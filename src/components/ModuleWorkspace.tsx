import {
  ArrowLeft,
  ArrowRight,
  BookmarkPlus,
  Check,
  CheckCircle2,
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
  Square,
  Trash2,
  Volume2,
  Waypoints,
  X,
} from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import type { LocalChatChunk } from "@app/lib/local-ai";
import type { ModuleState } from "@app/module-sdk";
import { CardInfoProvider, checkpointQuestions } from "@app/module-sdk";
import type { RegisteredModule } from "@app/modules/registry";
import { modules, modulesById } from "@app/modules/registry";
import { encodeState } from "@app/lib/math";
import { canUseSpeech, speakLesson, stopSpeaking, textFromLearnRoot } from "@app/lib/speech";
import { useAppStore, type Snapshot } from "@app/store/app-store";
import { checkpointTally } from "@app/store/checkpoint-progress";
import { Checkpoint } from "./Checkpoint";
import { PrerequisiteBar } from "./PrerequisiteBar";
import { GuideMarkdown } from "./GuideMarkdown";
import { IconButton } from "./IconButton";

function normalizeHeading(value: string) {
  return value.replace(/[’‘ʻʼ`']/g, "'").replace(/\s+/g, " ").trim().toLowerCase();
}

function LearnPanel({ module }: { module: RegisteredModule }) {
  const mode = useAppStore((state) => state.mode);
  const searchFocus = useAppStore((state) => state.searchFocus);
  const setSearchFocus = useAppStore((state) => state.setSearchFocus);
  const setView = useAppStore((state) => state.setView);
  const isLastModule = modules[modules.length - 1]?.id === module.id;
  const reducedMotion = useAppStore((state) => state.reducedMotion);
  const Content = module.content[mode];
  const [showGlossary, setShowGlossary] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const speechSupported = canUseSpeech();

  useEffect(() => () => stopSpeaking(), []);

  useEffect(() => {
    stopSpeaking();
    setSpeaking(false);
  }, [mode, module.id]);

  useEffect(() => {
    if (!searchFocus || !("moduleId" in searchFocus) || searchFocus.moduleId !== module.id) return;
    if (searchFocus.kind === "glossary") setShowGlossary(true);
    if (searchFocus.kind === "learn" || searchFocus.kind === "checkpoint") setShowGlossary(false);
  }, [module.id, searchFocus]);

  useEffect(() => {
    if (!searchFocus || !("moduleId" in searchFocus) || searchFocus.moduleId !== module.id) return;
    if (!["learn", "glossary", "checkpoint"].includes(searchFocus.kind)) return;
    const root = scrollRef.current;
    if (!root) return;

    const frame = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        let target: HTMLElement | null = null;
        if (searchFocus.kind === "glossary") {
          target = root.querySelector(`[data-glossary-term="${CSS.escape(searchFocus.term)}"]`);
        } else if (searchFocus.kind === "checkpoint") {
          target = root.querySelector("[data-search-target=checkpoint]");
        } else if (searchFocus.kind === "learn") {
          target =
            [...root.querySelectorAll<HTMLElement>("h2, h3")].find(
              (heading) => normalizeHeading(heading.textContent ?? "") === normalizeHeading(searchFocus.section),
            ) ?? null;
        }
        if (!target) return;
        target.dataset.searchHit = "true";
        target.scrollIntoView({
          behavior: reducedMotion ? "auto" : "smooth",
          block: "start",
          inline: "nearest",
        });
        window.setTimeout(() => {
          delete target.dataset.searchHit;
        }, 1600);
        setSearchFocus(null);
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [mode, module.id, reducedMotion, searchFocus, setSearchFocus, showGlossary]);

  return (
    <aside className="learn-panel" aria-label={`Learn about ${module.title}`}>
      <div className="learn-panel__header">
        <div>
          <span>Learn / {mode === "plain" ? "plain language" : "standard"}</span>
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
      <div className="learn-panel__scroll" ref={scrollRef}>
        <div className="objectives">
          <span>After this lab, you can</span>
          {module.objectives.map((objective) => <p key={objective}><CheckCircle2 />{objective}</p>)}
        </div>
        <article className="learn-copy">
          <Content />
        </article>
        <div data-search-target="checkpoint">
          <Checkpoint key={module.id} module={module} />
        </div>
        {module.references && module.references.length > 0 && (
          <section className="references" aria-labelledby={`references-${module.id}`}>
            <h2 id={`references-${module.id}`}>References</h2>
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
    </aside>
  );
}

function SnapshotPanel({
  module,
  currentState,
  onClose,
}: {
  module: RegisteredModule;
  currentState: Record<string, string | number | boolean | string[]>;
  onClose: () => void;
}) {
  const snapshots = useAppStore((state) => state.snapshots.filter((snapshot) => snapshot.moduleId === module.id));
  const saveSnapshot = useAppStore((state) => state.saveSnapshot);
  const loadSnapshot = useAppStore((state) => state.loadSnapshot);
  const deleteSnapshot = useAppStore((state) => state.deleteSnapshot);
  const [name, setName] = useState(`Experiment ${snapshots.length + 1}`);

  return (
    <div className="floating-panel snapshot-panel" role="dialog" aria-modal="true" aria-label="Snapshots">
      <div className="floating-panel__heading">
        <div><strong>Snapshots</strong><span>Return to any interesting state.</span></div>
        <IconButton label="Close snapshots" onClick={onClose}><X /></IconButton>
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          saveSnapshot(module.id, name.trim() || `Experiment ${snapshots.length + 1}`, currentState);
          setName(`Experiment ${snapshots.length + 2}`);
        }}
      >
        <input aria-label="Snapshot name" value={name} onChange={(event) => setName(event.target.value)} />
        <button type="submit"><BookmarkPlus /> Save state</button>
      </form>
      <div className="snapshot-list">
        {snapshots.length === 0 && <p>No saved states yet. Change a control, then capture it here.</p>}
        {snapshots.map((snapshot) => (
          <div key={snapshot.id}>
            <button type="button" onClick={() => { loadSnapshot(snapshot); onClose(); }}>
              <strong>{snapshot.name}</strong>
              <span>{new Date(snapshot.createdAt).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}</span>
            </button>
            <IconButton label={`Delete ${snapshot.name}`} onClick={() => deleteSnapshot(snapshot.id)}><Trash2 /></IconButton>
          </div>
        ))}
      </div>
    </div>
  );
}

interface GuideMessage {
  id: string;
  role: "guide" | "user";
  text: string;
  pending?: boolean;
}

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
  const [snapshotsOpen, setSnapshotsOpen] = useState(false);
  const [toast, setToast] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const exploreScrollRef = useRef<HTMLDivElement>(null);
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

  useEffect(() => {
    if (store.searchFocus?.kind !== "card" || store.searchFocus.moduleId !== module.id) return;
    const root = exploreScrollRef.current;
    if (!root) return;
    const frame = window.requestAnimationFrame(() => {
      const target = root.querySelector<HTMLElement>(
        `[data-lab-surface="${CSS.escape(store.searchFocus && store.searchFocus.kind === "card" ? store.searchFocus.label : "")}"]`,
      );
      if (!target) return;
      target.dataset.searchHit = "true";
      target.scrollIntoView({
        behavior: store.reducedMotion ? "auto" : "smooth",
        block: "center",
        inline: "nearest",
      });
      window.setTimeout(() => {
        delete target.dataset.searchHit;
      }, 1600);
      store.setSearchFocus(null);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [module.id, store.reducedMotion, store.searchFocus, store.setSearchFocus]);

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
        </section>

        {store.learnOpen && <LearnPanel module={module} />}
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
            store.resetModuleState(module.id, module.initialState);
            setToast("Interactive reset");
          }}><RotateCcw /></IconButton>
          <IconButton label="Save or load snapshot" tooltipSide="top" className={snapshotsOpen ? "is-active" : ""} onClick={() => setSnapshotsOpen(!snapshotsOpen)}><BookmarkPlus /></IconButton>
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

      {snapshotsOpen && <SnapshotPanel module={module} currentState={currentState} onClose={() => setSnapshotsOpen(false)} />}
      {toast && <ShareToast text={toast} />}
      <div className="sr-only" aria-live="polite">{announcement}</div>

      <div className="module-edge-nav" aria-label="Adjacent modules">
        {previousModule && <button type="button" onClick={() => store.openModule(previousModule.id)}><ArrowLeft /><span>Previous lab</span><strong>{previousModule.title}</strong></button>}
        {nextModule && <button type="button" onClick={() => store.openModule(nextModule.id)}><span>Next lab</span><strong>{nextModule.title}</strong><ArrowRight /></button>}
      </div>
    </main>
  );
}
