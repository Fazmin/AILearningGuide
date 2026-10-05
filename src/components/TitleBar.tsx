import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  Accessibility,
  Minus,
  Plus,
  Search,
  X,
  Maximize2,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { IconButton } from "./IconButton";
import { useAppStore } from "@app/store/app-store";

async function withWindow(action: (window: ReturnType<typeof getCurrentWindow>) => Promise<void>) {
  if (!window.__TAURI_INTERNALS__) return;
  await action(getCurrentWindow());
}

export function TitleBar() {
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  const isTauri = Boolean(window.__TAURI_INTERNALS__);
  const useNativeMacChrome = isMac && isTauri;
  const {
    mode,
    setMode,
    fontStep,
    setFontStep,
    sidebarCollapsed,
    toggleSidebar,
    accessibilityOpen,
    setAccessibilityOpen,
    setSearchOpen,
  } = useAppStore();
  const searchShortcut = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)
    ? "Search the guide, Command K"
    : "Search the guide, Control K";

  return (
    <header
      className="titlebar"
      data-mac-overlay={useNativeMacChrome || undefined}
      data-tauri-drag-region
      onDoubleClick={(event) => {
        if ((event.target as HTMLElement).closest("button")) return;
        void withWindow((appWindow) => appWindow.toggleMaximize());
      }}
      onMouseDown={(event) => {
        if (event.button !== 0 || (event.target as HTMLElement).closest("button, input")) return;
        void withWindow((appWindow) => appWindow.startDragging());
      }}
    >
      <div className="titlebar__left" data-tauri-drag-region>
        {isMac && !useNativeMacChrome && (
          <div className="mac-controls">
            <button type="button" aria-label="Close window" onClick={() => void withWindow((appWindow) => appWindow.close())}><i /></button>
            <button type="button" aria-label="Minimize window" onClick={() => void withWindow((appWindow) => appWindow.minimize())}><i /></button>
            <button type="button" aria-label="Maximize window" onClick={() => void withWindow((appWindow) => appWindow.toggleMaximize())}><i /></button>
          </div>
        )}
        <IconButton
          label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          onClick={toggleSidebar}
        >
          {sidebarCollapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
        </IconButton>
        <div className="wordmark" data-tauri-drag-region>
          <span className="wordmark__mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <strong>Discover AI</strong>
        </div>
      </div>

      <div className="mode-toggle" aria-label="Explanation mode">
        <button
          type="button"
          aria-pressed={mode === "plain"}
          onClick={() => setMode("plain")}
        >
          Plain
        </button>
        <button
          type="button"
          aria-pressed={mode === "standard"}
          onClick={() => setMode("standard")}
        >
          Standard
        </button>
      </div>

      <div className="titlebar__right">
        <button
          type="button"
          className="titlebar-search"
          aria-label={searchShortcut}
          onClick={() => setSearchOpen(true)}
        >
          <Search />
          <span>Search</span>
          <kbd>{/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "⌘K" : "Ctrl K"}</kbd>
        </button>
        <div className="font-stepper">
          <IconButton
            label="Decrease text size"
            disabled={fontStep === 0}
            onClick={() => setFontStep(fontStep - 1)}
          >
            <Minus />
          </IconButton>
          <span aria-live="polite">{Math.round([87.5, 100, 112.5, 125, 150, 175, 200][fontStep])}%</span>
          <IconButton
            label="Increase text size"
            disabled={fontStep === 6}
            onClick={() => setFontStep(fontStep + 1)}
          >
            <Plus />
          </IconButton>
        </div>
        <IconButton
          label="Accessibility options"
          aria-expanded={accessibilityOpen}
          className={accessibilityOpen ? "is-active" : ""}
          onClick={() => setAccessibilityOpen(!accessibilityOpen)}
        >
          <Accessibility />
        </IconButton>
        {!isMac && (
          <div className="window-controls">
            <IconButton label="Minimize window" onClick={() => void withWindow((appWindow) => appWindow.minimize())}>
              <Minus />
            </IconButton>
            <IconButton label="Maximize window" onClick={() => void withWindow((appWindow) => appWindow.toggleMaximize())}>
              <Maximize2 />
            </IconButton>
            <IconButton label="Close window" className="window-close" onClick={() => void withWindow((appWindow) => appWindow.close())}>
              <X />
            </IconButton>
          </div>
        )}
      </div>
    </header>
  );
}
