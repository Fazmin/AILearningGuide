import { ChevronDown, Home, Moon, Search, Settings, Sun, Waypoints } from "lucide-react";
import { useState } from "react";
import { fallbackModuleIcon, moduleIcons } from "./module-icons";
import { moduleGroups } from "@app/modules/groups";
import { modulesInGroup } from "@app/modules/registry";
import { useAppStore } from "@app/store/app-store";

function ProgressRing({ current, total }: { current: number; total: number }) {
  const percent = total ? current / total : 0;
  const circumference = 2 * Math.PI * 8;
  return (
    <span className="progress-ring" aria-label={`${current} of ${total} complete`}>
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="10" cy="10" r="8" />
        <circle
          cx="10"
          cy="10"
          r="8"
          style={{ strokeDasharray: circumference, strokeDashoffset: circumference * (1 - percent) }}
        />
      </svg>
      {current === total && total > 0 ? <span>✓</span> : null}
    </span>
  );
}

export function Sidebar() {
  const {
    sidebarCollapsed,
    view,
    activeModuleId,
    progress,
    setView,
    openModule,
    openConnections,
    setSearchOpen,
    theme,
    setTheme,
    provider,
    localModelReady,
  } = useAppStore();
  const [closedGroups, setClosedGroups] = useState<Record<string, boolean>>({});
  const searchShortcut = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "⌘K" : "Ctrl K";

  const modelReady = localModelReady || provider !== "none";
  const nextTheme = theme === "dark" ? "light" : "dark";

  return (
    <aside className={`sidebar ${sidebarCollapsed ? "sidebar--collapsed" : ""}`}>
      <nav className="sidebar__nav" aria-label="Learning modules">
        <button
          type="button"
          className={`sidebar-home ${view === "home" ? "is-active" : ""}`}
          aria-label="Learning map"
          data-tooltip={sidebarCollapsed ? "Learning map" : undefined}
          data-tooltip-side="right"
          onClick={() => setView("home")}
        >
          <Home />
          {!sidebarCollapsed && <span>Learning map</span>}
        </button>

        <button
          type="button"
          className={`sidebar-home ${view === "connections" ? "is-active" : ""}`}
          aria-label="Connections"
          data-tooltip={sidebarCollapsed ? "Connections" : undefined}
          data-tooltip-side="right"
          onClick={() => openConnections()}
        >
          <Waypoints />
          {!sidebarCollapsed && <span>Connections</span>}
        </button>

        <button
          type="button"
          className="module-search module-search--launch"
          aria-label="Search the guide"
          data-tooltip={sidebarCollapsed ? "Search the guide" : undefined}
          data-tooltip-side="right"
          onClick={() => setSearchOpen(true)}
        >
          <Search />
          {!sidebarCollapsed && <span>Search the guide</span>}
          {!sidebarCollapsed && <kbd>{searchShortcut}</kbd>}
        </button>

        <div className="module-groups">
          {moduleGroups.map((group) => {
            const groupModules = modulesInGroup(group.id);
            const complete = groupModules.filter((module) => progress[module.id]?.completed).length;
            const isClosed = closedGroups[group.id];

            return (
              <section className="module-group" key={group.id}>
                <button
                  type="button"
                  className="module-group__heading"
                  aria-expanded={!isClosed}
                  onClick={() => setClosedGroups((value) => ({ ...value, [group.id]: !value[group.id] }))}
                >
                  <ProgressRing current={complete} total={groupModules.length} />
                  {!sidebarCollapsed && (
                    <>
                      <span><strong>{group.title}</strong><small>{complete}/{groupModules.length} complete</small></span>
                      <ChevronDown className={isClosed ? "is-closed" : ""} />
                    </>
                  )}
                </button>
                {!isClosed && (
                  <div className="module-list">
                    {groupModules.map((module) => {
                      const ModuleIcon = moduleIcons[module.icon] ?? fallbackModuleIcon;
                      const isActive = view === "module" && module.id === activeModuleId;
                      return (
                        <button
                          type="button"
                          key={module.id}
                          aria-current={isActive ? "page" : undefined}
                          className={`module-link ${isActive ? "is-active" : ""} ${progress[module.id]?.completed ? "is-complete" : ""}`}
                          style={{ "--module-accent": module.accent } as React.CSSProperties}
                          onClick={() => openModule(module.id)}
                          data-tooltip={sidebarCollapsed ? module.title : undefined}
                          data-tooltip-side="right"
                        >
                          <span className="module-number">{String(module.order).padStart(2, "0")}</span>
                          <ModuleIcon />
                          {!sidebarCollapsed && <span>{module.title}</span>}
                          {!sidebarCollapsed && progress[module.id]?.completed && <i>✓</i>}
                        </button>
                      );
                    })}
                  </div>
                )}
              </section>
            );
          })}
        </div>
      </nav>

      <div className="sidebar__footer">
        <button
          type="button"
          className="footer-action"
          data-tooltip={sidebarCollapsed ? `${nextTheme === "dark" ? "Dark" : "Light"} theme` : undefined}
          data-tooltip-side="right"
          onClick={() => setTheme(nextTheme)}
        >
          {theme === "dark" ? <Sun /> : <Moon />}
          {!sidebarCollapsed && <span>{theme === "dark" ? "Light theme" : "Dark theme"}</span>}
        </button>
        <button
          type="button"
          className={`footer-action ${view === "settings" ? "is-active" : ""}`}
          data-tooltip={sidebarCollapsed ? "Settings" : undefined}
          data-tooltip-side="right"
          onClick={() => setView("settings")}
        >
          <Settings />
          {!sidebarCollapsed && <span>Settings</span>}
        </button>
        <div className="model-status" title={modelReady ? "AI guide ready" : "Optional AI guide not configured"}>
          <i className={modelReady ? "is-ready" : ""} />
          {!sidebarCollapsed && <span>{modelReady ? "AI guide ready" : "Offline lessons"}</span>}
          {!sidebarCollapsed && <small>v0.1</small>}
        </div>
      </div>
    </aside>
  );
}
