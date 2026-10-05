import { useEffect, useMemo, useState } from "react";
import { AccessibilityPopover } from "./components/AccessibilityPopover";
import { ConnectionsScreen } from "./components/ConnectionsScreen";
import { HomeScreen } from "./components/HomeScreen";
import { ModuleWorkspace } from "./components/ModuleWorkspace";
import { Onboarding } from "./components/Onboarding";
import { SearchPalette } from "./components/SearchPalette";
import { SettingsScreen } from "./components/SettingsScreen";
import { Sidebar } from "./components/Sidebar";
import { TitleBar } from "./components/TitleBar";
import { asSettingsTab } from "./search/navigate";
import { decodeState } from "./lib/math";
import { getLocalAiStatus, localAiAvailable } from "./lib/local-ai";
import { modulesById, modulesBySlug } from "./modules/registry";
import { fontScaleForStep, useAppStore } from "./store/app-store";
import type { ModuleState } from "./module-sdk";

interface SharedState {
  module: string;
  version: number;
  state: ModuleState;
}

export default function App() {
  const store = useAppStore();
  const [systemDark, setSystemDark] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);
  const [routeReady, setRouteReady] = useState(false);
  const activeModule = store.activeModuleId ? modulesById.get(store.activeModuleId) : undefined;
  const focusedModule = store.connectionsFocusId ? modulesById.get(store.connectionsFocusId) : undefined;
  const effectiveTheme = store.theme === "system" ? (systemDark ? "dark" : "light") : store.theme;
  const fontScale = useMemo(() => fontScaleForStep(store.fontStep), [store.fontStep]);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = (event: MediaQueryListEvent) => setSystemDark(event.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (!localAiAvailable()) return;
    void getLocalAiStatus()
      .then((status) => store.setLocalModelReady(status.engineInstalled && status.modelInstalled))
      .catch(() => store.setLocalModelReady(false));
    // The desktop filesystem is the source of truth; persisted UI state may be stale.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = effectiveTheme;
    document.documentElement.dataset.font = store.typeface;
    document.documentElement.dataset.contrast = store.highContrast ? "high" : "normal";
    document.documentElement.dataset.motion = store.reducedMotion ? "reduced" : "full";
    document.documentElement.dataset.palette = store.colorVision;
    document.documentElement.dataset.textSize = store.fontStep >= 4 ? "large" : "normal";
    document.documentElement.style.fontSize = `${fontScale * 100}%`;
    document.documentElement.style.colorScheme = effectiveTheme;
  }, [effectiveTheme, fontScale, store.fontStep, store.highContrast, store.reducedMotion, store.colorVision, store.typeface]);

  useEffect(() => {
    const applyRoute = () => {
      const route = window.location.hash.slice(1);
      if (!route.startsWith("/")) {
        store.setView("home");
        return;
      }
      const [path, query = ""] = route.split("?");
      const slug = path.slice(1);
      if (slug === "settings") {
        store.setView("settings");
        const tab = asSettingsTab(new URLSearchParams(query).get("tab"));
        if (tab) store.setSettingsTab(tab);
      } else if (slug === "connections") {
        const focus = new URLSearchParams(query).get("focus");
        store.setConnectionsFocus((focus && modulesBySlug.get(focus)?.id) || null);
        store.setView("connections");
      } else if (slug) {
        const module = modulesBySlug.get(slug);
        if (module) {
          store.openModule(module.id);
          const encoded = new URLSearchParams(query).get("state");
          if (encoded) {
            try {
              const shared = decodeState<SharedState>(encoded);
              if (shared.module === module.slug) {
                store.setModuleState(module.id, module.hydrateState(JSON.stringify(shared.state)));
              }
            } catch {
              // A malformed shared state should not prevent the lesson from opening.
            }
          }
        } else {
          store.setView("home");
        }
      } else {
        store.setView("home");
      }
    };
    applyRoute();
    setRouteReady(true);
    window.addEventListener("hashchange", applyRoute);
    return () => window.removeEventListener("hashchange", applyRoute);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!routeReady) return;
    const nextHash =
      store.view === "module" && activeModule
        ? `#/${activeModule.slug}`
        : store.view === "settings"
          ? store.settingsTab && store.settingsTab !== "appearance"
            ? `#/settings?tab=${store.settingsTab}`
            : "#/settings"
          : store.view === "connections"
            ? focusedModule
              ? `#/connections?focus=${focusedModule.slug}`
              : "#/connections"
            : "#/";
    if (window.location.hash !== nextHash) history.replaceState(null, "", nextHash);
    document.title =
      activeModule && store.view === "module"
        ? `${activeModule.title} · Discover AI`
        : store.view === "connections"
          ? "Connections · Discover AI"
          : "Discover AI: A Learning Guide";
  }, [activeModule, focusedModule, routeReady, store.settingsTab, store.view]);

  return (
    <div className="app">
      <TitleBar />
      <div className="app-shell">
        <Sidebar />
        <div className="app-content">
          {store.view === "home" && <HomeScreen />}
          {store.view === "settings" && <SettingsScreen />}
          {store.view === "connections" && <ConnectionsScreen />}
          {store.view === "module" && activeModule && <ModuleWorkspace key={activeModule.id} module={activeModule} />}
          {store.view === "module" && !activeModule && <HomeScreen />}
        </div>
      </div>
      <SearchPalette />
      <AccessibilityPopover />
      {/* After the first route is applied, so a shared link or deep link never flashes the tour. */}
      {routeReady && <Onboarding />}
    </div>
  );
}
