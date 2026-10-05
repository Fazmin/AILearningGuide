import {
  Accessibility,
  Check,
  Cloud,
  Database,
  Eye,
  FileJson,
  Info,
  Laptop,
  Moon,
  Palette,
  Play,
  RotateCcw,
  ShieldCheck,
  Sun,
  Type,
  Upload,
} from "lucide-react";
import { useRef, useState } from "react";
import { fontScaleForStep, useAppStore } from "@app/store/app-store";
import { CloudProviderSettings } from "./CloudProviderSettings";
import { LicencesPanel } from "./LicencesPanel";
import { LocalModelSettings } from "./LocalModelSettings";
import { SettingsSelect } from "./SettingsSelect";

const tabs = [
  ["appearance", "Appearance", Palette],
  ["accessibility", "Accessibility", Accessibility],
  ["explanations", "Explanation mode", Type],
  ["local", "Local model", Laptop],
  ["cloud", "Cloud providers", Cloud],
  ["data", "Data", Database],
  ["about", "About", Info],
] as const;

function SettingRow({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="setting-row">
      <div><strong>{title}</strong><span>{description}</span></div>
      <div>{children}</div>
    </div>
  );
}

function SectionHeading({ title, description }: { title: string; description: string }) {
  return (
    <header className="settings-section-heading">
      <h2>{title}</h2>
      <p>{description}</p>
    </header>
  );
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
}) {
  return (
    <button type="button" role="switch" aria-checked={checked} aria-label={label} className="settings-switch" onClick={() => onChange(!checked)}>
      <i><span /></i><b>{checked ? "On" : "Off"}</b>
    </button>
  );
}

export function SettingsScreen() {
  const store = useAppStore();
  const tab = store.settingsTab;
  const setTab = store.setSettingsTab;
  const [dataStatus, setDataStatus] = useState("");
  const importRef = useRef<HTMLInputElement>(null);

  const exportData = () => {
    const data = JSON.stringify({
      format: "the-ai-guide-export",
      version: 1,
      exportedAt: new Date().toISOString(),
      progress: store.progress,
      snapshots: store.snapshots,
      moduleStates: store.moduleStates,
      preferences: {
        theme: store.theme,
        typeface: store.typeface,
        mode: store.mode,
        fontStep: store.fontStep,
        highContrast: store.highContrast,
        reducedMotion: store.reducedMotion,
        verboseNarration: store.verboseNarration,
        colorVision: store.colorVision,
      },
    }, null, 2);
    const url = URL.createObjectURL(new Blob([data], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `discover-ai-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const resetData = () => {
    if (!window.confirm("Reset all learning progress, snapshots, and saved lab states on this device?")) return;
    store.resetLearningData();
    setDataStatus("Learning data reset.");
  };

  return (
    <main className="settings-screen">
      <header className="settings-header">
        <h1>Settings</h1>
        <p>Adjust reading, accessibility, privacy, and model options.</p>
      </header>

      <div className="settings-layout">
        <nav className="settings-tabs" aria-label="Settings sections">
          {tabs.map(([id, label, Icon]) => (
            <button
              type="button"
              key={id}
              className={tab === id ? "is-active" : ""}
              aria-current={tab === id ? "page" : undefined}
              onClick={() => setTab(id)}
            >
              <Icon /><span>{label}</span>
            </button>
          ))}
        </nav>

        <section className="settings-content">
          {tab === "appearance" && (
            <>
              <SectionHeading title="Appearance" description="Choose how the learning workspace looks." />
              <SettingRow title="Theme" description="Follow the system or choose a fixed appearance.">
                <div className="theme-options">
                  {([
                    ["light", "Light", Sun],
                    ["dark", "Dark", Moon],
                    ["system", "System", Laptop],
                  ] as const).map(([value, label, Icon]) => (
                    <button type="button" key={value} aria-pressed={store.theme === value} onClick={() => store.setTheme(value)}><Icon />{label}</button>
                  ))}
                </div>
              </SettingRow>
              <SettingRow title="Typeface" description="Sans for a compact interface, or an elegant serif for longer reading.">
                <div className="theme-options typeface-options">
                  <button
                    type="button"
                    className="typeface-option typeface-option--sans"
                    aria-pressed={store.typeface === "sans"}
                    onClick={() => store.setTypeface("sans")}
                  >
                    Sans
                  </button>
                  <button
                    type="button"
                    className="typeface-option typeface-option--elegant"
                    aria-pressed={store.typeface === "elegant"}
                    onClick={() => store.setTypeface("elegant")}
                  >
                    Elegant
                  </button>
                </div>
              </SettingRow>
              <SettingRow title="Text size" description="Seven steps from 87.5% to 200%. Layouts reflow at every step.">
                <div className="text-size-setting">
                  <button type="button" aria-label="Decrease text size" disabled={store.fontStep === 0} onClick={() => store.setFontStep(store.fontStep - 1)}>A</button>
                  <input
                    type="range"
                    min={0}
                    max={6}
                    step={1}
                    value={store.fontStep}
                    aria-label="Text size"
                    onChange={(event) => store.setFontStep(Number(event.target.value))}
                  />
                  <button type="button" aria-label="Increase text size" disabled={store.fontStep === 6} onClick={() => store.setFontStep(store.fontStep + 1)}>A</button>
                  <output aria-live="polite">{Math.round(fontScaleForStep(store.fontStep) * 100)}%</output>
                </div>
              </SettingRow>
              <div className="settings-preview">
                <span>Preview</span>
                <h3>Attention routes context.</h3>
                <p>Every token can retrieve information from the positions that matter to its current job.</p>
                <code>softmax(QKᵀ / √d) V</code>
              </div>
            </>
          )}

          {tab === "accessibility" && (
            <>
              <SectionHeading title="Accessibility" description="Alternatives for motion, colour, and visual updates." />
              <SettingRow title="Higher contrast" description="Strengthen borders and text separation."><Toggle label="Higher contrast" checked={store.highContrast} onChange={store.setHighContrast} /></SettingRow>
              <SettingRow title="Reduce motion" description="Disable ambient and transition animation."><Toggle label="Reduce motion" checked={store.reducedMotion} onChange={store.setReducedMotion} /></SettingRow>
              <SettingRow title="Verbose narration" description="Announce values when an interactive control changes."><Toggle label="Verbose narration" checked={store.verboseNarration} onChange={store.setVerboseNarration} /></SettingRow>
              <SettingRow title="Colour-vision palette" description="Remap semantic chart colours while preserving labels.">
                <SettingsSelect
                  ariaLabel="Colour-vision palette"
                  size="inline"
                  value={store.colorVision}
                  onChange={store.setColorVision}
                  options={[
                    { value: "default", label: "Default" },
                    { value: "deuteranopia", label: "Deuteranopia" },
                    { value: "protanopia", label: "Protanopia" },
                    { value: "tritanopia", label: "Tritanopia" },
                  ]}
                />
              </SettingRow>
              <div className="accessibility-promise"><Eye /><div><strong>No interaction depends on colour or dragging alone.</strong><span>Charts pair colour with shape, labels, keyboard controls, and text summaries.</span></div></div>
            </>
          )}

          {tab === "explanations" && (
            <>
              <SectionHeading title="Explanation mode" description="Change the writing without resetting your experiments." />
              <div className="explanation-choices">
                <button type="button" aria-pressed={store.mode === "plain"} onClick={() => store.setMode("plain")}>
                  <span>Plain</span><h3>Everyday words</h3><p>Written at a grade 8 to 10 reading level: the big picture first, then examples and short stories, a concrete nudge to try, and misconceptions corrected in place.</p><i>{store.mode === "plain" && <Check />}</i>
                </button>
                <button type="button" aria-pressed={store.mode === "standard"} onClick={() => store.setMode("standard")}>
                  <span>Standard</span><h3>Use the technical vocabulary</h3><p>Compact definitions, mechanisms, limitations, and links between ideas.</p><i>{store.mode === "standard" && <Check />}</i>
                </button>
              </div>
              <p className="privacy-note"><ShieldCheck /> Both modes cover the same learning objectives. You can switch at any time.</p>
            </>
          )}

          {tab === "local" && (
            <>
              <SectionHeading title="Local model" description="Private, offline topic chat through a llama.cpp sidecar." />
              <LocalModelSettings />
            </>
          )}

          {tab === "cloud" && (
            <>
              <SectionHeading title="Cloud providers" description="Choose a provider and model, then test your key before it goes in the OS keychain." />
              <CloudProviderSettings />
              <p className="privacy-note"><ShieldCheck /> Keys never enter SQLite, local exports, logs, or frontend persistence.</p>
            </>
          )}

          {tab === "data" && (
            <>
              <SectionHeading title="Data" description="Your progress and experiments belong to you." />
              <div className="data-summary">
                <div><strong>{Object.values(store.progress).filter((item) => item.completed).length}</strong><span>labs complete</span></div>
                <div><strong>{store.snapshots.length}</strong><span>snapshots</span></div>
                <div><strong>{Object.keys(store.moduleStates).length}</strong><span>labs explored</span></div>
              </div>
              <SettingRow title="Export learning data" description="Download progress, snapshots, and preferences as readable JSON."><button type="button" className="settings-action" onClick={exportData}><FileJson /> Export JSON</button></SettingRow>
              <SettingRow title="Import learning data" description="Restore a previous Discover AI export.">
                <>
                  <input
                    ref={importRef}
                    hidden
                    type="file"
                    accept="application/json"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (!file) return;
                      void file.text().then((content) => {
                        try {
                          const imported = store.importLearningData(JSON.parse(content));
                          setDataStatus(imported ? "Learning data imported." : "This is not a compatible export.");
                        } catch {
                          setDataStatus("The selected file is not valid JSON.");
                        }
                        event.target.value = "";
                      });
                    }}
                  />
                  <button type="button" className="settings-action" onClick={() => importRef.current?.click()}><Upload /> Choose file</button>
                </>
              </SettingRow>
              <SettingRow title="Reset local data" description="Remove progress and saved states from this device."><button type="button" className="settings-action settings-action--danger" onClick={resetData}><RotateCcw /> Reset learning data</button></SettingRow>
              {dataStatus && <p className="data-status" role="status">{dataStatus}</p>}
            </>
          )}

          {tab === "about" && (
            <>
              <SectionHeading title="About" description="An interactive field guide to modern AI." />
              <div className="about-mark"><i><span /><span /><span /></i><div><h3>Discover AI: A Learning Guide</h3><p>Version 0.1.0 · desktop preview</p></div></div>
              <div className="about-copy">
                <p>Built to make technical ideas inspectable rather than magical. Every lesson pairs a manipulable system with precise limits on what the picture can claim.</p>
                <p>The application shell is React, TypeScript, and Tauri. Desktop data is stored in bundled SQLite; provider secrets belong in the operating system keychain.</p>
              </div>
              <div className="about-links"><a href="https://tauri.app" target="_blank" rel="noreferrer">Tauri</a><a href="https://react.dev" target="_blank" rel="noreferrer">React</a><a href="https://github.com/lucide-icons/lucide" target="_blank" rel="noreferrer">Lucide</a></div>
              <SettingRow title="Welcome tour" description="Replay the three-step introduction: explanation modes, the optional guide, and accessibility."><button type="button" className="settings-action settings-action--secondary" onClick={() => store.replayOnboarding()}><Play /> Replay the welcome tour</button></SettingRow>
              <LicencesPanel />
              <p className="build-id">Build channel / source · Database schema / 1 · State contract / 1</p>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
