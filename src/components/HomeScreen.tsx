import { ArrowRight, Check, MousePointer2, Route, Sparkles, Waypoints } from "lucide-react";
import { moduleGroups } from "@app/modules/groups";
import { modulesInGroup, type RegisteredModule } from "@app/modules/registry";
import { useEffect } from "react";
import { nextInSequence, pathComplete } from "@app/lib/learning-path";
import { useAppStore } from "@app/store/app-store";

const TRACK_WRAP_AFTER = 6;

function splitTrackRows<T>(items: T[]): T[][] {
  if (items.length <= TRACK_WRAP_AFTER) return [items];
  const midpoint = Math.ceil(items.length / 2);
  return [items.slice(0, midpoint), items.slice(midpoint)];
}

function MapNodeButton({
  module,
  state,
  onOpen,
}: {
  module: RegisteredModule;
  state: string;
  onOpen: (id: string) => void;
}) {
  return (
    <button
      type="button"
      className={`map-node ${state}`}
      style={{ "--module-accent": module.accent } as React.CSSProperties}
      onClick={() => onOpen(module.id)}
    >
      <span>{state === "is-done" ? <Check /> : String(module.order).padStart(2, "0")}</span>
      <strong>{module.title}</strong>
      <small>{module.estimatedMinutes} min</small>
    </button>
  );
}

const signalCurve = "M 1 1 C 55 1 99 45 99 99";
const signalAnimation = {
  one: {
    keyPoints: "0; 0; 1; 1; 1",
    keyTimes: "0; 0.08; 0.42; 0.5; 1",
    opacity: "0; 1; 1; 0; 0",
  },
  two: {
    keyPoints: "0; 0; 0; 1; 1",
    keyTimes: "0; 0.48; 0.56; 0.9; 1",
    opacity: "0; 0; 1; 1; 0",
  },
  three: {
    keyPoints: "0; 0; 1; 1; 1",
    keyTimes: "0; 0.14; 0.48; 0.56; 1",
    opacity: "0; 1; 1; 0; 0",
  },
  four: {
    keyPoints: "0; 0; 0; 1; 1",
    keyTimes: "0; 0.52; 0.6; 0.94; 1",
    opacity: "0; 0; 1; 1; 0",
  },
} as const;
const signalKeySplines = "0.42 0 0.58 1; 0.42 0 0.58 1; 0.42 0 0.58 1; 0.42 0 0.58 1";

function SignalPath({ variant }: { variant: keyof typeof signalAnimation }) {
  const curveId = `signal-curve-${variant}`;
  const animation = signalAnimation[variant];

  return (
    <svg
      aria-hidden="true"
      className={`signal-path signal-path--${variant}`}
      preserveAspectRatio="none"
      viewBox="0 0 100 100"
    >
      <path className="signal-path__line" d={signalCurve} id={curveId} />
      <path className="signal-path__dot" d="M 0 0 h 0.01">
        <animateMotion
          calcMode="spline"
          dur="4.2s"
          keyPoints={animation.keyPoints}
          keySplines={signalKeySplines}
          keyTimes={animation.keyTimes}
          repeatCount="indefinite"
        >
          <mpath href={`#${curveId}`} />
        </animateMotion>
        <animate
          attributeName="opacity"
          dur="4.2s"
          keyTimes={animation.keyTimes}
          repeatCount="indefinite"
          values={animation.opacity}
        />
      </path>
    </svg>
  );
}

export function HomeScreen() {
  const { progress, openModule, openConnections, mode, searchFocus, reducedMotion, setView, setSettingsTab } = useAppStore();

  useEffect(() => {
    if (searchFocus?.kind !== "screen" || searchFocus.screen !== "home" || !searchFocus.groupId) return;
    const target = document.getElementById(`map-group-${searchFocus.groupId}`);
    target?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
  }, [reducedMotion, searchFocus]);
  const allModules = moduleGroups.flatMap((group) => modulesInGroup(group.id));
  const firstByOrder = allModules.reduce(
    (earliest, module) => (module.order < earliest.order ? module : earliest),
    allModules[0],
  );
  const completed = allModules.filter((module) => progress[module.id]?.completed).length;
  const visited = allModules.filter((module) => progress[module.id]?.visited);
  const continueModule = visited.length
    ? [...visited].sort(
        (a, b) =>
          new Date(progress[b.id]?.updatedAt ?? 0).getTime() -
          new Date(progress[a.id]?.updatedAt ?? 0).getTime(),
      )[0]
    : firstByOrder;
  const upNext = nextInSequence(allModules, progress);
  const finished = pathComplete(allModules, progress);
  const lastModule = allModules.reduce(
    (latest, module) => (module.order > latest.order ? module : latest),
    allModules[0],
  );

  return (
    <main className="home-screen">
      <section className="home-intro">
        <div className="home-intro__copy">
          <span className="intro-index">Learning guide / {allModules.length} labs</span>
          <h1>Learn AI.</h1>
          <p>
            A friendly guide to how artificial intelligence works, broken into the parts that make it.
            Understand each one, then put them together and create your own. You learn by trying, not just reading.
          </p>
          <div className="home-actions">
            <button
              type="button"
              className="start-action"
              onClick={() => openModule(continueModule.id)}
            >
              <span>{visited.length ? "Continue exploring" : "Start with the groundwork"}</span>
              <small>{continueModule.title} · {continueModule.estimatedMinutes} min</small>
              <ArrowRight />
            </button>
            <div
              className="home-progress"
              aria-label={`${completed} of ${allModules.length} modules completed`}
            >
              <strong>{String(completed).padStart(2, "0")}</strong>
              <span>of {allModules.length} labs<br />completed</span>
              <i
                style={{
                  "--progress": `${(completed / Math.max(1, allModules.length)) * 100}%`,
                } as React.CSSProperties}
              />
            </div>
          </div>
          {visited.length > 0 && upNext && upNext.id !== continueModule.id && (
            <p className="home-upnext">
              Next in the sequence:{" "}
              <button type="button" onClick={() => openModule(upNext.id)}>
                Lab {String(upNext.order).padStart(2, "0")} · {upNext.title}
              </button>
            </p>
          )}
        </div>

        <div className="signal-instrument" aria-label="Animated model signal illustration">
          <div className="instrument-grid" />
          <div className="signal-input"><span>the animal</span><i /></div>
          <SignalPath variant="one" />
          <SignalPath variant="two" />
          <SignalPath variant="three" />
          <SignalPath variant="four" />
          <div className="signal-nucleus">
            <i /><i /><i />
            <strong>meaning</strong>
            <span>[0.21, −0.74, 0.63]</span>
          </div>
          <div className="signal-output"><i /><span>crossed</span><b>42%</b></div>
          <div className="signal-output signal-output--two"><i /><span>was</span><b>31%</b></div>
          <div className="instrument-caption">
            <Sparkles />
            <span>Live model anatomy</span>
            <b>open a lab to steer it</b>
          </div>
        </div>
      </section>

      {finished && (
        <section className="home-finish" aria-labelledby="home-finish-title">
          <div>
            <h2 id="home-finish-title">You’ve been through the whole guide.</h2>
            <p>
              Every lab is marked complete. Revisit any lab to retake its checkpoint or try a different
              experiment, copy a state link to share one, or take your progress with you.
            </p>
          </div>
          <div className="home-finish__actions">
            <button type="button" onClick={() => openModule(lastModule.id)}>
              Revisit the last lab <ArrowRight />
            </button>
            <button
              type="button"
              onClick={() => {
                setSettingsTab("data");
                setView("settings");
              }}
            >
              Export your progress
            </button>
          </div>
        </section>
      )}

      <section className="learning-map" aria-labelledby="learning-map-title">
        <div className="learning-map__heading">
          <div>
            <h2 id="learning-map-title">Choose a path through the model</h2>
            <p>Follow the sequence or jump to the part you want to understand.</p>
          </div>
          <div className="map-key">
            <span><i className="is-done"><Check /></i> complete</span>
            <span><i className="is-active" /> in progress</span>
            <span><i /> not started</span>
            <button type="button" className="map-connections" onClick={() => openConnections()}>
              <Waypoints /> See how the labs connect
            </button>
          </div>
        </div>

        <div className="map-tracks">
          {moduleGroups.map((group, groupIndex) => {
            const groupModules = modulesInGroup(group.id);
            const rows = splitTrackRows(groupModules);
            return (
              <section
                className={`map-track${rows.length > 1 ? " map-track--stacked" : ""}`}
                key={group.id}
                id={`map-group-${group.id}`}
              >
                <div className="track-label">
                  <span>{groupIndex + 1}</span>
                  <div><h3>{group.title}</h3><p>{group.description}</p></div>
                </div>
                <div className="track-lines">
                  {rows.map((row) => (
                    <div className="track-line" key={row.map((module) => module.id).join("-")}>
                      {row.map((module) => {
                        const state = progress[module.id]?.completed
                          ? "is-done"
                          : progress[module.id]?.visited
                            ? "is-active"
                            : "";
                        return (
                          <MapNodeButton
                            key={module.id}
                            module={module}
                            state={state}
                            onOpen={openModule}
                          />
                        );
                      })}
                    </div>
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      </section>

      <section className="home-footnote">
        <div><MousePointer2 /><span><strong>Everything responds.</strong> Every diagram is a control, not a screenshot.</span></div>
        <div><Route /><span><strong>{mode === "plain" ? "Plain explanations are on." : "Standard explanations are on."}</strong> Switch modes without losing your place.</span></div>
      </section>
    </main>
  );
}
