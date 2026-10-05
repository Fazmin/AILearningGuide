import { ArrowRight, Check, Minus, Plus, Waypoints, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { pathState, type PathState } from "@app/lib/learning-path";
import {
  buildGraph,
  defaultLayoutOptions,
  focusOf,
  layoutGraph,
  type Focus,
  type Point,
} from "@app/lib/module-graph";
import { moduleGroups } from "@app/modules/groups";
import { modules, modulesById } from "@app/modules/registry";
import { useAppStore } from "@app/store/app-store";

const graph = buildGraph(modules);
const layout = layoutGraph(graph);
const { nodeWidth, nodeHeight } = defaultLayoutOptions;
const groupTitles = new Map(moduleGroups.map((group) => [group.id, group.title]));

const MIN_ZOOM = 0.4;
const MAX_ZOOM = 1.6;
const clampZoom = (zoom: number) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.round(zoom * 100) / 100));
const pad = (order: number) => String(order).padStart(2, "0");
const plural = (count: number, one: string, many = `${one}s`) => `${count} ${count === 1 ? one : many}`;

/** Each link leaves one lab vertically and arrives vertically, so it curves through the gap between rows. */
function linePath(points: Point[]) {
  return points.slice(1).reduce((path, to, index) => {
    const from = points[index];
    const middle = (from.y + to.y) / 2;
    return `${path} C ${from.x} ${middle}, ${to.x} ${middle}, ${to.x} ${to.y}`;
  }, `M ${points[0].x} ${points[0].y}`);
}

function headPath({ x, y }: Point) {
  return `M ${x - 0.3} ${y - 0.55} L ${x + 0.3} ${y - 0.55} L ${x} ${y} Z`;
}

const links = layout.links.map((link) => ({
  ...link,
  key: `${link.from}>${link.to}`,
  line: linePath(link.points),
  head: headPath(link.points[link.points.length - 1]),
}));

type Role = "none" | "selected" | "before" | "after" | "apart";

function roleOf(focus: Focus | null, id: string): Role {
  if (!focus) return "none";
  if (id === focus.id) return "selected";
  if (focus.beforeSet.has(id)) return "before";
  if (focus.afterSet.has(id)) return "after";
  return "apart";
}

type Tone = "idle" | "before" | "after" | "dim";
const toneRank: Record<Tone, number> = { dim: 0, idle: 1, before: 2, after: 2 };

/** A link is lit when both its ends sit on the route into the selected lab, or on the route out of it. */
function toneOf(focus: Focus | null, from: string, to: string): Tone {
  if (!focus) return "idle";
  const upstream = (id: string) => id === focus.id || focus.beforeSet.has(id);
  const downstream = (id: string) => id === focus.id || focus.afterSet.has(id);
  if (upstream(from) && upstream(to)) return "before";
  if (downstream(from) && downstream(to)) return "after";
  return "dim";
}

const stateLabel: Record<PathState, string> = {
  done: "completed",
  started: "started, not completed",
  new: "not started",
};

const roleLabel: Record<Role, string> = {
  none: "",
  selected: "selected",
  before: "the selected lab builds on this",
  after: "this builds on the selected lab",
  apart: "not connected to the selected lab",
};

export function ConnectionsScreen() {
  const focusId = useAppStore((state) => state.connectionsFocusId);
  const setFocus = useAppStore((state) => state.setConnectionsFocus);
  const progress = useAppStore((state) => state.progress);
  const reducedMotion = useAppStore((state) => state.reducedMotion);
  const [zoom, setZoom] = useState(1);
  const scroller = useRef<HTMLDivElement>(null);
  const nodes = useRef(new Map<string, HTMLButtonElement>());
  const arrived = useRef(false);

  const selected = focusId ? modulesById.get(focusId) : undefined;
  const focus = useMemo(() => (selected ? focusOf(graph, selected.id) : null), [selected]);
  const stateOf = (id: string) => pathState(progress[id]);

  // Arriving jumps straight to the right place; later selections scroll there. This is one effect so
  // that React running it twice in development cannot undo the first jump.
  useEffect(() => {
    const view = scroller.current;
    if (!view) return;
    const arriving = !arrived.current;
    arrived.current = true;
    const node = selected ? nodes.current.get(selected.id) : undefined;
    if (!node) {
      // The first lab sits at the middle of the layout, so open there rather than at the left edge.
      if (arriving) view.scrollLeft = (view.scrollWidth - view.clientWidth) / 2;
      return;
    }
    const frame = view.getBoundingClientRect();
    const box = node.getBoundingClientRect();
    const edge = 32;
    const visible =
      box.left >= frame.left + edge &&
      box.right <= frame.right - edge &&
      box.top >= frame.top + edge &&
      box.bottom <= frame.bottom - edge;
    if (visible) return;
    view.scrollTo({
      left: view.scrollLeft + box.left - frame.left - (frame.width - box.width) / 2,
      top: view.scrollTop + box.top - frame.top - (frame.height - box.height) / 2,
      behavior: reducedMotion || arriving ? "auto" : "smooth",
    });
  }, [selected, zoom, reducedMotion]);

  const fitWidth = () => {
    const view = scroller.current;
    if (!view) return;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
    setZoom(clampZoom(Math.floor((view.clientWidth / (layout.width * rem)) * 100) / 100));
  };

  const lit = [...links].sort(
    (a, b) => toneRank[toneOf(focus, a.from, a.to)] - toneRank[toneOf(focus, b.from, b.to)],
  );

  return (
    <main
      className="connections-screen"
      onKeyDown={(event) => {
        if (event.key === "Escape" && focusId) setFocus(null);
      }}
    >
      <header className="conn-header">
        <div>
          <span className="conn-eyebrow">
            <Waypoints /> Connections · {modules.length} labs, {layout.links.length} links
          </span>
          <h1>How the labs connect</h1>
          <p>
            Every lab sits below the labs it builds on. Select one to light up the route that leads
            to it and everything it opens up.
          </p>
        </div>
      </header>

      <div className="conn-body">
        <div className="conn-main">
          <div className="conn-toolbar">
            <ul className="conn-key" aria-label="Key">
              <li><i className="conn-key__line is-before" /> what it builds on (above)</li>
              <li><i className="conn-key__line is-after" /> what builds on it (below)</li>
              <li><i className="conn-key__dot is-done"><Check /></i> complete</li>
              <li><i className="conn-key__dot is-started" /> in progress</li>
            </ul>
            <div className="conn-zoom" role="group" aria-label="Zoom">
              <button type="button" aria-label="Zoom out" disabled={zoom <= MIN_ZOOM} onClick={() => setZoom(clampZoom(zoom - 0.1))}><Minus /></button>
              <output aria-live="off">{Math.round(zoom * 100)}%</output>
              <button type="button" aria-label="Zoom in" disabled={zoom >= MAX_ZOOM} onClick={() => setZoom(clampZoom(zoom + 0.1))}><Plus /></button>
              <button type="button" onClick={fitWidth}>Fit</button>
              <button type="button" onClick={() => setZoom(1)}>Reset</button>
            </div>
          </div>
          <div
            className="conn-canvas"
            ref={scroller}
            role="group"
            tabIndex={0}
            aria-label="Map of the labs and how they connect. Scroll to explore; each lab is a button."
          >
            <div
              className={`conn-plane${focus ? " has-focus" : ""}`}
              style={{
                width: `${layout.width * zoom}rem`,
                height: `${layout.height * zoom}rem`,
                "--zoom": zoom,
              } as React.CSSProperties}
              onClick={(event) => {
                if (event.target === event.currentTarget) setFocus(null);
              }}
            >
              <svg
                className="conn-links"
                aria-hidden="true"
                focusable="false"
                viewBox={`0 0 ${layout.width} ${layout.height}`}
                style={{ width: `${layout.width * zoom}rem`, height: `${layout.height * zoom}rem` }}
              >
                {lit.map((link) => {
                  const tone = toneOf(focus, link.from, link.to);
                  const direct = focus && (link.from === focus.id || link.to === focus.id);
                  return (
                    <g key={link.key} className={`conn-link is-${tone}${direct ? " is-direct" : ""}`}>
                      <path d={link.line} />
                      <path className="conn-link__head" d={link.head} />
                    </g>
                  );
                })}
              </svg>

              {layout.nodes.map((node) => {
                const module = modulesById.get(node.id)!;
                const role = roleOf(focus, node.id);
                const state = stateOf(node.id);
                const direct = focus ? focus.directBefore.includes(node.id) || focus.directAfter.includes(node.id) : false;
                return (
                  <button
                    type="button"
                    key={node.id}
                    ref={(element) => {
                      if (element) nodes.current.set(node.id, element);
                      else nodes.current.delete(node.id);
                    }}
                    className={`conn-node is-${role} is-${state}${direct ? " is-direct" : ""}`}
                    style={{
                      left: `${node.x * zoom}rem`,
                      top: `${node.y * zoom}rem`,
                      width: `${nodeWidth * zoom}rem`,
                      height: `${nodeHeight * zoom}rem`,
                      "--module-accent": module.accent,
                    } as React.CSSProperties}
                    aria-pressed={role === "selected"}
                    onClick={() => setFocus(role === "selected" ? null : node.id)}
                  >
                    <span className="conn-node__num">{state === "done" ? <Check aria-hidden="true" /> : pad(module.order)}</span>
                    <span className="conn-node__title">{module.title}</span>
                    <span className="sr-only">
                      {" "}
                      (lab {module.order}; {stateLabel[state]}
                      {role !== "none" ? `; ${roleLabel[role]}` : ""})
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <ConnectionsPanel focus={focus} stateOf={stateOf} onSelect={setFocus} />
      </div>
    </main>
  );
}

function LabButton({
  id,
  state,
  onSelect,
}: {
  id: string;
  state: PathState;
  onSelect: (id: string) => void;
}) {
  const module = modulesById.get(id)!;
  return (
    <button
      type="button"
      className={`conn-lab is-${state}`}
      style={{ "--module-accent": module.accent } as React.CSSProperties}
      onClick={() => onSelect(id)}
    >
      <span className="conn-lab__num">{state === "done" ? <Check aria-hidden="true" /> : pad(module.order)}</span>
      <span>{module.title}</span>
      <span className="sr-only"> ({stateLabel[state]})</span>
    </button>
  );
}

function LabList({
  label,
  ids,
  stateOf,
  onSelect,
}: {
  label: string;
  ids: string[];
  stateOf: (id: string) => PathState;
  onSelect: (id: string) => void;
}) {
  return (
    <section className="conn-list">
      <h3>{label}</h3>
      <ul>
        {ids.map((id) => (
          <li key={id}><LabButton id={id} state={stateOf(id)} onSelect={onSelect} /></li>
        ))}
      </ul>
    </section>
  );
}

function Everything({
  summary,
  ids,
  stateOf,
  onSelect,
  numbered,
}: {
  summary: string;
  ids: string[];
  stateOf: (id: string) => PathState;
  onSelect: (id: string) => void;
  numbered?: boolean;
}) {
  const List = numbered ? "ol" : "ul";
  return (
    <details className="conn-everything">
      <summary>{summary}</summary>
      <List>
        {ids.map((id) => (
          <li key={id}><LabButton id={id} state={stateOf(id)} onSelect={onSelect} /></li>
        ))}
      </List>
    </details>
  );
}

function ConnectionsPanel({
  focus,
  stateOf,
  onSelect,
}: {
  focus: Focus | null;
  stateOf: (id: string) => PathState;
  onSelect: (id: string | null) => void;
}) {
  const openModule = useAppStore((state) => state.openModule);
  const module = focus ? modulesById.get(focus.id) : undefined;

  if (!focus || !module) {
    return (
      <aside className="conn-panel" aria-label="Selected lab">
        <div className="conn-panel__empty">
          <Waypoints />
          <h2>Pick a lab</h2>
          <p>
            Select any lab on the map to see what it builds on, what builds on it, and which of
            those you have already finished.
          </p>
          <p>
            The map reads top to bottom: {layout.rows} steps from the first lab to the last. Labs
            side by side can be learned in either order.
          </p>
        </div>
      </aside>
    );
  }

  const unfinished = focus.before.filter((id) => stateOf(id) !== "done");
  const startHere = unfinished[0] ? modulesById.get(unfinished[0]) : undefined;

  return (
    <aside className="conn-panel" aria-label="Selected lab" style={{ "--module-accent": module.accent } as React.CSSProperties}>
      <header className="conn-panel__head">
        <span>Lab {pad(module.order)} · {groupTitles.get(module.group)}</span>
        <h2>{module.title}</h2>
        <p>{module.tagline}</p>
        <div className="conn-panel__actions">
          <button type="button" className="conn-primary" onClick={() => openModule(module.id)}>
            Open lab <ArrowRight />
          </button>
          <button type="button" onClick={() => onSelect(null)}>
            <X /> Clear
          </button>
        </div>
      </header>

      <p className="conn-panel__summary" role="status">
        <strong>Builds on {plural(focus.before.length, "lab")}</strong>
        <span>Opens up {plural(focus.after.length, "lab")}</span>
      </p>

      {focus.before.length === 0 ? (
        <p className="conn-panel__note">This is where the guide starts. Nothing needs to come first.</p>
      ) : startHere ? (
        <p className="conn-panel__note">
          You have finished {focus.before.length - unfinished.length} of the {focus.before.length} labs this builds on.
          The earliest one left is{" "}
          <button type="button" onClick={() => onSelect(startHere.id)}>
            Lab {pad(startHere.order)} · {startHere.title}
          </button>
          .
        </p>
      ) : (
        <p className="conn-panel__note">You have finished everything this lab builds on.</p>
      )}

      {focus.directBefore.length > 0 && (
        <LabList label="Builds directly on" ids={focus.directBefore} stateOf={stateOf} onSelect={onSelect} />
      )}
      {focus.directAfter.length > 0 ? (
        <LabList label="Leads directly to" ids={focus.directAfter} stateOf={stateOf} onSelect={onSelect} />
      ) : (
        <p className="conn-panel__note">Nothing else builds on this lab. It is an end point of the guide.</p>
      )}

      {focus.before.length > 1 && (
        <Everything
          summary={`The whole route here · ${plural(focus.before.length, "lab")}`}
          ids={focus.before}
          stateOf={stateOf}
          onSelect={onSelect}
          numbered
        />
      )}
      {focus.after.length > focus.directAfter.length && (
        <Everything
          summary={`Everything it opens up · ${plural(focus.after.length, "lab")}`}
          ids={focus.after}
          stateOf={stateOf}
          onSelect={onSelect}
        />
      )}
    </aside>
  );
}
