import { Check, ChevronDown } from "lucide-react";
import { useCallback, useId, useRef, useState } from "react";
import type { RegisteredModule } from "@app/modules/registry";
import { modulesById } from "@app/modules/registry";
import { prerequisitesOf, type PathState } from "@app/lib/learning-path";
import { useAppStore } from "@app/store/app-store";
import { useDismiss } from "./useDismiss";

const stateLabel: Record<PathState, string> = {
  done: "completed",
  started: "started, not completed",
  new: "not started",
};

/**
 * The labs this one builds on, with the learner's progress in each, as links back to them.
 * A single lab sits in the lab header as a chip. A longer list, or any list in a narrow header,
 * folds behind a count that opens it, so the header stays one row.
 */
export function PrerequisiteBar({ module }: { module: RegisteredModule }) {
  const progress = useAppStore((state) => state.progress);
  const openModule = useAppStore((state) => state.openModule);
  const [open, setOpen] = useState(false);
  const barRef = useRef<HTMLElement>(null);
  const listId = useId();
  const close = useCallback(() => setOpen(false), []);
  useDismiss(barRef, open, close);
  const items = prerequisitesOf(module, modulesById, progress);
  if (items.length === 0) return null;
  const unfinished = items.filter((item) => item.state !== "done");
  const finished = items.length - unfinished.length;

  return (
    <nav
      ref={barRef}
      className="module-prereqs"
      aria-label="Labs this one builds on"
      data-many={items.length > 1 ? "" : undefined}
      data-open={open ? "" : undefined}
    >
      <span>Builds on</span>
      <button
        type="button"
        className="module-prereqs__toggle"
        aria-expanded={open}
        aria-controls={listId}
        onClick={() => setOpen(!open)}
      >
        {items.length} {items.length === 1 ? "lab" : "labs"}
        <small>{finished} done</small>
        <ChevronDown />
      </button>
      <div className="module-prereqs__list" id={listId}>
        {items.map(({ module: target, state }) => (
          <button
            type="button"
            key={target.id}
            className={`prereq-chip is-${state}`}
            onClick={() => openModule(target.id)}
          >
            <i aria-hidden="true">{state === "done" ? <Check /> : null}</i>
            {target.title}
            <span className="sr-only"> ({stateLabel[state]})</span>
          </button>
        ))}
      </div>
      {unfinished.length > 0 && (
        <small>
          {unfinished.length === items.length
            ? "Worth a look first."
            : `${unfinished.length} still to finish.`}
        </small>
      )}
    </nav>
  );
}
