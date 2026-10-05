import { Check } from "lucide-react";
import type { RegisteredModule } from "@app/modules/registry";
import { modulesById } from "@app/modules/registry";
import { prerequisitesOf, type PathState } from "@app/lib/learning-path";
import { useAppStore } from "@app/store/app-store";

const stateLabel: Record<PathState, string> = {
  done: "completed",
  started: "started, not completed",
  new: "not started",
};

/** The labs this one builds on, with the learner's progress in each, as links back to them. */
export function PrerequisiteBar({ module }: { module: RegisteredModule }) {
  const progress = useAppStore((state) => state.progress);
  const openModule = useAppStore((state) => state.openModule);
  const items = prerequisitesOf(module, modulesById, progress);
  if (items.length === 0) return null;
  const unfinished = items.filter((item) => item.state !== "done");

  return (
    <nav className="module-prereqs" aria-label="Labs this one builds on">
      <span>Builds on</span>
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
