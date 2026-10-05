/** What the learning map needs to know about a module, so these helpers stay testable without the registry. */
interface PathModule {
  id: string;
  order: number;
  prerequisites: string[];
}

interface PathProgress {
  completed?: boolean;
  visited?: boolean;
}

export type PathState = "done" | "started" | "new";

export function pathState(progress: PathProgress | undefined): PathState {
  return progress?.completed ? "done" : progress?.visited ? "started" : "new";
}

/** The modules a lab builds on, each with how far the learner has got with it. */
export function prerequisitesOf<T extends PathModule>(
  module: PathModule,
  byId: ReadonlyMap<string, T>,
  progress: Record<string, PathProgress | undefined>,
) {
  return module.prerequisites.flatMap((id) => {
    const target = byId.get(id);
    return target ? [{ module: target, state: pathState(progress[id]) }] : [];
  });
}

/**
 * The earliest lab, in learning-map order, that is not yet complete. Registry tests require every
 * prerequisite to come earlier, so everything this lab builds on is already complete.
 */
export function nextInSequence<T extends PathModule>(
  modules: readonly T[],
  progress: Record<string, PathProgress | undefined>,
): T | undefined {
  return [...modules].sort((a, b) => a.order - b.order).find((module) => !progress[module.id]?.completed);
}

export function pathComplete(
  modules: readonly PathModule[],
  progress: Record<string, PathProgress | undefined>,
) {
  return modules.length > 0 && modules.every((module) => progress[module.id]?.completed);
}
