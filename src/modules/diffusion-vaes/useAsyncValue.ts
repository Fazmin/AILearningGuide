import { useEffect, useRef, useState } from "react";

export type AsyncStatus = "idle" | "loading" | "ready" | "error";

export interface AsyncValue<T> {
  status: AsyncStatus;
  value?: T;
  error?: string;
  /** Progress in [0, 1] for long sequential runs. */
  progress: number;
}

export interface RunToken {
  cancelled: boolean;
  report: (fraction: number) => void;
}

/**
 * Run an async job whenever `key` changes. The previous value is kept while a
 * new one loads, so dragging a control never blanks the picture. A null key
 * idles the job.
 */
export function useAsyncValue<T>(
  key: string | null,
  run: (token: RunToken) => Promise<T>,
  delay = 0,
): AsyncValue<T> {
  const [state, setState] = useState<AsyncValue<T>>({
    status: key === null ? "idle" : "loading",
    progress: 0,
  });
  const runRef = useRef(run);
  runRef.current = run;

  useEffect(() => {
    if (key === null) {
      setState((previous) => ({ ...previous, status: previous.value ? "ready" : "idle" }));
      return;
    }
    const token: RunToken = {
      cancelled: false,
      report: (fraction) => {
        if (!token.cancelled) setState((previous) => ({ ...previous, progress: fraction }));
      },
    };
    setState((previous) => ({ ...previous, status: "loading", progress: 0 }));
    const timer = window.setTimeout(() => {
      runRef
        .current(token)
        .then((value) => {
          if (!token.cancelled) setState({ status: "ready", value, progress: 1 });
        })
        .catch((reason: unknown) => {
          if (token.cancelled) return;
          setState((previous) => ({
            status: "error",
            value: previous.value,
            progress: 0,
            error: reason instanceof Error ? reason.message : String(reason),
          }));
        });
    }, delay);
    return () => {
      token.cancelled = true;
      window.clearTimeout(timer);
    };
  }, [key, delay]);

  return state;
}
