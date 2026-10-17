import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { useAppStore } from "@app/store/app-store";

/** Thrown into a running hero script once it should stop. */
export class Cancelled extends Error {}

/** Reports whether the script that received it is still the current one. */
export type Live = () => boolean;

export interface StageSize {
  width: number;
  height: number;
}

export const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
export const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
export const linear = (t: number) => t;

export function wait(ms: number, live: Live) {
  return new Promise<void>((resolve, reject) => {
    window.setTimeout(() => (live() ? resolve() : reject(new Cancelled())), ms);
  });
}

/** Calls `onFrame` with eased progress from 0 to 1 over `ms`, one call per animation frame. */
export function tween(ms: number, live: Live, onFrame: (t: number) => void, curve = easeInOut) {
  return new Promise<void>((resolve, reject) => {
    const start = performance.now();
    const frame = (now: number) => {
      if (!live()) {
        reject(new Cancelled());
        return;
      }
      const t = Math.min(1, (now - start) / ms);
      onFrame(curve(t));
      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
}

/**
 * Runs `script` while `active`, cancelling it when `active` turns false, the component unmounts,
 * or `restartKey` changes. The script reads `live()` through `wait` and `tween`, so it stops at
 * its next pause instead of writing state after it has been replaced.
 */
export function useHeroScript(active: boolean, script: (live: Live) => Promise<void>, restartKey = 0) {
  const scriptRef = useRef(script);
  useLayoutEffect(() => {
    scriptRef.current = script;
  });
  useEffect(() => {
    if (!active) return undefined;
    let alive = true;
    scriptRef.current(() => alive).catch((error: unknown) => {
      if (!(error instanceof Cancelled)) console.error(error);
    });
    return () => {
      alive = false;
    };
  }, [active, restartKey]);
}

const reducedMotionQuery = "(prefers-reduced-motion: reduce)";

function systemPrefersReducedMotion() {
  return typeof window.matchMedia === "function" && window.matchMedia(reducedMotionQuery).matches;
}

/**
 * True when the hero should animate: it is on screen, and neither the app's reduced-motion setting
 * nor the system preference asks for stillness. Without IntersectionObserver it stays still.
 */
export function useHeroMotion(ref: RefObject<HTMLElement | null>) {
  const reducedMotion = useAppStore((state) => state.reducedMotion);
  const [systemReduced, setSystemReduced] = useState(systemPrefersReducedMotion);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return undefined;
    const media = window.matchMedia(reducedMotionQuery);
    const update = () => setSystemReduced(media.matches);
    media.addEventListener?.("change", update);
    return () => media.removeEventListener?.("change", update);
  }, []);

  useEffect(() => {
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") return undefined;
    const observer = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), { threshold: 0.2 });
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref]);

  return inView && !reducedMotion && !systemReduced;
}

/** The stage's size in CSS pixels, with a typical desktop size until the first measurement. */
export function useStageSize(ref: RefObject<HTMLElement | null>) {
  const [size, setSize] = useState<StageSize>({ width: 640, height: 480 });
  useEffect(() => {
    const node = ref.current;
    if (!node || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.round(entry.contentRect.width);
      const height = Math.round(entry.contentRect.height);
      if (width > 0 && height > 0) {
        setSize((current) => (current.width === width && current.height === height ? current : { width, height }));
      }
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

export interface HeroVisualProps {
  size: StageSize;
  animate: boolean;
}
