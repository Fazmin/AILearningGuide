import { useAppStore } from "@app/store/app-store";

export const heroVariants = ["next-word", "attention", "meaning-map", "descent"] as const;
export type HeroVariant = (typeof heroVariants)[number];

export const isHeroVariant = (value: unknown): value is HeroVariant =>
  heroVariants.includes(value as HeroVariant);

/** A random variant, other than the one shown last time when there is a choice. */
export function pickHeroVariant(previous: string | null, random: () => number = Math.random): HeroVariant {
  const pool = heroVariants.filter((variant) => variant !== previous);
  return pool[Math.min(pool.length - 1, Math.floor(random() * pool.length))];
}

let launchVariant: HeroVariant | null = null;

/**
 * The hero for this launch: picked once, then kept while the app stays open, so going back to the
 * home screen shows the same one. In development, `?hero=<variant>` forces a variant for review.
 */
export function heroForThisLaunch(): HeroVariant {
  if (launchVariant) return launchVariant;
  const forced = import.meta.env.DEV ? new URLSearchParams(window.location.search).get("hero") : null;
  launchVariant = isHeroVariant(forced) ? forced : pickHeroVariant(useAppStore.getState().lastHomeHero);
  return launchVariant;
}

/** Test hook: forget this launch's pick. */
export function resetLaunchHero() {
  launchVariant = null;
}
