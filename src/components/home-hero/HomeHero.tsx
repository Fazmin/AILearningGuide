import { ArrowUpRight } from "lucide-react";
import { useEffect, useRef, useState, type ComponentType } from "react";
import { modulesBySlug } from "@app/modules/registry";
import { useAppStore } from "@app/store/app-store";
import { AttentionHero } from "./AttentionHero";
import { DescentHero } from "./DescentHero";
import { useHeroMotion, useStageSize, type HeroVisualProps } from "./hero-motion";
import { MeaningMapHero } from "./MeaningMapHero";
import { NextWordHero } from "./NextWordHero";
import { heroForThisLaunch, type HeroVariant } from "./pick-hero";

interface HeroDefinition {
  /** The lab this visual previews; the corner button opens it. */
  slug: string;
  tag: string;
  description: string;
  accent: string;
  Visual: ComponentType<HeroVisualProps>;
}

const heroes: Record<HeroVariant, HeroDefinition> = {
  "next-word": {
    slug: "next-token-prediction",
    tag: "next-word prediction",
    description: "A sentence written one word at a time, with the odds of each next word",
    accent: "var(--forward)",
    Visual: NextWordHero,
  },
  attention: {
    slug: "attention",
    tag: "attention · where “it” looks",
    description: "Attention arcs from the word “it” to the other words of a sentence",
    accent: "var(--attention)",
    Visual: AttentionHero,
  },
  "meaning-map": {
    slug: "tokens-embeddings",
    tag: "word vectors · GloVe, 50 dimensions",
    description: "A map of word vectors where words with similar meanings sit close together",
    accent: "var(--positive)",
    Visual: MeaningMapHero,
  },
  descent: {
    slug: "loss-gradient-descent",
    tag: "gradient descent",
    description: "Gradient descent: a dot stepping down a loss curve to its lowest point",
    accent: "var(--loss)",
    Visual: DescentHero,
  },
};

/** The home screen's visual: one of four live previews of a lab, chosen at random each launch. */
export function HomeHero() {
  const openModule = useAppStore((state) => state.openModule);
  const rememberHomeHero = useAppStore((state) => state.rememberHomeHero);
  const [variant] = useState(heroForThisLaunch);
  const stageRef = useRef<HTMLDivElement>(null);
  const size = useStageSize(stageRef);
  const animate = useHeroMotion(stageRef);

  useEffect(() => {
    rememberHomeHero(variant);
  }, [rememberHomeHero, variant]);

  const hero = heroes[variant];
  const lab = modulesBySlug.get(hero.slug);
  const { Visual } = hero;

  return (
    <div
      className={`home-hero home-hero--${variant}`}
      role="group"
      aria-label={hero.description}
      style={{ "--hero-accent": hero.accent } as React.CSSProperties}
    >
      <span className="home-hero__tag">{hero.tag}</span>
      <div className="home-hero__stage" ref={stageRef}>
        <Visual size={size} animate={animate} />
      </div>
      {lab && (
        <button type="button" className="home-hero__lab" onClick={() => openModule(lab.id)}>
          Lab {String(lab.order).padStart(2, "0")} <b>{lab.title}</b>
          <ArrowUpRight aria-hidden="true" />
        </button>
      )}
    </div>
  );
}
