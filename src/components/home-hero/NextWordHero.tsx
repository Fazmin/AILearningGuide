import { Fragment, useEffect, useState } from "react";
import { useHeroScript, wait, type HeroVisualProps } from "./hero-motion";

type Candidate = readonly [token: string, percent: number];

interface Sentence {
  prompt: string;
  /** One list of candidates per generated token; the first candidate is the one written. */
  steps: readonly (readonly Candidate[])[];
}

// Hand-set odds that read as a short story about how generation works.
const sentences: readonly Sentence[] = [
  {
    prompt: "You learn AI by",
    steps: [
      [["trying", 46], ["reading", 31], ["watching", 12]],
      [["things", 38], ["it", 29], ["examples", 14]],
      [["out", 71], ["yourself", 13], ["together", 6]],
      [[".", 82], [",", 9], ["and", 5]],
    ],
  },
  {
    prompt: "A model writes one word at a",
    steps: [
      [["time", 93], ["glance", 2], ["go", 1]],
      [[".", 88], [",", 6], ["and", 3]],
    ],
  },
  {
    prompt: "Then it reads it all and",
    steps: [
      [["guesses", 41], ["writes", 24], ["starts", 11]],
      [["again", 57], ["the", 19], ["what", 9]],
      [[".", 84], [",", 8], ["until", 4]],
    ],
  },
];

type Phase = "showing" | "picked" | "leaving";

interface View {
  prompt: string;
  generated: string[];
  candidates: readonly Candidate[] | null;
  /** Remounts the candidate list so each step replays its entrance. */
  step: number;
  phase: Phase;
  hidden: boolean;
}

const still: View = {
  prompt: sentences[0].prompt,
  generated: ["trying", "things"],
  candidates: sentences[0].steps[2],
  step: 0,
  phase: "picked",
  hidden: false,
};

const isPunctuation = (token: string) => /^[.,!?;:]$/.test(token);

export function NextWordHero({ animate }: HeroVisualProps) {
  const [view, setView] = useState<View>(still);

  useEffect(() => {
    if (!animate) setView(still);
  }, [animate]);

  useHeroScript(animate, async (live) => {
    for (let index = 0; ; index = (index + 1) % sentences.length) {
      const { prompt, steps } = sentences[index];
      setView((current) => ({ ...current, hidden: true, phase: "leaving" }));
      await wait(380, live);
      setView({ prompt, generated: [], candidates: null, step: 0, phase: "showing", hidden: false });
      await wait(800, live);
      for (const candidates of steps) {
        setView((current) => ({ ...current, candidates, step: current.step + 1, phase: "showing" }));
        await wait(1050, live);
        setView((current) => ({ ...current, phase: "picked" }));
        await wait(380, live);
        setView((current) => ({ ...current, generated: [...current.generated, candidates[0][0]], phase: "leaving" }));
        await wait(480, live);
      }
      await wait(2600, live);
    }
  });

  const newest = view.generated.length - 1;

  return (
    <div className={`hero-next${animate ? "" : " is-still"}`}>
      <p className={`hero-next__line${view.hidden ? " is-hidden" : ""}`}>
        {view.prompt.split(" ").map((word, index) => (
          <Fragment key={`p${index}`}>
            {index > 0 && " "}
            <span className="hero-next__word">{word}</span>
          </Fragment>
        ))}
        {view.generated.map((token, index) => (
          <Fragment key={`g${index}`}>
            {!isPunctuation(token) && " "}
            <span className={`hero-next__word is-generated${animate && index === newest ? " is-new" : ""}`}>{token}</span>
          </Fragment>
        ))}
        <span className="hero-next__caret" aria-hidden="true" />
      </p>
      <div className="hero-next__odds">
        <span className="hero-next__label">next word, by probability</span>
        <ol className={`hero-next__candidates is-${view.phase}`} key={view.step}>
          {view.candidates?.map(([token, percent], index) => (
            <li
              key={token}
              className={index === 0 && view.phase !== "showing" ? "is-picked" : undefined}
              style={{ "--p": `${percent}%`, "--i": index } as React.CSSProperties}
            >
              <span className="hero-next__token">{token}</span>
              <span className="hero-next__bar"><i /></span>
              <span className="hero-next__percent">{percent}%</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
