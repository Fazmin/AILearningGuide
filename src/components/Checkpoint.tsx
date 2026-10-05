import { Check, CircleHelp, RotateCcw } from "lucide-react";
import { useMemo, useState } from "react";
import { checkpointQuestions, shuffleCheckpoint } from "@app/module-sdk";
import type { RegisteredModule } from "@app/modules/registry";
import { checkpointTally } from "@app/store/checkpoint-progress";
import { useAppStore } from "@app/store/app-store";

export function Checkpoint({ module }: { module: RegisteredModule }) {
  const questions = useMemo(() => checkpointQuestions(module), [module]);
  const shuffled = useMemo(
    () => questions.map((question, index) => shuffleCheckpoint(question, `${module.id}:${index}`)),
    [module.id, questions],
  );
  const results = useAppStore((state) => state.progress[module.id]?.checkpointResults);
  const markCheckpoint = useAppStore((state) => state.markCheckpoint);
  const resetCheckpoint = useAppStore((state) => state.resetCheckpoint);
  // The option picked during this visit. The store keeps only whether the first answer was right.
  const [chosen, setChosen] = useState<Array<number | null>>(() => questions.map(() => null));
  const tally = checkpointTally(results, questions.length);
  const many = questions.length > 1;

  return (
    <section className="checkpoint">
      <div className="checkpoint__heading">
        <CircleHelp />
        <div>
          <strong>Check your model</strong>
          <span>{many ? `${questions.length} questions before you move on` : "One question before you move on"}</span>
        </div>
      </div>
      {questions.map((question, questionIndex) => {
        const { options, answer } = shuffled[questionIndex];
        const pick = chosen[questionIndex] ?? null;
        const earlier = results?.[questionIndex] ?? null;
        const revealed = pick !== null || earlier !== null;
        const isCorrect = pick === answer;
        const promptId = `${module.id}-checkpoint-${questionIndex}`;
        return (
          <div className="checkpoint__question" key={`${module.id}-${questionIndex}`}>
            {many && <small className="checkpoint__count">Question {questionIndex + 1} of {questions.length}</small>}
            <p id={promptId}>{question.prompt}</p>
            <div className="checkpoint__options" role="group" aria-labelledby={promptId}>
              {options.map((option, index) => (
                <button
                  type="button"
                  key={option}
                  className={
                    !revealed
                      ? ""
                      : index === answer
                        ? "is-correct"
                        : pick === index
                          ? "is-wrong"
                          : ""
                  }
                  onClick={() => {
                    setChosen((current) => current.map((item, position) => (position === questionIndex ? index : item)));
                    markCheckpoint(module.id, questionIndex, index === answer, questions.length);
                  }}
                >
                  <i>{String.fromCharCode(65 + index)}</i>
                  <span>{option}</span>
                  {revealed && index === answer && <Check />}
                </button>
              ))}
            </div>
            {revealed && (
              <div
                className={`checkpoint__feedback ${pick !== null ? (isCorrect ? "is-correct" : "") : earlier ? "is-correct" : ""}`}
                role="status"
              >
                <strong>
                  {pick !== null
                    ? isCorrect
                      ? "That’s it."
                      : "Not quite yet."
                    : earlier
                      ? "You got this one on the first try."
                      : "You missed this one the first time."}
                </strong>
                <span>{question.explanation}</span>
              </div>
            )}
          </div>
        );
      })}
      {many && tally.answered > 0 && (
        <div className="checkpoint__summary" role="status">
          <strong>
            {tally.correct} of {tally.total} correct on the first try
          </strong>
          {tally.answered === tally.total && (
            <button
              type="button"
              onClick={() => {
                resetCheckpoint(module.id);
                setChosen(questions.map(() => null));
              }}
            >
              <RotateCcw /> Try again
            </button>
          )}
        </div>
      )}
    </section>
  );
}
