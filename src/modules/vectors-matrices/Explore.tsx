import { LabSurface, PredictCheck, StepPager, SurfaceHeading, type ModuleContext } from "@app/module-sdk";
import { AimCard, ForgetCard, MatrixCard, OrderCard, PiecesCard } from "./matrix-cards";
import { readState } from "./state";
import { AddCard, AlikeCard, WordCard } from "./vector-cards";

/** Shown at the foot of the lab when the shell does not pass the step names (a test harness). */
const CARD_COUNT = 8;

/**
 * Vectors & matrices: one story, a tiny map of eight words, told in eight cards. Each card adds
 * one idea to the last, and the lab shows one card per step, like pages. Everything the reader
 * set lives in module state, so paging forward and back loses nothing.
 */
export default function Explore({ state, setState, currentStep, narrate, steps, goToStep }: ModuleContext) {
  const s = readState(state);
  const step = Math.min(CARD_COUNT - 1, Math.max(0, Math.floor(Number.isFinite(currentStep) ? currentStep : 0)));
  const card = (index: number) => `vm-card vm-card--${index}`;
  const stepNames = steps ?? Array.from({ length: CARD_COUNT }, (_, index) => `step ${index + 1}`);

  return (
    <div className="vm-shell">
      <div className="vm-lab">
        {step === 0 && (
        <LabSurface label="A word as two numbers" className={card(0)}>
          <SurfaceHeading kicker="1 · A list is an arrow" title="A word as two numbers" />
          <p className="vm-lead">
            Each word is described by two numbers: how big the thing is and how alive it is. Go that far across
            and that far up, and the list becomes an arrow. A list of numbers like this is called a{" "}
            <strong>vector</strong>.
          </p>
          <WordCard s={s} setState={setState} />
        </LabSurface>
        )}

        {step === 1 && (
        <LabSurface label="Add and stretch" className={card(1)}>
          <SurfaceHeading kicker="2 · The two things you can do to an arrow" title="Add and stretch" />
          <p className="vm-lead">
            Add a step to a word by putting the step’s tail on the word’s tip. Multiply the step by a number to
            stretch it, shrink it, or (with a minus sign) flip it.
          </p>
          <AddCard s={s} setState={setState} />
        </LabSurface>
        )}

        {step === 2 && (
        <LabSurface label="How alike are two arrows" className={card(2)}>
          <SurfaceHeading kicker="3 · The dot product and the cosine" title="How alike are two arrows?" />
          <p className="vm-lead">
            Multiply the two sizes, multiply the two alive numbers, and add. That sum is the{" "}
            <strong>dot product</strong>: positive when the arrows agree, negative when they pull opposite ways.
          </p>
          <PredictCheck
            prompt="Two arrows have a positive dot product. You turn one of them half a turn, so it points the opposite way. The dot product will be…"
            options={["the same: no length has changed", "zero: the two arrows cancel", "the same size, but negative"]}
            answer={2}
            picked={s.predictDot}
            onPick={(index) => setState({ predictDot: index })}
            explanation="A half turn flips the sign of both of an arrow’s numbers, so each product flips sign and so does their sum. Set Turn the second arrow to 180° and watch: for horse and whale, +2.88 becomes −2.88. Zero is what you pass on the way, when the two arrows are at right angles."
          />
          <AlikeCard s={s} setState={setState} />
        </LabSurface>
        )}

        {step === 3 && (
        <LabSurface label="A matrix moves every arrow" className={card(3)}>
          <SurfaceHeading kicker="4 · Four numbers that move the whole map" title="A matrix moves every arrow at once" />
          <p className="vm-lead">
            A <strong>matrix</strong> is a small grid of numbers. Its first column says where the size arrow (1, 0)
            lands and its second column says where the alive arrow (0, 1) lands. Every word follows along.
          </p>
          <MatrixCard s={s} setState={setState} narrate={narrate} />
        </LabSurface>
        )}

        {step === 4 && (
        <LabSurface label="One after another" className={card(4)}>
          <SurfaceHeading kicker="5 · Two moves in a row" title="One after another" />
          <p className="vm-lead">
            Do one matrix, then another. The pair can always be replaced by a single matrix that does both, but
            which one you get depends on the order.
          </p>
          <PredictCheck
            prompt="Step 1 turns the map a quarter turn. Step 2 stretches it sideways. If you swap the two steps, where do the words land?"
            options={["In the same places", "In different places", "The same, but upside down"]}
            answer={1}
            picked={s.predictOrder}
            onPick={(index) => setState({ predictOrder: index })}
            explanation="The stretch always pulls sideways, but what lies sideways depends on whether the turn has happened yet. Turn first and the stretch widens the turned map; stretch first and the turn stands the wide map on end. Press Swap the order and count the words that move."
          />
          <OrderCard s={s} setState={setState} narrate={narrate} />
        </LabSurface>
        )}

        {step === 5 && (
        <LabSurface label="When a matrix forgets" className={card(5)}>
          <SurfaceHeading kicker="6 · Squeezing the map onto a line" title="When a matrix forgets" />
          <p className="vm-lead">
            This matrix multiplies every alive number by one factor and leaves size alone. Slide the factor down
            to 0 and the whole map is pressed flat onto the size line.
          </p>
          <PredictCheck
            prompt="At × 0, cat and bicycle land on the same spot. What second matrix would pull them apart again?"
            options={["One that stretches alive", "None: they are now one list", "One that turns the map"]}
            answer={1}
            picked={s.predictUndo}
            onPick={(index) => setState({ predictUndo: index })}
            explanation="At × 0 both words become the list (−0.8, 0.0). A matrix gives one output for one input, so two identical lists can never come out different. At × 0.05 the gap is tiny but still there, and stretching by 20 brings it back. Choose “stretch it back” at both settings and compare."
          />
          <ForgetCard s={s} setState={setState} />
        </LabSurface>
        )}

        {step === 6 && (
        <LabSurface label="A table from a few pieces" className={card(6)}>
          <SurfaceHeading kicker="7 · The flip side: simple tables are cheap" title="A table from a few pieces" />
          <p className="vm-lead">
            A bigger matrix: 8 rows by 8 columns, 64 numbers. This one was built by adding four simple pieces,
            each made from one column times one row. Keep fewer pieces and you store fewer numbers.
          </p>
          <PredictCheck
            prompt="One piece is a column of 8 numbers times a row of 8 numbers. How many numbers must you store to keep 2 pieces?"
            options={["16", "32", "128"]}
            answer={1}
            picked={s.predictPieces}
            onPick={(index) => setState({ predictPieces: index })}
            explanation="A piece costs 8 + 8 = 16 numbers, not 8 × 8 = 64, because every cell of a piece is worked out from its column and its row. Two pieces cost 32, half of the full table’s 64. Set Pieces kept to 2 and read Numbers stored."
          />
          <PiecesCard s={s} setState={setState} />
        </LabSurface>
        )}

        {step === 7 && (
        <LabSurface label="Challenge: aim the map" className={card(7)}>
          <SurfaceHeading kicker="8 · Put it together" title="Challenge: aim the map" />
          <p className="vm-lead">
            The dashed hollow marks show where each word should land. Set the four numbers of the matrix until
            every word sits on its target. Three rounds, each a little harder.
          </p>
          <AimCard s={s} setState={setState} narrate={narrate} />
        </LabSurface>
        )}
        <StepPager steps={stepNames} current={step} goToStep={goToStep} />
      </div>
    </div>
  );
}
