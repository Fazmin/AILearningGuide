import { Metric, RangeControl, SegmentedControl, Stage3D, type ModuleContext, type StageView } from "@app/module-sdk";
import { WORDS, fmt } from "./math";
import { closestPair3, directionsLeft, squeeze3, vec3Of } from "./space";
import { RANGES, type LabState } from "./state";

type SetState = ModuleContext["setState"];

const loadScene = () => import("./space-scene");
/** From the front right and a little above: size runs right, alive up, fast toward the reader. */
const VIEW: StageView = { azimuth: 32, elevation: 20, distance: 6.2, target: [0, 0.05, 0] };

const triple = (point: readonly [number, number, number]) => `(${fmt(point[0])}, ${fmt(point[1])}, ${fmt(point[2])})`;
const SHAPES = ["", "a line", "a flat sheet", "a cloud that fills space"] as const;

/**
 * The rank card with three numbers per word. Multiplying a direction by zero presses the cloud of
 * words from space onto a sheet, and a second zero presses the sheet onto a line. The number of
 * directions left is the matrix's rank.
 */
export function ForgetSpace({ s, setState }: { s: LabState; setState: SetState }) {
  const trying = s.undo === "stretch";
  const canUndo = s.keep > 1e-9 && s.keepFast > 1e-9;
  const restored = trying && canUndo;
  const keepAlive = restored ? 1 : s.keep;
  const keepFast = restored ? 1 : s.keepFast;
  const left = directionsLeft(keepAlive, keepFast);
  const closest = closestPair3(keepAlive, keepFast);
  const [one, two] = closest.pair;
  const spot = squeeze3(vec3Of(one), keepAlive, keepFast);
  const volume = keepAlive * keepFast;
  const gone = [s.keep <= 1e-9 ? "alive" : "", s.keepFast <= 1e-9 ? "fast" : ""].filter(Boolean);
  const message = restored
    ? `Multiplying alive by 1 ÷ ${fmt(s.keep, 2)} and fast by 1 ÷ ${fmt(s.keepFast, 2)} puts every word back. Nothing was lost.`
    : trying
      ? `Stretching ${gone.join(" and ")} back would need 1 ÷ 0, and there is no such number: 0 times anything is still 0.`
      : left === 1
        ? `Only size is left, so the words sit on a line. ${one.id} and ${two.id} have the same size, so they are now the same list, ${triple(spot)}.`
        : left === 2
          ? `The cloud is pressed onto a flat sheet: every ${gone[0]} number is 0 and cannot be read back. ${
              closest.merged
                ? `${one.id} and ${two.id} now have the same list, ${triple(spot)}.`
                : `All 8 words still have their own spot, because the two numbers that are left still tell them apart.`
            }`
          : s.keep < 1 || s.keepFast < 1
            ? "The cloud is thinner but it still fills space: all three directions are left, so a stretch could undo this."
            : "Nothing has moved yet. Each word is a list of three numbers, a point in space. Slide left to squeeze.";

  const cells = [
    [1, 0, 0],
    [0, s.keep, 0],
    [0, 0, s.keepFast],
  ];

  return (
    <>
      <div className="vm-space">
        <Stage3D
          scene={loadScene}
          props={{
            words: WORDS.map((word) => ({
              id: word.id,
              // Two words on one spot share one label, so neither name hides the other.
              label: !closest.merged || (word.id !== one.id && word.id !== two.id) ? word.id : word.id === one.id ? `${one.id} = ${two.id}` : "",
              at: vec3Of(word),
              flagged: closest.merged && (word.id === one.id || word.id === two.id),
            })),
            keepAlive,
            keepFast,
          }}
          view={VIEW}
          aspect={0.6}
          label={`Eight words as points in space, each a list of three numbers: size, alive and fast. Every alive number is multiplied by ${fmt(
            s.keep,
            2,
          )} and every fast number by ${fmt(s.keepFast, 2)}${restored ? ", and then both are stretched back" : ""}. The words now form ${
            SHAPES[left]
          }: ${left} of 3 directions are left. The shaded box, which began as a cube with volume 1, has volume ${fmt(volume, 2)}. ${
            closest.distinct
          } of 8 words have their own spot. ${message}`}
        />
        <div className="vm-side vm-space__side">
          <div className="vm-target">
            <RangeControl
              label="Multiply every alive number by"
              min={RANGES.keep[0]}
              max={RANGES.keep[1]}
              step={0.05}
              value={s.keep}
              format={(v) => `× ${fmt(v, 2)}`}
              onChange={(value) => setState({ keep: value })}
            />
            <RangeControl
              label="Multiply every fast number by"
              min={RANGES.keepFast[0]}
              max={RANGES.keepFast[1]}
              step={0.05}
              value={s.keepFast}
              format={(v) => `× ${fmt(v, 2)}`}
              onChange={(value) => setState({ keepFast: value })}
            />
          </div>
          <figure
            className="vm-box vm-box--three"
            aria-label={`Matrix with three rows: 1, 0, 0; then 0, ${fmt(s.keep, 2)}, 0; then 0, 0, ${fmt(s.keepFast, 2)}`}
          >
            <div className="vm-box__grid" aria-hidden="true">
              {cells.flat().map((value, index) => (
                <span key={index} className={index % 3 === 0 ? "is-size" : index % 3 === 1 ? "is-alive" : "is-fast"}>
                  {fmt(value, Number.isInteger(value * 10) ? 1 : 2)}
                </span>
              ))}
            </div>
            <figcaption>the squeezing matrix: 3 rows, 3 columns</figcaption>
          </figure>
          <SegmentedControl
            label="Then"
            value={s.undo}
            options={[
              { value: "leave", label: "leave it" },
              { value: "stretch", label: "stretch it back" },
            ]}
            onChange={(value) => setState({ undo: value })}
          />
          <div className="metric-row vm-metrics">
            <Metric label="Directions left" value={`${left} of 3`} tone={left < 3 ? "loss" : undefined} />
            <Metric label="Volume of the shaded box" value={fmt(volume, 2)} tone={volume < 1e-9 ? "loss" : undefined} />
          </div>
          <div className="metric-row vm-metrics">
            <Metric
              label="Words with their own spot"
              value={`${closest.distinct} of 8`}
              tone={closest.distinct < 8 ? "loss" : undefined}
            />
            <Metric
              label="Closest pair"
              value={closest.merged ? `${one.id} = ${two.id}` : `${one.id}, ${two.id}: ${fmt(closest.gap, 2)}`}
              tone={closest.merged ? "loss" : undefined}
            />
          </div>
        </div>
      </div>
      <p className={`vm-callout ${closest.merged ? "is-loss" : ""}`.trim()} role="status">
        {message}
      </p>
      <p className="vm-note">
        The count of directions left is the matrix&apos;s <b>rank</b>. A table with 3 rows and 3 columns can have rank 3, 2
        or 1; the real tables inside a model have thousands of rows, and the same count says how much they can keep apart.
      </p>
    </>
  );
}
