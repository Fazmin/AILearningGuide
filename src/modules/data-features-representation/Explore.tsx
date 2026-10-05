import { useEffect, useMemo, useRef } from "react";
import {
  LabSurface,
  Metric,
  RangeControl,
  SegmentedControl,
  SurfaceHeading,
  VectorChip,
  type ModuleContext,
} from "@app/module-sdk";
import {
  AXES,
  DEFAULT_CODES,
  EMBEDDED,
  ENCODINGS,
  MAX_COLOR_CODE,
  MAX_ROOMS,
  ROWS,
  axisValue,
  encodeRow,
  euclid,
  featureNames,
  nearestTies,
  neighborOrder,
  planeNeighbor,
  type Apartment,
  type ColorName,
  type Encoding,
  type FeatureAxis,
} from "./encode";
import {
  CLEAN,
  HANDLINGS,
  K,
  MISLABEL_ROW,
  MISSING_ROW,
  ONE_HOT_GAP,
  QUERY,
  SENTENCES,
  SENTENCE_IDS,
  VOCABULARY,
  distanceToQuery,
  encodeSentence,
  idGap,
  leaveOneOut,
  messyRows,
  nearestAnswer,
  parsePixels,
  pixelVector,
  summarize,
  templateDistances,
  togglePixel,
  type Handling,
  type Mess,
} from "./preview";

const asString = (state: ModuleContext["state"], key: string, fallback: string) =>
  typeof state[key] === "string" ? state[key] : fallback;
const asNumber = (state: ModuleContext["state"], key: string, fallback: number) =>
  typeof state[key] === "number" && Number.isFinite(state[key]) ? state[key] : fallback;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const pick = <T extends string>(value: string, allowed: readonly T[], fallback: T): T =>
  (allowed as readonly string[]).includes(value) ? (value as T) : fallback;

const COLOR_LABEL: Record<ColorName, string> = {
  red: "red door",
  blue: "blue door",
  green: "green door",
};

function HouseMark({
  row,
  selected = false,
  miss = false,
  compact = false,
  onClick,
}: {
  row: Apartment;
  selected?: boolean;
  miss?: boolean;
  compact?: boolean;
  onClick?: () => void;
}) {
  const windows = Array.from({ length: row.rooms }, (_, index) => index);
  const label = [
    row.name,
    `${row.rooms} room${row.rooms === 1 ? "" : "s"}`,
    COLOR_LABEL[row.color],
    row.park === "yes" ? "near a park" : "no park",
    row.price ? "high price tag" : "low price tag",
  ].join(", ");

  return (
    <button
      type="button"
      className={`dfr-house is-${row.color}${selected ? " is-selected" : ""}${miss ? " is-miss" : ""}${compact ? " is-compact" : ""}`}
      aria-pressed={selected}
      aria-label={label}
      onClick={onClick}
    >
      <svg viewBox="0 0 88 92" aria-hidden="true">
        <path className="dfr-house__roof" d="M10 36 L44 10 L78 36 Z" />
        <rect className="dfr-house__wall" x="16" y="34" width="56" height="48" />
        {windows.map((index) => {
          const col = index % 2;
          const rowIndex = Math.floor(index / 2);
          return (
            <rect
              key={index}
              className="dfr-house__window"
              x={22 + col * 20}
              y={40 + rowIndex * 16}
              width="12"
              height="11"
            />
          );
        })}
        <rect className="dfr-house__door" x="35" y="56" width="18" height="26" />
        {row.color === "blue" && (
          <>
            <line className="dfr-house__hatch" x1="38" x2="50" y1="64" y2="64" />
            <line className="dfr-house__hatch" x1="38" x2="50" y1="71" y2="71" />
          </>
        )}
        {row.color === "green" && (
          <>
            <circle className="dfr-house__dot" cx="40" cy="66" r="2" />
            <circle className="dfr-house__dot" cx="48" cy="74" r="2" />
          </>
        )}
        {row.park === "yes" && (
          <g className="dfr-house__tree">
            <rect x="70" y="64" width="4" height="18" />
            <circle cx="72" cy="58" r="11" />
          </g>
        )}
      </svg>
      <strong>{row.name.split(" ")[0]}</strong>
      <span>
        {row.rooms} room{row.rooms === 1 ? "" : "s"} · {row.color}
        {row.park === "yes" ? " · park" : ""}
      </span>
      <b className={`dfr-price${row.price ? " is-high" : ""}`}>{row.price ? "high" : "low"}</b>
    </button>
  );
}

function StampPad({
  color,
  value,
  disabled,
  onChange,
}: {
  color: ColorName;
  value: number;
  disabled: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <fieldset className={`dfr-stamp is-${color}${disabled ? " is-idle" : ""}`} disabled={disabled}>
      <legend>{COLOR_LABEL[color]}</legend>
      <div>
        {Array.from({ length: MAX_COLOR_CODE }, (_, index) => {
          const n = index + 1;
          return (
            <button
              type="button"
              key={n}
              aria-pressed={value === n}
              aria-label={`Stamp ${color} as ${n}`}
              onClick={() => onChange(n)}
            >
              {n}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

function PackingSlip({
  row,
  encoding,
  values,
  names,
}: {
  row: Apartment;
  encoding: Encoding;
  values: readonly number[];
  names: readonly string[];
}) {
  const max = Math.max(1, ...values.map((value) => Math.abs(value)));
  return (
    <div className="dfr-slip" aria-label={`Packing slip for ${row.name}`}>
      <header>
        <span>Packing slip</span>
        <strong>{row.name}</strong>
      </header>
      <ol>
        {names.map((name, index) => {
          const value = values[index] ?? 0;
          return (
            <li key={name}>
              <span>{name}</span>
              <i style={{ width: `${(Math.abs(value) / max) * 100}%` }} />
              <code>{value.toFixed(encoding === "normalized" ? 2 : 0)}</code>
            </li>
          );
        })}
      </ol>
      <footer>High price stays off the slip. It is the tag, not a coordinate.</footer>
    </div>
  );
}

export default function Explore({ state, setState, narrate }: ModuleContext) {
  const encoding = pick(asString(state, "encoding", "integer"), ENCODINGS, "integer");
  const featureAxis = pick(asString(state, "featureAxis", "rooms"), AXES, "rooms");
  const redCode = clamp(Math.round(asNumber(state, "redCode", DEFAULT_CODES.red)), 1, MAX_COLOR_CODE);
  const blueCode = clamp(Math.round(asNumber(state, "blueCode", DEFAULT_CODES.blue)), 1, MAX_COLOR_CODE);
  const greenCode = clamp(Math.round(asNumber(state, "greenCode", DEFAULT_CODES.green)), 1, MAX_COLOR_CODE);
  const selected = clamp(Math.round(asNumber(state, "selected", 0)), 0, ROWS.length - 1);
  const threshold = clamp(asNumber(state, "threshold", 2.5), 0, 5);
  const codes = { red: redCode, blue: blueCode, green: greenCode };
  const names = featureNames(encoding);
  const stampsIdle = encoding === "onehot";

  const packed = useMemo(
    () => ROWS.map((row) => ({ row, features: encodeRow(row, encoding, codes) })),
    [blueCode, encoding, greenCode, redCode],
  );
  const current = packed[selected];
  const ranks = useMemo(() => neighborOrder(packed.map((item) => item.features), selected), [packed, selected]);
  const planeRanks = useMemo(() => planeNeighbor(packed.map((item) => item.features), selected), [packed, selected]);
  const nearest = packed[ranks[0]?.id ?? 0];
  const ties = nearestTies(ranks);
  const tied = ties.length > 1;
  const nearestLabel = ties.map((rank) => packed[rank.id].row.name).join(" = ");
  const planeNearest = packed[planeRanks[0]?.id ?? 0];
  const projectionDisagrees = !ties.some((rank) => rank.id === planeNearest.row.id);

  const lastNearest = useRef<number | null>(null);

  useEffect(() => {
    const id = nearest.row.id;
    if (lastNearest.current !== null && lastNearest.current !== id) {
      narrate(`${nearest.row.name} is now nearest to ${current.row.name} under this encoding.`);
    }
    lastNearest.current = id;
  }, [current.row.name, narrate, nearest.row.id, nearest.row.name]);

  const deltas = names.map((name, index) => ({
    name,
    value: Math.abs((current.features[index] ?? 0) - (nearest.features[index] ?? 0)),
  }));
  const deltaMax = Math.max(0.001, ...deltas.map((item) => item.value));

  const cuts = packed.map((item) => {
    const value = axisValue(item.row, featureAxis, encoding, codes);
    const predicted = value >= threshold ? 1 : 0;
    return { ...item, value, predicted, hit: predicted === item.row.price };
  });
  const matched = cuts.filter((item) => item.hit).length;
  const lowYard = cuts.filter((item) => item.predicted === 0);
  const highYard = cuts.filter((item) => item.predicted === 1);

  const embedRanks = useMemo(() => {
    const here = EMBEDDED[selected];
    return ROWS.filter((row) => row.id !== selected)
      .map((row) => ({
        row,
        embed: euclid(EMBEDDED[row.id], here),
        color: euclid(
          encodeRow(row, "onehot", codes).slice(1, 4),
          encodeRow(current.row, "onehot", codes).slice(1, 4),
        ),
      }))
      .sort((left, right) => left.embed - right.embed || left.row.id - right.row.id);
  }, [codes, current.row, selected]);
  const embedNear = embedRanks[0];
  const colorNear = [...embedRanks].sort((left, right) => left.color - right.color || left.row.id - right.row.id)[0];

  const axisNote =
    featureAxis === "rooms"
      ? encoding === "normalized"
        ? "The gate reads rooms/4. Same rule family, a shorter ruler."
        : "The gate reads the raw room count. Same rule as the other stamps: is this number big enough?"
      : featureAxis === "park"
        ? "Park is already 0 or 1. A threshold between them is the only cut that splits the street."
        : encoding === "onehot"
          ? "One-hot color gives the gate only the red bit. Blue and green both read 0, so it cannot tell them apart."
          : "The gate treats your color stamps as amounts. Blue between red and green is an order you wrote.";

  const pixels = parsePixels(state.pixels);
  const pixelValues = pixelVector(pixels);
  const templateRanks = templateDistances(pixels);
  const sentenceId = pick(asString(state, "sentence", "cat-mat"), SENTENCE_IDS, "cat-mat");
  const sentence = SENTENCES.find((item) => item.id === sentenceId) ?? SENTENCES[0];
  const tokens = encodeSentence(sentence.text);
  const mess: Mess = {
    missing: state.messMissing === true,
    duplicate: state.messDuplicate === true,
    mislabel: state.messMislabel === true,
    handling: pick(asString(state, "missingHandling", "zero"), HANDLINGS, "zero") as Handling,
  };
  const messRows = messyRows(mess);
  const messSummary = summarize(messRows);
  const messAnswer = nearestAnswer(messRows);
  const messLoo = leaveOneOut(messRows);
  const cleanRows = messyRows(CLEAN);
  const cleanSummary = summarize(cleanRows);
  const cleanAnswer = nearestAnswer(cleanRows);
  const cleanLoo = leaveOneOut(cleanRows);
  const answerWord = (answer: number) => (answer === 1 ? "high" : "low");
  const messActive = mess.missing || mess.duplicate || mess.mislabel;
  const messNotes: string[] = [];
  if (mess.missing) {
    const target = messRows[MISSING_ROW];
    messNotes.push(
      mess.handling === "zero"
        ? `Missing value, read as 0: ${target.name} now has 0 rooms, which moves it from ${distanceToQuery(cleanRows[MISSING_ROW]).toFixed(2)} to ${distanceToQuery(target).toFixed(2)} away from the new listing.`
        : mess.handling === "drop"
          ? `Missing value, row dropped: the table loses ${target.name}, which was the nearest listing to the new one.`
          : `Missing value, filled with the mean: ${target.name} gets ${target.rooms.toFixed(1)} rooms, the mean of the other five, and sits ${distanceToQuery(target).toFixed(2)} from the new listing.`,
    );
  }
  if (mess.duplicate) {
    messNotes.push(
      "Duplicate row: Alder 12 is listed twice, so it counts twice in the summary, and each copy's nearest other row is its twin at distance 0.",
    );
  }
  if (mess.mislabel) {
    messNotes.push(
      `Mislabelled row: ${cleanRows[MISLABEL_ROW].name}'s tag is flipped from ${answerWord(cleanRows[MISLABEL_ROW].price)} to ${answerWord(messRows[MISLABEL_ROW].price)}, so the listing nearest to the new one now votes the other way.`,
    );
  }

  const encodingNote =
    encoding === "onehot"
      ? "Each color gets its own slot. Distinct doors are orthogonal — distance √2 — and the stamps never enter the slip."
      : encoding === "normalized"
        ? `Rooms are divided by ${MAX_ROOMS} and color stamps by ${MAX_COLOR_CODE}. Scaling is not a new measurement.`
        : "Integer stamps invent an order. If blue sits between red and green, the lineup treats that gap as real.";

  return (
    <div className="gw-shell">
    <div className="tg-lab tg-lab--hero gw-lab dfr-lab">
      <LabSurface label="The street" className="dfr-street-card">
        <SurfaceHeading
          kicker="Six listings · rooms, door color, park, and a price tag"
          title="The buildings did not move. Only the numbers you write for them will."
        />
        <div className="dfr-street">
          {ROWS.map((row) => (
            <HouseMark
              key={row.id}
              row={row}
              selected={row.id === selected}
              onClick={() => setState({ selected: row.id })}
            />
          ))}
          <i className="dfr-street__curb" aria-hidden="true" />
        </div>
        <p className="lab-note">
          Windows are rooms. The door paint is a category, not an amount. The tree is a yes/no park.
          The high/low pill is the label y — the answer — and it is not a feature unless you leak it.
          The selected house stays marked on the slip and the lineup.
        </p>
      </LabSurface>

      <LabSurface label="Packing slip" className="dfr-bench-card">
        <SurfaceHeading
          kicker={`${current.features.length} numbers standing in for ${current.row.name}`}
          title="You choose how a door color becomes a coordinate"
        />
        <div className="dfr-bench">
          <div className="dfr-bench__house">
            <HouseMark row={current.row} selected onClick={() => setState({ selected: current.row.id })} />
          </div>
          <div className="dfr-encoding encoding-control">
            <SegmentedControl
              label="How to pack the door"
              value={encoding}
              options={[
                { value: "integer", label: "integer stamps" },
                { value: "onehot", label: "one cubby each" },
                { value: "normalized", label: "same-length rulers" },
              ]}
              onChange={(value) => setState({ encoding: value })}
            />
          </div>
          <div className="dfr-stamps">
            <StampPad color="red" value={redCode} disabled={stampsIdle} onChange={(value) => setState({ redCode: value })} />
            <StampPad color="blue" value={blueCode} disabled={stampsIdle} onChange={(value) => setState({ blueCode: value })} />
            <StampPad color="green" value={greenCode} disabled={stampsIdle} onChange={(value) => setState({ greenCode: value })} />
          </div>
          <PackingSlip row={current.row} encoding={encoding} values={current.features} names={names} />
        </div>
        <p className="lab-note">{encodingNote}</p>
      </LabSurface>

      <LabSurface label="Who stands nearest" className="dfr-queue-card">
        <SurfaceHeading
          kicker={`Lined up by slip distance from ${current.row.name}`}
          title="A neighbor flip is the apartments staying put and the packing changing"
        />
        <ol className="dfr-queue">
          <li className="is-self">
            <HouseMark row={current.row} selected compact onClick={() => setState({ selected: current.row.id })} />
            <span>
              <b>selected</b>
              d 0.00
            </span>
          </li>
          {ranks.map((rank, index) => {
            const item = packed[rank.id];
            const near = index < ties.length;
            return (
              <li key={rank.id} className={near ? "is-near" : ""}>
                <HouseMark
                  row={item.row}
                  compact
                  selected={false}
                  onClick={() => setState({ selected: item.row.id })}
                />
                <span>
                  <b>{near ? (tied ? "tied nearest" : "nearest") : `#${index + 1}`}</b>
                  d {rank.dist.toFixed(2)}
                </span>
              </li>
            );
          })}
        </ol>
        <div className="dfr-deltas" aria-label={`Why ${nearest.row.name} is nearest`}>
          <span>
            Why {nearest.row.name} sits closest{tied ? ` (tied with ${ties
              .slice(1)
              .map((rank) => packed[rank.id].row.name)
              .join(", ")})` : ""} — each bar is |Δ| on one slip slot
          </span>
          {deltas.map((item) => (
            <div key={item.name}>
              <strong>{item.name}</strong>
              <i style={{ width: `${(item.value / deltaMax) * 100}%` }} />
              <code>{item.value.toFixed(2)}</code>
            </div>
          ))}
        </div>
        {projectionDisagrees && (
          <p className="dfr-callout">
            If you only plotted the first two slip numbers, {planeNearest.row.name} would look closer.
            The extra slots disagree. The plane is a projection, not the whole representation.
          </p>
        )}
        <div className="metric-row">
          <Metric label={tied ? "Nearest on the slip · tie" : "Nearest on the slip"} value={nearestLabel} tone="forward" />
          <Metric label="Distance" value={ranks[0]?.dist.toFixed(2) ?? "—"} />
          <Metric label="First-two nearest" value={planeNearest.row.name} />
        </div>
      </LabSurface>

      <LabSurface label="The gate" className="dfr-gate-card">
        <SurfaceHeading
          kicker="ŷ = 1[one stamp ≥ t] · the buildings walk, the rule does not"
          title="Same threshold family, a different cut when the stamp changes"
        />
        <div className="dfr-gate-control">
          <SegmentedControl
            label="Stamp the gate reads"
            value={featureAxis}
            options={[
              { value: "rooms", label: "rooms" },
              { value: "color", label: "door color" },
              { value: "park", label: "park" },
            ]}
            onChange={(value) => setState({ featureAxis: value })}
          />
          <RangeControl
            label="Gate height t"
            min={0}
            max={5}
            step={0.1}
            value={threshold}
            format={(value) => value.toFixed(1)}
            onChange={(value) => setState({ threshold: value })}
          />
        </div>
        <div
          className="dfr-yards"
          role="img"
          aria-label={`Gate at ${threshold.toFixed(1)} on ${featureAxis}. ${matched} of 6 tags match.`}
        >
          <section>
            <header>Predicted low</header>
            <div>
              {lowYard.map((item) => (
                <HouseMark
                  key={item.row.id}
                  row={item.row}
                  compact
                  selected={item.row.id === selected}
                  miss={!item.hit}
                  onClick={() => setState({ selected: item.row.id })}
                />
              ))}
              {lowYard.length === 0 && <p>Empty yard</p>}
            </div>
          </section>
          <div className="dfr-fence">
            <b />
            <span>t = {threshold.toFixed(1)}</span>
            <small>1[{featureAxis} ≥ t]</small>
          </div>
          <section>
            <header>Predicted high</header>
            <div>
              {highYard.map((item) => (
                <HouseMark
                  key={item.row.id}
                  row={item.row}
                  compact
                  selected={item.row.id === selected}
                  miss={!item.hit}
                  onClick={() => setState({ selected: item.row.id })}
                />
              ))}
              {highYard.length === 0 && <p>Empty yard</p>}
            </div>
          </section>
        </div>
        <div className="metric-row">
          <Metric label="Tags the gate matches" value={`${matched} of 6`} tone={matched < 5 ? "loss" : "forward"} />
          <Metric label={`${current.row.name} ŷ`} value={cuts[selected]?.predicted ? "high" : "low"} />
        </div>
        <p className="lab-note">
          A struck-through pill is a miss: the price tag and the yard disagree. {axisNote} Accuracy is on
          these six authored rows, not a holdout.
        </p>
      </LabSurface>

      <LabSurface label="Two maps" className="dfr-maps-card">
        <SurfaceHeading
          kicker="Color cubbies versus a later geometry"
          title="Nearby means a map you chose, or a map a loss might later fit"
        />
        <div className="dfr-maps">
          <section className="dfr-docks">
            <header>
              <span>Color cubbies · one-hot</span>
              <strong>Every other color is equally far (√2)</strong>
            </header>
            <div>
              {(["red", "blue", "green"] as const).map((color) => (
                <div key={color} className={`dfr-dock is-${color}`}>
                  <span>{color}</span>
                  {ROWS.filter((row) => row.color === color).map((row) => (
                    <HouseMark
                      key={row.id}
                      row={row}
                      compact
                      selected={row.id === selected}
                      onClick={() => setState({ selected: row.id })}
                    />
                  ))}
                </div>
              ))}
            </div>
          </section>
          <section className="dfr-lot">
            <header>
              <span>A later map · authored</span>
              <strong>Park-and-price neighbors were placed nearby by hand</strong>
            </header>
            <div className="dfr-lot__plot">
              <div className="dfr-lot__field">
                {embedNear && (
                  <svg className="dfr-lot__link" viewBox="0 0 1 1" preserveAspectRatio="none">
                    <line
                      x1={EMBEDDED[selected][0]}
                      y1={1 - EMBEDDED[selected][1]}
                      x2={EMBEDDED[embedNear.row.id][0]}
                      y2={1 - EMBEDDED[embedNear.row.id][1]}
                    />
                  </svg>
                )}
                {ROWS.map((row) => {
                  const [x, y] = EMBEDDED[row.id];
                  return (
                    <div
                      key={row.id}
                      className="dfr-lot__pin"
                      style={{ left: `${x * 100}%`, top: `${(1 - y) * 100}%` }}
                    >
                      <HouseMark
                        row={row}
                        compact
                        selected={row.id === selected}
                        onClick={() => setState({ selected: row.id })}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          </section>
        </div>
        <div className="metric-row">
          <Metric label="Nearest on the later map" value={embedNear?.row.name ?? "—"} tone="forward" />
          <Metric label="Embed dist" value={embedNear?.embed.toFixed(2) ?? "—"} />
          <Metric label="One-hot color nearest" value={colorNear?.row.name ?? "—"} />
        </div>
        <p className="lab-note">
          One-hot color distance is 0 for the same door and √2 otherwise. It ignores rooms, park, and
          the tag. The map on the right is authored so similar park-and-price listings sit closer — a
          geometry a later model might fit, not a measurement from this street. Tokens & embeddings
          will learn coordinates instead of writing them.
        </p>
      </LabSurface>

      <LabSurface label="Pixels and words" className="dfr-numbers-card">
        <SurfaceHeading
          kicker="Pictures and text are numbers too"
          title="A 3×3 picture is nine numbers; a sentence is a list of IDs"
        />
        <div className="dfr-pixel-control">
          <div
            className="pixel-pad dfr-pixel-pad"
            role="group"
            aria-label="3 by 3 picture. Each cell is one pixel, value 1 for ink or 0 for blank, read row by row."
          >
            {Array.from(pixels).map((char, index) => (
              <button
                key={index}
                type="button"
                className={char === "1" ? "is-on" : ""}
                aria-pressed={char === "1"}
                aria-label={`Row ${Math.floor(index / 3) + 1}, column ${(index % 3) + 1}, value ${char}`}
                onClick={() => setState({ pixels: togglePixel(pixels, index) })}
              >
                <b aria-hidden="true">{char}</b>
              </button>
            ))}
          </div>
          <VectorChip label="Picture as nine numbers, row by row" values={pixelValues} precision={0} tone="forward" />
        </div>
        <table className="tg-board fit-table" aria-label="Distance from the drawing to each template picture">
          <thead>
            <tr>
              <th scope="col">Template</th>
              <th scope="col">Pixels, row by row</th>
              <th scope="col">Distance</th>
              <th scope="col">Rank</th>
            </tr>
          </thead>
          <tbody>
            {templateRanks.map((item, index) => (
              <tr key={item.id} className={index === 0 ? "is-leader" : ""}>
                <th scope="row">{item.label}</th>
                <td>{`${item.pixels.slice(0, 3)} ${item.pixels.slice(3, 6)} ${item.pixels.slice(6)}`}</td>
                <td>{item.distance.toFixed(2)}</td>
                <td>{index === 0 ? "nearest" : `#${index + 1}`}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="dfr-sentence-control">
          <SegmentedControl
            label="Sentence"
            value={sentenceId}
            options={SENTENCES.map((item) => ({ value: item.id, label: item.text }))}
            onChange={(value) => {
              setState({ sentence: value });
              narrate(
                `Sentence set to ${SENTENCES.find((item) => item.id === value)?.text ?? value}.`,
              );
            }}
          />
        </div>
        <table className="tg-board fit-table" aria-label="Each word of the sentence and its ID in the fixed vocabulary">
          <thead>
            <tr>
              <th scope="col">Position</th>
              <th scope="col">Word</th>
              <th scope="col">ID</th>
              <th scope="col">In the list?</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((token, index) => (
              <tr key={`${token.word}-${index}`} className={token.known ? "" : "is-leader"}>
                <td>{index + 1}</td>
                <th scope="row">{token.word}</th>
                <td>{token.id}</td>
                <td>{token.known ? "yes" : "no, so ID 0 (<unk>)"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="metric-row">
          <Metric label="Nearest template" value={templateRanks[0].label} tone="forward" />
          <Metric label="Distance to it" value={templateRanks[0].distance.toFixed(2)} />
          <Metric label="Sentence as IDs" value={`[${tokens.map((token) => token.id).join(", ")}]`} />
          <Metric label="ID gap cat–dog" value={`${idGap("cat", "dog")}`} />
          <Metric label="ID gap cat–mat" value={`${idGap("cat", "mat")}`} />
          <Metric label="One-hot gap, any two words" value={ONE_HOT_GAP.toFixed(2)} />
        </div>
        <p className="lab-note">
          Each pixel is 0 or 1, so the squared distance between two pictures is the number of pixels where they
          differ, and one flipped pixel changes it by exactly 1. The vocabulary is fixed here:{" "}
          {VOCABULARY.map((word, id) => `${word} = ${id}`).join(", ")}. A word that is not on the list becomes ID 0. IDs
          are labels, not amounts: by subtraction cat is nearer to mat than to dog, an order the numbering invented.
          This eight-word list is not a real tokenizer.
        </p>
      </LabSurface>

      <LabSurface label="Messy rows" className="dfr-messy-card">
        <SurfaceHeading
          kicker={`Same six listings, ${messActive ? "spoiled" : "clean"} · new listing: ${QUERY.rooms} rooms, ${QUERY.color} door, no park`}
          title="One bad cell can move a summary and flip a neighbour"
        />
        <div className="dfr-mess-control">
          <SegmentedControl
            label="Missing value"
            value={mess.missing ? "on" : "off"}
            options={[
              { value: "off", label: "off" },
              { value: "on", label: "Dock 2 rooms blank" },
            ]}
            onChange={(value) => setState({ messMissing: value === "on" })}
          />
          <SegmentedControl
            label="Handle missing"
            value={mess.handling}
            options={[
              { value: "zero", label: "read as 0" },
              { value: "drop", label: "drop the row" },
              { value: "mean", label: "fill with the mean" },
            ]}
            onChange={(value) => setState({ missingHandling: value })}
          />
          <SegmentedControl
            label="Duplicate row"
            value={mess.duplicate ? "on" : "off"}
            options={[
              { value: "off", label: "off" },
              { value: "on", label: "Alder 12 twice" },
            ]}
            onChange={(value) => setState({ messDuplicate: value === "on" })}
          />
          <SegmentedControl
            label="Mislabelled row"
            value={mess.mislabel ? "on" : "off"}
            options={[
              { value: "off", label: "off" },
              { value: "on", label: "Dock 2 tag flipped" },
            ]}
            onChange={(value) => setState({ messMislabel: value === "on" })}
          />
        </div>
        <table className="tg-board fit-table" aria-label="The six listings with the chosen mess applied">
          <thead>
            <tr>
              <th scope="col">Listing</th>
              <th scope="col">Rooms</th>
              <th scope="col">Color</th>
              <th scope="col">Park</th>
              <th scope="col">Tag</th>
              <th scope="col">What is wrong</th>
            </tr>
          </thead>
          <tbody>
            {messRows.map((row) => (
              <tr key={row.key} className={row.flag ? "is-leader" : ""}>
                <th scope="row">{row.name}</th>
                <td>{Number.isInteger(row.rooms) ? row.rooms : row.rooms.toFixed(1)}</td>
                <td>{row.color}</td>
                <td>{row.park ? "yes" : "no"}</td>
                <td>{answerWord(row.price)}</td>
                <td>{row.flag || "nothing"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="metric-row">
          <Metric label="Rows in the summary" value={`${messSummary.count}`} />
          <Metric label="Mean rooms" value={messSummary.meanRooms.toFixed(2)} tone={messSummary.meanRooms !== cleanSummary.meanRooms ? "loss" : undefined} />
          <Metric label="High-price share" value={`${(messSummary.highShare * 100).toFixed(1)}%`} tone={messSummary.highShare !== cleanSummary.highShare ? "loss" : undefined} />
          <Metric
            label={`${K}-nearest answer`}
            value={answerWord(messAnswer.answer)}
            tone={messAnswer.answer !== cleanAnswer.answer ? "loss" : "forward"}
          />
          <Metric
            label="Leave-one-out accuracy"
            value={`${messLoo.correct} of ${messLoo.total} · ${(messLoo.accuracy * 100).toFixed(1)}%`}
            tone={messLoo.accuracy !== cleanLoo.accuracy ? "loss" : undefined}
          />
          <Metric label="Answer against the clean table" value={messAnswer.answer === cleanAnswer.answer ? "same" : "changed"} />
        </div>
        <p className="lab-note">
          The {K} listings nearest to the new one are{" "}
          {messAnswer.neighbors.map((item) => `${item.row.name} (${item.distance.toFixed(2)}, ${answerWord(item.row.price)})`).join(", ")}:{" "}
          {messAnswer.high} of {messAnswer.neighbors.length} are high-priced, so the answer is {answerWord(messAnswer.answer)}.{" "}
          {messNotes.length === 0
            ? `This is the clean table: ${cleanSummary.count} rows, mean rooms ${cleanSummary.meanRooms.toFixed(2)}, ${(cleanSummary.highShare * 100).toFixed(0)}% high-priced, and a leave-one-out accuracy of ${cleanLoo.correct} of ${cleanLoo.total}.`
            : messNotes.join(" ")}{" "}
          Leave-one-out predicts each row from its nearest other row. Distances use the default integer stamps (red 1, blue
          2, green 3).
        </p>
      </LabSurface>
    </div>
    </div>
  );
}
