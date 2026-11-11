import { HELD_OUT_QUERIES, PASSAGES, passageIndex, TOPICS, DUPLICATE_OF } from "./data";
import { project, type Projection, type Space } from "./metrics";

/**
 * Module-local scatter: every passage and held-out query, projected onto the two principal directions
 * of the passage vectors. Every vector has length 1, so every shadow lands inside the unit circle; a
 * tight clump means the vectors point nearly the same way, whatever the axes are.
 */
const SIZE = 340;
const CENTRE = SIZE / 2;
const RADIUS = 150;

const at = (point: [number, number]) => ({ x: CENTRE + RADIUS * point[0], y: CENTRE - RADIUS * point[1] });

export function EmbeddingMap({
  space,
  projection,
  selectedQuery,
  step,
  meanCosine,
  heading,
  quiet = false,
}: {
  space: Space;
  projection: Projection;
  selectedQuery: number;
  step: number;
  meanCosine: number;
  /** Printed above the map, so two maps side by side say which is which. */
  heading: string;
  /** The reference map: points and the selected query only, with no topic names or other links. */
  quiet?: boolean;
}) {
  const passagePoints = space.passages.map((vector) => at(project(vector, projection)));
  const queryPoints = space.queries.map((vector) => at(project(vector, projection)));
  const topicIndex = new Map(TOPICS.map((topic, index) => [topic.id, index]));
  const centroids = TOPICS.map((topic) => {
    const members = PASSAGES.map((passage, index) => ({ passage, index })).filter(({ passage }) => passage.topic === topic.id);
    const x = members.reduce((sum, { index }) => sum + passagePoints[index].x, 0) / members.length;
    const y = members.reduce((sum, { index }) => sum + passagePoints[index].y, 0) / members.length;
    return { topic, x, y };
  });
  const selected = HELD_OUT_QUERIES[selectedQuery];
  const answer = passageIndex(selected.answer);
  const twin = DUPLICATE_OF[answer];
  const queryPoint = queryPoints[selectedQuery];
  const answerPoint = passagePoints[answer];
  const labelFor = (point: { x: number; y: number }) => ({
    x: point.x + (point.x > CENTRE ? -6 : 6),
    anchor: point.x > CENTRE ? ("end" as const) : ("start" as const),
  });
  // The query, its answer and the duplicate often sit within a few pixels of each other, so their
  // three labels are spread into separate lines instead of printing on top of one another.
  const callouts = [
    { id: "query", text: `query ${selected.id}`, point: queryPoint, y: queryPoint.y - 8, isQuery: true },
    { id: "answer", text: `answer: ${PASSAGES[answer].title}`, point: answerPoint, y: answerPoint.y + 14, isQuery: false },
    ...(twin >= 0
      ? [{ id: "twin", text: `duplicate: ${PASSAGES[twin].title}`, point: passagePoints[twin], y: passagePoints[twin].y - 9, isQuery: false }]
      : []),
  ].sort((a, b) => a.y - b.y);
  for (let index = 1; index < callouts.length; index += 1) {
    if (callouts[index].y - callouts[index - 1].y < 13) callouts[index].y = callouts[index - 1].y + 13;
  }
  const overflow = Math.max(0, callouts[callouts.length - 1].y - (SIZE - 4));
  const underflow = Math.max(0, 12 - (callouts[0].y - overflow));
  for (const callout of callouts) callout.y += underflow - overflow;

  return (
    <figure className={`etr-map-figure${quiet ? " is-quiet" : ""}`}>
      <figcaption>
        <strong>{heading}</strong>
        <span>mean cosine between passages {meanCosine.toFixed(2)}</span>
      </figcaption>
      <svg
        className="etr-map"
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        role="img"
        aria-label={`Projection at training step ${step} of ${PASSAGES.length} passages and ${HELD_OUT_QUERIES.length} held-out queries onto two directions. Mean cosine between passages ${meanCosine.toFixed(2)}. The selected query, “${selected.text}”, is joined to its labelled answer, ${PASSAGES[answer].title}.`}
      >
        <circle className="etr-map__circle" cx={CENTRE} cy={CENTRE} r={RADIUS} />
        <line className="etr-map__axis" x1={CENTRE - RADIUS} x2={CENTRE + RADIUS} y1={CENTRE} y2={CENTRE} />
        <line className="etr-map__axis" x1={CENTRE} x2={CENTRE} y1={CENTRE - RADIUS} y2={CENTRE + RADIUS} />
        <text className="etr-map__tick" x={CENTRE + RADIUS + 2} y={CENTRE - 4} textAnchor="end">
          axis 1
        </text>
        <text className="etr-map__tick" x={CENTRE + 4} y={CENTRE - RADIUS + 10}>
          axis 2
        </text>
        {HELD_OUT_QUERIES.map((query, index) => {
          if (quiet || index === selectedQuery) return null;
          const from = queryPoints[index];
          const to = passagePoints[passageIndex(query.answer)];
          return <line key={query.id} className="etr-map__link" x1={from.x} y1={from.y} x2={to.x} y2={to.y} />;
        })}
        {passagePoints.map((point, index) => (
          <circle
            key={PASSAGES[index].id}
            className={`etr-map__passage etr-topic-${topicIndex.get(PASSAGES[index].topic) ?? 0}`}
            cx={point.x}
            cy={point.y}
            r={3.4}
          >
            <title>{`${PASSAGES[index].title} (${TOPICS[topicIndex.get(PASSAGES[index].topic) ?? 0].label})`}</title>
          </circle>
        ))}
        {queryPoints.map((point, index) =>
          index === selectedQuery ? null : (
            <rect key={HELD_OUT_QUERIES[index].id} className="etr-map__query" x={point.x - 2.2} y={point.y - 2.2} width={4.4} height={4.4}>
              <title>{`Held-out query: ${HELD_OUT_QUERIES[index].text}`}</title>
            </rect>
          ),
        )}
        {!quiet &&
          centroids.map(({ topic, x, y }) => (
            <text
              key={topic.id}
              className="etr-map__topic"
              x={x}
              y={y - 7}
              textAnchor={x < 56 ? "start" : x > SIZE - 56 ? "end" : "middle"}
            >
              {topic.label}
            </text>
          ))}
        <line className="etr-map__link is-selected" x1={queryPoint.x} y1={queryPoint.y} x2={answerPoint.x} y2={answerPoint.y} />
        {twin >= 0 && (
          <line
            className="etr-map__link is-twin"
            x1={queryPoint.x}
            y1={queryPoint.y}
            x2={passagePoints[twin].x}
            y2={passagePoints[twin].y}
          />
        )}
        <circle className="etr-map__answer" cx={answerPoint.x} cy={answerPoint.y} r={6} />
        {twin >= 0 && (
          <>
            <circle className="etr-map__answer is-twin" cx={passagePoints[twin].x} cy={passagePoints[twin].y} r={6} />
          </>
        )}
        <rect className="etr-map__query is-selected" x={queryPoint.x - 4} y={queryPoint.y - 4} width={8} height={8} />
        {callouts.map((callout) => (
          <text
            key={callout.id}
            className={`etr-map__label${callout.isQuery ? " is-query" : ""}`}
            x={labelFor(callout.point).x + (labelFor(callout.point).anchor === "end" ? -4 : 4)}
            y={callout.y}
            textAnchor={labelFor(callout.point).anchor}
          >
            {callout.text}
          </text>
        ))}
      </svg>
    </figure>
  );
}
