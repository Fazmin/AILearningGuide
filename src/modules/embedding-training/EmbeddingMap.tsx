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
}: {
  space: Space;
  projection: Projection;
  selectedQuery: number;
  step: number;
  meanCosine: number;
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
  const answerLabel = labelFor(answerPoint);
  const queryLabel = labelFor(queryPoint);

  return (
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
        if (index === selectedQuery) return null;
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
      {centroids.map(({ topic, x, y }) => (
        <text key={topic.id} className="etr-map__topic" x={x} y={y - 7} textAnchor="middle">
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
      <text className="etr-map__label" x={answerLabel.x} y={answerPoint.y + 14} textAnchor={answerLabel.anchor}>
        {`answer: ${PASSAGES[answer].title}`}
      </text>
      {twin >= 0 && (
        <>
          <circle className="etr-map__answer is-twin" cx={passagePoints[twin].x} cy={passagePoints[twin].y} r={6} />
          <text
            className="etr-map__label"
            x={labelFor(passagePoints[twin]).x}
            y={passagePoints[twin].y - 9}
            textAnchor={labelFor(passagePoints[twin]).anchor}
          >
            {`duplicate: ${PASSAGES[twin].title}`}
          </text>
        </>
      )}
      <rect className="etr-map__query is-selected" x={queryPoint.x - 4} y={queryPoint.y - 4} width={8} height={8} />
      <text className="etr-map__label is-query" x={queryLabel.x} y={queryPoint.y - 8} textAnchor={queryLabel.anchor}>
        {`query ${selected.id}`}
      </text>
    </svg>
  );
}
