import { formatCount, formatIntensity, type OperationId } from "./roofline";

const SIDE = 72;

interface Block {
  id: string;
  name: string;
  shape: string;
  role: string;
  rows: number;
  columns: number;
  bytes: number;
}

/**
 * The operands drawn to scale: each block's area is the bytes it moves across the memory bus.
 * At a batch of one the weight matrix is a full square and the input is a hairline beside it.
 */
export function TrafficShapes({
  operation,
  batch,
  width,
  bytesPerElement,
  opUnit,
  intensity,
}: {
  operation: OperationId;
  batch: number;
  width: number;
  bytesPerElement: number;
  opUnit: string;
  intensity: number;
}) {
  const strip = { rows: batch, columns: width, bytes: batch * width * bytesPerElement, shape: `${batch}×${width}` };
  const blocks: Block[] =
    operation === "matmul"
      ? [
          { id: "x", name: "X", role: "read", ...strip },
          { id: "w", name: "W", role: "read once", rows: width, columns: width, bytes: width * width * bytesPerElement, shape: `${width}×${width}` },
          { id: "y", name: "Y", role: "written", ...strip },
        ]
      : [
          { id: "x", name: "X", role: "read", ...strip },
          { id: "r", name: "R", role: "read", ...strip },
          { id: "y", name: "Y", role: "written", ...strip },
        ];
  const unit = Math.max(batch, width);
  const total = blocks.reduce((sum, block) => sum + block.bytes, 0);
  const share = (bytes: number) => {
    const percent = (bytes / total) * 100;
    return percent >= 99.95 ? ">99.9%" : percent < 0.05 ? "<0.1%" : `${percent.toFixed(1)}%`;
  };
  const rowsText = `${batch.toLocaleString("en-US")} row${batch === 1 ? "" : "s"}`;
  const summary =
    operation === "matmul"
      ? `Bytes moved, drawn to scale. X ${strip.shape} is ${formatCount(strip.bytes, "B")}, W ${width}×${width} is ${formatCount(blocks[1].bytes, "B")} (${share(blocks[1].bytes)} of the traffic), Y ${strip.shape} is ${formatCount(strip.bytes, "B")}. Each weight is fetched once and used by ${rowsText}, giving ${formatIntensity(intensity)} ${opUnit} per byte.`
      : `Bytes moved, drawn to scale. X, R and Y are each ${strip.shape} and ${formatCount(strip.bytes, "B")}. Every value is touched once for one add, giving ${formatIntensity(intensity)} ${opUnit} per byte.`;

  return (
    <div className="wg-shapes" role="img" aria-label={summary}>
      <div className="wg-shapes__blocks" aria-hidden="true">
        {blocks.map((block, index) => (
          <div className="wg-shapes__item" key={block.id}>
            {index > 0 && (
              <span className="wg-shapes__operator">
                {index === 2 ? "→" : operation === "matmul" ? "·" : "+"}
              </span>
            )}
            <div className="wg-shapes__cell">
              <div className="wg-shapes__frame" style={{ height: SIDE }}>
                <i
                  className={`wg-shapes__block wg-shapes__block--${block.id}`}
                  style={{
                    width: Math.max(3, (block.columns / unit) * SIDE),
                    height: Math.max(2, (block.rows / unit) * SIDE),
                  }}
                />
              </div>
              <span className="wg-shapes__name">
                <b>{block.name}</b> {block.shape}
              </span>
              <span className="wg-shapes__bytes">
                {formatCount(block.bytes, "B")} · {share(block.bytes)}
              </span>
              <span className="wg-shapes__role">{block.role}</span>
            </div>
          </div>
        ))}
      </div>
      <div className="wg-shapes__readout" aria-hidden="true">
        <span className="wg-shapes__caption">Area = bytes that cross the memory bus</span>
        <span className="wg-shapes__reuse">
          {operation === "matmul" ? (
            <>
              Each weight is fetched once and used by <b>{rowsText}</b>
            </>
          ) : (
            <>
              Each value is fetched once and used <b>once</b>
            </>
          )}
        </span>
        <span className="wg-shapes__intensity">
          <b>
            {formatIntensity(intensity)} {opUnit}
          </b>{" "}
          of work per byte moved
        </span>
      </div>
    </div>
  );
}
