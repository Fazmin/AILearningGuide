/**
 * Pictures for the LoRA lab. Each one draws numbers the lab has already computed; nothing here trains
 * or counts anything itself.
 */

/* -------------------------------------------------------------------------- */
/* The path an input takes when the model is served                            */
/* -------------------------------------------------------------------------- */

export function ServingPaths({
  mode,
  baseWeights,
  adapterWeights,
  scaleText,
}: {
  mode: "base" | "adapted" | "merged";
  baseWeights: number;
  adapterWeights: number;
  /** α/r as printed, for the merged table's name. */
  scaleText: string;
}) {
  const served = mode === "adapted" ? baseWeights + adapterWeights : baseWeights;
  const reading =
    mode === "merged"
      ? `Merged: the input passes through one table of ${baseWeights} weights that already holds the adapter's update. No adapter is left.`
      : mode === "adapted"
        ? `Base plus adapter: the input passes through the frozen table W, ${baseWeights} weights, and through the adapter A then B, ${adapterWeights} weights, and the two results are added. ${served} weights are served.`
        : `Base only: the input passes through the frozen table W, ${baseWeights} weights. The adapter, ${adapterWeights} weights, is detached and unused.`;
  return (
    <figure className={`lra-serve lra-serve--${mode}`}>
      <svg viewBox={mode === "merged" ? "0 0 240 76" : "0 0 240 126"} role="img" aria-label={reading}>
        {/* the input */}
        <circle className="lra-serve__node" cx="15" cy="38" r="10.5" />
        <text className="lra-serve__glyph" x="15" y="42" textAnchor="middle">
          x
        </text>

        {/* the base path, always live */}
        <path className="lra-serve__wire is-live" d="M26 38 H58" />
        <path className="lra-serve__wire is-live" d={mode === "merged" ? "M162 38 H222" : "M162 38 H184"} />
        <rect className={`lra-serve__box lra-serve__box--base${mode === "merged" ? " is-merged" : ""}`} x="58" y="18" width="104" height="40" rx="5" />
        <text className="lra-serve__name" x="110" y="35" textAnchor="middle">
          {mode === "merged" ? `W + ${scaleText}·B·A` : "W"}
        </text>
        <text className="lra-serve__count" x="110" y="50" textAnchor="middle">
          {mode === "merged" ? `${baseWeights} · one table` : `${baseWeights} · frozen`}
        </text>

        {mode !== "merged" && (
          <>
            {/* the adapter path, live only while attached */}
            <path className={`lra-serve__wire${mode === "adapted" ? " is-live is-adapter" : ""}`} d="M20 48 C24 78 34 94 58 94" />
            <path className={`lra-serve__wire${mode === "adapted" ? " is-live is-adapter" : ""}`} d="M162 94 C186 94 195 78 195 50" />
            <rect className={`lra-serve__box lra-serve__box--adapter${mode === "adapted" ? " is-live" : ""}`} x="58" y="76" width="104" height="36" rx="5" />
            <text className="lra-serve__name" x="110" y="92" textAnchor="middle">
              A, then B
            </text>
            <text className="lra-serve__count" x="110" y="105" textAnchor="middle">
              {mode === "adapted" ? `${adapterWeights} · attached` : `${adapterWeights} · detached`}
            </text>
            <circle className={`lra-serve__node${mode === "adapted" ? " is-live" : ""}`} cx="195" cy="38" r="10.5" />
            <text className="lra-serve__glyph" x="195" y="42.5" textAnchor="middle">
              +
            </text>
            <path className="lra-serve__wire is-live" d="M206 38 H222" />
          </>
        )}
        <path className="lra-serve__tip" d="M222 33 L232 38 L222 43 Z" />
      </svg>
      <figcaption>
        {mode === "merged"
          ? `One table, ${baseWeights} weights. The adapter is folded in and gone.`
          : mode === "adapted"
            ? `Two paths, added: ${baseWeights} + ${adapterWeights} = ${served} weights served.`
            : `One path: ${baseWeights} weights served. The adapter waits, unused.`}
      </figcaption>
    </figure>
  );
}

/* -------------------------------------------------------------------------- */
/* The trainable share of the toy against a real model                         */
/* -------------------------------------------------------------------------- */

export function ShareBars({ rows }: { rows: ReadonlyArray<{ id: string; label: string; share: number; text: string }> }) {
  return (
    <div
      className="lra-share"
      role="img"
      aria-label={rows.map((row) => `${row.label}: ${row.text} of its weights train.`).join(" ")}
    >
      <span className="lra-share__title">Each bar is a whole model. The filled part is what the adapter trains.</span>
      {rows.map((row) => (
        <div className="lra-share__row" key={row.id}>
          <b>{row.label}</b>
          <span className="lra-share__track">
            <i style={{ width: `${Math.min(100, row.share * 100)}%` }} />
          </span>
          <code>{row.text}</code>
          {row.share > 0 && row.share < 0.02 && <small>the thin line at the left edge, drawn wider than it truly is so that it shows</small>}
        </div>
      ))}
    </div>
  );
}
