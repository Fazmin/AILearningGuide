import type { CSSProperties } from "react";

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

export const showChar = (character: string) => (character === " " ? "␣" : character === "\n" ? "↵" : character);

/**
 * The 256 numbers of one residual vector as signed bars, with the SAE's
 * reconstruction drawn as a dot on each bar so the two can be compared
 * dimension by dimension.
 */
export function ResidualBars({
  x,
  reconstruction,
  label,
}: {
  x: ArrayLike<number> | null;
  reconstruction: ArrayLike<number> | null;
  label: string;
}) {
  const width = 512;
  const height = 110;
  const middle = height / 2;
  const values = x ? Array.from(x) : [];
  const peak = Math.max(1e-6, ...values.map(Math.abs), ...(reconstruction ? Array.from(reconstruction).map(Math.abs) : []));
  const scale = (middle - 6) / peak;
  const step = width / 256;
  return (
    <svg className="sf-residual" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} preserveAspectRatio="none">
      <line x1={0} x2={width} y1={middle} y2={middle} className="sf-residual__axis" />
      {values.map((value, index) => (
        <rect
          key={index}
          x={index * step}
          y={value >= 0 ? middle - value * scale : middle}
          width={Math.max(0.6, step - 0.4)}
          height={Math.abs(value) * scale}
          className="sf-residual__bar"
        />
      ))}
      {reconstruction &&
        Array.from(reconstruction).map((value, index) => (
          <rect
            key={`r-${index}`}
            x={index * step}
            y={middle - value * scale - 0.9}
            width={Math.max(0.6, step - 0.4)}
            height={1.8}
            className="sf-residual__recon"
          />
        ))}
    </svg>
  );
}

/** 1,024 dictionary slots with a spike for every active feature. */
export function SparseCode({
  active,
  total,
  selected,
  label,
}: {
  active: ReadonlyArray<{ feature: number; value: number }>;
  total: number;
  selected: number;
  label: string;
}) {
  const width = 1024;
  const height = 120;
  const base = 100;
  const peak = Math.max(1e-6, ...active.map((entry) => entry.value));
  return (
    <svg className="sf-code-strip" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} preserveAspectRatio="none">
      <rect x={0} y={base} width={width} height={4} className="sf-code-strip__track" />
      {[0, 256, 512, 768, 1024].map((tick) => (
        <line key={tick} x1={tick} x2={tick} y1={base} y2={base + 10} className="sf-code-strip__tick" />
      ))}
      {active.map(({ feature, value }) => {
        const h = (value / peak) * (base - 8);
        return (
          <g key={feature} className={feature === selected ? "is-selected" : ""}>
            <rect x={(feature / total) * width - 3} y={base - h} width={6} height={h} className="sf-code-strip__spike" />
            <title>{`feature ${feature}: ${value.toFixed(2)}`}</title>
          </g>
        );
      })}
    </svg>
  );
}

/** Characters of the probe text, shaded by one feature's activation, marking which fire and which match a label. */
export function ActivationText({
  text,
  activations,
  threshold,
  matches,
  selected,
  onSelect,
  maximum,
  earlyPositions = 0,
}: {
  text: string;
  activations: ReadonlyArray<number>;
  threshold: number;
  matches: (character: string, position: number) => boolean;
  selected?: number;
  onSelect?: (position: number) => void;
  maximum: number;
  /** Positions below this are the start of the model's window and get a marked top edge. */
  earlyPositions?: number;
}) {
  return (
    <ol className="sf-text" aria-label="Probe text with the feature's activation on each character">
      {Array.from(text).map((character, position) => {
        const value = activations[position] ?? 0;
        const fires = value > threshold;
        const match = matches(character, position);
        const early = position < earlyPositions;
        const spoken = character === " " ? "space" : character === "\n" ? "line break" : character;
        const body = (
          <>
            <span>{showChar(character)}</span>
            <small>{value > 0 ? value.toFixed(1) : "·"}</small>
          </>
        );
        return (
          <li
            key={position}
            className={[fires ? "is-firing" : "", match ? "is-match" : "", early ? "is-early" : "", selected === position ? "is-selected" : ""]
              .filter(Boolean)
              .join(" ")}
            style={{ "--a": clamp(value / Math.max(1e-6, maximum), 0, 1) } as CSSProperties}
          >
            {onSelect ? (
              <button
                type="button"
                aria-pressed={selected === position}
                aria-label={`position ${position}, ${spoken}, activation ${value.toFixed(2)}${fires ? ", fires" : ""}${match ? ", matches the label" : ""}${early ? ", window start" : ""}`}
                onClick={() => onSelect(position)}
              >
                {body}
              </button>
            ) : (
              <div
                role="group"
                aria-label={`position ${position}, ${spoken}, activation ${value.toFixed(2)}${fires ? ", fires" : ""}${match ? ", matches the label" : ""}${early ? ", window start" : ""}`}
              >
                {body}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
