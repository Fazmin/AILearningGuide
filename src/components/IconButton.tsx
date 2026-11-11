import type { ButtonHTMLAttributes, ReactNode, Ref } from "react";

interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label"> {
  label: string;
  children: ReactNode;
  ref?: Ref<HTMLButtonElement>;
  tooltipSide?: "top" | "right" | "bottom" | "left";
  /** Hang a top or bottom tooltip from one edge of the button, for a button beside a clipping edge. */
  tooltipAlign?: "start" | "end";
}

export function IconButton({
  label,
  children,
  className = "",
  tooltipSide = "bottom",
  tooltipAlign,
  ...props
}: IconButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      data-tooltip={label}
      data-tooltip-side={tooltipSide}
      data-tooltip-align={tooltipAlign}
      className={`icon-button ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
