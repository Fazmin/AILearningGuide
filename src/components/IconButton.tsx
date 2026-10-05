import type { ButtonHTMLAttributes, ReactNode } from "react";

interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label"> {
  label: string;
  children: ReactNode;
  tooltipSide?: "top" | "right" | "bottom" | "left";
}

export function IconButton({
  label,
  children,
  className = "",
  tooltipSide = "bottom",
  ...props
}: IconButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      data-tooltip={label}
      data-tooltip-side={tooltipSide}
      className={`icon-button ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
