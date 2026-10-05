import { Check, ChevronDown } from "lucide-react";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface SettingsSelectOption<T extends string> {
  value: T;
  label: string;
  description?: string;
}

interface SettingsSelectProps<T extends string> {
  value: T;
  options: SettingsSelectOption<T>[];
  onChange: (value: T) => void;
  ariaLabel?: string;
  labelledBy?: string;
  disabled?: boolean;
  size?: "field" | "inline";
}

interface MenuPosition {
  top: number;
  left: number;
  width: number;
  maxHeight: number;
  openUp: boolean;
}

export function SettingsSelect<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  labelledBy,
  disabled = false,
  size = "field",
}: SettingsSelectProps<T>) {
  const listboxId = useId();
  const valueId = useId();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLUListElement>(null);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(() => Math.max(0, options.findIndex((option) => option.value === value)));
  const [position, setPosition] = useState<MenuPosition | null>(null);
  const selected = options.find((option) => option.value === value) ?? options[0];

  const updatePosition = () => {
    const trigger = buttonRef.current;
    if (!trigger) return;
    const rect = trigger.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom - 12;
    const spaceAbove = rect.top - 12;
    const openUp = spaceBelow < 16 * 12 && spaceAbove > spaceBelow;
    const maxHeight = Math.max(8 * 16, Math.min(18 * 16, openUp ? spaceAbove : spaceBelow));
    const width = Math.min(Math.max(rect.width, 12 * 16), window.innerWidth - 24);
    setPosition({
      top: openUp ? rect.top - maxHeight - 6 : rect.bottom + 6,
      left: Math.min(Math.max(12, rect.left), window.innerWidth - width - 12),
      width,
      maxHeight,
      openUp,
    });
  };

  useLayoutEffect(() => {
    if (!open) return;
    updatePosition();
    const selectedIndex = Math.max(0, options.findIndex((option) => option.value === value));
    setHighlight(selectedIndex);
    const frame = window.requestAnimationFrame(() => {
      const active = menuRef.current?.querySelector<HTMLElement>("[data-active=true]");
      active?.focus();
    });
    const onReposition = () => updatePosition();
    window.addEventListener("resize", onReposition);
    document.addEventListener("scroll", onReposition, true);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", onReposition);
      document.removeEventListener("scroll", onReposition, true);
    };
  }, [open, options, value]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (buttonRef.current?.contains(target) || menuRef.current?.contains(target)) return;
      setOpen(false);
      buttonRef.current?.focus();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setOpen(false);
        buttonRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const move = (next: number) => {
    if (options.length === 0) return;
    const index = (next + options.length) % options.length;
    setHighlight(index);
    menuRef.current?.querySelectorAll<HTMLElement>("[role=option]")[index]?.focus();
  };

  const choose = (next: T) => {
    onChange(next);
    setOpen(false);
    buttonRef.current?.focus();
  };

  return (
    <div className={`settings-select settings-select--${size} ${open ? "is-open" : ""}`}>
      <button
        ref={buttonRef}
        type="button"
        className="settings-select__button"
        aria-label={labelledBy ? undefined : (ariaLabel ? `${ariaLabel}: ${selected?.label ?? ""}` : undefined)}
        aria-labelledby={labelledBy ? `${labelledBy} ${valueId}` : undefined}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listboxId}
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            if (!open) setOpen(true);
          }
        }}
      >
        <span>
          <strong id={valueId}>{selected?.label ?? "Choose"}</strong>
        </span>
        <ChevronDown />
      </button>
      {open && position && createPortal(
        <ul
          ref={menuRef}
          id={listboxId}
          role="listbox"
          className={`settings-select__menu${position.openUp ? " is-above" : ""}`}
          aria-label={ariaLabel}
          style={{
            top: position.top,
            left: position.left,
            width: position.width,
            maxHeight: position.maxHeight,
          }}
        >
          {options.map((option, index) => {
            const isSelected = option.value === value;
            const isActive = index === highlight;
            return (
              <li key={option.value} role="presentation">
                <button
                  type="button"
                  role="option"
                  data-active={isActive}
                  aria-selected={isSelected}
                  className={isSelected ? "is-selected" : ""}
                  onMouseEnter={() => setHighlight(index)}
                  onClick={() => choose(option.value)}
                  onKeyDown={(event) => {
                    if (event.key === "ArrowDown") {
                      event.preventDefault();
                      move(index + 1);
                    } else if (event.key === "ArrowUp") {
                      event.preventDefault();
                      move(index - 1);
                    } else if (event.key === "Home") {
                      event.preventDefault();
                      move(0);
                    } else if (event.key === "End") {
                      event.preventDefault();
                      move(options.length - 1);
                    } else if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      choose(option.value);
                    }
                  }}
                >
                  <span>
                    <strong>{option.label}</strong>
                    {option.description && <small>{option.description}</small>}
                  </span>
                  {isSelected && <Check />}
                </button>
              </li>
            );
          })}
        </ul>,
        document.body,
      )}
    </div>
  );
}
