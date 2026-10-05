import { Check, Eye, MessageSquareText, MousePointer2, X } from "lucide-react";
import { useAppStore, type ColorVision } from "@app/store/app-store";
import { IconButton } from "./IconButton";

export function Switch({
  checked,
  label,
  description,
  onChange,
  icon,
}: {
  checked: boolean;
  label: string;
  description: string;
  onChange: (checked: boolean) => void;
  icon: React.ReactNode;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className="accessibility-switch"
      onClick={() => onChange(!checked)}
    >
      <span className="accessibility-switch__icon">{icon}</span>
      <span><strong>{label}</strong><small>{description}</small></span>
      <i>{checked && <Check />}</i>
    </button>
  );
}

export const palettes: { value: ColorVision; label: string }[] = [
  { value: "default", label: "Default" },
  { value: "deuteranopia", label: "Deuteranopia" },
  { value: "protanopia", label: "Protanopia" },
  { value: "tritanopia", label: "Tritanopia" },
];

export function AccessibilityPopover() {
  const store = useAppStore();
  if (!store.accessibilityOpen) return null;

  return (
    <aside className="accessibility-popover" aria-label="Accessibility options">
      <div className="popover-heading">
        <div><strong>Make the guide yours</strong><span>These settings apply immediately.</span></div>
        <IconButton label="Close accessibility options" onClick={() => store.setAccessibilityOpen(false)}>
          <X />
        </IconButton>
      </div>
      <Switch
        checked={store.highContrast}
        label="Higher contrast"
        description="Stronger edges and text separation"
        icon={<Eye />}
        onChange={store.setHighContrast}
      />
      <Switch
        checked={store.reducedMotion}
        label="Reduce motion"
        description="Remove nonessential animation"
        icon={<MousePointer2 />}
        onChange={store.setReducedMotion}
      />
      <Switch
        checked={store.verboseNarration}
        label="Verbose narration"
        description="Announce interactive changes"
        icon={<MessageSquareText />}
        onChange={store.setVerboseNarration}
      />
      <fieldset className="palette-picker">
        <legend>Colour-vision palette</legend>
        <div>
          {palettes.map((palette) => (
            <button
              type="button"
              key={palette.value}
              aria-pressed={store.colorVision === palette.value}
              onClick={() => store.setColorVision(palette.value)}
            >
              <i className={`palette-swatch palette-swatch--${palette.value}`} />
              {palette.label}
            </button>
          ))}
        </div>
      </fieldset>
    </aside>
  );
}
