import { BookOpen, FlaskConical, Undo2, X } from "lucide-react";
import type { Ref } from "react";
import type { LabScene } from "@app/module-sdk";
import { IconButton } from "./IconButton";

/**
 * Says which lesson section set the lab up, with a way back to the reader's own setup. Laid over the
 * bottom of the lab, like the Learn panel's resume prompt, so the card it points at is not covered.
 */
export function LabSceneBanner({
  ref,
  scene,
  section,
  learnClosed,
  onRestore,
  onReturnToReading,
  onDismiss,
}: {
  ref?: Ref<HTMLDivElement>;
  scene: LabScene;
  section: string;
  /** True when the Learn panel is closed, as it is after a scene opened over a narrow window. */
  learnClosed: boolean;
  onRestore: () => void;
  onReturnToReading: () => void;
  onDismiss: () => void;
}) {
  return (
    <div ref={ref} className="lab-scene-banner" role="status" tabIndex={-1}>
      <FlaskConical className="lab-scene-banner__icon" />
      <div className="lab-scene-banner__text">
        <span>From the lesson · {section}</span>
        <strong>{scene.title}</strong>
        <p>{scene.look}</p>
      </div>
      <div className="lab-scene-banner__actions">
        {learnClosed && (
          <button type="button" className="lab-scene-banner__action" onClick={onReturnToReading}>
            <BookOpen />
            <span>Back to reading</span>
          </button>
        )}
        <button type="button" className="lab-scene-banner__action" onClick={onRestore}>
          <Undo2 />
          <span>Back to my lab</span>
        </button>
        <IconButton label="Keep this setup" tooltipSide="left" onClick={onDismiss}><X /></IconButton>
      </div>
    </div>
  );
}
