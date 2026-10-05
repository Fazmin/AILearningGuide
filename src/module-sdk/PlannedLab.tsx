import { CalendarClock } from "lucide-react";
import { LabSurface, SurfaceHeading } from "./visualizations";

export function PlannedLab({
  heading,
  description,
  controls,
  currentStep = 0,
}: {
  heading: string;
  description: string;
  controls: string[];
  currentStep?: number;
}) {
  return (
    <div className="lab-stack">
      <LabSurface label="Planned interactive" className="planned-lab">
        <div className={`planned-lab-preview${currentStep === 0 ? " is-current" : ""}`}>
          <SurfaceHeading
            kicker="Planned interactive"
            title={heading}
            icon={<CalendarClock size={20} />}
          />
          <p className="planned-lab-copy">{description}</p>
        </div>
        <ol
          className={`planned-lab-controls${currentStep === 1 ? " is-current" : ""}`}
          aria-label="Controls the finished lab will expose"
        >
          {controls.map((control) => (
            <li key={control}>{control}</li>
          ))}
        </ol>
        <p className={`planned-lab-limits lab-note${currentStep === 2 ? " is-current" : ""}`}>
          This card is a preview of a lab that is not built yet. The list names future
          controls; nothing here is wired to a live experiment.
        </p>
      </LabSurface>
    </div>
  );
}
