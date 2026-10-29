import type { LabScene } from "@app/module-sdk";
import { presetById } from "./host";

/**
 * Lab setups the lesson opens from its sections. A preset loads the way its button does, with the
 * preset's own text in both call boxes and Turn back on 1, so the host checks the authored call
 * rather than an edit. Approve send_email stays on no approval unless a scene says otherwise.
 */
const preset = (id: string) => ({
  preset: id,
  draft: presetById(id).output,
  retry: presetById(id).retry ?? "",
  turn: "1",
});

const scenes: LabScene[] = [
  {
    id: "json-call",
    sections: ["JSON: a way to write labelled values"],
    title: "A call written as JSON",
    look: "Model output holds the call as raw text: name says which tool to use, and arguments holds the dock and the day.",
    step: 0,
    state: preset("valid"),
    focus: "Model output",
  },
  {
    id: "ten-stages",
    sections: ["One round trip, in ten stages", "What it is"],
    title: "One round trip, ten stages",
    look: "The model’s stages sit on the left and the host’s on the right, and only the host’s stages carry a pass or fail mark.",
    step: 0,
    state: preset("valid"),
    focus: "Host pipeline",
  },
  {
    id: "six-ways",
    sections: ["Six ways a call can be broken"],
    title: "A broken call, caught",
    look: "Stage 6 fails on /day, the later checks read – skipped, and stage 9 still appends a tool message that names the problem.",
    step: 1,
    state: preset("type"),
    focus: "Host pipeline",
  },
  {
    id: "email-schema",
    sections: ["Neat is not the same as safe: host policy"],
    title: "What the email schema checks",
    look: "Schema check shows send_email’s pattern for the address and its length limit for the body, and its heading says policy needs approval.",
    step: 3,
    state: preset("email"),
    focus: "Schema check",
  },
  {
    id: "grammar",
    sections: ["What constrained decoding can and cannot do"],
    title: "What a grammar can stop",
    look: "Grammar blocks it reads no for this neat email call; click missing field and watch it change to yes.",
    step: 4,
    state: preset("email"),
    focus: "Transcript",
  },
  {
    id: "error-as-data",
    sections: ["When a call fails, the error comes back as data"],
    title: "The error the retry reads",
    look: "The tool message sits between the two calls, and the retry changes only the value that message names.",
    step: 5,
    state: preset("retry-type"),
    focus: "Transcript",
  },
  {
    id: "read-contract",
    sections: ["Read the contract"],
    title: "Read the contract",
    look: "Schema check lists get_dock_hours’ two enums, its required list and additionalProperties: false.",
    step: 0,
    state: preset("valid"),
    focus: "Schema check",
  },
  {
    id: "break-arguments",
    sections: ["Break the arguments"],
    title: "Break the arguments",
    look: "Stages 4 and 5 pass, stage 6 fails and the rest read – skipped; then click bad enum, wrong type and extra field in turn.",
    step: 1,
    state: preset("missing"),
    focus: "Host pipeline",
  },
  {
    id: "break-text",
    sections: ["Break the text"],
    title: "Break the text",
    look: "Stage 4 fails because the text stops early; click unknown tool and watch the failure move one stage later.",
    step: 2,
    state: preset("broken"),
    focus: "Host pipeline",
  },
  {
    id: "valid-not-safe",
    sections: ["Valid is not safe"],
    title: "A neat call, refused",
    look: "Stages 4 to 6 pass and stage 7 refuses; set Approve send_email to human approves and watch stages 7 and 8 change.",
    step: 3,
    state: preset("email"),
    focus: "Host pipeline",
  },
  {
    id: "read-transcript",
    sections: ["Read the transcript"],
    title: "What the next call reads",
    look: "Transcript lists the system, user, assistant and tool messages in order, and its last line marks where the next call would start.",
    step: 4,
    state: preset("valid"),
    focus: "Transcript",
  },
  {
    id: "repair",
    sections: ["Repair a bad call"],
    title: "Wrong type, then fixed",
    look: "Turn shows the first call failing at stage 6; switch it to 2 · retry and watch every stage pass.",
    step: 5,
    state: preset("retry-type"),
    focus: "Host pipeline",
  },
];

export default scenes;
