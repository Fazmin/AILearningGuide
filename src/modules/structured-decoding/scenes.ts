import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Inside `state`, `step` is the lab's Decode step,
 * counted from 0, so `step: 3` is the button the lesson calls step 4. It is not the shell step set
 * by the scene's own `step`. The JSON walk is {" ok ": true } ⟨eos⟩, so `step: 3` stands at
 * {"ok": and `step: 4` at {"ok":true. The function-call walk is get _weather (city =" Os lo ") ⟨eos⟩.
 */
const scenes: LabScene[] = [
  {
    id: "renormalization",
    sections: ["Rescaling what is left: renormalization", "What it is"],
    title: "Dividing by Z",
    look: "The formula under the table divides the picked token’s chance by Z, the legal mass named in the card’s heading.",
    step: 2,
    focus: "Masked distribution",
  },
  {
    id: "token-boundaries",
    sections: ["How the mask decides what is legal", "How legality is decided"],
    title: "A token across two pieces",
    look: "With the text at {\"ok, the legal \": token carries a ↔ tag naming the two grammar pieces it covers.",
    step: 2,
    state: { step: 2 },
    focus: "Masked distribution",
  },
  {
    id: "end-masked",
    sections: ["Knowing when to stop"],
    title: "Too early to stop",
    look: "With the text at {\"ok\":true, ⟨eos⟩ is marked ✗ masked, because stopping here would leave the object without its closing bracket.",
    step: 2,
    state: { step: 4 },
    focus: "Masked distribution",
  },
  {
    id: "schema-stricter",
    sections: ["Any JSON is not enough: the schema"],
    title: "Stricter than JSON",
    look: "With the text at {\"ok\":, null and the quotation mark are ✗ masked even though plain JSON would allow them, so only true and false are left.",
    step: 2,
    state: { step: 3 },
    focus: "Masked distribution",
  },
  {
    id: "choose-grammar",
    sections: ["Step 1: choose a format on Grammar", "How to play with it"],
    title: "Two formats",
    look: "Read the Prompt, Allowed output and Rule lines, then switch Grammar to Function call and read them again.",
    step: 0,
    focus: "Grammar",
  },
  {
    id: "prefix-walk",
    sections: ["Step 2: step through on Prefix walk"],
    title: "Walk the prefix",
    look: "Click the Decode step buttons in order and compare Unconstrained pick with Pick after mask at each one.",
    step: 1,
    focus: "Prefix walk",
  },
  {
    id: "first-step-numbers",
    sections: ["Step 3: read the numbers on Masked distribution"],
    title: "The first token, masked",
    look: "At the first step, Sure falls to 0 in the after mask column while {\" rises; then click Decode step button 4 and find the masked quotation mark.",
    step: 2,
    focus: "Masked distribution",
  },
  {
    id: "city-field",
    sections: ["Step 4: compare with the function call"],
    title: "A free text field",
    look: "Where the city name starts, Legal tokens jumps up, because almost any word made of letters fits; then click Decode step button 6.",
    step: 1,
    state: { grammar: "call", step: 4 },
    focus: "Prefix walk",
  },
  {
    id: "mostly-agree",
    sections: ["The model and the mask mostly agree"],
    title: "After the closing bracket",
    look: "With the object finished, Unconstrained pick is a word that would start a sentence, while Pick after mask is ⟨eos⟩.",
    step: 1,
    state: { step: 5 },
    focus: "Prefix walk",
  },
  {
    id: "distortion-slider",
    sections: ["Step 5: drag the slider on Mask distortion"],
    title: "Mask against conditioning",
    look: "Drag P(slo\" after \"O) up from 10% and watch the conditioned bar climb while the Per-step mask bar stays put.",
    step: 3,
    focus: "Mask distortion",
  },
  {
    id: "bars-match",
    sections: ["Watching the slider"],
    title: "Where the bars meet",
    look: "At 90% the two bars show the same split; drag a little higher and the conditioned bar edges past the per-step one.",
    step: 3,
    state: { detour: 0.9 },
    focus: "Mask distortion",
  },
];

export default scenes;
