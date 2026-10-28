import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Word indices refer to the sentence each scene loads
 * (the default is "The animal did not cross the street because it was tired", where 8 is "it" and 1
 * is "animal"). Layer and head are 0-based, so `layer: 1, head: 3` is Layer 2 head 4.
 */
const scenes: LabScene[] = [
  {
    id: "follow-one-word",
    sections: ["Follow one word", "How to play with it"],
    title: "Follow one word",
    look: "“it” is the query. Thicker arcs carry more weight, and the strip under the arcs names its strongest route.",
    step: 0,
    focus: "Model attention routing diagram",
  },
  {
    id: "five-stages",
    sections: ["The five stages", "How the five stages fit together"],
    title: "The five stages",
    look: "Press Query, Keys, Score, Softmax and Mix under the arcs, and the inspector card shows each stage in turn.",
    step: 0,
    state: { stage: 0 },
    focus: "Model attention routing diagram",
  },
  {
    id: "heads-layers",
    sections: ["Heads and layers"],
    title: "Eight heads, one sentence",
    look: "Switch Layer and move the Attention head slider at the top of this card. The sentence stays the same while the routes change.",
    step: 0,
    state: { layer: 0, head: 0 },
    focus: "Query key value inspector",
  },
  {
    id: "why-divide",
    sections: ["Why divide by 8?"],
    title: "The division by 8",
    look: "The Scale line divides the dot product by √64, which is 8, before softmax turns the scores into shares.",
    step: 2,
    state: { stage: 2 },
    focus: "Query key value inspector",
  },
  {
    id: "check-pair",
    sections: ["Check the arithmetic for one pair"],
    title: "One pair, checked by hand",
    look: "The Softmax line works out the share “it” gives “animal” and prints the model’s own figure beside it.",
    step: 2,
    state: { stage: 3 },
    focus: "Query key value inspector",
  },
  {
    id: "edit-key",
    sections: ["Edit a key"],
    title: "Edit a key",
    look: "Type a new number into one of the K boxes, then watch every bar under the word chips move, not only the one you edited.",
    step: 1,
    state: { stage: 1 },
    focus: "Query key value inspector",
  },
  {
    id: "no-scaling",
    sections: ["Switch off the scaling", "Divide, or don't"],
    title: "Scaling switched off",
    look: "Read Largest weight and Saturated rows, then set Score scaling back to ÷ √64 and read them again.",
    step: 2,
    state: { scaling: "raw", stage: 3 },
    focus: "Query key value inspector",
  },
  {
    id: "zero-values",
    sections: ["Empty the packets", "Zero the values"],
    title: "Every value set to zero",
    look: "The weights and arcs stay where they were, while the Mixed value chip and Mix length drop to zero.",
    step: 3,
    state: { values: "all", stage: 4 },
    focus: "Query key value inspector",
  },
  {
    id: "causal-triangle",
    sections: ["The one-way rule"],
    title: "The causal mask",
    look: "In the Characters view, the hatched cells above the diagonal are later characters that each row may not read.",
    step: 4,
    state: { matrixView: "characters" },
    focus: "Attention matrix",
  },
  {
    id: "mask-lifted",
    sections: ["Lift the one-way rule", "Lift the mask"],
    title: "Mask lifted",
    look: "Layer 2 head 4 with the mask lifted: shares appear above the diagonal, and the strip under the arcs adds on later characters.",
    step: 4,
    state: { mask: "bidirectional", layer: 1, head: 3 },
    focus: "Attention matrix",
  },
  {
    id: "previous-char",
    sections: ["A route that is not what it looks like", "Test a route before you believe it"],
    title: "A look-one-back head",
    look: "Layer 1 head 3 in the Characters view: the dark stripe just below the diagonal means each character reads the one before it.",
    step: 5,
    state: { layer: 0, head: 2, matrixView: "characters" },
    focus: "Attention matrix",
  },
  {
    id: "repeated-name",
    sections: ["A route that follows the content"],
    title: "A route that follows the name",
    look: "Layer 2 head 3 with the second Mark as the query: its thick arc goes to Jones. Then press Name moved earlier.",
    step: 5,
    state: { text: "Mark Jones met Anna Blake and then Mark", selected: 7, inspected: 1, layer: 1, head: 2 },
    focus: "Model attention routing diagram",
  },
];

export default scenes;
