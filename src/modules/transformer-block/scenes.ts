import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Every scene starts from the lab's defaults: a
 * pre-norm block with every sublayer on, Input scale 1×, the token `mat` in slot 5 (`position: 5`),
 * the GPT-2 small parameter preset, and the Position card on Swap cat ↔ mat with position
 * information on.
 */
const scenes: LabScene[] = [
  {
    id: "words-as-numbers",
    sections: ["Words as lists of numbers"],
    title: "Eight numbers per word",
    look: "The Input x row draws mat’s eight numbers as bars above and below a middle line, with its norm printed beside them; press another Token button to compare.",
    focus: "Transformer block",
  },
  {
    id: "belt-without-specialists",
    sections: ["A conveyor belt that only adds", "The residual stream"],
    title: "The belt with nothing added",
    look: "With Self-attention and MLP both off, each diamond sits at the end of its input lane: the belt still carries mat’s numbers through to the output.",
    step: 2,
    state: { attention: false, mlp: false },
    focus: "Residual stream",
  },
  {
    id: "one-trip",
    sections: ["One trip through the block"],
    title: "One trip through the block",
    look: "Follow mat down the Transformer block card: rows on the left rail are the belt, and each indented row works on a copy and writes it back at the next +.",
    focus: "Transformer block",
  },
  {
    id: "post-norm-rows",
    sections: ["Pre-norm and post-norm", "Normalization"],
    title: "Post-norm placement",
    look: "With Placement on Post-norm, the Layer norm rows move onto the left rail, so they resize the belt itself instead of a copy.",
    step: 2,
    state: { placement: "post" },
    focus: "Transformer block",
  },
  {
    id: "order-from-position",
    sections: ["Where order comes from"],
    title: "Position information on",
    look: "Word order is on Swap cat ↔ mat with Position information On, so every word’s output changes; set Position information to Off and the block can no longer tell the sentences apart.",
    step: 4,
    focus: "Position",
  },
  {
    id: "attention-off",
    sections: ["Switch off the only step that crosses words", "How to play with it"],
    title: "Self-attention switched off",
    look: "With Self-attention off, Output shift if other words change has dropped to zero; click Self-attention back on in the Transformer block card and watch it return.",
    step: 1,
    state: { attention: false },
    focus: "Residual stream",
  },
  {
    id: "huge-input",
    sections: ["Make the input huge"],
    title: "Huge input, no tidying",
    look: "With Layer norm off and Input scale at 8×, both heads in the Self-attention row give their whole share to one word; turn Layer norm back on and the shares spread out again.",
    step: 0,
    state: { norm: false, inputScale: 8 },
    focus: "Transformer block",
  },
  {
    id: "additions-line-up",
    sections: ["Watch the additions line up"],
    title: "The additions line up",
    look: "In each row from d0 to d7, each lane starts where the one above it ended, and the diamond sits at the end of the bottom lane; switch MLP off and the bottom lanes vanish.",
    step: 2,
    focus: "Residual stream",
  },
  {
    id: "post-norm-stream",
    sections: ["Move the tidying onto the belt"],
    title: "Tidying on the belt",
    look: "With Placement on Post-norm, the diamonds leave the ends of the lanes; drag Input scale and watch Output norm hold still.",
    step: 2,
    state: { placement: "post" },
    focus: "Residual stream",
  },
  {
    id: "parameter-count",
    sections: ["Count where the parameters live"],
    title: "Where the parameters live",
    look: "The bar splits one GPT-2 small block into attention, the hatched MLP and a thin sliver for the norms; click Llama 2 7B and compare MLP share.",
    step: 3,
    focus: "Parameter count",
  },
  {
    id: "word-order-off",
    sections: ["Test word order"],
    title: "Swapped words, no position",
    look: "With Position information Off, every Result in the table says same output; switch it to On and watch the Output shift column fill in.",
    step: 4,
    state: { positionInfo: false },
    focus: "Position",
  },
];

export default scenes;
