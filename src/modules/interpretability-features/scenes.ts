import type { LabScene } from "@app/module-sdk";

/**
 * Lab setups the lesson opens from its sections. Positions index the default probe text, "hath the
 * king a good horse? KING HENRY: aye, he hath.", where 4 is the first space, 28 the K of KING and 43
 * the comma after "aye". Every scene keeps that text, because the lesson's readouts come from it.
 * `examples: "context"` is Examples shown on Top 8 from position 4 on.
 */
const scenes: LabScene[] = [
  {
    id: "sparse-code",
    sections: ["Unpacking with a sparse autoencoder", "The proposed fix: sparse autoencoders"],
    title: "A dense list, unpacked",
    look: "The lower panel has a slot for each of the 1,024 features, and only the spikes are switched on at this character.",
    step: 0,
    focus: "Residual to sparse code",
  },
  {
    id: "measure-rebuild",
    sections: ["Measuring a rebuild"],
    title: "How much the rebuild misses",
    look: "The dashes on the bars mark the rebuild, and the ‖x − x̂‖² / ‖x‖² readout gives the share it misses at the first space.",
    step: 0,
    focus: "Residual to sparse code",
  },
  {
    id: "residual-walk",
    sections: ["Residual to sparse code", "How to play with it"],
    title: "From the first space to KING",
    look: "Click position 0, then 1, then the K of KING, and watch how many of the largest features carry a window start mark.",
    step: 0,
    state: { token: 4 },
    focus: "Residual to sparse code",
  },
  {
    id: "keep-one",
    sections: ["Sparsity and superposition"],
    title: "Keep one feature",
    look: "Drag Keep top k features up from 1 and watch the unexplained share fall on the chart, then compare Keep 1 with Keep all 32.",
    step: 1,
    state: { keep: 1 },
    focus: "Sparsity and superposition",
  },
  {
    id: "feature-876",
    sections: ["Feature evidence"],
    title: "Read the position badges",
    look: "Read the position badge on each of #876’s eight examples, then switch Examples shown to Top 8 from position 4 on and compare.",
    step: 2,
    focus: "Feature evidence",
  },
  {
    id: "steer-160",
    sections: ["Steer with a feature"],
    title: "Push along #160",
    look: "Compare Change in log p with the random range beside it, then press Continue 12 characters and read the three versions.",
    step: 4,
    focus: "Steer with a feature",
  },
  {
    id: "few-features",
    sections: ["A few features carry most of the list", "Reconstruction"],
    title: "The biggest few do most",
    look: "Read Keep 1, Keep 8 and Keep all 32 under the chart: the curve drops steeply at first and then flattens.",
    step: 1,
    focus: "Sparsity and superposition",
  },
  {
    id: "overlap",
    sections: ["Overlap: superposition you can measure", "Overlap"],
    title: "Overlap you can measure",
    look: "Under Overlap between decoder directions, read All: mean · max, then Most similar pair and the sign of Their cosine (signed).",
    step: 1,
    focus: "Sparsity and superposition",
  },
  {
    id: "away-from-start",
    sections: ["Where a feature's examples come from", "Where examples come from"],
    title: "#876 away from the window start",
    look: "These are #876’s best examples from position 4 on, so compare their activations with Top 8 overall, where every example sits at the window’s start.",
    step: 2,
    state: { examples: "context" },
    focus: "Feature evidence",
  },
  {
    id: "score-space",
    sections: ["Scoring a label: precision and recall", "Scoring labels"],
    title: "Score the Space label",
    look: "Read Precision, Recall and the line of hits, misses and false alarms, then lower Activation threshold to 2 and read them again.",
    step: 3,
    state: { feature: 121, hypothesis: "space" },
    focus: "Feature evidence",
  },
  {
    id: "steer-683",
    sections: ["Steering: does turning a feature up do what its label says?", "Steering"],
    title: "A y feature that lowers y",
    look: "Read p(“y”) before → after and Change in log p: adding #683 makes its own label character less likely, and Feature vs random, in SDs says that is no accident.",
    step: 4,
    state: { steerFeature: 683, strength: 5 },
    focus: "Steer with a feature",
  },
];

export default scenes;
