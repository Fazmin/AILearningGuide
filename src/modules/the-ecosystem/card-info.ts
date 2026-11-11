import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Sample model card": {
    title: "A fictional 7B release with real-shaped fields",
    summary:
      "harbor-lm-7b is authored for this lab, but every field is one a real model card should make you read. Three release choices, License, Training code, and Data information, decide which kind of release it is. Two fields are computed from the card's own numbers: the evaluation error and the training energy.",
    whatYouSee: [
      "The release drawn as a box. License sits on the lid as the rules printed on the box, and Weights, Inference code, Training code, and Data information are tiles marked ✓ in the box or ✗ held back.",
      "Rows for base model, fine-tunes, weights, training code, data information, license, evaluation harness, and training energy. Red and green text marks what is missing or present.",
      "In the Evaluation harness row, 40 dots with the 28 right answers filled, then the 70% score on a 0–100% line with a thick bar for one standard error and a thin line for the 95% interval. In the Training energy row, three linked boxes: compute, GPU-hours, MWh.",
      "License (Research only, Custom community, Apache-2.0), Training code (not released or released), and Data information (not published or published).",
      "A badge naming the release tier, and a note listing what is still missing for the Open Source AI Definition.",
    ],
    howItWorks: [
      "Evaluation harness: standard error = √(p(1 − p)/n) = √(0.7 × 0.3 / 40) ≈ 0.072, so ±7.2 points, and a 95% interval of about 56–84%.",
      "Training energy: compute ≈ 6·N·D = 6 × 7×10⁹ × 2×10¹² = 8.4×10²² FLOP. At 40% of an A100's 312 TFLOP/s dense BF16 peak that is about 187,000 GPU-hours; at 400 W per GPU, about 75 MWh.",
      "The tier is computed: a license that is not any-purpose makes it restricted open weights; Apache-2.0 without training code or data information makes it permissive open weights; Apache-2.0 with both has the components OSAID 1.0 asks for.",
    ],
    controls: [
      "License, Training code, and Data information on this card. Intended use is on Is this use allowed.",
      "Comparison worth running: Apache-2.0 with nothing else released, then with Training code and Data information. The tier changes; the weights file does not.",
    ],
    notice: [
      "For scale, Meta reported 184,320 A100 GPU-hours to pretrain Llama 2 7B on 2T tokens, close to this 6ND estimate at 40% utilization.",
      "Forty items cannot separate two models whose scores differ by a few points.",
    ],
    limits: [
      "In this lab: the model, the harness, and the licenses are fictional. The energy figure counts GPU board power only, for one successful run, with no cooling overhead, failed runs, or experiments.",
      "In general: a real card can omit the field you need. A missing field is information, not a blank to fill with hope, and nothing forces a card to be accurate.",
    ],
  },

  "Release spectrum": {
    title: "Four kinds of release, six things you might get",
    summary:
      "Columns are categories of release, from API-only to open-source AI as the Open Source Initiative defines it. Rows are what you might need. The column this card falls into is highlighted and moves as you change License, Training code, and Data information.",
    whatYouSee: [
      "A table with columns API only, Open weights with a restricted license, Open weights with a permissive license, and Open source AI (OSAID 1.0 components). The current card's column is shaded and marked this card.",
      "Above the columns, a road from less open to more open with a gate before each stop. Under each gate is what the card needs to pass it, marked ✓ or ✗, and the large dot is where the card has got to.",
      "Each cell pairs a mark with its word: a filled ✓ for yes or required, a half-filled disc for limited or per terms of service, a dashed ring for not required or not always required, and ✗ for no. Color is never the only signal.",
    ],
    howItWorks: [
      "OSAID 1.0 (Open Source Initiative, October 2024) requires the model parameters, the complete code used to train and run the system, and data information detailed enough that a skilled person could build a substantially equivalent system, under terms that allow any purpose.",
      "It does not require releasing all training data in every case, which is why the last row says \"not always required\".",
      "\"Not required\" in an open-weights column means the category promises nothing; individual releases sometimes include more.",
    ],
    controls: [
      "No controls on this card. The highlighted column follows Sample model card.",
      "Comparison worth running: Research only with everything released still sits in the restricted column, because the terms are not any-purpose.",
    ],
    notice: [
      "Open weights and open source differ on code, data information, and terms, not on whether you can download a file.",
      "Only the API column withholds the weights entirely, and it is the only one where the provider can change behavior under you.",
    ],
    limits: [
      "In this lab: four columns simplify a continuum. Real releases mix components, and OSI's own review weighs the exact terms.",
      "In general: OSAID 1.0 is one definition. Other groups define openness differently, and some prominent releases called open by their authors do not meet it.",
    ],
  },

  "Is this use allowed": {
    title: "A use needs clauses; the strictest one decides",
    summary:
      "Each intended use needs one or more permissions. The lab looks up each needed clause in the card's license and reports the strictest outcome. A separate readout keeps what the license cannot answer, where the data came from, in view.",
    whatYouSee: [
      "Intended use: Research paper, Hosted product, Fine-tune and ship, or Train on its outputs.",
      "Under Permission, the license's five clauses in a column. Each clause the use needs is a row marked ✓ allowed, ◐ allowed with conditions, or ✗ not allowed, with the clause text; the others read not needed for this use.",
      "A line that runs down through the marks. It is solid until a ◐, dashed after it, and dotted after a ✗, so what reaches the bottom is the strictest mark.",
      "Under Provenance, Data sources and Consent, each marked ? while it is unknown.",
      "License match, Clauses checked, and Still unknown.",
    ],
    howItWorks: [
      "Needs: research paper → research use; hosted product → commercial use; fine-tune and ship → commercial use, modification, redistribution; train on outputs → commercial use and use of outputs to train another model.",
      "Verdict = worst outcome among the needed clauses. Apache-2.0 redistribution is conditional because the license and notices must travel with the copy.",
      "Still unknown depends only on Data information, never on the license.",
    ],
    controls: [
      "Intended use here; License, Training code, and Data information on Sample model card.",
      "Comparison worth running: Train on its outputs under Custom community, then under Apache-2.0.",
    ],
    notice: [
      "A download button is not a grant. Research only blocks the hosted product before any cost or quality question arises.",
      "Under Apache-2.0 the hosted product is allowed, and consent for the training data is still unknown.",
    ],
    limits: [
      "In this lab: three fictional licenses with clause types drawn from real ones. This is not legal advice, and real terms vary by jurisdiction and version.",
      "In general: API terms of service also govern outputs, and a fine-tune usually inherits the base model's license along with any data problems the name \"fine-tune\" does not mention.",
    ],
  },
};

export default cardInfo;
