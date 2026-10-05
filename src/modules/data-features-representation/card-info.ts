import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "The street": {
    title: "Six listings before they become numbers",
    summary:
      "A short street of townhouses. Windows count rooms, door paint is a category, a tree is a yes/no park, and the hanging tag is the price label. Clicking a house only selects that listing; it does not change the raw values the later packing slip will read.",
    whatYouSee: [
      "Six illustrated facades: Alder 12 through Fir 1. The selected house is ringed.",
      "Windows equal the room count. Red doors are solid, blue doors have two bars, green doors have two dots.",
      "A tree only when the row is near a park. A high or low pill is the price label, hung off the building.",
    ],
    howItWorks: [
      "The six rows are constants. Rooms is already a number. Color is red, blue, or green. Park is yes or no. High price is y and is drawn as a tag, not as a window.",
      "Clicking a facade writes `selected`. Later cards read that id for the packing slip, the lineup, the gate, and both maps.",
    ],
    controls: [
      "Click a house to select that listing.",
      "Comparison worth running: select Alder 12, then Fir 1, and notice only the ring moves. The later packing is what reorders the street.",
    ],
    notice: [
      "Color and park are categorical. Writing them as integers is already a design choice.",
      "The tag is not a feature unless you leak it. This card keeps it hanging off the building.",
    ],
    limits: [
      "In this lab: six authored listings. The street cards have no missing-value path, no scaler fitted on a holdout, and no pipeline object; the Messy rows card spoils a copy of the table to show what those cost.",
      "In general: a production table has types, leakage, and target definition problems this street does not simulate. Train/val/test is a later data concept; this page does not split the six rows.",
    ],
  },

  "Packing slip": {
    title: "You assign the numbers a door color will use",
    summary:
      "The selected house is packed into a named slip. Integer stamps, one cubby per color, or same-length rulers choose the recipe. The red, blue, and green pads are the hand step: you decide what integer each door becomes before distance can run.",
    whatYouSee: [
      "The selected facade beside How to pack the door: integer stamps, one cubby each, and same-length rulers.",
      "Three stamp pads, 1 through 5, idle when one-hot is on.",
      "A packing slip listing each slot, a bar, and the number actually stored. High price is named as off the slip.",
    ],
    howItWorks: [
      "Integer: the slip is [rooms, colorCode, park]. Park is 1 or 0.",
      "One-hot: the slip is [rooms, red, blue, green, park]. The stamp pads are ignored. Distinct colors are orthogonal, distance √2.",
      "Normalized: rooms / 4 and colorCode / 5, plus park. The pads still set the codes that get rescaled. After rescaling, park's 0/1 gap is the widest slot.",
    ],
    controls: [
      "How to pack the door switches the recipe. The three pads rewrite the integer codes.",
      "Comparison worth running: on integer stamps, set blue to 5, then switch to one cubby each and watch the slip drop those codes.",
    ],
    notice: [
      "Integer stamps invent an order. One-hot invents equal distance instead.",
      "Normalization only rescales what you already chose. It does not recover a feature you never wrote down.",
    ],
    limits: [
      "In this lab: you type the codes on pads. There is no learned embedding and no target encoding.",
      "In general: a fitted scaler must be frozen on training statistics. Fitting it on the test rows is leakage — a later lab.",
    ],
  },

  "Who stands nearest": {
    title: "The same listings after your packing",
    summary:
      "The six houses line up by Euclidean distance on the current packing slip. The nearest house is labelled nearest. Bars show which slip slots created that gap. A callout appears if the first two coordinates would have named someone else.",
    whatYouSee: [
      "A lineup starting at the selected house, then the other five in slip-distance order, each labelled with its rank and its distance d; the selected house reads d 0.00. When two houses are equally near, both are marked tied nearest.",
      "One bar per slip slot: |Δ| between the selected house and the nearest house.",
      "Metrics Nearest on the slip, Distance, and First-two nearest.",
    ],
    howItWorks: [
      "Distance is ordinary Euclidean distance on the whole slip, including park and extra one-hot slots. Rows at exactly the nearest distance are all reported; the order among them is only by id.",
      "First-two nearest uses only the first two numbers. When those names disagree, the plane is a projection of a longer vector.",
    ],
    controls: [
      "Click a house in the lineup to make it the selected listing. Encoding and the stamp pads reorder everyone else.",
      "Comparison worth running: keep Alder 12 selected, stamp blue as 5, then switch How to pack the door and see whether the nearest name stays put.",
    ],
    notice: [
      "A neighbor flip is the lesson. The apartments did not move; the representation did.",
      "The hanging tag is decoration here. Distance does not read High price.",
    ],
    limits: [
      "In this lab: six points and Euclidean distance. The first-two callout is the only projection shown; there is no scatter plot.",
      "In general: neighbors in a learned embedding can be useful and still be wrong as facts. Distance measures the representation you chose.",
    ],
  },

  "The gate": {
    title: "One threshold, three possible stamps",
    summary:
      "The model class is a 1-D threshold: ŷ = 1[stamp ≥ t]. Stamp the gate reads chooses rooms, door color, or park. The same six tags therefore meet a different yard when the coordinate changes.",
    whatYouSee: [
      "Stamp the gate reads, three segments, and Gate height t from 0 to 5.",
      "Two yards — Predicted low and Predicted high — with the six facades. A struck-through pill marks a miss.",
      "Metrics Tags the gate matches, and the selected listing's ŷ.",
    ],
    howItWorks: [
      "Rooms uses the raw count, or rooms/4 when packing is normalized. Park uses 0/1. Color uses the integer (or scaled) stamp, except under one-hot, where it uses the red bit only.",
      "A match is a yard that agrees with the hanging tag. There is no holdout on this page.",
    ],
    controls: [
      "Stamp the gate reads and Gate height t. How to pack the door changes how color and rooms are read.",
      "Comparison worth running: rooms at t = 2.5 versus door color at t = 2.5. The rule family is the same; the partitions are not.",
    ],
    notice: [
      "A better score on these six is not a better representation of the world. It is a better cut of this sample.",
      "Under one-hot, a threshold on the red bit cannot separate blue from green.",
    ],
    limits: [
      "In this lab: one axis, one threshold, six houses. This is not the linear model of the next lab, and it is not cross-validated.",
      "In general: the same hypothesis class on a different feature map is a different learning problem. That is why representation work precedes model work.",
    ],
  },

  "Two maps": {
    title: "Nearby means a geometry you chose or fitted",
    summary:
      "Left, the six houses sit in three color cubbies: one-hot says every other door is equally far. Right, an authored map places similar park-and-price listings nearby. The two nearest names can disagree.",
    whatYouSee: [
      "Three color docks with the houses that share that door paint.",
      "A later map: six facades at fixed authored coordinates, with a line to the embedding neighbor.",
      "Metrics Nearest on the later map, Embed dist, and One-hot color nearest.",
    ],
    howItWorks: [
      "Embedding coordinates are constants in the source. Distance is ordinary Euclidean distance in that plane.",
      "One-hot color distance is 0 when the doors match and √2 otherwise. It does not use rooms, park, or the tag.",
    ],
    controls: [
      "Click a house on either map to change the selected listing.",
      "Comparison worth running: select Alder 12 (red, park, high) and notice which neighbor is close on the later map but √2 away in one-hot color.",
    ],
    notice: [
      "One-hot says every pair of distinct colors is equally far. An embedding is allowed to disagree.",
      "Nearby on the later map is a claim about the drawn geometry, not a fact about the apartments.",
    ],
    limits: [
      "In this lab: the embedding is hand-placed to make the geometry readable. No loss was minimized to put the houses there.",
      "In general: a trained embedding moves coordinates so a later loss is small. Similar items nearby is the usual hope, not a guarantee, and not a proof about the world.",
    ],
  },

  "Pixels and words": {
    title: "A picture is nine numbers; a sentence is a list of IDs",
    summary:
      "A 3×3 picture of 0/1 pixels, read row by row, is a vector of nine numbers, and it can be compared with three template pictures by the same straight-line distance used on the street. A sentence is turned into IDs by looking each word up in a fixed eight-word vocabulary. Both are previews of how later labs turn pictures and text into numbers.",
    whatYouSee: [
      "A 3×3 pad of buttons. Each cell prints its value, 1 for ink or 0 for blank, and is a pressed or unpressed button, so the state never depends on colour alone.",
      "A vector chip with the nine numbers, row by row, and a table of the three template pictures (plus, ring, bar) with their pixels, their distance to your drawing, and their rank.",
      "Sentence, four presets, and a table of each word, its ID, and whether it is on the list. A word that is not on the list shows ID 0.",
      "Metrics for the nearest template and its distance, the sentence as a list of IDs, the ID gap from cat to dog and from cat to mat, and the one-hot gap between any two words.",
    ],
    howItWorks: [
      "The drawing is the string of nine 0/1 characters stored in state; pressing a cell flips one character. The vector is those characters read as numbers. Distance to a template is the Euclidean distance between the two nine-number vectors, so the squared distance is the count of pixels that differ and one flipped pixel changes it by exactly 1.",
      "The vocabulary is <unk> 0, the 1, cat 2, sat 3, on 4, mat 5, dog 6, ran 7. A sentence is split on spaces and each word is replaced by its position; a word not on the list becomes 0. Nothing here is learned.",
      "The ID gap is the absolute difference of two IDs: 4 from cat to dog and 3 from cat to mat. One-hot would put any two different words exactly √2 apart.",
    ],
    controls: [
      "Press pixels to draw, and use Sentence to choose a sentence.",
      "Comparison worth running: with the default plus, switch off the left and right arms (row 2, columns 1 and 3). The drawing becomes the bar, which is now 0.00 away, and the plus is 1.41 away. Then set Sentence to the one with a bird and watch the word bird become ID 0.",
    ],
    notice: [
      "A picture is a point in nine dimensions, so the street's nearest-neighbour idea carries over unchanged.",
      "IDs are labels. By subtraction cat is nearer to mat than to dog, an order the numbering made up, which is the integer-stamp trap with words.",
    ],
    limits: [
      "In this lab: nine pixels that are only 0 or 1, three fixed templates, and eight fixed words. The vocabulary is defined in the card and is not a real tokenizer; no learned embedding is involved.",
      "In general: real images have thousands of pixels with brightness and color channels, and real text is split into subword pieces from a vocabulary of tens of thousands, then mapped to learned vectors. Tokens & embeddings is where that happens.",
    ],
  },

  "Messy rows": {
    title: "One bad cell can move a summary and flip a neighbour",
    summary:
      "The six listings, spoiled three ways: one room count is blank, one listing appears twice, and one price tag is flipped. A small table summary (rows, mean rooms, share of high prices) and a three-nearest-neighbour answer for a new listing show what each problem does, and how handling a blank cell changes it.",
    whatYouSee: [
      "Missing value, Handle missing (read as 0, drop the row, fill with the mean), Duplicate row, and Mislabelled row, each a set of buttons.",
      "A table of the listings with a What is wrong column in words, so the spoiled rows are named and not only shaded.",
      "Metrics: rows in the summary, mean rooms, high-price share, the three-nearest answer, leave-one-out accuracy, and whether the answer changed against the clean table.",
      "A note that names the three nearest listings with their distances and tags, and says what each switched-on problem did.",
    ],
    howItWorks: [
      "The new listing is 3 rooms, red door, no park. Distances use the default integer stamps (red 1, blue 2, green 3). The answer is the majority tag among the three nearest rows, ties in distance broken by table order.",
      "Missing value blanks Dock 2's room count. Read as 0 puts the row at 0 rooms; drop the row removes it; fill with the mean uses the mean of the other five rooms, 1.8. Duplicate row appends a copy of Alder 12. Mislabelled row flips Dock 2's price tag from high to low.",
      "Leave-one-out accuracy predicts each row from its nearest other row and counts how many predictions match the row's own tag. A copy's nearest other row is its twin at distance 0.",
    ],
    controls: [
      "Missing value, Handle missing, Duplicate row, and Mislabelled row.",
      "Comparison worth running: switch Missing value on and move Handle missing across its three settings. Reading it as 0 and dropping the row both flip the answer from high to low; filling with the mean keeps it high. Then switch Duplicate row on and watch leave-one-out accuracy rise from 3 of 6 to 5 of 7 with no new information.",
    ],
    notice: [
      "The same blank cell gives different answers depending on how it is handled. The handling is part of the encoding.",
      "A duplicate raises a leave-one-out score without adding anything: each copy is checked against its twin.",
      "A single flipped tag moves the share of high prices from 50 percent to 33.3 percent and the neighbour vote with it.",
    ],
    limits: [
      "In this lab: six rows, three authored problems chosen so that each one can be seen, one fixed query, and a table that is a copy of the street. Real tables are larger, and one bad cell rarely flips an answer.",
      "In general: real data has many more kinds of mess (outliers, inconsistent units, sampling bias, rows that leak the label), and a duplicate that straddles a train and a test split inflates the score the same way the leave-one-out check does here.",
    ],
  },
};

export default cardInfo;
