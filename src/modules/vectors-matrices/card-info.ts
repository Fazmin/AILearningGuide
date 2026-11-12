import type { ModuleCardInfo } from "@app/module-sdk";

/**
 * One explanation per card. Every card in this lab is exact arithmetic on a hand-written map of
 * eight words, so each entry says which numbers were authored and where the real idea differs.
 */
const cardInfo: ModuleCardInfo = {
  "A word as two numbers": {
    title: "A list of numbers is an arrow",
    summary:
      "A vector is an ordered list of numbers, and with two numbers you can draw it: go the first number across and the second number up. The arrow's direction says what kind of thing it describes and its length says how strongly.",
    whatYouSee: [
      "A map with two axes. Across is size: right of the centre is bigger than a person, left is smaller. Up is alive: above the centre is a living thing, below is a machine.",
      "Eight words placed on the map. Round dots are living things and square dots are machines, so the two kinds differ by shape as well as by position.",
      "A green arrow from the centre to the current list, with a solid blue leg for the size number and a dashed purple leg for the alive number. Its length is written beside it.",
      "The list itself in large type, and readouts for How much: length, What kind: direction, and Nearest word on the map.",
      "A box that works the length out from the two legs with the live numbers.",
    ],
    howItWorks: [
      "The arrow tip is the point (size, alive). Picking a word sets the tip to that word's two authored numbers.",
      "Length is the square root of size squared plus alive squared. For horse that is the square root of 0.81 + 1.44 = 2.25, which is 1.5.",
      "What kind: direction is read from the signs of the two numbers: big or small, living or machine.",
      "Nearest word is the word whose dot is the shortest straight-line distance from the tip.",
    ],
    controls: [
      "Word to read picks one of the eight words; you can also click a word on the map.",
      "Drag the round handle on the arrow tip to change both numbers at once.",
      "Size (first number) and Alive (second number) move the tip from the keyboard, one tenth at a time.",
      "Put the arrow back on the word appears once you have moved the tip away from it.",
    ],
    notice: [
      "Drag the tip straight outward and the length grows while What kind: direction stays the same.",
      "Drag it across an axis and the direction readout changes even if the length barely does.",
    ],
    limits: [
      "In this lab: the sixteen numbers were typed in by hand and tuned so the arithmetic is clean. The names size and alive were chosen for the lesson. Nothing was learned or measured.",
      "In general: a real model's word vectors have hundreds or thousands of numbers, they are learned, and their individual numbers rarely have a meaning you can name. Length and direction still exist; the picture does not.",
    ],
  },
  "Add and stretch": {
    title: "The two things you can do to a vector",
    summary:
      "Adding two vectors adds their matching numbers, which puts the tail of one arrow on the tip of the other. Multiplying a vector by a number stretches, shrinks or flips it along its own line. Everything else in this lab is built from these two moves.",
    whatYouSee: [
      "A black arrow from the centre to the word you start from.",
      "A magenta arrow, the step after multiplying, with its tail on the tip of the word. The label beside it is the multiplier.",
      "A grey dashed arrow showing the same step at × 1, for comparison.",
      "A green arrow from the centre to the result, and a dashed ring round the word nearest to it.",
      "A small table that does the addition one number at a time, and a readout naming the nearest word and its distance.",
    ],
    howItWorks: [
      "The step is one of three fixed vectors: bigger (1, 0), more alive (0, 1), or bigger and less alive (1, −1).",
      "Times multiplies both numbers of the step. The result is the word's two numbers plus the multiplied step's two numbers.",
      "Horse (0.9, 1.2) plus 1.0 × bigger is (1.9, 1.2), which is 0.32 from whale. With × −1.7 it is (−0.8, 1.2), exactly cat.",
    ],
    controls: [
      "Start from picks the word.",
      "Step to add picks one of the three steps.",
      "Times (multiply the step by) runs from × −2.0 to × 2.0.",
    ],
    notice: [
      "A multiplier between 0 and 1 shrinks the step, above 1 stretches it, and below 0 flips it. The step never turns to face a new direction.",
      "With the step bigger, only the size number of the result changes, because the step's alive number is zero.",
    ],
    limits: [
      "In this lab: the three steps and the eight words are authored, and cat sits exactly 1.7 to the left of horse by design.",
      "In general: learned word vectors sometimes show the same kind of arithmetic, as in the well-known king, man, woman and queen example, but it is a tendency that holds for some word pairs and fails for others.",
    ],
  },
  "How alike are two arrows": {
    title: "The dot product, the shadow and the cosine",
    summary:
      "The dot product multiplies the matching numbers of two vectors and adds the results. It is positive when the arrows agree and negative when they point apart, but it also grows when either arrow gets longer. Dividing by both lengths gives the cosine similarity, which keeps direction and ignores length.",
    whatYouSee: [
      "Two arrows from the centre: the first word in green and the second in black. A grey dashed arrow shows the second word's own position once you have turned or stretched it.",
      "A dotted line through the second arrow, a thick magenta bar along it for the shadow of the first arrow, and a fine dotted drop from the first arrow's tip to the end of the shadow.",
      "An arc with the angle between the two arrows.",
      "Three bars that grow right for positive and left, with stripes, for negative: size × size, alive × alive, and their sum, the dot product.",
      "A cosine scale from −1 (opposite) through 0 to +1 (same way), with a round marker and a verdict in words.",
    ],
    howItWorks: [
      "Dot product = first size × second size + first alive × second alive. For horse and whale: 0.9 × 2.0 + 1.2 × 0.9 = 1.80 + 1.08 = 2.88.",
      "The shadow is the dot product divided by the length of the second arrow, so shadow × length of second gives the dot product back.",
      "Cosine = dot product ÷ (length of first × length of second). For horse and whale: 2.88 ÷ (1.50 × 2.19) = 0.88.",
      "Turn the second arrow rotates it about the centre without changing its length. Stretch the second arrow multiplies both its numbers.",
    ],
    controls: [
      "First arrow and Second arrow pick the two words. Picking a new second word resets the turn and the stretch.",
      "Turn the second arrow goes from 0° to 360° in steps of 15°.",
      "Stretch the second arrow goes from × 0.5 to × 1.5.",
      "The Predict first question at the top can be answered before you try the turn.",
    ],
    notice: [
      "At 180° every bar flips to the other side and the dot product becomes its own negative.",
      "Stretching changes the dot product and leaves the cosine marker where it was.",
      "Horse and bus give bars of +1.44 and −1.44 that cancel to exactly 0: the arrows meet at 90°.",
    ],
    limits: [
      "In this lab: similar things point the same way only because the map was drawn that way. Every number is computed from the sixteen authored ones; the bar scale is fixed for each pair of words so that stretching visibly lengthens the bars.",
      "In general: a model's dot products and cosines mean what training made them mean. For some learned embeddings the cosine has been shown to give arbitrary results, so a high cosine is evidence of likeness, not proof.",
    ],
  },
  "A matrix moves every arrow": {
    title: "Four numbers that move the whole map",
    summary:
      "A matrix turns any vector into a new vector: each output number is the dot product of one row with the input. Its first column is where the size arrow (1, 0) lands and its second column is where the alive arrow (0, 1) lands, and every other arrow follows from those two.",
    whatYouSee: [
      "The map after the move. Pale green lines are the grid carried along by the matrix; the shaded shape is one grid square after the move.",
      "A thick blue arrow for the first column and a thick purple arrow for the second, with grey dashed arrows showing where they started.",
      "All eight words at their new positions. The word you follow is green, with a hollow mark and a dotted line back to where it started.",
      "A black dashed path that walks the word's size number along the first column and then its alive number along the second, ending on the word.",
      "A table under the map with the two row calculations and the same result worked by columns.",
    ],
    howItWorks: [
      "New size = top left × size + top right × alive. New alive = bottom left × size + bottom right × alive.",
      "The same result is size × first column + alive × second column, which is what the dashed path draws.",
      "The area in the note is the size of top left × bottom right − top right × bottom left, the determinant.",
      "Pressing a ready-made matrix glides the map to its new position so you can watch it travel; the numbers change at once.",
    ],
    controls: [
      "Word to follow picks the word whose calculation is written out.",
      "Ready-made matrices sets all four numbers: leave alone, stretch, rotate, shear, flip or squash.",
      "Top left, Top right, Bottom left and Bottom right each run from −2.0 to 2.0. The left pair is the first column and the right pair is the second.",
    ],
    notice: [
      "The thick arrows always end at the points the two columns name.",
      "Grid lines stay straight and evenly spaced, and the centre never moves, whatever the four numbers are.",
      "Raising Top right moves living words right and machines left, in proportion to their alive number.",
    ],
    limits: [
      "In this lab: the matrix has four numbers and acts on a hand-written two-number map. A word thrown beyond the edge is pinned to the edge and marked off the map. There is no bias, so the centre cannot move.",
      "In general: a layer's matrix has thousands to millions of numbers and is followed by an added bias and a bending step. A matrix alone can only make straight-line moves, however large it is.",
    ],
  },
  "One after another": {
    title: "Two matrices in a row, and why order matters",
    summary:
      "Doing one matrix and then another is the same as doing a single matrix, their product. Its columns are where the two axis arrows end up after both steps. Swapping the order usually gives a different product and puts the words somewhere else.",
    whatYouSee: [
      "The map at the stage you choose: at the start, after step 1, or after both steps.",
      "After both steps, hollow marks and a dashed outline show where the swapped order would put each word and the grid square, with dotted lines joining each word to its other landing.",
      "Thick blue and purple arrows for the columns of the matrix shown.",
      "A row of small written matrices: step 1, then step 2, equals one matrix that does both; and the matrix the swapped order gives.",
      "Tiles counting the words that land elsewhere if you swap, and the widest gap between the two landings.",
    ],
    howItWorks: [
      "The single matrix is step 2 times step 1: each of its columns is step 2 applied to the matching column of step 1.",
      "Rotate then stretch gives top row 0.0, −1.5 and bottom row 1.0, 0.0. Stretch then rotate gives top row 0.0, −1.0 and bottom row 1.5, 0.0.",
      "A word counts as landing elsewhere when its two landings differ at all.",
    ],
    controls: [
      "Step 1 and Step 2 each pick rotate, stretch, shear or flip.",
      "Swap the order exchanges the two steps.",
      "Show the map steps through the start, after step 1, and after both.",
      "The Predict first question at the top can be answered before you swap.",
    ],
    notice: [
      "With rotate and stretch, 8 of 8 words land elsewhere when you swap.",
      "With stretch and flip, 0 of 8 do: both act along the same two axes, so their order makes no difference.",
    ],
    limits: [
      "In this lab: only four authored moves are offered, and each is a clean two-by-two matrix. Some pairs throw a word off the map; it is then pinned to the edge.",
      "In general: a deep network is many matrices applied in order with bending steps between them. Without the bending steps the whole stack would collapse into one matrix, exactly as two steps do here.",
    ],
  },
  "When a matrix forgets": {
    title: "Flattening the map loses a direction for good",
    summary:
      "This matrix multiplies every alive number by one factor. Above zero the squeeze can be undone by stretching back. At exactly zero the map lies on one line, two words become the same list, and no later matrix can tell them apart. The matrix has dropped from rank 2 to rank 1.",
    whatYouSee: [
      "The map after the squeeze, with hollow marks and dotted lines showing where each word started.",
      "The shaded grid square and its area, which is the factor on the slider.",
      "The squeezing matrix written out: 1.0 and 0.0 on top, 0.0 and the factor underneath.",
      "Tiles for the Area of the shaded square, Words with their own spot, and the Closest pair.",
      "At a factor of zero, a red ring round cat and bicycle and a message naming the list they now share.",
      "Each word is switches between Two numbers and Three numbers, in 3D. In 3D every word has a third number, fast, and is a point in space. Two sliders multiply the alive and the fast numbers, the shaded box shows what the squeeze does to a cube, and a stem joins each word to the sheet where alive is 0.",
    ],
    howItWorks: [
      "Every word (size, alive) is sent to (size, factor × alive).",
      "Stretch it back multiplies alive by 1 ÷ factor, which restores every word whenever the factor is above zero. At zero there is nothing to divide by.",
      "Cat is (−0.8, 1.2) and bicycle is (−0.8, −1.0). They share a size, so at a factor of zero both become (−0.8, 0.0).",
      "Words with their own spot counts distinct landing points; Closest pair is the pair of words with the smallest distance between them.",
    ],
    controls: [
      "Multiply every alive number by runs from × 1.00 down to × 0.00 in steps of 0.05.",
      "Then chooses between leave it and stretch it back.",
      "The Predict first question at the top can be answered before you try to undo the squeeze.",
      "In 3D, drag to turn the cloud. Slide alive to 0 to press the cloud onto a sheet, then fast to 0 to press the sheet onto a line. Directions left counts what remains: 3, 2 or 1.",
    ],
    notice: [
      "At × 0.05 the words look merged but the tiles still read 8 of 8, and stretch it back still works.",
      "Only at × 0.00 does the area reach zero, the count fall to 7 of 8, and stretch it back fail.",
    ],
    limits: [
      "In this lab: the matrix only ever squeezes the alive direction, and cat and bicycle were given the same size so that exactly one pair merges.",
      "In general: any matrix whose columns lie along one line flattens the plane the same way, and which inputs merge depends on which direction it drops. Models often drop directions on purpose, to ignore what does not matter, and they pay this same price for it.",
      "In this lab: the third number, fast, is invented like the other two, and the 3D squeeze only ever multiplies a direction that lines up with one of the three numbers.",
    ],
  },
  "A table from a few pieces": {
    title: "A low-rank table is cheap to store",
    summary:
      "An 8 by 8 table has 64 numbers, but a table that is the sum of a few column-times-row pieces can be rebuilt from far fewer: each piece costs one column and one row, 16 numbers. Keeping r pieces stores r × 16 numbers. This is the saving that LoRA uses on tables with millions of numbers.",
    whatYouSee: [
      "Four small pictures, one per piece. Down the left is the piece's column of eight numbers, along the top its row of plus and minus signs, and each cell is their product. Blue cells are positive and orange striped cells are negative.",
      "Pieces that are kept are shown at full strength; pieces left out are faded and say so.",
      "The full table on the left and the table rebuilt from the kept pieces on the right, with every number written in its cell.",
      "Tiles for Numbers stored, the share of the full table that is, Average miss per cell and Largest miss.",
    ],
    howItWorks: [
      "The four pieces have strengths 8, 4, 2 and 1. Each cell of a piece is plus or minus its strength.",
      "The full table is the four pieces added cell by cell, so its entries are odd whole numbers from −15 to 15.",
      "The rebuilt table is the sum of the first r pieces. The miss in a cell is the size of the difference from the full table.",
      "Numbers stored = r × (8 + 8). With 1, 2, 3 and 4 pieces that is 16, 32, 48 and 64, and the average miss is 4, 2, 1 and 0.",
    ],
    controls: [
      "Pieces kept runs from 1 to 4.",
      "The Predict first question at the top asks for the cost of two pieces before you look.",
    ],
    notice: [
      "Each piece you add halves the average miss, because each piece is half as strong as the one before.",
      "With all four pieces you store 64 numbers, as many as the table: low rank only saves when the pieces are few.",
    ],
    limits: [
      "In this lab: the table was built from four pieces written by hand, with patterns chosen to be orthogonal so that keeping the strongest pieces is provably the best you can do with that many. The table is exactly rank 4 by construction.",
      "In general: most tables are not exactly low rank, so a few pieces only approximate them. LoRA bets that the change a model needs during fine-tuning is close to low rank, and when that bet is wrong some of the change is lost.",
    ],
  },
  "Challenge: aim the map": {
    title: "Find the matrix that puts inputs on targets",
    summary:
      "Each round fixes where the eight words should land and asks you to find the matrix that sends them there. You can read the answer from where the two axis arrows must land, or nudge one number at a time and keep what lowers the distance. The second way is training in miniature.",
    whatYouSee: [
      "Dashed hollow marks for the eight targets, and the eight words where your matrix currently puts them, each joined to its own target by a red dashed line.",
      "The grid, the shaded grid square and the two thick column arrows of your matrix.",
      "With the hint on, two dashed arrows for the columns of the target matrix, and their lists in the text beside the map.",
      "A meter for Average distance from the targets, which turns green and reads All eight words on target when every word is on its mark.",
    ],
    howItWorks: [
      "Each round has a hidden target matrix. A target mark is that matrix applied to the word.",
      "Average distance is the mean straight-line distance between each word's landing and its target.",
      "A round is solved when every word is within a hair of its target, which with these sliders means your four numbers equal the target's.",
      "The three targets are a stretch with a squash, a quarter turn to the right, and a flip followed by a shear.",
    ],
    controls: [
      "Round picks one of the three rounds and starts it from leave alone.",
      "Top left, Top right, Bottom left and Bottom right set your matrix.",
      "Hint: show where the two axis arrows should land turns the hint on or off.",
      "Start this round again resets the four numbers. Next round appears once a round is solved.",
    ],
    notice: [
      "Each slider has one best position for where the other three are now, and the meter falls as you move toward it. Going round the sliders a few times brings the meter to zero in all three rounds.",
      "With the hint on, the two dashed arrows are the two columns you need.",
    ],
    limits: [
      "In this lab: the targets were made by matrices the sliders can reach exactly, so a perfect score exists and there are only four numbers to find.",
      "In general: training a layer searches for millions of numbers at once, uses slopes to choose every nudge, and almost never reaches a miss of zero, because real targets are not produced by a matrix.",
    ],
  },
};

export default cardInfo;
