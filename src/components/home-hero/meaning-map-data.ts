/**
 * 75 words from the GloVe subset the Tokens & embeddings lab ships
 * (src/modules/tokens-embeddings/assets/embedding-10k.*), baked so the home screen
 * never loads the 2 MB vector file.
 *
 * - Labelled words carry their three nearest labelled words, by real cosine similarity.
 * - The rest are each group centroid's nine nearest words in the full vocabulary.
 * - x and y (0 to 1) are a t-SNE fit (perplexity 10, seed 7) over just these 75 vectors.
 */

export type MeaningGroup = "animals" | "colours" | "cities" | "food" | "numbers";

export interface MeaningWord {
  word: string;
  group: MeaningGroup;
  x: number;
  y: number;
  neighbours?: readonly (readonly [string, number])[];
}

export const meaningMapWords: readonly MeaningWord[] = [
  { word: "cat", group: "animals", x: 0.361, y: 0.705, neighbours: [["dog", 0.92], ["mouse", 0.77], ["horse", 0.70]] },
  { word: "dog", group: "animals", x: 0.366, y: 0.729, neighbours: [["cat", 0.92], ["horse", 0.79], ["cow", 0.66]] },
  { word: "horse", group: "animals", x: 0.402, y: 0.742, neighbours: [["dog", 0.79], ["cat", 0.70], ["lion", 0.61]] },
  { word: "mouse", group: "animals", x: 0.354, y: 0.632, neighbours: [["cat", 0.77], ["dog", 0.65], ["lion", 0.49]] },
  { word: "lion", group: "animals", x: 0.295, y: 0.653, neighbours: [["cat", 0.70], ["dog", 0.64], ["horse", 0.61]] },
  { word: "cow", group: "animals", x: 0.273, y: 0.763, neighbours: [["dog", 0.66], ["milk", 0.66], ["cat", 0.65]] },
  { word: "elephant", group: "animals", x: 0.293, y: 0.693 },
  { word: "pig", group: "animals", x: 0.279, y: 0.752 },
  { word: "dogs", group: "animals", x: 0.36, y: 0.791 },
  { word: "snake", group: "animals", x: 0.383, y: 0.682 },
  { word: "pet", group: "animals", x: 0.329, y: 0.766 },
  { word: "cats", group: "animals", x: 0.362, y: 0.805 },
  { word: "monster", group: "animals", x: 0.382, y: 0.639 },
  { word: "bite", group: "animals", x: 0.343, y: 0.691 },
  { word: "animal", group: "animals", x: 0.312, y: 0.791 },
  { word: "red", group: "colours", x: 0.424, y: 0.13, neighbours: [["yellow", 0.90], ["blue", 0.89], ["green", 0.86]] },
  { word: "blue", group: "colours", x: 0.409, y: 0.115, neighbours: [["red", 0.89], ["pink", 0.85], ["green", 0.83]] },
  { word: "green", group: "colours", x: 0.39, y: 0.166, neighbours: [["red", 0.86], ["blue", 0.83], ["yellow", 0.79]] },
  { word: "yellow", group: "colours", x: 0.441, y: 0.11, neighbours: [["red", 0.90], ["purple", 0.87], ["pink", 0.84]] },
  { word: "purple", group: "colours", x: 0.441, y: 0.065, neighbours: [["pink", 0.92], ["yellow", 0.87], ["red", 0.83]] },
  { word: "pink", group: "colours", x: 0.441, y: 0.063, neighbours: [["purple", 0.92], ["blue", 0.85], ["yellow", 0.84]] },
  { word: "black", group: "colours", x: 0.382, y: 0.126 },
  { word: "colored", group: "colours", x: 0.419, y: 0.014 },
  { word: "orange", group: "colours", x: 0.444, y: 0.165 },
  { word: "white", group: "colours", x: 0.374, y: 0.144 },
  { word: "bright", group: "colours", x: 0.374, y: 0.029 },
  { word: "gray", group: "colours", x: 0.346, y: 0.125 },
  { word: "colors", group: "colours", x: 0.411, y: 0 },
  { word: "dark", group: "colours", x: 0.362, y: 0.04 },
  { word: "grey", group: "colours", x: 0.331, y: 0.116 },
  { word: "paris", group: "cities", x: 0.965, y: 0.477, neighbours: [["london", 0.73], ["berlin", 0.73], ["rome", 0.71]] },
  { word: "tokyo", group: "cities", x: 0.872, y: 0.428, neighbours: [["london", 0.61], ["berlin", 0.57], ["paris", 0.46]] },
  { word: "rome", group: "cities", x: 1, y: 0.4, neighbours: [["paris", 0.71], ["berlin", 0.68], ["cairo", 0.57]] },
  { word: "cairo", group: "cities", x: 0.893, y: 0.347, neighbours: [["paris", 0.58], ["rome", 0.57], ["london", 0.54]] },
  { word: "london", group: "cities", x: 0.902, y: 0.546, neighbours: [["paris", 0.73], ["berlin", 0.65], ["tokyo", 0.61]] },
  { word: "berlin", group: "cities", x: 0.963, y: 0.38, neighbours: [["paris", 0.73], ["rome", 0.68], ["london", 0.65]] },
  { word: "vienna", group: "cities", x: 0.953, y: 0.393 },
  { word: "opened", group: "cities", x: 0.896, y: 0.557 },
  { word: "amsterdam", group: "cities", x: 0.94, y: 0.471 },
  { word: "moscow", group: "cities", x: 0.952, y: 0.334 },
  { word: "brussels", group: "cities", x: 0.965, y: 0.447 },
  { word: "stockholm", group: "cities", x: 0.925, y: 0.397 },
  { word: "prague", group: "cities", x: 0.96, y: 0.351 },
  { word: "istanbul", group: "cities", x: 0.898, y: 0.376 },
  { word: "copenhagen", group: "cities", x: 0.94, y: 0.429 },
  { word: "bread", group: "food", x: 0.099, y: 0.958, neighbours: [["soup", 0.84], ["cheese", 0.81], ["milk", 0.73]] },
  { word: "rice", group: "food", x: 0, y: 0.873, neighbours: [["sugar", 0.65], ["bread", 0.59], ["milk", 0.59]] },
  { word: "soup", group: "food", x: 0.098, y: 0.921, neighbours: [["bread", 0.84], ["cheese", 0.77], ["milk", 0.66]] },
  { word: "cheese", group: "food", x: 0.097, y: 1, neighbours: [["bread", 0.81], ["soup", 0.77], ["milk", 0.74]] },
  { word: "milk", group: "food", x: 0.091, y: 0.819, neighbours: [["sugar", 0.82], ["cheese", 0.74], ["bread", 0.73]] },
  { word: "sugar", group: "food", x: 0.083, y: 0.811, neighbours: [["milk", 0.82], ["bread", 0.71], ["rice", 0.65]] },
  { word: "butter", group: "food", x: 0.1, y: 0.986 },
  { word: "beans", group: "food", x: 0.037, y: 0.917 },
  { word: "vegetable", group: "food", x: 0.047, y: 0.869 },
  { word: "potato", group: "food", x: 0.072, y: 0.944 },
  { word: "vegetables", group: "food", x: 0.038, y: 0.878 },
  { word: "chicken", group: "food", x: 0.125, y: 0.89 },
  { word: "potatoes", group: "food", x: 0.051, y: 0.938 },
  { word: "meat", group: "food", x: 0.131, y: 0.873 },
  { word: "fruit", group: "food", x: 0.04, y: 0.838 },
  { word: "two", group: "numbers", x: 0.796, y: 0.929, neighbours: [["three", 0.99], ["four", 0.98], ["five", 0.96]] },
  { word: "three", group: "numbers", x: 0.783, y: 0.911, neighbours: [["two", 0.99], ["four", 0.99], ["five", 0.99]] },
  { word: "four", group: "numbers", x: 0.772, y: 0.942, neighbours: [["three", 0.99], ["five", 0.99], ["two", 0.98]] },
  { word: "five", group: "numbers", x: 0.765, y: 0.91, neighbours: [["three", 0.99], ["four", 0.99], ["seven", 0.99]] },
  { word: "seven", group: "numbers", x: 0.752, y: 0.944, neighbours: [["five", 0.99], ["nine", 0.99], ["three", 0.98]] },
  { word: "nine", group: "numbers", x: 0.741, y: 0.919, neighbours: [["seven", 0.99], ["five", 0.98], ["three", 0.97]] },
  { word: "six", group: "numbers", x: 0.765, y: 0.884 },
  { word: "eight", group: "numbers", x: 0.743, y: 0.894 },
  { word: "ten", group: "numbers", x: 0.718, y: 0.956 },
  { word: "one", group: "numbers", x: 0.839, y: 0.854 },
  { word: "with", group: "numbers", x: 0.818, y: 0.821 },
  { word: "only", group: "numbers", x: 0.844, y: 0.866 },
  { word: "least", group: "numbers", x: 0.849, y: 0.917 },
  { word: "several", group: "numbers", x: 0.812, y: 0.979 },
  { word: "while", group: "numbers", x: 0.823, y: 0.809 },
];
