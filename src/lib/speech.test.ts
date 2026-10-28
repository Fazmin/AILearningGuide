import { describe, expect, it } from "vitest";
import { normalizeSpeechText, splitForSpeech, textFromLearnRoot } from "./speech";

describe("learn panel speech", () => {
  it("reads objectives and lesson copy, not the checkpoint", () => {
    const root = document.createElement("div");
    root.innerHTML = `
      <div class="objectives">After this lab, you can Relate depth to capacity</div>
      <article class="learn-copy"><h2>What it is</h2><p>Put many neurons together.</p></article>
      <div class="checkpoint">Why add a nonlinear activation?</div>
    `;
    const text = textFromLearnRoot(root);
    expect(text).toContain("Relate depth to capacity");
    expect(text).toContain("What it is");
    expect(text).toContain("Put many neurons together.");
    expect(text).not.toContain("nonlinear activation");
  });

  it("skips the lab scene cards inside the lesson", () => {
    const root = document.createElement("div");
    root.innerHTML = `
      <article class="learn-copy">
        <h3>Lift the one-way rule</h3>
        <button data-speech-skip="">See it in the lab Mask lifted</button>
        <p>Set Mask to Bidirectional.</p>
      </article>
    `;
    const text = textFromLearnRoot(root);
    expect(text).toContain("Lift the one-way rule");
    expect(text).toContain("Set Mask to Bidirectional.");
    expect(text).not.toContain("See it in the lab");
  });

  it("collapses extra whitespace", () => {
    expect(normalizeSpeechText("  Hello   \n\n\n  world\t ")).toBe("Hello\n\nworld");
  });

  it("splits long copy into speakable chunks", () => {
    const chunks = splitForSpeech(
      "First sentence is short. Second sentence is also fine. Third sentence is here.",
      40,
    );
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.join(" ")).toContain("First sentence is short.");
    expect(chunks.every((chunk) => chunk.length <= 80)).toBe(true);
  });
});
