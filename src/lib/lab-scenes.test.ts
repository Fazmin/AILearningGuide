import { describe, expect, it } from "vitest";
import type { LabScene } from "@app/module-sdk";
import { placeScenes, sceneState } from "./lab-scenes";

const scene = (id: string, sections: string[], state?: LabScene["state"]): LabScene => ({
  id,
  sections,
  title: id,
  look: "Look here.",
  state,
});

describe("placeScenes", () => {
  it("puts a scene under the first heading in its list that the lesson has", () => {
    const mask = scene("mask", ["Lift the one-way rule", "Lift the mask"]);
    expect([...placeScenes([mask], ["How to play with it", "Lift the mask"]).keys()]).toEqual(["lift the mask"]);
    expect([...placeScenes([mask], ["Lift the mask", "Lift the one-way rule"]).keys()]).toEqual([
      "lift the one-way rule",
    ]);
  });

  it("matches headings whatever their quotes and case", () => {
    const divide = scene("divide", ["Divide, or don't"]);
    expect(placeScenes([divide], ["Divide, or don’t"]).get("divide, or don't")).toBe(divide);
  });

  it("leaves out a scene whose headings this lesson lacks", () => {
    expect(placeScenes([scene("missing", ["Not here"])], ["What it is"]).size).toBe(0);
  });

  it("gives a heading to the first scene that claims it", () => {
    const first = scene("first", ["What it is"]);
    const second = scene("second", ["What it is"]);
    expect(placeScenes([first, second], ["What it is"]).get("what it is")).toBe(first);
  });
});

describe("sceneState", () => {
  it("lays the scene's settings over the defaults, not over the reader's state", () => {
    const initial = { mask: "causal", head: 1, layer: 1 };
    expect(sceneState(initial, scene("mask", [], { mask: "bidirectional", head: 3 }))).toEqual({
      mask: "bidirectional",
      head: 3,
      layer: 1,
    });
    expect(sceneState(initial, scene("plain", []))).toEqual(initial);
    expect(initial.mask).toBe("causal");
  });
});
