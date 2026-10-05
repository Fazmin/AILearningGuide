import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition, { hydrateToolsState } from "./module";

/** The text a learner reads on the page, with the markup removed and entities decoded. */
const rendered = (state: Record<string, unknown>) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: hydrateToolsState(JSON.stringify(state)),
      setState: () => undefined,
      currentStep: 0,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  )
    .replace(/<[^>]*>/g, " ")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ");

describe("tools quoted values appear on the page", () => {
  it("shows a bad first call and its corrected retry in the transcript", () => {
    const page = rendered({ preset: "retry-type" });
    expect(page).toContain("Messages, in order, across both turns");
    expect(page).toContain('{"error":"invalid arguments","details":["/day: expected string, got integer"]}');
    expect(page).toContain('{"dock":"north","day":"sunday","closes":"16:00"}');
    expect(page).toContain("the six messages above");
    expect(page).toContain("Calls made 2");
  });

  it("lets the Turn control choose which call the pipeline shows", () => {
    expect(rendered({ preset: "retry-type", turn: "1" })).toContain("Turn 1 of 2 · stopped at validation");
    expect(rendered({ preset: "retry-type", turn: "2" })).toContain("Turn 2 of 2 · get_dock_hours ran on the host");
  });

  it("shows no retry and no Turn control for a single-turn preset", () => {
    const page = rendered({ preset: "valid" });
    expect(page).toContain("the four messages above");
    expect(page).toContain("Calls made 1");
    expect(page).not.toContain("Turn 1 of 2");
    expect(page).not.toContain("authored retry");
  });

  it("keeps a repeated mistake failing, and drops the retry once the first call is fixed", () => {
    const repeated = rendered({ preset: "retry-type", retry: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": 7}}', turn: "2" });
    expect(repeated).toContain("Turn 2 of 2 · stopped at validation");
    const fixedFirst = rendered({ preset: "retry-type", draft: '{"name": "get_dock_hours", "arguments": {"dock": "north", "day": "sunday"}}' });
    expect(fixedFirst).toContain("the four messages above");
    expect(fixedFirst).not.toContain("Turn 1 of 2");
  });
});
