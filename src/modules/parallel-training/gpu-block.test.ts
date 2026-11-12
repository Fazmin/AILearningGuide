import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { GpuBlock, SHOWN_REPLICAS } from "./GpuBlock";
import definition from "./module";
import { defaultState, readState } from "./state";

const block = (tensor: number, pipeline: number, data: number) =>
  renderToStaticMarkup(createElement(GpuBlock, { tensor, pipeline, data, gpus: tensor * pipeline * data }));
const label = (html: string) => html.match(/<canvas[^>]*aria-label="([^"]*)"/)?.[1] ?? "";

describe("the 3D block of GPUs", () => {
  it("names the three edges, the total and where each group's traffic goes", () => {
    const html = block(8, 16, 128);
    const words = label(html);
    expect(words).toContain("All 16,384 GPUs as a block of cubes");
    expect(words).toContain("16 pipeline stages across, 8 tensor slices up, and 128 data replicas going back");
    expect(words).toContain(`of which the first ${SHOWN_REPLICAS} are drawn`);
    expect(words).toContain("tensor group of 8 on NVLink");
    expect(words).toContain("pipeline group of 16 on InfiniBand");
    expect(words).toContain("data group of 128 on InfiniBand");
    expect(html).toContain("16 × 8 × 128 = 16,384 GPUs");
  });

  it("handles a single GPU and a layout with nothing split", () => {
    const one = label(block(1, 1, 1));
    expect(one).toContain("The one GPU as a single cube: 1 pipeline stage across");
    expect(one).toContain("1 pipeline stage across, 1 tensor slice up, and 1 data replica going back.");
    expect(one).toContain("tensor off; pipeline off; data off");
    expect(one).not.toContain("are drawn");
  });

  it("is offered beside the slab and rack, which stay the default", () => {
    const render = (patch: Record<string, unknown>) =>
      renderToStaticMarkup(
        createElement(definition.Explore, {
          state: { ...definition.initialState, ...patch } as typeof definition.initialState,
          setState: () => undefined,
          currentStep: 0,
          mode: "plain" as const,
          narrate: () => undefined,
        }),
      );
    const flat = render({});
    expect(flat).toContain("par-slab");
    expect(flat).not.toContain("stage3d__frame");
    const lifted = render({ layoutView: "block" });
    expect(lifted).toContain("stage3d__frame");
    expect(lifted).not.toContain("par-slab");
    expect(defaultState.layoutView).toBe("flat");
    expect(readState({ layoutView: "block" }).layoutView).toBe("block");
    expect(readState({ layoutView: "hologram" }).layoutView).toBe("flat");
  });
});
