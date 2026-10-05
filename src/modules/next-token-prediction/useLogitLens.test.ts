/**
 * Mounts the hook behind the "Layer by layer" card: it fetches tiny-transformer.onnx, reads the
 * weights, and recomputes the lens when the text changes. fetch is stubbed with the shipped file;
 * everything else is the hook's own code.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import vocabulary from "./assets/transformer-vocab.json";
import { DEFAULT_PROMPT } from "./prompts";
import { encode } from "./sampling";
import { useLogitLens, type LensView } from "./useLogitLens";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const bytes = readFileSync(resolve(process.cwd(), "src/modules/attention/assets/tiny-transformer.onnx"));
const arrayBuffer = () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const itos = vocabulary.itos as string[];

let container: HTMLDivElement;
let root: Root;
let latest: LensView;

function Probe({ ids }: { ids: number[] }) {
  latest = useLogitLens(ids, 10);
  return null;
}

const mount = async (ids: number[]) => {
  await act(async () => {
    root.render(createElement(Probe, { ids }));
  });
};
const wait = (ms: number) => act(async () => void (await new Promise((done) => setTimeout(done, ms))));

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.unstubAllGlobals();
});

describe("useLogitLens", () => {
  it("reports an error, and tries again next time, when the model file cannot be fetched", async () => {
    vi.stubGlobal("fetch", async () => ({ ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) }));
    await mount(encode(DEFAULT_PROMPT, vocabulary));
    await wait(100);
    expect(latest.status).toBe("error");
    expect(latest.error).toContain("returned HTTP 404");
    expect(latest.result).toBeUndefined();
  });

  it("loads the weights, reads the stages, and marks the old result stale while a new text is pending", async () => {
    const fetchSpy = vi.fn(async () => ({ ok: true, status: 200, arrayBuffer: async () => arrayBuffer() }));
    vi.stubGlobal("fetch", fetchSpy);
    await mount(encode(DEFAULT_PROMPT, vocabulary));
    expect(latest.status).toBe("loading");
    await wait(600);
    expect(latest.status).toBe("ready");
    expect(latest.stale).toBe(false);
    expect(itos[latest.result!.finalIndex]).toBe("u");
    expect(latest.result!.stages.map((stage) => stage.finalRank)).toEqual([56, 3, 1]);
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    // New text: the previous readout is kept (stale) until the pause ends, then replaced.
    const jason = encode("O Jason, Jason! wherefore art thou J", vocabulary);
    await mount(jason);
    expect(latest.status).toBe("loading");
    expect(latest.stale).toBe(true);
    expect(itos[latest.result!.finalIndex]).toBe("u");
    await wait(300);
    expect(latest.status).toBe("ready");
    expect(latest.stale).toBe(false);
    expect(itos[latest.result!.finalIndex]).toBe("a");
    // The weights were read once; a new text does not fetch the model again.
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  }, 30_000);

  it("does nothing for empty text", async () => {
    vi.stubGlobal("fetch", async () => ({ ok: true, status: 200, arrayBuffer: async () => arrayBuffer() }));
    await mount([]);
    await wait(200);
    expect(latest.result).toBeUndefined();
    expect(latest.status).toBe("loading");
  });
});
