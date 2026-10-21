import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

class SilentWorker {
  static instances: SilentWorker[] = [];
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  posted: unknown[] = [];
  terminated = false;
  constructor() {
    SilentWorker.instances.push(this);
  }
  postMessage(message: unknown) {
    this.posted.push(message);
  }
  terminate() {
    this.terminated = true;
  }
}

describe("runTeachingModel", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
    SilentWorker.instances = [];
    vi.stubGlobal("Worker", SilentWorker);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("rejects and restarts the worker when it stops responding", async () => {
    const { runTeachingModel } = await import("./client");
    const pending = runTeachingModel("model.onnx", {});
    const outcome = pending.catch((error: Error) => error.message);

    await vi.advanceTimersByTimeAsync(30_000);
    expect(await outcome).toBe("The model runtime stopped responding.");
    expect(SilentWorker.instances[0].terminated).toBe(true);

    void runTeachingModel("model.onnx", {}).catch(() => undefined);
    expect(SilentWorker.instances).toHaveLength(2);
  });

  it("keeps waiting while the worker is still replying", async () => {
    const { runTeachingModel } = await import("./client");
    const first = runTeachingModel("model.onnx", {});
    const second = runTeachingModel("model.onnx", {});
    const worker = SilentWorker.instances[0];
    const reply = (id: number) =>
      worker.onmessage?.({
        data: { id, ok: true, type: "model-result", provider: "wasm", outputs: {} },
      } as MessageEvent);

    await vi.advanceTimersByTimeAsync(20_000);
    reply(1);
    await expect(first).resolves.toMatchObject({ provider: "wasm" });

    // The reply restarted the countdown, so 20 more seconds is not a stall.
    await vi.advanceTimersByTimeAsync(20_000);
    reply(2);
    await expect(second).resolves.toMatchObject({ provider: "wasm" });
    expect(worker.terminated).toBe(false);
  });
});
