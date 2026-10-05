import { describe, expect, it } from "vitest";
import { formatBytes } from "./local-ai";

describe("formatBytes", () => {
  it("labels powers of 1024 as binary units", () => {
    // Settings shows the pinned Qwen3.5 2B Q4_K_M file (1,280,835,840 bytes). Divided by 1024^3 that is 1.19, a GiB figure;
    // the old label said "GB", which would mean 1,000,000,000 bytes (1.28 GB).
    expect(formatBytes(1_280_835_840)).toBe("1.19 GiB");
    expect((1_280_835_840 / 1e9).toFixed(2)).toBe("1.28");
    expect(formatBytes(1_073_741_824)).toBe("1.00 GiB");
    expect(formatBytes(1_048_576)).toBe("1.0 MiB");
    expect(formatBytes(2_048)).toBe("2.0 KiB");
    expect(formatBytes(512)).toBe("512 B");
  });
});
