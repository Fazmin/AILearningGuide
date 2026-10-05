import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { ONBOARDING_SAMPLE } from "./onboarding-sample";

const lab = path.resolve(__dirname, "..", "modules", ONBOARDING_SAMPLE.moduleSlug);
const read = (file: string) => readFileSync(path.join(lab, file), "utf8");

describe("the welcome tour's sample paragraphs", () => {
  it("are word for word what the lab's Plain and Standard text say", () => {
    expect(
      read("content/plain.mdx"),
      "The tour's Plain sample no longer matches the lab. Update src/lib/onboarding-sample.ts.",
    ).toContain(ONBOARDING_SAMPLE.plain);
    expect(
      read("content/standard.mdx"),
      "The tour's Standard sample no longer matches the lab. Update src/lib/onboarding-sample.ts.",
    ).toContain(ONBOARDING_SAMPLE.standard);
  });

  it("come from the section they are credited to, in a lab that exists", () => {
    expect(read("module.ts")).toContain(`title: "${ONBOARDING_SAMPLE.moduleTitle}"`);
    for (const file of ["content/plain.mdx", "content/standard.mdx"]) {
      const text = read(file);
      const section = text.split(/^## /m).find((part) => part.startsWith(`${ONBOARDING_SAMPLE.section}\n`));
      expect(section, `${file} has no "${ONBOARDING_SAMPLE.section}" section`).toBeDefined();
    }
    const plainSection = read("content/plain.mdx").split(/^## /m).find((part) => part.startsWith("What it is\n"))!;
    const standardSection = read("content/standard.mdx").split(/^## /m).find((part) => part.startsWith("What it is\n"))!;
    expect(plainSection).toContain(ONBOARDING_SAMPLE.plain);
    expect(standardSection).toContain(ONBOARDING_SAMPLE.standard);
  });

  it("are short enough to read at a glance", () => {
    for (const text of [ONBOARDING_SAMPLE.plain, ONBOARDING_SAMPLE.standard]) {
      expect(text.split(/\s+/).length).toBeLessThan(80);
    }
  });
});
