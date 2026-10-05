import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { licenceEntries } from "@app/lib/licenses";
import { LicencesPanel } from "./LicencesPanel";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

const mount = () =>
  act(() => {
    root.render(createElement(LicencesPanel));
  });

const input = () => container.querySelector<HTMLInputElement>('input[type="search"]')!;
const type = (value: string) =>
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input(), value);
    input().dispatchEvent(new Event("input", { bubbles: true }));
  });
const rows = () => Array.from(container.querySelectorAll<HTMLElement>(".licence-entry"));
const rowNames = () => rows().map((row) => row.querySelector("strong")!.textContent);
const groupTitles = () => Array.from(container.querySelectorAll("h4")).map((heading) => heading.firstChild?.textContent?.trim());

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

describe("LicencesPanel", () => {
  it("groups everything as Libraries, Fonts, Runtime & models, and Data", () => {
    mount();
    expect(groupTitles()).toEqual(["Libraries", "Fonts", "Runtime & models", "Data"]);
    expect(container.querySelector(".licences__count")!.textContent).toContain(`${licenceEntries.length} components`);
  });

  it("gives each entry its use, licence, copyright holder, and source", () => {
    mount();
    const react = rows().find((row) => row.querySelector("strong")!.textContent === "React")!;
    expect(react.textContent).toContain("The user interface");
    expect(react.querySelector(".licence-badge")!.textContent).toBe("MIT");
    expect(react.textContent).toContain("Meta Platforms, Inc. and affiliates");
    expect(react.querySelector("a")!.getAttribute("href")).toMatch(/^https:\/\//);
    const llama = rows().find((row) => row.querySelector("strong")!.textContent === "llama.cpp")!;
    expect(llama.textContent).toContain("Downloaded on request");
    expect(llama.textContent).toContain("b10991");
  });

  it("says plainly when a licence could not be verified, and points at the upstream project", () => {
    mount();
    const mnist = rows().find((row) => row.querySelector("strong")!.textContent === "MNIST handwritten digits")!;
    expect(mnist.querySelector(".licence-badge")!.textContent).toContain("Check the upstream project");
    expect(mnist.getAttribute("data-status")).toBe("check-upstream");
    expect(mnist.querySelector("a")!.getAttribute("href")).toBe("http://yann.lecun.com/exdb/mnist/");
    expect(mnist.textContent).toContain("could not verify");
  });

  it("keeps the installed-with and Rust-crate lists collapsed until you filter", () => {
    mount();
    const more = Array.from(container.querySelectorAll<HTMLDetailsElement>("details.licence-more"));
    expect(more.map((item) => item.querySelector("summary")!.textContent)).toEqual([
      expect.stringContaining("Packages installed with these libraries"),
      expect.stringContaining("Rust crates in the desktop shell"),
    ]);
    expect(more.every((item) => !item.open)).toBe(true);
    type("serde");
    expect(container.querySelector<HTMLDetailsElement>("details.licence-more")!.open).toBe(true);
    expect(rowNames()).toContain("serde");
  });

  it("filters as you type, announces the count, and can be cleared", () => {
    mount();
    type("OFL-1.1");
    expect(rowNames().sort()).toEqual(["Inter", "JetBrains Mono", "Newsreader"]);
    expect(groupTitles()).toEqual(["Fonts"]);
    const status = container.querySelector('[role="status"]')!;
    expect(status.textContent).toContain(`3 of ${licenceEntries.length} components match`);
    expect(input().getAttribute("aria-describedby")).toBe(status.id);
    type("");
    expect(groupTitles()).toHaveLength(4);
  });

  it("says so when nothing matches", () => {
    mount();
    type("zzzz-no-such-thing");
    expect(rows()).toHaveLength(0);
    expect(container.textContent).toContain("Nothing matches that");
    expect(container.querySelector('[role="status"]')!.textContent).toContain("0 of");
  });

  it("is operable from the keyboard: a labelled input, native disclosure widgets, and focusable text regions", () => {
    mount();
    const label = container.querySelector<HTMLLabelElement>(".licences__search label")!;
    expect(label.htmlFor).toBe(input().id);
    expect(label.textContent).toBe("Filter the list");
    const texts = container.querySelector("details.licence-texts")!;
    expect(texts.querySelector("summary")!.textContent).toContain("Full licence texts");
    const regions = Array.from(texts.querySelectorAll<HTMLElement>('pre[role="region"]'));
    expect(regions.length).toBeGreaterThan(4);
    for (const region of regions) {
      expect(region.getAttribute("tabindex")).toBe("0");
      expect(region.getAttribute("aria-label")).toBeTruthy();
    }
  });

  it("carries the full licence texts and copyright notices offline", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    mount();
    const texts = container.querySelector("details.licence-texts")!;
    const summaries = Array.from(texts.querySelectorAll("details.licence-text > summary")).map((item) => item.textContent);
    for (const title of ["MIT License", "ISC License", "BSD 3-Clause License", "Apache License, Version 2.0", "SIL Open Font License, Version 1.1"]) {
      expect(summaries.some((summary) => summary?.includes(title)), title).toBe(true);
    }
    expect(texts.textContent).toContain("Permission is hereby granted, free of charge");
    expect(texts.textContent).toContain("SIL OPEN FONT LICENSE Version 1.1");
    expect(texts.textContent).toContain("The Inter Project Authors");
    expect(texts.textContent).toContain("Copyright (c) Microsoft Corporation");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("shows the app's own copyright line and no licence claim for the app", () => {
    mount();
    expect(container.querySelector(".licences__copyright")!.textContent).toBe("Discover AI itself: Copyright © 2026 Discover AI contributors");
    expect(container.textContent).not.toMatch(/Discover AI[^.]*(is licensed|open source|released under)/i);
  });

  it("explains that cloud providers are not part of the list", () => {
    mount();
    expect(container.querySelector(".licences__cloud")!.textContent).toContain("Nothing from a provider is bundled or downloaded");
  });
});
