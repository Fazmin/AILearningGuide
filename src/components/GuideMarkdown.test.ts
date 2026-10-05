import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { GuideMarkdown } from "./GuideMarkdown";

const html = (text: string) => renderToStaticMarkup(createElement(GuideMarkdown, { text }));

describe("GuideMarkdown", () => {
  it("renders emphasis, lists, and fenced code", () => {
    const markup = html("**Watch** the `bias`\n\n- first\n- second\n\n```ts\nconst x = 1\n```");
    expect(markup).toContain("<strong>Watch</strong>");
    expect(markup).toContain("<code>bias</code>");
    expect(markup).toContain("<li>first</li>");
    expect(markup).toContain("<pre>");
    expect(markup).toContain("const x = 1");
  });

  it("keeps only http(s) and mailto links", () => {
    const markup = html("[docs](https://example.com) [bad](javascript:alert(1))");
    expect(markup).toContain('href="https://example.com"');
    expect(markup).toContain('target="_blank"');
    expect(markup).not.toContain("javascript:");
    expect(markup).toContain("<span>bad</span>");
  });

  it("does not execute raw HTML", () => {
    const markup = html("Hello <script>alert(1)</script>");
    expect(markup).not.toContain("<script>");
    expect(markup).toContain("&lt;script&gt;");
  });
});
