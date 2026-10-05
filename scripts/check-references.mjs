// Usage: npm run check:references [-- <module slug | module.ts path | url> ...]
// With no arguments, checks every module that lists references.
// Each URL is fetched with curl and a browser user agent, then, if that fails or lands on a bot wall,
// loaded in headless Chrome (macOS path). Prints status, final URL, and page title for each.
// Needs network access, so it is not part of `npm run check`.
import { execFileSync } from "node:child_process";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const modulesDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "src", "modules");

const UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const args = process.argv.slice(2);
const targets = args.length
  ? args.map((arg) => (/^https?:/.test(arg) || existsSync(arg) ? arg : path.join(modulesDir, arg, "module.ts")))
  : readdirSync(modulesDir)
      .map((slug) => path.join(modulesDir, slug, "module.ts"))
      .filter((file) => existsSync(file) && readFileSync(file, "utf8").includes("references: ["));

const urls = [];
for (const arg of targets) {
  if (existsSync(arg)) {
    const source = readFileSync(arg, "utf8");
    const start = source.indexOf("references: [");
    if (start < 0) { console.log(`${arg}: no references block`); continue; }
    const block = source.slice(start, source.indexOf("\n  ],", start));
    for (const match of block.matchAll(/url: "([^"]+)"/g)) urls.push(match[1]);
  } else urls.push(arg);
}

const title = (html) => (html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] ?? "").replace(/\s+/g, " ").trim().slice(0, 110);

let failures = 0;
for (const url of urls) {
  let status = 0, final = url, body = "";
  try {
    const out = execFileSync("curl", ["-sL", "--compressed", "--max-time", "25", "-A", UA, "-H", "Accept: text/html,application/xhtml+xml,application/pdf,*/*", "-w", "\n__META__%{http_code} %{url_effective}", url], { maxBuffer: 64 * 1024 * 1024 }).toString("latin1");
    const at = out.lastIndexOf("\n__META__");
    body = out.slice(0, at);
    [status, final] = out.slice(at + 9).split(" ");
    status = Number(status);
  } catch { status = 0; }
  let via = "curl";
  // Cloudflare injects its challenge script into ordinary pages too, so it only counts on a near-empty page.
  const walled = (html) =>
    /_Incapsula_Resource/i.test(html) ||
    (/cf-chl-|challenge-platform/i.test(html) && html.length < 20000) ||
    /just a moment|attention required|client challenge|captcha|are you a robot|verifying your browser|checking your browser|not a bot|^redirecting/i.test(title(html)) ||
    (!html.startsWith("%PDF") && !title(html));
  if (status >= 200 && status < 300 && (walled(body) || /\/challenge\b|captcha/i.test(final))) status = 0;
  if (!(status >= 200 && status < 300)) {
    try {
      const dom = execFileSync(CHROME, ["--headless=new", "--disable-gpu", `--user-agent=${UA}`, "--virtual-time-budget=15000", "--dump-dom", url], { timeout: 45000, maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "ignore"] }).toString();
      const t = title(dom);
      const blocked = /access denied|just a moment|attention required|forbidden|not found|404|error/i.test(t);
      if (dom.length > 2000 && t && !blocked && !walled(dom)) { status = 200; body = dom; via = "chrome"; }
    } catch { /* keep the curl status */ }
  }
  const ok = status >= 200 && status < 300;
  if (!ok) failures += 1;
  const pdf = body.startsWith("%PDF");
  console.log(`${ok ? "OK  " : "FAIL"} ${status} [${via}] ${url}${final && final !== url ? `\n       -> ${final}` : ""}\n       ${pdf ? "(PDF)" : title(body)}`);
}
console.log(`\n${urls.length - failures}/${urls.length} links work.`);
process.exitCode = failures ? 1 : 0;
