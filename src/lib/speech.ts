import { invoke } from "@tauri-apps/api/core";

const desktop = () => Boolean(window.__TAURI_INTERNALS__);

export function canUseSpeech() {
  return desktop() || typeof window !== "undefined" && "speechSynthesis" in window;
}

export function normalizeSpeechText(text: string) {
  return text
    .replace(/\u00a0/g, " ")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function textFromLearnRoot(root: ParentNode) {
  return [".objectives", ".learn-copy"]
    .map((selector) => normalizeSpeechText(root.querySelector(selector)?.textContent ?? ""))
    .filter(Boolean)
    .join("\n\n");
}

export function splitForSpeech(text: string, maxLength = 240) {
  const cleaned = normalizeSpeechText(text);
  if (!cleaned) return [];
  const sentences = cleaned.match(/[^.!?\n]+[.!?]+|[^.!?\n]+/g) ?? [cleaned];
  const chunks: string[] = [];
  let current = "";
  for (const sentence of sentences) {
    const piece = sentence.trim();
    if (!piece) continue;
    const next = current ? `${current} ${piece}` : piece;
    if (next.length > maxLength && current) {
      chunks.push(current);
      current = piece;
    } else {
      current = next;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

let webCancelled = false;

function pickVoice() {
  const voices = window.speechSynthesis.getVoices();
  const lang = (document.documentElement.lang || navigator.language || "en").toLowerCase();
  const prefix = lang.slice(0, 2);
  const local = voices.filter((voice) => voice.localService);
  const pool = local.length > 0 ? local : voices;
  return (
    pool.find((voice) => voice.default && voice.lang.toLowerCase().startsWith(lang)) ??
    pool.find((voice) => voice.default) ??
    pool.find((voice) => voice.lang.toLowerCase().startsWith(prefix)) ??
    pool[0] ??
    null
  );
}

function waitForVoices() {
  if (window.speechSynthesis.getVoices().length > 0) return Promise.resolve();
  return new Promise<void>((resolve) => {
    const done = () => {
      window.speechSynthesis.removeEventListener("voiceschanged", done);
      resolve();
    };
    window.speechSynthesis.addEventListener("voiceschanged", done);
    window.setTimeout(done, 400);
  });
}

function speakChunk(text: string, voice: SpeechSynthesisVoice | null) {
  return new Promise<void>((resolve, reject) => {
    const utterance = new SpeechSynthesisUtterance(text);
    if (voice) utterance.voice = voice;
    utterance.onend = () => resolve();
    utterance.onerror = (event) => {
      if (event.error === "canceled" || event.error === "interrupted") resolve();
      else reject(new Error(`Speech failed: ${event.error}`));
    };
    window.speechSynthesis.speak(utterance);
  });
}

async function speakWithWebApi(text: string) {
  if (!("speechSynthesis" in window)) {
    throw new Error("Speech is not available in this window.");
  }
  await waitForVoices();
  const voice = pickVoice();
  const chunks = splitForSpeech(text);
  window.speechSynthesis.cancel();
  for (const chunk of chunks) {
    if (webCancelled) return;
    await speakChunk(chunk, voice);
  }
}

export function stopSpeaking() {
  webCancelled = true;
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  if (desktop()) void invoke("stop_speaking").catch(() => undefined);
}

export async function speakLesson(text: string) {
  webCancelled = false;
  const cleaned = normalizeSpeechText(text);
  if (!cleaned) throw new Error("Nothing to read.");
  if (desktop()) {
    try {
      await invoke("speak_text", { text: cleaned });
      return;
    } catch {
      if (webCancelled) return;
    }
  }
  await speakWithWebApi(cleaned);
}
