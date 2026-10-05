export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value));
}

export function softmax(values: number[], temperature = 1) {
  const safeTemperature = Math.max(0.01, temperature);
  const scaled = values.map((value) => value / safeTemperature);
  const max = Math.max(...scaled);
  const exponentials = scaled.map((value) => Math.exp(value - max));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

export function mse(prediction: number, target: number) {
  return (prediction - target) ** 2;
}

export function seededNoise(seed: number) {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

export function encodeState(value: unknown) {
  const json = JSON.stringify(value);
  const bytes = new TextEncoder().encode(json);
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

export function decodeState<T>(value: string): T {
  const padded = value.replaceAll("-", "+").replaceAll("_", "/").padEnd(
    Math.ceil(value.length / 4) * 4,
    "=",
  );
  const binary = atob(padded);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}
