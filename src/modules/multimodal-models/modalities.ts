/**
 * Token arithmetic for the modalities beyond a still image. Only computed values are quoted in the
 * lesson, and every input is either a figure the source states or an assumption the lesson labels.
 */
import { realScale } from "./patches";

/**
 * Whisper (Radford et al., 2022) computes its spectrogram on 25 ms windows with a 10 ms stride and
 * works on 30-second segments; its encoder stem's second convolution has stride two.
 */
export const AUDIO = { segmentSeconds: 30, strideMs: 10, convStride: 2 } as const;

export const audioFrames = (seconds: number, strideMs: number) => Math.floor((seconds * 1000) / strideMs);

/** Positions the encoder's transformer sees after the stride-two convolution. */
export const audioPositions = (seconds: number, strideMs: number, convStride: number) =>
  Math.ceil(audioFrames(seconds, strideMs) / convStride);

/**
 * Tokens for a clip if every sampled frame is cut like one image. An assumption for illustration, not
 * any model's setting: real video models skip, merge or pool frames.
 */
export const videoTokens = (seconds: number, framesPerSecond: number, tokensPerFrame: number) =>
  Math.round(seconds * framesPerSecond) * tokensPerFrame;

/** A page fed to the model as one image costs (side / patch)² tokens, whatever is printed on it. */
export const pageImageTokens = (side: number, patch: number) => realScale(side, patch, false, "llava").patchTokens;
