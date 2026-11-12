/**
 * The three.js half of a 3D lab picture. Nothing outside this folder's scene files imports it
 * directly: `Stage3D` loads it on demand, so three.js stays out of the app's first download.
 *
 * A stage owns one WebGL canvas, a camera that turns around a target, colours read from the app's
 * CSS variables, pixel-width lines, and a layer of HTML labels pinned to points in the scene. It
 * draws only when something changed.
 */
import * as THREE from "three";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

export { THREE };

export interface StageView {
  /** Degrees around the vertical axis. 0 looks from the front. */
  azimuth: number;
  /** Degrees above the floor. 90 looks straight down. */
  elevation: number;
  distance: number;
  target: readonly [number, number, number];
}

export interface StageLimits {
  minElevation: number;
  maxElevation: number;
  minDistance: number;
  maxDistance: number;
}

export interface StageLabel {
  id: string;
  at: readonly [number, number, number];
  text: string;
  /** Extra class names, such as `is-accent` or `is-faint`. */
  tone?: string;
  align?: "center" | "start" | "end";
  /** Pixel nudge from the projected point. */
  dx?: number;
  dy?: number;
}

export interface LineStyle {
  /** A CSS variable name such as `--loss`, or any CSS colour. */
  color: string;
  /** Width in CSS pixels. */
  width?: number;
  opacity?: number;
  /** Dash and gap length in scene units. */
  dash?: readonly [number, number];
  /** Draw order among things at the same depth; higher is drawn later. */
  order?: number;
  /** Draw over everything, even when something is in front. */
  onTop?: boolean;
}

export interface StagePointer {
  /** Return true to take the drag instead of turning the view. */
  down?: (ray: THREE.Ray, event: PointerEvent) => boolean;
  move?: (ray: THREE.Ray) => void;
  up?: () => void;
  /** A press and release without a drag. */
  click?: (ray: THREE.Ray) => void;
}

export interface Stage {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  /** Everything a scene draws goes in here; `clear()` empties it and frees the memory. */
  root: THREE.Group;
  reducedMotion: () => boolean;
  color: (css: string) => THREE.Color;
  alpha: (css: string) => number;
  line: (points: ArrayLike<number>, style: LineStyle) => Line2;
  segments: (points: ArrayLike<number>, style: LineStyle) => LineSegments2;
  setLabels: (labels: StageLabel[]) => void;
  invalidate: () => void;
  /** Runs `frame` with 0…1 over `ms`, eased. With reduced motion it jumps straight to 1. Returns a cancel. */
  tween: (ms: number, frame: (t: number) => void) => () => void;
  view: () => StageView;
  setView: (view: Partial<StageView>, ms?: number) => void;
  resetView: () => void;
  /** Screen position of a scene point, in CSS pixels from the canvas corner. */
  project: (point: THREE.Vector3) => { x: number; y: number; visible: boolean };
  pointer: StagePointer;
  clear: () => void;
  dispose: () => void;
}

export interface StageOptions {
  view: StageView;
  limits?: Partial<StageLimits>;
  fov?: number;
  /** Called when the app's theme, contrast or palette changes: rebuild the scene with fresh colours. */
  onTheme?: () => void;
}

const RAD = Math.PI / 180;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

/** True when this browser can give a canvas a WebGL context. */
export function webglAvailable(): boolean {
  try {
    const probe = document.createElement("canvas");
    return Boolean(probe.getContext("webgl2") ?? probe.getContext("webgl"));
  } catch {
    return false;
  }
}

/** Frees every geometry and material under an object. */
export function disposeTree(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
    else material?.dispose?.();
  });
}

export function createStage(
  frame: HTMLElement,
  canvas: HTMLCanvasElement,
  labelLayer: HTMLElement,
  options: StageOptions,
): Stage {
  // The picture is drawn only when it changes, so the last frame has to stay in the canvas: without
  // this a browser may hand back an empty buffer when the page is repainted, printed or captured.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.localClippingEnabled = true;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(options.fov ?? 30, 4 / 3, 0.05, 100);
  const root = new THREE.Group();
  scene.add(root);
  // One soft light from everywhere and one from the upper left, so slopes shade the same at every turn.
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8fa8, 1.55));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(-2.2, 4, 2.6);
  scene.add(sun);

  const limits: StageLimits = {
    minElevation: 4,
    maxElevation: 89.4,
    minDistance: options.view.distance * 0.55,
    maxDistance: options.view.distance * 1.8,
    ...options.limits,
  };
  const home: StageView = { ...options.view };
  let current: StageView = { ...options.view };
  let disposed = false;
  let ready = false;
  let frameRequest = 0;
  let width = 0;
  let height = 0;

  /* ---------------- colours ---------------- */
  const probe = document.createElement("span");
  probe.style.display = "none";
  frame.appendChild(probe);
  const swatch = document.createElement("canvas");
  swatch.width = 1;
  swatch.height = 1;
  const ink = swatch.getContext("2d", { willReadFrequently: true });
  const read = (css: string): [number, number, number, number] => {
    probe.style.color = "";
    probe.style.color = css.startsWith("--") ? `var(${css})` : css;
    const computed = getComputedStyle(probe).color;
    if (!ink) return [128, 128, 128, 255];
    ink.clearRect(0, 0, 1, 1);
    ink.fillStyle = "#808080";
    ink.fillStyle = computed;
    ink.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = ink.getImageData(0, 0, 1, 1).data;
    return [r, g, b, a];
  };
  const color = (css: string) => {
    const [r, g, b] = read(css);
    return new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
  };
  const alpha = (css: string) => read(css)[3] / 255;

  /* ---------------- lines ---------------- */
  const lineMaterial = (style: LineStyle) => {
    const opacity = style.opacity ?? 1;
    const material = new LineMaterial({
      color: color(style.color),
      linewidth: style.width ?? 1.5,
      transparent: true,
      opacity,
      dashed: Boolean(style.dash),
      dashSize: style.dash?.[0] ?? 1,
      gapSize: style.dash?.[1] ?? 0,
      depthTest: !style.onTop,
      depthWrite: false,
    });
    return material;
  };
  const line = (points: ArrayLike<number>, style: LineStyle) => {
    const geometry = new LineGeometry();
    geometry.setPositions(Array.from(points));
    const object = new Line2(geometry, lineMaterial(style));
    if (style.dash) object.computeLineDistances();
    object.renderOrder = style.order ?? 1;
    return object;
  };
  const segments = (points: ArrayLike<number>, style: LineStyle) => {
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(Array.from(points));
    const object = new LineSegments2(geometry, lineMaterial(style));
    if (style.dash) object.computeLineDistances();
    object.renderOrder = style.order ?? 1;
    return object;
  };

  /* ---------------- labels ---------------- */
  let labels: StageLabel[] = [];
  const labelNodes = new Map<string, HTMLSpanElement>();
  const scratch = new THREE.Vector3();
  const project = (point: THREE.Vector3) => {
    scratch.copy(point).project(camera);
    return {
      x: (scratch.x * 0.5 + 0.5) * width,
      y: (-scratch.y * 0.5 + 0.5) * height,
      visible: scratch.z > -1 && scratch.z < 1,
    };
  };
  const setLabels = (next: StageLabel[]) => {
    labels = next;
    const wanted = new Set(next.map((label) => label.id));
    for (const [id, node] of labelNodes) {
      if (wanted.has(id)) continue;
      node.remove();
      labelNodes.delete(id);
    }
    for (const label of next) {
      let node = labelNodes.get(label.id);
      if (!node) {
        node = document.createElement("span");
        labelLayer.appendChild(node);
        labelNodes.set(label.id, node);
      }
      node.className = `stage3d__label ${label.tone ?? ""}`.trim();
      node.dataset.align = label.align ?? "center";
      if (node.textContent !== label.text) node.textContent = label.text;
    }
    invalidate();
  };
  const placeLabels = () => {
    const spot = new THREE.Vector3();
    for (const label of labels) {
      const node = labelNodes.get(label.id);
      if (!node) continue;
      const at = project(spot.set(label.at[0], label.at[1], label.at[2]));
      const x = at.x + (label.dx ?? 0);
      const y = at.y + (label.dy ?? 0);
      const inside = at.visible && x > -20 && x < width + 20 && y > -10 && y < height + 10;
      node.style.display = inside ? "" : "none";
      if (inside) node.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    }
  };

  /* ---------------- camera ---------------- */
  const applyView = () => {
    const azimuth = current.azimuth * RAD;
    const elevation = clamp(current.elevation, limits.minElevation, limits.maxElevation) * RAD;
    const [tx, ty, tz] = current.target;
    const flat = Math.cos(elevation) * current.distance;
    camera.position.set(tx + Math.sin(azimuth) * flat, ty + Math.sin(elevation) * current.distance, tz + Math.cos(azimuth) * flat);
    camera.lookAt(tx, ty, tz);
    camera.updateMatrixWorld();
  };

  /* ---------------- drawing ---------------- */
  type Tween = { start: number; ms: number; frame: (t: number) => void; done: boolean };
  const tweens = new Set<Tween>();
  const reducedMotion = () =>
    document.documentElement.dataset.motion === "reduced" ||
    (typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  const draw = (now: number) => {
    frameRequest = 0;
    if (disposed) return;
    for (const tween of [...tweens]) {
      const t = tween.ms <= 0 ? 1 : clamp((now - tween.start) / tween.ms, 0, 1);
      tween.frame(ease(t));
      if (t >= 1) {
        tween.done = true;
        tweens.delete(tween);
      }
    }
    applyView();
    renderer.render(scene, camera);
    placeLabels();
    if (tweens.size > 0) invalidate();
  };
  function invalidate() {
    if (disposed || frameRequest) return;
    frameRequest = requestAnimationFrame(draw);
  }
  const tween = (ms: number, frameFn: (t: number) => void) => {
    if (reducedMotion() || ms <= 0) {
      frameFn(1);
      invalidate();
      return () => undefined;
    }
    const entry: Tween = { start: performance.now(), ms, frame: frameFn, done: false };
    tweens.add(entry);
    invalidate();
    return () => {
      tweens.delete(entry);
    };
  };

  let cancelViewTween: () => void = () => undefined;
  const setView = (next: Partial<StageView>, ms = 0) => {
    cancelViewTween();
    const from = { ...current };
    const to: StageView = { ...current, ...next };
    // Turn the short way round.
    let turn = (to.azimuth - from.azimuth) % 360;
    if (turn > 180) turn -= 360;
    if (turn < -180) turn += 360;
    const mix = (a: number, b: number, t: number) => a + (b - a) * t;
    cancelViewTween = tween(ms, (t) => {
      current = {
        azimuth: from.azimuth + turn * t,
        elevation: mix(from.elevation, to.elevation, t),
        distance: mix(from.distance, to.distance, t),
        target: [
          mix(from.target[0], to.target[0], t),
          mix(from.target[1], to.target[1], t),
          mix(from.target[2], to.target[2], t),
        ],
      };
    });
  };

  /* ---------------- size ---------------- */
  const resize = () => {
    const nextWidth = Math.max(1, frame.clientWidth);
    const nextHeight = Math.max(1, frame.clientHeight);
    if (nextWidth === width && nextHeight === height) return;
    width = nextWidth;
    height = nextHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    // A narrow frame keeps the whole scene in view by widening the angle, not by cropping the sides.
    const base = options.fov ?? 30;
    camera.fov = camera.aspect >= 1.2 ? base : clamp((base * 1.2) / camera.aspect, base, 58);
    camera.updateProjectionMatrix();
    // Resizing empties the canvas, so redraw before the browser paints rather than a frame later.
    if (ready) {
      if (frameRequest) cancelAnimationFrame(frameRequest);
      draw(performance.now());
    }
  };
  const sizeWatch = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(resize);
  sizeWatch?.observe(frame);
  resize();

  /* ---------------- theme ---------------- */
  const themeWatch = new MutationObserver(() => options.onTheme?.());
  themeWatch.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme", "data-contrast", "data-palette"],
  });

  /* ---------------- pointer and keys ---------------- */
  const stagePointer: StagePointer = {};
  const raycaster = new THREE.Raycaster();
  const rayAt = (event: { clientX: number; clientY: number }) => {
    const box = canvas.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * 2 - 1;
    const y = -((event.clientY - box.top) / box.height) * 2 + 1;
    applyView();
    raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
    return raycaster.ray.clone();
  };
  let press: { id: number; x: number; y: number; moved: number; mode: "turn" | "scene"; touch: boolean } | null = null;
  const onDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    cancelViewTween();
    const taken = stagePointer.down?.(rayAt(event), event) ?? false;
    press = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: 0, mode: taken ? "scene" : "turn", touch: event.pointerType === "touch" };
    canvas.setPointerCapture?.(event.pointerId);
    canvas.dataset.dragging = "true";
  };
  const onMove = (event: PointerEvent) => {
    if (!press || press.id !== event.pointerId) return;
    const dx = event.clientX - press.x;
    const dy = event.clientY - press.y;
    press.x = event.clientX;
    press.y = event.clientY;
    press.moved += Math.abs(dx) + Math.abs(dy);
    if (press.mode === "scene") {
      stagePointer.move?.(rayAt(event));
      return;
    }
    current = {
      ...current,
      azimuth: current.azimuth - dx * 0.45,
      // A finger moving up or down scrolls the page, so only a mouse or pen tilts the view.
      elevation: press.touch ? current.elevation : clamp(current.elevation + dy * 0.35, limits.minElevation, limits.maxElevation),
    };
    invalidate();
  };
  const onUp = (event: PointerEvent) => {
    if (!press || press.id !== event.pointerId) return;
    const finished = press;
    press = null;
    delete canvas.dataset.dragging;
    canvas.releasePointerCapture?.(event.pointerId);
    if (finished.mode === "scene") stagePointer.up?.();
    else if (finished.moved < 5 && event.type === "pointerup") stagePointer.click?.(rayAt(event));
  };
  const zoomBy = (factor: number) => {
    current = { ...current, distance: clamp(current.distance * factor, limits.minDistance, limits.maxDistance) };
    invalidate();
  };
  const onWheel = (event: WheelEvent) => {
    // A plain wheel scrolls the lab. Pinching a trackpad, or the wheel with Ctrl or ⌘, zooms the view.
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    zoomBy(Math.exp(event.deltaY * 0.004));
  };
  const onKey = (event: KeyboardEvent) => {
    const turn = event.shiftKey ? 5 : 15;
    const actions: Record<string, () => void> = {
      ArrowLeft: () => setView({ azimuth: current.azimuth - turn }, 140),
      ArrowRight: () => setView({ azimuth: current.azimuth + turn }, 140),
      ArrowUp: () => setView({ elevation: clamp(current.elevation + turn / 1.5, limits.minElevation, limits.maxElevation) }, 140),
      ArrowDown: () => setView({ elevation: clamp(current.elevation - turn / 1.5, limits.minElevation, limits.maxElevation) }, 140),
      "+": () => zoomBy(0.88),
      "=": () => zoomBy(0.88),
      "-": () => zoomBy(1 / 0.88),
      Home: () => setView(home, 320),
      "0": () => setView(home, 320),
    };
    const action = actions[event.key];
    if (!action) return;
    event.preventDefault();
    action();
  };
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("keydown", onKey);

  const clear = () => {
    disposeTree(root);
    root.clear();
    setLabels([]);
    stagePointer.down = undefined;
    stagePointer.move = undefined;
    stagePointer.up = undefined;
    stagePointer.click = undefined;
  };

  ready = true;
  applyView();
  invalidate();

  return {
    scene,
    camera,
    root,
    reducedMotion,
    color,
    alpha,
    line,
    segments,
    setLabels,
    invalidate,
    tween,
    view: () => ({ ...current }),
    setView,
    resetView: () => setView(home, 320),
    project,
    pointer: stagePointer,
    clear,
    dispose: () => {
      disposed = true;
      if (frameRequest) cancelAnimationFrame(frameRequest);
      tweens.clear();
      sizeWatch?.disconnect();
      themeWatch.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("keydown", onKey);
      clear();
      probe.remove();
      renderer.dispose();
    },
  };
}

/** What a scene file exports as its default: build the picture once, then redraw it for each new set of props. */
export type SceneFactory<P> = (stage: Stage) => { update: (props: P) => void; dispose?: () => void };
