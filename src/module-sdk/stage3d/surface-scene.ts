/**
 * A height surface over two settings: the shape itself, its equal-height lines lying on it, and
 * whatever a lab draws on top (a path, a ball, an arrow, a cut). Pressing it flat turns it into
 * its own contour map, which is how the two pictures are tied together.
 *
 * Loaded on demand by `SurfaceStage`; it is the only reason a surface lab downloads three.js.
 */
import { contourLines } from "./contours";
import { THREE, disposeTree, type SceneFactory, type StageLabel, type StageView } from "./runtime";
import type { SurfaceSceneProps, SurfaceXY } from "./surface-types";

/** The longest floor side and the box height, in scene units. */
const SPAN = 2;
const HEIGHT = 1;
const GRID = 72;
const FLAT_VIEW = { azimuth: 0, elevation: 89.4 } as const;

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

const factory: SceneFactory<SurfaceSceneProps> = (stage) => {
  let props: SurfaceSceneProps | null = null;
  let builtKey = "";
  let overlayKey = "";
  let lift = 1;
  let started = false;
  let cancelLift: () => void = () => undefined;
  let liftedView: StageView | null = null;
  let surfaceMesh: THREE.Mesh | null = null;
  const ringCache = new Map<string, number[]>();

  const meshGroup = new THREE.Group();
  const lineGroup = new THREE.Group();
  const frameGroup = new THREE.Group();
  const overlay = new THREE.Group();
  lineGroup.position.y = 0.003;
  stage.root.add(meshGroup, lineGroup, frameGroup, overlay);
  const clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), HEIGHT);

  const mapping = (p: SurfaceSceneProps) => {
    const [x0, x1] = p.xRange;
    const [y0, y1] = p.yRange;
    const [z0, z1] = p.zRange;
    const scale = SPAN / Math.max(x1 - x0, y1 - y0);
    return {
      scale,
      halfX: ((x1 - x0) * scale) / 2,
      halfZ: ((y1 - y0) * scale) / 2,
      X: (x: number) => (x - (x0 + x1) / 2) * scale,
      Z: (y: number) => -(y - (y0 + y1) / 2) * scale,
      /** Height in scene units at full lift. */
      Y: (value: number) => ((value - z0) / (z1 - z0)) * HEIGHT,
      x: (X: number) => X / scale + (x0 + x1) / 2,
      y: (Z: number) => -Z / scale + (y0 + y1) / 2,
      inside: (point: SurfaceXY) => point.x >= x0 - 1e-9 && point.x <= x1 + 1e-9 && point.y >= y0 - 1e-9 && point.y <= y1 + 1e-9,
    };
  };

  const clipped = <T extends THREE.Material>(material: T): T => {
    material.clippingPlanes = [clip];
    return material;
  };

  /* ---------------- the surface, its grid and its contour lines ---------------- */
  const buildSurface = (p: SurfaceSceneProps) => {
    for (const group of [meshGroup, lineGroup, frameGroup]) {
      disposeTree(group);
      group.clear();
    }
    ringCache.clear();
    const m = mapping(p);
    const low = stage.color("color-mix(in srgb, var(--module-accent) 9%, var(--panel-solid))");
    const high = stage.color("color-mix(in srgb, var(--module-accent) 74%, var(--panel-solid))");

    const positions = new Float32Array((GRID + 1) * (GRID + 1) * 3);
    const colors = new Float32Array((GRID + 1) * (GRID + 1) * 3);
    const shade = new THREE.Color();
    let cursor = 0;
    for (let row = 0; row <= GRID; row += 1) {
      for (let column = 0; column <= GRID; column += 1) {
        const x = p.xRange[0] + ((p.xRange[1] - p.xRange[0]) * column) / GRID;
        const y = p.yRange[0] + ((p.yRange[1] - p.yRange[0]) * row) / GRID;
        const height = m.Y(p.value(x, y));
        positions[cursor] = m.X(x);
        positions[cursor + 1] = Number.isFinite(height) ? height : HEIGHT * 4;
        positions[cursor + 2] = m.Z(y);
        shade.lerpColors(low, high, clamp(height / HEIGHT, 0, 1) ** 0.8);
        colors[cursor] = shade.r;
        colors[cursor + 1] = shade.g;
        colors[cursor + 2] = shade.b;
        cursor += 3;
      }
    }
    const indices: number[] = [];
    for (let row = 0; row < GRID; row += 1) {
      for (let column = 0; column < GRID; column += 1) {
        const a = row * (GRID + 1) + column;
        const b = a + 1;
        const c = a + GRID + 1;
        const d = c + 1;
        indices.push(a, b, d, a, d, c);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    surfaceMesh = new THREE.Mesh(
      geometry,
      clipped(
        new THREE.MeshLambertMaterial({
          vertexColors: true,
          side: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: 1.5,
          polygonOffsetUnits: 2,
        }),
      ),
    );
    meshGroup.add(surfaceMesh);

    // The grid: the surface along each tick of one setting, with the other running its whole range.
    const gridLines: number[] = [];
    const samples = 48;
    const trace = (at: (t: number) => SurfaceXY) => {
      for (let index = 0; index < samples; index += 1) {
        const a = at(index / samples);
        const b = at((index + 1) / samples);
        gridLines.push(m.X(a.x), m.Y(p.value(a.x, a.y)), m.Z(a.y), m.X(b.x), m.Y(p.value(b.x, b.y)), m.Z(b.y));
      }
    };
    for (const tick of p.xTicks) trace((t) => ({ x: tick, y: p.yRange[0] + (p.yRange[1] - p.yRange[0]) * t }));
    for (const tick of p.yTicks) trace((t) => ({ x: p.xRange[0] + (p.xRange[1] - p.xRange[0]) * t, y: tick }));
    const grid = stage.segments(gridLines, { color: "--ink", width: 1, opacity: 0.13, order: 1 });
    clipped(grid.material);
    lineGroup.add(grid);

    const contourPositions: number[] = [];
    for (const level of p.levels) {
      const lines = contourLines(p.value, level, p.xRange, p.yRange, 64);
      const y = m.Y(level);
      for (let index = 0; index < lines.length; index += 4) {
        contourPositions.push(m.X(lines[index]), y, m.Z(lines[index + 1]), m.X(lines[index + 2]), y, m.Z(lines[index + 3]));
      }
    }
    if (contourPositions.length > 0) {
      const contours = stage.segments(contourPositions, { color: "--ink", width: 1.35, opacity: 0.62, order: 2 });
      clipped(contours.material);
      lineGroup.add(contours);
    }

    // The floor: the square of settings the surface stands on.
    const { halfX, halfZ } = m;
    frameGroup.add(
      stage.line([-halfX, 0, halfZ, halfX, 0, halfZ, halfX, 0, -halfZ, -halfX, 0, -halfZ, -halfX, 0, halfZ], {
        color: "--ink",
        width: 1.2,
        opacity: 0.4,
        order: 0,
      }),
    );
    const marks: number[] = [];
    for (const tick of p.xTicks) marks.push(m.X(tick), 0, halfZ, m.X(tick), 0, halfZ + 0.035);
    for (const tick of p.yTicks) marks.push(-halfX, 0, m.Z(tick), -halfX - 0.035, 0, m.Z(tick));
    if (marks.length > 0) frameGroup.add(stage.segments(marks, { color: "--ink", width: 1, opacity: 0.4, order: 0 }));
  };

  /* ---------------- everything that sits on the surface ---------------- */
  const buildOverlay = (p: SurfaceSceneProps) => {
    disposeTree(overlay);
    overlay.clear();
    const m = mapping(p);
    const { halfX, halfZ } = m;
    const labels: StageLabel[] = [];
    const up = lift > 0.04;
    /** Scene height of the surface over (x, y) at the current lift, never above the cut. */
    const heightAt = (x: number, y: number) => Math.min(m.Y(p.value(x, y)), HEIGHT * 1.5) * lift;
    const world = (point: SurfaceXY, raise = 0.006) => [m.X(point.x), heightAt(point.x, point.y) + raise, m.Z(point.y)] as const;
    const shown = (point: SurfaceXY) => m.inside(point) && m.Y(p.value(point.x, point.y)) <= HEIGHT + 1e-6;

    /** A straight move between two settings, bent to follow the surface, stopped where it leaves the square. */
    const drape = (from: SurfaceXY, to: SurfaceXY, raise: number, out: number[]) => {
      const length = Math.hypot(to.x - from.x, to.y - from.y) * m.scale;
      const pieces = clamp(Math.ceil(length / 0.035), 1, 40);
      for (let index = out.length === 0 ? 0 : 1; index <= pieces; index += 1) {
        const t = index / pieces;
        const point = { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t };
        if (!m.inside(point)) return false;
        out.push(...world(point, raise));
      }
      return true;
    };

    for (const slice of p.slices ?? []) {
      const range = slice.along === "x" ? p.xRange : p.yRange;
      const at = (t: number): SurfaceXY => {
        const along = range[0] + (range[1] - range[0]) * t;
        return slice.along === "x" ? { x: along, y: slice.at } : { x: slice.at, y: along };
      };
      const samples = 72;
      const curve: number[] = [];
      for (let index = 0; index <= samples; index += 1) curve.push(...world(at(index / samples), 0.008));
      const line = stage.line(curve, { color: slice.tone, width: 2.6, order: 4 });
      clipped(line.material);
      overlay.add(line);
      if (up) {
        // A curtain from the floor up to the cut, so the cut reads as a one-setting curve standing on end.
        const wall = new Float32Array((samples + 1) * 6);
        const index: number[] = [];
        for (let step = 0; step <= samples; step += 1) {
          const point = at(step / samples);
          const [x, y, z] = world(point, 0);
          wall.set([x, 0, z, x, y, z], step * 6);
          if (step < samples) index.push(step * 2, step * 2 + 1, step * 2 + 3, step * 2, step * 2 + 3, step * 2 + 2);
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(wall, 3));
        geometry.setIndex(index);
        const curtain = new THREE.Mesh(
          geometry,
          clipped(
            new THREE.MeshBasicMaterial({
              color: stage.color(slice.tone),
              transparent: true,
              opacity: 0.16,
              side: THREE.DoubleSide,
              depthWrite: false,
            }),
          ),
        );
        curtain.renderOrder = 3;
        overlay.add(curtain);
      }
    }

    for (const ring of p.rings ?? []) {
      const key = `${ring.level}`;
      let lines = ringCache.get(key);
      if (!lines) {
        lines = contourLines(p.value, ring.level, p.xRange, p.yRange, 96);
        ringCache.set(key, lines);
      }
      const y = clamp(m.Y(ring.level), 0, HEIGHT) * lift + 0.007;
      const positions: number[] = [];
      for (let index = 0; index < lines.length; index += 4) {
        positions.push(m.X(lines[index]), y, m.Z(lines[index + 1]), m.X(lines[index + 2]), y, m.Z(lines[index + 3]));
      }
      if (positions.length === 0) continue;
      // Marching squares gives loose pieces, so a dash pattern cannot run along them; a dashed ring is drawn finer instead.
      overlay.add(
        stage.segments(positions, {
          color: ring.tone,
          width: ring.width ?? (ring.dashed ? 1.6 : 2.4),
          opacity: ring.dashed ? 0.85 : 1,
          order: 4,
        }),
      );
    }

    for (const path of p.paths ?? []) {
      if (path.points.length < 2) continue;
      const positions: number[] = [];
      for (let index = 0; index < path.points.length - 1; index += 1) {
        if (!m.inside(path.points[index])) break;
        if (!drape(path.points[index], path.points[index + 1], 0.01, positions)) break;
      }
      if (positions.length >= 6) {
        const line = stage.line(positions, {
          color: path.tone,
          width: path.width ?? 2.6,
          opacity: path.opacity ?? 1,
          dash: path.dashed ? [0.045, 0.035] : undefined,
          order: path.dashed ? 5 : 6,
        });
        clipped(line.material);
        overlay.add(line);
      }
      if (path.dots) {
        const dotGeometry = new THREE.SphereGeometry(0.017, 12, 8);
        const dotMaterial = clipped(new THREE.MeshBasicMaterial({ color: stage.color(path.tone) }));
        for (const point of path.points.slice(1, -1)) {
          if (!shown(point)) continue;
          const dot = new THREE.Mesh(dotGeometry, dotMaterial);
          dot.position.set(...world(point, 0.012));
          overlay.add(dot);
        }
      }
    }

    for (const arrow of p.arrows ?? []) {
      if (!m.inside(arrow.from)) continue;
      // Shorten an arrow that would leave the square, so its head stays on the surface.
      let reach = 1;
      for (const [value, range] of [
        [[arrow.from.x, arrow.to.x], p.xRange],
        [[arrow.from.y, arrow.to.y], p.yRange],
      ] as const) {
        const delta = value[1] - value[0];
        if (delta > 1e-9) reach = Math.min(reach, (range[1] - value[0]) / delta);
        if (delta < -1e-9) reach = Math.min(reach, (range[0] - value[0]) / delta);
      }
      const to = {
        x: arrow.from.x + (arrow.to.x - arrow.from.x) * reach,
        y: arrow.from.y + (arrow.to.y - arrow.from.y) * reach,
      };
      const positions: number[] = [];
      drape(arrow.from, to, 0.012, positions);
      if (positions.length < 6) continue;
      const count = positions.length / 3;
      const tip = new THREE.Vector3(positions[count * 3 - 3], positions[count * 3 - 2], positions[count * 3 - 1]);
      const before = new THREE.Vector3(positions[count * 3 - 6], positions[count * 3 - 5], positions[count * 3 - 4]);
      const direction = tip.clone().sub(before).normalize();
      const headLength = 0.085;
      const total = new THREE.Vector3(positions[0], positions[1], positions[2]).distanceTo(tip);
      const line = stage.line(positions, {
        color: arrow.tone,
        width: arrow.width ?? 3,
        dash: arrow.dashed ? [0.04, 0.03] : undefined,
        order: 7,
      });
      clipped(line.material);
      overlay.add(line);
      if (total > headLength * 0.7) {
        const head = new THREE.Mesh(
          new THREE.ConeGeometry(0.03, headLength, 18),
          clipped(new THREE.MeshBasicMaterial({ color: stage.color(arrow.tone) })),
        );
        head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
        head.position.copy(tip).addScaledVector(direction, headLength * 0.25);
        head.renderOrder = 7;
        overlay.add(head);
      }
      if (arrow.label) {
        labels.push({ id: `arrow-${arrow.id}`, at: [tip.x, tip.y + 0.06, tip.z], text: arrow.label, tone: arrow.labelTone, dy: -8 });
      }
    }

    for (const point of p.points ?? []) {
      if (!shown(point)) continue;
      const size = point.size ?? 1;
      const [x, y, z] = world(point, 0);
      const tint = stage.color(point.tone);
      if (point.kind === "ball") {
        const radius = 0.05 * size;
        const ball = new THREE.Mesh(new THREE.SphereGeometry(radius, 28, 18), new THREE.MeshLambertMaterial({ color: tint }));
        ball.position.set(x, y + radius, z);
        ball.renderOrder = 8;
        overlay.add(ball);
        if (point.drop && up && y > 0.03) {
          overlay.add(stage.line([x, 0, z, x, y, z], { color: point.tone, width: 1.4, opacity: 0.75, dash: [0.03, 0.025], order: 3 }));
          const foot = new THREE.Mesh(new THREE.CircleGeometry(0.022, 20), new THREE.MeshBasicMaterial({ color: tint }));
          foot.rotation.x = -Math.PI / 2;
          foot.position.set(x, 0.002, z);
          overlay.add(foot);
        }
        if (point.label) {
          labels.push({ id: `point-${point.id}`, at: [x, y + radius * 2, z], text: point.label, tone: point.labelTone, dy: -12 - (point.labelLift ?? 0) });
        }
        continue;
      }
      if (point.kind === "ring") {
        const ring = new THREE.Mesh(
          new THREE.RingGeometry(0.03 * size, 0.045 * size, 28),
          new THREE.MeshBasicMaterial({ color: tint, side: THREE.DoubleSide }),
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(x, y + 0.012, z);
        ring.renderOrder = 6;
        overlay.add(ring);
      } else {
        const arm = 0.04 * size;
        overlay.add(
          stage.segments([x - arm, y + 0.012, z - arm, x + arm, y + 0.012, z + arm, x - arm, y + 0.012, z + arm, x + arm, y + 0.012, z - arm], {
            color: point.tone,
            width: 2.2,
            order: 6,
          }),
        );
      }
      // A cross is named underneath, a ring above, so the two do not fight over one spot.
      if (point.label) {
        labels.push({ id: `point-${point.id}`, at: [x, y, z], text: point.label, tone: point.labelTone, dy: point.kind === "cross" ? 15 : -14 });
      }
    }

    // The height scale stands at the back left corner and grows with the surface.
    if (up) {
      const top = HEIGHT * lift;
      overlay.add(stage.line([-halfX, 0, -halfZ, -halfX, top, -halfZ], { color: "--ink", width: 1.2, opacity: 0.45, order: 0 }));
      const marks: number[] = [];
      for (const tick of p.zTicks) {
        const y = m.Y(tick) * lift;
        marks.push(-halfX, y, -halfZ, -halfX - 0.04, y, -halfZ);
        if (lift > 0.55) labels.push({ id: `z-${tick}`, at: [-halfX - 0.05, y, -halfZ], text: `${tick}`, tone: "is-tick", align: "end", dx: -3 });
      }
      if (marks.length > 0) overlay.add(stage.segments(marks, { color: "--ink", width: 1, opacity: 0.45, order: 0 }));
      if (lift > 0.55) {
        labels.push({ id: "axis-z", at: [-halfX, top, -halfZ], text: p.axes.z, tone: "is-axis", dy: -12 });
      }
    }
    for (const tick of p.xTicks) {
      labels.push({ id: `x-${tick}`, at: [m.X(tick), 0, halfZ + 0.04], text: `${tick}`.replace("-", "−"), tone: "is-tick", dy: 9 });
    }
    for (const tick of p.yTicks) {
      labels.push({ id: `y-${tick}`, at: [-halfX - 0.04, 0, m.Z(tick)], text: `${tick}`.replace("-", "−"), tone: "is-tick", align: "end", dx: -5 });
    }
    labels.push({ id: "axis-x", at: [halfX * 0.72, 0, halfZ + 0.04], text: p.axes.x, tone: "is-axis", dy: 26 });
    labels.push({ id: "axis-y", at: [-halfX - 0.04, 0, -halfZ * 0.72], text: p.axes.y, tone: "is-axis", align: "end", dx: -26 });
    stage.setLabels(labels);
  };

  const setLift = (next: number) => {
    lift = next;
    const scale = Math.max(next, 1e-4);
    meshGroup.scale.y = scale;
    lineGroup.scale.y = scale;
    clip.constant = HEIGHT * next + 0.02;
    if (props) buildOverlay(props);
    stage.invalidate();
  };

  /* ---------------- pointing at the surface ---------------- */
  const caster = new THREE.Raycaster();
  const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const settingsUnder = (ray: THREE.Ray): SurfaceXY | null => {
    if (!props) return null;
    const m = mapping(props);
    let hit: THREE.Vector3 | null = null;
    if (lift < 0.04 || !surfaceMesh) {
      hit = ray.intersectPlane(floor, new THREE.Vector3());
    } else {
      caster.set(ray.origin, ray.direction);
      const found = caster.intersectObject(surfaceMesh, false).find((entry) => entry.point.y <= HEIGHT * lift + 0.02);
      hit = found?.point ?? null;
    }
    if (!hit) return null;
    const point = { x: m.x(hit.x), y: m.y(hit.z) };
    return m.inside(point) ? point : null;
  };
  const place = (ray: THREE.Ray) => {
    const point = settingsUnder(ray);
    if (point) props?.onPlace?.(point.x, point.y);
  };
  stage.pointer.down = (_ray, event) => {
    if (!props?.onPlace || !props.handle || event.pointerType === "touch") return false;
    const handle = props.points?.find((point) => point.id === props?.handle);
    if (!handle) return false;
    const m = mapping(props);
    const height = Math.min(m.Y(props.value(handle.x, handle.y)), HEIGHT) * lift;
    const at = stage.project(new THREE.Vector3(m.X(handle.x), height + 0.05, m.Z(handle.y)));
    return Math.hypot(at.x - event.offsetX, at.y - event.offsetY) < 24;
  };
  stage.pointer.move = place;
  stage.pointer.click = (ray) => {
    if (props?.onPlace) place(ray);
  };

  return {
    update(next) {
      const previous = props;
      props = next;
      const surfaceKey = JSON.stringify([next.surfaceKey, next.xRange, next.yRange, next.zRange, next.levels, next.xTicks, next.yTicks]);
      const rebuilt = surfaceKey !== builtKey;
      if (rebuilt) {
        builtKey = surfaceKey;
        buildSurface(next);
      }
      if (!started) {
        started = true;
        if (next.flat) stage.setView({ ...FLAT_VIEW, target: [0, 0, 0] });
        setLift(next.flat ? 0 : 1);
        overlayKey = "";
      } else if (previous && previous.flat !== next.flat) {
        cancelLift();
        const from = lift;
        const to = next.flat ? 0 : 1;
        if (next.flat) {
          liftedView = stage.view();
          stage.setView({ ...FLAT_VIEW, target: [0, 0, 0] }, 900);
        } else if (liftedView) {
          stage.setView(liftedView, 900);
        } else {
          stage.resetView();
        }
        cancelLift = stage.tween(900, (t) => setLift(from + (to - from) * t));
      }
      const nextOverlayKey = JSON.stringify([next.paths, next.points, next.arrows, next.slices, next.rings, next.zTicks, next.axes]);
      if (rebuilt || nextOverlayKey !== overlayKey) {
        overlayKey = nextOverlayKey;
        buildOverlay(next);
      }
    },
    dispose() {
      cancelLift();
    },
  };
};

export default factory;
