/**
 * Eight words as points in space, and the unit cube that a squeezing matrix flattens: into a slab,
 * then a sheet, then a line. Loaded on demand by the rank card's 3D picture, with three.js.
 */
import { THREE, type SceneFactory, type StageLabel } from "@app/module-sdk/stage3d/runtime";

export interface SpaceSceneProps {
  /** Each word's three numbers before the squeeze: size, alive, fast. */
  /** `label` is what is printed beside the point; an empty label prints nothing (the word shares a spot with another). */
  words: ReadonlyArray<{ id: string; label: string; at: readonly [number, number, number]; flagged: boolean }>;
  /** What the alive and fast numbers are multiplied by. */
  keepAlive: number;
  keepFast: number;
}

/** Scene units per unit of a word's number, and how far each axis is drawn. */
const UNIT = 0.62;
const REACH = 2.3;

const factory: SceneFactory<SpaceSceneProps> = (stage) => {
  const fixed = new THREE.Group();
  const moving = new THREE.Group();
  stage.root.add(fixed, moving);
  let props: SpaceSceneProps | null = null;
  let shown = { alive: 1, fast: 1 };
  let cancel: () => void = () => undefined;

  // size runs right, alive runs up, fast runs toward the viewer.
  const world = (point: readonly [number, number, number]) => new THREE.Vector3(point[0] * UNIT, point[1] * UNIT, point[2] * UNIT);
  const reach = REACH * UNIT;

  // The three axes, and a faint grid on the sheet where alive is 0.
  const grid: number[] = [];
  for (let tick = -2; tick <= 2; tick += 1) {
    const at = tick * UNIT;
    grid.push(at, 0, -reach, at, 0, reach, -reach, 0, at, reach, 0, at);
  }
  fixed.add(stage.segments(grid, { color: "--ink", width: 1, opacity: 0.12, order: 0 }));
  fixed.add(stage.line([-reach, 0, 0, reach, 0, 0], { color: "--vm-size", width: 1.8, opacity: 0.9, order: 1 }));
  fixed.add(stage.line([0, -reach, 0, 0, reach, 0], { color: "--vm-alive", width: 1.8, opacity: 0.9, order: 1 }));
  fixed.add(stage.line([0, 0, -reach, 0, 0, reach], { color: "--gradient", width: 1.8, opacity: 0.9, order: 1 }));

  const draw = () => {
    if (!props) return;
    for (const child of [...moving.children]) {
      const mesh = child as THREE.Mesh;
      mesh.geometry?.dispose?.();
      const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
      else material?.dispose?.();
    }
    moving.clear();
    const { alive, fast } = shown;
    const accent = stage.color("--module-accent");
    const flag = stage.color("--loss");

    // The unit cube, squeezed: one corner at the origin, one unit along each direction to begin with.
    const sy = Math.max(alive, 0.004);
    const sz = Math.max(fast, 0.004);
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(UNIT, UNIT * sy, UNIT * sz),
      new THREE.MeshLambertMaterial({ color: accent, transparent: true, opacity: 0.26, depthWrite: false, side: THREE.DoubleSide }),
    );
    box.position.set(UNIT / 2, (UNIT * sy) / 2, (UNIT * sz) / 2);
    box.renderOrder = 2;
    moving.add(box);
    const [x1, y1, z1] = [UNIT, UNIT * sy, UNIT * sz];
    const corners = [
      [0, 0, 0], [x1, 0, 0], [x1, y1, 0], [0, y1, 0],
      [0, 0, z1], [x1, 0, z1], [x1, y1, z1], [0, y1, z1],
    ];
    const edges: number[] = [];
    for (const [a, b] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]) {
      edges.push(...corners[a], ...corners[b]);
    }
    moving.add(stage.segments(edges, { color: "--module-accent", width: 1.8, order: 3 }));

    const labels: StageLabel[] = [
      { id: "axis-size", at: [reach, 0, 0], text: "size →", tone: "is-forward", align: "start", dx: 6 },
      { id: "axis-alive", at: [0, reach, 0], text: "alive ↑", tone: "is-attention", dy: -12 },
      { id: "axis-fast", at: [0, 0, reach], text: "fast ↙", tone: "is-gradient", align: "end", dx: -6, dy: 10 },
    ];
    const trails: number[] = [];
    const homes: number[] = [];
    const stems: number[] = [];
    const ball = new THREE.SphereGeometry(0.045, 22, 14);
    for (const word of props.words) {
      const home = world(word.at);
      const now = world([word.at[0], word.at[1] * alive, word.at[2] * fast]);
      const dot = new THREE.Mesh(ball, new THREE.MeshLambertMaterial({ color: word.flagged ? flag : accent }));
      dot.position.copy(now);
      dot.renderOrder = 6;
      moving.add(dot);
      if (home.distanceTo(now) > 0.02) {
        trails.push(home.x, home.y, home.z, now.x, now.y, now.z);
        // A small cross where the word started.
        const arm = 0.022;
        homes.push(home.x - arm, home.y, home.z, home.x + arm, home.y, home.z, home.x, home.y - arm, home.z, home.x, home.y + arm, home.z);
      }
      // A stem down to the sheet where alive is 0, so the point's place in depth can be read.
      if (Math.abs(now.y) > 0.03) stems.push(now.x, now.y, now.z, now.x, 0, now.z);
      if (word.label) {
        labels.push({
          id: `word-${word.id}`,
          at: [now.x, now.y, now.z],
          text: word.label,
          tone: word.flagged ? "is-loss" : "is-ink",
          dy: -13,
        });
      }
    }
    if (stems.length > 0) moving.add(stage.segments(stems, { color: "--module-accent", width: 1.2, opacity: 0.55, order: 4 }));
    if (trails.length > 0) {
      moving.add(stage.segments(trails, { color: "--ink", width: 1, opacity: 0.3, dash: [0.03, 0.03], order: 4 }));
      moving.add(stage.segments(homes, { color: "--ink", width: 1.2, opacity: 0.5, order: 4 }));
    }
    stage.setLabels(labels);
    stage.invalidate();
  };

  return {
    update(next) {
      const first = props === null;
      props = next;
      const from = { ...shown };
      const to = { alive: next.keepAlive, fast: next.keepFast };
      cancel();
      if (first || (Math.abs(from.alive - to.alive) < 1e-6 && Math.abs(from.fast - to.fast) < 1e-6)) {
        shown = to;
        draw();
        return;
      }
      // Glide to the new squeeze, so a jump from 1 to 0 is seen as a press and not a cut.
      cancel = stage.tween(260, (t) => {
        shown = { alive: from.alive + (to.alive - from.alive) * t, fast: from.fast + (to.fast - from.fast) * t };
        draw();
      });
    },
    dispose() {
      cancel();
    },
  };
};

export default factory;
