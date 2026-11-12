/**
 * The layout as a block of GPUs. One small cube is one GPU, and the three ways of splitting a
 * training step are the block's three directions: pipeline stages across, tensor slices up, data
 * replicas back. Loaded on demand by `GpuBlock`, with three.js.
 */
import { THREE, type SceneFactory, type StageLabel } from "@app/module-sdk/stage3d/runtime";

export interface GpuBlockProps {
  tensor: number;
  pipeline: number;
  data: number;
  /** Replicas actually drawn; the rest are named in a label behind the block. */
  shownData: number;
  gpusPerNode: number;
}

/** The longest side of the block, and the largest a single cube is drawn, in scene units. */
const SPAN = 2.3;
const LARGEST_CUBE = 0.46;
const count = (value: number) => value.toLocaleString("en-US");

const factory: SceneFactory<GpuBlockProps> = (stage) => {
  const group = new THREE.Group();
  stage.root.add(group);
  let built = "";

  const build = (p: GpuBlockProps) => {
    for (const child of [...group.children]) {
      const mesh = child as THREE.Mesh;
      mesh.geometry?.dispose?.();
      (mesh.material as THREE.Material | undefined)?.dispose?.();
    }
    group.clear();

    const across = p.pipeline;
    const up = p.tensor;
    const back = p.shownData;
    const pitch = Math.min(SPAN / Math.max(across, up, back), LARGEST_CUBE);
    const size = pitch * 0.8;
    const width = across * pitch;
    const height = up * pitch;
    const depth = back * pitch;
    const at = (i: number, j: number, k: number) =>
      new THREE.Vector3(-width / 2 + (i + 0.5) * pitch, (j + 0.5) * pitch, depth / 2 - (k + 0.5) * pitch);

    const plain = stage.color("color-mix(in srgb, var(--ink) 24%, var(--panel-solid))");
    const self = stage.color("--module-accent");
    const tensorTone = stage.color("--attention");
    const pipelineTone = stage.color("--gradient");
    const dataTone = stage.color("--loss");

    const total = across * up * back;
    const cubes = new THREE.InstancedMesh(new THREE.BoxGeometry(size, size, size), new THREE.MeshLambertMaterial(), total);
    const place = new THREE.Matrix4();
    let index = 0;
    for (let i = 0; i < across; i += 1) {
      for (let j = 0; j < up; j += 1) {
        for (let k = 0; k < back; k += 1) {
          place.setPosition(at(i, j, k));
          cubes.setMatrixAt(index, place);
          // GPU 0 sits at the front bottom left. Its three groups are the three lines of cubes that meet there.
          const tone =
            i === 0 && j === 0 && k === 0
              ? self
              : i === 0 && k === 0
                ? tensorTone
                : j === 0 && k === 0
                  ? pipelineTone
                  : i === 0 && j === 0
                    ? dataTone
                    : plain;
          cubes.setColorAt(index, tone);
          index += 1;
        }
      }
    }
    cubes.instanceMatrix.needsUpdate = true;
    if (cubes.instanceColor) cubes.instanceColor.needsUpdate = true;
    group.add(cubes);

    // One node: the first `gpusPerNode` ranks. Ranks count up the tensor slices first, then back through
    // the replicas, then across the stages, so a node is a box of cubes inside each stage it touches.
    const perStage = up * p.data;
    const nodeCells = Math.min(p.gpusPerNode, across * perStage);
    const edges: number[] = [];
    const box = (low: THREE.Vector3, high: THREE.Vector3) => {
      const [x0, y0, z0, x1, y1, z1] = [low.x, low.y, low.z, high.x, high.y, high.z];
      const corners = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
      ];
      for (const [a, b] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]) {
        edges.push(...corners[a], ...corners[b]);
      }
    };
    const margin = pitch * 0.5;
    for (let i = 0; i * perStage < nodeCells; i += 1) {
      const inStage = Math.min(perStage, nodeCells - i * perStage);
      const slices = Math.min(up, inStage);
      const replicas = Math.min(back, Math.ceil(inStage / up));
      const low = at(i, 0, replicas - 1).addScalar(-margin);
      const high = at(i, slices - 1, 0).addScalar(margin);
      box(low, high);
    }
    if (edges.length > 0) group.add(stage.segments(edges, { color: "--ink", width: 1.8, opacity: 0.85, order: 5 }));

    const nodeSlices = Math.min(up, nodeCells);
    const nodeReplicas = Math.min(back, Math.ceil(Math.min(perStage, nodeCells) / up));
    const labels: StageLabel[] = [
      { id: "self", at: [-width / 2 + pitch / 2, 0, depth / 2], text: "GPU 0", tone: "is-accent", dy: 14 },
      {
        id: "pipeline",
        at: [width / 2, 0, depth / 2],
        text: `pipeline × ${p.pipeline} →`,
        tone: "is-gradient",
        align: across > 1 ? "end" : "start",
        dx: across > 1 ? 0 : 10,
        dy: 16,
      },
      {
        id: "tensor",
        at: [-width / 2, height, depth / 2],
        text: `tensor × ${p.tensor} ↑`,
        tone: "is-attention",
        align: "end",
        dx: -10,
        dy: -4,
      },
      {
        id: "data",
        at: [-width / 2, 0, -depth / 2],
        text: `data × ${count(p.data)} ↗`,
        tone: "is-loss",
        align: "end",
        dx: -10,
        dy: 6,
      },
      {
        id: "node",
        at: [-width / 2 + pitch / 2, nodeSlices * pitch + margin * 0.4, depth / 2 - nodeReplicas * pitch],
        text: `one node: ${nodeCells} GPUs on NVLink`,
        tone: "is-ink",
        align: "start",
        dx: 6,
        dy: -12,
      },
    ];
    if (p.data > back) {
      labels.push({
        id: "more",
        at: [0, height, -depth / 2],
        text: `+ ${count(p.data - back)} more replicas behind`,
        tone: "is-faint",
        dy: -14,
      });
    }
    stage.setLabels(labels);
    // Stand back far enough to see the whole block, however long it is.
    const radius = Math.hypot(width, height, depth) / 2;
    stage.setView({ target: [0, height * 0.42, 0], distance: Math.max(3.6, radius * 3.7) }, 260);
  };

  return {
    update(next) {
      const signature = JSON.stringify(next);
      if (signature === built) return;
      built = signature;
      build(next);
    },
  };
};

export default factory;
