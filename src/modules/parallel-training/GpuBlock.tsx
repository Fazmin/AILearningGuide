import type { CSSProperties } from "react";
import { Stage3D, type StageView } from "@app/module-sdk";
import { HARDWARE, linkFor } from "./parallel";

const loadScene = () => import("./gpu-block-scene");
/** From the front left and a little above, so all three directions of the block show their length. */
const VIEW: StageView = { azimuth: -36, elevation: 25, distance: 5.9, target: [0, 0.6, 0] };
/** Replicas drawn before the rest are summed up in a label. */
export const SHOWN_REPLICAS = 16;

const count = (value: number) => value.toLocaleString("en-US");
const key = (tone: string): CSSProperties => ({ "--key": `var(${tone})` }) as CSSProperties;

/**
 * The Parallel layout card's third picture: every GPU as a cube. GPU 0 is at the front bottom left,
 * and the three lines of coloured cubes that meet there are the three groups it talks to.
 */
export function GpuBlock({ tensor, pipeline, data, gpus }: { tensor: number; pipeline: number; data: number; gpus: number }) {
  const shownData = Math.min(data, SHOWN_REPLICAS);
  const node = HARDWARE.gpusPerNode;
  const tensorLink = linkFor(1, tensor);
  const dataLink = linkFor(tensor, data);
  const pipelineLink = linkFor(tensor * data, pipeline);
  const group = (name: string, size: number, link: { label: string }) => (size > 1 ? `${name} group of ${count(size)} on ${link.label}` : `${name} off`);
  return (
    <div className="par-block">
      <Stage3D
        scene={loadScene}
        props={{ tensor, pipeline, data, shownData, gpusPerNode: node }}
        view={VIEW}
        aspect={0.52}
        label={`${gpus === 1 ? "The one GPU as a single cube" : `All ${count(gpus)} GPUs as a block of cubes, one cube per GPU`}: ${pipeline} pipeline stage${pipeline === 1 ? "" : "s"} across, ${tensor} tensor slice${
          tensor === 1 ? "" : "s"
        } up, and ${count(data)} data replica${data === 1 ? "" : "s"} going back${
          data > shownData ? `, of which the first ${shownData} are drawn` : ""
        }. GPU 0 is the front bottom left cube. The cubes above it are its tensor group, the cubes to its right its pipeline group, and the cubes behind it its data group: ${group(
          "tensor",
          tensor,
          tensorLink,
        )}; ${group("pipeline", pipeline, pipelineLink)}; ${group("data", data, dataLink)}. The outlined box is one node of ${node} GPUs.`}
      />
      <ul className="stage3d-legend" aria-hidden="true">
        <li><i className="is-ball" style={key("--module-accent")} />GPU 0</li>
        <li><i className="is-ball" style={key("--attention")} />its tensor group, up</li>
        <li><i className="is-ball" style={key("--gradient")} />its pipeline group, across</li>
        <li><i className="is-ball" style={key("--loss")} />its data group, back</li>
        <li><i style={key("--ink")} />one node of {node}: inside it, NVLink</li>
      </ul>
      <p className="par-block__note">
        Count the cubes along each edge and multiply: {pipeline} × {tensor} × {count(data)} = {count(gpus)} GPUs. A group
        that stays inside the outlined box talks over NVLink; a line of cubes that leaves it has to use InfiniBand.
      </p>
    </div>
  );
}
