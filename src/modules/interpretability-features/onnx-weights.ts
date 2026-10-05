/**
 * A minimal reader for the float initializers inside an ONNX file. (The
 * circuits module carries the same reader; modules do not share code.)
 *
 * ONNX is a protobuf: ModelProto.graph (field 7) holds GraphProto, whose
 * node (field 1) and initializer (field 5) entries are what a forward pass
 * needs. Only the wire types protobuf actually uses are handled.
 */

export interface OnnxTensor {
  name: string;
  dims: number[];
  data: Float32Array;
}

export interface OnnxNode {
  name: string;
  opType: string;
  inputs: string[];
}

export interface OnnxGraph {
  initializers: Map<string, OnnxTensor>;
  nodes: OnnxNode[];
}

const FLOAT = 1;
const decoder = new TextDecoder();

class Reader {
  position: number;
  constructor(
    readonly bytes: Uint8Array,
    start = 0,
    readonly end = bytes.length,
  ) {
    this.position = start;
  }

  get done() {
    return this.position >= this.end;
  }

  varint() {
    let result = 0;
    let scale = 1;
    for (;;) {
      if (this.position >= this.end) throw new Error("Truncated varint in ONNX file.");
      const byte = this.bytes[this.position++];
      result += (byte & 0x7f) * scale;
      if (byte < 0x80) return result;
      scale *= 128;
    }
  }

  /** Returns [start, end) of a length-delimited field and skips past it. */
  span(): [number, number] {
    const length = this.varint();
    const start = this.position;
    this.position += length;
    if (this.position > this.end) throw new Error("Truncated field in ONNX file.");
    return [start, this.position];
  }

  skip(wireType: number) {
    if (wireType === 0) this.varint();
    else if (wireType === 1) this.position += 8;
    else if (wireType === 2) this.span();
    else if (wireType === 5) this.position += 4;
    else throw new Error(`Unsupported protobuf wire type ${wireType}.`);
  }
}

function readTensor(bytes: Uint8Array, start: number, end: number): OnnxTensor | null {
  const reader = new Reader(bytes, start, end);
  const dims: number[] = [];
  let name = "";
  let dataType = 0;
  let raw: [number, number] | null = null;
  const floats: number[] = [];

  while (!reader.done) {
    const key = reader.varint();
    const field = Math.floor(key / 8);
    const wireType = key & 7;
    if (field === 1 && wireType === 0) dims.push(reader.varint());
    else if (field === 1 && wireType === 2) {
      const [from, to] = reader.span();
      const packed = new Reader(bytes, from, to);
      while (!packed.done) dims.push(packed.varint());
    } else if (field === 2 && wireType === 0) dataType = reader.varint();
    else if (field === 8 && wireType === 2) {
      const [from, to] = reader.span();
      name = decoder.decode(bytes.subarray(from, to));
    } else if (field === 9 && wireType === 2) raw = reader.span();
    else if (field === 4 && wireType === 2) {
      const [from, to] = reader.span();
      const view = new DataView(bytes.buffer, bytes.byteOffset + from, to - from);
      for (let offset = 0; offset + 4 <= to - from; offset += 4) floats.push(view.getFloat32(offset, true));
    } else if (field === 4 && wireType === 5) {
      const view = new DataView(bytes.buffer, bytes.byteOffset + reader.position, 4);
      floats.push(view.getFloat32(0, true));
      reader.position += 4;
    } else reader.skip(wireType);
  }

  if (dataType !== FLOAT) return null;
  let data: Float32Array;
  if (raw) {
    // Copy so the array is 4-byte aligned regardless of where it sat in the file.
    data = new Float32Array(bytes.slice(raw[0], raw[1]).buffer);
  } else {
    data = Float32Array.from(floats);
  }
  const expected = dims.reduce((product, value) => product * value, 1);
  if (data.length !== expected) {
    throw new Error(`Initializer ${name} has ${data.length} values for shape [${dims.join(", ")}].`);
  }
  return { name, dims, data };
}

function readNode(bytes: Uint8Array, start: number, end: number): OnnxNode {
  const reader = new Reader(bytes, start, end);
  const node: OnnxNode = { name: "", opType: "", inputs: [] };
  while (!reader.done) {
    const key = reader.varint();
    const field = Math.floor(key / 8);
    const wireType = key & 7;
    if (wireType === 2 && (field === 1 || field === 3 || field === 4)) {
      const [from, to] = reader.span();
      const text = decoder.decode(bytes.subarray(from, to));
      if (field === 1) node.inputs.push(text);
      else if (field === 3) node.name = text;
      else node.opType = text;
    } else reader.skip(wireType);
  }
  return node;
}

/** Read every float initializer and every node of an ONNX model. */
export function readOnnxGraph(input: ArrayBuffer | Uint8Array): OnnxGraph {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const model = new Reader(bytes);
  let graph: [number, number] | null = null;
  while (!model.done) {
    const key = model.varint();
    const field = Math.floor(key / 8);
    const wireType = key & 7;
    if (field === 7 && wireType === 2) graph = model.span();
    else model.skip(wireType);
  }
  if (!graph) throw new Error("The ONNX file has no graph.");

  const initializers = new Map<string, OnnxTensor>();
  const nodes: OnnxNode[] = [];
  const reader = new Reader(bytes, graph[0], graph[1]);
  while (!reader.done) {
    const key = reader.varint();
    const field = Math.floor(key / 8);
    const wireType = key & 7;
    if (field === 5 && wireType === 2) {
      const [from, to] = reader.span();
      const tensor = readTensor(bytes, from, to);
      if (tensor) initializers.set(tensor.name, tensor);
    } else if (field === 1 && wireType === 2) {
      const [from, to] = reader.span();
      nodes.push(readNode(bytes, from, to));
    } else reader.skip(wireType);
  }
  return { initializers, nodes };
}
