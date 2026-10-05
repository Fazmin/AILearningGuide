import { describe, expect, it } from "vitest";
import definition, { hydrateEmbeddingsState } from "./module";
import {
  approximateSearch,
  buildIndex,
  cosine,
  dot,
  embed,
  exactSearch,
  kmeans,
  lookup,
  norm,
  PRESETS,
  recallAtK,
  sharedWords,
} from "./vectors";

const byId = (query: string, metric: "cosine" | "dot") => exactSearch(buildIndex(""), query, metric);
const preset = (label: string) => PRESETS.find((item) => item.label === label)!.query;

describe("embedding arithmetic", () => {
  it("computes cosine and dot product from components", () => {
    expect(dot([1, 2, 0], [3, 1, 5])).toBe(5);
    expect(cosine([1, 0], [1, 0])).toBeCloseTo(1, 12);
    expect(cosine([1, 0], [0, 1])).toBe(0);
    const q = [0.3, 0.4];
    const d = [0.6, 0.1];
    expect(dot(q, d)).toBeCloseTo(norm(q) * norm(d) * cosine(q, d), 12);
  });

  it("mean-pools known words and ignores unknown ones", () => {
    expect(embed("dog")).toEqual([0, 0, 0, 0, 1, 0, 0, 0]);
    expect(embed("dog zebra")).toEqual([0, 0, 0, 0, 1, 0, 0, 0]);
    expect(embed("dog dock")).toEqual([0, 0.5, 0, 0, 0.5, 0, 0, 0]);
    expect(norm(embed("zebra quokka"))).toBe(0);
    expect(cosine(embed("zebra"), embed("dog"))).toBe(0);
  });

  it("reads plurals and clock times through the lexicon", () => {
    expect(lookup("docks")).toBe("dock");
    expect(lookup("closes")).toBe("close");
    expect(lookup("18:00")).toBe("<clock>");
    expect(lookup("bluebell")).toBeNull();
  });

  it("gives identical rankings for cosine and dot when every vector has length 1", () => {
    const vectors = [
      [1, 0],
      [0.6, 0.8],
      [0, 1],
    ];
    const q = [0.8, 0.6];
    const cosOrder = vectors.map((v, i) => ({ i, s: cosine(q, v) })).sort((a, b) => b.s - a.s).map((e) => e.i);
    const dotOrder = vectors.map((v, i) => ({ i, s: dot(q, v) })).sort((a, b) => b.s - a.s).map((e) => e.i);
    expect(dotOrder).toEqual(cosOrder);
  });
});

describe("the claims the lesson makes about the harbor corpus", () => {
  it("pier shut: Dock hours is nearest by cosine", () => {
    const ranked = byId(preset("pier shut"), "cosine");
    expect(ranked[0].id).toBe("hours");
    expect(ranked[0].cosine.toFixed(3)).toBe("0.999");
    expect(ranked[0].angle.toFixed(1)).toBe("1.9");
    expect(ranked[1].id).toBe("south");
    expect(ranked[1].cosine.toFixed(3)).toBe("0.989");
    expect(norm(embed(preset("pier shut"))).toFixed(3)).toBe("0.720");
  });

  it("dog on the boat: dense finds Pets on board with zero shared words, keyword prefers Mooring fees", () => {
    const ranked = byId(preset("dog on the boat"), "cosine");
    expect(ranked[0].id).toBe("pets");
    expect(ranked[0].cosine.toFixed(3)).toBe("0.881");
    expect(ranked[0].shared).toEqual([]);
    expect(ranked[0].keywordRank).toBe(6);
    const keywordLeader = [...ranked].sort((a, b) => a.keywordRank - b.keywordRank)[0];
    expect(keywordLeader.id).toBe("mooring");
    expect(sharedWords(preset("dog on the boat"), "Mooring a boat overnight")).toEqual(["boat"]);
  });

  it("password: cosine and dot product disagree on the top passage", () => {
    const cos = byId(preset("password"), "cosine");
    const raw = byId(preset("password"), "dot");
    expect(cos[0].id).toBe("lost");
    expect(cos[1].id).toBe("notice");
    expect(raw[0].id).toBe("notice");
    expect(raw[1].id).toBe("lost");
    const notice = cos.find((item) => item.id === "notice")!;
    const lost = cos.find((item) => item.id === "lost")!;
    expect(notice.length.toFixed(3)).toBe("0.565");
    expect(lost.length.toFixed(3)).toBe("0.427");
    expect(lost.cosine.toFixed(3)).toBe("0.981");
    expect(notice.cosine.toFixed(3)).toBe("0.955");
    expect(notice.dot.toFixed(3)).toBe("0.306");
    expect(lost.dot.toFixed(3)).toBe("0.238");
  });

  it("password with 4 cells: one probe misses a top-3 neighbour, two probes recover it", () => {
    const index = buildIndex("");
    const query = preset("password");
    const cells = kmeans(index.map((item) => item.unit), 4);
    const exact = exactSearch(index, query, "cosine").slice(0, 3).map((item) => item.id);
    const one = approximateSearch(index, cells, query, "cosine", 1, 3);
    const two = approximateSearch(index, cells, query, "cosine", 2, 3);
    expect(recallAtK(exact, one.top)).toBeCloseTo(2 / 3, 12);
    expect(one.scoredIds.length).toBe(2);
    expect(recallAtK(exact, two.top)).toBe(1);
    expect(two.scoredIds.length).toBe(7);
  });

  it("three cells and one probe keep recall at 1 for every preset", () => {
    const index = buildIndex("");
    const cells = kmeans(index.map((item) => item.unit), 3);
    for (const { query } of PRESETS) {
      const exact = exactSearch(index, query, "cosine").slice(0, 3).map((item) => item.id);
      expect(recallAtK(exact, approximateSearch(index, cells, query, "cosine", 1, 3).top)).toBe(1);
    }
  });

  it("probing every cell is exact search", () => {
    const index = buildIndex("The café by the south dock opens at dawn.");
    for (const { query } of PRESETS) {
      for (const cellsCount of [1, 2, 3, 4]) {
        const cells = kmeans(index.map((item) => item.unit), cellsCount);
        const exact = exactSearch(index, query, "dot").slice(0, 5).map((item) => item.id);
        const all = approximateSearch(index, cells, query, "dot", cellsCount, 5);
        expect(all.top).toEqual(exact);
        expect(all.scoredIds.length).toBe(index.length);
      }
    }
  });
});

describe("state", () => {
  it("migrates a version 1 payload and drops the hybrid and chunk keys", () => {
    const old = JSON.stringify({
      query: "When does the north dock close?",
      userText: "Bring a dog on the afternoon ferry and check when the north dock shuts.",
      chunkSize: 12,
      hybrid: 0.4,
    });
    const state = hydrateEmbeddingsState(old);
    expect(state.query).toBe(definition.initialState.query);
    expect(state.userText).toBe("");
    expect(state).not.toHaveProperty("hybrid");
    expect(state).not.toHaveProperty("chunkSize");
    expect(state.metric).toBe("cosine");
  });

  it("keeps a learner's own version 1 text and clamps probe to cells", () => {
    const state = hydrateEmbeddingsState(
      JSON.stringify({ query: "ferry dog", userText: "mine", hybrid: 1, cells: 2, probe: 9 }),
    );
    expect(state.query).toBe("ferry dog");
    expect(state.userText).toBe("mine");
    expect(state.probe).toBe(2);
  });
});
