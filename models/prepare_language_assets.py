"""Prepare the 10k embedding projection and BPE tokenizer assets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

GLOVE_50_SHA256 = "5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_embeddings(path: Path, limit: int) -> tuple[list[str], np.ndarray]:
    words: list[str] = []
    vectors: list[list[float]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            pieces = line.rstrip().split(" ")
            if len(pieces) < 3:
                continue
            word, values = pieces[0], pieces[1:]
            if not word.isalpha() or not word.isascii():
                continue
            try:
                vector = [float(value) for value in values]
            except ValueError:
                continue
            words.append(word)
            vectors.append(vector)
            if len(words) == limit:
                break
    if len(words) < limit:
        raise ValueError(f"{path} yielded {len(words):,} words; expected {limit:,}.")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embedding source has inconsistent vector widths.")
    return words, matrix


def prepare_embeddings(
    source: Path,
    output: Path,
    publish: Path,
    *,
    limit: int,
    seed: int,
) -> dict[str, object]:
    words, matrix = read_embeddings(source, limit)
    standardized = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=seed).fit_transform(standardized)
    umap_projection = umap.UMAP(
        n_components=2,
        n_neighbors=18,
        min_dist=0.12,
        metric="cosine",
        random_state=seed,
    ).fit_transform(matrix)

    vector_path = output / "embedding-10k.f32"
    matrix.astype("<f4").tofile(vector_path)
    index = {
        "format": 1,
        "words": words,
        "count": len(words),
        "dimensions": matrix.shape[1],
        "dtype": "little-endian float32",
        "vector_file": vector_path.name,
        "source": source.name,
        "source_sha256": sha256(source),
        "source_project": "GloVe",
        "source_url": "https://nlp.stanford.edu/projects/glove/",
        "licence": "Public Domain Dedication and License v1.0",
    }
    projection = {
        "format": 1,
        "words": words,
        "pca": pca.round(6).tolist(),
        "umap": umap_projection.round(6).tolist(),
    }
    index_path = output / "embedding-10k.index.json"
    projection_path = output / "embedding-10k.projections.json"
    index_path.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    projection_path.write_text(
        json.dumps(projection, separators=(",", ":")), encoding="utf-8"
    )
    for path in [vector_path, index_path, projection_path]:
        (publish / path.name).write_bytes(path.read_bytes())
    return index


def prepare_tokenizer(
    corpus: Path,
    output: Path,
    publish: Path,
    *,
    vocabulary_size: int,
) -> dict[str, object]:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(), normalizers.Replace("\n", " \n ")]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=2,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        show_progress=True,
    )
    tokenizer.train([str(corpus)], trainer)
    tokenizer_path = output / "bpe-tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    (publish / tokenizer_path.name).write_bytes(tokenizer_path.read_bytes())
    return {
        "format": 1,
        "vocabulary_size": tokenizer.get_vocab_size(),
        "model": "byte-level BPE",
        "corpus": corpus.name,
        "special_tokens": ["<unk>", "<pad>", "<bos>", "<eos>"],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=root / "data" / "glove-wiki-gigaword-50.gz",
        help="Path to a GloVe-format text file. Only the first 10k ASCII words ship.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=root / "data" / "tiny-shakespeare.txt"
    )
    parser.add_argument("--output", type=Path, default=root / "exports")
    parser.add_argument(
        "--publish",
        type=Path,
        default=root.parent / "src" / "modules" / "tokens-embeddings" / "assets",
    )
    parser.add_argument("--words", type=int, default=10_000)
    parser.add_argument("--bpe-vocabulary", type=int, default=512)
    parser.add_argument("--seed", type=int, default=707)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.publish.mkdir(parents=True, exist_ok=True)
    if args.embeddings.name == "glove-wiki-gigaword-50.gz":
        digest = sha256(args.embeddings)
        if digest != GLOVE_50_SHA256:
            raise SystemExit(
                "The pinned GloVe archive checksum changed. "
                f"Expected {GLOVE_50_SHA256}, got {digest}."
            )
    embedding_metadata = prepare_embeddings(
        args.embeddings,
        args.output,
        args.publish,
        limit=args.words,
        seed=args.seed,
    )
    tokenizer_metadata = prepare_tokenizer(
        args.corpus,
        args.output,
        args.publish,
        vocabulary_size=args.bpe_vocabulary,
    )
    (args.output / "language-assets.metadata.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "embeddings": embedding_metadata,
                "tokenizer": tokenizer_metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
