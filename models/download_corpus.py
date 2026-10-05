"""Download and validate the public-domain teaching corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import urllib.request
from pathlib import Path

CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
)
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_corpus(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    cleaned = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or character.isprintable()
    )
    return cleaned.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "tiny-shakespeare.txt",
    )
    parser.add_argument(
        "--accept-upstream-change",
        action="store_true",
        help="Keep a download whose checksum differs after manually reviewing it.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".download")
    print(f"Downloading {CORPUS_URL}")
    urllib.request.urlretrieve(CORPUS_URL, temporary)
    source_digest = sha256(temporary)

    if source_digest != CORPUS_SHA256 and not args.accept_upstream_change:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            "Corpus checksum changed. Review the pinned source before using "
            f"--accept-upstream-change.\nexpected {CORPUS_SHA256}\nactual   {source_digest}"
        )

    cleaned = clean_corpus(temporary.read_text(encoding="utf-8"))
    args.output.write_text(cleaned, encoding="utf-8")
    temporary.unlink()
    cleaned_digest = sha256(args.output)
    provenance = {
        "name": "Tiny Shakespeare",
        "source": CORPUS_URL,
        "sourceSha256": source_digest,
        "sha256": cleaned_digest,
        "bytes": args.output.stat().st_size,
        "licence": "Public-domain works of William Shakespeare; transcription source pinned above.",
        "purpose": "Shared character corpus for the bigram, RNN, and transformer teaching models.",
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} "
        f"({provenance['bytes']:,} bytes, {cleaned_digest})"
    )


if __name__ == "__main__":
    main()
