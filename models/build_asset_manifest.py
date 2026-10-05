"""Create and verify the checksum manifest for every shipped teaching asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BUDGET_BYTES = 150 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def onnx_contract(path: Path) -> dict[str, Any]:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

        def tensor_info(value: Any) -> dict[str, Any]:
            return {
                "name": value.name,
                "type": value.type,
                "shape": value.shape,
            }

        return {
            "opset": 18,
            "minimumOnnxRuntimeWeb": "1.20.0",
            "inputs": [tensor_info(value) for value in session.get_inputs()],
            "outputs": [tensor_info(value) for value in session.get_outputs()],
        }
    except ImportError:
        return {
            "opset": 18,
            "minimumOnnxRuntimeWeb": "1.20.0",
            "contractInspection": "Install models/requirements.txt to record inputs and outputs.",
        }


def collect(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    modules_root = root / "src" / "modules"
    records: list[dict[str, Any]] = []
    by_module: dict[str, list[dict[str, Any]]] = {}

    for assets_directory in sorted(modules_root.glob("*/assets")):
        module = assets_directory.parent.name
        for path in sorted(item for item in assets_directory.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            module_relative = path.relative_to(assets_directory.parent).as_posix()
            record: dict[str, Any] = {
                "id": f"{module}/{path.name}",
                "module": module,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "kind": path.suffix.lstrip(".") or "binary",
            }
            if path.suffix == ".onnx":
                record["runtime"] = onnx_contract(path)
            records.append(record)
            by_module.setdefault(module, []).append(
                {
                    "path": module_relative,
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
            )
    return records, by_module


def manifest_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(record["bytes"] for record in records)
    return {
        "format": 1,
        "generatedBy": "python models/build_asset_manifest.py",
        "budgetBytes": BUDGET_BYTES,
        "totalBytes": total,
        "withinBudget": total <= BUDGET_BYTES,
        "assets": records,
    }


def verify_existing(root: Path, manifest_path: Path) -> tuple[int, int]:
    if not manifest_path.exists():
        raise SystemExit(
            "Teaching asset manifest is missing. Run "
            "`python models/build_asset_manifest.py`."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("assets", [])
    expected_paths = {record["path"] for record in records}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "src" / "modules").glob("*/assets/**/*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        added = sorted(actual_paths - expected_paths)
        removed = sorted(expected_paths - actual_paths)
        raise SystemExit(
            "Teaching asset manifest paths are stale."
            f"\nunlisted: {added}\nmissing: {removed}"
        )

    total = 0
    module_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        path = root / record["path"]
        size = path.stat().st_size
        digest = sha256(path)
        if size != record["bytes"] or digest != record["sha256"]:
            raise SystemExit(
                f"Teaching asset changed: {record['path']}. "
                "Refresh and review the manifest."
            )
        total += size
        module = record["module"]
        module_root = root / "src" / "modules" / module
        module_records.setdefault(module, []).append(
            {
                "path": path.relative_to(module_root).as_posix(),
                "bytes": size,
                "sha256": digest,
            }
        )

    if total != payload.get("totalBytes"):
        raise SystemExit("Teaching asset total is stale.")
    for module, assets in module_records.items():
        module_manifest = root / "src" / "modules" / module / "assets.json"
        declared = json.loads(module_manifest.read_text(encoding="utf-8"))
        if declared.get("assets") != assets:
            raise SystemExit(f"{module}/assets.json is stale.")
    return len(records), total


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest_path = root / "assets" / "manifest.json"
    if args.check:
        count, total = verify_existing(root, manifest_path)
        within_budget = total <= BUDGET_BYTES
    else:
        records, by_module = collect(root)
        payload = manifest_payload(records)
        serialized = json.dumps(payload, indent=2) + "\n"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(serialized, encoding="utf-8")
        for module, assets in by_module.items():
            module_manifest = root / "src" / "modules" / module / "assets.json"
            module_manifest.write_text(
                json.dumps({"assets": assets}, indent=2) + "\n",
                encoding="utf-8",
            )
        count = len(records)
        total = payload["totalBytes"]
        within_budget = payload["withinBudget"]

    print(
        f"{'Verified' if args.check else 'Wrote'} {count} assets: "
        f"{total / 1024 / 1024:.2f} MiB / "
        f"{BUDGET_BYTES / 1024 / 1024:.0f} MiB."
    )
    if not within_budget:
        raise SystemExit("Teaching assets exceed the 150 MiB shipping budget.")


if __name__ == "__main__":
    main()
