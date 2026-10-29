#!/usr/bin/env python3
"""First real model-pack benchmark: ONNX -> EXM v3 on real neural networks.

Measures the full ingestion pipeline on actual artifacts (never assumed):
  ingest ms      parse ONNX, fold constants, extract weights, build graph JSON
  write ms       encode v3 chunked container (per-tensor DEFLATE level 1)
  ratio          stored / original payload bytes
  header ms      header-only open (manifest + indexes, zero weight I/O)
  stream ms      per-tensor streaming of every tensor (the runtime load path)
  assemble ms    full decode + per-chunk digest + manifest digest
  integrity      must be ok for every cell, or the run fails

Chunk sweep mirrors ExmBenchmark.defaultConfigs(): 256k (RAM-tight phones),
1m, 4m (default). Cross-language numbers (Kotlin reader on the same files)
are recorded separately by tools/crosscheck/CrossCheck.kt.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import onnx
from onnx import numpy_helper

sys_dir = Path(__file__).resolve().parent
sys_path_list = None
import sys
sys.path.insert(0, str(sys_dir))
import onnx_to_exm  # noqa: E402

CHUNK_SWEEP = [
    ("256k-chunks", 256 * 1024),
    ("1m-chunks", 1024 * 1024),
    ("4m-chunks", onnx_to_exm.DEFAULT_CHUNK_BYTES),
]


def timed_ms(fn, *args, **kw):
    t0 = time.perf_counter()
    value = fn(*args, **kw)
    return (time.perf_counter() - t0) * 1000.0, value


def _payload_only(model_path: Path) -> tuple[list, bytes, dict]:
    """Re-extract weights+graph without writing a container (ingest cost)."""
    model = onnx.load(str(model_path), load_external_data=False)
    folded = onnx_to_exm._fold_constant_nodes(model)
    parts, meta = [], {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.size == 0:
            continue
        raw = arr.tobytes(order="C")
        parts.append(raw)
        meta[init.name] = len(raw)
    return parts, b"".join(parts), {"folded": folded, "tensors": len(meta)}


def run_bench(model_path: Path, model_id: str, name: str, license_name: str,
              out_dir: Path) -> list[dict]:
    rows = []
    ingest_ms, _ = timed_ms(_payload_only, model_path)
    for label, chunk_bytes in CHUNK_SWEEP:
        with tempfile.TemporaryDirectory() as td:
            container = Path(td) / "bench.exm"
            write_ms, report = timed_ms(
                onnx_to_exm.ingest, model_path, container, model_id, name,
                license_name, "0.0.0", chunk_bytes, 2048, 26)
            blob = container.read_bytes()

            header_ms, header = timed_ms(onnx_to_exm.read_header, blob)

            def stream_all() -> tuple[int, bool]:
                pos = header["dataStart"]
                total = 0
                ok = True
                offsets = []
                acc = pos
                for c in header["chunks"]:
                    offsets.append(acc)
                    acc += c["stored"]
                for t in header["tensors"]:
                    start = min(t["chunkStart"], len(header["chunks"]) - 1)
                    stop = min(t["chunkStart"] + t["chunkCount"], len(header["chunks"]))
                    for i in range(t["chunkStart"], stop):
                        c = header["chunks"][i]
                        stored = blob[offsets[i]:offsets[i] + c["stored"]]
                        if onnx_to_exm.sha256_hex(stored) != c["sha"]:
                            ok = False
                        original = onnx_to_exm.zlib.decompress(stored) if c["codec"] == 1 else stored
                        total += len(original)
                return total, ok

            stream_ms, (streamed, stream_ok) = timed_ms(stream_all)
            verify_ms, vreport = timed_ms(onnx_to_exm.verify_container, blob, model_path)
            if not (stream_ok and vreport["integrity"] and streamed == report["payloadBytes"]):
                raise SystemExit(f"integrity failure at {model_id}/{label}")
            rows.append({
                "model": model_id, "chunkConfig": label, "chunkBytes": chunk_bytes,
                "ingestMs": round(ingest_ms, 1), "writeMs": round(write_ms, 1),
                "originalBytes": report["payloadBytes"], "storedBytes": report["storedBytes"],
                "ratio": report["ratio"], "tensors": report["tensors"],
                "chunks": report["chunks"],
                "headerOnlyOpenMs": round(header_ms, 2),
                "headerBytes": header["headerBytes"],
                "streamAllTensorsMs": round(stream_ms, 1),
                "assembleVerifyMs": round(verify_ms, 1),
                "integrity": "ok",
            })
    return rows


def render(rows: list[dict]) -> str:
    header = (f"{'model':<26} {'config':<12} {'orig':>11} {'stored':>11} {'ratio':>6} "
              f"{'tensors':>7} {'chunks':>6} {'ingest':>8} {'write':>8} {'header':>8} "
              f"{'stream':>8} {'verify':>8} {'ok':>3}")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['model']:<26} {r['chunkConfig']:<12} {r['originalBytes']:>11,} "
            f"{r['storedBytes']:>11,} {r['ratio']:>6.3f} {r['tensors']:>7} {r['chunks']:>6} "
            f"{r['ingestMs']:>8.0f} {r['writeMs']:>8.0f} {r['headerOnlyOpenMs']:>8.1f} "
            f"{r['streamAllTensorsMs']:>8.0f} {r['assembleVerifyMs']:>8.0f} {r['integrity']:>3}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="write JSON report here")
    args = ap.parse_args()

    models = [
        (Path(__file__).resolve().parent.parent / "models/artifacts/super-resolution-10.onnx",
         "pipeline/super-resolution-10", "Super Resolution (sub-pixel CNN 2016)", "MIT"),
        (Path(__file__).resolve().parent.parent / "models/artifacts/resnet50-v2-7.onnx",
         "pipeline/resnet50-v2", "ResNet-50 v2 (ImageNet classification)", "Apache-2.0"),
    ]
    all_rows = []
    for path, mid, name, lic in models:
        if not path.is_file():
            raise SystemExit(f"missing model artifact: {path} (see models/artifacts/README)")
        all_rows += run_bench(path, mid, name, lic, args.out)

    print(render(all_rows))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\n")
        print(f"\njson -> {args.out}")


if __name__ == "__main__":
    main()
