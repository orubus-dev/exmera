#!/usr/bin/env python3
"""Test suite for the ONNX -> EXM v3 ingestion compiler.

Runs with plain python3 (no pytest dependency):
    python3 tools/tests/test_onnx_to_exm.py

Covers:
  1. synthetic model ingestion (Conv/Relu/Reshape, fp32 + INT64 + folded Constant)
  2. byte determinism: two compilations are identical
  3. verify_container: header-only open, graph parse, deep source-tensor compare
  4. corruption detection: any flipped payload byte must fail verification
  5. dtype mapping: INT64 flattened to bytes, true shape kept in graph JSON
  6. Constant nodes with inline tensors fold into named initializers
  7. rejection: control flow (If) and external data
"""
from __future__ import annotations

import json
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import onnx_to_exm  # noqa: E402

PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    if not cond:
        raise AssertionError(f"FAIL: {label}")
    PASS += 1
    print(f"  ok: {label}")


def expect_error(fn, needle: str, label: str) -> None:
    try:
        fn()
    except onnx_to_exm.IngestionError as e:
        check(needle in str(e), f"{label} (message contains {needle!r})")
        return
    raise AssertionError(f"FAIL: {label} did not raise IngestionError")


def build_synthetic_model() -> onnx.ModelProto:
    """Conv(fp32) -> Relu -> Reshape(shape from INT64 initializer + folded Constant)."""
    w = helper.make_tensor("conv.weight", TensorProto.FLOAT, [4, 1, 3, 3],
                           np.arange(36, dtype=np.float32).tobytes(), raw=True)
    b = helper.make_tensor("conv.bias", TensorProto.FLOAT, [4],
                           np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes(), raw=True)
    shape_init = helper.make_tensor("reshape.shape", TensorProto.INT64, [4],
                                    np.array([1, 4, 8, 8], dtype=np.int64).tobytes(), raw=True)
    const_shape = helper.make_tensor("cst.out", TensorProto.INT64, [3],
                                     np.array([4, 4, 4], dtype=np.int64).tobytes(), raw=True)
    nodes = [
        helper.make_node("Conv", ["input", "conv.weight", "conv.bias"], ["conv1"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Relu", ["conv1"], ["relu1"]),
        helper.make_node("Constant", [], ["cst.out"], value=const_shape),
        helper.make_node("Reshape", ["relu1", "reshape.shape"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "synthetic",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=[w, b, shape_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    model.producer_name = "exm-ingestion-tests"
    return model


def assemble_payload(blob: bytes) -> tuple[bytes, dict]:
    """Header + chunk decode (mirrors the runtime path), returns (payload, header)."""
    header = onnx_to_exm.read_header(blob)
    off = header["dataStart"]
    acc = bytearray()
    for c in header["chunks"]:
        stored = blob[off:off + c["stored"]]
        acc += zlib.decompress(stored) if c["codec"] == onnx_to_exm.CODEC_DEFLATE else stored
        off += c["stored"]
    return bytes(acc), header


def read_tensor_from(payload: bytes, header: dict, name: str) -> bytes:
    cursor = 0
    for t in header["tensors"]:
        size = onnx_to_exm.tensor_byte_size(t["dtype"], t["shape"])
        if t["name"] == name:
            return payload[cursor:cursor + size]
        cursor += size
    raise KeyError(name)


def ingest_synthetic(tmp: Path, tag: str) -> tuple[Path, dict]:
    src = tmp / f"synthetic-{tag}.onnx"
    out = tmp / f"synthetic-{tag}.exm"
    src.write_bytes(build_synthetic_model().SerializeToString())
    report = onnx_to_exm.ingest(src, out, "test/synthetic", "Synthetic Net", "MIT",
                                "1.0.0", onnx_to_exm.DEFAULT_CHUNK_BYTES, 1024, 26)
    return out, report


def test_ingest_and_verify(tmp: Path) -> None:
    out, report = ingest_synthetic(tmp, "a")
    check(report["tensors"] == 5, "tensor index = 4 weights + @graph")
    check(report["quantization"] == "fp32", "quantization label honest (fp32 weights)")
    check(report["weightTensors"] == 4, "4 weight tensors (incl. folded constant)")
    check(report["nodes"] == 4 and report["ops"] == ["Constant", "Conv", "Relu", "Reshape"],
          "graph JSON carries every node/op")

    blob = out.read_bytes()
    header = onnx_to_exm.read_header(blob)
    names = {t["name"] for t in header["tensors"]}
    check(names == {"conv.weight", "conv.bias", "reshape.shape", "cst.out", "@graph"},
          "folded Constant named after its output")
    int64_entry = next(t for t in header["tensors"] if t["name"] == "reshape.shape")
    check(int64_entry["dtype"] == "bytes" and int64_entry["shape"] == [32],
          "INT64 flattened to opaque little-endian bytes")
    check(header["manifest"]["license"] == "MIT", "license travels in the manifest")

    v = onnx_to_exm.verify_container(blob, tmp / "synthetic-a.onnx")
    check(v["integrity"] and v["chunksVerified"] == v["chunks"], "all chunk digests verify")
    check(v["sourceTensorCheck"] == "ok", "deep compare vs ONNX source: byte-identical weights")
    check(v["graphParsed"] and v["graphFormat"] == "exm-graph", "@graph parses as exm-graph JSON")

    payload, header = assemble_payload(blob)
    graph_doc = json.loads(read_tensor_from(payload, header, "@graph"))
    check(graph_doc["format"] == "exm-graph" and graph_doc["version"] == 1,
          "graph doc self-describing")
    check(graph_doc["tensors"]["reshape.shape"]["onnxDtype"] == "INT64"
          and graph_doc["tensors"]["reshape.shape"]["shape"] == [4],
          "true dtype/shape preserved in graph JSON")
    check(graph_doc["tensors"]["cst.out"]["onnxDtype"] == "INT64", "folded constant dtype recorded")
    check(graph_doc["inputs"][0]["name"] == "input" and graph_doc["outputs"][0]["name"] == "output",
          "runtime I/O signature excludes weights")
    check(graph_doc["foldedConstants"] == ["cst.out"], "constant fold recorded")


def test_determinism(tmp: Path) -> None:
    a, _ = ingest_synthetic(tmp, "det1")
    b, _ = ingest_synthetic(tmp, "det2")
    check(a.read_bytes() == b.read_bytes(), "two compilations are byte-identical")


def test_corruption_detection(tmp: Path) -> None:
    out, _ = ingest_synthetic(tmp, "corrupt")
    blob = bytearray(out.read_bytes())
    blob[-5] ^= 0xFF  # flip a bit inside the last chunk payload
    expect_error(lambda: onnx_to_exm.verify_container(bytes(blob)), "integrity",
                 "flipped payload byte caught")


def test_rejections(tmp: Path) -> None:
    # control flow
    graph = helper.make_graph(
        [helper.make_node("If", ["cond"], ["out"])], "cf",
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    src = tmp / "controlflow.onnx"
    src.write_bytes(model.SerializeToString())
    expect_error(
        lambda: onnx_to_exm.ingest(src, tmp / "cf.exm", "t/cf", "CF", "MIT", "1.0.0", 1024, 1024, 26),
        "control-flow", "If node rejected")

    # external data
    t = helper.make_tensor("ext", TensorProto.FLOAT, [2], np.zeros(2, np.float32).tobytes(), raw=True)
    t.data_location = TensorProto.EXTERNAL
    graph = helper.make_graph([], "ext", [], [], initializer=[t])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    src = tmp / "external.onnx"
    src.write_bytes(model.SerializeToString())
    expect_error(
        lambda: onnx_to_exm.ingest(src, tmp / "ext.exm", "t/ext", "EXT", "MIT", "1.0.0", 1024, 1024, 26),
        "external data", "external-data model rejected")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("[1/4] ingest + verify")
        test_ingest_and_verify(tmp)
        print("[2/4] determinism")
        test_determinism(tmp)
        print("[3/4] corruption detection")
        test_corruption_detection(tmp)
        print("[4/4] rejection gates")
        test_rejections(tmp)
        print(f"all {PASS} assertions passed")
    print("RESULT=PASS")


if __name__ == "__main__":
    main()
