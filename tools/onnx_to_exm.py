#!/usr/bin/env python3
"""ONNX ingestion compiler: real neural networks -> EXM v3 chunked containers.

Pipeline position (docs/ARCHITECTURE_SHIFT_MODEL_ECOSYSTEM.md):
    open model/checkpoint -> license audit -> conversion -> optimization -> compression -> .EXM

This tool implements the *conversion + packaging* stage for single-file ONNX graphs:

  - every initializer (weights/biases/running stats) becomes a declared EXM v3
    tensor, chunked independently so the runtime can stream one tensor at a time;
  - the ONNX graph itself (ops, attributes, edges, true input/output signatures,
    per-tensor dtype provenance) is serialized as deterministic JSON and stored
    as a reserved "@graph" bytes tensor — the execution planner's source of truth;
  - integrity is structural: per-chunk SHA-256 over stored bytes, manifest
    SHA-256 over the whole original payload, and tensor coverage validation
    identical to the Kotlin reader's (ExmChunked.readHeader).

Honesty boundaries (unchanged from exm_compiler.py):
  - no quantization happens here; the manifest quantization label reflects the
    source weights (FP32 unless every initializer really is fp16/int8);
  - no backend claims beyond CPU (no GPU/NPU execution backends exist yet);
  - control flow (If/Loop/Scan), subgraphs, sparse initializers and external
    data are rejected, not silently degraded.

Binary layout mirrored byte-for-byte from ExmChunked.kt (format v3, big-endian):
  magic "EXM1" | version=3 | flags=1 | manifest | chunkSize i64 |
  tensorCount + per-tensor (name|dtype|rank+dims|quant i32|chunkStart|chunkCount) |
  chunkCount + per-chunk (storedSize i64|originalSize i64|codec i32|sha256 hex64) |
  chunk payloads in index order (each DEFLATE level-1, zlib wrapper)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import onnx
from onnx import TensorProto, numpy_helper

MAGIC = b"EXM1"
VERSION = 3
FLAG_CHUNKED = 1
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
MAX_STRING_BYTES = 1_048_576
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
MAX_CHUNKS = 10_000_000

QUANT = {"fp32": 0, "fp16": 1, "int8": 2, "int4": 3}
BACKENDS = {"cpu": 0, "gpu": 1, "npu": 2}
CODEC_NONE, CODEC_DEFLATE = 0, 1

GRAPH_TENSOR = "@graph"
GRAPH_FORMAT = "exm-graph"
GRAPH_VERSION = 1

# ONNX elem type -> (EXM dtype, bytes-per-element or None for flatten-to-bytes)
DTYPE_MAP = {
    TensorProto.FLOAT: ("fp32", 4),
    TensorProto.FLOAT16: ("fp16", 2),
    TensorProto.INT8: ("int8", 1),
    TensorProto.UINT8: ("bytes", 1),
    TensorProto.BOOL: ("bytes", 1),
    # non-byte-aligned / wide numerics: stored as opaque little-endian bytes,
    # shape flattened to [nbytes]; true dtype+shape live in the graph JSON
    TensorProto.INT16: ("bytes", None),
    TensorProto.INT32: ("bytes", None),
    TensorProto.INT64: ("bytes", None),
    TensorProto.UINT16: ("bytes", None),
    TensorProto.UINT32: ("bytes", None),
    TensorProto.UINT64: ("bytes", None),
    TensorProto.DOUBLE: ("bytes", None),
}
UNSUPPORTED_DTYPES = {
    TensorProto.BFLOAT16, TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E4M3FNUZ,
    TensorProto.FLOAT8E5M2, TensorProto.FLOAT8E5M2FNUZ,
}
CONTROL_FLOW_OPS = {"If", "Loop", "Scan", "SequenceConstruct", "SequenceAt", "SequenceEmpty"}


class IngestionError(Exception):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack_str(value: str) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) > MAX_STRING_BYTES:
        raise IngestionError(f"string too long for EXM: {len(raw)} bytes")
    return struct.pack(">I", len(raw)) + raw


def deflate_chunk(data: bytes) -> bytes:
    """Mirror of ExmChunked.deflate: DEFLATE level 1 (BEST_SPEED), zlib wrapper."""
    comp = zlib.compressobj(level=1, method=zlib.DEFLATED, wbits=15,
                            memLevel=8, strategy=zlib.Z_DEFAULT_STRATEGY)
    return comp.compress(data) + comp.flush()


def _attr_value(attr) -> object:
    """Serializable ONNX attribute value, or raise for unsupported kinds."""
    t = attr.type
    if t == onnx.AttributeProto.INT:
        return attr.i
    if t == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if t == onnx.AttributeProto.FLOAT:
        return attr.f
    if t == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if t == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    if t == onnx.AttributeProto.STRINGS:
        return [x.decode("utf-8", errors="replace") for x in attr.s]
    if t == onnx.AttributeProto.UNDEFINED:
        return None
    kinds = {onnx.AttributeProto.GRAPH: "graph", onnx.AttributeProto.GRAPHS: "graphs",
             onnx.AttributeProto.SPARSE_TENSOR: "sparse tensor",
             onnx.AttributeProto.TENSOR: "dense tensor",
             onnx.AttributeProto.TENSORS: "dense tensors"}
    raise IngestionError(
        f"node {attr.name!r}: attribute kind {kinds.get(t, t)} is not representable "
        f"in {GRAPH_FORMAT} v{GRAPH_VERSION} (control flow / inline tensors unsupported)")


def _io_entry(value_info, exclude: set[str]) -> dict | None:
    name = value_info.name
    if name in exclude or not value_info.type.tensor_type:
        return None
    tt = value_info.type.tensor_type
    if tt.elem_type == 0:
        return {"name": name, "dtype": "undefined", "shape": None}
    shape = []
    for d in tt.shape.dim:
        shape.append(d.dim_value if d.HasField("dim_value") else d.dim_param or "?")
    return {"name": name, "dtype": TensorProto.DataType.Name(tt.elem_type), "shape": shape}


def tensor_bytes(exm_dtype: str, shape: list[int]) -> int:
    """Mirror of ExmChunked.tensorBytes."""
    count = 1 if not shape else 1
    for d in shape:
        count *= max(d, 0) if d else 1
    if exm_dtype == "fp32":
        return count * 4
    if exm_dtype == "fp16":
        return count * 2
    if exm_dtype in ("int8", "bytes"):
        return count
    if exm_dtype == "int4":
        return (count + 1) // 2
    raise IngestionError(f"unsupported dtype {exm_dtype}")


def _fold_constant_nodes(model) -> list[str]:
    """Fold Constant nodes' inline TENSOR attributes into initializers.

    The ONNX Model Zoo and most exporters ship small weights (shape vectors,
    scale constants) inside Constant nodes rather than the initializer list.
    Every real converter hoists them; we do the same, naming the synthetic
    initializer after the node's output so all consumers keep resolving.
    Returns the folded tensor names (in encounter order, deterministic).
    """
    folded: list[str] = []
    existing = {i.name for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        keep_attrs = []
        for a in node.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                out_name = node.output[0] if node.output and node.output[0] else f"const_{len(folded)}"
                if out_name.startswith("@"):
                    raise IngestionError(f"constant output {out_name!r} collides with reserved '@' namespace")
                if out_name in existing:
                    raise IngestionError(f"constant fold target {out_name!r} already exists")
                tensor = onnx.TensorProto(); tensor.CopyFrom(a.t)
                tensor.name = out_name
                tensor.ClearField("external_data")
                if tensor.data_location == TensorProto.EXTERNAL:
                    raise IngestionError("constant node uses external data; single-file models only")
                model.graph.initializer.append(tensor)
                existing.add(out_name)
                folded.append(out_name)
            elif a.type == onnx.AttributeProto.TENSORS:
                raise IngestionError(f"Constant node {node.name!r}: multiple inline tensors unsupported")
            else:
                keep_attrs.append(a)
        del node.attribute[:]
        node.attribute.extend(keep_attrs)
    return folded


def ingest(model_path: Path, out_path: Path, model_id: str, name: str, license_name: str,
           model_version: str, chunk_bytes: int, min_ram: int, min_android: int) -> dict:
    model = onnx.load(str(model_path), load_external_data=False)
    folded = _fold_constant_nodes(model)

    # --- safety gates: only self-contained, static, control-flow-free graphs ---
    for t in model.graph.initializer:
        if t.data_location == TensorProto.EXTERNAL or t.external_data:
            raise IngestionError(f"initializer {t.name!r} uses external data; single-file models only")
        if t.data_type in UNSUPPORTED_DTYPES:
            raise IngestionError(f"initializer {t.name!r}: dtype {TensorProto.DataType.Name(t.data_type)} unsupported")
    if model.graph.sparse_initializer:
        raise IngestionError("sparse initializers are not supported")
    for node in model.graph.node:
        if node.op_type in CONTROL_FLOW_OPS:
            raise IngestionError(f"node {node.name!r}: control-flow op {node.op_type} not supported")
        for a in node.attribute:
            if a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise IngestionError(f"node {node.name!r}: subgraph attributes not supported")

    # --- extract initializers in declaration order (deterministic) ---
    tensors_meta: dict[str, dict] = {}
    blob_parts: list[bytes] = []
    specs: list[tuple[str, str, list[int]]] = []  # (name, exm dtype, exm shape)
    empty_tensors: list[str] = []

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype.kind not in "biufc":
            raise IngestionError(f"initializer {init.name!r}: dtype {arr.dtype} not numeric")
        raw = arr.tobytes(order="C")
        true_shape = list(init.dims)
        if arr.size == 0:
            empty_tensors.append(init.name)
            tensors_meta[init.name] = {
                "exmDtype": None, "onnxDtype": TensorProto.DataType.Name(init.data_type),
                "shape": true_shape, "bytes": 0, "stored": False}
            continue
        if init.name.startswith("@"):
            raise IngestionError(f"initializer name {init.name!r} collides with reserved '@' namespace")
        exm_dtype, elem = DTYPE_MAP[init.data_type]
        if elem is None:  # flatten: EXM stores opaque little-endian bytes
            exm_shape = [len(raw)]
        else:
            exm_shape = true_shape
            if tensor_bytes(exm_dtype, exm_shape) != len(raw):
                raise IngestionError(f"initializer {init.name!r}: size mismatch shape vs raw bytes")
        tensors_meta[init.name] = {
            "exmDtype": exm_dtype, "onnxDtype": TensorProto.DataType.Name(init.data_type),
            "shape": true_shape, "bytes": len(raw), "stored": True}
        specs.append((init.name, exm_dtype, exm_shape))
        blob_parts.append(raw)

    # --- graph JSON (the execution planner's source of truth) ---
    init_names = {i.name for i in model.graph.initializer}
    graph_inputs = [e for e in (_io_entry(v, init_names) for v in model.graph.input) if e]
    if not graph_inputs:
        raise IngestionError("model has no runtime inputs (all graph inputs are weights?)")
    graph_outputs = [e for e in (_io_entry(v, set()) for v in model.graph.output) if e]
    nodes = []
    for node in model.graph.node:
        attrs = {}
        for a in node.attribute:
            attrs[a.name] = _attr_value(a)
        nodes.append({"op": node.op_type, "name": node.name,
                      "inputs": list(node.input), "outputs": list(node.output),
                      "attrs": attrs})
    graph_doc = {
        "format": GRAPH_FORMAT,
        "version": GRAPH_VERSION,
        "byteOrder": "little",
        "source": {
            "format": "onnx",
            "irVersion": model.ir_version,
            "opsets": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
            "producer": model.producer_name,
            "sha256": sha256_hex(model_path.read_bytes()),
        },
        "inputs": graph_inputs,
        "outputs": graph_outputs,
        "nodes": nodes,
        "tensors": tensors_meta,
        "emptyTensors": empty_tensors,
        "foldedConstants": folded,
        "opCoverage": sorted({n["op"] for n in nodes}),
    }
    graph_bytes = json.dumps(graph_doc, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True).encode("utf-8")
    specs.append((GRAPH_TENSOR, "bytes", [len(graph_bytes)]))
    blob_parts.append(graph_bytes)

    payload = b"".join(blob_parts)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise IngestionError(f"payload {len(payload)} exceeds EXM limit {MAX_PAYLOAD_BYTES}")

    # --- quantization label reflects the actual weights; no pretending ---
    present = {m["exmDtype"] for m in tensors_meta.values() if m["stored"]}
    if not present:
        quant = "fp32"
    elif present == {"fp32"}:
        quant = "fp32"
    elif present == {"fp16"}:
        quant = "fp16"
    elif present <= {"int8", "int4"}:
        quant = "int8"
    else:
        quant = "fp32"  # mixed (e.g. fp32 weights + int64 shape tensors)

    # --- encode v3 chunked container (mirror of ExmChunked.write) ---
    # 1. resolve contiguous byte ranges per tensor
    ranges: list[tuple[tuple, int, int]] = []
    cursor = 0
    for (tname, dtype, shape) in specs:
        size = tensor_bytes(dtype, shape)
        if size <= 0:
            raise IngestionError(f"tensor {tname}: zero byte size")
        ranges.append(((tname, dtype, shape), cursor, size))
        cursor += size
    if cursor != len(payload):
        raise IngestionError(f"tensor coverage {cursor} != payload {len(payload)}")

    # 2. chunk each tensor independently (chunks never straddle tensors)
    chunk_entries: list[dict] = []
    stored_chunks: list[bytes] = []
    for (_, _, _), start, size in ranges:
        offset, end = start, start + size
        while offset < end:
            stop = min(offset + chunk_bytes, end)
            original = payload[offset:stop]
            stored = deflate_chunk(original)
            chunk_entries.append({"stored": len(stored), "original": len(original),
                                  "codec": CODEC_DEFLATE, "sha": sha256_hex(stored)})
            stored_chunks.append(stored)
            offset = stop
    if not chunk_entries:  # unreachable (every tensor > 0) but keep the contract
        stored = deflate_chunk(b"")
        chunk_entries.append({"stored": len(stored), "original": 0,
                              "codec": CODEC_DEFLATE, "sha": sha256_hex(stored)})
        stored_chunks.append(stored)

    # 3. map tensors onto the chunk grid
    tensor_entries: list[dict] = []
    chunk_cursor = 0
    for (tname, dtype, shape), _, size in ranges:
        spans = max(-(-size // chunk_bytes), 1)
        tensor_entries.append({"name": tname, "dtype": dtype, "shape": shape,
                               "quant": -1, "chunkStart": chunk_cursor, "chunkCount": spans})
        chunk_cursor += spans

    # 4. serialize manifest + indexes (big-endian, deterministic field order)
    quant_ordinal = QUANT[quant]
    backend_mask = 1 << BACKENDS["cpu"]  # honest: CPU only
    out = bytearray()
    out += MAGIC
    out += struct.pack(">i", VERSION)
    out += struct.pack(">i", FLAG_CHUNKED)
    out += pack_str(model_id) + pack_str(model_version) + pack_str(name)
    out += struct.pack(">i", quant_ordinal)
    out += struct.pack(">i", backend_mask)
    out += struct.pack(">i", CODEC_NONE)  # v3 compresses per chunk
    out += pack_str(sha256_hex(payload))  # manifest digest covers ORIGINAL payload
    out += struct.pack(">Q", len(payload))   # sizeBytes (original, v3 semantics)
    out += struct.pack(">Q", len(payload))   # originalSizeBytes
    out += struct.pack(">i", min_ram)
    out += struct.pack(">i", min_android)
    out += pack_str(license_name)
    out += struct.pack(">q", chunk_bytes)
    out += struct.pack(">i", len(tensor_entries))
    for t in tensor_entries:
        out += pack_str(t["name"]) + pack_str(t["dtype"])
        out += struct.pack(">i", len(t["shape"]))
        for d in t["shape"]:
            out += struct.pack(">i", d)
        out += struct.pack(">i", t["quant"])
        out += struct.pack(">i", t["chunkStart"])
        out += struct.pack(">i", t["chunkCount"])
    out += struct.pack(">i", len(chunk_entries))
    for c in chunk_entries:
        out += struct.pack(">Q", c["stored"])
        out += struct.pack(">Q", c["original"])
        out += struct.pack(">i", c["codec"])
        out += c["sha"].encode("ascii")
    for s in stored_chunks:
        out += s

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(out))
    return {
        "output": str(out_path), "id": model_id, "version": model_version, "name": name,
        "license": license_name, "quantization": quant, "backends": ["cpu"],
        "payloadBytes": len(payload), "storedBytes": len(out),
        "ratio": round(len(out) / len(payload), 4) if payload else 0.0,
        "tensors": len(tensor_entries),
        "weightTensors": len(specs) - 1,
        "emptyTensors": len(empty_tensors),
        "nodes": len(nodes), "ops": graph_doc["opCoverage"],
        "opsets": graph_doc["source"]["opsets"],
        "chunks": len(chunk_entries), "chunkBytes": chunk_bytes,
        "graphBytes": len(graph_bytes),
        "manifestSha256": sha256_hex(payload),
        "containerSha256": sha256_hex(bytes(out)),
    }


# --------------------------------------------------------------------------
# verification — mirror of ExmChunked.readHeader + Reader, pure python
# --------------------------------------------------------------------------

def _read_str(buf: memoryview, pos: int) -> tuple[str, int]:
    (n,) = struct.unpack_from(">I", buf, pos)
    pos += 4
    if n > MAX_STRING_BYTES:
        raise IngestionError("invalid string length")
    return str(buf[pos:pos + n], "utf-8"), pos + n


def read_header(container: bytes) -> dict:
    buf = memoryview(container)
    pos = 0
    if bytes(buf[0:4]) != MAGIC:
        raise IngestionError("Invalid EXM magic")
    pos = 4
    (version,) = struct.unpack_from(">i", buf, pos); pos += 4
    if version != VERSION:
        raise IngestionError(f"Unsupported EXM version {version}")
    (flags,) = struct.unpack_from(">i", buf, pos); pos += 4
    if not flags & FLAG_CHUNKED:
        raise IngestionError("not a chunked EXM container")
    mid, pos = _read_str(buf, pos)
    mver, pos = _read_str(buf, pos)
    mname, pos = _read_str(buf, pos)
    (quant,) = struct.unpack_from(">i", buf, pos); pos += 4
    (mask,) = struct.unpack_from(">i", buf, pos); pos += 4
    (compression,) = struct.unpack_from(">i", buf, pos); pos += 4
    msha, pos = _read_str(buf, pos)
    size_bytes, original_size, = struct.unpack_from(">Q", buf, pos)[0], struct.unpack_from(">Q", buf, pos + 8)[0]
    pos += 16
    (min_ram,) = struct.unpack_from(">i", buf, pos); pos += 4
    (min_android,) = struct.unpack_from(">i", buf, pos); pos += 4
    mlicense, pos = _read_str(buf, pos)
    (chunk_bytes,) = struct.unpack_from(">q", buf, pos); pos += 8
    if chunk_bytes <= 0:
        raise IngestionError("invalid chunkSize")

    (tensor_count,) = struct.unpack_from(">i", buf, pos); pos += 4
    if not 0 <= tensor_count <= 1_000_000:
        raise IngestionError("invalid tensor count")
    tensors = []
    for _ in range(tensor_count):
        tname, pos = _read_str(buf, pos)
        tdtype, pos = _read_str(buf, pos)
        (rank,) = struct.unpack_from(">i", buf, pos); pos += 4
        if not 0 <= rank <= 64:
            raise IngestionError("invalid tensor rank")
        shape = []
        for _ in range(rank):
            (d,) = struct.unpack_from(">i", buf, pos); pos += 4
            shape.append(d)
        (tquant,) = struct.unpack_from(">i", buf, pos); pos += 4
        (cstart,) = struct.unpack_from(">i", buf, pos); pos += 4
        (ccount,) = struct.unpack_from(">i", buf, pos); pos += 4
        tensors.append({"name": tname, "dtype": tdtype, "shape": shape,
                        "quant": tquant, "chunkStart": cstart, "chunkCount": ccount})

    (chunk_count,) = struct.unpack_from(">i", buf, pos); pos += 4
    if not 1 <= chunk_count <= MAX_CHUNKS:
        raise IngestionError("invalid chunk count")
    chunks = []
    for _ in range(chunk_count):
        stored, original = struct.unpack_from(">QQ", buf, pos)
        (codec,) = struct.unpack_from(">i", buf, pos + 16)
        pos += 20  # 8 + 8 + 4 — digest follows immediately
        digest = str(buf[pos:pos + 64], "ascii"); pos += 64
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise IngestionError("malformed chunk digest")
        chunks.append({"stored": stored, "original": original, "codec": codec, "sha": digest})

    manifest = {"id": mid, "version": mver, "name": mname, "quant": quant, "backends": mask,
                "compression": compression, "sha256": msha, "sizeBytes": size_bytes,
                "originalSizeBytes": original_size, "minRamMb": min_ram,
                "minAndroid": min_android, "license": mlicense}

    # structural sanity (identical to Kotlin)
    total_original = sum(c["original"] for c in chunks)
    total_stored = sum(c["stored"] for c in chunks)
    if total_original != size_bytes:
        raise IngestionError("chunk index does not cover manifest size")
    if total_stored > MAX_PAYLOAD_BYTES:
        raise IngestionError("stored payload exceeds format limit")
    for t in tensors:
        if t["quant"] >= 4:
            raise IngestionError("invalid tensor quantization")
    data_start = pos
    return {"manifest": manifest, "chunkBytes": chunk_bytes, "tensors": tensors,
            "chunks": chunks, "dataStart": data_start, "headerBytes": pos}


def tensor_byte_size(dtype: str, shape: list[int]) -> int:
    count = 1 if not shape else 1
    for d in shape:
        count *= d if d else 1
    return {"fp32": count * 4, "fp16": count * 2, "int8": count, "int4": (count + 1) // 2,
            "bytes": count}[dtype.lower()]


def verify_container(container: bytes, source_onnx: Path | None = None) -> dict:
    header = read_header(container)
    chunks = header["chunks"]
    tensors = header["tensors"]
    payload_size = header["manifest"]["sizeBytes"]

    # header-only open worked; now full integrity pass
    payload = bytearray()
    verified = 0
    for i, c in enumerate(chunks):
        stored = bytes(container[header["dataStart"] + sum(x["stored"] for x in chunks[:i]):
                                 header["dataStart"] + sum(x["stored"] for x in chunks[:i + 1])])
        if sha256_hex(stored) != c["sha"]:
            raise IngestionError(f"chunk {i} integrity check failed")
        original = zlib.decompress(stored) if c["codec"] == CODEC_DEFLATE else stored
        if len(original) != c["original"]:
            raise IngestionError(f"chunk {i} size mismatch")
        payload += original
        verified += 1
    payload = bytes(payload)
    if len(payload) != payload_size:
        raise IngestionError("assembled size mismatch")
    if sha256_hex(payload) != header["manifest"]["sha256"]:
        raise IngestionError("manifest integrity check failed")

    # tensor coverage + per-tensor streaming (the runtime path)
    cursor = 0
    streamed = {}
    for t in tensors:
        size = tensor_byte_size(t["dtype"], t["shape"])
        blob = payload[cursor:cursor + size]
        if len(blob) != size:
            raise IngestionError(f"tensor {t['name']}: payload short")
        streamed[t["name"]] = blob
        cursor += size
    if cursor != payload_size and not (len(tensors) == 0):
        raise IngestionError("tensor coverage != payload")

    report = {
        "headerBytes": header["headerBytes"],
        "headerOnlyOpen": True,
        "tensors": len(tensors),
        "chunks": len(chunks),
        "chunksVerified": verified,
        "integrity": True,
        "graphParsed": False,
    }
    if GRAPH_TENSOR in streamed:
        graph = json.loads(streamed[GRAPH_TENSOR])
        report["graphParsed"] = True
        report["graphFormat"] = graph.get("format")
        report["graphNodes"] = len(graph.get("nodes", []))
        report["graphOps"] = graph.get("opCoverage")

    if source_onnx is not None:
        model = onnx.load(str(source_onnx), load_external_data=False)
        mismatches = []
        for init in model.graph.initializer:
            if init.name not in streamed:
                if numpy_helper.to_array(init).size == 0:
                    continue
                mismatches.append(f"{init.name}: missing")
                continue
            want = numpy_helper.to_array(init).tobytes(order="C")
            if streamed[init.name] != want:
                mismatches.append(f"{init.name}: bytes differ")
        report["sourceTensorCheck"] = "ok" if not mismatches else mismatches
        if mismatches:
            raise IngestionError(f"source tensor mismatch: {mismatches}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="ONNX -> EXM v3 ingestion compiler")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="compile a single-file ONNX model into an EXM v3 container")
    pi.add_argument("source", type=Path)
    pi.add_argument("output", type=Path)
    pi.add_argument("--id", required=True)
    pi.add_argument("--name", required=True)
    pi.add_argument("--version", default="1.0.0")
    pi.add_argument("--license", dest="license_name", required=True)
    pi.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    pi.add_argument("--min-ram", type=int, default=2048)
    pi.add_argument("--min-android", type=int, default=26)

    pv = sub.add_parser("verify", help="verify an EXM v3 container (header-only open + integrity)")
    pv.add_argument("container", type=Path)
    pv.add_argument("--source", type=Path, default=None,
                    help="deep-compare every tensor against this ONNX source")

    args = p.parse_args()
    if args.cmd == "ingest":
        if args.chunk_bytes <= 0:
            p.error("--chunk-bytes must be positive")
        report = ingest(args.source, args.output, args.id, args.name, args.license_name,
                        args.version, args.chunk_bytes, args.min_ram, args.min_android)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        report = verify_container(args.container.read_bytes(), args.source)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except IngestionError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
