# EXM container

`.exm` is Exmera's proposed model container. It is a packaging/runtime format, not a new neural-network architecture.

## Goals

1. Deterministic metadata and integrity verification.
2. Chunked weights for memory-aware loading.
3. Multiple quantization variants.
4. CPU/GPU/NPU compatibility declarations.
5. License provenance carried with every artifact.
6. Forward-compatible versioning.

## Implemented layouts

### v2 — whole payload (legacy, still readable)

```text
magic "EXM1" + version i32 (=2)
manifest (id, version, name, quantization, backend bitset, compression,
          sha-256 of stored payload, stored size, original size,
          min RAM, min Android, license)
payload (single blob, optionally DEFLATE-compressed)
```

`ExmContainer` writes and reads this layout. Kept for compatibility.

### v3 — chunked (canonical, `ExmChunked`)

```text
magic "EXM1" + version i32 (=3) + flags i32 (bit0 = chunked)
manifest  (same field order as v2; compression must be NONE — v3 compresses per chunk)
chunkSize i64 (max original bytes per chunk; last chunk of a tensor may be smaller)
tensorCount i32; per tensor:
    name str | dtype str | rank i32 + dims i32* |
    quantization i32 (-1 = inherit manifest) | chunkStart i32 | chunkCount i32
chunkCount i32; per chunk:
    storedSize i64 | originalSize i64 | codec i32 | sha-256 (64 ASCII bytes)
chunk payloads in index order
```

Contract:

- **Chunks never straddle tensors.** Each tensor's byte range is chunked independently,
  so the runtime can stream exactly one tensor without touching the others.
- **Per-chunk integrity.** Every chunk carries the SHA-256 of its stored bytes; the
  manifest carries the SHA-256 of the whole original payload. Corruption is localized,
  not file-fatal.
- **Header-only open.** `ExmChunked.open()` reads just the manifest and indexes — no
  weight bytes — so an execution planner can decide what to load before any I/O.
- **Deterministic encoding.** Identical manifest/tensors/payload/chunkSize produce
  byte-identical containers.
- **Empty payloads are valid** (a single empty chunk keeps `chunkCount >= 1`).

Declared tensor dtypes: `fp32` (4 B), `fp16` (2 B), `int8` (1 B), `int4` (0.5 B, rounded
up), `bytes` (1 B). Declared tensors must exactly cover the payload; otherwise the
payload is stored as one anonymous tensor and per-tensor streaming is unavailable.

## Quantization

Supported target labels: FP32, FP16, INT8 and INT4. Actual quality/size/runtime results must be measured per model and device. EXM does not claim a fixed compression ratio — use the benchmark harness (`ExmBenchmark`), which measures stored size, ratio, write/header/assemble latency and integrity on real artifacts.

## Streaming

The runtime locates and decodes only the chunks required by an execution plan (`ExmChunked.Reader.readTensor`). It never assumes the complete model fits in RAM.

## Graph tensor convention (`@graph`)

ONNX ingestion (`tools/onnx_to_exm.py`) stores every initializer as a declared
v3 tensor and serializes the network itself into one reserved bytes tensor:

```text
name    = "@graph"        (reserved namespace; initializer names starting with '@' are rejected)
dtype   = "bytes"
shape   = [len(json)]     (one dimension carrying the serialized byte length)
payload = UTF-8 JSON, sorted keys, compact separators — deterministic
```

The graph document (`"format": "exm-graph", "version": 1`) is the execution
planner's source of truth and carries:

- `byteOrder: "little"` — all tensor bytes are little-endian
- `source` — origin format (`onnx`), IR version, opsets, producer, source SHA-256
- `inputs` / `outputs` — runtime I/O signatures (weights excluded)
- `nodes` — every op with its attributes (primitive attrs only), inputs, outputs
- `tensors` — per-initializer provenance: EXM dtype used, ONNX dtype, true
  shape, byte length. Wide numerics (INT32/INT64/DOUBLE/…) are stored as
  opaque little-endian bytes with the shape flattened to `[nbytes]`; the true
  shape lives here
- `emptyTensors` — zero-size initializers, recorded but not stored
- `foldedConstants` — `Constant` nodes hoisted into named initializers
- `opCoverage` — sorted set of op types (runtime capability checks)

Scope limits, enforced as hard rejections: control flow (`If`/`Loop`/`Scan`),
subgraph attributes, sparse initializers, external data, and non-numeric
dtypes (bfloat16/float8). Quantization is *not* claimed by ingestion — the
manifest label reflects the source weights.
