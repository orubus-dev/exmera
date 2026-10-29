# Exmera Open-Model / EXM Architecture

Exmera no longer treats every AI capability as a model that must be trained internally. The product architecture is now:

```text
Open model/checkpoint
        |
        v
License audit -> conversion -> optimization -> compression
        |
        v
      .EXM
        |
        v
   Exmera Runtime
     /   |   \
   CPU  GPU  NPU
        |
        v
 Camera / Video / Studio / Exllery
```

## Curated redistribution policy

A model is not automatically commercially redistributable because its repository is called open source. Exmera tracks code, checkpoint, dataset, and bundled-asset licensing separately.

### Approved starting catalog

- Real-ESRGAN — BSD-3-Clause repository; package only a checkpoint whose terms have been verified.
- SAM 2 — the upstream project states its model checkpoints, demo code, and training code are Apache-2.0.
- MMPose — Apache-2.0 project; audit selected checkpoint/data terms.
- MMSegmentation — Apache-2.0 project; audit selected model/data terms.
- OpenCV — Apache-2.0 for OpenCV 4.5.0+.

### Review before bundling

PaddleOCR, RAFT, RIFE, Depth Anything, and DINOv2 remain in the review queue until the exact checkpoint and asset terms are recorded.

### Blocked by default

Pretrained InsightFace weights and current Ultralytics YOLO distributions are not bundled until the appropriate commercial redistribution rights are obtained.

## EXM v2

EXM v2 now records:

- model identity/version
- quantization metadata
- CPU/GPU/NPU backend availability
- compression method
- stored SHA-256
- stored and original sizes
- minimum Android/RAM
- license identifier

The runtime verifies the stored payload before decompression and enforces a maximum decompressed payload size.

## Compiler

`tools/exm_compiler.py` produces deterministic EXM v2 packages with DEFLATE compression and integrity metadata. It intentionally does not claim to perform neural-network quantization merely by changing a label. Graph rewriting and true quantization are separate compiler passes and must be implemented with a compatible model-conversion backend.

`tools/onnx_to_exm.py` is the first real conversion pass: it ingests single-file
ONNX graphs into chunked EXM v3 containers — every initializer becomes a
per-tensor-chunked EXM tensor, the network itself is serialized into the
reserved `@graph` tensor (see `EXM_FORMAT.md`), and the reader/writer is
byte-compatible with the Kotlin runtime: Kotlin rebuilds Python-written
containers byte-for-byte, and the contract is locked in CI
(`ExmIngestedContainerTest` + the `exm-ingestion` workflow job). Constant
nodes are folded, control flow / external data / sparse / bfloat16 inputs are
rejected. Quantization and graph *rewriting* remain future passes — ingestion
packages the source graph faithfully.

## First real model-pack benchmark (2026-10-29)

Two ONNX Model Zoo networks went through the full pipeline
(ingest → package → header-only open → per-tensor streaming → integrity):

- super-resolution-10 (MIT): 11 tensors / 11 chunks, 241,467 → 222,250 bytes
- ResNet-50 v2 (Apache-2.0): 261 tensors / 269 chunks, 102,456,489 → 95,119,741 bytes

Measured (see `models/artifacts/README.md`): fp32 weights deflate to only
~0.92–0.93, chunk size changes stored size by <0.1% (it is a streaming
granularity knob), header-only open stays ≤3 ms at 100 MB scale, and Kotlin
writer rebuilds were byte-identical to the Python-written containers.
Conclusion unchanged: real size/runtime wins require true FP16/INT8 passes.

## Remaining production tracks

1. EXM graph importer — ONNX ingestion shipped (packaging-faithful); graph optimizer and true FP16/INT8 conversion remain.
2. Android GPU/NPU execution backends.
3. Model benchmark matrix and per-device model selection.
4. Camera RAW/gyro/HDR-bracket/full-resolution fusion.
5. Production dense optical flow and temporal video reconstruction.
6. Studio timeline/editor UI and render graph.
7. Exllery indexing/search UI and background indexing.
8. Lens Store distribution, signatures, permissions and sandboxing.
9. Export codec matrix, metadata privacy and crash-safe background rendering.
10. Real-device validation across low/mid/high-tier Android hardware.
11. Release signing, staged rollout, crash/ANR telemetry and certification.

The architecture deliberately allows these tracks to progress independently because all learned components are exposed through EXM model/backend contracts.
