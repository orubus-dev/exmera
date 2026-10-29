# Model artifacts

Real artifacts used to develop and benchmark the ONNX ingestion pipeline.
Everything in this directory is a **pipeline validation artifact** — it proves
the conversion/packaging path works on real neural networks; none of it is a
bundled pack model (see `models/manifest.json` for the redistribution policy).

## Committed files

- `super-resolution-10.onnx` — ONNX Model Zoo, sub-pixel CNN 2016 (MIT, as
  published by the zoo). SHA-256 `85f36ff8…86035e`. This is the audit source
  for the Kotlin contract fixture at
  `exm-runtime/src/test/resources/ingested/super-resolution-10.exm`.

## Regenerable (not committed)

- `super-resolution-10.exm` — produced by the compiler (222,250 bytes,
  SHA-256 `c00c5249…cc9b3b6`):
  ```sh
  python3 tools/onnx_to_exm.py ingest super-resolution-10.onnx super-resolution-10.exm \
      --id pipeline/super-resolution-10 --name "Super Resolution (sub-pixel CNN 2016)" --license MIT
  ```
- `resnet50-v2-7.onnx` (102 MB) — ONNX Model Zoo, ResNet-50 v2 (Apache-2.0, as
  published by the zoo). SHA-256 `79102261…f64fefe31`:
  `https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx`
- `resnet50-v2-7.exm` — 95,119,741 bytes, 261 tensors / 269 chunks, used by
  `tools/bench_model_pack.py` for the per-tensor streaming benchmark.

## Benchmark

```sh
python3 tools/bench_model_pack.py
```

Measured results (2026-10-29, 2-vCPU sandbox, DEFLATE level 1, per-tensor
chunking; ratios are payload-relative and exclude container indexes):

| model | config | original | stored | ratio | tensors | chunks | header-only open | stream all tensors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| super-resolution-10 | 4m-chunks | 241,467 | 222,250 | 0.920 | 11 | 11 | 0.1 ms | 1 ms |
| resnet50-v2 | 256k-chunks | 102,456,489 | 95,151,029 | 0.929 | 261 | 608 | 2.9 ms | 484 ms |
| resnet50-v2 | 1m-chunks | 102,456,489 | 95,126,482 | 0.928 | 261 | 328 | 1.9 ms | 484 ms |
| resnet50-v2 | 4m-chunks | 102,456,489 | 95,119,741 | 0.928 | 261 | 269 | 1.7 ms | 498 ms |

Findings that shape the roadmap:

1. **fp32 weights barely deflate** (~0.92–0.93). Real size wins must come from
   true FP16/INT8 conversion (a separate compiler pass), not compression.
2. **Chunk size is a latency/overhead knob, not a compression knob** — 256k vs
   4m differs by <0.1% stored bytes. Small chunks buy streaming granularity for
   RAM-tight devices at the cost of more digests to verify.
3. **Header-only open stays in milliseconds** at 100 MB scale — execution
   planning before weight I/O is viable on budget phones.
4. **Cross-language byte identity**: the Kotlin runtime rebuilds the Python
   compiler's container byte-for-byte (both artifacts verified locally; the
   super-resolution contract is locked in CI via the fixture test).
