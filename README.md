# Exmera

Exmera is a hardware-aware computational photography and media engine for Android.

## Principles

- Professional capability, consumer language.
- Budget phones are first-class devices.
- Small base application; engines and models are downloadable.
- Prefer deterministic CV/GPU processing where AI is unnecessary.
- Multi-frame information is preserved for reconstruction instead of relying only on single-frame hallucination.
- `.exm` is a proposed model container; compression targets are measured, not assumed.

## Repository layout

```text
app/             Android application and UI
exmera-engine/   camera/media orchestration and device profiling
exm-runtime/     EXM format and runtime primitives
models/          model manifests and licensing metadata
native/          future C++/Vulkan acceleration
runtime/         future execution backends
```

## Model packs

- Core — camera intelligence and basic enhancement
- Smart — computational photography and automatic editing
- Creator — segmentation, matting, tracking, restoration and creative tools
- Cinema — heavy restoration, video reconstruction and high-quality processing

## Development status

v0.1 establishes the Android multi-module foundation, hardware profile selection, consumer UI, and initial EXM format primitives.

v0.2 adds the real model pipeline: EXM v3 chunked container with per-tensor streaming, the ONNX ingestion compiler (`tools/onnx_to_exm.py` — weights as streamed tensors, network serialized into the reserved `@graph` tensor), and the first model-pack benchmark on real neural networks, with a cross-language guarantee: the Kotlin runtime rebuilds Python-written containers byte-for-byte (locked in CI).

v0.2.0 is the client milestone (`app.orubus.exmera`): the app ships with zero AI models. Every camera mode works without a pack — PRO (full manual Camera2: ISO, shutter, focus, white balance, zoom, RAW DNG where the sensor supports it, scrubbed JPEG fallback), VIDEO (no location metadata, ever), IMAGE (auto metadata scrubbing: pixel-only re-encode on every save), PORTRAIT (model-free software bokeh), PANO (on-device coarse-to-fine stitching), plus multi-frame boost fusion. The EXM runtime entry point is built in: `.exm` packs are imported via the packs sheet, header-validated and chunk-integrity-checked before install. Next milestones: true FP16/INT8 conversion passes, execution planner over `@graph`, and the first tiny in-house models (depth, face/body tracking) as EXM packs.
