#!/usr/bin/env python3
"""Inspect ONNX models: op set, initializers, dtypes — ingestion feasibility check."""
import sys, collections
import onnx
from onnx import TensorProto

DT = {v: k for k, v in TensorProto.DataType.items()}

def inspect(path):
    m = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(m)  # full check (file is local, no external data)
    g = m.graph
    print(f"== {path}")
    print(f"  ir_version={m.ir_version} opsets={[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]} producer={m.producer_name!r}")
    print(f"  inputs:  {[(i.name, [d.dim_value if d.HasField('dim_value') else d.dim_param for d in i.type.tensor_type.shape.dim], DT[i.type.tensor_type.elem_type]) for i in g.input]}")
    print(f"  outputs: {[(o.name, [d.dim_value if d.HasField('dim_value') else d.dim_param for d in o.type.tensor_type.shape.dim], DT[o.type.tensor_type.elem_type]) for o in g.output]}")
    ops = collections.Counter(n.op_type for n in g.node)
    print(f"  nodes={len(g.node)} ops={dict(ops)}")
    total = 0
    by_dt = collections.Counter()
    for init in g.initializer:
        n = 1
        for d in init.dims:
            n *= d
        total += n
        by_dt[DT[init.data_type]] += 1
        raw = len(init.raw_data)
        if raw and raw != n * {"FLOAT": 4, "FLOAT16": 2, "INT8": 1, "INT64": 8, "INT32": 4, "BOOL": 1, "DOUBLE": 8, "UINT8": 1}.get(DT[init.data_type], 0):
            print(f"    !! {init.name}: {DT[init.data_type]} count={n} raw={raw}")
    print(f"  initializers={len(g.initializer)} total_elems={total:,} dtypes={dict(by_dt)}")
    # spot-check raw sizes
    big = sorted(g.initializer, key=lambda i: -len(i.raw_data))[:3]
    for i in big:
        print(f"    largest: {i.name} {DT[i.data_type]} {list(i.dims)} {len(i.raw_data):,}B")
    # external data?
    for t in list(g.initializer):
        for e in t.external_data:
            print(f"    EXTERNAL DATA: {t.name} -> {e.key}={e.value}")

for p in sys.argv[1:]:
    inspect(p)
