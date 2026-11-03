#!/usr/bin/env python3
"""Measure actual inference graphs at1x3x224x224; report unsupported FLOP ops."""
import argparse
import importlib.util
import json
import hashlib
import math
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / 'src/ff2_expanded_models.py'
spec = importlib.util.spec_from_file_location('profile_zoo', SRC)
zoo = importlib.util.module_from_spec(spec); spec.loader.exec_module(zoo)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=zoo.TRAINABLE_METHODS, required=True)
    p.add_argument('--sources', required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise ValueError('Do not overwrite measurements')
    torch.set_num_threads(2); torch.manual_seed(17)
    model = zoo.Adapter(a.method, a.sources, fixture=True).eval()
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import get_shape
    def attention(inputs, outputs):
        q,k,v=(get_shape(x) for x in inputs[:3])
        if not q or not k or not v: raise ValueError('Unknown attention shape')
        return math.prod(q[:-2])*q[-2]*k[-2]*(q[-1]+v[-1])
    def fused_mha(inputs, outputs):
        q,k,v=(get_shape(x) for x in inputs[:3])
        if not q or len(q)!=3: raise ValueError('Unknown fused MHA shape')
        b,n,d=q; s=k[1]
        return b*(n+k[1]+v[1])*d*d + 2*b*n*s*d + b*n*d*d
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        f = FlopCountAnalysis(model, x)
        f.set_op_handle('aten::scaled_dot_product_attention',attention,
                        'aten::_native_multi_head_attention',fused_mha)
        f.unsupported_ops_warnings(False); f.uncalled_modules_warnings(False)
        counted = float(f.total()); unsupported = dict(f.unsupported_ops())
    result = {'method': a.method, **model.counts(), 'input_shape': [1,3,224,224],
              'scope': 'detector inference only; loss, preprocessing and backward excluded',
              'flop_convention': 'fvcore one fused multiply-add counts as one operation',
              'counted_gflops': counted/1e9, 'complete_flop_coverage': not bool(unsupported),
              'unsupported_operations': unsupported, 'uncalled_modules': sorted(f.uncalled_modules()),
              'attention_accounted': 'QK/AV matmuls; fused MHA additionally includes QKV/output projections; softmax remains unsupported',
              'profiler_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'architecture_source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),
              'torch_version':str(torch.__version__),
              'architecture_only_random_weights': True, 'accuracy_evaluated': False}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result))


if __name__ == '__main__': main()
