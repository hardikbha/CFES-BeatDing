"""Convert the verified cached official V1 legacy archive, preserving tensors.

No download and no change to model initialization values or loss mathematics.
The legacy trusted official archive requires weights_only=False once; every
subsequent model load uses the unchanged weights_only=True model implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--protocol', type=Path, required=True); args = p.parse_args()
    protocol = json.loads(args.protocol.read_text()); root = Path(protocol['run_root'])
    source = Path(protocol['resnet_weights']['path'])
    assert sha(source) == protocol['resnet_weights']['sha256'] and sha(source).startswith('19c8e357')
    assert not list((root/'full_experiment_runs').iterdir()), 'Normalize before any new training'
    original = torch.load(source, map_location='cpu', weights_only=False)
    assert isinstance(original, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in original.items())
    target = root/'preflight/resnet50_imagenet1k_v1_modern.pth'
    assert not target.exists()
    torch.save(original, target)
    restored = torch.load(target, map_location='cpu', weights_only=True)
    assert list(original) == list(restored) and all(torch.equal(original[k], restored[k]) for k in original)
    assert getattr(original, '_metadata', None) == getattr(restored, '_metadata', None)
    conversion = {'source': str(source), 'source_sha256': sha(source), 'target': str(target), 'target_sha256': sha(target),
                  'all_tensor_values_and_metadata_identical': True, 'tensor_entries': len(original),
                  'reason': 'PyTorch2.7 weights_only=True cannot load the legacy official V1 tar format; serialization-only compatibility conversion before new training/test scoring'}
    (root/'preflight/weight_format_conversion.json').write_text(json.dumps(conversion, indent=2)+'\n')
    shutil.copyfile(args.protocol, args.protocol.with_name('protocol_before_weight_format_conversion.json'))
    shutil.copyfile(root/'FROZEN_PROTOCOL.json', root/'FROZEN_PROTOCOL.before_weight_format_conversion.json')
    protocol['resnet_weights'].update(source_path=str(source), source_sha256=sha(source), path=str(target), sha256=sha(target),
                                      tensor_values_preserved=True)
    protocol['serialization_compatibility_note'] = conversion
    for relative in protocol['configs'].values():
        # The protocol is under REPO/CES_FL/proposals/<id>/.
        repo = args.protocol.resolve().parents[3]
        cfg_path = repo/relative
        c = json.loads(cfg_path.read_text())
        backup = cfg_path.parent/'before_weight_format_conversion'/cfg_path.name
        backup.parent.mkdir(exist_ok=True); shutil.copyfile(cfg_path, backup)
        assert c['weights'] == str(source)
        c['weights'] = str(target); cfg_path.write_text(json.dumps(c, indent=2)+'\n')
    args.protocol.write_text(json.dumps(protocol, indent=2)+'\n')
    (root/'FROZEN_PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print('WEIGHT_FORMAT_CONVERSION_PASS', json.dumps(conversion), flush=True)


if __name__ == '__main__': main()
