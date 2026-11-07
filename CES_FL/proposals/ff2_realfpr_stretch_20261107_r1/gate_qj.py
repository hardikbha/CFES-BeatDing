# Gate Q-J (pre-declared): crop-level q75 4:2:0 emulation must cut L_A's FFHQ-real VAL flag rate from 67.8% to <= 20%.
import json, sys, importlib.util, numpy as np, torch
sys.path.insert(0, '/home/sachin.chaudhary/ces_fl_sources/ff2-lh-ps-v2-20261106/CES_FL/src')
spec = importlib.util.spec_from_file_location('rf', '/home/sachin.chaudhary/ces_fl_sources/ff2-lh-ps-v2-20261106/CES_FL/src/ces_fl_realfpr.py')
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf); ham = rf.ham
torch.set_num_threads(8)
R = '/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A/'
v = json.load(open(R + 'validation_scores_PRIVATE.json')); thr = v['threshold']
ck = torch.load(R + 'best.pt', map_location='cpu', weights_only=True); c = ck['config']
m = ham.Detector(c['hidden'], c['gate_hidden'], c['encoder_policy'], c['head_type']); m.load_state_dict(ck['model']); m.eval()
out = {}
for src, lab in [('ffhq', 0), ('ffhq', 1), ('img', 0)]:
    rows = [r for r in v['rows'] if r['source_ids'].startswith(src + ':') and r['label'] == lab]
    if lab == 1: rows = rows[::max(1, len(rows) // 300)][:300]
    if src == 'img': rows = rows[::max(1, len(rows) // 300)][:300]
    for arm, plan in [('unchanged', None), ('q75_420_s1.158', (75, 2, None)), ('q75_444_s1.158', (75, 0, None)), ('q95_420_s1.158', (95, 2, None))]:
        xs = []
        for r in rows:
            if plan is None: x = ham.load_pixels(r, 224)
            else:
                img = rf.Image.open(r['image']).convert('RGB'); img = rf.jpeg_emulate(img, plan[0], plan[1], 1.158)
                x = torch.from_numpy(np.asarray(img.resize((224, 224), rf.Image.Resampling.BILINEAR), np.float32).copy() / 255.).permute(2, 0, 1)
            xs.append(ham.normalize(x))
        with torch.no_grad(): z = torch.cat([m(torch.stack(xs[i:i + 32]))['logit'] for i in range(0, len(xs), 32)]).numpy()
        out[f'{src}|{lab}|{arm}'] = {'n': len(rows), 'flag_fake_rate': float((z >= thr).mean())}
        print(src, lab, arm, len(rows), round(100 * float((z >= thr).mean()), 1), flush=True)
k = 'ffhq|0|q75_420_s1.158'; out['gate_QJ'] = {'rule': 'FFHQ real flag rate <= 0.20 under crop-level q75 4:2:0', 'value': out[k]['flag_fake_rate'],
                                               'pass': out[k]['flag_fake_rate'] <= 0.20}
json.dump(out, open('/home/sachin.chaudhary/ces_fl_runs/ff2_rfa_20261107/gate_qj.json', 'w'), indent=1); print(out['gate_QJ'])
