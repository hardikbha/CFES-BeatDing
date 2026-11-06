# Aggregate the counterfactual re-crop scores (L_A, frozen VAL threshold). Writes recrop_summary.json (aggregates only).
import json, numpy as np, pandas as pd
OUT = '/home/sachin.chaudhary/ces_fl_runs/ff2_source_diag_20261106'
THR = float(json.load(open('/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A/validation_scores_PRIVATE.json'))['threshold'])
ARMS = ['A0_stored', 'A1_v1_recrop', 'A2_v2_profile_from_master', 'A3_fairface_geometry', 'A4_fairface_geometry_jpeg75_420',
        'A5_v1_after_master_jpeg75_420', 'A6_fairface_geometry_2x_upsampled_jpeg75_420']
rng = np.random.default_rng(0)
S = {'threshold_logit': THR}
for tag in ['reals', 'fakes']:
    d = pd.read_pickle(f'{OUT}/_recrop_scores_{tag}_PRIVATE.pkl')
    if 'err' in d: d = d[d.err.isna()]
    kind = {}
    kind['n'] = int(len(d)); kind['stored_vs_file_max_abs_dlogit'] = float((d.A0_stored - d.A0_stored_logit_file).abs().max())
    kind['A1_byte_identical_frac'] = float(d.A1_bytes_identical.mean())
    kind['A0_A1_pearson_logit'] = float(np.corrcoef(d.A0_stored, d.A1_v1_recrop)[0, 1])
    kind['A0_A1_flag_agreement'] = float(((d.A0_stored >= THR) == (d.A1_v1_recrop >= THR)).mean())
    for a in ARMS:
        f = (d[a] >= THR).values.astype(float)
        bs = [f[rng.integers(0, len(f), len(f))].mean() for _ in range(2000)]
        kind[a] = dict(flag_fake_rate=float(f.mean()), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                       median_logit=float(d[a].median()),
                       median_align_scale=float(d[a + '_scale'].median()) if a + '_scale' in d else None,
                       median_log_laplacian=float(d[a + '_log_lap'].median()), median_hf_y=float(d[a + '_hf_y'].median()),
                       median_noise=float(d[a + '_noise'].median()))
    # paired change A0 -> A4
    f0 = (d.A0_stored >= THR).values; f4 = (d.A4_fairface_geometry_jpeg75_420 >= THR).values
    kind['paired_A0_to_A4'] = dict(flag_to_pass=int((f0 & ~f4).sum()), pass_to_flag=int((~f0 & f4).sum()), both_flag=int((f0 & f4).sum()))
    if tag == 'fakes':
        kind['per_method_flag_rate'] = {a: d.groupby('method')[a].apply(lambda s: round(float((s >= THR).mean()), 3)).to_dict()
                                        for a in ['A0_stored', 'A3_fairface_geometry', 'A4_fairface_geometry_jpeg75_420', 'A5_v1_after_master_jpeg75_420']}
        kind['n_methods'] = int(d.method.nunique())
    S[tag] = kind
json.dump(S, open(OUT + '/recrop_summary.json', 'w'), indent=1)
for tag in ['reals', 'fakes']:
    k = S[tag]; print(tag, 'n', k['n'], 'A1 identical', k['A1_byte_identical_frac'], 'A0/A1 r', round(k['A0_A1_pearson_logit'], 3),
                      'agree', round(k['A0_A1_flag_agreement'], 3), 'dlogit', round(k['stored_vs_file_max_abs_dlogit'], 3))
    for a in ARMS:
        x = k[a]; print('  ', a.ljust(46), '%.3f [%.3f, %.3f]' % (x['flag_fake_rate'], *x['ci95']), 'med logit %.2f' % x['median_logit'],
                        'scale', x['median_align_scale'] and round(x['median_align_scale'], 3), 'loglap %.2f hf %.2f noise %.2f' % (x['median_log_laplacian'], x['median_hf_y'], x['median_noise']))
    print('  paired', k['paired_A0_to_A4'])
    if 'per_method_flag_rate' in k: print(pd.DataFrame(k['per_method_flag_rate']).to_string())
