# -*- coding: utf-8 -*-
"""NB04_analysis_v2.py
NB02 + NB03 V2에서 저장된 CSV들을 통합 분석.

변경 내역 (vs 구버전):
- V2 파일명(nomadic_v2_*) 대응 + 구버전 파일명 fallback
- 새 컬럼(mean_final_switch_prob, mean_gap_proxy) 방어 처리
- Decision Diagnostics 테이블 추가 (raw/final switch prob, gap proxy)
- Coupling Diagnostics 추가 (gap ↔ switch_rate 상관관계)
- Figure 1: 4-subplot (ΔH / PPL / RepRate / ΔFinalSwitch)
- Figure 2: Decision Decomposition (Nomadic 계열)
- Figure 3: Entropy by Domain (기존 Figure 2 → 번호만 이동)
- Limitation flags v2 (decision-level 경고 추가)
- saturation_check_v2.json 우선 로드, 구버전 fallback
"""

# ============================================================
# STEP 0: 환경 로드
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DRIVE_BASE = '/content/drive/MyDrive/nomadic_llm_results'
with open(os.path.join(DRIVE_BASE, 'latest_run_dir.txt')) as f:
    lines = f.read().strip().split('\n')
    RUN_DIR, MODEL_KEY = lines[0], lines[1]

# saturation check: v2 우선, 구버전 fallback
for sat_file in ['saturation_check_v2.json', 'saturation_check.json']:
    sat_path = os.path.join(RUN_DIR, sat_file)
    if os.path.exists(sat_path):
        with open(sat_path) as f:
            saturation = json.load(f)
        print(f'saturation 로드: {sat_file}')
        break
else:
    saturation = {'sp_diff': 1.0}
    print('⚠️  saturation_check JSON 없음 — 경고 비활성화')

# sp_diff 키 통합 (v2는 'sp_diff_raw', 구버전은 'sp_diff')
sp_diff_val = saturation.get('sp_diff_raw', saturation.get('sp_diff', 1.0))
SATURATION_WARNING = sp_diff_val < 0.1

print(f'RUN_DIR : {RUN_DIR}')
print(f'MODEL   : {MODEL_KEY}')
if SATURATION_WARNING:
    print(f'⚠️  PolicyNet saturation 감지 (switch_sep_gap={sp_diff_val:.3f} < 0.1). 결과 해석 시 주의.')

# ============================================================
# STEP 1: 시나리오 A — 통합 비교 테이블
# ============================================================

dfs = []
# V2 파일명 우선, 구버전 fallback
for fname in ['baseline_scenario_A.csv',
              'nomadic_v2_scenario_A.csv',
              'nomadic_scenario_A.csv']:
    fpath = os.path.join(RUN_DIR, fname)
    if os.path.exists(fpath):
        df_tmp = pd.read_csv(fpath)
        # Nomadic_Full_V2 → Nomadic_Full 로 이름 통일 (비교 테이블 일관성)
        if 'model' in df_tmp.columns:
            df_tmp['model'] = df_tmp['model'].replace({'Nomadic_Full_V2': 'Nomadic_Full'})
        dfs.append(df_tmp)
        print(f'로드: {fname} ({len(df_tmp)}행)')
    else:
        print(f'없음 (스킵): {fname}')

if not dfs:
    raise FileNotFoundError('시나리오 A CSV가 없습니다. NB02, NB03을 먼저 실행하세요.')

df_all = pd.concat(dfs, ignore_index=True)

# ── STEP 1A: 새 컬럼 방어 처리 (NB03 V2 대응) ─────────────────
for col in ['mean_switch_prob', 'mean_final_switch_prob',
            'mean_gap_proxy', 'mean_gap', 'std_entropy']:
    if col not in df_all.columns:
        df_all[col] = np.nan

# mean_gap_proxy / mean_gap 통합 (NB03 V2는 mean_gap으로 저장)
if df_all['mean_gap_proxy'].isna().all() and not df_all['mean_gap'].isna().all():
    df_all['mean_gap_proxy'] = df_all['mean_gap']

# phase 이름 정규화
if 'phase' in df_all.columns:
    df_all['phase'] = df_all['phase'].astype(str).str.strip().str.lower()

MODEL_ORDER = ['Vanilla', 'DynamicTemp', 'StaticLoRA', 'FixedRouting',
               'Nomadic_NoDx', 'Nomadic_Full']

# ── summary 집계 ───────────────────────────────────────────────
summary = df_all.groupby(['model', 'phase']).agg(
    mean_H        =('mean_entropy',           'mean'),
    std_H         =('std_entropy',            'mean'),
    mean_ppl      =('perplexity',             'mean'),
    mean_rep      =('repeat_rate',            'mean'),
    mean_sw       =('switch_rate',            'mean'),
    mean_raw_sw   =('mean_switch_prob',        'mean'),
    mean_final_sw =('mean_final_switch_prob',  'mean'),
    mean_gap      =('mean_gap_proxy',          'mean'),
).round(4).reset_index()

# ── entropy pivot ──────────────────────────────────────────────
pivot = summary.pivot_table(
    index='model', columns='phase', values='mean_H'
).reset_index()
pivot['ΔH'] = pivot.get('transition', np.nan) - pivot.get('stable', np.nan)

ppl_lang = df_all[df_all['domain']=='language'].groupby('model')['perplexity'].mean().reset_index()
ppl_lang.columns = ['model', 'lang_ppl']
pivot = pivot.merge(ppl_lang, on='model', how='left')

rep_lang = df_all[df_all['domain']=='language'].groupby('model')['repeat_rate'].mean().reset_index()
rep_lang.columns = ['model', 'lang_rep']
pivot = pivot.merge(rep_lang, on='model', how='left')

# ── 메인 테이블 출력 ───────────────────────────────────────────
print('='*90)
print(f'§4.5 LLM Experiment — {MODEL_KEY} (3-Expert, PolicyNet V2)')
print('='*90)
print('\n--- Table: ΔH + PPL + RepeatRate ---')
print(f'{"Model":20s} | Stable H | Trans H |   ΔH  | Lang PPL | RepRate')
print('-'*75)
for m in MODEL_ORDER:
    row = pivot[pivot['model']==m]
    if len(row) == 0: continue
    sh  = row['stable'].values[0]     if 'stable'     in row.columns else float('nan')
    th  = row['transition'].values[0] if 'transition' in row.columns else float('nan')
    dh  = row['ΔH'].values[0]
    ppl = row['lang_ppl'].values[0]
    rep = row['lang_rep'].values[0]
    print(f'{m:20s} | {sh:.3f}    | {th:.3f}   | {dh:+.3f} | {ppl:.3f}    | {rep:.3f}')

print('\n--- PAPER.md 반영용 Markdown ---')
print('| Model | Stable H | Trans H | ΔH | Lang PPL | Rep Rate |')
print('|-------|----------|---------|-----|----------|----------|')
for m in MODEL_ORDER:
    row = pivot[pivot['model']==m]
    if len(row) == 0: continue
    sh  = row['stable'].values[0]     if 'stable'     in row.columns else float('nan')
    th  = row['transition'].values[0] if 'transition' in row.columns else float('nan')
    dh  = row['ΔH'].values[0]
    ppl = row['lang_ppl'].values[0]
    rep = row['lang_rep'].values[0]
    print(f'| {m} | {sh:.3f} | {th:.3f} | {dh:+.3f} | {ppl:.3f} | {rep:.3f} |')

pivot.to_csv(os.path.join(RUN_DIR, 'table_scenario_A.csv'), index=False)

# ============================================================
# STEP 1B: Decision-level pivots
# ============================================================
pivot_raw_sw = summary.pivot_table(
    index='model', columns='phase', values='mean_raw_sw'
).reset_index()
pivot_final_sw = summary.pivot_table(
    index='model', columns='phase', values='mean_final_sw'
).reset_index()
pivot_gap = summary.pivot_table(
    index='model', columns='phase', values='mean_gap'
).reset_index()

for df_pv, col_name in [
    (pivot_raw_sw,   'ΔRawSwitch'),
    (pivot_final_sw, 'ΔFinalSwitch'),
    (pivot_gap,      'ΔGap'),
]:
    if 'stable' in df_pv.columns and 'transition' in df_pv.columns:
        df_pv[col_name] = df_pv['transition'] - df_pv['stable']
    else:
        df_pv[col_name] = np.nan

# ── Decision Diagnostics 통합 테이블 ──────────────────────────
decision_table = pivot[['model']].copy()
decision_table = decision_table.merge(
    pivot_raw_sw[['model'] + ([c for c in ['stable','transition','ΔRawSwitch'] if c in pivot_raw_sw.columns])
    ].rename(columns={'stable': 'raw_stable', 'transition': 'raw_transition'}),
    on='model', how='left'
)
decision_table = decision_table.merge(
    pivot_final_sw[['model'] + ([c for c in ['stable','transition','ΔFinalSwitch'] if c in pivot_final_sw.columns])
    ].rename(columns={'stable': 'final_stable', 'transition': 'final_transition'}),
    on='model', how='left'
)
decision_table = decision_table.merge(
    pivot_gap[['model'] + ([c for c in ['stable','transition','ΔGap'] if c in pivot_gap.columns])
    ].rename(columns={'stable': 'gap_stable', 'transition': 'gap_transition'}),
    on='model', how='left'
)

print('\n--- Decision Diagnostics Table ---')
print(f'{"Model":20s} | Raw S | Raw T | ΔRaw | Final S | Final T | ΔFinal | Gap S | Gap T | ΔGap')
print('-'*110)
for m in MODEL_ORDER:
    row = decision_table[decision_table['model']==m]
    if len(row) == 0: continue
    r = row.iloc[0]

    def _f(key, fmt='.3f'):
        v = r.get(key, np.nan)
        try:
            v = float(v)
            return format(v, fmt)
        except (TypeError, ValueError):
            return ' nan'

    print(
        f'{m:20s} | '
        f'{_f("raw_stable")} | {_f("raw_transition")} | {_f("ΔRawSwitch", "+.3f")} | '
        f'{_f("final_stable")} | {_f("final_transition")} | {_f("ΔFinalSwitch", "+.3f")} | '
        f'{_f("gap_stable")} | {_f("gap_transition")} | {_f("ΔGap", "+.3f")}'
    )

print('\n--- PAPER.md / appendix용 Markdown ---')
print('| Model | Raw S | Raw T | ΔRaw | Final S | Final T | ΔFinal | Gap S | Gap T | ΔGap |')
print('|-------|------:|------:|-----:|--------:|--------:|-------:|------:|------:|-----:|')
for m in MODEL_ORDER:
    row = decision_table[decision_table['model']==m]
    if len(row) == 0: continue
    r = row.iloc[0]
    def _fm(key, fmt='.3f'):
        v = r.get(key, np.nan)
        try: return format(float(v), fmt)
        except: return 'nan'
    print(
        f'| {m} | '
        f'{_fm("raw_stable")} | {_fm("raw_transition")} | {_fm("ΔRawSwitch", "+.3f")} | '
        f'{_fm("final_stable")} | {_fm("final_transition")} | {_fm("ΔFinalSwitch", "+.3f")} | '
        f'{_fm("gap_stable")} | {_fm("gap_transition")} | {_fm("ΔGap", "+.3f")} |'
    )

decision_table.to_csv(os.path.join(RUN_DIR, 'table_decision_diagnostics.csv'), index=False)

# ============================================================
# STEP 2: Ablation-style Comparison (ΔH 중심)
# ============================================================
# 기존 "컴포넌트 기여분 분해"에서 이름 변경.
# "기여분 정량화"가 아닌 behavioral progression으로 해석.

print('\n--- Ablation-style Comparison (ΔH 중심) ---')
dh_map = {}
for m in MODEL_ORDER:
    row = pivot[pivot['model']==m]
    if len(row):
        dh_map[m] = float(row['ΔH'].values[0])

steps = [
    ('Baseline (Vanilla)',            'Vanilla',       None,           dh_map.get('Vanilla',       float('nan'))),
    ('+Δx signal (DynamicTemp)',      'DynamicTemp',   'Vanilla',      dh_map.get('DynamicTemp',   float('nan'))),
    ('+LoRA expert (StaticLoRA)',     'StaticLoRA',    'Vanilla',      dh_map.get('StaticLoRA',    float('nan'))),
    ('+Domain routing (FixedRouting)','FixedRouting',  'StaticLoRA',   dh_map.get('FixedRouting',  float('nan'))),
    ('+PolicyNet (NoDx)',             'Nomadic_NoDx',  'FixedRouting', dh_map.get('Nomadic_NoDx',  float('nan'))),
    ('+Δx (Full)',                    'Nomadic_Full',  'Nomadic_NoDx', dh_map.get('Nomadic_Full',  float('nan'))),
]

print(f'{"Component":40s} | ΔH    | Gain vs prev')
print('-'*65)
for label, model, prev_model, dh in steps:
    if prev_model is None:
        print(f'{label:40s} | {dh:+.4f} | —')
    else:
        gain = dh - dh_map.get(prev_model, dh)
        arrow = '↑' if gain > 0 else '↓'
        print(f'{label:40s} | {dh:+.4f} | {arrow} {abs(gain):.4f}')

print('\n--- PAPER.md 반영용 ---')
print('| Component | ΔH | Gain vs prev |')
print('|-----------|-----|------|')
for label, model, prev_model, dh in steps:
    gain_str = '—' if not prev_model else f'{dh - dh_map.get(prev_model, dh):+.4f}'
    print(f'| {label} | {dh:+.4f} | {gain_str} |')

# ============================================================
# STEP 2A: Coupling Diagnostics
# ============================================================
print('\n--- Coupling Diagnostics ---')

def safe_corr(a, b):
    x = pd.Series(a, dtype=float)
    y = pd.Series(b, dtype=float)
    mask = ~(x.isna() | y.isna())
    if mask.sum() < 3: return np.nan
    if x[mask].std() < 1e-8 or y[mask].std() < 1e-8: return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])

coupling_rows = []
for model_name in df_all['model'].dropna().unique():
    sub = df_all[df_all['model']==model_name].copy()
    coupling_rows.append({
        'model':                       model_name,
        'corr_gap_switch_rate':        safe_corr(sub.get('mean_gap_proxy',          pd.Series(dtype=float)),
                                                  sub.get('switch_rate',             pd.Series(dtype=float))),
        'corr_finalSwitch_switch_rate':safe_corr(sub.get('mean_final_switch_prob',  pd.Series(dtype=float)),
                                                  sub.get('switch_rate',             pd.Series(dtype=float))),
        'corr_rawSwitch_switch_rate':  safe_corr(sub.get('mean_switch_prob',         pd.Series(dtype=float)),
                                                  sub.get('switch_rate',             pd.Series(dtype=float))),
    })

df_coupling = pd.DataFrame(coupling_rows).round(4)
print(df_coupling.to_string(index=False))
df_coupling.to_csv(os.path.join(RUN_DIR, 'table_coupling_diagnostics.csv'), index=False)

# ============================================================
# STEP 3: 시나리오 B~G 통합 집계
# ============================================================

def load_scenario(scenario):
    """baseline + nomadic (v2 우선, 구버전 fallback) CSV 통합 로드"""
    dfs_s = []
    for fname in [f'baseline_scenario_{scenario}.csv',
                  f'nomadic_v2_scenario_{scenario}.csv',
                  f'nomadic_scenario_{scenario}.csv']:
        fpath = os.path.join(RUN_DIR, fname)
        if os.path.exists(fpath):
            df_tmp = pd.read_csv(fpath)
            if 'model' in df_tmp.columns:
                df_tmp['model'] = df_tmp['model'].replace({'Nomadic_Full_V2': 'Nomadic_Full'})
            dfs_s.append(df_tmp)
    return pd.concat(dfs_s, ignore_index=True) if dfs_s else pd.DataFrame()

# ── 시나리오 E ────────────────────────────────────────────────
df_E = load_scenario('E')
if not df_E.empty:
    e_summary = df_E.groupby('model').agg(
        mean_lure_spike     =('lure_spike',      'mean'),
        mean_recovery_steps =('recovery_steps',  'mean'),
    ).round(3).reset_index()
    print('\n--- 시나리오 E: 유혹-회복 ---')
    print(e_summary.to_string(index=False))

# ── 시나리오 F ────────────────────────────────────────────────
df_F = load_scenario('F')
if not df_F.empty:
    print('\n--- 시나리오 F: 정보 밀도 ---')
    cols_F = [c for c in ['model','density','mean_dx','corr_dx_conf'] if c in df_F.columns]
    print(df_F[cols_F].to_string(index=False))

# ── 시나리오 G ────────────────────────────────────────────────
df_G = load_scenario('G')
if not df_G.empty:
    g_summary = df_G.groupby('model').agg(
        mean_conv  =('convergence_step', 'mean'),
        mean_decay =('decay_rate',       'mean'),
    ).round(3).reset_index()
    print('\n--- 시나리오 G: 장기 고착도 ---')
    print(g_summary.to_string(index=False))

# ── 시나리오 D ────────────────────────────────────────────────
for fname_D in ['nomadic_v2_scenario_D.csv', 'nomadic_scenario_D.csv']:
    fpath_D = os.path.join(RUN_DIR, fname_D)
    if os.path.exists(fpath_D):
        df_D = pd.read_csv(fpath_D)
        print('\n--- 시나리오 D: 도메인 중첩 (Nomadic_Full) ---')
        cols_D = [c for c in ['prompt','oscillation','dominant_expert','switch_rate','mean_gap']
                  if c in df_D.columns]
        print(df_D[cols_D].to_string(index=False))
        break

# ============================================================
# STEP 4: Figure 1 — main + decision comparison (4-subplot)
# ============================================================
COLORS = {
    'Vanilla':      '#888888',
    'DynamicTemp':  '#E07B54',
    'StaticLoRA':   '#4CAF50',
    'FixedRouting': '#9C27B0',
    'Nomadic_NoDx': '#FF9800',
    'Nomadic_Full': '#2C6FAC',
}

models_present = [m for m in MODEL_ORDER if m in pivot['model'].values]

fig, axes = plt.subplots(1, 4, figsize=(21, 5))
fig.suptitle(f'LLM Experiment — {MODEL_KEY} (Behavior + Decision Diagnostics)', fontsize=12, fontweight='bold')

# 1) ΔH
ax = axes[0]
dh_vals = [float(pivot[pivot['model']==m]['ΔH'].values[0])
           if m in pivot['model'].values else float('nan') for m in models_present]
bars = ax.bar(models_present, dh_vals,
              color=[COLORS.get(m,'gray') for m in models_present], width=0.55)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_title('ΔH = Trans H − Stable H', fontsize=9)
ax.set_ylabel('ΔH'); ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, dh_vals):
    if not np.isnan(v):
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.003, f'{v:+.3f}',
                ha='center', fontsize=7, fontweight='bold')

# 2) Language PPL
ax = axes[1]
ppl_vals = [float(pivot[pivot['model']==m]['lang_ppl'].values[0])
            if m in pivot['model'].values else float('nan') for m in models_present]
bars = ax.bar(models_present, ppl_vals,
              color=[COLORS.get(m,'gray') for m in models_present], width=0.55)
ax.set_title('Language PPL', fontsize=9)
ax.set_ylabel('PPL'); ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, ppl_vals):
    if not np.isnan(v):
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.01, f'{v:.3f}',
                ha='center', fontsize=7)

# 3) Repeat Rate
ax = axes[2]
rep_vals = [float(pivot[pivot['model']==m]['lang_rep'].values[0])
            if m in pivot['model'].values else float('nan') for m in models_present]
bars = ax.bar(models_present, rep_vals,
              color=[COLORS.get(m,'gray') for m in models_present], width=0.55)
ax.set_title('Language Repeat Rate', fontsize=9)
ax.set_ylabel('Repeat Rate'); ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, rep_vals):
    if not np.isnan(v):
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.002, f'{v:.3f}',
                ha='center', fontsize=7)

# 4) ΔFinalSwitch (new)
ax = axes[3]
final_switch_vals = []
for m in models_present:
    row = pivot_final_sw[pivot_final_sw['model']==m]
    if len(row) == 0 or 'ΔFinalSwitch' not in row.columns:
        final_switch_vals.append(float('nan'))
    else:
        final_switch_vals.append(float(row['ΔFinalSwitch'].values[0]))
bars = ax.bar(models_present, final_switch_vals,
              color=[COLORS.get(m,'gray') for m in models_present], width=0.55)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_title('ΔFinalSwitch = Trans − Stable', fontsize=9)
ax.set_ylabel('ΔFinalSwitch'); ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, final_switch_vals):
    if not np.isnan(v):
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.003, f'{v:+.3f}',
                ha='center', fontsize=7)

if SATURATION_WARNING:
    fig.text(0.5, 0.01,
             f'⚠️ Raw PolicyNet saturation warning (switch_sep_gap={sp_diff_val:.3f}) — '
             'interpret raw switch outputs with caution',
             ha='center', fontsize=8, color='red')

plt.tight_layout()
fig.savefig(os.path.join(RUN_DIR, 'fig1_main_comparison_v2.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('fig1_main_comparison_v2.png 저장')

# ============================================================
# STEP 4A: Figure 2 — Decision Decomposition (Nomadic 계열)
# ============================================================
nomadic_models = [m for m in ['Nomadic_NoDx', 'Nomadic_Full']
                  if m in decision_table['model'].values]

if len(nomadic_models) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f'Decision Diagnostics — {MODEL_KEY}', fontsize=12, fontweight='bold')
    bw = 0.35
    x  = np.arange(len(nomadic_models))

    for ax_idx, (col_s, col_t, title) in enumerate([
        ('raw_stable',   'raw_transition',   'Raw Switch Probability'),
        ('final_stable', 'final_transition', 'Final Switch Probability'),
        ('gap_stable',   'gap_transition',   'Gap Proxy'),
    ]):
        ax = axes[ax_idx]
        stable_vals = []
        trans_vals  = []
        for m in nomadic_models:
            row = decision_table[decision_table['model']==m]
            if len(row) == 0:
                stable_vals.append(float('nan'))
                trans_vals.append(float('nan'))
            else:
                r = row.iloc[0]
                stable_vals.append(r.get(col_s, np.nan))
                trans_vals.append(r.get(col_t, np.nan))

        # nan 방어
        stable_vals = [v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0
                       for v in stable_vals]
        trans_vals  = [v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0
                       for v in trans_vals]

        ax.bar(x - bw/2, stable_vals, bw, label='Stable',     color='#378ADD', alpha=0.8)
        ax.bar(x + bw/2, trans_vals,  bw, label='Transition', color='#1D9E75', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(nomadic_models, rotation=20, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, 'fig2_decision_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('fig2_decision_decomposition.png 저장')

# ============================================================
# STEP 4B: Figure 3 — Entropy by Domain (기존 Figure 2 → 번호 이동)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Entropy by Domain — {MODEL_KEY}', fontsize=12, fontweight='bold')

domains = ['math', 'code', 'language']
for col, domain in enumerate(domains):
    ax = axes[col]
    for i, m in enumerate(models_present):
        sub = df_all[(df_all['model']==m) & (df_all['domain']==domain)]
        if sub.empty: continue
        val = sub['mean_entropy'].mean()
        err = sub['mean_entropy'].std()
        ax.bar(i, val, 0.6, yerr=err, capsize=3,
               color=COLORS.get(m,'gray'), alpha=0.8,
               label=m if col==0 else None)
    ax.set_title(f'{domain.capitalize()} Domain', fontsize=10)
    ax.set_xticks(range(len(models_present)))
    ax.set_xticklabels(models_present, rotation=30, fontsize=7)
    ax.set_ylabel('Entropy'); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RUN_DIR, 'fig3_entropy_by_domain.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('fig3_entropy_by_domain.png 저장')

# ============================================================
# STEP 5: Limitation Flags v2
# ============================================================
print('='*60)
print('한계 탐지 리포트 (v2)')
print('='*60)

flags = []

# 1. Raw PolicyNet saturation
if SATURATION_WARNING:
    flags.append(
        f'[WARN] Raw PolicyNet saturation: math/language switch_prob gap = '
        f'{sp_diff_val:.3f} (< 0.1). ΔH 차이가 나타나더라도 PolicyNet 기여가 아닐 수 있음.'
    )

def get_decision_row(model_name):
    row = decision_table[decision_table['model']==model_name]
    return row.iloc[0] if len(row) > 0 else None

# 2. Nomadic_Full decision-level separation
row_full = get_decision_row('Nomadic_Full')
if row_full is not None:
    for key, label, threshold in [
        ('ΔRawSwitch',   'raw switch separation',   0.05),
        ('ΔFinalSwitch', 'final switch separation', 0.05),
        ('ΔGap',         'gap proxy separation',    0.02),
    ]:
        v = row_full.get(key, np.nan)
        try:
            v = float(v)
            if np.isfinite(v) and v < threshold:
                flags.append(f'[WARN] Nomadic_Full {label} weak: {key} = {v:.3f} (< {threshold}).')
        except (TypeError, ValueError):
            pass

    # raw → final 변화량 (evidence fusion 효과)
    try:
        raw_s = float(row_full.get('raw_stable',    np.nan))
        raw_t = float(row_full.get('raw_transition', np.nan))
        fin_s = float(row_full.get('final_stable',   np.nan))
        fin_t = float(row_full.get('final_transition',np.nan))
        if all(np.isfinite(v) for v in [raw_s, raw_t, fin_s, fin_t]):
            delta_adjust = abs(fin_s - raw_s) + abs(fin_t - raw_t)
            if delta_adjust < 0.02:
                flags.append(
                    f'[WARN] Evidence fusion effect minimal: |final-raw| total = {delta_adjust:.3f} (< 0.02). '
                    'Δx/gap/dwell이 PolicyNet output을 거의 보정하지 못하고 있음.'
                )
    except (TypeError, ValueError):
        pass

# 3. Coupling diagnostics
if 'df_coupling' in locals() and len(df_coupling) > 0:
    row_c = df_coupling[df_coupling['model']=='Nomadic_Full']
    if len(row_c) > 0:
        r = row_c.iloc[0]
        for key, label, threshold in [
            ('corr_gap_switch_rate',        'gap↔switch_rate coupling',        0.20),
            ('corr_finalSwitch_switch_rate', 'finalSwitch↔switch_rate coupling', 0.20),
        ]:
            v = r.get(key, np.nan)
            try:
                v = float(v)
                if np.isfinite(v) and v < threshold:
                    flags.append(
                        f'[WARN] Nomadic_Full {label} weak: corr = {v:.3f} (< {threshold}). '
                        '실제 switching behavior와 decision signal의 연결이 약함.'
                    )
            except (TypeError, ValueError):
                pass

# 4. ΔH 비교 경고 (기존 유지)
full_dh = dh_map.get('Nomadic_Full', float('nan'))
dt_dh   = dh_map.get('DynamicTemp',  float('nan'))
fr_dh   = dh_map.get('FixedRouting', float('nan'))

if not (np.isnan(full_dh) or np.isnan(dt_dh)):
    if (full_dh - dt_dh) < 0.05:
        flags.append(
            f'[WARN] Nomadic_Full ΔH ({full_dh:.3f}) vs DynamicTemp ΔH ({dt_dh:.3f}) gap < 0.05. '
            'LoRA+PolicyNet의 추가 기여가 미미할 수 있음.'
        )

if not (np.isnan(full_dh) or np.isnan(fr_dh)):
    if full_dh < fr_dh:
        flags.append(
            f'[WARN] Nomadic_Full ΔH ({full_dh:.3f}) < FixedRouting ΔH ({fr_dh:.3f}). '
            '동적 라우팅이 정적 라우팅보다 entropy 구조가 나쁨. PolicyNet 학습 실패 가능성.'
        )

# 5. 재현성
high_std = df_all.groupby('model')['std_entropy'].mean()
for m, s in high_std.items():
    if s > 0.15:
        flags.append(
            f'[WARN] {m}: mean std_entropy = {s:.3f} (> 0.15). '
            '결과 재현성 낮음. N_RUNS 증가 권장.'
        )

if not flags:
    print('✅ 자동 탐지된 한계 없음')
else:
    for flag in flags:
        print(flag)
        print()

with open(os.path.join(RUN_DIR, 'limitation_flags_v2.txt'), 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(flags) if flags else 'No limitations detected.')

# ============================================================
# STEP 6: 결과 파일 목록 출력
# ============================================================
print('\n' + '='*60)
print(f'✅ NB04 V2 완료')
print(f'모든 결과: {RUN_DIR}')
print('='*60)

expected_files = [
    'table_scenario_A.csv',
    'table_decision_diagnostics.csv',
    'table_coupling_diagnostics.csv',
    'fig1_main_comparison_v2.png',
    'fig2_decision_decomposition.png',
    'fig3_entropy_by_domain.png',
    'limitation_flags_v2.txt',
]
for fname in expected_files:
    exists = os.path.exists(os.path.join(RUN_DIR, fname))
    print(f'  {"✅" if exists else "❌"} {fname}')

print('\n전체 RUN_DIR 파일 목록:')
for fname in sorted(os.listdir(RUN_DIR)):
    print(f'  {fname}')
