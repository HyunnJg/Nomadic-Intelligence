# -*- coding: utf-8 -*-
"""NB03_nomadic_eval_v2.py
Nomadic Full V2 — B방향 inference
- build_meta_signals_v2: NB01과 완전 동일한 함수 사용 (train/inference 정합)
- gap proxy: 1-step confidence score 비교 (훈련 교사와 동일한 방식)
- routing_decision_v2: PolicyNet 중심 score fusion
- PolicyNet V2 (12-dim meta) 로드
"""

# ============================================================
# STEP 0: 환경 로드
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, json, math

DRIVE_BASE = '/content/drive/MyDrive/nomadic_llm_results'
with open(os.path.join(DRIVE_BASE, 'latest_run_dir.txt')) as f:
    lines     = f.read().strip().split('\n')
    RUN_DIR   = lines[0]
    MODEL_KEY = lines[1]

with open(os.path.join(RUN_DIR, 'run_config.json')) as f:
    cfg = json.load(f)
MODEL_PATH = cfg['MODEL_PATH']
HIDDEN_DIM = cfg['HIDDEN_DIM']

with open(os.path.join(RUN_DIR, 'meta_config.json')) as f:
    mcfg = json.load(f)
META_DIM    = mcfg['META_DIM']
NUM_EXPERTS = mcfg['NUM_EXPERTS']

with open(os.path.join(RUN_DIR, 'prompts.json'), encoding='utf-8') as f:
    P = json.load(f)
EVAL_PROMPTS          = P['EVAL_PROMPTS']
SCENARIO_B_SEQUENCE   = P['SCENARIO_B_SEQUENCE']
SCENARIO_B_SHIFT_POINT= P['SCENARIO_B_SHIFT_POINT']
SCENARIO_C_CLEAN      = P['SCENARIO_C_CLEAN']
SCENARIO_C_NOISY      = P['SCENARIO_C_NOISY']
SCENARIO_D_HYBRID     = P['SCENARIO_D_HYBRID']
SCENARIO_E_CASES      = P['SCENARIO_E_CASES']
SCENARIO_F_DENSE      = P['SCENARIO_F_DENSE']
SCENARIO_F_SPARSE     = P['SCENARIO_F_SPARSE']
SCENARIO_G_PROMPTS    = P['SCENARIO_G_PROMPTS']

with open(os.path.join(RUN_DIR, 'expert_paths.json')) as f:
    expert_paths = json.load(f)

print(f'RUN_DIR  : {RUN_DIR}')
print(f'MODEL_KEY: {MODEL_KEY}')
print(f'META_DIM : {META_DIM}')

# ============================================================
# STEP 1: 패키지 + 모델 로드
# ============================================================
import subprocess
subprocess.run(['pip', 'install', '-q', '-U',
                'transformers', 'accelerate', 'bitsandbytes', 'peft'])

import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from collections import deque, Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    device_map='auto', trust_remote_code=True)
base_model.eval()
print('✅ 모델 로드 완료')

# ============================================================
# STEP 2: 컴포넌트 정의
# ============================================================

class HybridDeltaTracker:
    def __init__(self, alpha=0.8, beta=0.85, tau_min=2.0, tau_max=8.0,
                 tau_var_scale=6.0, tau_window=8):
        self.alpha=alpha; self.beta=beta
        self.tau_min=tau_min; self.tau_max=tau_max
        self.tau_var_scale=tau_var_scale; self.tau_window=tau_window
        self.reset()

    def reset(self):
        self.prev_x=None; self.ema_err=0.0; self.baseline_err=0.0
        self.recent_de=deque(maxlen=self.tau_window); self.history=[]

    def compute(self, current_x, current_err):
        if self.prev_x is None:
            delta_env = 0.0
        else:
            cos = F.cosine_similarity(
                current_x.float().flatten(), self.prev_x.float().flatten(), dim=0)
            delta_env = float((1.0 - cos).clamp(0).item())
        self.prev_x = current_x.detach().clone()
        self.recent_de.append(delta_env)
        self.ema_err      = self.alpha * self.ema_err + (1 - self.alpha) * current_err
        self.baseline_err = self.beta  * self.baseline_err + (1 - self.beta) * current_err
        delta_err    = max(0.0, self.ema_err - self.baseline_err)
        delta_hybrid = float(torch.tanh(torch.tensor(delta_env + delta_err)).item())
        sigma2       = float(np.var(self.recent_de)) if len(self.recent_de) >= 2 else 0.0
        tau_dyn      = self.tau_min + (self.tau_max - self.tau_min) / (
            1.0 + self.tau_var_scale * sigma2)
        rec = dict(delta_env=delta_env, delta_err=delta_err,
                   delta_hybrid=delta_hybrid, sigma2=sigma2, tau_dyn=tau_dyn)
        self.history.append(rec)
        return rec


class NomadicPolicyNetV2(nn.Module):
    """NB01과 동일한 구조 — 가중치 호환 필수"""
    def __init__(self, hidden_dim, meta_dim, num_experts=3, policy_hidden=128):
        super().__init__()
        self.num_experts = num_experts
        self.proj   = nn.Linear(hidden_dim, policy_hidden)
        self.shared = nn.Sequential(
            nn.Linear(policy_hidden + meta_dim, policy_hidden), nn.ReLU(),
            nn.Linear(policy_hidden, policy_hidden),            nn.ReLU(),
        )
        self.stay_switch_head = nn.Linear(policy_hidden, 2)
        self.target_head      = nn.Linear(policy_hidden, num_experts)
        self.mode_head        = nn.Linear(policy_hidden, 2)

    def forward(self, hidden, meta_signals):
        if hidden.dim() == 3:
            hidden = hidden.squeeze(1)
        h = F.relu(self.proj(hidden.float()))
        x = torch.cat([h, meta_signals.float()], dim=-1)
        z = self.shared(x)
        return self.stay_switch_head(z), self.target_head(z), self.mode_head(z)


def build_meta_signals_v2(meta: dict, num_experts: int = 3,
                           device=None) -> torch.Tensor:
    """
    NB01과 완전 동일한 함수 — 이 함수를 공유하는 것이 B방향의 핵심.
    12-dim: [9개 연속값] + [3개 expert one-hot]
    """
    onehot = [0.0] * num_experts
    onehot[meta['current_expert_id']] = 1.0
    vec = [
        meta['delta_env'],
        meta['delta_err'],
        meta['delta_hybrid'],
        math.tanh(meta['sigma2'] * 10.0),
        math.tanh((meta['tau_dyn'] - 5.0) / 5.0),
        math.tanh(meta['dwell_ratio']),
        meta['current_conf'],
        meta['logit_margin'],
        math.tanh(meta['gap']),
    ] + onehot
    t = torch.tensor([vec], dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def compute_expert_confidence_score(logits: torch.Tensor) -> float:
    """
    NB01과 완전 동일한 함수 — B방향 정합의 핵심.
    gap = best_score - current_score
    """
    probs     = F.softmax(logits, dim=-1)
    top2      = probs.topk(2, dim=-1).values
    top1_prob = float(top2[0, 0].item())
    margin    = float((top2[0, 0] - top2[0, 1]).item())
    return 0.7 * top1_prob + 0.3 * margin


def topk_entropy(logits, k=50):
    probs = F.softmax(logits, dim=-1)
    topk  = probs.topk(k).values; topk = topk / topk.sum()
    return float(-(topk * (topk + 1e-10).log()).sum().item())


def repeat_rate(tokens):
    if len(tokens) < 3: return 0.0
    ngrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


EXPERT_KEYS  = ['math', 'code', 'language']
MAX_NEW_TOKENS = 40
T_STABLE     = 0.35; T_TRANSITION = 0.90
N_RUNS       = 3
DOMAIN_PHASE = {'math': 'stable', 'code': 'stable', 'language': 'transition'}


def _load_expert(key):
    m = PeftModel.from_pretrained(base_model, expert_paths[key])
    m.eval(); return m

def _unload_expert(m):
    del m; torch.cuda.empty_cache()

def _next_tok(logits, temp):
    probs = F.softmax(logits / max(temp, 1e-4), dim=-1)
    tok   = torch.multinomial(probs, 1)
    lp    = F.log_softmax(logits / max(temp, 1e-4), dim=-1).gather(1, tok).item()
    return tok, lp

def base_result(text, entropies, log_probs, expert_trace=None, dx_trace=None):
    ppl     = float(np.exp(-np.mean(log_probs))) if log_probs else float('nan')
    trace   = expert_trace or []
    switches = sum(1 for i in range(1, len(trace)) if trace[i] != trace[i-1])
    tokens   = tokenizer.encode(text)
    return {
        'text':         text,
        'mean_entropy': float(np.mean(entropies)) if entropies else float('nan'),
        'perplexity':   ppl,
        'switch_rate':  switches / max(1, len(trace)),
        'repeat_rate':  repeat_rate(tokens),
        'mean_dx':      float(np.mean(dx_trace)) if dx_trace else 0.0,
        'expert_trace': trace,
        'dx_trace':     dx_trace or [],
    }


# ── PolicyNet V2 로드 ─────────────────────────────────────────
policy_net = NomadicPolicyNetV2(
    hidden_dim=HIDDEN_DIM, meta_dim=META_DIM, num_experts=NUM_EXPERTS)
policy_net.load_state_dict(
    torch.load(os.path.join(RUN_DIR, 'policy_net.pt'),
               map_location=base_model.device))
policy_net = policy_net.to(base_model.device)
policy_net.eval()
print(f'✅ PolicyNet V2 로드 완료 | params: '
      f'{sum(p.numel() for p in policy_net.parameters()):,}')

# ============================================================
# STEP 3: inference 유틸 — B방향 gap proxy + routing decision
# ============================================================

def estimate_gap_proxy_1step(
    ids: torch.Tensor,
    current_key: str,
    expert_keys=EXPERT_KEYS,
) -> tuple:
    """
    1-step confidence score를 모든 expert에서 계산.
    훈련 교사(compute_gap_and_label)와 완전히 동일한 방식.

    반환: (gap, best_key, score_dict)
    """
    score_dict = {}
    for ek in expert_keys:
        m = _load_expert(ek)
        with torch.no_grad():
            out = m(ids)
        logits = out.logits[:, -1, :]
        score_dict[ek] = compute_expert_confidence_score(logits)
        _unload_expert(m)

    current_score = score_dict[current_key]
    best_key      = max(score_dict, key=score_dict.get)
    best_score    = score_dict[best_key]
    gap           = max(0.0, best_score - current_score)
    return gap, best_key, score_dict


def sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def logit_scalar(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def routing_decision_v2(
    policy_switch_prob: float,
    delta_hybrid: float,
    dwell_count: int,
    tau_dyn: float,
    gap: float,
) -> float:
    """
    최종 전환 확률.
    PolicyNet이 결정의 중심이고, Δx / dwell / gap이 evidence를 제공.

    가중치:
      PolicyNet  1.8  — 결정의 중심
      gap        1.2  — 1-step counterfactual utility (훈련과 동일한 신호)
      delta      0.6  — 환경 변화 evidence
      dwell      0.5  — commitment pressure

    이 가중치 구조는 논문 §2.3의 Φ 정의와 대응:
      Φ = Δx(env) + Δx(err) + L_task + gap_t
    여기서 PolicyNet이 Φ를 통합해 최종 결정을 내림.
    """
    dwell_pressure = max(0.0, dwell_count - tau_dyn) / max(tau_dyn, 1e-6)
    score = (
        1.8 * logit_scalar(policy_switch_prob) +
        1.2 * gap +
        0.6 * delta_hybrid +
        0.5 * dwell_pressure
    )
    return sigmoid_scalar(score)


class DwellTracker:
    def __init__(self):
        self.current_expert = None; self.dwell_count = 0

    def update(self, expert_key):
        if expert_key == self.current_expert:
            self.dwell_count += 1
        else:
            self.current_expert = expert_key; self.dwell_count = 1
        return self.dwell_count


# ============================================================
# STEP 4: Nomadic_Full V2 생성 함수
# ============================================================

def generate_nomadic_full_v2(prompt: str) -> dict:
    """
    B방향 Nomadic Full:
      Δx    = evidence (HybridDeltaTracker)
      gap   = 1-step counterfactual utility (훈련과 동일한 방식)
      dwell = commitment pressure (DwellTracker)
      PolicyNet V2 = 최종 결정 (routing_decision_v2 통해)
    """
    tracker = HybridDeltaTracker()
    dwell   = DwellTracker()
    ids     = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)

    ents, lps, expert_trace, dxs = [], [], [], []
    switch_probs_log, final_switch_probs_log, gap_trace = [], [], []

    current_key   = 'math'
    current_model = _load_expert(current_key)

    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = current_model(ids, output_hidden_states=True)

        logits = out.logits[:, -1, :]
        hidden = out.hidden_states[-1][:, -1, :]

        probs    = F.softmax(logits, dim=-1)
        conf     = float(probs.max().item())
        margin   = float((probs.topk(2, dim=-1).values[0, 0] -
                          probs.topk(2, dim=-1).values[0, 1]).item())
        err      = 1.0 - conf
        rec      = tracker.compute(hidden, err)
        dwell_count = dwell.update(current_key)
        dwell_ratio = dwell_count / max(rec['tau_dyn'], 1e-6)

        # 1-step gap proxy — NB01 교사와 동일한 방식 (B방향 핵심)
        gap, best_key, score_dict = estimate_gap_proxy_1step(
            ids=ids, current_key=current_key, expert_keys=EXPERT_KEYS)

        meta = {
            'delta_env':         rec['delta_env'],
            'delta_err':         rec['delta_err'],
            'delta_hybrid':      rec['delta_hybrid'],
            'sigma2':            rec['sigma2'],
            'tau_dyn':           rec['tau_dyn'],
            'dwell_ratio':       dwell_ratio,
            'current_conf':      conf,
            'logit_margin':      margin,
            'gap':               gap,
            'current_expert_id': EXPERT_KEYS.index(current_key),
        }

        meta_tensor = build_meta_signals_v2(
            meta, num_experts=NUM_EXPERTS, device=base_model.device)

        with torch.no_grad():
            sw_logits, tg_logits, md_logits = policy_net(
                hidden.unsqueeze(0), meta_tensor)

        sw_probs    = F.softmax(sw_logits, dim=-1)
        switch_prob = float(sw_probs[0, 1].item())
        tgt_key     = EXPERT_KEYS[int(F.softmax(tg_logits, dim=-1).argmax(-1).item())]
        hard_mode   = int(F.softmax(md_logits, dim=-1).argmax(-1).item()) == 1

        # routing_decision_v2: PolicyNet 중심 score fusion
        final_switch_prob = routing_decision_v2(
            policy_switch_prob=switch_prob,
            delta_hybrid=rec['delta_hybrid'],
            dwell_count=dwell_count,
            tau_dyn=rec['tau_dyn'],
            gap=gap,
        )

        # target expert 선택:
        # gap이 충분히 크면 proxy best_key 우선, 그 외엔 PolicyNet tgt_key
        next_candidate = best_key if gap > 0.05 else tgt_key
        next_key       = next_candidate if final_switch_prob >= 0.5 else current_key

        if next_key != current_key:
            _unload_expert(current_model)
            current_model = _load_expert(next_key)
            current_key   = next_key
            dwell.update(next_key)

        # 생성 온도: Δx 기반 + hard mode 조정
        temp = T_STABLE + (T_TRANSITION - T_STABLE) * rec['delta_hybrid']
        if hard_mode and rec['delta_hybrid'] < 0.2 and gap < 0.05:
            temp = 0.10

        ents.append(topk_entropy(logits / max(temp, 1e-4)))
        tok, lp = _next_tok(logits, temp)
        lps.append(lp)
        expert_trace.append(current_key)
        dxs.append(rec['delta_hybrid'])
        switch_probs_log.append(switch_prob)
        final_switch_probs_log.append(final_switch_prob)
        gap_trace.append(gap)

        ids = torch.cat([ids, tok], dim=-1)
        if tok.item() == tokenizer.eos_token_id:
            break

    _unload_expert(current_model)
    text   = tokenizer.decode(ids[0], skip_special_tokens=True)
    result = base_result(text, ents, lps, expert_trace=expert_trace, dx_trace=dxs)
    result['mean_switch_prob']       = float(np.mean(switch_probs_log))       if switch_probs_log       else 0.0
    result['mean_final_switch_prob'] = float(np.mean(final_switch_probs_log)) if final_switch_probs_log else 0.0
    result['mean_gap']               = float(np.mean(gap_trace))               if gap_trace               else 0.0
    result['switch_prob_trace']      = switch_probs_log
    result['final_switch_prob_trace']= final_switch_probs_log
    result['gap_trace']              = gap_trace
    return result


print('✅ Nomadic_Full V2 생성 함수 정의 완료')

# ============================================================
# STEP 5: Switch saturation 사전 체크
# ============================================================
print('PolicyNet V2 switch saturation 체크...')
saturation_data = {}

for domain in ['math', 'language']:
    raw_probs, final_probs = [], []
    for prompt in EVAL_PROMPTS[domain][:3]:
        r = generate_nomadic_full_v2(prompt)
        raw_probs.extend(r['switch_prob_trace'])
        final_probs.extend(r['final_switch_prob_trace'])

    mean_raw   = float(np.mean(raw_probs))   if raw_probs   else 0.0
    mean_final = float(np.mean(final_probs)) if final_probs else 0.0
    saturation_data[domain] = {'raw': mean_raw, 'final': mean_final}

    flag_raw   = ' ⚠️ SATURATED' if mean_raw   > 0.90 or mean_raw   < 0.10 else ' ✅'
    flag_final = ' ⚠️ SATURATED' if mean_final > 0.90 or mean_final < 0.10 else ' ✅'
    print(f'  {domain:10s}: raw={mean_raw:.3f}{flag_raw} | final={mean_final:.3f}{flag_final}')

sp_diff_raw = abs(saturation_data['math']['raw'] - saturation_data['language']['raw'])
print(f'\n  math vs language |Δ switch_prob| = {sp_diff_raw:.3f}')
if sp_diff_raw < 0.15:
    print('  ⚠️  switch_sep_gap < 0.15 — PolicyNet이 도메인을 충분히 구분하지 못할 수 있음.')
    print('     switch_cost를 낮추거나 epochs를 늘려 재학습을 고려하세요.')
else:
    print('  ✅ PolicyNet이 stable/transition을 구분하고 있음.')

with open(os.path.join(RUN_DIR, 'saturation_check_v2.json'), 'w') as f:
    json.dump({'saturation': saturation_data, 'sp_diff_raw': sp_diff_raw}, f, indent=2)

# ============================================================
# STEP 6: 시나리오 A — 도메인별 기본 평가
# ============================================================
print('시나리오 A: Nomadic_Full V2 도메인 평가...')
results_A = []

for domain in ['math', 'code', 'language']:
    prompts = EVAL_PROMPTS[domain]
    for prompt in prompts:
        run_ents, run_ppls, run_reps, run_sws, run_dxs = [], [], [], [], []
        run_sp, run_fsp, run_gap = [], [], []

        for run in range(N_RUNS):
            r = generate_nomadic_full_v2(prompt)
            run_ents.append(r['mean_entropy'])
            run_ppls.append(r['perplexity'])
            run_reps.append(r['repeat_rate'])
            run_sws.append(r['switch_rate'])
            run_dxs.append(r['mean_dx'])
            run_sp.append(r['mean_switch_prob'])
            run_fsp.append(r['mean_final_switch_prob'])
            run_gap.append(r['mean_gap'])

        results_A.append({
            'model':                 'Nomadic_Full_V2',
            'domain':                domain,
            'phase':                 DOMAIN_PHASE[domain],
            'prompt':                prompt[:50],
            'mean_entropy':          float(np.mean(run_ents)),
            'std_entropy':           float(np.std(run_ents)),
            'perplexity':            float(np.mean(run_ppls)),
            'repeat_rate':           float(np.mean(run_reps)),
            'switch_rate':           float(np.mean(run_sws)),
            'mean_dx':               float(np.mean(run_dxs)),
            'mean_switch_prob':      float(np.mean(run_sp)),
            'mean_final_switch_prob':float(np.mean(run_fsp)),
            'mean_gap':              float(np.mean(run_gap)),
        })
    print(f'  {domain}: 완료')

df_A = pd.DataFrame(results_A)
df_A.to_csv(os.path.join(RUN_DIR, 'nomadic_v2_scenario_A.csv'),
            index=False, encoding='utf-8-sig')
print('✅ 시나리오 A 완료')

# 논문용 집계 테이블 출력
print('\n─── 시나리오 A 집계 (논문용) ───')
agg = df_A.groupby('phase').agg(
    stable_H=('mean_entropy',  lambda x: x[df_A.loc[x.index,'phase']=='stable'].mean()  if 'stable'  in df_A.loc[x.index,'phase'].values else float('nan')),
    trans_H =('mean_entropy',  lambda x: x[df_A.loc[x.index,'phase']=='transition'].mean() if 'transition' in df_A.loc[x.index,'phase'].values else float('nan')),
).reset_index()

stable_H = df_A[df_A['phase']=='stable']['mean_entropy'].mean()
trans_H  = df_A[df_A['phase']=='transition']['mean_entropy'].mean()
lang_ppl = df_A[df_A['domain']=='language']['perplexity'].mean()
lang_rep = df_A[df_A['domain']=='language']['repeat_rate'].mean()
dH       = trans_H - stable_H

print(f'  Stable H   : {stable_H:.4f}')
print(f'  Trans H    : {trans_H:.4f}')
print(f'  ΔH         : {dH:.4f}')
print(f'  Language PPL: {lang_ppl:.4f}')
print(f'  Language Rep: {lang_rep:.4f}')

# ============================================================
# STEP 7: 시나리오 B~G
# ============================================================

# ── 시나리오 B: 도메인 급변 Δx ───────────────────────────────
print('시나리오 B: 도메인 급변 Δx...')
tracker_b = HybridDeltaTracker(); tracker_b.reset()
dx_trace_b = []
for prompt in SCENARIO_B_SEQUENCE:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
    with torch.no_grad():
        out = base_model(ids, output_hidden_states=True)
    hidden = out.hidden_states[-1][:, -1, :]
    err    = float(1.0 - F.softmax(out.logits[:, -1, :], dim=-1).max().item())
    rec    = tracker_b.compute(hidden, err)
    dx_trace_b.append(rec['delta_hybrid'])

shift_pt = SCENARIO_B_SHIFT_POINT
result_B = {
    'model':        'Nomadic_Full_V2',
    'pre_mean_dx':  float(np.mean(dx_trace_b[:shift_pt])),
    'post_mean_dx': float(np.mean(dx_trace_b[shift_pt:])),
    'delta_dx':     float(np.mean(dx_trace_b[shift_pt:])) - float(np.mean(dx_trace_b[:shift_pt])),
}
print(f'  pre={result_B["pre_mean_dx"]:.4f} → post={result_B["post_mean_dx"]:.4f}  Δ={result_B["delta_dx"]:+.4f}')
pd.DataFrame([result_B]).to_csv(os.path.join(RUN_DIR, 'nomadic_v2_scenario_B.csv'), index=False)

# ── 시나리오 C: 노이즈 면역력 ────────────────────────────────
print('시나리오 C: 노이즈 면역력...')
results_C = []
for condition, prompts in [('clean', SCENARIO_C_CLEAN), ('noisy', SCENARIO_C_NOISY)]:
    ents, ppls = [], []
    for prompt in prompts:
        r = generate_nomadic_full_v2(prompt)
        ents.append(r['mean_entropy']); ppls.append(r['perplexity'])
    results_C.append({'model': 'Nomadic_Full_V2', 'condition': condition,
                      'mean_entropy': float(np.mean(ents)),
                      'perplexity':   float(np.mean(ppls))})
pd.DataFrame(results_C).to_csv(os.path.join(RUN_DIR, 'nomadic_v2_scenario_C.csv'), index=False)

# ── 시나리오 D: 도메인 중첩 ──────────────────────────────────
print('시나리오 D: 도메인 중첩...')
results_D = []
for prompt in SCENARIO_D_HYBRID:
    r = generate_nomadic_full_v2(prompt)
    tr = r['expert_trace']
    osc = sum(1 for k in range(2, len(tr)) if tr[k] == tr[k-2] and tr[k] != tr[k-1])
    osc_rate = osc / max(1, len(tr) - 2)
    dom_exp  = Counter(tr).most_common(1)[0][0] if tr else 'n/a'
    results_D.append({'model': 'Nomadic_Full_V2', 'prompt': prompt[:50],
                      'oscillation': osc_rate, 'dominant_expert': dom_exp,
                      'switch_rate': r['switch_rate'], 'mean_dx': r['mean_dx'],
                      'mean_gap': r['mean_gap']})
pd.DataFrame(results_D).to_csv(os.path.join(RUN_DIR, 'nomadic_v2_scenario_D.csv'),
                                index=False, encoding='utf-8-sig')

# ── 시나리오 E: 유혹-회복 ────────────────────────────────────
print('시나리오 E: 유혹-회복...')

def run_lure_scenario_v2(cases):
    rows = []
    for math_prefix, lure, math_cont in cases:
        tracker = HybridDeltaTracker()
        baseline_dxs = []
        ids = tokenizer(math_prefix, return_tensors='pt').input_ids.to(base_model.device)
        for _ in range(8):
            with torch.no_grad():
                out = base_model(ids, output_hidden_states=True)
            h   = out.hidden_states[-1][:, -1, :]
            e   = float(1.0 - F.softmax(out.logits[:, -1, :], dim=-1).max().item())
            rec = tracker.compute(h, e)
            baseline_dxs.append(rec['delta_hybrid'])
            ids = torch.cat([ids, out.logits[:, -1, :].argmax(-1, keepdim=True)], dim=-1)
        baseline_mean = float(np.mean(baseline_dxs))

        lure_ids = tokenizer(math_prefix + lure, return_tensors='pt').input_ids.to(base_model.device)
        with torch.no_grad():
            out = base_model(lure_ids, output_hidden_states=True)
        h        = out.hidden_states[-1][:, -1, :]
        e        = float(1.0 - F.softmax(out.logits[:, -1, :], dim=-1).max().item())
        lure_rec = tracker.compute(h, e)
        lure_spike = lure_rec['delta_hybrid'] - baseline_mean

        rec_ids = tokenizer(math_prefix + lure + math_cont,
                            return_tensors='pt').input_ids.to(base_model.device)
        recovery_steps = 0
        for step in range(20):
            with torch.no_grad():
                out = base_model(rec_ids, output_hidden_states=True)
            h   = out.hidden_states[-1][:, -1, :]
            e   = float(1.0 - F.softmax(out.logits[:, -1, :], dim=-1).max().item())
            rec = tracker.compute(h, e)
            recovery_steps += 1
            if rec['delta_hybrid'] <= baseline_mean + 0.05: break
            tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            rec_ids = torch.cat([rec_ids, tok], dim=-1)
        rows.append({'model': 'Nomadic_Full_V2',
                     'baseline_dx': baseline_mean,
                     'lure_spike': lure_spike,
                     'recovery_steps': recovery_steps})
    return rows

results_E = run_lure_scenario_v2(SCENARIO_E_CASES)
print(f'  avg recovery_steps={np.mean([r["recovery_steps"] for r in results_E]):.1f}')
pd.DataFrame(results_E).to_csv(os.path.join(RUN_DIR, 'nomadic_v2_scenario_E.csv'),
                                index=False, encoding='utf-8-sig')

# ── 시나리오 F: 정보 밀도 ────────────────────────────────────
print('시나리오 F: 정보 밀도...')
results_F = []
for density, prompts in [('dense', SCENARIO_F_DENSE), ('sparse', SCENARIO_F_SPARSE)]:
    tracker = HybridDeltaTracker(); tracker.reset()
    dx_list, conf_list = [], []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
        with torch.no_grad():
            out = base_model(ids, output_hidden_states=True)
        h   = out.hidden_states[-1][:, -1, :]
        e   = float(1.0 - F.softmax(out.logits[:, -1, :], dim=-1).max().item())
        rec = tracker.compute(h, e)
        dx_list.append(rec['delta_hybrid']); conf_list.append(e)
    corr = float(np.corrcoef(dx_list, conf_list)[0, 1]) if len(dx_list) > 1 else 0.0
    results_F.append({'model': 'Nomadic_Full_V2', 'density': density,
                      'mean_dx': float(np.mean(dx_list)),
                      'mean_conf': float(np.mean(conf_list)),
                      'corr_dx_conf': corr})
pd.DataFrame(results_F).to_csv(os.path.join(RUN_DIR, 'nomadic_v2_scenario_F.csv'), index=False)

# ── 시나리오 G: 장기 고착도 ──────────────────────────────────
print('시나리오 G: 장기 고착도...')
results_G = []
LONG_STEPS = 60; CONV_THRESHOLD = 0.02

for domain, prompt in SCENARIO_G_PROMPTS.items():
    tracker = HybridDeltaTracker()
    expert  = _load_expert(domain)
    ids     = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
    phi_trace = []; conv_step = LONG_STEPS

    for step in range(LONG_STEPS):
        with torch.no_grad():
            out = expert(ids, output_hidden_states=True)
        h   = out.hidden_states[-1][:, -1, :]
        e   = float(1.0 - F.softmax(out.logits[:, -1, :], dim=-1).max().item())
        rec = tracker.compute(h, e)
        phi_trace.append(rec['delta_hybrid'])
        if rec['delta_hybrid'] <= CONV_THRESHOLD and conv_step == LONG_STEPS:
            conv_step = step
        tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, tok], dim=-1)
        if tok.item() == tokenizer.eos_token_id: break

    _unload_expert(expert)
    phi_arr = np.array(phi_trace)
    slope   = float(np.polyfit(np.arange(len(phi_arr)), phi_arr, 1)[0]) if len(phi_arr) > 5 else 0.0
    results_G.append({
        'model':           'Nomadic_Full_V2',
        'domain':          domain,
        'convergence_step': conv_step,
        'final_phi_mean':  float(np.mean(phi_trace[-5:])) if len(phi_trace) >= 5 else float('nan'),
        'decay_rate':      slope,
    })

pd.DataFrame(results_G).to_csv(os.path.join(RUN_DIR, 'nomadic_v2_scenario_G.csv'), index=False)
print('✅ NB03 V2 완료. 다음: NB04_analysis.ipynb')
print(f'\n저장된 파일:')
for fn in ['nomadic_v2_scenario_A.csv', 'nomadic_v2_scenario_B.csv',
           'nomadic_v2_scenario_C.csv', 'nomadic_v2_scenario_D.csv',
           'nomadic_v2_scenario_E.csv', 'nomadic_v2_scenario_F.csv',
           'nomadic_v2_scenario_G.csv', 'saturation_check_v2.json']:
    fp = os.path.join(RUN_DIR, fn)
    print(f'  {"✅" if os.path.exists(fp) else "❌"} {fn}')
