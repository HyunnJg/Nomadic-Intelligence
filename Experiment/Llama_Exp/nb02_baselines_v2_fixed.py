# -*- coding: utf-8 -*-
"""NB02_baselines_v2.py
Baseline Models — V2 수정본

수정 내역:
  [Fix 1] NomadicPolicyNet(구버전 5-dim) → NomadicPolicyNetV2(12-dim) 로 교체
          policy_net.pt 로드 실패(shape mismatch) 방지
  [Fix 2] build_meta_signals(5-dim) → build_meta_signals_v2(12-dim) 로 교체
          NB01 V2 / NB03 V2와 완전 정합
  [Fix 3] bnb_4bit_compute_dtype bfloat16 → float16 (T4 호환, 타 NB와 일치)
  [Fix 4] 시나리오 A 저장 컬럼에 mean_switch_prob / mean_final_switch_prob / mean_gap
          추가 (NaN) — NB04 Decision Diagnostics 테이블 호환
  [Fix 5] EXPERT_KEYS 중복 선언 제거 (382번 줄)
"""

# ============================================================
# STEP 0: 환경 로드 (NB01에서 저장된 설정 참조)
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

# [Fix 1/2] meta_config.json 로드 — PolicyNet V2 구조 정보
with open(os.path.join(RUN_DIR, 'meta_config.json')) as f:
    mcfg = json.load(f)
META_DIM    = mcfg['META_DIM']     # 12
NUM_EXPERTS = mcfg['NUM_EXPERTS']  # 3

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
print(f'Experts  : {list(expert_paths.keys())}')

# ============================================================
# STEP 1: 패키지 + 모델 로드
# ============================================================
import subprocess
subprocess.run(['pip', 'install', '-q', '-U',
                'transformers', 'accelerate', 'bitsandbytes', 'peft'])

import warnings; warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# [Fix 3] bfloat16 → float16 (T4 호환, NB01 V2 / NB03 V2와 일치)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
)
base_model.eval()
print('✅ 모델 로드 완료')

# ============================================================
# STEP 2: 공통 컴포넌트
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


# [Fix 1] PolicyNet V2 — NB01 V2 / NB03 V2와 동일한 클래스 정의
class NomadicPolicyNetV2(nn.Module):
    """
    NB01 V2에서 학습한 가중치와 호환되는 구조.
    meta input: 12-dim (9개 연속값 + 3개 expert one-hot)
    출력: logits (softmax 미적용) — loss fn 또는 F.softmax로 처리
    """
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


# [Fix 2] build_meta_signals_v2 — NB01 V2 / NB03 V2와 완전 동일한 함수
def build_meta_signals_v2(meta: dict, num_experts: int = 3,
                           device=None) -> torch.Tensor:
    """
    12-dim meta feature vector.
    NB01/NB03와 동일한 함수를 공유해야 train/inference 정합이 보장됨.
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


def topk_entropy(logits, k=50):
    probs = F.softmax(logits, dim=-1)
    topk  = probs.topk(k).values; topk = topk / topk.sum()
    return float(-(topk * (topk + 1e-10).log()).sum().item())


def repeat_rate(tokens):
    if len(tokens) < 3: return 0.0
    ngrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


# [Fix 5] EXPERT_KEYS 단일 선언 (중복 제거)
EXPERT_KEYS    = ['math', 'code', 'language']
MAX_NEW_TOKENS = 40
T_STABLE       = 0.35
T_TRANSITION   = 0.90
VANILLA_TEMP   = 0.70
N_RUNS         = 3
DOMAIN_PHASE   = {'math': 'stable', 'code': 'stable', 'language': 'transition'}


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
    ppl      = float(np.exp(-np.mean(log_probs))) if log_probs else float('nan')
    trace    = expert_trace or []
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

print('✅ 공통 컴포넌트 정의 완료')

# ============================================================
# STEP 3: Baseline 생성 함수 정의
# ============================================================

# ─── Vanilla ────────────────────────────────────────────────
def generate_vanilla(prompt):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
    ents, lps = [], []
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = base_model(ids)
        logits = out.logits[:, -1, :]
        ents.append(topk_entropy(logits))
        tok, lp = _next_tok(logits, VANILLA_TEMP)
        lps.append(lp)
        ids = torch.cat([ids, tok], dim=-1)
        if tok.item() == tokenizer.eos_token_id: break
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return base_result(text, ents, lps)


# ─── DynamicTemp (Δx→temperature, LoRA 없음) ────────────────
def generate_dynamic_temp(prompt):
    tracker = HybridDeltaTracker()
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
    ents, lps, dxs = [], [], []
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = base_model(ids, output_hidden_states=True)
        logits = out.logits[:, -1, :]
        hidden = out.hidden_states[-1][:, -1, :]
        err    = float(1.0 - F.softmax(logits, dim=-1).max().item())
        rec    = tracker.compute(hidden, err)
        d      = rec['delta_hybrid']
        temp   = T_STABLE + (T_TRANSITION - T_STABLE) * d
        ents.append(topk_entropy(logits / max(temp, 1e-4)))
        tok, lp = _next_tok(logits, temp)
        lps.append(lp); dxs.append(d)
        ids = torch.cat([ids, tok], dim=-1)
        if tok.item() == tokenizer.eos_token_id: break
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return base_result(text, ents, lps, dx_trace=dxs)


# ─── StaticLoRA (math LoRA 고정, 라우팅 없음) ───────────────
def generate_static_lora(prompt):
    expert = _load_expert('math')
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
    ents, lps = [], []
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = expert(ids)
        logits = out.logits[:, -1, :]
        ents.append(topk_entropy(logits))
        tok, lp = _next_tok(logits, VANILLA_TEMP)
        lps.append(lp)
        ids = torch.cat([ids, tok], dim=-1)
        if tok.item() == tokenizer.eos_token_id: break
    _unload_expert(expert)
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return base_result(text, ents, lps, expert_trace=['math'] * len(ents))


# ─── FixedRouting (키워드 기반 expert 선택, Δx 없음) ────────
KEYWORD_DOMAINS = {
    'math':     ['integral', 'derivative', 'equation', 'matrix', 'vector',
                 '수식', '방정식', '적분', '미분', '행렬', '함수', '수학', 'log', 'sin', 'cos'],
    'code':     ['def ', 'class ', 'import ', 'function', 'return', 'SELECT',
                 '#', 'print(', 'var ', '\n    ', 'const ', 'for (', 'if ('],
    'language': [],
}

def keyword_domain(prompt):
    p = prompt.lower()
    scores = {domain: sum(1 for kw in kws if kw.lower() in p)
              for domain, kws in KEYWORD_DOMAINS.items()}
    if max(scores.values()) == 0:
        return 'language'
    return max(scores, key=scores.get)

def generate_fixed_routing(prompt):
    domain = keyword_domain(prompt)
    expert = _load_expert(domain)
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
    ents, lps = [], []
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = expert(ids)
        logits = out.logits[:, -1, :]
        ents.append(topk_entropy(logits))
        tok, lp = _next_tok(logits, VANILLA_TEMP)
        lps.append(lp)
        ids = torch.cat([ids, tok], dim=-1)
        if tok.item() == tokenizer.eos_token_id: break
    _unload_expert(expert)
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return base_result(text, ents, lps, expert_trace=[domain] * len(ents))


# ─── Nomadic_NoDx (Δx=0, PolicyNet V2) ──────────────────────
# [Fix 1] PolicyNet V2 로드
policy_net = NomadicPolicyNetV2(
    hidden_dim=HIDDEN_DIM, meta_dim=META_DIM, num_experts=NUM_EXPERTS)
policy_net.load_state_dict(
    torch.load(os.path.join(RUN_DIR, 'policy_net.pt'),
               map_location=base_model.device))
policy_net = policy_net.to(base_model.device)
policy_net.eval()
print(f'✅ PolicyNet V2 로드 완료 | params: '
      f'{sum(p.numel() for p in policy_net.parameters()):,}')


def generate_nomadic_nodx(prompt):
    """
    Δx=0 고정. PolicyNet V2는 hidden state + zero Δx로만 라우팅 결정.
    목적: Δx 신호의 추가 기여분 분리.

    [Fix 2] build_meta_signals_v2 사용 (12-dim).
    meta의 Δx 관련 항(delta_env, delta_err, delta_hybrid, gap)을 0으로 고정.
    나머지(sigma2, tau_dyn, dwell_ratio, conf, logit_margin, expert_onehot)는 실제값.
    """
    tracker     = HybridDeltaTracker()
    ids         = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
    ents, lps, expert_trace, dxs = [], [], [], []
    current_key = 'math'
    current_model = _load_expert(current_key)
    dwell_count = 1

    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = current_model(ids, output_hidden_states=True)
        logits = out.logits[:, -1, :]
        hidden = out.hidden_states[-1][:, -1, :]

        probs  = F.softmax(logits, dim=-1)
        conf   = float(probs.max().item())
        top2   = probs.topk(2, dim=-1).values
        margin = float((top2[0, 0] - top2[0, 1]).item())
        err    = 1.0 - conf
        rec    = tracker.compute(hidden, err)

        # dwell 추적
        dwell_ratio = dwell_count / max(rec['tau_dyn'], 1e-6)

        # [Fix 2] Δx=0 고정 meta — delta_env/err/hybrid/gap을 0으로, 나머지는 실제값
        meta_nodx = {
            'delta_env':         0.0,   # Δx=0 고정
            'delta_err':         0.0,
            'delta_hybrid':      0.0,
            'sigma2':            rec['sigma2'],
            'tau_dyn':           rec['tau_dyn'],
            'dwell_ratio':       dwell_ratio,
            'current_conf':      conf,
            'logit_margin':      margin,
            'gap':               0.0,   # Δx=0이므로 gap proxy도 0
            'current_expert_id': EXPERT_KEYS.index(current_key),
        }
        meta_tensor = build_meta_signals_v2(
            meta_nodx, num_experts=NUM_EXPERTS, device=base_model.device)

        with torch.no_grad():
            sw_logits, tg_logits, _ = policy_net(hidden.unsqueeze(0), meta_tensor)

        switch_prob = float(F.softmax(sw_logits, dim=-1)[0, 1].item())
        tgt_key     = EXPERT_KEYS[int(F.softmax(tg_logits, dim=-1).argmax(-1).item())]

        next_key = tgt_key if switch_prob >= 0.5 else current_key
        if next_key != current_key:
            _unload_expert(current_model)
            current_model = _load_expert(next_key)
            current_key   = next_key
            dwell_count   = 1
        else:
            dwell_count += 1

        # Δx=0이므로 온도는 T_STABLE 고정
        ents.append(topk_entropy(logits / T_STABLE))
        tok, lp = _next_tok(logits, T_STABLE)
        lps.append(lp)
        expert_trace.append(current_key)
        dxs.append(0.0)
        ids = torch.cat([ids, tok], dim=-1)
        if tok.item() == tokenizer.eos_token_id: break

    _unload_expert(current_model)
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return base_result(text, ents, lps, expert_trace=expert_trace, dx_trace=dxs)


BASELINE_MODELS = {
    'Vanilla':      generate_vanilla,
    'DynamicTemp':  generate_dynamic_temp,
    'StaticLoRA':   generate_static_lora,
    'FixedRouting': generate_fixed_routing,
    'Nomadic_NoDx': generate_nomadic_nodx,
}

print('✅ Baseline 생성 함수 정의 완료')

# ============================================================
# STEP 4: 시나리오 A — 도메인별 기본 평가
# ============================================================
print('시나리오 A: 도메인별 기본 평가 시작...')
results_A = []

for model_name, gen_fn in BASELINE_MODELS.items():
    print(f'  [{model_name}]')
    for domain in ['math', 'code', 'language']:
        prompts = EVAL_PROMPTS[domain]
        for prompt in prompts:
            run_ents, run_ppls, run_reps, run_sws, run_dxs = [], [], [], [], []
            for run in range(N_RUNS):
                r = gen_fn(prompt)
                run_ents.append(r['mean_entropy'])
                run_ppls.append(r['perplexity'])
                run_reps.append(r['repeat_rate'])
                run_sws.append(r['switch_rate'])
                run_dxs.append(r['mean_dx'])

            results_A.append({
                'model':        model_name,
                'domain':       domain,
                'phase':        DOMAIN_PHASE[domain],
                'prompt':       prompt[:50],
                'mean_entropy': float(np.mean(run_ents)),
                'std_entropy':  float(np.std(run_ents)),
                'perplexity':   float(np.mean(run_ppls)),
                'repeat_rate':  float(np.mean(run_reps)),
                'switch_rate':  float(np.mean(run_sws)),
                'mean_dx':      float(np.mean(run_dxs)),
                # [Fix 4] NB04 Decision Diagnostics 호환용 컬럼 — baseline은 NaN
                'mean_switch_prob':       float('nan'),
                'mean_final_switch_prob': float('nan'),
                'mean_gap':               float('nan'),
            })
    print(f'    완료')

df_A = pd.DataFrame(results_A)
df_A.to_csv(os.path.join(RUN_DIR, 'baseline_scenario_A.csv'),
            index=False, encoding='utf-8-sig')
print(f'\n✅ 시나리오 A 완료 | {len(results_A)} rows 저장')

# ============================================================
# STEP 5: 시나리오 B — 도메인 급변 Δx 반응성
# ============================================================
print('시나리오 B: 도메인 급변 Δx 반응성...')
results_B = []

def run_shift_scenario(model_name, gen_fn, sequence, shift_point):
    dx_trace = []
    tracker  = HybridDeltaTracker()
    for prompt in sequence:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
        with torch.no_grad():
            out = base_model(ids, output_hidden_states=True)
        logits = out.logits[:, -1, :]
        hidden = out.hidden_states[-1][:, -1, :]
        err    = float(1.0 - F.softmax(logits, dim=-1).max().item())
        rec    = tracker.compute(hidden, err)
        dx_trace.append(rec['delta_hybrid'])

    pre  = float(np.mean(dx_trace[:shift_point]))
    post = float(np.mean(dx_trace[shift_point:]))
    return {
        'model':        model_name,
        'pre_mean_dx':  pre,
        'post_mean_dx': post,
        'delta_dx':     post - pre,
        'dx_trace':     dx_trace,
    }

for model_name in BASELINE_MODELS:
    r = run_shift_scenario(model_name, None,
                           SCENARIO_B_SEQUENCE, SCENARIO_B_SHIFT_POINT)
    results_B.append({k: v for k, v in r.items() if k != 'dx_trace'})
    print(f'  {model_name:20s}: pre={r["pre_mean_dx"]:.4f} '
          f'→ post={r["post_mean_dx"]:.4f}  Δ={r["delta_dx"]:+.4f}')

# Δx trace 시각화
trace_r = run_shift_scenario('trace', None, SCENARIO_B_SEQUENCE, SCENARIO_B_SHIFT_POINT)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(trace_r['dx_trace'], marker='o', markersize=4)
ax.axvline(SCENARIO_B_SHIFT_POINT - 0.5, color='red', linestyle='--', label='Domain shift')
ax.set_xlabel('Prompt Index'); ax.set_ylabel('Δx_hybrid')
ax.set_title('Scenario B: Δx Response to Domain Shift')
ax.legend(); ax.grid(alpha=0.3)
fig.savefig(os.path.join(RUN_DIR, 'scenario_B_dx_trace.png'), dpi=120, bbox_inches='tight')
plt.close(fig)

pd.DataFrame(results_B).to_csv(
    os.path.join(RUN_DIR, 'baseline_scenario_B.csv'), index=False)
print('✅ 시나리오 B 완료')

# ============================================================
# STEP 6: 시나리오 C — 노이즈 면역력
# ============================================================
print('시나리오 C: 노이즈 면역력...')
results_C = []

for model_name, gen_fn in BASELINE_MODELS.items():
    for condition, prompts in [('clean', SCENARIO_C_CLEAN), ('noisy', SCENARIO_C_NOISY)]:
        ents, ppls = [], []
        for prompt in prompts:
            r = gen_fn(prompt)
            ents.append(r['mean_entropy'])
            ppls.append(r['perplexity'])
        results_C.append({
            'model':        model_name,
            'condition':    condition,
            'mean_entropy': float(np.mean(ents)),
            'perplexity':   float(np.mean(ppls)),
        })
    print(f'  {model_name} 완료')

pd.DataFrame(results_C).to_csv(
    os.path.join(RUN_DIR, 'baseline_scenario_C.csv'), index=False)
print('✅ 시나리오 C 완료')

# ============================================================
# STEP 7: 시나리오 E — 유혹-회복
# ============================================================
print('시나리오 E: 유혹-회복...')
results_E = []

def run_lure_scenario(model_name, gen_fn, cases):
    tracker = HybridDeltaTracker()
    rows = []
    for math_prefix, lure, math_cont in cases:
        baseline_dxs = []
        ids = tokenizer(math_prefix, return_tensors='pt').input_ids.to(base_model.device)
        tracker.reset()
        for _ in range(8):
            with torch.no_grad():
                out = base_model(ids, output_hidden_states=True)
            logits = out.logits[:, -1, :]
            hidden = out.hidden_states[-1][:, -1, :]
            err    = float(1.0 - F.softmax(logits, dim=-1).max().item())
            rec    = tracker.compute(hidden, err)
            baseline_dxs.append(rec['delta_hybrid'])
            tok = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, tok], dim=-1)
        baseline_mean = float(np.mean(baseline_dxs))

        lure_ids = tokenizer(math_prefix + lure, return_tensors='pt').input_ids.to(base_model.device)
        with torch.no_grad():
            out = base_model(lure_ids, output_hidden_states=True)
        lure_hidden = out.hidden_states[-1][:, -1, :]
        lure_err    = float(1.0 - F.softmax(out.logits[:, -1, :], dim=-1).max().item())
        lure_rec    = tracker.compute(lure_hidden, lure_err)
        lure_spike  = lure_rec['delta_hybrid'] - baseline_mean

        recovery_ids = tokenizer(math_prefix + lure + math_cont,
                                  return_tensors='pt').input_ids.to(base_model.device)
        recovery_steps = 0
        threshold = baseline_mean + 0.05
        for step in range(20):
            with torch.no_grad():
                out = base_model(recovery_ids, output_hidden_states=True)
            h   = out.hidden_states[-1][:, -1, :]
            e   = float(1.0 - F.softmax(out.logits[:, -1, :], dim=-1).max().item())
            rec = tracker.compute(h, e)
            recovery_steps += 1
            if rec['delta_hybrid'] <= threshold: break
            tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            recovery_ids = torch.cat([recovery_ids, tok], dim=-1)

        rows.append({
            'model':          model_name,
            'baseline_dx':    baseline_mean,
            'lure_spike':     lure_spike,
            'recovery_steps': recovery_steps,
        })
    return rows

for model_name in BASELINE_MODELS:
    rows    = run_lure_scenario(model_name, None, SCENARIO_E_CASES)
    results_E.extend(rows)
    avg_sp  = float(np.mean([r['lure_spike']     for r in rows]))
    avg_rec = float(np.mean([r['recovery_steps'] for r in rows]))
    print(f'  {model_name:20s}: lure_spike={avg_sp:+.4f}  recovery_steps={avg_rec:.1f}')

pd.DataFrame(results_E).to_csv(
    os.path.join(RUN_DIR, 'baseline_scenario_E.csv'),
    index=False, encoding='utf-8-sig')
print('✅ 시나리오 E 완료')

# ============================================================
# STEP 8: 시나리오 F — 정보 밀도
# ============================================================
print('시나리오 F: 정보 밀도...')
results_F = []

for model_name in BASELINE_MODELS:
    for density, prompts in [('dense', SCENARIO_F_DENSE), ('sparse', SCENARIO_F_SPARSE)]:
        dx_list, conf_drop_list = [], []
        tracker = HybridDeltaTracker(); tracker.reset()
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
            with torch.no_grad():
                out = base_model(ids, output_hidden_states=True)
            logits = out.logits[:, -1, :]
            hidden = out.hidden_states[-1][:, -1, :]
            err    = float(1.0 - F.softmax(logits, dim=-1).max().item())
            rec    = tracker.compute(hidden, err)
            dx_list.append(rec['delta_hybrid'])
            conf_drop_list.append(err)

        corr = float(np.corrcoef(dx_list, conf_drop_list)[0, 1]) if len(dx_list) > 1 else 0.0
        results_F.append({
            'model':        model_name,
            'density':      density,
            'mean_dx':      float(np.mean(dx_list)),
            'mean_conf':    float(np.mean(conf_drop_list)),
            'corr_dx_conf': corr,
        })

pd.DataFrame(results_F).to_csv(
    os.path.join(RUN_DIR, 'baseline_scenario_F.csv'), index=False)
print('✅ 시나리오 F 완료')

# ============================================================
# STEP 9: 시나리오 G — 장기 고착도
# ============================================================
print('시나리오 G: 장기 고착도...')
results_G = []
LONG_STEPS     = 60
CONV_THRESHOLD = 0.02

for model_name in BASELINE_MODELS:
    for domain, prompt in SCENARIO_G_PROMPTS.items():
        tracker = HybridDeltaTracker()
        ids     = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
        phi_trace         = []
        convergence_step  = LONG_STEPS

        for step in range(LONG_STEPS):
            with torch.no_grad():
                out = base_model(ids, output_hidden_states=True)
            logits = out.logits[:, -1, :]
            hidden = out.hidden_states[-1][:, -1, :]
            err    = float(1.0 - F.softmax(logits, dim=-1).max().item())
            rec    = tracker.compute(hidden, err)
            phi_trace.append(rec['delta_hybrid'])

            if rec['delta_hybrid'] <= CONV_THRESHOLD and convergence_step == LONG_STEPS:
                convergence_step = step

            tok = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, tok], dim=-1)
            if tok.item() == tokenizer.eos_token_id: break

        phi_arr = np.array(phi_trace)
        slope   = float(np.polyfit(np.arange(len(phi_arr)), phi_arr, 1)[0]) \
                  if len(phi_arr) > 5 else 0.0

        results_G.append({
            'model':            model_name,
            'domain':           domain,
            'convergence_step': convergence_step,
            'final_phi_mean':   float(np.mean(phi_trace[-5:])) if len(phi_trace) >= 5 else float('nan'),
            'decay_rate':       slope,
        })
    print(f'  {model_name} 완료')

pd.DataFrame(results_G).to_csv(
    os.path.join(RUN_DIR, 'baseline_scenario_G.csv'), index=False)
print('✅ NB02 완료. 다음: NB03_nomadic_eval_v2.py')
