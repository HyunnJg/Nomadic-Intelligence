# -*- coding: utf-8 -*-
"""
Nomadic Intelligence — §4.9 Task Generalization Experiment (Lagged v3)
인과율(Lagged Feedback, k-1)이 강제된 ML적 추론 버전
"""

import os, random, math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams['figure.dpi'] = 120

# ============================================================
# STEP 0: 환경 및 실험 조건 설정
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS  = [42, 123, 456]

# 선택지: 'nonlinear' | 'abrupt' | 'gradual' | 'heavy_tail' | 'combined'
TASK_VARIANT = 'nonlinear'

print(f'Device: {DEVICE}')
print(f'Variant: {TASK_VARIANT}')

# ============================================================
# STEP 1: Config
# ============================================================
@dataclass
class Config:
    seed:   int   = 42
    device: str   = DEVICE

    input_dim:  int   = 2
    output_dim: int   = 1
    overlap_std: float = 0.9       

    hidden_dim:      int   = 64
    num_experts:     int   = 3
    gate_hidden_dim: int   = 64
    temperature:     float = 0.60

    epochs:       int   = 220
    lr:           float = 2e-3
    weight_decay: float = 1e-5

    phase_batch_size:    int = 64
    phase_train_cycles:  int = 40
    phase_test_cycles:   int = 12
    transition_steps:    int = 8       

    ema_decay:              float = 0.80
    err_baseline_momentum:  float = 0.85
    w_env: float = 1.0
    w_err: float = 2.0

    alpha_dogma:     float = 0.04
    beta_nomad:      float = 0.05
    beta_phi:        float = 0.02
    gamma_diversity: float = 0.08
    lambda_sep:      float = 0.08
    lambda_cons:     float = 0.03
    lambda_load:     float = 0.03
    tau_k_min:       int   = 3
    tau_k_penalty:   float = 0.05

    use_dynamic_tau: bool  = True
    tau_min:   float = 2.0
    tau_max:   float = 8.0
    tau_var_scale:  float = 6.0
    tau_var_window: int   = 8

    phi_scale_env:     float = 1.0
    phi_scale_err:     float = 1.5
    phi_scale_explain: float = 1.5
    phi_scale_gap:     float = 0.8

    temp_stable:     float = 0.35
    temp_transition: float = 0.90

    use_hard_switch:     bool  = True
    phi_hard_threshold:  float = 0.30

    policy_hidden_dim:       int   = 64
    policy_mix_weight:       float = 0.25
    policy_weight_stay:      float = 0.20
    policy_weight_target:    float = 0.20
    policy_weight_mode:      float = 0.10
    policy_switch_threshold: float = 0.50

def make_config(seed: int, variant: str) -> Config:
    cfg = Config(seed=seed)
    if variant == 'abrupt': cfg.transition_steps = 2
    elif variant == 'gradual': cfg.transition_steps = 24
    elif variant == 'combined': cfg.transition_steps = 2
    return cfg

# ============================================================
# STEP 2: Data Generation 
# ============================================================
REGIME_TO_ID = {'A': 0, 'B': 1, 'C': 2}
ID_TO_REGIME = {0: 'A', 1: 'B', 2: 'C'}
REGIME_ORDER = ['A', 'B', 'C']

def sample_noise(n: int, std: float, device: str, variant: str) -> torch.Tensor:
    if variant in ('heavy_tail', 'combined'):
        dist = torch.distributions.StudentT(df=2.0)
        noise = dist.sample((n, 2)).to(device) * std
    else:
        noise = std * torch.randn(n, 2, device=device)
    return noise

def sample_regime_x(regime: str, n: int, cfg: Config, variant: str) -> torch.Tensor:
    centers = {'A': (2.5, 2.5), 'B': (-2.5, -2.5), 'C': (2.5, -2.5)}
    c = torch.tensor(centers[regime], device=cfg.device)
    noise = sample_noise(n, cfg.overlap_std, cfg.device, variant)
    return noise + c

def regime_function(x: torch.Tensor, regime: str, variant: str) -> torch.Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    if variant in ('nonlinear', 'combined'):
        if regime == 'A':   y = torch.sin(x1) * x2
        elif regime == 'B': y = x1 ** 2 - x2 ** 2
        elif regime == 'C': y = torch.tanh(x1 + x2) * x2.abs()
    else:
        if regime == 'A':   y = x1 + x2
        elif regime == 'B': y = x1 - x2
        elif regime == 'C': y = -x1 + 0.5 * x2
    return y.unsqueeze(-1)

def generate_phase_sequence(cfg: Config, cycles: int, variant: str):
    xs, ys, rs, tags = [], [], [], []
    for _ in range(cycles):
        for i, curr_r in enumerate(REGIME_ORDER):
            next_r = REGIME_ORDER[(i + 1) % 3]
            x_s = sample_regime_x(curr_r, cfg.phase_batch_size, cfg, variant)
            y_s = regime_function(x_s, curr_r, variant)
            r_s = torch.full((cfg.phase_batch_size,), REGIME_TO_ID[curr_r], dtype=torch.long, device=cfg.device)
            xs.append(x_s); ys.append(y_s); rs.append(r_s)
            tags.extend([f'stable_{curr_r}'] * cfg.phase_batch_size)
            for step in range(cfg.transition_steps):
                alpha = (step + 1) / cfg.transition_steps
                x_a = sample_regime_x(curr_r, cfg.phase_batch_size, cfg, variant)
                x_b = sample_regime_x(next_r, cfg.phase_batch_size, cfg, variant)
                x_mix = (1 - alpha) * x_a + alpha * x_b
                y_mix = ((1 - alpha) * regime_function(x_mix, curr_r, variant) + alpha * regime_function(x_mix, next_r, variant))
                dominant = curr_r if alpha < 0.5 else next_r
                r_mix = torch.full((cfg.phase_batch_size,), REGIME_TO_ID[dominant], dtype=torch.long, device=cfg.device)
                xs.append(x_mix); ys.append(y_mix); rs.append(r_mix)
                tags.extend([f'transition_{curr_r}_to_{next_r}'] * cfg.phase_batch_size)
    return torch.cat(xs), torch.cat(ys), torch.cat(rs), tags

def iterate_sequence_minibatches(X, Y, R, batch_size):
    n = X.size(0)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        yield X[s:e], Y[s:e], R[s:e]

# ============================================================
# STEP 3: Models & Utilities
# ============================================================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.net(x)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.net(x)

class StandardMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, gate_hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(input_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, num_experts))
    def forward(self, x, hard=False):
        gate_probs = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        routing = F.one_hot(gate_probs.argmax(-1), self.num_experts).float() if hard else gate_probs
        return (routing.unsqueeze(-1) * expert_outputs).sum(1), gate_probs, None, expert_outputs

class NomadicMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, gate_hidden_dim, policy_hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(input_dim + 2, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, num_experts))
        self.policy_shared = nn.Sequential(nn.Linear(input_dim + 5, policy_hidden_dim), nn.ReLU(), nn.Linear(policy_hidden_dim, policy_hidden_dim), nn.ReLU())
        self.stay_head   = nn.Linear(policy_hidden_dim, 2)
        self.target_head = nn.Linear(policy_hidden_dim, num_experts)
        self.mode_head   = nn.Linear(policy_hidden_dim, 2)

    def gate_forward(self, x, dh, de, temperature):
        return F.softmax(self.gate(torch.cat([x, dh, de], dim=-1)) / temperature, dim=-1)

    def policy_forward(self, policy_input):
        h = self.policy_shared(policy_input)
        return F.softmax(self.stay_head(h), dim=-1), F.softmax(self.target_head(h), dim=-1), F.softmax(self.mode_head(h), dim=-1)

    def forward(self, x, delta_hybrid, delta_err, temperature, hard=False):
        gate_probs = self.gate_forward(x, delta_hybrid, delta_err, temperature)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        routing = F.one_hot(gate_probs.argmax(-1), self.num_experts).float() if hard else gate_probs
        return (routing.unsqueeze(-1) * expert_outputs).sum(1), gate_probs, None, expert_outputs

class HybridDeltaTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.recent_delta_env = deque(maxlen=cfg.tau_var_window)
        self.reset()
    def reset(self):
        self.prev_x_mean = None; self.err_ema = None; self.err_baseline = None
        self.recent_delta_env.clear()
    def compute(self, x, batch_mse):
        x_mean = x.mean(0, keepdim=True)
        de = 0.0 if self.prev_x_mean is None else float(torch.norm(x_mean - self.prev_x_mean, p=2).item())
        be = batch_mse.detach()
        if self.err_ema is None:
            self.err_ema = be; self.err_baseline = be; derr = 0.0
        else:
            self.err_ema = self.cfg.ema_decay * self.err_ema + (1 - self.cfg.ema_decay) * be
            self.err_baseline = self.cfg.err_baseline_momentum * self.err_baseline + (1 - self.cfg.err_baseline_momentum) * self.err_ema
            derr = float(torch.relu(self.err_ema - self.err_baseline).item())
        dh = float(torch.tanh(torch.tensor(self.cfg.w_env * de + self.cfg.w_err * derr)).item())
        self.prev_x_mean = x_mean.detach(); self.recent_delta_env.append(de)
        sigma2 = float(np.var(self.recent_delta_env)) if len(self.recent_delta_env) >= 2 else 0.0
        dyn_tau = float(np.clip(self.cfg.tau_min + (self.cfg.tau_max - self.cfg.tau_min) / (1.0 + self.cfg.tau_var_scale * sigma2), self.cfg.tau_min, self.cfg.tau_max))
        return torch.full((x.size(0), 1), dh, device=self.cfg.device), de, derr, dh, sigma2, dyn_tau

class DwellTimeRegularizer:
    def __init__(self, cfg): self.cfg = cfg; self.reset()
    def reset(self): self.current_expert = None; self.dwell_count = 0
    def compute(self, gate_probs, tau_dynamic=None):
        dominant = int(torch.bincount(gate_probs.argmax(-1), minlength=gate_probs.size(-1)).argmax().item())
        if dominant == self.current_expert: self.dwell_count += 1
        else: self.current_expert = dominant; self.dwell_count = 1
        entropy = -(gate_probs * (gate_probs + 1e-8).log()).sum(-1).mean()
        tau_cap = float(tau_dynamic if tau_dynamic is not None else self.cfg.tau_k_min)
        if self.dwell_count <= tau_cap: return -self.cfg.tau_k_penalty * entropy
        else: return min(float(self.dwell_count - tau_cap) * self.cfg.tau_k_penalty, self.cfg.tau_k_penalty * 10) * entropy

def gate_entropy(gate_probs): return -(gate_probs * (gate_probs + 1e-8).log()).sum(-1)
def compute_load_balancing_loss(gate_probs): return gate_probs.size(-1) * (torch.bincount(gate_probs.argmax(-1), minlength=gate_probs.size(-1)).float() / gate_probs.size(0) * gate_probs.mean(0)).sum()
def compute_dogma_penalty(gate_probs): return gate_probs.mean(0).pow(2).sum() - 1.0 / gate_probs.size(1)
def compute_nomad_bonus(gate_probs): return -(gate_probs * (gate_probs + 1e-8).log()).sum(-1).mean()
def compute_diversity_loss(expert_outputs):
    K = expert_outputs.size(1)
    if K < 2: return expert_outputs.new_zeros(1).squeeze()
    idx_i, idx_j = zip(*[(i,j) for i in range(K) for j in range(i+1,K)])
    return F.cosine_similarity(expert_outputs[:, idx_i, :], expert_outputs[:, idx_j, :], dim=-1).mean()

def compute_explanation_signals(y_true, y_hat, expert_outputs, gate_probs):
    expl_err = F.mse_loss(y_hat, y_true)
    per_err  = ((expert_outputs - y_true.unsqueeze(1)) ** 2).mean(-1)
    top1_err = per_err.gather(1, gate_probs.argmax(-1).unsqueeze(1)).mean()
    best_err = per_err.min(1).values.mean()
    return expl_err, torch.relu(top1_err - best_err)

def compute_phi(de, derr, expl_err, gap, cfg):
    dev = expl_err.device
    return torch.tanh(cfg.phi_scale_env * torch.tensor(de, device=dev) + cfg.phi_scale_err * torch.tensor(derr, device=dev) + cfg.phi_scale_explain * expl_err.detach() + cfg.phi_scale_gap * gap.detach())

def compute_temp(phi, cfg): return cfg.temp_stable + (cfg.temp_transition - cfg.temp_stable) * float(phi.mean().item())

def build_policy_input(xb, dh_t, de_t, phi, sigma2, dyn_tau):
    return torch.cat([xb.mean(0, keepdim=True).expand(xb.size(0), -1), dh_t, de_t, torch.full((xb.size(0),1), float(phi.mean().item()), device=xb.device), torch.full((xb.size(0),1), float(np.tanh(sigma2*10.0)), device=xb.device), torch.full((xb.size(0),1), float(np.tanh((dyn_tau-5.0)/5.0)), device=xb.device)], dim=-1)

def build_policy_targets(yb, exp_out, phi, sigma2, dyn_tau, cfg):
    tgt = ((exp_out - yb.unsqueeze(1))**2).mean(-1).mean(0).argmin().long()
    pv = float(phi.mean().item())
    return 1 if (pv > cfg.policy_switch_threshold or sigma2 > 0.05) else 0, tgt, 1 if (pv <= cfg.policy_switch_threshold and dyn_tau >= 5.5) else 0

def compute_regime_gate_stats(gate_probs, regime_ids):
    dev = gate_probs.device; valid_means = []; l_cons = torch.tensor(0.0, device=dev); cnt = 0
    for rid in range(3):
        mask = regime_ids == rid
        if mask.sum() > 0:
            g_r = gate_probs[mask]; u_r = g_r.mean(0)
            valid_means.append(u_r)
            l_cons = l_cons + ((g_r - u_r.unsqueeze(0))**2).sum(-1).mean(); cnt += 1
    if cnt > 0: l_cons = l_cons / cnt
    if len(valid_means) < 2: return torch.tensor(0.0, device=dev), l_cons
    return -torch.stack([torch.norm(valid_means[i]-valid_means[j], p=2) for i in range(len(valid_means)) for j in range(i+1, len(valid_means))]).mean(), l_cons

def regimewise_usage(gate_probs, regime_ids, num_experts):
    top1 = gate_probs.argmax(-1); usage = {}
    for rid in range(3):
        mask = regime_ids == rid; name = ID_TO_REGIME[rid]
        usage[name] = (torch.bincount(top1[mask], minlength=num_experts).float() / max(mask.sum().item(), 1.0)).cpu().numpy() if mask.sum() > 0 else np.zeros(num_experts)
    return usage

def infer_regime_to_expert(usage): return {r: int(np.argmax(usage[r])) for r in ['A','B','C']}

def compute_switch_latency(regime_seq, top1_seq, r2e):
    lats = []; prev = regime_seq[0] if regime_seq else None
    for t in range(1, len(regime_seq)):
        curr = regime_seq[t]
        if curr != prev and r2e.get(curr) is not None:
            for k in range(t, len(top1_seq)):
                if int(top1_seq[k]) == int(r2e[curr]): lats.append(k-t); break
        prev = curr
    return lats

# ============================================================
# STEP 4: Evaluation Functions (Lagged v3 반영)
# ============================================================
def eval_fixed_seq(model, X, Y, R, cfg):
    model.eval()
    with torch.no_grad(): return F.mse_loss(model(X), Y).item()

def eval_stdmoe_seq(model, X, Y, R, phase_tags, cfg):
    model.eval(); all_y, all_g, tags, ents = [], [], [], []
    with torch.no_grad():
        for bi, (xb,yb,rb) in enumerate(iterate_sequence_minibatches(X,Y,R,cfg.phase_batch_size)):
            y_hat, gp, _, _ = model(xb)
            all_y.append(y_hat); all_g.append(gp); tags.append(phase_tags[bi*cfg.phase_batch_size]); ents.append(gate_entropy(gp).mean().item())
    seq_mse = F.mse_loss(torch.cat(all_y), Y).item()
    sh = [e for t,e in zip(tags,ents) if t.startswith('stable_')]
    th = [e for t,e in zip(tags,ents) if t.startswith('transition_')]
    return seq_mse, {'stable_entropy_mean': float(np.mean(sh)) if sh else float('nan'), 'transition_entropy_mean': float(np.mean(th)) if th else float('nan'), 'delta_h': float(np.mean(th)-np.mean(sh)) if (sh and th) else float('nan')}

def eval_nomadic_seq(model, X, Y, R, phase_tags, cfg):
    model.eval()
    tracker = HybridDeltaTracker(cfg); tracker.reset()
    all_y, all_g, tags, ents, top1_list, reg_list = [], [], [], [], [], []
    
    # [핵심] 이전 배치 정답 버퍼
    prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

    with torch.no_grad():
        for bi, (xb,yb,rb) in enumerate(iterate_sequence_minibatches(X,Y,R,cfg.phase_batch_size)):
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            z = torch.zeros((xb.size(0),1), device=cfg.device)
            # [핵심] yb -> prev_yb
            warm_mse = F.mse_loss(model(xb,z,z,cfg.temperature)[0], prev_yb)
            dh_t, de, derr, dh, sigma2, dyn_tau = tracker.compute(xb, warm_mse)
            de_t = torch.full((xb.size(0),1), derr, device=cfg.device)

            p_y, p_g, _, p_e = model(xb, dh_t, de_t, cfg.temperature)
            # [핵심] yb -> prev_yb
            expl, gap = compute_explanation_signals(prev_yb, p_y, p_e, p_g)
            phi = compute_phi(de, derr, expl, gap, cfg)
            temp = compute_temp(phi, cfg)

            pol_in = build_policy_input(xb, dh_t, de_t, phi, sigma2, dyn_tau)
            sw_p, tgt_p, mode_p = model.policy_forward(pol_in)

            y_hat, gp, _, exp_out = model(xb, dh_t, de_t, temp)
            eff_mix = cfg.policy_mix_weight * float(sw_p[:,1].mean().item())
            tgt_ste = (F.one_hot(tgt_p.mean(0).argmax(), cfg.num_experts).float().unsqueeze(0).expand(xb.size(0),-1) - gp).detach() + gp
            mixed = (1-eff_mix)*gp + eff_mix*tgt_ste
            final_r = F.one_hot(mixed.argmax(-1), cfg.num_experts).float() if cfg.use_hard_switch and (mode_p[:,1].mean().item()>0.5) and not (dh > cfg.phi_hard_threshold) else mixed
            y_hat = (final_r.unsqueeze(-1)*exp_out).sum(1)

            all_y.append(y_hat); all_g.append(final_r); tags.append(phase_tags[bi*cfg.phase_batch_size])
            ents.append(gate_entropy(final_r).mean().item())
            top1_list.append(int(torch.bincount(final_r.argmax(-1), minlength=cfg.num_experts).argmax().item()))
            reg_list.append(ID_TO_REGIME[int(rb[0].item())])

            # [핵심] 버퍼 업데이트
            prev_yb = yb.detach()

    seq_mse = F.mse_loss(torch.cat(all_y), Y).item()
    sh = [e for t,e in zip(tags,ents) if t.startswith('stable_')]
    th = [e for t,e in zip(tags,ents) if t.startswith('transition_')]
    lats = compute_switch_latency(reg_list, np.array(top1_list), infer_regime_to_expert(regimewise_usage(torch.cat(all_g), R, cfg.num_experts)))
    return seq_mse, {'stable_entropy_mean': float(np.mean(sh)) if sh else float('nan'), 'transition_entropy_mean': float(np.mean(th)) if th else float('nan'), 'delta_h': float(np.mean(th)-np.mean(sh)) if (sh and th) else float('nan'), 'mean_switch_latency': float(np.mean(lats)) if lats else float('nan')}

# ============================================================
# STEP 5: Training Functions (Lagged v3 반영)
# ============================================================
def train_fixed(cfg, Xtr, Ytr, Rtr, Xte, Yte, Rte, tags_te):
    model = MLPRegressor(cfg.input_dim, cfg.hidden_dim, cfg.output_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse_log = []
    for ep in range(cfg.epochs):
        model.train()
        for xb,yb,_ in iterate_sequence_minibatches(Xtr,Ytr,Rtr,cfg.phase_batch_size):
            opt.zero_grad(); F.mse_loss(model(xb),yb).backward(); opt.step()
        mse_log.append(eval_fixed_seq(model, Xte, Yte, Rte, cfg))
    return model, mse_log

def train_stdmoe(cfg, Xtr, Ytr, Rtr, Xte, Yte, Rte, tags_te):
    model = StandardMoE(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_experts, cfg.gate_hidden_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse_log, dyn_log = [], []
    for ep in range(cfg.epochs):
        model.train()
        for xb,yb,_ in iterate_sequence_minibatches(Xtr,Ytr,Rtr,cfg.phase_batch_size):
            opt.zero_grad()
            y_hat, gp, _, exp = model(xb)
            (F.mse_loss(y_hat,yb) + cfg.gamma_diversity * compute_diversity_loss(exp) + cfg.lambda_load * compute_load_balancing_loss(gp)).backward(); opt.step()
        seq_mse, dyn = eval_stdmoe_seq(model, Xte, Yte, Rte, tags_te, cfg)
        mse_log.append(seq_mse); dyn_log.append(dyn)
        if (ep+1) % 55 == 0 or ep == 0: print(f'  [StdMoE] Ep {ep+1:03d} | MSE {seq_mse:.4f} | ΔH {dyn["delta_h"]:.3f}')
    return model, mse_log, dyn_log

def train_nomadic(cfg, Xtr, Ytr, Rtr, Xte, Yte, Rte, tags_te):
    model = NomadicMoE(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_experts, cfg.gate_hidden_dim, cfg.policy_hidden_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse_log, dyn_log = [], []

    for ep in range(cfg.epochs):
        model.train()
        tracker = HybridDeltaTracker(cfg); tracker.reset()
        dwell_reg = DwellTimeRegularizer(cfg); dwell_reg.reset()

        # [핵심] 이전 배치 정답 버퍼
        prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

        for xb, yb, rb in iterate_sequence_minibatches(Xtr,Ytr,Rtr,cfg.phase_batch_size):
            opt.zero_grad()
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            z = torch.zeros((xb.size(0),1), device=cfg.device)
            with torch.no_grad():
                # [핵심] yb -> prev_yb
                warm_mse = F.mse_loss(model(xb,z,z,cfg.temperature)[0], prev_yb)
            dh_t, de, derr, dh, sigma2, dyn_tau = tracker.compute(xb, warm_mse)
            de_t = torch.full((xb.size(0),1), derr, device=cfg.device)

            with torch.no_grad(): p_y, p_g, _, p_e = model(xb, dh_t, de_t, cfg.temperature)
            
            # [핵심] yb -> prev_yb
            expl, gap = compute_explanation_signals(prev_yb, p_y, p_e, p_g)
            phi = compute_phi(de, derr, expl, gap, cfg); temp = compute_temp(phi, cfg)

            pol_in = build_policy_input(xb, dh_t, de_t, phi, sigma2, dyn_tau)
            sw_p, tgt_p, mode_p = model.policy_forward(pol_in)

            y_hat, gp, _, exp_out = model(xb, dh_t, de_t, temp)
            eff_mix = cfg.policy_mix_weight * float(sw_p[:,1].mean().item())
            tgt_ste = (F.one_hot(tgt_p.mean(0).argmax(), cfg.num_experts).float().unsqueeze(0).expand(xb.size(0),-1) - gp).detach() + gp
            mixed = (1-eff_mix)*gp + eff_mix*tgt_ste
            final_r = F.one_hot(mixed.argmax(-1), cfg.num_experts).float() if cfg.use_hard_switch and (mode_p[:,1].mean().item()>0.5) and not (dh > cfg.phi_hard_threshold) else mixed
            y_hat = (final_r.unsqueeze(-1)*exp_out).sum(1)

            # Gradient Flow: Forward pass with CURRENT yb for true target loss
            _, gap2 = compute_explanation_signals(yb, y_hat, exp_out, final_r)
            sep_l, cons_l = compute_regime_gate_stats(final_r, rb)
            
            # [핵심] Policy Targets도 prev_yb 기준
            sw_t, tgt_t, mod_t = build_policy_targets(prev_yb, p_e, phi, sigma2, dyn_tau, cfg)

            loss = (F.mse_loss(y_hat, yb) 
                  + cfg.beta_phi * (phi.detach() * gap2) + cfg.alpha_dogma * compute_dogma_penalty(final_r) - cfg.beta_nomad * compute_nomad_bonus(final_r)
                  + cfg.gamma_diversity * compute_diversity_loss(exp_out) + cfg.lambda_sep * sep_l + cfg.lambda_cons * cons_l + cfg.lambda_load * compute_load_balancing_loss(final_r)
                  + cfg.policy_weight_stay * F.nll_loss(torch.log(sw_p+1e-8), torch.full((xb.size(0),), sw_t, dtype=torch.long, device=cfg.device))
                  + cfg.policy_weight_target * F.nll_loss(torch.log(tgt_p+1e-8), torch.full((xb.size(0),), int(tgt_t.item()), dtype=torch.long, device=cfg.device))
                  + cfg.policy_weight_mode * F.nll_loss(torch.log(mode_p+1e-8), torch.full((xb.size(0),), mod_t, dtype=torch.long, device=cfg.device))
                  - dwell_reg.compute(final_r, tau_dynamic=dyn_tau if cfg.use_dynamic_tau else float(cfg.tau_k_min)))
            loss.backward(); opt.step()

            # [핵심] 버퍼 업데이트
            prev_yb = yb.detach()

        seq_mse, dyn = eval_nomadic_seq(model, Xte, Yte, Rte, tags_te, cfg)
        mse_log.append(seq_mse); dyn_log.append(dyn)
        if (ep+1) % 55 == 0 or ep == 0:
            print(f'  [Nomadic] Ep {ep+1:03d} | MSE {seq_mse:.4f} | StH {dyn["stable_entropy_mean"]:.3f} | TrH {dyn["transition_entropy_mean"]:.3f} | ΔH {dyn["delta_h"]:.3f}')
    return model, mse_log, dyn_log

# ============================================================
# STEP 6: Run & Evaluate
# ============================================================
import time
all_results = {'Fixed': {}, 'StdMoE': {}, 'Nomadic': {}}

for seed in SEEDS:
    t0 = time.time()
    print(f'\n========== Seed {seed} | variant={TASK_VARIANT} (Lagged v3) ==========')
    set_seed(seed); cfg = make_config(seed, TASK_VARIANT)
    Xtr, Ytr, Rtr, tags_tr = generate_phase_sequence(cfg, cfg.phase_train_cycles, TASK_VARIANT)
    Xte, Yte, Rte, tags_te = generate_phase_sequence(cfg, cfg.phase_test_cycles,  TASK_VARIANT)

    print('--- Fixed ---')
    _, mse_log = train_fixed(cfg, Xtr, Ytr, Rtr, Xte, Yte, Rte, tags_te)
    all_results['Fixed'][seed] = {'mse_log': mse_log, 'dyn_log': None}

    print('--- Standard MoE ---')
    _, mse_log, dyn_log = train_stdmoe(cfg, Xtr, Ytr, Rtr, Xte, Yte, Rte, tags_te)
    all_results['StdMoE'][seed] = {'mse_log': mse_log, 'dyn_log': dyn_log}

    print('--- Nomadic Full ---')
    _, mse_log, dyn_log = train_nomadic(cfg, Xtr, Ytr, Rtr, Xte, Yte, Rte, tags_te)
    all_results['Nomadic'][seed] = {'mse_log': mse_log, 'dyn_log': dyn_log}

    print(f'  Seed {seed} done ({time.time()-t0:.0f}s)')

print('\n=== All seeds complete ===')

rows = []
for mn in ['Fixed', 'StdMoE', 'Nomadic']:
    mse_vals, dh_vals, sh_vals, th_vals = [], [], [], []
    for seed in SEEDS:
        r = all_results[mn][seed]
        mse_vals.append(r['mse_log'][-1])
        if r['dyn_log'] is not None:
            d = r['dyn_log'][-1]
            sh_vals.append(d['stable_entropy_mean'])
            th_vals.append(d['transition_entropy_mean'])
            dh_vals.append(d['delta_h'])
    rows.append({'Model': mn, 'Variant': TASK_VARIANT, 'Seq MSE mean': np.nanmean(mse_vals), 'ΔH mean': np.nanmean(dh_vals) if dh_vals else float('nan'), 'Stable Ent': np.nanmean(sh_vals) if sh_vals else float('nan'), 'Trans Ent': np.nanmean(th_vals) if th_vals else float('nan')})

df = pd.DataFrame(rows)
print('\n' + '='*70)
print(f'TASK GENERALIZATION: {TASK_VARIANT.upper()} (Lagged v3) — 3-seed mean')
print('='*70)
print(df.to_string(float_format=lambda x: f'{x:.4f}', index=False))