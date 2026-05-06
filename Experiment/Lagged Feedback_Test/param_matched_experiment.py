# -*- coding: utf-8 -*-
"""
Nomadic Intelligence — Parameter-Matched Baseline Experiment (Lagged v3)
인과율(Lagged Feedback, k-1)이 강제된 버전
"""

import os, random, math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams['figure.dpi'] = 120

# ============================================================
# STEP 0: 환경 확인
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# ============================================================
# STEP 1: Config
# ============================================================
@dataclass
class Config:
    seed: int = 42
    device: str = DEVICE

    input_dim: int = 2
    output_dim: int = 1
    overlap_std: float = 0.9

    hidden_dim: int = 64
    num_experts: int = 3
    gate_hidden_dim: int = 64
    temperature: float = 0.60

    epochs: int = 220
    lr: float = 2e-3
    weight_decay: float = 1e-5

    phase_batch_size: int = 64
    phase_train_cycles: int = 40
    phase_test_cycles: int = 12
    transition_steps: int = 8

    ema_decay: float = 0.80
    err_baseline_momentum: float = 0.85
    w_env: float = 1.0
    w_err: float = 2.0

    alpha_dogma: float = 0.04
    beta_nomad: float = 0.05
    beta_phi: float = 0.02
    gamma_diversity: float = 0.08
    lambda_sep: float = 0.08
    lambda_cons: float = 0.03
    lambda_load: float = 0.03
    tau_k_min: int = 3
    tau_k_penalty: float = 0.05

    use_dynamic_tau: bool = True
    tau_min: float = 2.0
    tau_max: float = 8.0
    tau_var_scale: float = 6.0
    tau_var_window: int = 8

    phi_scale_env: float = 1.0
    phi_scale_err: float = 1.5
    phi_scale_explain: float = 1.5
    phi_scale_gap: float = 0.8

    temp_stable: float = 0.35
    temp_transition: float = 0.90

    use_hard_switch: bool = True
    phi_hard_threshold: float = 0.30

    policy_hidden_dim: int = 64
    policy_mix_weight: float = 0.25
    policy_weight_stay: float = 0.20
    policy_weight_target: float = 0.20
    policy_weight_mode: float = 0.10
    policy_switch_threshold: float = 0.50

SEEDS = [42, 123, 456]
EXPERIMENT_VARIANTS = [
    ('Fixed_orig',    64,  False),
    ('StdMoE_orig',   64,  False),
    ('Fixed_matched', 150, True),
    ('StdMoE_matched', 74, True),
    ('NoPolicy_orig', 64,  False),
    ('Nomadic_Full',  64,  False),
]

def count_params(model): return sum(p.numel() for p in model.parameters())

# ============================================================
# STEP 2: Data Generation
# ============================================================
REGIME_TO_ID = {'A': 0, 'B': 1, 'C': 2}
ID_TO_REGIME = {0: 'A', 1: 'B', 2: 'C'}
REGIME_ORDER = ['A', 'B', 'C']

def sample_regime_x(regime, n, std, device='cpu'):
    noise = std * torch.randn(n, 2, device=device)
    centers = {'A': (2.5, 2.5), 'B': (-2.5, -2.5), 'C': (2.5, -2.5)}
    return noise + torch.tensor(centers[regime], device=device)

def regime_function(x, regime):
    x1, x2 = x[:, 0], x[:, 1]
    if regime == 'A':   y = x1 + x2
    elif regime == 'B': y = x1 - x2
    elif regime == 'C': y = -x1 + 0.5 * x2
    return y.unsqueeze(-1)

def generate_phase_sequence(cfg, cycles, device='cpu'):
    xs, ys, rs, phase_tags = [], [], [], []
    for _ in range(cycles):
        for i, curr_r in enumerate(REGIME_ORDER):
            next_r = REGIME_ORDER[(i + 1) % len(REGIME_ORDER)]
            x_s = sample_regime_x(curr_r, cfg.phase_batch_size, cfg.overlap_std, device)
            y_s = regime_function(x_s, curr_r)
            r_s = torch.full((cfg.phase_batch_size,), REGIME_TO_ID[curr_r], dtype=torch.long, device=device)
            xs.append(x_s); ys.append(y_s); rs.append(r_s)
            phase_tags.extend([f'stable_{curr_r}'] * cfg.phase_batch_size)
            for step in range(cfg.transition_steps):
                alpha = (step + 1) / cfg.transition_steps
                x_a = sample_regime_x(curr_r, cfg.phase_batch_size, cfg.overlap_std, device)
                x_b = sample_regime_x(next_r, cfg.phase_batch_size, cfg.overlap_std, device)
                x_mix = (1.0 - alpha) * x_a + alpha * x_b
                y_mix = (1.0 - alpha) * regime_function(x_mix, curr_r) + alpha * regime_function(x_mix, next_r)
                dominant = curr_r if alpha < 0.5 else next_r
                r_mix = torch.full((cfg.phase_batch_size,), REGIME_TO_ID[dominant], dtype=torch.long, device=device)
                xs.append(x_mix); ys.append(y_mix); rs.append(r_mix)
                phase_tags.extend([f'transition_{curr_r}_to_{next_r}'] * cfg.phase_batch_size)
    return torch.cat(xs), torch.cat(ys), torch.cat(rs), phase_tags

def iterate_sequence_minibatches(X, Y, R, batch_size):
    n = X.size(0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield X[start:end], Y[start:end], R[start:end]

# ============================================================
# STEP 3: Model Definitions
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

class GateNet(nn.Module):
    def __init__(self, input_dim, gate_hidden_dim, num_experts, use_delta=True):
        super().__init__()
        in_dim = input_dim + 2 if use_delta else input_dim
        self.use_delta = use_delta
        self.net = nn.Sequential(nn.Linear(in_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, num_experts))
    def forward(self, x, delta_hybrid=None, delta_err=None, temperature=1.0):
        gate_input = torch.cat([x, delta_hybrid, delta_err], dim=-1) if self.use_delta else x
        logits = self.net(gate_input)
        return F.softmax(logits / temperature, dim=-1), logits

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim + 5, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.stay_switch_head = nn.Linear(hidden_dim, 2)
        self.target_head      = nn.Linear(hidden_dim, num_experts)
        self.mode_head        = nn.Linear(hidden_dim, 2)
    def forward(self, policy_input):
        h = self.shared(policy_input)
        return F.softmax(self.stay_switch_head(h), dim=-1), F.softmax(self.target_head(h), dim=-1), F.softmax(self.mode_head(h), dim=-1)

class StandardMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, gate_hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(input_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, num_experts))
    def forward(self, x, hard=False):
        logits = self.gate(x)
        gate_probs = F.softmax(logits, dim=-1)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        routing = F.one_hot(gate_probs.argmax(-1), self.num_experts).float() if hard else gate_probs
        return (routing.unsqueeze(-1) * expert_outputs).sum(dim=1), gate_probs, logits, expert_outputs

class NomadicMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, gate_hidden_dim, policy_hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate   = GateNet(input_dim, gate_hidden_dim, num_experts, use_delta=True)
        self.policy = PolicyNet(input_dim, policy_hidden_dim, num_experts)
    def forward(self, x, delta_hybrid, delta_err, temperature, hard=False):
        gate_probs, gate_logits = self.gate(x, delta_hybrid, delta_err, temperature)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        routing = F.one_hot(gate_probs.argmax(-1), self.num_experts).float() if hard else gate_probs
        return (routing.unsqueeze(-1) * expert_outputs).sum(dim=1), gate_probs, gate_logits, expert_outputs

# ============================================================
# STEP 4: Utilities
# ============================================================
class HybridDeltaTracker:
    def __init__(self, cfg, device):
        self.cfg = cfg; self.device = device
        self.recent_delta_env = deque(maxlen=cfg.tau_var_window)
        self.reset()
    def reset(self):
        self.prev_x_mean = None; self.err_ema = None; self.err_baseline = None
        self.recent_delta_env.clear()
    def compute_dynamic_tau(self, sigma2):
        tau = self.cfg.tau_min + (self.cfg.tau_max - self.cfg.tau_min) / (1.0 + self.cfg.tau_var_scale * sigma2)
        return float(np.clip(tau, self.cfg.tau_min, self.cfg.tau_max))
    def compute(self, x, batch_mse):
        x_mean = x.mean(dim=0, keepdim=True)
        de = 0.0 if self.prev_x_mean is None else float(torch.norm(x_mean - self.prev_x_mean, p=2).item())
        batch_err = batch_mse.detach()
        if self.err_ema is None:
            self.err_ema = batch_err; self.err_baseline = batch_err; derr = 0.0
        else:
            self.err_ema = self.cfg.ema_decay * self.err_ema + (1 - self.cfg.ema_decay) * batch_err
            self.err_baseline = self.cfg.err_baseline_momentum * self.err_baseline + (1 - self.cfg.err_baseline_momentum) * self.err_ema
            derr = float(torch.relu(self.err_ema - self.err_baseline).item())
        dh = float(torch.tanh(torch.tensor(self.cfg.w_env * de + self.cfg.w_err * derr)).item())
        self.prev_x_mean = x_mean.detach()
        self.recent_delta_env.append(de)
        sigma2 = float(np.var(self.recent_delta_env)) if len(self.recent_delta_env) >= 2 else 0.0
        dynamic_tau = self.compute_dynamic_tau(sigma2)
        return torch.full((x.size(0), 1), dh, device=self.device), de, derr, dh, sigma2, dynamic_tau

class DwellTimeRegularizer:
    def __init__(self, tau_k_min=3, penalty=0.05):
        self.tau_k_min = tau_k_min; self.penalty = penalty
        self.reset()
    def reset(self):
        self.current_expert = None; self.dwell_count = 0
    def compute(self, gate_probs, tau_dynamic=None):
        dominant = int(torch.bincount(gate_probs.argmax(-1), minlength=gate_probs.size(-1)).argmax().item())
        if dominant == self.current_expert: self.dwell_count += 1
        else: self.current_expert = dominant; self.dwell_count = 1
        entropy = -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=-1).mean()
        tau_cap = float(tau_dynamic if tau_dynamic is not None else self.tau_k_min)
        if self.dwell_count <= tau_cap: return -self.penalty * entropy
        else: return min(float(self.dwell_count - tau_cap) * self.penalty, self.penalty * 10) * entropy

def gate_entropy(gate_probs): return -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=-1)
def compute_load_balancing_loss(gate_probs): return gate_probs.size(-1) * (torch.bincount(gate_probs.argmax(dim=-1), minlength=gate_probs.size(-1)).float() / gate_probs.size(0) * gate_probs.mean(dim=0)).sum()
def compute_dogma_penalty(gate_probs): return torch.sum(gate_probs.mean(dim=0) ** 2) - 1.0 / gate_probs.size(1)
def compute_nomad_bonus(gate_probs): return -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=-1).mean()
def compute_diversity_loss(expert_outputs):
    K = expert_outputs.size(1)
    if K < 2: return expert_outputs.new_zeros(1).squeeze()
    idx_i, idx_j = zip(*[(i,j) for i in range(K) for j in range(i+1,K)])
    return F.cosine_similarity(expert_outputs[:, idx_i, :], expert_outputs[:, idx_j, :], dim=-1).mean()

def compute_explanation_signals(y_true, y_hat, expert_outputs, gate_probs):
    per_expert_sqerr = ((expert_outputs - y_true.unsqueeze(1)) ** 2).mean(dim=-1)
    top1_err = per_expert_sqerr.gather(1, gate_probs.argmax(dim=-1).unsqueeze(1)).mean()
    best_expert_err = per_expert_sqerr.min(dim=1).values.mean()
    return F.mse_loss(y_hat, y_true), torch.relu(top1_err - best_expert_err)

def compute_phi_signal(de, derr, explanation_error, best_expert_gap, cfg):
    dev = explanation_error.device
    return torch.tanh(cfg.phi_scale_env * torch.tensor(de, device=dev) + cfg.phi_scale_err * torch.tensor(derr, device=dev) + cfg.phi_scale_explain * explanation_error.detach() + cfg.phi_scale_gap * best_expert_gap.detach())

def compute_adaptive_temperature(phi_signal, cfg): return cfg.temp_stable + (cfg.temp_transition - cfg.temp_stable) * float(phi_signal.mean().item())

def build_policy_input(xb, delta_hybrid, delta_err_t, phi_signal, sigma2, dynamic_tau):
    return torch.cat([xb.mean(dim=0, keepdim=True).expand(xb.size(0), -1), delta_hybrid, delta_err_t, torch.full((xb.size(0), 1), float(phi_signal.mean().item()), device=xb.device), torch.full((xb.size(0), 1), float(np.tanh(sigma2 * 10.0)), device=xb.device), torch.full((xb.size(0), 1), float(np.tanh((dynamic_tau - 5.0) / 5.0)), device=xb.device)], dim=-1)

def build_policy_targets(yb, expert_outputs, phi_signal, sigma2, dynamic_tau, cfg):
    target_expert = ((expert_outputs - yb.unsqueeze(1)) ** 2).mean(dim=-1).mean(dim=0).argmin().long()
    phi_val = float(phi_signal.mean().item())
    return (1 if (phi_val > cfg.policy_switch_threshold) or (sigma2 > 0.05) else 0), target_expert, (1 if (phi_val <= cfg.policy_switch_threshold) and (dynamic_tau >= 5.5) else 0)

def compute_regime_gate_stats(gate_probs, regime_ids, num_regimes=3):
    dev = gate_probs.device; valid_means = []; l_cons = torch.tensor(0.0, device=dev); cnt = 0
    for rid in range(num_regimes):
        mask = regime_ids == rid
        if mask.sum() > 0:
            g_r = gate_probs[mask]; u_r = g_r.mean(dim=0)
            valid_means.append(u_r); l_cons = l_cons + ((g_r - u_r.unsqueeze(0)) ** 2).sum(dim=-1).mean(); cnt += 1
    if cnt > 0: l_cons = l_cons / cnt
    if len(valid_means) < 2: return torch.tensor(0.0, device=dev), l_cons
    return -torch.stack([torch.norm(valid_means[i] - valid_means[j], p=2) for i in range(len(valid_means)) for j in range(i+1, len(valid_means))]).mean(), l_cons

def regimewise_usage(gate_probs, regime_ids, num_experts):
    usage = {}; top1 = gate_probs.argmax(dim=-1)
    for rid in range(3):
        mask = regime_ids == rid; name = ID_TO_REGIME[rid]
        if mask.sum() == 0: usage[name] = np.zeros(num_experts); continue
        usage[name] = (torch.bincount(top1[mask], minlength=num_experts).float() / mask.sum()).cpu().numpy()
    return usage

def infer_regime_to_expert(usage): return {r: int(np.argmax(usage[r])) for r in ['A','B','C']}

def compute_switch_latency(regime_seq, top1_seq, regime_to_expert):
    latencies = []; prev = regime_seq[0] if regime_seq else None
    for t in range(1, len(regime_seq)):
        curr = regime_seq[t]
        if curr != prev and regime_to_expert.get(curr) is not None:
            for k in range(t, len(top1_seq)):
                if int(top1_seq[k]) == int(regime_to_expert[curr]): latencies.append(k - t); break
        prev = curr
    return latencies

# ============================================================
# STEP 5: Evaluation Functions (Lagged v3 반영)
# ============================================================
def eval_fixed(model, X, Y, R, cfg):
    model.eval()
    with torch.no_grad(): return F.mse_loss(model(X), Y).item()

def eval_stdmoe_seq(model, X, Y, R, phase_tags, cfg):
    model.eval()
    all_y, all_gate, batch_tags, batch_ents = [], [], [], []
    with torch.no_grad():
        for bi, (xb, yb, rb) in enumerate(iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)):
            y_hat, gate_probs, _, _ = model(xb, hard=False)
            all_y.append(y_hat); all_gate.append(gate_probs)
            batch_tags.append(phase_tags[bi * cfg.phase_batch_size])
            batch_ents.append(gate_entropy(gate_probs).mean().item())
    seq_mse = F.mse_loss(torch.cat(all_y), Y).item()
    stable_h = [e for t,e in zip(batch_tags, batch_ents) if t.startswith('stable_')]
    transition_h = [e for t,e in zip(batch_tags, batch_ents) if t.startswith('transition_')]
    return seq_mse, {'stable_entropy_mean': float(np.mean(stable_h)) if stable_h else float('nan'), 'transition_entropy_mean': float(np.mean(transition_h)) if transition_h else float('nan')}

def eval_nomadic_seq(model, X, Y, R, phase_tags, cfg, use_policy=True):
    model.eval()
    tracker = HybridDeltaTracker(cfg, cfg.device); tracker.reset()
    all_y, all_gate, batch_tags, batch_ents, batch_top1, batch_regimes = [], [], [], [], [], []

    # [핵심] 이전 배치 정답 버퍼
    prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

    with torch.no_grad():
        for bi, (xb, yb, rb) in enumerate(iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)):
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            z = torch.zeros((xb.size(0), 1), device=cfg.device)
            # [핵심] yb -> prev_yb (시차 기동)
            warm_mse = F.mse_loss(model(xb, z, z, cfg.temperature)[0], prev_yb)
            delta_hybrid, de, derr, dh, sigma2, dyn_tau = tracker.compute(xb, warm_mse)
            delta_err_t = torch.full((xb.size(0), 1), derr, device=cfg.device)

            probe_y, probe_gate, _, probe_exp = model(xb, delta_hybrid, delta_err_t, cfg.temperature)
            # [핵심] yb -> prev_yb
            expl_err, gap = compute_explanation_signals(prev_yb, probe_y, probe_exp, probe_gate)
            phi = compute_phi_signal(de, derr, expl_err, gap, cfg)
            temp_now = compute_adaptive_temperature(phi, cfg)

            y_hat, gate_probs, _, exp_out = model(xb, delta_hybrid, delta_err_t, temp_now)

            if use_policy:
                policy_input = build_policy_input(xb, delta_hybrid, delta_err_t, phi, sigma2, dyn_tau)
                stay_sw, tgt_probs, mode_probs = model.policy(policy_input)
                effective_mix = cfg.policy_mix_weight * float(stay_sw[:, 1].mean().item())
                tgt_ste = (F.one_hot(torch.argmax(tgt_probs.mean(dim=0)), cfg.num_experts).float().unsqueeze(0).expand(xb.size(0), -1) - gate_probs).detach() + gate_probs
                mixed = (1.0 - effective_mix) * gate_probs + effective_mix * tgt_ste
                final_routing = F.one_hot(mixed.argmax(-1), cfg.num_experts).float() if cfg.use_hard_switch and (mode_probs[:, 1].mean().item() > 0.5) and not (dh > cfg.phi_hard_threshold) else mixed
                y_hat = (final_routing.unsqueeze(-1) * exp_out).sum(dim=1)
                gate_probs = final_routing

            all_y.append(y_hat); all_gate.append(gate_probs)
            batch_tags.append(phase_tags[bi * cfg.phase_batch_size]); batch_ents.append(gate_entropy(gate_probs).mean().item())
            batch_top1.append(int(torch.bincount(gate_probs.argmax(-1), minlength=cfg.num_experts).argmax().item()))
            batch_regimes.append(ID_TO_REGIME[int(rb[0].item())])

            # [핵심] 버퍼 업데이트
            prev_yb = yb.detach()

    Y_hat = torch.cat(all_y); G = torch.cat(all_gate)
    seq_mse = F.mse_loss(Y_hat, Y).item()
    sh = [e for t,e in zip(batch_tags, batch_ents) if t.startswith('stable_')]
    th = [e for t,e in zip(batch_tags, batch_ents) if t.startswith('transition_')]
    lats = compute_switch_latency(batch_regimes, np.array(batch_top1), infer_regime_to_expert(regimewise_usage(G, R, cfg.num_experts)))
    return seq_mse, {'stable_entropy_mean': float(np.mean(sh)) if sh else float('nan'), 'transition_entropy_mean': float(np.mean(th)) if th else float('nan'), 'mean_switch_latency': float(np.mean(lats)) if lats else float('nan')}

# ============================================================
# STEP 6: Training Functions (Lagged v3 반영)
# ============================================================
def train_fixed(cfg, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test):
    h = cfg.hidden_dim
    model = MLPRegressor(cfg.input_dim, h, cfg.output_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    test_seq_mse_log = []
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb, _ in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            opt.zero_grad(); F.mse_loss(model(xb), yb).backward(); opt.step()
        seq_mse = eval_fixed(model, X_test, Y_test, R_test, cfg)
        test_seq_mse_log.append(seq_mse)
    return model, test_seq_mse_log

def train_stdmoe(cfg, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test):
    h = cfg.hidden_dim
    model = StandardMoE(cfg.input_dim, h, cfg.output_dim, cfg.num_experts, h).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    test_seq_mse_log, dyn_log = [], []
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb, _ in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            opt.zero_grad()
            y_hat, gate_probs, _, exp_out = model(xb)
            (F.mse_loss(y_hat, yb) + cfg.gamma_diversity * compute_diversity_loss(exp_out) + cfg.lambda_load * compute_load_balancing_loss(gate_probs)).backward(); opt.step()
        seq_mse, dyn = eval_stdmoe_seq(model, X_test, Y_test, R_test, phase_tags_test, cfg)
        test_seq_mse_log.append(seq_mse); dyn_log.append(dyn)
    return model, test_seq_mse_log, dyn_log

def train_nomadic_full(cfg, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test):
    model = NomadicMoE(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_experts, cfg.gate_hidden_dim, cfg.policy_hidden_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    test_seq_mse_log, dyn_log = [], []

    for epoch in range(cfg.epochs):
        model.train()
        tracker = HybridDeltaTracker(cfg, cfg.device); tracker.reset()
        dwell_reg = DwellTimeRegularizer(cfg.tau_k_min, cfg.tau_k_penalty); dwell_reg.reset()

        # [핵심] 이전 배치 정답 버퍼
        prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

        for xb, yb, rb in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            opt.zero_grad()
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            with torch.no_grad():
                z = torch.zeros((xb.size(0),1), device=cfg.device)
                # [핵심] yb -> prev_yb
                warm_mse = F.mse_loss(model(xb, z, z, cfg.temperature)[0], prev_yb)
            delta_hybrid, de, derr, dh, sigma2, dyn_tau = tracker.compute(xb, warm_mse)
            delta_err_t = torch.full((xb.size(0),1), derr, device=cfg.device)

            with torch.no_grad(): probe_y, probe_gate, _, probe_exp = model(xb, delta_hybrid, delta_err_t, cfg.temperature)
            
            # [핵심] yb -> prev_yb
            expl_err, gap = compute_explanation_signals(prev_yb, probe_y, probe_exp, probe_gate)
            phi = compute_phi_signal(de, derr, expl_err, gap, cfg); temp_now = compute_adaptive_temperature(phi, cfg)

            policy_input = build_policy_input(xb, delta_hybrid, delta_err_t, phi, sigma2, dyn_tau)
            stay_sw, tgt_probs, mode_probs = model.policy(policy_input)

            y_hat, gate_probs, _, exp_out = model(xb, delta_hybrid, delta_err_t, temp_now)
            effective_mix = cfg.policy_mix_weight * float(stay_sw[:,1].mean().item())
            tgt_ste = (F.one_hot(torch.argmax(tgt_probs.mean(0)), cfg.num_experts).float().unsqueeze(0).expand(xb.size(0),-1) - gate_probs).detach() + gate_probs
            mixed = (1.0 - effective_mix)*gate_probs + effective_mix*tgt_ste
            final_routing = F.one_hot(mixed.argmax(-1), cfg.num_experts).float() if cfg.use_hard_switch and (mode_probs[:,1].mean().item()>0.5) and not (dh > cfg.phi_hard_threshold) else mixed
            y_hat = (final_routing.unsqueeze(-1) * exp_out).sum(1)

            # Gradient Flow: Forward pass with CURRENT yb for true target loss
            _, gap_loss = compute_explanation_signals(yb, y_hat, exp_out, final_routing)
            sep_loss, cons_loss = compute_regime_gate_stats(final_routing, rb)
            
            # [핵심] Policy Targets도 prev_yb 기준
            sw_lbl, tgt_lbl, mode_lbl = build_policy_targets(prev_yb, probe_exp, phi, sigma2, dyn_tau, cfg)

            loss = (F.mse_loss(y_hat, yb)
                    + cfg.beta_phi * (phi.detach() * gap_loss) + cfg.alpha_dogma * compute_dogma_penalty(final_routing) - cfg.beta_nomad * compute_nomad_bonus(final_routing)
                    + cfg.gamma_diversity * compute_diversity_loss(exp_out) + cfg.lambda_sep * sep_loss + cfg.lambda_cons * cons_loss + cfg.lambda_load * compute_load_balancing_loss(final_routing)
                    + cfg.policy_weight_stay * F.nll_loss(torch.log(stay_sw+1e-8), torch.full((xb.size(0),), sw_lbl, dtype=torch.long, device=cfg.device))
                    + cfg.policy_weight_target * F.nll_loss(torch.log(tgt_probs+1e-8), torch.full((xb.size(0),), int(tgt_lbl.item()), dtype=torch.long, device=cfg.device))
                    + cfg.policy_weight_mode * F.nll_loss(torch.log(mode_probs+1e-8), torch.full((xb.size(0),), mode_lbl, dtype=torch.long, device=cfg.device))
                    - dwell_reg.compute(final_routing, tau_dynamic=dyn_tau if cfg.use_dynamic_tau else float(cfg.tau_k_min)))
            loss.backward(); opt.step()

            # [핵심] 버퍼 업데이트
            prev_yb = yb.detach()

        seq_mse, dyn = eval_nomadic_seq(model, X_test, Y_test, R_test, phase_tags_test, cfg, use_policy=True)
        test_seq_mse_log.append(seq_mse); dyn_log.append(dyn)
        if (epoch+1) % 55 == 0 or epoch == 0:
            print(f'  [Nomadic Full] Ep {epoch+1:03d} | Seq MSE: {seq_mse:.4f} | StableH: {dyn["stable_entropy_mean"]:.4f} | TransH: {dyn["transition_entropy_mean"]:.4f}')
    return model, test_seq_mse_log, dyn_log

# ============================================================
# STEP 7: 실험 실행
# ============================================================
import time
all_results = {}
variant_cfgs = {
    'Fixed_orig':     {'hidden_dim': 64},
    'StdMoE_orig':    {'hidden_dim': 64},
    'Fixed_matched':  {'hidden_dim': 150},
    'StdMoE_matched': {'hidden_dim': 74},
    'NoPolicy_orig':  {'hidden_dim': 64},
    'Nomadic_Full':   {'hidden_dim': 64},
}

for variant, v_cfg in variant_cfgs.items():
    all_results[variant] = {}
    for seed in SEEDS:
        t0 = time.time()
        print(f'\n>>> {variant} | seed={seed} (Lagged v3)')
        set_seed(seed); cfg = Config(seed=seed, hidden_dim=v_cfg['hidden_dim'])
        X_tr, Y_tr, R_tr, tags_tr = generate_phase_sequence(cfg, cfg.phase_train_cycles, cfg.device)
        X_te, Y_te, R_te, tags_te = generate_phase_sequence(cfg, cfg.phase_test_cycles,  cfg.device)

        if variant in ('Fixed_orig', 'Fixed_matched'):
            _, mse_log = train_fixed(cfg, X_tr, Y_tr, R_tr, X_te, Y_te, R_te, tags_te)
            all_results[variant][seed] = {'seq_mse_log': mse_log, 'dyn_log': None}

        elif variant in ('StdMoE_orig', 'StdMoE_matched'):
            cfg.gate_hidden_dim = v_cfg['hidden_dim']
            _, mse_log, dyn_log = train_stdmoe(cfg, X_tr, Y_tr, R_tr, X_te, Y_te, R_te, tags_te)
            all_results[variant][seed] = {'seq_mse_log': mse_log, 'dyn_log': dyn_log}

        elif variant == 'NoPolicy_orig':
            model = NomadicMoE(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_experts, cfg.gate_hidden_dim, cfg.policy_hidden_dim).to(cfg.device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            mse_log, dyn_log = [], []
            for epoch in range(cfg.epochs):
                model.train()
                tracker = HybridDeltaTracker(cfg, cfg.device); tracker.reset()
                dwell_reg = DwellTimeRegularizer(cfg.tau_k_min, cfg.tau_k_penalty); dwell_reg.reset()
                prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

                for xb, yb, rb in iterate_sequence_minibatches(X_tr, Y_tr, R_tr, cfg.phase_batch_size):
                    opt.zero_grad()
                    if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)
                    with torch.no_grad():
                        z = torch.zeros((xb.size(0),1), device=cfg.device)
                        warm_mse = F.mse_loss(model(xb, z, z, cfg.temperature)[0], prev_yb) # Lagged
                    dh_t, de, derr, dh, sigma2, dyn_tau = tracker.compute(xb, warm_mse)
                    delta_err_t = torch.full((xb.size(0),1), derr, device=cfg.device)
                    with torch.no_grad(): probe_y, probe_gate, _, probe_exp = model(xb, dh_t, delta_err_t, cfg.temperature)
                    expl_err, gap = compute_explanation_signals(prev_yb, probe_y, probe_exp, probe_gate) # Lagged
                    phi = compute_phi_signal(de, derr, expl_err, gap, cfg); temp_now = compute_adaptive_temperature(phi, cfg)
                    y_hat, gate_probs, _, exp_out = model(xb, dh_t, delta_err_t, temp_now)
                    final_routing = gate_probs
                    y_hat = (final_routing.unsqueeze(-1) * exp_out).sum(1)
                    _, gap_loss = compute_explanation_signals(yb, y_hat, exp_out, final_routing)
                    sep_loss, cons_loss = compute_regime_gate_stats(final_routing, rb)
                    loss = (F.mse_loss(y_hat, yb) + cfg.beta_phi * (phi.detach() * gap_loss) + cfg.alpha_dogma * compute_dogma_penalty(final_routing) - cfg.beta_nomad * compute_nomad_bonus(final_routing) + cfg.gamma_diversity * compute_diversity_loss(exp_out) + cfg.lambda_sep * sep_loss + cfg.lambda_cons * cons_loss + cfg.lambda_load * compute_load_balancing_loss(final_routing) - dwell_reg.compute(final_routing, tau_dynamic=dyn_tau if cfg.use_dynamic_tau else float(cfg.tau_k_min)))
                    loss.backward(); opt.step()
                    prev_yb = yb.detach()

                seq_mse, dyn = eval_nomadic_seq(model, X_te, Y_te, R_te, tags_te, cfg, use_policy=False)
                mse_log.append(seq_mse); dyn_log.append(dyn)
            all_results[variant][seed] = {'seq_mse_log': mse_log, 'dyn_log': dyn_log}

        elif variant == 'Nomadic_Full':
            _, mse_log, dyn_log = train_nomadic_full(cfg, X_tr, Y_tr, R_tr, X_te, Y_te, R_te, tags_te)
            all_results[variant][seed] = {'seq_mse_log': mse_log, 'dyn_log': dyn_log}

print('\n=== All experiments complete ===')

# ============================================================
# STEP 8: 결과 집계
# ============================================================
rows = []
for variant in variant_cfgs:
    mse_vals, dh_vals, stable_vals, trans_vals = [], [], [], []
    for seed in SEEDS:
        r = all_results[variant][seed]
        mse_vals.append(r['seq_mse_log'][-1])
        if r['dyn_log'] is not None:
            d = r['dyn_log'][-1]
            sh = d['stable_entropy_mean']; th = d['transition_entropy_mean']
            stable_vals.append(sh); trans_vals.append(th)
            dh_vals.append(th - sh if not (math.isnan(th) or math.isnan(sh)) else float('nan'))
        else:
            stable_vals.append(float('nan')); trans_vals.append(float('nan')); dh_vals.append(float('nan'))
    rows.append({'Variant': variant, 'hidden_dim': variant_cfgs[variant]['hidden_dim'], 'Seq MSE mean': np.nanmean(mse_vals), 'ΔH mean': np.nanmean(dh_vals), 'Stable Ent': np.nanmean(stable_vals), 'Trans Ent': np.nanmean(trans_vals)})

df = pd.DataFrame(rows)
print('\n' + '='*80)
print('PARAMETER-MATCHED BASELINE EXPERIMENT: Final Results (Lagged v3)')
print('='*80)
print(df.to_string(float_format=lambda x: f'{x:.4f}', index=False))