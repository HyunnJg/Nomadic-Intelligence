# -*- coding: utf-8 -*-
"""
Nomadic Intelligence — Scenario D (v4) Lagged
Φ Variant Comparison: JSD_v2 포함 & 인과율(k-1 시차 기동) 강제 버전
"""

import os
import random
import math
from abc import ABC, abstractmethod
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
    phi_scale_explain: float = 2.0
    phi_scale_gap: float = 1.0

    temp_stable: float = 0.30
    temp_transition: float = 1.00

    use_hard_switch: bool = True
    phi_hard_threshold: float = 0.35

    policy_hidden_dim: int = 64
    policy_mix_weight: float = 0.25
    policy_weight_stay: float = 0.20
    policy_weight_target: float = 0.20
    policy_weight_mode: float = 0.10
    policy_switch_threshold: float = 0.50

PHI_VARIANTS = ['Phi_EMA', 'Phi_JSD', 'Phi_KL', 'Phi_Switch', 'Phi_JSD_v2']
SEEDS = [42, 123, 456]

# ============================================================
# STEP 2: Data Generation
# ============================================================
REGIME_TO_ID = {'A': 0, 'B': 1, 'C': 2}
ID_TO_REGIME = {0: 'A', 1: 'B', 2: 'C'}
REGIME_ORDER = ['A', 'B', 'C']

def sample_regime_x(regime: str, n: int, std: float, device: str = 'cpu') -> torch.Tensor:
    noise = std * torch.randn(n, 2, device=device)
    centers = {'A': (2.5, 2.5), 'B': (-2.5, -2.5), 'C': (2.5, -2.5)}
    return noise + torch.tensor(centers[regime], device=device)

def regime_function(x: torch.Tensor, regime: str) -> torch.Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    if regime == 'A': y = x1 + x2
    elif regime == 'B': y = x1 - x2
    elif regime == 'C': y = -x1 + 0.5 * x2
    else: raise ValueError(regime)
    return y.unsqueeze(-1)

def generate_phase_sequence(cfg: Config, cycles: int, device: str = 'cpu'):
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

def iterate_sequence_minibatches(X, Y, R, batch_size: int):
    n = X.size(0)
    for start in range(0, n, batch_size):
        yield X[start:min(start + batch_size, n)], Y[start:min(start + batch_size, n)], R[start:min(start + batch_size, n)]

# ============================================================
# STEP 3: Model Definitions
# ============================================================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.net(x)

class GateNet(nn.Module):
    def __init__(self, input_dim, gate_hidden_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim + 2, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, num_experts))
    def forward(self, x, delta_hybrid, delta_err, temperature):
        logits = self.net(torch.cat([x, delta_hybrid, delta_err], dim=-1))
        return F.softmax(logits / temperature, dim=-1), logits

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim + 5, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.stay_switch_head = nn.Linear(hidden_dim, 2)
        self.target_head = nn.Linear(hidden_dim, num_experts)
        self.mode_head = nn.Linear(hidden_dim, 2)
    def forward(self, policy_input):
        h = self.shared(policy_input)
        return F.softmax(self.stay_switch_head(h), dim=-1), F.softmax(self.target_head(h), dim=-1), F.softmax(self.mode_head(h), dim=-1)

class NomadicMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, gate_hidden_dim, policy_hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GateNet(input_dim, gate_hidden_dim, num_experts)
        self.policy = PolicyNet(input_dim, policy_hidden_dim, num_experts)
    def forward(self, x, delta_hybrid, delta_err, temperature, hard=False):
        gate_probs, gate_logits = self.gate(x, delta_hybrid, delta_err, temperature)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        routing = F.one_hot(gate_probs.argmax(dim=-1), self.num_experts).float() if hard else gate_probs
        return (routing.unsqueeze(-1) * expert_outputs).sum(dim=1), gate_probs, gate_logits, expert_outputs

# ============================================================
# STEP 4: PhiComputer (5 Variants)
# ============================================================
class PhiComputer(ABC):
    @abstractmethod
    def compute(self, delta_env, delta_err, explanation_error, best_expert_gap, gate_probs, stay_switch_probs=None, cfg=None): pass
    def reset(self): pass

class PhiEMA(PhiComputer):
    def compute(self, delta_env, delta_err, explanation_error, best_expert_gap, gate_probs, stay_switch_probs=None, cfg=None):
        return torch.tanh(cfg.phi_scale_env * torch.tensor(delta_env, device=explanation_error.device) + cfg.phi_scale_err * torch.tensor(delta_err, device=explanation_error.device) + cfg.phi_scale_explain * explanation_error.detach() + cfg.phi_scale_gap * best_expert_gap.detach())

class PhiJSD(PhiComputer):
    def __init__(self, scale=4.0, eps=1e-8): self.scale = scale; self.eps = eps; self.prev_gate = None
    def reset(self): self.prev_gate = None
    def compute(self, delta_env, delta_err, explanation_error, best_expert_gap, gate_probs, stay_switch_probs=None, cfg=None):
        curr = gate_probs.mean(dim=0)
        if self.prev_gate is None: self.prev_gate = curr.detach(); return torch.tensor(0.0, device=gate_probs.device)
        P, Q = curr, self.prev_gate.detach()
        M = 0.5 * (P + Q)
        jsd_norm = (0.5 * (P * (torch.log(P + self.eps) - torch.log(M + self.eps))).sum() + 0.5 * (Q * (torch.log(Q + self.eps) - torch.log(M + self.eps))).sum()) / math.log(2)
        self.prev_gate = curr.detach()
        return torch.tanh(self.scale * jsd_norm)

class PhiKL(PhiComputer):
    def __init__(self, scale=3.0, eps=1e-8): self.scale = scale; self.eps = eps; self.prev_gate = None
    def reset(self): self.prev_gate = None
    def compute(self, delta_env, delta_err, explanation_error, best_expert_gap, gate_probs, stay_switch_probs=None, cfg=None):
        curr = gate_probs.mean(dim=0)
        if self.prev_gate is None: self.prev_gate = curr.detach(); return torch.tensor(0.0, device=gate_probs.device)
        kl = torch.clamp((curr * (torch.log(curr + self.eps) - torch.log(self.prev_gate.detach() + self.eps))).sum(), min=0.0)
        self.prev_gate = curr.detach()
        return torch.tanh(self.scale * kl)

class PhiSwitch(PhiComputer):
    def __init__(self, warm_start_phi=0.3): self.warm_start_phi = warm_start_phi
    def compute(self, delta_env, delta_err, explanation_error, best_expert_gap, gate_probs, stay_switch_probs=None, cfg=None):
        if stay_switch_probs is None: return torch.tensor(self.warm_start_phi, device=gate_probs.device)
        return stay_switch_probs[:, 1].mean().detach()

class PhiJSD_v2(PhiComputer):
    def __init__(self, s_div=4.0, s_ema=3.0, ema_decay=0.85, eps=1e-8):
        self.s_div = s_div; self.s_ema = s_ema; self.ema_decay = ema_decay; self.eps = eps; self.ema_mean_jsd = None
    def reset(self): self.ema_mean_jsd = None
    def _per_sample_jsd(self, gate_probs):
        batch_mean = gate_probs.mean(dim=0, keepdim=True)
        M = 0.5 * (gate_probs + batch_mean.expand_as(gate_probs))
        kl_pm = (gate_probs * (torch.log(gate_probs + self.eps) - torch.log(M + self.eps))).sum(dim=-1)
        kl_qm = (batch_mean.expand_as(gate_probs) * (torch.log(batch_mean.expand_as(gate_probs) + self.eps) - torch.log(M + self.eps))).sum(dim=-1)
        return (0.5 * kl_pm + 0.5 * kl_qm).clamp(min=0.0) / math.log(2)
    def compute(self, delta_env, delta_err, explanation_error, best_expert_gap, gate_probs, stay_switch_probs=None, cfg=None):
        per_jsd = self._per_sample_jsd(gate_probs)
        std_term, mean_term = per_jsd.std().detach(), per_jsd.mean().detach()
        if self.ema_mean_jsd is None: self.ema_mean_jsd = mean_term
        else: self.ema_mean_jsd = (self.ema_decay * self.ema_mean_jsd + (1.0 - self.ema_decay) * mean_term)
        return torch.tanh(self.s_div * std_term + self.s_ema * self.ema_mean_jsd)

def make_phi(name: str) -> PhiComputer:
    return {'Phi_EMA': PhiEMA, 'Phi_JSD': PhiJSD, 'Phi_KL': PhiKL, 'Phi_Switch': PhiSwitch, 'Phi_JSD_v2': PhiJSD_v2}[name]()

# ============================================================
# STEP 5: Utilities (Lagged에 맞춰 GPU 최적화 유지)
# ============================================================
class HybridDeltaTracker:
    def __init__(self, cfg: Config, device: str):
        self.cfg = cfg; self.device = device
        self.delta_env_buf = torch.zeros(cfg.tau_var_window, device=device)
        self.reset()
    def reset(self):
        self.prev_x_mean = self.err_ema = self.err_baseline = None
        self.delta_env_buf.zero_(); self.buf_fill = 0
    def compute(self, x: torch.Tensor, batch_mse: torch.Tensor):
        x_mean = x.mean(dim=0, keepdim=True)
        delta_env_scalar = torch.tensor(0.0, device=self.device) if self.prev_x_mean is None else torch.norm(x_mean - self.prev_x_mean, p=2)
        batch_err = batch_mse.detach()
        if self.err_ema is None:
            self.err_ema = batch_err; self.err_baseline = batch_err; delta_err_scalar = torch.tensor(0.0, device=self.device)
        else:
            self.err_ema = self.cfg.ema_decay * self.err_ema + (1 - self.cfg.ema_decay) * batch_err
            self.err_baseline = self.cfg.err_baseline_momentum * self.err_baseline + (1 - self.cfg.err_baseline_momentum) * self.err_ema
            delta_err_scalar = torch.relu(self.err_ema - self.err_baseline)
        dh = float(torch.tanh(self.cfg.w_env * delta_env_scalar + self.cfg.w_err * delta_err_scalar).item())
        self.prev_x_mean = x_mean.detach()
        self.delta_env_buf = torch.roll(self.delta_env_buf, -1)
        self.delta_env_buf[-1] = delta_env_scalar.detach()
        self.buf_fill = min(self.buf_fill + 1, self.cfg.tau_var_window)
        sigma2 = float(self.delta_env_buf[-self.buf_fill:].var(unbiased=False).item()) if self.buf_fill >= 2 else 0.0
        dyn_tau = float(max(self.cfg.tau_min, min(self.cfg.tau_max, self.cfg.tau_min + (self.cfg.tau_max - self.cfg.tau_min) / (1.0 + self.cfg.tau_var_scale * sigma2))))
        return torch.full((x.size(0), 1), dh, device=self.device), delta_err_scalar.detach().view(1, 1).expand(x.size(0), 1), float(delta_env_scalar.item()), float(delta_err_scalar.item()), dh, sigma2, dyn_tau

class DwellTimeRegularizer:
    def __init__(self, tau_k_min=3, penalty=0.05): self.tau_k_min = tau_k_min; self.penalty = penalty; self.reset()
    def reset(self): self.current_expert = None; self.dwell_count = 0
    def compute(self, gate_probs, tau_dynamic=None):
        dominant = int(torch.bincount(gate_probs.argmax(dim=-1), minlength=gate_probs.size(-1)).argmax().item())
        if dominant == self.current_expert: self.dwell_count += 1
        else: self.current_expert = dominant; self.dwell_count = 1
        entropy = -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=-1).mean()
        tau_cap = float(tau_dynamic if tau_dynamic is not None else self.tau_k_min)
        if self.dwell_count <= tau_cap: return -self.penalty * entropy
        else: return min(float(self.dwell_count - tau_cap) * self.penalty, self.penalty * 10) * entropy

def gate_entropy(gate_probs): return -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=-1)
def compute_load_balancing_loss(gp): return gp.size(-1) * (torch.bincount(gp.argmax(dim=-1), minlength=gp.size(-1)).float() / gp.size(0) * gp.mean(dim=0)).sum()
def compute_dogma_penalty(gp): return torch.sum(gp.mean(dim=0) ** 2) - 1.0 / gp.size(1)
def compute_nomad_bonus(gp): return -(gp * (gp + 1e-8).log()).sum(dim=-1).mean()
def compute_diversity_loss(eo):
    if eo.size(1) < 2: return eo.new_zeros(1).squeeze()
    idx_i, idx_j = zip(*[(i, j) for i in range(eo.size(1)) for j in range(i+1, eo.size(1))])
    return F.cosine_similarity(eo[:, idx_i, :], eo[:, idx_j, :], dim=-1).mean()
def compute_explanation_signals(y_true, y_hat, eo, gp):
    pe = ((eo - y_true.unsqueeze(1)) ** 2).mean(dim=-1)
    return F.mse_loss(y_hat, y_true), torch.relu(pe.gather(1, gp.argmax(dim=-1).unsqueeze(1)).mean() - pe.min(dim=1).values.mean())
def compute_regime_gate_stats(gp, rb, num_regimes=3):
    vm = []; lc = torch.tensor(0.0, device=gp.device); cnt = 0
    for rid in range(num_regimes):
        mask = rb == rid
        if mask.sum() > 0:
            ur = gp[mask].mean(dim=0)
            vm.append(ur); lc = lc + ((gp[mask] - ur.unsqueeze(0)) ** 2).sum(dim=-1).mean(); cnt += 1
    if cnt > 0: lc = lc / cnt
    if len(vm) < 2: return torch.tensor(0.0, device=gp.device), lc
    return -torch.stack([torch.norm(vm[i] - vm[j], p=2) for i in range(len(vm)) for j in range(i + 1, len(vm))]).mean(), lc
def regimewise_usage(gp, rb, K):
    usage = {}; top1 = gp.argmax(dim=-1)
    for rid in range(3):
        mask = rb == rid
        usage[ID_TO_REGIME[rid]] = (torch.bincount(top1[mask], minlength=K).float() / mask.sum().clamp_min(1.0)).cpu().numpy() if mask.sum() > 0 else np.zeros(K)
    return usage
def compute_switch_latency(rs, top1, r2e):
    lats = []; pr = rs[0] if rs else None
    for t in range(1, len(rs)):
        cr = rs[t]
        if cr != pr and r2e.get(cr) is not None:
            for k in range(t, len(top1)):
                if int(top1[k]) == int(r2e[cr]): lats.append(k - t); break
        pr = cr
    return lats
def build_policy_input(xb, dh_t, de_t, phi, s2, dtau):
    return torch.cat([xb.mean(dim=0, keepdim=True).expand(xb.size(0), -1), dh_t, de_t, torch.full((xb.size(0), 1), float(phi.mean().item()), device=xb.device), torch.tanh(torch.tensor(s2 * 10.0, device=xb.device)).view(1, 1).expand(xb.size(0), 1), torch.tanh(torch.tensor((dtau - 5.0) / 5.0, device=xb.device)).view(1, 1).expand(xb.size(0), 1)], dim=-1)
def build_policy_targets(yb, eo, phi, s2, dtau, th):
    pv = float(phi.mean().item())
    return (1 if (pv > th) or (s2 > 0.05) else 0), ((eo - yb.unsqueeze(1)) ** 2).mean(dim=-1).mean(dim=0).argmin().long(), (1 if (pv <= th) and (dtau >= 5.5) else 0)

# ============================================================
# STEP 6: 통합 학습 및 평가 함수 (Lagged v3 강제)
# ============================================================
def evaluate_sequence(model, X, Y, R, phase_tags, cfg, phi_computer, phi_name):
    model.eval()
    phi_computer.reset()
    tracker = HybridDeltaTracker(cfg=cfg, device=cfg.device)
    
    all_y, all_gate, batch_tags, batch_ents, batch_top1 = [], [], [], [], []
    prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

    with torch.no_grad():
        for batch_idx, (xb, yb, rb) in enumerate(iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)):
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            z = torch.zeros((xb.size(0), 1), device=cfg.device)
            # [핵심] 시차 기동
            warm_mse = F.mse_loss(model(xb, z, z, cfg.temperature)[0], prev_yb)
            dh_t, de_t, de, derr, dh, s2, dtau = tracker.compute(xb, warm_mse)

            probe_y, probe_gate, _, probe_exp = model(xb, dh_t, de_t, cfg.temperature)
            # [핵심] 시차 성찰
            expl_err, gap = compute_explanation_signals(prev_yb, probe_y, probe_exp, probe_gate)

            phi_prov = phi_computer.compute(de, derr, expl_err, gap, probe_gate, None, cfg)
            if phi_name == 'Phi_Switch':
                ss_probs, _, _ = model.policy(build_policy_input(xb, dh_t, de_t, phi_prov, s2, dtau))
                phi_signal = phi_computer.compute(de, derr, expl_err, gap, probe_gate, ss_probs, cfg)
            else:
                phi_signal = phi_prov

            temp_now = cfg.temp_stable + (cfg.temp_transition - cfg.temp_stable) * float(phi_signal.item() if phi_signal.dim() == 0 else phi_signal.mean().item())

            y_hat, gate_probs, _, _ = model(xb, dh_t, de_t, temp_now)
            all_y.append(y_hat); all_gate.append(gate_probs)

            batch_tags.append(phase_tags[batch_idx * cfg.phase_batch_size])
            batch_ents.append(gate_entropy(gate_probs).mean().item())
            batch_top1.append(int(torch.bincount(gate_probs.argmax(dim=-1), minlength=cfg.num_experts).argmax().item()))

            prev_yb = yb.detach()

    Y_hat = torch.cat(all_y); G = torch.cat(all_gate)
    seq_mse = F.mse_loss(Y_hat, Y).item()
    sh = [e for t, e in zip(batch_tags, batch_ents) if t.startswith('stable_')]
    th = [e for t, e in zip(batch_tags, batch_ents) if t.startswith('transition_')]
    lats = compute_switch_latency([ID_TO_REGIME[int(rb[0].item())] for _, _, rb in iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)], np.array(batch_top1), {r: int(np.argmax(regimewise_usage(G, R, cfg.num_experts)[r])) for r in ['A', 'B', 'C']})

    return seq_mse, float(np.mean(sh)) if sh else float('nan'), float(np.mean(th)) if th else float('nan'), float(np.mean(lats)) if lats else float('nan')


def train_nomadic_phi_variant(cfg, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test, phi_computer, phi_name, verbose=True):
    model = NomadicMoE(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_experts, cfg.gate_hidden_dim, cfg.policy_hidden_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    logs = {'train_mse': [], 'train_total': [], 'test_seq_mse': [], 'test_stable_entropy': [], 'test_transition_entropy': [], 'test_switch_latency': [], 'phi_values': [], 'policy_switch_rate': []}
    tracker = HybridDeltaTracker(cfg=cfg, device=cfg.device)
    dwell_reg = DwellTimeRegularizer(tau_k_min=cfg.tau_k_min, penalty=cfg.tau_k_penalty)

    for epoch in range(cfg.epochs):
        model.train()
        phi_computer.reset(); tracker.reset(); dwell_reg.reset()
        epoch_mse = epoch_total = epoch_switch_rate = 0.0; epoch_phi_vals = []; n_batches = 0

        prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

        for xb, yb, rb in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            optimizer.zero_grad()
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            with torch.no_grad():
                z = torch.zeros((xb.size(0), 1), device=cfg.device)
                warm_mse = F.mse_loss(model(xb, z, z, cfg.temperature)[0], prev_yb) # Lagged
            dh_t, de_t, de, derr, dh, s2, dtau = tracker.compute(xb, warm_mse)

            with torch.no_grad():
                probe_y, probe_gate, _, probe_exp = model(xb, dh_t, de_t, cfg.temperature)
            
            expl_err, gap = compute_explanation_signals(prev_yb, probe_y, probe_exp, probe_gate) # Lagged
            phi_prov = phi_computer.compute(de, derr, expl_err, gap, probe_gate, None, cfg)

            policy_input = build_policy_input(xb, dh_t, de_t, phi_prov, s2, dtau)
            stay_sw, tgt_probs, mode_probs = model.policy(policy_input)

            if phi_name == 'Phi_Switch': phi_signal = phi_computer.compute(de, derr, expl_err, gap, probe_gate, stay_sw, cfg)
            else: phi_signal = phi_prov

            phi_val = float(phi_signal.item() if phi_signal.dim() == 0 else phi_signal.mean().item())
            epoch_phi_vals.append(phi_val)

            temp_now = cfg.temp_stable + (cfg.temp_transition - cfg.temp_stable) * phi_val
            y_hat, gate_probs, _, exp_out = model(xb, dh_t, de_t, temp_now)

            eff_mix = cfg.policy_mix_weight * float(stay_sw[:, 1].mean().item())
            tgt_ste = (F.one_hot(torch.argmax(tgt_probs.mean(dim=0), dim=-1), cfg.num_experts).float().unsqueeze(0).expand(xb.size(0), -1) - gate_probs).detach() + gate_probs
            mixed = (1.0 - eff_mix) * gate_probs + eff_mix * tgt_ste
            final_routing = F.one_hot(mixed.argmax(dim=-1), cfg.num_experts).float() if cfg.use_hard_switch and (mode_probs[:, 1].mean().item() > 0.5) and not (dh > cfg.phi_hard_threshold) else mixed
            y_hat = (final_routing.unsqueeze(-1) * exp_out).sum(dim=1)

            # Gradient Flow
            mse_loss = F.mse_loss(y_hat, yb)
            _, gap_loss = compute_explanation_signals(yb, y_hat, exp_out, final_routing)
            l_sep, l_cons = compute_regime_gate_stats(final_routing, rb)

            sw_lbl, tgt_lbl, mode_lbl = build_policy_targets(prev_yb, probe_exp, phi_signal, s2, dtau, cfg.policy_switch_threshold) # Lagged

            loss = (mse_loss
                  + cfg.beta_phi * ((phi_signal if phi_signal.dim() > 0 else phi_signal.unsqueeze(0)).detach().mean() * gap_loss)
                  + cfg.alpha_dogma * compute_dogma_penalty(final_routing) - cfg.beta_nomad * compute_nomad_bonus(final_routing)
                  + cfg.gamma_diversity * compute_diversity_loss(exp_out) + cfg.lambda_sep * l_sep + cfg.lambda_cons * l_cons + cfg.lambda_load * compute_load_balancing_loss(final_routing)
                  + cfg.policy_weight_stay * F.nll_loss(torch.log(stay_sw + 1e-8), torch.full((xb.size(0),), sw_lbl, dtype=torch.long, device=cfg.device))
                  + cfg.policy_weight_target * F.nll_loss(torch.log(tgt_probs + 1e-8), torch.full((xb.size(0),), int(tgt_lbl.item()), dtype=torch.long, device=cfg.device))
                  + cfg.policy_weight_mode * F.nll_loss(torch.log(mode_probs + 1e-8), torch.full((xb.size(0),), mode_lbl, dtype=torch.long, device=cfg.device))
                  - dwell_reg.compute(final_routing, tau_dynamic=dtau if cfg.use_dynamic_tau else float(cfg.tau_k_min)))
            
            loss.backward(); optimizer.step()
            epoch_mse += mse_loss.item(); epoch_total += loss.item(); epoch_switch_rate += float(stay_sw[:, 1].mean().item()); n_batches += 1
            prev_yb = yb.detach()

        logs['train_mse'].append(epoch_mse / max(n_batches, 1))
        logs['train_total'].append(epoch_total / max(n_batches, 1))
        logs['phi_values'].append(float(np.mean(epoch_phi_vals)) if epoch_phi_vals else 0.0)
        logs['policy_switch_rate'].append(epoch_switch_rate / max(n_batches, 1))

        seq_mse, stable_ent, trans_ent, sw_lat = evaluate_sequence(model, X_test, Y_test, R_test, phase_tags_test, cfg, phi_computer, phi_name)
        logs['test_seq_mse'].append(seq_mse); logs['test_stable_entropy'].append(stable_ent); logs['test_transition_entropy'].append(trans_ent); logs['test_switch_latency'].append(sw_lat)

        if verbose and ((epoch + 1) % 55 == 0 or epoch == 0):
            print(f'[{phi_name}] Ep {epoch+1:03d} | Train MSE: {logs["train_mse"][-1]:.4f} | Seq MSE: {seq_mse:.4f} | dH: {trans_ent - stable_ent:+.4f} | Lat: {sw_lat:.3f}')

    return model, logs

# ============================================================
# STEP 7: 실험 실행 및 결과 집계
# ============================================================
def main():
    import time
    datasets = {}
    print('Generating datasets...')
    for seed in SEEDS:
        set_seed(seed); cfg_tmp = Config(seed=seed, device=DEVICE)
        datasets[seed] = {'train': generate_phase_sequence(cfg_tmp, cfg_tmp.phase_train_cycles, cfg_tmp.device), 'test': generate_phase_sequence(cfg_tmp, cfg_tmp.phase_test_cycles, cfg_tmp.device)}
    print('Datasets ready.\n')

    all_results = {}
    total_start = time.time()

    for phi_name in PHI_VARIANTS:
        all_results[phi_name] = {}
        print(f'\n{"="*60}\n  Variant: {phi_name} (Lagged v3)\n{"="*60}')
        for seed in SEEDS:
            print(f'\n--- Seed {seed} ---')
            set_seed(seed); cfg = Config(seed=seed, device=DEVICE)
            phi_computer = make_phi(phi_name)
            t0 = time.time()
            _, logs = train_nomadic_phi_variant(cfg, *datasets[seed]['train'], *datasets[seed]['test'], phi_computer, phi_name)
            print(f'  → Seed {seed} done in {(time.time() - t0)/60:.1f} min')
            all_results[phi_name][seed] = logs

    print(f'\n\n✅ 모든 실험 완료. 총 소요시간: {(time.time() - total_start)/60:.1f} min')

    summary_rows = []
    for phi_name in PHI_VARIANTS:
        seed_mse, seed_dh, seed_lat = [], [], []
        for seed in SEEDS:
            l = all_results[phi_name][seed]
            seed_mse.append(l['test_seq_mse'][-1])
            seed_dh.append(l['test_transition_entropy'][-1] - l['test_stable_entropy'][-1])
            seed_lat.append(l['test_switch_latency'][-1] if not math.isnan(l['test_switch_latency'][-1]) else 0.0)
        summary_rows.append({'Phi Variant': phi_name, 'Seq MSE (mean)': np.mean(seed_mse), 'ΔH (mean)': np.mean(seed_dh), 'Switch Lat (mean)': np.mean(seed_lat)})

    df = pd.DataFrame(summary_rows).set_index('Phi Variant')
    print('\n' + '='*72)
    print('SCENARIO D: Φ Variant Comparison (Lagged v3, avg over 3 seeds)')
    print('='*72)
    print(df.to_string(float_format=lambda x: f'{x:.4f}'))
    
    # 디렉토리 생성 및 로컬 저장
    save_dir = './outputs_scenario_D_lagged'
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'results_summary.csv'))
    print(f'\nResults saved to: {save_dir}')

if __name__ == "__main__":
    main()