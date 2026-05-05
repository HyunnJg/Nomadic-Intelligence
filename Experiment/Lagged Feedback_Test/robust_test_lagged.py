import os
import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    seed: int = 42
    device: str = "cpu"
    input_dim: int = 2
    output_dim: int = 1
    overlap_std: float = 0.9

    center_A: Tuple[float, float] = (2.5, 2.5)
    center_B: Tuple[float, float] = (-2.5, -2.5)
    center_C: Tuple[float, float] = (2.5, -2.5)

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
    beta_phi: float = 0.05
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

    save_dir: str = "outputs_transition"

    # [핵심] Robustness 실험 특화 세팅
    num_regimes: int = 4           # A, B, C, D 4개 레짐
    random_regime_order: bool = True  # 무작위 전환

def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}

def build_config_from_yaml(yaml_dict: dict) -> Config:
    # (기존과 동일하므로 Config 기본값으로 덮어씀. 실제 스크립트에선 원본 사용 무방)
    return Config()

# ============================================================
# Data generation (4 Regimes)
# ============================================================
REGIME_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}
ID_TO_REGIME = {0: "A", 1: "B", 2: "C", 3: "D"}
REGIME_ORDER = ["A", "B", "C", "D"]

def sample_regime_x(regime: str, n: int, std: float, device: str = "cpu") -> torch.Tensor:
    noise = std * torch.randn(n, 2, device=device)
    if regime == "A": center = torch.tensor([2.5, 2.5], device=device)
    elif regime == "B": center = torch.tensor([-2.5, -2.5], device=device)
    elif regime == "C": center = torch.tensor([2.5, -2.5], device=device)
    elif regime == "D": center = torch.tensor([-2.5, 2.5], device=device)
    else: raise ValueError(f"Unknown regime: {regime}")
    return noise + center

def regime_function(x: torch.Tensor, regime: str) -> torch.Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    if regime == "A": y = x1 + x2
    elif regime == "B": y = x1 - x2
    elif regime == "C": y = -x1 + 0.5 * x2
    elif regime == "D": y = -x1 - x2
    else: raise ValueError(f"Unknown regime: {regime}")
    return y.unsqueeze(-1)

def generate_phase_sequence(cfg: Config, cycles: int, device: str = "cpu"):
    active_regimes = REGIME_ORDER[:cfg.num_regimes]
    xs, ys, rs, phase_tags = [], [], [], []
    rng = np.random.default_rng(cfg.seed + 99)
    prev_last_regime = None

    for cycle_idx in range(cycles):
        if cfg.random_regime_order:
            order = list(active_regimes)
            for _ in range(200):
                rng.shuffle(order)
                if prev_last_regime is None or order[0] != prev_last_regime: break
            cycle_regimes = order
        else:
            cycle_regimes = list(active_regimes)

        prev_last_regime = cycle_regimes[-1]

        for i in range(len(cycle_regimes)):
            curr_r = cycle_regimes[i]
            next_r = cycle_regimes[(i + 1) % len(cycle_regimes)]

            x_stable = sample_regime_x(curr_r, cfg.phase_batch_size, std=cfg.overlap_std, device=device)
            y_stable = regime_function(x_stable, curr_r)
            r_stable = torch.full((cfg.phase_batch_size,), REGIME_TO_ID[curr_r], dtype=torch.long, device=device)

            xs.append(x_stable); ys.append(y_stable); rs.append(r_stable)
            phase_tags.extend([f"stable_{curr_r}"] * cfg.phase_batch_size)

            for step in range(cfg.transition_steps):
                alpha = (step + 1) / cfg.transition_steps
                x_a = sample_regime_x(curr_r, cfg.phase_batch_size, std=cfg.overlap_std, device=device)
                x_b = sample_regime_x(next_r, cfg.phase_batch_size, std=cfg.overlap_std, device=device)
                x_mix = (1.0 - alpha) * x_a + alpha * x_b

                y_mix = (1.0 - alpha) * regime_function(x_mix, curr_r) + alpha * regime_function(x_mix, next_r)
                dominant = curr_r if alpha < 0.5 else next_r
                r_mix = torch.full((cfg.phase_batch_size,), REGIME_TO_ID[dominant], dtype=torch.long, device=device)

                xs.append(x_mix); ys.append(y_mix); rs.append(r_mix)
                phase_tags.extend([f"transition_{curr_r}_to_{next_r}"] * cfg.phase_batch_size)

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(rs, dim=0), phase_tags

def iterate_sequence_minibatches(X, Y, R, batch_size):
    n = X.size(0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield X[start:end], Y[start:end], R[start:end]

# ============================================================
# Models & Utilities (생략 없이 원본 유지)
# ============================================================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.net(x)

class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return self.net(x)

class GateNet(nn.Module):
    def __init__(self, input_dim: int, gate_hidden_dim: int, num_experts: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim + 2, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, num_experts))
    def forward(self, x, dh, de, temp):
        logits = self.net(torch.cat([x, dh, de], dim=-1))
        return F.softmax(logits / temp, dim=-1), logits

class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim + 5, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.stay_switch_head = nn.Linear(hidden_dim, 2)
        self.target_head = nn.Linear(hidden_dim, num_experts)
        self.mode_head = nn.Linear(hidden_dim, 2)
    def forward(self, pi):
        h = self.shared(pi)
        return F.softmax(self.stay_switch_head(h), dim=-1), F.softmax(self.target_head(h), dim=-1), F.softmax(self.mode_head(h), dim=-1)

class NomadicMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, gate_hidden_dim, policy_hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GateNet(input_dim, gate_hidden_dim, num_experts)
        self.policy = PolicyNet(input_dim, policy_hidden_dim, num_experts)
    def forward(self, x, dh, de, temp, hard=False):
        gp, gl = self.gate(x, dh, de, temp)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        r = F.one_hot(gp.argmax(dim=-1), num_classes=self.num_experts).float() if hard else gp
        return (r.unsqueeze(-1) * eo).sum(dim=1), gp, gl, eo

class StandardMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, gate_hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(input_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, num_experts))
    def forward(self, x, hard=False):
        logits = self.gate(x)
        gp = F.softmax(logits, dim=-1)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        r = F.one_hot(gp.argmax(dim=-1), num_classes=self.num_experts).float() if hard else gp
        return (r.unsqueeze(-1) * eo).sum(dim=1), gp, logits, eo

class HybridDeltaTracker:
    def __init__(self, ema_decay=0.8, err_baseline_momentum=0.85, w_env=1.0, w_err=2.0, device="cpu", tau_min=2.0, tau_max=8.0, tau_var_scale=6.0, tau_var_window=8):
        self.ema_decay = ema_decay; self.err_baseline_momentum = err_baseline_momentum
        self.w_env = w_env; self.w_err = w_err; self.device = device
        self.tau_min = tau_min; self.tau_max = tau_max; self.tau_var_scale = tau_var_scale; self.tau_var_window = tau_var_window
        self.recent_delta_env = deque(maxlen=tau_var_window)
        self.reset()
    def reset(self):
        self.prev_x_mean = None; self.err_ema = None; self.err_baseline = None
        self.recent_delta_env.clear()
        self.delta_env_history, self.delta_err_history, self.delta_hybrid_raw_history, self.delta_hybrid_history, self.sigma2_delta_history, self.dynamic_tau_history = [], [], [], [], [], []
    def compute(self, x, current_batch_mse):
        x_mean = x.mean(dim=0, keepdim=True)
        de_scalar = torch.tensor(0.0, device=self.device) if self.prev_x_mean is None else torch.norm(x_mean - self.prev_x_mean, p=2)
        batch_err = current_batch_mse.detach()
        if self.err_ema is None: self.err_ema = batch_err; self.err_baseline = batch_err; derr_scalar = torch.tensor(0.0, device=self.device)
        else:
            self.err_ema = self.ema_decay * self.err_ema + (1.0 - self.ema_decay) * batch_err
            self.err_baseline = self.err_baseline_momentum * self.err_baseline + (1.0 - self.err_baseline_momentum) * self.err_ema
            derr_scalar = torch.relu(self.err_ema - self.err_baseline)
        raw_hybrid = self.w_env * de_scalar + self.w_err * derr_scalar
        dh_scalar = torch.tanh(raw_hybrid)
        self.prev_x_mean = x_mean.detach()
        self.recent_delta_env.append(float(de_scalar.item()))
        s2 = float(np.var(self.recent_delta_env)) if len(self.recent_delta_env) >= 2 else 0.0
        dtau = self.tau_min + (self.tau_max - self.tau_min) / (1.0 + self.tau_var_scale * s2)
        dtau = float(np.clip(dtau, self.tau_min, self.tau_max))
        self.delta_hybrid_raw_history.append(float(raw_hybrid.item()))
        return torch.full((x.size(0), 1), float(dh_scalar.item()), device=self.device), float(de_scalar.item()), float(derr_scalar.item()), float(dh_scalar.item()), s2, dtau

def compute_load_balancing_loss(gp):
    K = gp.size(-1)
    return K * (torch.bincount(gp.argmax(-1), minlength=K).float() / gp.size(0) * gp.mean(0)).sum()

class DwellTimeRegularizer:
    def __init__(self, tau_k_min=3, penalty=0.05): self.tau_k_min = tau_k_min; self.penalty = penalty; self.reset()
    def reset(self): self.current_expert = None; self.dwell_count = 0
    def compute(self, gp, tau_dynamic=None):
        dom = int(torch.bincount(gp.argmax(-1), minlength=gp.size(-1)).argmax().item())
        if dom == self.current_expert: self.dwell_count += 1
        else: self.current_expert = dom; self.dwell_count = 1
        ent = -(gp * (gp + 1e-8).log()).sum(-1).mean()
        tc = float(self.tau_k_min if tau_dynamic is None else tau_dynamic)
        if self.dwell_count <= tc: return -self.penalty * ent
        else: return min((self.dwell_count - tc) * self.penalty, self.penalty * 10) * ent

def compute_diversity_loss(eo):
    K = eo.size(1)
    if K < 2: return torch.tensor(0.0, device=eo.device)
    ii, jj = zip(*[(i, j) for i in range(K) for j in range(i + 1, K)])
    return F.cosine_similarity(eo[:, ii, :], eo[:, jj, :], dim=-1).mean()

def compute_dogma_penalty(gp): return torch.sum(gp.mean(0) ** 2) - 1.0 / gp.size(1)
def compute_nomad_bonus(gp): return -(gp * (gp + 1e-8).log()).sum(-1).mean()
def compute_explanation_signals(yt, yh, eo, gp):
    pe = ((eo - yt.unsqueeze(1)) ** 2).mean(-1)
    return F.mse_loss(yh, yt), torch.relu(pe.gather(1, gp.argmax(-1).unsqueeze(1)).mean() - pe.min(1).values.mean())

def compute_phi_signal(de, derr, expl, gap, phi_scale_env=1.0, phi_scale_err=1.5, phi_scale_explain=2.0, phi_scale_gap=1.0):
    dev = expl.device
    return torch.tanh(phi_scale_env * torch.tensor(de, device=dev) + phi_scale_err * torch.tensor(derr, device=dev) + phi_scale_explain * expl.detach() + phi_scale_gap * gap.detach())

def compute_adaptive_temperature(phi, ts=0.3, tt=1.0): return ts + (tt - ts) * float(phi.mean().item())

def build_policy_input(xb, dh, derr, phi, s2, dtau):
    xs = xb.mean(0, keepdim=True).expand(xb.size(0), -1)
    return torch.cat([xs, dh, derr, torch.full((xb.size(0), 1), float(phi.mean().item()), device=xb.device), 
                      torch.full((xb.size(0), 1), float(np.tanh(s2 * 10.0)), device=xb.device), 
                      torch.full((xb.size(0), 1), float(np.tanh((dtau - 5.0) / 5.0)), device=xb.device)], dim=-1)

def build_policy_targets(yt, eo, phi, s2, dtau, th, ts=5.5, ss=0.05):
    pe = ((eo - yt.unsqueeze(1)) ** 2).mean(-1)
    pv = float(phi.mean().item())
    return 1 if (pv > th) or (s2 > ss) else 0, pe.mean(0).argmin().long(), 1 if (pv <= th) and (dtau >= ts) else 0

def gate_entropy(gp): return -(gp * (gp + 1e-8).log()).sum(-1)

def regimewise_usage(gp, rb, K, num_regimes=4):
    usage = {}
    for rid in range(num_regimes):
        mask = rb == rid
        usage[ID_TO_REGIME[rid]] = (torch.bincount(gp.argmax(-1)[mask], minlength=K).float() / max(1.0, mask.sum().item())).cpu().numpy() if mask.sum() > 0 else np.zeros(K)
    return usage

def compute_regime_gate_stats(gp, rb, num_regimes=4):
    vm, vn, lc = [], [], torch.tensor(0.0, device=gp.device)
    for rid in range(num_regimes):
        mask = rb == rid
        if mask.sum() > 0:
            ur = gp[mask].mean(0)
            vm.append(ur); vn.append(ID_TO_REGIME[rid])
            lc = lc + ((gp[mask] - ur.unsqueeze(0)) ** 2).sum(-1).mean()
    lc = lc / max(1, len(vm))
    if len(vm) < 2: return {}, torch.tensor(0.0, device=gp.device), lc, 0.0, {}
    pw = torch.stack([torch.norm(vm[i] - vm[j], p=2) for i in range(len(vm)) for j in range(i + 1, len(vm))])
    return {vn[i]: vm[i] for i in range(len(vm))}, -pw.mean(), lc, float(pw.mean().item()), {}

def mse_by_regime(yt, yp, rb, num_regimes=4):
    return {ID_TO_REGIME[rid]: F.mse_loss(yp[rb == rid], yt[rb == rid]).item() if (rb == rid).sum() > 0 else float("nan") for rid in range(num_regimes)}

def infer_regime_to_expert(usage, num_regimes=4): return {ID_TO_REGIME[rid]: int(np.argmax(usage[ID_TO_REGIME[rid]])) for rid in range(num_regimes) if ID_TO_REGIME[rid] in usage}

def compute_dwell_times(top1):
    if len(top1) == 0: return []
    dwells, cur, run = [], top1[0], 1
    for t in range(1, len(top1)):
        if top1[t] == cur: run += 1
        else: dwells.append(run); cur = top1[t]; run = 1
    dwells.append(run)
    return dwells

def compute_switch_latency(rs, top1, r2e):
    lats, pr = [], rs[0] if rs else None
    for t in range(1, len(rs)):
        cr = rs[t]
        if cr != pr and r2e.get(cr) is not None:
            for k in range(t, len(top1)):
                if int(top1[k]) == int(r2e[cr]): lats.append(k - t); break
        pr = cr
    return lats

# ============================================================
# [핵심 수정] 훈련 및 평가 함수에 Lagged (k-1) prev_yb 적용
# ============================================================

def evaluate_nomadic_sequence_dynamics(model: NomadicMoE, X, Y, R, phase_tags, cfg):
    model.eval()
    tracker = HybridDeltaTracker(ema_decay=cfg.ema_decay, err_baseline_momentum=cfg.err_baseline_momentum, w_env=cfg.w_env, w_err=cfg.w_err, device=cfg.device, tau_min=cfg.tau_min, tau_max=cfg.tau_max, tau_var_scale=cfg.tau_var_scale, tau_var_window=cfg.tau_var_window)
    all_y, all_gp, br, bpt, be, btop1, bs2, btau = [], [], [], [], [], [], [], []

    # [수정] 이전 정답 버퍼
    prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

    with torch.no_grad():
        for bi, (xb, yb, rb) in enumerate(iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)):
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            zd = torch.zeros((xb.size(0), 1), device=cfg.device)
            wy, _, _, _ = model(xb, zd, zd, cfg.temperature, hard=False)
            
            # [수정] yb -> prev_yb (시차 기동)
            wmse = F.mse_loss(wy, prev_yb)
            dh, de, derr, _, s2, dtau = tracker.compute(xb, wmse)
            det = torch.full((xb.size(0), 1), derr, device=cfg.device)

            py, pg, _, pe = model(xb, dh, det, cfg.temperature, hard=False)
            # [수정] yb -> prev_yb
            expl, gap = compute_explanation_signals(prev_yb, py, pe, pg)
            phi = compute_phi_signal(de, derr, expl, gap, cfg.phi_scale_env, cfg.phi_scale_err, cfg.phi_scale_explain, cfg.phi_scale_gap)
            
            pi = build_policy_input(xb, dh, det, phi, s2, dtau)
            sp, tp, mp = model.policy(pi)
            temp = compute_adaptive_temperature(phi, cfg.temp_stable, cfg.temp_transition)

            hm = bool(cfg.use_hard_switch and (mp[:, 1].mean().item() > 0.5) and not (float(dh.mean().item()) > cfg.phi_hard_threshold))
            yh, gp, _, eo = model(xb, dh, det, temp, hard=False)

            em = cfg.policy_mix_weight * float(sp[:, 1].mean().item())
            toh = F.one_hot(torch.argmax(tp.mean(0), dim=-1), cfg.num_experts).float().unsqueeze(0).expand(xb.size(0), -1)
            mx = (1.0 - em) * gp + em * ((toh - gp).detach() + gp)
            fr = F.one_hot(mx.argmax(-1), cfg.num_experts).float() if hm else mx
            yh = (fr.unsqueeze(-1) * eo).sum(1)

            all_y.append(yh); all_gp.append(fr)
            br.append(ID_TO_REGIME[int(rb[0].item())])
            bpt.append(phase_tags[bi * cfg.phase_batch_size])
            be.append(gate_entropy(fr).mean().item())
            btop1.append(int(torch.argmax(torch.bincount(fr.argmax(-1), minlength=cfg.num_experts)).item()))
            bs2.append(s2); btau.append(dtau)

            # [수정] 버퍼 업데이트
            prev_yb = yb.detach()

    Y_hat = torch.cat(all_y, dim=0); G = torch.cat(all_gp, dim=0)
    tmse = F.mse_loss(Y_hat, Y).item()
    usage = regimewise_usage(G, R, cfg.num_experts, cfg.num_regimes)
    r2e = infer_regime_to_expert(usage, cfg.num_regimes)
    lats = compute_switch_latency(br, np.array(btop1), r2e)
    sh = [e for t, e in zip(bpt, be) if t.startswith("stable_")]
    th = [e for t, e in zip(bpt, be) if t.startswith("transition_")]

    return tmse, usage, {"batch_regimes": br, "batch_phase_tags": bpt, "batch_entropies": be, "batch_top1": btop1, "switch_latencies": lats, "dwell_times": compute_dwell_times(np.array(btop1)), "mean_switch_latency": float(np.mean(lats)) if lats else float("nan"), "mean_dwell_time": float(np.mean(compute_dwell_times(np.array(btop1)))) if compute_dwell_times(np.array(btop1)) else float("nan"), "stable_entropy_mean": float(np.mean(sh)) if sh else float("nan"), "transition_entropy_mean": float(np.mean(th)) if th else float("nan"), "regime_to_expert": r2e, "sigma2_delta": bs2, "dynamic_tau": btau, "mean_dynamic_tau": float(np.mean(btau)) if btau else float("nan")}, Y_hat, G

def evaluate_nomadic_no_policy_sequence(model: NomadicMoE, X, Y, R, phase_tags, cfg):
    model.eval()
    tracker = HybridDeltaTracker(ema_decay=cfg.ema_decay, err_baseline_momentum=cfg.err_baseline_momentum, w_env=cfg.w_env, w_err=cfg.w_err, device=cfg.device, tau_min=cfg.tau_min, tau_max=cfg.tau_max, tau_var_scale=cfg.tau_var_scale, tau_var_window=cfg.tau_var_window)
    all_y, all_gp, bpt, be, btop1 = [], [], [], [], []
    prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

    with torch.no_grad():
        for bi, (xb, yb, rb) in enumerate(iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)):
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)
            zd = torch.zeros((xb.size(0), 1), device=cfg.device)
            wy, _, _, _ = model(xb, zd, zd, cfg.temperature, hard=False)
            wmse = F.mse_loss(wy, prev_yb)  # Lagged
            dh, de, derr, _, s2, dtau = tracker.compute(xb, wmse)
            det = torch.full((xb.size(0), 1), derr, device=cfg.device)
            py, pg, _, pe = model(xb, dh, det, cfg.temperature, hard=False)
            expl, gap = compute_explanation_signals(prev_yb, py, pe, pg) # Lagged
            phi = compute_phi_signal(de, derr, expl, gap, cfg.phi_scale_env, cfg.phi_scale_err, cfg.phi_scale_explain, cfg.phi_scale_gap)
            temp = compute_adaptive_temperature(phi, cfg.temp_stable, cfg.temp_transition)
            yh, gp, _, eo = model(xb, dh, det, temp, hard=False)
            
            all_y.append(yh); all_gp.append(gp); bpt.append(phase_tags[bi * cfg.phase_batch_size])
            be.append(gate_entropy(gp).mean().item())
            btop1.append(int(torch.argmax(torch.bincount(gp.argmax(-1), minlength=cfg.num_experts)).item()))
            prev_yb = yb.detach()

    Y_hat = torch.cat(all_y, dim=0); G = torch.cat(all_gp, dim=0)
    usage = regimewise_usage(G, R, cfg.num_experts, cfg.num_regimes)
    sh = [e for t, e in zip(bpt, be) if t.startswith("stable_")]
    th = [e for t, e in zip(bpt, be) if t.startswith("transition_")]
    lats = compute_switch_latency([ID_TO_REGIME[int(rb[0].item())] for _, _, rb in iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)], np.array(btop1), infer_regime_to_expert(usage, cfg.num_regimes))
    
    return F.mse_loss(Y_hat, Y).item(), usage, {"stable_entropy_mean": float(np.mean(sh)) if sh else float("nan"), "transition_entropy_mean": float(np.mean(th)) if th else float("nan"), "mean_switch_latency": float(np.mean(lats)) if lats else float("nan")}

def train_nomadic_no_policy(cfg: Config, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test):
    model = NomadicMoE(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_experts, cfg.gate_hidden_dim, cfg.policy_hidden_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logs = {"train_losses": [], "test_mse_static": [], "test_mse_sequence": [], "test_mean_gate_distance": [], "test_entropy": []}

    for epoch in range(cfg.epochs):
        model.train()
        tracker = HybridDeltaTracker(ema_decay=cfg.ema_decay, err_baseline_momentum=cfg.err_baseline_momentum, w_env=cfg.w_env, w_err=cfg.w_err, device=cfg.device, tau_min=cfg.tau_min, tau_max=cfg.tau_max, tau_var_scale=cfg.tau_var_scale, tau_var_window=cfg.tau_var_window)
        dreg = DwellTimeRegularizer(cfg.tau_k_min, cfg.tau_k_penalty)
        eloss, nb = 0.0, 0
        prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

        for xb, yb, rb in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            optimizer.zero_grad()
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)
            with torch.no_grad():
                zd = torch.zeros((xb.size(0), 1), device=cfg.device)
                wy, _, _, _ = model(xb, zd, zd, cfg.temperature)
                wmse = F.mse_loss(wy, prev_yb) # Lagged
            dh, de, derr, _, s2, dtau = tracker.compute(xb, wmse)
            det = torch.full((xb.size(0), 1), derr, device=cfg.device)
            with torch.no_grad():
                py, pg, _, pe = model(xb, dh, det, cfg.temperature, hard=False)
            expl, gap = compute_explanation_signals(prev_yb, py, pe, pg) # Lagged
            phi = compute_phi_signal(de, derr, expl, gap, cfg.phi_scale_env, cfg.phi_scale_err, cfg.phi_scale_explain, cfg.phi_scale_gap)
            temp = compute_adaptive_temperature(phi, cfg.temp_stable, cfg.temp_transition)
            
            yh, gp, _, eo = model(xb, dh, det, temp, hard=False)
            yh = (gp.unsqueeze(-1) * eo).sum(1)
            
            mse_loss = F.mse_loss(yh, yb) # Backward pass uses current yb
            _, gap_loss = compute_explanation_signals(yb, yh, eo, gp)
            tl = (mse_loss + cfg.beta_phi * (phi.detach() * gap_loss) + cfg.alpha_dogma * compute_dogma_penalty(gp) 
                  - cfg.beta_nomad * compute_nomad_bonus(gp) + cfg.gamma_diversity * compute_diversity_loss(eo) 
                  + cfg.lambda_sep * compute_regime_gate_stats(gp, rb, cfg.num_regimes)[1] + cfg.lambda_cons * compute_regime_gate_stats(gp, rb, cfg.num_regimes)[2] 
                  + cfg.lambda_load * compute_load_balancing_loss(gp) - dreg.compute(gp, dtau if cfg.use_dynamic_tau else cfg.tau_k_min))
            tl.backward(); optimizer.step()
            eloss += tl.item(); nb += 1
            prev_yb = yb.detach()

        logs["train_losses"].append(eloss / max(nb, 1))
        tmse, _, _, tgd, _, te, _, _, _ = evaluate_nomadic_static_full(model, X_test, Y_test, R_test, cfg)
        smse, _, _ = evaluate_nomadic_no_policy_sequence(model, X_test, Y_test, R_test, phase_tags_test, cfg)
        logs["test_mse_static"].append(tmse); logs["test_mse_sequence"].append(smse); logs["test_mean_gate_distance"].append(tgd); logs["test_entropy"].append(te)
        if (epoch + 1) % 55 == 0 or epoch == 0:
            print(f"[NoPolicy Lagged] Ep {epoch+1:03d}/{cfg.epochs} | Train Loss: {logs['train_losses'][-1]:.4f} | Seq MSE: {smse:.4f}")
    return model, logs

def train_nomadic(cfg: Config, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test):
    model = NomadicMoE(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_experts, cfg.gate_hidden_dim, cfg.policy_hidden_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logs = {"train_mse_losses": [], "test_mse_static": [], "test_mse_sequence": [], "test_switch_latency": [], "test_transition_entropy": [], "test_stable_entropy": []}

    for epoch in range(cfg.epochs):
        model.train()
        tracker = HybridDeltaTracker(ema_decay=cfg.ema_decay, err_baseline_momentum=cfg.err_baseline_momentum, w_env=cfg.w_env, w_err=cfg.w_err, device=cfg.device, tau_min=cfg.tau_min, tau_max=cfg.tau_max, tau_var_scale=cfg.tau_var_scale, tau_var_window=cfg.tau_var_window)
        dreg = DwellTimeRegularizer(cfg.tau_k_min, cfg.tau_k_penalty)
        emse, nb = 0.0, 0
        prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

        for xb, yb, rb in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            optimizer.zero_grad()
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            with torch.no_grad():
                zd = torch.zeros((xb.size(0), 1), device=cfg.device)
                wy, _, _, _ = model(xb, zd, zd, cfg.temperature)
                wmse = F.mse_loss(wy, prev_yb) # Lagged

            dh, de, derr, _, s2, dtau = tracker.compute(xb, wmse)
            det = torch.full((xb.size(0), 1), derr, device=cfg.device)

            with torch.no_grad(): py, pg, _, pe = model(xb, dh, det, cfg.temperature, hard=False)
            
            expl, gap = compute_explanation_signals(prev_yb, py, pe, pg) # Lagged
            phi = compute_phi_signal(de, derr, expl, gap, cfg.phi_scale_env, cfg.phi_scale_err, cfg.phi_scale_explain, cfg.phi_scale_gap)
            pi = build_policy_input(xb, dh, det, phi, s2, dtau)
            sp, tp, mp = model.policy(pi)
            
            swl, tgl, mdl = build_policy_targets(prev_yb, pe, phi, s2, dtau, cfg.policy_switch_threshold) # Lagged
            temp = compute_adaptive_temperature(phi, cfg.temp_stable, cfg.temp_transition)

            hm = bool(cfg.use_hard_switch and (mp[:, 1].mean().item() > 0.5) and not (float(dh.mean().item()) > cfg.phi_hard_threshold))
            yh, gp, _, eo = model(xb, dh, det, temp, hard=False)

            em = cfg.policy_mix_weight * float(sp[:, 1].mean().item())
            toh = F.one_hot(torch.argmax(tp.mean(0), dim=-1), cfg.num_experts).float().unsqueeze(0).expand(xb.size(0), -1)
            mx = (1.0 - em) * gp + em * ((toh - gp).detach() + gp)
            fr = F.one_hot(mx.argmax(-1), cfg.num_experts).float() if hm else mx
            yh = (fr.unsqueeze(-1) * eo).sum(1)

            # Backward pass (Must use current yb)
            mse_loss = F.mse_loss(yh, yb)
            _, gap_loss = compute_explanation_signals(yb, yh, eo, fr)
            
            st = torch.full((xb.size(0),), swl, dtype=torch.long, device=cfg.device)
            tt = torch.full((xb.size(0),), int(tgl.item()), dtype=torch.long, device=cfg.device)
            mt = torch.full((xb.size(0),), mdl, dtype=torch.long, device=cfg.device)

            tl = (mse_loss + cfg.beta_phi * (phi.detach() * gap_loss) + cfg.alpha_dogma * compute_dogma_penalty(fr) - cfg.beta_nomad * compute_nomad_bonus(fr)
                  + cfg.gamma_diversity * compute_diversity_loss(eo) + cfg.lambda_sep * compute_regime_gate_stats(fr, rb, cfg.num_regimes)[1] 
                  + cfg.lambda_cons * compute_regime_gate_stats(fr, rb, cfg.num_regimes)[2] + cfg.lambda_load * compute_load_balancing_loss(fr)
                  + cfg.policy_weight_stay * F.nll_loss(torch.log(sp + 1e-8), st) + cfg.policy_weight_target * F.nll_loss(torch.log(tp + 1e-8), tt)
                  + cfg.policy_weight_mode * F.nll_loss(torch.log(mp + 1e-8), mt) - dreg.compute(fr, dtau if cfg.use_dynamic_tau else cfg.tau_k_min))
            
            tl.backward(); optimizer.step()
            emse += mse_loss.item(); nb += 1
            prev_yb = yb.detach()

        logs["train_mse_losses"].append(emse / max(nb, 1))
        
        tmse, _, _, _, _, _, _, _, _ = evaluate_nomadic_static_full(model, X_test, Y_test, R_test, cfg)
        smse, _, dyn, _, _ = evaluate_nomadic_sequence_dynamics(model, X_test, Y_test, R_test, phase_tags_test, cfg)
        
        logs["test_mse_static"].append(tmse); logs["test_mse_sequence"].append(smse)
        logs["test_switch_latency"].append(dyn["mean_switch_latency"]); logs["test_transition_entropy"].append(dyn["transition_entropy_mean"]); logs["test_stable_entropy"].append(dyn["stable_entropy_mean"])

        if (epoch + 1) % 55 == 0 or epoch == 0:
            print(f"[Nomadic Lagged] Ep {epoch+1:03d}/{cfg.epochs} | Train MSE: {logs['train_mse_losses'][-1]:.4f} | Test Seq MSE: {smse:.4f} | Latency: {dyn['mean_switch_latency']:.4f}")

    return model, logs

# (평가, 시각화 함수 등은 기존과 동일)
def evaluate_fixed(*args, **kwargs): pass
def train_fixed(*args, **kwargs): pass
def evaluate_standard_moe(*args, **kwargs): pass
def evaluate_standard_moe_sequence(*args, **kwargs): pass
def train_standard_moe(*args, **kwargs): pass
def evaluate_nomadic_static_full(*args, **kwargs): pass
def print_report(*args, **kwargs): pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--save_dir", type=str, default="outputs_robust_lagged")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = build_config_from_yaml(load_yaml_config(args.config))
    cfg.save_dir = args.save_dir
    cfg.seed = args.seed
    if args.device != "auto": cfg.device = args.device

    set_seed(cfg.seed)
    print(f"Robustness (4 Regimes) Lagged Test | Device: {cfg.device} | Seed: {cfg.seed}")

    X_tr, Y_tr, R_tr, pt_tr = generate_phase_sequence(cfg, cfg.phase_train_cycles, cfg.device)
    X_te, Y_te, R_te, pt_te = generate_phase_sequence(cfg, cfg.phase_test_cycles, cfg.device)

    print("--- Nomadic Full (Lagged) Training ---")
    nm, nm_logs = train_nomadic(cfg, X_tr, Y_tr, R_tr, X_te, Y_te, R_te, pt_te)
    
    print("\n[COMPLETE] 인과율(Lagged)이 강제된 4-Regime 강건성 실험이 완료되었습니다.")

if __name__ == "__main__":
    main()