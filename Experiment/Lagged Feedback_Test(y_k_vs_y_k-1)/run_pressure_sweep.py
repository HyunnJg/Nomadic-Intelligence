"""
run_pressure_sweep_lagged.py
─────────────────────────────────────────────────────────────────────────────
§5.1 "Intermediate Regime" 실증 실험 — Prediction Pressure Sweep (Lagged v3)

목적:
  "Nomadic Routing은 intermediate change pressure에서 작동한다"는 핵심 주장을
  인과율(k-1 시차 기동)이 강제된 환경에서 연속적으로 검증한다.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import random
import csv
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int):
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
    overlap_std: float = 0.9          # ← sweep 변수 A

    hidden_dim: int = 64
    num_experts: int = 3
    gate_hidden_dim: int = 64
    temperature: float = 0.60

    epochs: int = 150
    lr: float = 2e-3
    weight_decay: float = 1e-5

    phase_batch_size: int = 64
    phase_train_cycles: int = 40
    phase_test_cycles: int = 12
    transition_steps: int = 8         # ← sweep 변수 B
    num_regimes: int = 3
    random_regime_order: bool = False

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

    save_dir: str = "outputs_sweep_lagged"

# ============================================================
# Data Generation
# ============================================================
REGIME_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_REGIME = {0: "A", 1: "B", 2: "C"}

def sample_regime_x(regime: str, n: int, std: float, device: str) -> torch.Tensor:
    centers = {"A": (2.5, 2.5), "B": (-2.5, -2.5), "C": (2.5, -2.5)}
    c = torch.tensor(centers[regime], device=device)
    return std * torch.randn(n, 2, device=device) + c

def regime_function(x: torch.Tensor, regime: str) -> torch.Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    if regime == "A": return (x1 + x2).unsqueeze(-1)
    if regime == "B": return (x1 - x2).unsqueeze(-1)
    if regime == "C": return (-x1 + 0.5 * x2).unsqueeze(-1)
    raise ValueError(regime)

def generate_phase_sequence(cfg: Config, cycles: int):
    active = ["A", "B", "C"]
    xs, ys, rs, tags = [], [], [], []
    for _ in range(cycles):
        for i, curr in enumerate(active):
            nxt = active[(i + 1) % len(active)]
            xb = sample_regime_x(curr, cfg.phase_batch_size, cfg.overlap_std, cfg.device)
            xs.append(xb); ys.append(regime_function(xb, curr))
            rs.append(torch.full((cfg.phase_batch_size,), REGIME_TO_ID[curr], dtype=torch.long, device=cfg.device))
            tags.extend([f"stable_{curr}"] * cfg.phase_batch_size)
            for step in range(cfg.transition_steps):
                alpha = (step + 1) / cfg.transition_steps
                xa = sample_regime_x(curr, cfg.phase_batch_size, cfg.overlap_std, cfg.device)
                xn = sample_regime_x(nxt,  cfg.phase_batch_size, cfg.overlap_std, cfg.device)
                xm = (1 - alpha) * xa + alpha * xn
                ym = (1 - alpha) * regime_function(xm, curr) + alpha * regime_function(xm, nxt)
                dom = curr if alpha < 0.5 else nxt
                xs.append(xm); ys.append(ym)
                rs.append(torch.full((cfg.phase_batch_size,), REGIME_TO_ID[dom], dtype=torch.long, device=cfg.device))
                tags.extend([f"transition_{curr}_to_{nxt}"] * cfg.phase_batch_size)
    return torch.cat(xs), torch.cat(ys), torch.cat(rs), tags

def iter_batches(X, Y, R, bs):
    n = X.size(0)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        yield X[s:e], Y[s:e], R[s:e]

# ============================================================
# Models
# ============================================================
class Expert(nn.Module):
    def __init__(self, in_d, h, out_d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, out_d))
    def forward(self, x): return self.net(x)

class GateNet(nn.Module):
    def __init__(self, in_d, h, n_exp):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d + 2, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_exp))
    def forward(self, x, dh, de, temp):
        logits = self.net(torch.cat([x, dh, de], dim=-1))
        return F.softmax(logits / temp, dim=-1), logits

class PolicyNet(nn.Module):
    def __init__(self, in_d, h, n_exp):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_d + 5, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.ss_head  = nn.Linear(h, 2)
        self.tgt_head = nn.Linear(h, n_exp)
        self.mode_head= nn.Linear(h, 2)
    def forward(self, x):
        h = self.shared(x)
        return F.softmax(self.ss_head(h), dim=-1), F.softmax(self.tgt_head(h), dim=-1), F.softmax(self.mode_head(h), dim=-1)

class NomadicMoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.experts = nn.ModuleList([Expert(cfg.input_dim, cfg.hidden_dim, cfg.output_dim) for _ in range(cfg.num_experts)])
        self.gate = GateNet(cfg.input_dim, cfg.gate_hidden_dim, cfg.num_experts)
        self.policy = PolicyNet(cfg.input_dim, cfg.policy_hidden_dim, cfg.num_experts)
    def forward(self, x, dh, de, temp, hard=False):
        gp, gl = self.gate(x, dh, de, temp)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        r = F.one_hot(gp.argmax(-1), self.num_experts).float() if hard else gp
        return (r.unsqueeze(-1) * eo).sum(1), gp, gl, eo

class StandardMoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.experts = nn.ModuleList([Expert(cfg.input_dim, cfg.hidden_dim, cfg.output_dim) for _ in range(cfg.num_experts)])
        self.gate = nn.Sequential(nn.Linear(cfg.input_dim, cfg.gate_hidden_dim), nn.ReLU(), nn.Linear(cfg.gate_hidden_dim, cfg.gate_hidden_dim), nn.ReLU(), nn.Linear(cfg.gate_hidden_dim, cfg.num_experts))
    def forward(self, x, hard=False):
        gp = F.softmax(self.gate(x), dim=-1)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        r = F.one_hot(gp.argmax(-1), self.num_experts).float() if hard else gp
        return (r.unsqueeze(-1) * eo).sum(1), gp

# ============================================================
# Utilities
# ============================================================
class HybridDeltaTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.recent = deque(maxlen=cfg.tau_var_window); self.reset()
    def reset(self):
        self.prev_mean = self.err_ema = self.err_base = None; self.recent.clear()
    def compute(self, x, mse):
        xm = x.mean(0, keepdim=True)
        de = 0.0 if self.prev_mean is None else float(torch.norm(xm - self.prev_mean, 2).item())
        self.prev_mean = xm.detach(); self.recent.append(de)
        err = float(mse.detach().item())
        if self.err_ema is None:
            self.err_ema = err; self.err_base = err; derr = 0.0
        else:
            self.err_ema  = self.cfg.ema_decay * self.err_ema  + (1 - self.cfg.ema_decay) * err
            self.err_base = self.cfg.err_baseline_momentum * self.err_base + (1 - self.cfg.err_baseline_momentum) * self.err_ema
            derr = max(0.0, self.err_ema - self.err_base)
        dh_val = float(torch.tanh(torch.tensor(self.cfg.w_env * de + self.cfg.w_err * derr)).item())
        sig2 = float(np.var(list(self.recent))) if len(self.recent) >= 2 else 0.0
        tau  = self.cfg.tau_min + (self.cfg.tau_max - self.cfg.tau_min) / (1 + self.cfg.tau_var_scale * sig2)
        return torch.full((x.size(0), 1), dh_val, device=x.device), de, derr, dh_val, sig2, tau

def explanation_signals(yb, yhat, eo, gp):
    err = F.mse_loss(yhat, yb)
    per = ((eo - yb.unsqueeze(1)) ** 2).mean(-1)
    return err, torch.relu(per.gather(1, gp.argmax(-1, keepdim=True)).mean() - per.min(1).values.mean())

def phi_signal(de, derr, expl, gap, cfg):
    return torch.tanh(cfg.phi_scale_env * torch.tensor(de, device=expl.device) + cfg.phi_scale_err * torch.tensor(derr, device=expl.device) + cfg.phi_scale_explain * expl.detach() + cfg.phi_scale_gap * gap.detach())

def adaptive_temp(phi, cfg): return cfg.temp_stable + (cfg.temp_transition - cfg.temp_stable) * float(phi.mean().item())

def build_policy_input(xb, dh, de_t, phi, sig2, tau):
    xs = xb.mean(0, keepdim=True).expand(xb.size(0), -1)
    return torch.cat([xs, dh, de_t, torch.full((xb.size(0), 1), float(phi.mean().item()), device=xb.device), torch.full((xb.size(0), 1), float(np.tanh(sig2 * 10)), device=xb.device), torch.full((xb.size(0), 1), float(np.tanh((tau - 5) / 5)), device=xb.device)], dim=-1)

def policy_targets(yb, eo, phi, sig2, tau, cfg):
    tgt = ((eo - yb.unsqueeze(1)) ** 2).mean(-1).mean(0).argmin().long()
    pv = float(phi.mean().item())
    return 1 if (pv > cfg.policy_switch_threshold or sig2 > 0.05) else 0, tgt, 1 if (pv <= cfg.policy_switch_threshold and tau >= 5.5) else 0

def load_balance_loss(gp):
    E = gp.size(-1); mg = gp.mean(0); t1 = torch.zeros(E, device=gp.device)
    for i in range(E): t1[i] = (gp.argmax(-1) == i).float().mean()
    return E * (t1 * mg).sum()

def diversity_loss(eo):
    E = eo.size(1); loss = 0.0; n = 0
    for i in range(E):
        for j in range(i+1, E):
            loss += F.cosine_similarity(eo[:, i], eo[:, j], dim=-1).mean(); n += 1
    return loss / max(n, 1)

def gate_entropy(gp): return -(gp * (gp + 1e-8).log()).sum(-1)

def regime_gate_stats(gp, rb, n_reg=3):
    reg_means = []
    for r in range(n_reg):
        m = (rb == r)
        if m.sum() > 0: reg_means.append(gp[m].mean(0))
    if len(reg_means) < 2: return torch.tensor(0., device=gp.device), torch.tensor(0., device=gp.device)
    sep = 0.0; pairs = 0
    for i in range(len(reg_means)):
        for j in range(i+1, len(reg_means)):
            sep += F.cosine_similarity(reg_means[i].unsqueeze(0), reg_means[j].unsqueeze(0)).mean(); pairs += 1
    sep_loss = sep / max(pairs, 1)
    cons_loss = torch.tensor(0., device=gp.device)
    for r in range(n_reg):
        m = (rb == r)
        if m.sum() > 1: cons_loss = cons_loss + gp[m].var(0).sum()
    return sep_loss, cons_loss

class DwellReg:
    def __init__(self, cfg: Config): self.tau_min = cfg.tau_k_min; self.pen = cfg.tau_k_penalty; self.reset()
    def reset(self): self.cur = None; self.cnt = 0
    def compute(self, gp, tau):
        dom = int(torch.bincount(gp.argmax(-1), minlength=gp.size(-1)).argmax().item())
        self.cnt = (self.cnt + 1) if dom == self.cur else 1
        self.cur = dom
        ent = gate_entropy(gp).mean(); cap = float(tau)
        if self.cnt <= cap: return -self.pen * ent
        return min((self.cnt - cap) * self.pen, self.pen * 10) * ent

# ============================================================
# Train / Eval (Lagged v3 반영)
# ============================================================
def train_stdmoe(cfg: Config, Xtr, Ytr, Rtr):
    model = StandardMoE(cfg).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        model.train()
        for xb, yb, rb in iter_batches(Xtr, Ytr, Rtr, cfg.phase_batch_size):
            opt.zero_grad()
            yh, gp = model(xb)
            F.mse_loss(yh, yb).backward(); opt.step()
    return model

def train_nomadic(cfg: Config, Xtr, Ytr, Rtr):
    model = NomadicMoE(cfg).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    for ep in range(cfg.epochs):
        model.train()
        tracker = HybridDeltaTracker(cfg); tracker.reset()
        dwell   = DwellReg(cfg); dwell.reset()
        
        # [핵심] 이전 배치 정답 버퍼
        prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

        for xb, yb, rb in iter_batches(Xtr, Ytr, Rtr, cfg.phase_batch_size):
            opt.zero_grad()
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            z = torch.zeros(xb.size(0), 1, device=cfg.device)
            with torch.no_grad():
                wy, _, _, _ = model(xb, z, z, cfg.temperature)
                # [핵심] yb -> prev_yb (시차 기동)
                wmse = F.mse_loss(wy, prev_yb)
            
            dh, de, derr, _, sig2, tau = tracker.compute(xb, wmse)
            de_t = torch.full((xb.size(0), 1), derr, device=cfg.device)
            
            with torch.no_grad(): py, pgp, _, peo = model(xb, dh, de_t, cfg.temperature)
            
            # [핵심] yb -> prev_yb (시차 성찰)
            ee, gap = explanation_signals(prev_yb, py, peo, pgp)
            phi = phi_signal(de, derr, ee, gap, cfg); tmp = adaptive_temp(phi, cfg)

            pi = build_policy_input(xb, dh, de_t, phi, sig2, tau)
            ss, tp, mp = model.policy(pi)

            # [핵심] yb -> prev_yb
            sw_lbl, tgt_lbl, md_lbl = policy_targets(prev_yb, peo, phi, sig2, tau, cfg)

            yh, gp, _, eo = model(xb, dh, de_t, tmp)
            eff_mix = cfg.policy_mix_weight * float(ss[:, 1].mean().item())
            ste = (F.one_hot(torch.argmax(tp.mean(0)), cfg.num_experts).float().unsqueeze(0).expand(xb.size(0), -1) - gp).detach() + gp
            mr  = (1 - eff_mix) * gp + eff_mix * ste
            if cfg.use_hard_switch and mp[:, 1].mean().item() > 0.5 and float(dh.mean().item()) <= cfg.phi_hard_threshold:
                mr = F.one_hot(mr.argmax(-1), cfg.num_experts).float()
            yh = (mr.unsqueeze(-1) * eo).sum(1)

            mse_l = F.mse_loss(yh, yb)
            _, gap2 = explanation_signals(yb, yh, eo, mr)
            sl, cl = regime_gate_stats(mr, rb, cfg.num_regimes)

            loss = (mse_l
                    + cfg.beta_phi * phi.detach() * gap2 + cfg.alpha_dogma * (mr.mean(0) ** 2).sum() - cfg.beta_nomad * gate_entropy(mr).mean()
                    + cfg.gamma_diversity * diversity_loss(eo) + cfg.lambda_sep * sl + cfg.lambda_cons * cl + cfg.lambda_load * load_balance_loss(mr)
                    + cfg.policy_weight_stay * F.nll_loss(torch.log(ss + 1e-8), torch.full((xb.size(0),), sw_lbl, dtype=torch.long, device=cfg.device))
                    + cfg.policy_weight_target * F.nll_loss(torch.log(tp + 1e-8), torch.full((xb.size(0),), int(tgt_lbl.item()), dtype=torch.long, device=cfg.device))
                    + cfg.policy_weight_mode * F.nll_loss(torch.log(mp + 1e-8), torch.full((xb.size(0),), md_lbl, dtype=torch.long, device=cfg.device))
                    - dwell.compute(mr, tau))
            loss.backward(); opt.step()

            # [핵심] 버퍼 업데이트
            prev_yb = yb.detach()
            
    return model

@torch.no_grad()
def eval_seq(model, X, Y, R, tags, cfg, is_nomadic: bool):
    model.eval(); ys, ents, phase_ents = [], [], []
    if is_nomadic: tracker = HybridDeltaTracker(cfg); tracker.reset()
    prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

    for bi, (xb, yb, rb) in enumerate(iter_batches(X, Y, R, cfg.phase_batch_size)):
        if is_nomadic:
            if prev_yb.size(0) != xb.size(0): prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)
            z = torch.zeros(xb.size(0), 1, device=cfg.device)
            wy, _, _, _ = model(xb, z, z, cfg.temperature)
            wmse = F.mse_loss(wy, prev_yb) # Lagged
            dh, de, derr, _, sig2, tau = tracker.compute(xb, wmse)
            de_t = torch.full((xb.size(0), 1), derr, device=cfg.device)
            py, pgp, _, peo = model(xb, dh, de_t, cfg.temperature)
            ee, gap = explanation_signals(prev_yb, py, peo, pgp) # Lagged
            phi = phi_signal(de, derr, ee, gap, cfg); tmp = adaptive_temp(phi, cfg)
            yh, gp, _, _ = model(xb, dh, de_t, tmp)
            prev_yb = yb.detach() # Update
        else:
            yh, gp = model(xb)
        ys.append(yh); ent = float(gate_entropy(gp).mean().item())
        phase_ents.append((tags[bi * cfg.phase_batch_size], ent))

    s_h = [e for t, e in phase_ents if t.startswith("stable_")]
    tr_h = [e for t, e in phase_ents if t.startswith("transition_")]
    return float(F.mse_loss(torch.cat(ys), Y).item()), float(np.mean(s_h)) if s_h else float("nan"), float(np.mean(tr_h)) if tr_h else float("nan")

# ============================================================
# Sweep Execution & Plotting
# ============================================================
def run_one_condition(cfg: Config, seeds: List[int]):
    results = {"std_mse": [], "nom_mse": [], "nom_sh": [], "nom_th": []}
    for seed in seeds:
        set_seed(seed); cfg_s = Config(**{k: v for k, v in cfg.__dict__.items()}); cfg_s.seed = seed
        Xtr, Ytr, Rtr, _ = generate_phase_sequence(cfg_s, cfg.phase_train_cycles)
        Xte, Yte, Rte, tags = generate_phase_sequence(cfg_s, cfg.phase_test_cycles)
        std = train_stdmoe(cfg_s, Xtr, Ytr, Rtr); nom = train_nomadic(cfg_s, Xtr, Ytr, Rtr)
        results["std_mse"].append(eval_seq(std, Xte, Yte, Rte, tags, cfg_s, is_nomadic=False)[0])
        mse, sh, th = eval_seq(nom, Xte, Yte, Rte, tags, cfg_s, is_nomadic=True)
        results["nom_mse"].append(mse); results["nom_sh"].append(sh); results["nom_th"].append(th)
    
    std_m, nom_m = float(np.mean(results["std_mse"])), float(np.mean(results["nom_mse"]))
    dh_vals = [th - sh for th, sh in zip(results["nom_th"], results["nom_sh"]) if not (np.isnan(th) or np.isnan(sh))]
    return {"std_mse_mean": std_m, "nom_mse_mean": nom_m, "mse_impr_pct": (std_m - nom_m) / max(std_m, 1e-9) * 100, "dh_mean": float(np.mean(dh_vals)) if dh_vals else float("nan"), "dh_std": float(np.std(dh_vals)) if dh_vals else float("nan"), "stable_h_mean": float(np.mean(results["nom_sh"])), "stable_h_std": float(np.std(results["nom_sh"]))}

def run_sigma_sweep(seeds, quick=False):
    rows = []
    for σ in ([0.3, 0.9, 1.5] if quick else [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]):
        print(f"  [sigma sweep] σ={σ:.1f} ... ", end="", flush=True)
        t0 = time.time(); cfg = Config(overlap_std=σ, transition_steps=8)
        if quick: cfg.epochs = 100
        r = run_one_condition(cfg, seeds); r["sigma"] = σ; rows.append(r)
        print(f"MSE_impr={r['mse_impr_pct']:+.1f}%  ΔH={r['dh_mean']:.3f}  ({time.time()-t0:.0f}s)")
    return rows

def run_transition_sweep(seeds, quick=False):
    rows = []
    for ts in ([2, 8, 24] if quick else [2, 4, 8, 16, 24]):
        print(f"  [transition sweep] steps={ts} ... ", end="", flush=True)
        t0 = time.time(); cfg = Config(overlap_std=0.9, transition_steps=ts)
        if quick: cfg.epochs = 100
        r = run_one_condition(cfg, seeds); r["transition_steps"] = ts; rows.append(r)
        print(f"MSE_impr={r['mse_impr_pct']:+.1f}%  ΔH={r['dh_mean']:.3f}  ({time.time()-t0:.0f}s)")
    return rows

def save_csv(rows, path, key_col):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[key_col] + [k for k in rows[0] if k != key_col])
        w.writeheader()
        for r in rows: w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})

def plot_sigma_sweep(rows, save_path):
    sigmas = [r["sigma"] for r in rows]; impr = [r["mse_impr_pct"] for r in rows]; dh = [r["dh_mean"] for r in rows]; dh_std = [r["dh_std"] for r in rows]; sh = [r["stable_h_mean"] for r in rows]
    fig = plt.figure(figsize=(12, 4.5)); gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(range(len(sigmas)), impr, color=["#4A7BA7" if v > 0 else "#E07B54" for v in impr], edgecolor="white", linewidth=0.6)
    ax1.axhline(0, color="gray", lw=0.8, ls="--"); ax1.set_xticks(range(len(sigmas))); ax1.set_xticklabels([f"{s}" for s in sigmas], fontsize=9)
    ax1.set_xlabel("Noise σ (overlap_std)", fontsize=10); ax1.set_ylabel("MSE Improvement over StdMoE (%)", fontsize=9); ax1.set_title("Prediction Accuracy\nvs Noise Level", fontsize=10, fontweight="bold")
    for bar, val in zip(bars, impr): ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.5 if val >= 0 else -1.5), f"{val:+.1f}%", ha="center", va="bottom", fontsize=8)
    
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(sigmas, dh, "o-", color="#4A7BA7", lw=2, ms=7, zorder=3)
    ax2.fill_between(sigmas, [d - e for d, e in zip(dh, dh_std)], [d + e for d, e in zip(dh, dh_std)], alpha=0.18, color="#4A7BA7")
    ax2.axhline(0, color="gray", lw=0.8, ls="--"); ax2.set_xlabel("Noise σ", fontsize=10); ax2.set_ylabel("ΔH  (Trans H − Stable H)", fontsize=9); ax2.set_title("Homeomorphic Fixation\nStrength vs Noise Level", fontsize=10, fontweight="bold")
    y_top = max(dh) * 1.25 if max(dh) > 0 else 0.3
    ax2.annotate("Info\nStarvation", xy=(sigmas[0], dh[0]), xytext=(sigmas[0] + 0.05, y_top * 0.7), fontsize=7.5, color="#888", arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))
    mid_idx = int(len(sigmas) / 2)
    ax2.annotate("Intermediate\nRegime", xy=(sigmas[mid_idx], dh[mid_idx]), xytext=(sigmas[mid_idx] - 0.15, y_top * 1.0), fontsize=7.5, color="#4A7BA7", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#4A7BA7", lw=0.8))
    ax2.annotate("High-noise\nCollapse", xy=(sigmas[-1], dh[-1]), xytext=(sigmas[-1] - 0.5, y_top * 0.7), fontsize=7.5, color="#888", arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(sigmas, sh, "s-", color="#5BAD72", lw=2, ms=7, zorder=3)
    ax3.fill_between(sigmas, [s - r["stable_h_std"] for s, r in zip(sh, rows)], [s + r["stable_h_std"] for s, r in zip(sh, rows)], alpha=0.18, color="#5BAD72")
    ax3.axhline(np.log(3), color="#ccc", lw=0.8, ls=":", label="log(3) uniform"); ax3.set_xlabel("Noise σ", fontsize=10); ax3.set_ylabel("Stable-phase Entropy", fontsize=9); ax3.set_title("Fixation Depth\nvs Noise Level", fontsize=10, fontweight="bold"); ax3.legend(fontsize=7.5)

    fig.suptitle("§5.1 Pressure Sweep A (Lagged v3) — Noise σ\n\"Intermediate change pressure is the operational regime of Nomadic Routing\"", fontsize=10, y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_transition_sweep(rows, save_path):
    tvals = [r["transition_steps"] for r in rows]; impr = [r["mse_impr_pct"] for r in rows]; dh = [r["dh_mean"] for r in rows]; dh_std = [r["dh_std"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    
    ax1.bar(range(len(tvals)), impr, color=["#4A7BA7" if v > 0 else "#E07B54" for v in impr], edgecolor="white", linewidth=0.6)
    ax1.axhline(0, color="gray", lw=0.8, ls="--"); ax1.set_xticks(range(len(tvals))); ax1.set_xticklabels([str(t) for t in tvals], fontsize=9); ax1.set_xlabel("Transition Steps", fontsize=10); ax1.set_ylabel("MSE Improvement (%)", fontsize=9); ax1.set_title("Prediction Accuracy\nvs Transition Speed", fontsize=10, fontweight="bold")
    
    ax2.plot(tvals, dh, "o-", color="#4A7BA7", lw=2, ms=7)
    ax2.fill_between(tvals, [d - e for d, e in zip(dh, dh_std)], [d + e for d, e in zip(dh, dh_std)], alpha=0.18, color="#4A7BA7")
    ax2.axhline(0, color="gray", lw=0.8, ls="--"); ax2.set_xlabel("Transition Steps", fontsize=10); ax2.set_ylabel("ΔH", fontsize=9); ax2.set_title("Homeomorphic Fixation\nvs Transition Speed", fontsize=10, fontweight="bold")

    fig.suptitle("§5.1 Pressure Sweep B (Lagged v3) — Transition Speed (σ=0.9 fixed)", fontsize=9, y=1.02)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["sigma", "transition", "both"], default="both")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    SEEDS = [42] if args.quick else [42, 123, 456]; os.makedirs("outputs_sweep_lagged", exist_ok=True)
    print(f"\n{'='*60}\nNomadic Routing — Pressure Sweep (Lagged v3)\n  seeds={SEEDS}  device={args.device}  quick={args.quick}\n{'='*60}\n")

    if args.sweep in ("sigma", "both"):
        print("▶ Sweep A: Noise σ")
        sigma_rows = run_sigma_sweep(SEEDS, quick=args.quick)
        save_csv(sigma_rows, "outputs_sweep_lagged/sigma_sweep_results.csv", "sigma")
        plot_sigma_sweep(sigma_rows, "outputs_sweep_lagged/fig_sigma_sweep.png")

    if args.sweep in ("transition", "both"):
        print("\n▶ Sweep B: Transition Steps (σ=0.9 fixed)")
        trans_rows = run_transition_sweep(SEEDS, quick=args.quick)
        save_csv(trans_rows, "outputs_sweep_lagged/transition_sweep_results.csv", "transition_steps")
        plot_transition_sweep(trans_rows, "outputs_sweep_lagged/fig_transition_sweep.png")

    print("\n✅ Sweep complete (Lagged v3). Check 'outputs_sweep_lagged/' folder.")

if __name__ == "__main__":
    main()