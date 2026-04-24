"""
run_pressure_sweep.py
─────────────────────────────────────────────────────────────────────────────
§5.1 "Intermediate Regime" 실증 실험 — Prediction Pressure Sweep

목적:
  "Nomadic Routing은 intermediate change pressure에서 작동한다"는 핵심 주장을
  합성 환경에서 연속적으로 검증한다.

  Concept.md 언어로:
    - σ 너무 낮음  → Δx≈0, R_sync 굶주림 → 나선이 점으로 수렴 (정보 기아 한계)
    - σ 너무 높음  → Φ signal 붕괴, R_nomad만 남아 부유 → 나선이 발산
    - σ 중간       → 세 힘(Sync/Dogma/Trans) 균형 → homeomorphic fixation 발현

실험 설계:
  독립 변수 A: overlap_std (입력 노이즈) — [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]
  독립 변수 B: transition_steps (전환 속도) — [2, 4, 8, 16, 24]  (σ=0.9 고정)

  각 조건: StandardMoE vs Nomadic Full, 3 seeds (42/123/456)
  에폭: 150 (sweep이므로 220→150으로 단축, 결과 안정성 확인됨)

  측정:
    - Seq MSE improvement: (StdMoE - Nomadic) / StdMoE × 100
    - ΔH: transition_entropy - stable_entropy (homeomorphic fixation 지표)
    - Stable Entropy: fixation 강도

출력:
  outputs_sweep/
    sigma_sweep_results.csv
    transition_sweep_results.csv
    fig_sigma_sweep.png          ← 논문 figure
    fig_transition_sweep.png     ← 논문 figure
    sweep_summary.txt

사용법:
  python run_pressure_sweep.py                     # 전체 sweep
  python run_pressure_sweep.py --sweep sigma       # sigma만
  python run_pressure_sweep.py --sweep transition  # transition만
  python run_pressure_sweep.py --quick             # 빠른 확인 (1 seed, 100 epoch)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import random
import csv
import time
from collections import deque
from dataclasses import dataclass, field
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

    # data
    input_dim: int = 2
    output_dim: int = 1
    overlap_std: float = 0.9          # ← sweep 변수 A

    # model
    hidden_dim: int = 64
    num_experts: int = 3
    gate_hidden_dim: int = 64
    temperature: float = 0.60

    # training
    epochs: int = 150
    lr: float = 2e-3
    weight_decay: float = 1e-5

    # phase sequence
    phase_batch_size: int = 64
    phase_train_cycles: int = 40
    phase_test_cycles: int = 12
    transition_steps: int = 8         # ← sweep 변수 B
    num_regimes: int = 3
    random_regime_order: bool = False

    # hybrid delta
    ema_decay: float = 0.80
    err_baseline_momentum: float = 0.85
    w_env: float = 1.0
    w_err: float = 2.0

    # loss weights
    alpha_dogma: float = 0.04
    beta_nomad: float = 0.05
    beta_phi: float = 0.05
    gamma_diversity: float = 0.08
    lambda_sep: float = 0.08
    lambda_cons: float = 0.03
    lambda_load: float = 0.03
    tau_k_min: int = 3
    tau_k_penalty: float = 0.05

    # dynamic tau
    use_dynamic_tau: bool = True
    tau_min: float = 2.0
    tau_max: float = 8.0
    tau_var_scale: float = 6.0
    tau_var_window: int = 8

    # phi / switching
    phi_scale_env: float = 1.0
    phi_scale_err: float = 1.5
    phi_scale_explain: float = 2.0
    phi_scale_gap: float = 1.0
    temp_stable: float = 0.30
    temp_transition: float = 1.00
    use_hard_switch: bool = True
    phi_hard_threshold: float = 0.35

    # policy
    policy_hidden_dim: int = 64
    policy_mix_weight: float = 0.25
    policy_weight_stay: float = 0.20
    policy_weight_target: float = 0.20
    policy_weight_mode: float = 0.10
    policy_switch_threshold: float = 0.50

    save_dir: str = "outputs_sweep"


# ============================================================
# Data Generation  (run_extended_robust_test_.py와 동일)
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
            rs.append(torch.full((cfg.phase_batch_size,), REGIME_TO_ID[curr],
                                  dtype=torch.long, device=cfg.device))
            tags.extend([f"stable_{curr}"] * cfg.phase_batch_size)
            for step in range(cfg.transition_steps):
                alpha = (step + 1) / cfg.transition_steps
                xa = sample_regime_x(curr, cfg.phase_batch_size, cfg.overlap_std, cfg.device)
                xn = sample_regime_x(nxt,  cfg.phase_batch_size, cfg.overlap_std, cfg.device)
                xm = (1 - alpha) * xa + alpha * xn
                ym = (1 - alpha) * regime_function(xm, curr) + alpha * regime_function(xm, nxt)
                dom = curr if alpha < 0.5 else nxt
                xs.append(xm); ys.append(ym)
                rs.append(torch.full((cfg.phase_batch_size,), REGIME_TO_ID[dom],
                                      dtype=torch.long, device=cfg.device))
                tags.extend([f"transition_{curr}_to_{nxt}"] * cfg.phase_batch_size)
    return torch.cat(xs), torch.cat(ys), torch.cat(rs), tags


def iter_batches(X, Y, R, bs):
    n = X.size(0)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        yield X[s:e], Y[s:e], R[s:e]


# ============================================================
# Models  (run_extended_robust_test_.py와 동일)
# ============================================================

class Expert(nn.Module):
    def __init__(self, in_d, h, out_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, h), nn.Tanh(),
            nn.Linear(h, h),    nn.Tanh(),
            nn.Linear(h, out_d),
        )
    def forward(self, x): return self.net(x)


class GateNet(nn.Module):
    def __init__(self, in_d, h, n_exp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d + 2, h), nn.ReLU(),
            nn.Linear(h, h),        nn.ReLU(),
            nn.Linear(h, n_exp),
        )
    def forward(self, x, dh, de, temp):
        logits = self.net(torch.cat([x, dh, de], dim=-1))
        return F.softmax(logits / temp, dim=-1), logits


class PolicyNet(nn.Module):
    def __init__(self, in_d, h, n_exp):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_d + 5, h), nn.ReLU(),
            nn.Linear(h, h),        nn.ReLU(),
        )
        self.ss_head  = nn.Linear(h, 2)
        self.tgt_head = nn.Linear(h, n_exp)
        self.mode_head= nn.Linear(h, 2)
    def forward(self, x):
        h = self.shared(x)
        return (F.softmax(self.ss_head(h),   dim=-1),
                F.softmax(self.tgt_head(h),  dim=-1),
                F.softmax(self.mode_head(h), dim=-1))


class NomadicMoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.experts = nn.ModuleList(
            [Expert(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
             for _ in range(cfg.num_experts)])
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
        self.experts = nn.ModuleList(
            [Expert(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
             for _ in range(cfg.num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.gate_hidden_dim), nn.ReLU(),
            nn.Linear(cfg.gate_hidden_dim, cfg.gate_hidden_dim), nn.ReLU(),
            nn.Linear(cfg.gate_hidden_dim, cfg.num_experts),
        )
    def forward(self, x, hard=False):
        gp = F.softmax(self.gate(x), dim=-1)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        r = F.one_hot(gp.argmax(-1), self.num_experts).float() if hard else gp
        return (r.unsqueeze(-1) * eo).sum(1), gp


# ============================================================
# Hybrid Delta / Phi / Loss utilities
# ============================================================

class HybridDeltaTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.recent = deque(maxlen=cfg.tau_var_window)
        self.reset()

    def reset(self):
        self.prev_mean = None
        self.err_ema = None
        self.err_base = None
        self.recent.clear()

    def compute(self, x, mse):
        xm = x.mean(0, keepdim=True)
        de = 0.0 if self.prev_mean is None else float(torch.norm(xm - self.prev_mean, 2).item())
        self.prev_mean = xm.detach()
        self.recent.append(de)

        err = float(mse.detach().item())
        if self.err_ema is None:
            self.err_ema = err; self.err_base = err; derr = 0.0
        else:
            self.err_ema  = self.cfg.ema_decay * self.err_ema  + (1 - self.cfg.ema_decay) * err
            self.err_base = self.cfg.err_baseline_momentum * self.err_base + \
                            (1 - self.cfg.err_baseline_momentum) * self.err_ema
            derr = max(0.0, self.err_ema - self.err_base)

        dh_val = float(torch.tanh(torch.tensor(self.cfg.w_env * de + self.cfg.w_err * derr)).item())
        sig2 = float(np.var(list(self.recent))) if len(self.recent) >= 2 else 0.0
        tau  = self.cfg.tau_min + (self.cfg.tau_max - self.cfg.tau_min) / (1 + self.cfg.tau_var_scale * sig2)

        dh_t  = torch.full((x.size(0), 1), dh_val,  device=x.device)
        de_t  = torch.full((x.size(0), 1), derr,     device=x.device)
        return dh_t, de, derr, dh_val, sig2, tau


def explanation_signals(yb, yhat, eo, gp):
    err = F.mse_loss(yhat, yb)
    per = ((eo - yb.unsqueeze(1)) ** 2).mean(-1)
    top1_err  = per.gather(1, gp.argmax(-1, keepdim=True)).mean()
    best_err  = per.min(1).values.mean()
    gap = torch.relu(top1_err - best_err)
    return err, gap


def phi_signal(de, derr, expl, gap, cfg):
    return torch.tanh(
        cfg.phi_scale_env  * torch.tensor(de,   device=expl.device) +
        cfg.phi_scale_err  * torch.tensor(derr, device=expl.device) +
        cfg.phi_scale_explain * expl.detach() +
        cfg.phi_scale_gap  * gap.detach()
    )


def adaptive_temp(phi, cfg):
    return cfg.temp_stable + (cfg.temp_transition - cfg.temp_stable) * float(phi.mean().item())


def build_policy_input(xb, dh, de_t, phi, sig2, tau):
    xs = xb.mean(0, keepdim=True).expand(xb.size(0), -1)
    phi_t  = torch.full((xb.size(0), 1), float(phi.mean().item()), device=xb.device)
    s2_t   = torch.full((xb.size(0), 1), float(np.tanh(sig2 * 10)),   device=xb.device)
    tau_t  = torch.full((xb.size(0), 1), float(np.tanh((tau - 5) / 5)), device=xb.device)
    return torch.cat([xs, dh, de_t, phi_t, s2_t, tau_t], dim=-1)


def policy_targets(yb, eo, phi, sig2, tau, cfg):
    per = ((eo - yb.unsqueeze(1)) ** 2).mean(-1)
    tgt = per.mean(0).argmin().long()
    pv  = float(phi.mean().item())
    sw  = 1 if (pv > cfg.policy_switch_threshold or sig2 > 0.05) else 0
    md  = 1 if (pv <= cfg.policy_switch_threshold and tau >= 5.5)  else 0
    return sw, tgt, md


def load_balance_loss(gp):
    E = gp.size(-1)
    mg = gp.mean(0)
    t1 = torch.zeros(E, device=gp.device)
    for i in range(E): t1[i] = (gp.argmax(-1) == i).float().mean()
    return E * (t1 * mg).sum()


def diversity_loss(eo):
    E = eo.size(1)
    loss = 0.0; n = 0
    for i in range(E):
        for j in range(i+1, E):
            loss += F.cosine_similarity(eo[:, i], eo[:, j], dim=-1).mean()
            n += 1
    return loss / max(n, 1)


def gate_entropy(gp):
    return -(gp * (gp + 1e-8).log()).sum(-1)


def regime_gate_stats(gp, rb, n_reg=3):
    """Returns sep_loss, cons_loss."""
    eps = 1e-8
    E = gp.size(-1)
    reg_means = []
    for r in range(n_reg):
        m = (rb == r)
        if m.sum() > 0: reg_means.append(gp[m].mean(0))
    if len(reg_means) < 2:
        return torch.tensor(0., device=gp.device), torch.tensor(0., device=gp.device)
    # sep: maximize distance between regime centroids
    sep = 0.0; pairs = 0
    for i in range(len(reg_means)):
        for j in range(i+1, len(reg_means)):
            sep += F.cosine_similarity(reg_means[i].unsqueeze(0),
                                        reg_means[j].unsqueeze(0)).mean()
            pairs += 1
    sep_loss = sep / max(pairs, 1)
    # cons: penalize within-regime variance
    cons_loss = torch.tensor(0., device=gp.device)
    for r in range(n_reg):
        m = (rb == r)
        if m.sum() > 1:
            rm = gp[m]
            cons_loss = cons_loss + rm.var(0).sum()
    return sep_loss, cons_loss


class DwellReg:
    def __init__(self, cfg: Config):
        self.tau_min = cfg.tau_k_min; self.pen = cfg.tau_k_penalty
        self.cur = None; self.cnt = 0

    def reset(self): self.cur = None; self.cnt = 0

    def compute(self, gp, tau):
        dom = int(torch.bincount(gp.argmax(-1), minlength=gp.size(-1)).argmax().item())
        self.cnt = (self.cnt + 1) if dom == self.cur else 1
        self.cur = dom
        ent = gate_entropy(gp).mean()
        cap = float(tau)
        if self.cnt <= cap: return -self.pen * ent
        excess = self.cnt - cap
        return min(excess * self.pen, self.pen * 10) * ent


# ============================================================
# Train / Eval
# ============================================================

def train_stdmoe(cfg: Config, Xtr, Ytr, Rtr):
    model = StandardMoE(cfg).to(cfg.device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        model.train()
        for xb, yb, rb in iter_batches(Xtr, Ytr, Rtr, cfg.phase_batch_size):
            opt.zero_grad()
            yh, gp = model(xb)
            loss = F.mse_loss(yh, yb)
            loss.backward(); opt.step()
    return model


def train_nomadic(cfg: Config, Xtr, Ytr, Rtr):
    model = NomadicMoE(cfg).to(cfg.device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        model.train()
        tracker = HybridDeltaTracker(cfg); tracker.reset()
        dwell   = DwellReg(cfg); dwell.reset()
        for xb, yb, rb in iter_batches(Xtr, Ytr, Rtr, cfg.phase_batch_size):
            opt.zero_grad()
            z = torch.zeros(xb.size(0), 1, device=cfg.device)
            with torch.no_grad():
                wy, _, _, _ = model(xb, z, z, cfg.temperature)
                wmse = F.mse_loss(wy, yb)
            dh, de, derr, _, sig2, tau = tracker.compute(xb, wmse)
            de_t = torch.full((xb.size(0), 1), derr, device=cfg.device)
            with torch.no_grad():
                py, pgp, _, peo = model(xb, dh, de_t, cfg.temperature)
            ee, gap = explanation_signals(yb, py, peo, pgp)
            phi = phi_signal(de, derr, ee, gap, cfg)
            tmp = adaptive_temp(phi, cfg)

            pi  = build_policy_input(xb, dh, de_t, phi, sig2, tau)
            ss, tp, mp = model.policy(pi)

            sw_lbl, tgt_lbl, md_lbl = policy_targets(yb, peo, phi, sig2, tau, cfg)

            yh, gp, _, eo = model(xb, dh, de_t, tmp)
            eff_mix = cfg.policy_mix_weight * float(ss[:, 1].mean().item())
            ti  = torch.argmax(tp.mean(0))
            toh = F.one_hot(ti, cfg.num_experts).float().unsqueeze(0).expand(xb.size(0), -1)
            ste = (toh - gp).detach() + gp
            mr  = (1 - eff_mix) * gp + eff_mix * ste
            hard_mode = (cfg.use_hard_switch
                         and mp[:, 1].mean().item() > 0.5
                         and float(dh.mean().item()) <= cfg.phi_hard_threshold)
            if hard_mode:
                mr = F.one_hot(mr.argmax(-1), cfg.num_experts).float()
            yh = (mr.unsqueeze(-1) * eo).sum(1)

            mse_l = F.mse_loss(yh, yb)
            _, gap2 = explanation_signals(yb, yh, eo, mr)
            sl, cl = regime_gate_stats(mr, rb, cfg.num_regimes)

            stay_t = torch.full((xb.size(0),), sw_lbl,              dtype=torch.long, device=cfg.device)
            tgt_t  = torch.full((xb.size(0),), int(tgt_lbl.item()), dtype=torch.long, device=cfg.device)
            md_t   = torch.full((xb.size(0),), md_lbl,              dtype=torch.long, device=cfg.device)

            loss = (mse_l
                    + cfg.beta_phi     * phi.detach() * gap2
                    + cfg.alpha_dogma  * (mr.mean(0) ** 2).sum()
                    - cfg.beta_nomad   * gate_entropy(mr).mean()
                    + cfg.gamma_diversity * diversity_loss(eo)
                    + cfg.lambda_sep   * sl
                    + cfg.lambda_cons  * cl
                    + cfg.lambda_load  * load_balance_loss(mr)
                    + cfg.policy_weight_stay   * F.nll_loss(torch.log(ss   + 1e-8), stay_t)
                    + cfg.policy_weight_target * F.nll_loss(torch.log(tp   + 1e-8), tgt_t)
                    + cfg.policy_weight_mode   * F.nll_loss(torch.log(mp   + 1e-8), md_t)
                    - dwell.compute(mr, tau))
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_seq(model, X, Y, R, tags, cfg, is_nomadic: bool):
    """Returns (seq_mse, stable_H_mean, trans_H_mean)."""
    model.eval()
    ys, ents, phase_ents = [], [], []
    if is_nomadic:
        tracker = HybridDeltaTracker(cfg); tracker.reset()

    for bi, (xb, yb, rb) in enumerate(iter_batches(X, Y, R, cfg.phase_batch_size)):
        if is_nomadic:
            z = torch.zeros(xb.size(0), 1, device=cfg.device)
            wy, _, _, _ = model(xb, z, z, cfg.temperature)
            wmse = F.mse_loss(wy, yb)
            dh, de, derr, _, sig2, tau = tracker.compute(xb, wmse)
            de_t = torch.full((xb.size(0), 1), derr, device=cfg.device)
            py, pgp, _, peo = model(xb, dh, de_t, cfg.temperature)
            ee, gap = explanation_signals(yb, py, peo, pgp)
            phi = phi_signal(de, derr, ee, gap, cfg)
            tmp = adaptive_temp(phi, cfg)
            yh, gp, _, _ = model(xb, dh, de_t, tmp)
        else:
            yh, gp = model(xb)

        ys.append(yh)
        ent = float(gate_entropy(gp).mean().item())
        tag = tags[bi * cfg.phase_batch_size]
        phase_ents.append((tag, ent))

    Yhat = torch.cat(ys)
    mse  = float(F.mse_loss(Yhat, Y).item())
    s_h  = [e for t, e in phase_ents if t.startswith("stable_")]
    tr_h = [e for t, e in phase_ents if t.startswith("transition_")]
    sh   = float(np.mean(s_h))  if s_h  else float("nan")
    th   = float(np.mean(tr_h)) if tr_h else float("nan")
    return mse, sh, th


# ============================================================
# Single-condition experiment
# ============================================================

def run_one_condition(cfg: Config, seeds: List[int]):
    """Returns dict with mean/std of Seq MSE, ΔH, Stable H for both models."""
    results = {"std_mse": [], "nom_mse": [], "nom_sh": [], "nom_th": []}
    for seed in seeds:
        set_seed(seed)
        cfg_s = Config(**{k: v for k, v in cfg.__dict__.items()})
        cfg_s.seed = seed
        Xtr, Ytr, Rtr, _    = generate_phase_sequence(cfg_s, cfg.phase_train_cycles)
        Xte, Yte, Rte, tags = generate_phase_sequence(cfg_s, cfg.phase_test_cycles)

        std = train_stdmoe(cfg_s, Xtr, Ytr, Rtr)
        nom = train_nomadic(cfg_s, Xtr, Ytr, Rtr)

        std_mse, _, _ = eval_seq(std, Xte, Yte, Rte, tags, cfg_s, is_nomadic=False)
        nom_mse, sh, th = eval_seq(nom, Xte, Yte, Rte, tags, cfg_s, is_nomadic=True)

        results["std_mse"].append(std_mse)
        results["nom_mse"].append(nom_mse)
        results["nom_sh"].append(sh)
        results["nom_th"].append(th)

    def m(k): return float(np.mean(results[k]))
    def s(k): return float(np.std(results[k]))

    std_mse_m = m("std_mse")
    nom_mse_m = m("nom_mse")
    dh_vals   = [th - sh for th, sh in zip(results["nom_th"], results["nom_sh"])
                 if not (np.isnan(th) or np.isnan(sh))]

    return {
        "std_mse_mean":  std_mse_m,
        "nom_mse_mean":  nom_mse_m,
        "mse_impr_pct":  (std_mse_m - nom_mse_m) / max(std_mse_m, 1e-9) * 100,
        "dh_mean":       float(np.mean(dh_vals)) if dh_vals else float("nan"),
        "dh_std":        float(np.std(dh_vals))  if dh_vals else float("nan"),
        "stable_h_mean": m("nom_sh"),
        "stable_h_std":  s("nom_sh"),
    }


# ============================================================
# Sweep A: noise σ
# ============================================================

SIGMA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]

def run_sigma_sweep(seeds, quick=False):
    sigmas = [0.3, 0.9, 1.5] if quick else SIGMA_VALUES
    rows = []
    for σ in sigmas:
        print(f"  [sigma sweep] σ={σ:.1f} ... ", end="", flush=True)
        t0 = time.time()
        cfg = Config(overlap_std=σ, transition_steps=8)
        if quick: cfg.epochs = 100
        r = run_one_condition(cfg, seeds)
        r["sigma"] = σ
        rows.append(r)
        print(f"MSE_impr={r['mse_impr_pct']:+.1f}%  ΔH={r['dh_mean']:.3f}  ({time.time()-t0:.0f}s)")
    return rows


# ============================================================
# Sweep B: transition_steps
# ============================================================

TRANSITION_VALUES = [2, 4, 8, 16, 24]

def run_transition_sweep(seeds, quick=False):
    tvals = [2, 8, 24] if quick else TRANSITION_VALUES
    rows = []
    for ts in tvals:
        print(f"  [transition sweep] steps={ts} ... ", end="", flush=True)
        t0 = time.time()
        cfg = Config(overlap_std=0.9, transition_steps=ts)
        if quick: cfg.epochs = 100
        r = run_one_condition(cfg, seeds)
        r["transition_steps"] = ts
        rows.append(r)
        print(f"MSE_impr={r['mse_impr_pct']:+.1f}%  ΔH={r['dh_mean']:.3f}  ({time.time()-t0:.0f}s)")
    return rows


# ============================================================
# Save CSV
# ============================================================

def save_csv(rows: list, path: str, key_col: str):
    if not rows: return
    fieldnames = [key_col] + [k for k in rows[0] if k != key_col]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"  → saved: {path}")


# ============================================================
# Figures
# ============================================================

def plot_sigma_sweep(rows: list, save_path: str):
    sigmas   = [r["sigma"]      for r in rows]
    impr     = [r["mse_impr_pct"] for r in rows]
    dh       = [r["dh_mean"]    for r in rows]
    dh_std   = [r["dh_std"]     for r in rows]
    sh       = [r["stable_h_mean"] for r in rows]

    fig = plt.figure(figsize=(12, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    clr_main  = "#4A7BA7"
    clr_acc   = "#E07B54"
    clr_green = "#5BAD72"

    # — Panel 1: MSE improvement
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(range(len(sigmas)), impr,
                   color=[clr_main if v > 0 else clr_acc for v in impr],
                   edgecolor="white", linewidth=0.6)
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_xticks(range(len(sigmas)))
    ax1.set_xticklabels([f"{s}" for s in sigmas], fontsize=9)
    ax1.set_xlabel("Noise σ (overlap_std)", fontsize=10)
    ax1.set_ylabel("MSE Improvement over StdMoE (%)", fontsize=9)
    ax1.set_title("Prediction Accuracy\nvs Noise Level", fontsize=10, fontweight="bold")
    for bar, val in zip(bars, impr):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.5 if val >= 0 else -1.5),
                 f"{val:+.1f}%", ha="center", va="bottom", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # — Panel 2: ΔH (homeomorphic fixation signature)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(sigmas, dh, "o-", color=clr_main, lw=2, ms=7, zorder=3)
    ax2.fill_between(sigmas,
                     [d - e for d, e in zip(dh, dh_std)],
                     [d + e for d, e in zip(dh, dh_std)],
                     alpha=0.18, color=clr_main)
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Noise σ", fontsize=10)
    ax2.set_ylabel("ΔH  (Trans H − Stable H)", fontsize=9)
    ax2.set_title("Homeomorphic Fixation\nStrength vs Noise Level", fontsize=10, fontweight="bold")

    # Annotate three zones
    y_top = max(dh) * 1.25 if max(dh) > 0 else 0.3
    ax2.annotate("Info\nStarvation", xy=(sigmas[0], dh[0]),
                 xytext=(sigmas[0] + 0.05, y_top * 0.7),
                 fontsize=7.5, color="#888", arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))
    mid_idx = int(len(sigmas) / 2)
    ax2.annotate("Intermediate\nRegime", xy=(sigmas[mid_idx], dh[mid_idx]),
                 xytext=(sigmas[mid_idx] - 0.15, y_top * 1.0),
                 fontsize=7.5, color=clr_main, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=clr_main, lw=0.8))
    ax2.annotate("High-noise\nCollapse", xy=(sigmas[-1], dh[-1]),
                 xytext=(sigmas[-1] - 0.5, y_top * 0.7),
                 fontsize=7.5, color="#888", arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))
    ax2.grid(alpha=0.3)

    # — Panel 3: Stable Entropy (fixation depth)
    ax3 = fig.add_subplot(gs[2])
    sh_std = [r["stable_h_std"] for r in rows]
    ax3.plot(sigmas, sh, "s-", color=clr_green, lw=2, ms=7, zorder=3)
    ax3.fill_between(sigmas,
                     [s - e for s, e in zip(sh, sh_std)],
                     [s + e for s, e in zip(sh, sh_std)],
                     alpha=0.18, color=clr_green)
    ax3.axhline(np.log(3), color="#ccc", lw=0.8, ls=":", label="log(3) uniform")
    ax3.set_xlabel("Noise σ", fontsize=10)
    ax3.set_ylabel("Stable-phase Entropy (Nomadic Full)", fontsize=9)
    ax3.set_title("Fixation Depth\nvs Noise Level", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=7.5)
    ax3.grid(alpha=0.3)

    fig.suptitle("§5.1 Pressure Sweep A — Noise σ\n"
                 "\"Intermediate change pressure is the operational regime of Nomadic Routing\"",
                 fontsize=10, y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → figure saved: {save_path}")


def plot_transition_sweep(rows: list, save_path: str):
    tvals  = [r["transition_steps"] for r in rows]
    impr   = [r["mse_impr_pct"] for r in rows]
    dh     = [r["dh_mean"]      for r in rows]
    dh_std = [r["dh_std"]       for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    clr_main = "#4A7BA7"; clr_acc = "#E07B54"

    ax1.bar(range(len(tvals)), impr,
            color=[clr_main if v > 0 else clr_acc for v in impr],
            edgecolor="white", linewidth=0.6)
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_xticks(range(len(tvals)))
    ax1.set_xticklabels([str(t) for t in tvals], fontsize=9)
    ax1.set_xlabel("Transition Steps", fontsize=10)
    ax1.set_ylabel("MSE Improvement (%)", fontsize=9)
    ax1.set_title("Prediction Accuracy\nvs Transition Speed", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(tvals, dh, "o-", color=clr_main, lw=2, ms=7)
    ax2.fill_between(tvals,
                     [d - e for d, e in zip(dh, dh_std)],
                     [d + e for d, e in zip(dh, dh_std)],
                     alpha=0.18, color=clr_main)
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Transition Steps", fontsize=10)
    ax2.set_ylabel("ΔH", fontsize=9)
    ax2.set_title("Homeomorphic Fixation\nvs Transition Speed", fontsize=10, fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.suptitle("§5.1 Pressure Sweep B — Transition Speed (σ=0.9 fixed)\n"
                 "Reference: §4.8 Gradual (steps=24) identified as structural failure mode",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → figure saved: {save_path}")


# ============================================================
# Summary text
# ============================================================

def write_summary(sigma_rows, trans_rows, save_path):
    lines = ["=" * 68,
             "PRESSURE SWEEP — Summary for §5.1",
             "=" * 68, ""]

    if sigma_rows:
        lines += ["Sweep A: Noise σ (transition_steps=8 fixed)",
                  "-" * 50]
        lines.append(f"{'σ':>6}  {'StdMSE':>8}  {'NomMSE':>8}  {'Impr%':>7}  {'ΔH':>7}  {'StableH':>8}")
        for r in sigma_rows:
            lines.append(
                f"{r['sigma']:>6.2f}  {r['std_mse_mean']:>8.4f}  {r['nom_mse_mean']:>8.4f}  "
                f"{r['mse_impr_pct']:>+7.1f}%  {r['dh_mean']:>7.3f}  {r['stable_h_mean']:>8.3f}"
            )
        best = max(sigma_rows, key=lambda r: r["dh_mean"])
        lines += ["",
                  f"Peak ΔH at σ={best['sigma']:.2f} (ΔH={best['dh_mean']:.3f}, "
                  f"Impr={best['mse_impr_pct']:+.1f}%)",
                  "→ This σ defines the center of the 'intermediate regime'.", ""]

    if trans_rows:
        lines += ["Sweep B: Transition Steps (σ=0.9 fixed)",
                  "-" * 50]
        lines.append(f"{'steps':>7}  {'StdMSE':>8}  {'NomMSE':>8}  {'Impr%':>7}  {'ΔH':>7}")
        for r in trans_rows:
            lines.append(
                f"{r['transition_steps']:>7}  {r['std_mse_mean']:>8.4f}  {r['nom_mse_mean']:>8.4f}  "
                f"{r['mse_impr_pct']:>+7.1f}%  {r['dh_mean']:>7.3f}"
            )
        lines += ["", "→ Compare steps=24 result to §4.8 Gradual finding.", ""]

    lines += ["=" * 68,
              "Concept.md interpretation:",
              "  σ low  → Δx≈0, R_sync starvation → spiral collapses to point",
              "  σ mid  → three forces (Sync/Dogma/Trans) balanced → fixation emerges",
              "  σ high → Φ signal corrupted by noise → spiral diverges",
              "=" * 68]
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → summary saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Nomadic Routing — Pressure Sweep")
    parser.add_argument("--sweep", choices=["sigma", "transition", "both"], default="both")
    parser.add_argument("--quick", action="store_true",
                        help="Fast sanity check: 1 seed, 3 sigma values, 100 epochs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    SEEDS = [42] if args.quick else [42, 123, 456]
    os.makedirs("outputs_sweep", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Nomadic Routing — Prediction Pressure Sweep")
    print(f"  seeds={SEEDS}  device={args.device}  quick={args.quick}")
    print(f"{'='*60}\n")

    sigma_rows = []
    trans_rows = []

    if args.sweep in ("sigma", "both"):
        print("▶ Sweep A: Noise σ")
        sigma_rows = run_sigma_sweep(SEEDS, quick=args.quick)
        save_csv(sigma_rows, "outputs_sweep/sigma_sweep_results.csv", "sigma")
        plot_sigma_sweep(sigma_rows, "outputs_sweep/fig_sigma_sweep.png")

    if args.sweep in ("transition", "both"):
        print("\n▶ Sweep B: Transition Steps (σ=0.9 fixed)")
        trans_rows = run_transition_sweep(SEEDS, quick=args.quick)
        save_csv(trans_rows, "outputs_sweep/transition_sweep_results.csv", "transition_steps")
        plot_transition_sweep(trans_rows, "outputs_sweep/fig_transition_sweep.png")

    write_summary(sigma_rows, trans_rows, "outputs_sweep/sweep_summary.txt")

    print("\n✅ Sweep complete.")
    print("   outputs_sweep/")
    if sigma_rows:
        print("     sigma_sweep_results.csv")
        print("     fig_sigma_sweep.png        ← §5.1 figure (σ sweep)")
    if trans_rows:
        print("     transition_sweep_results.csv")
        print("     fig_transition_sweep.png   ← §5.1 figure (transition sweep)")
    print("     sweep_summary.txt")


if __name__ == "__main__":
    main()
