"""
run_rl_sweep.py
─────────────────────────────────────────────────────────────────────────────
RL-only Sigma Sweep + Transition Sweep
기존 SL/StdMoE 결과를 CSV로 받아서 RL 결과와 합산 비교.

사용법:
  # SL 결과 없이 RL만 sweep (StdMoE는 내부 재실행)
  python run_rl_sweep.py --sweep sigma

  # 기존 SL 결과 CSV 제공 → RL만 새로 훈련
  python run_rl_sweep.py --sweep sigma --sl_csv sigma_sweep_sl.csv

  # transition sweep
  python run_rl_sweep.py --sweep transition --sl_csv transition_sweep_sl.csv

  # 둘 다
  python run_rl_sweep.py --sweep both --sl_sigma_csv s.csv --sl_trans_csv t.csv

  # 빠른 확인
  python run_rl_sweep.py --sweep sigma --quick

  # CUDA
  python run_rl_sweep.py --sweep sigma --device cuda

SL CSV 포맷 (run_pressure_sweep.py 출력 그대로):
  sigma_sweep_results.csv  : sigma, std_mse_mean, nom_mse_mean, mse_impr_pct, dh_mean, ...
  transition_sweep_results.csv : transition_steps, std_mse_mean, nom_mse_mean, ...

출력:
  outputs_rl_sweep/
    rl_sigma_sweep.csv
    rl_transition_sweep.csv
    fig_rl_sigma_sweep.png      ← StdMoE / SL / RL 3-way 비교
    fig_rl_transition_sweep.png
    fig_rl_reward_dynamics.png  ← σ별 R_total / entropy / transition_ratio
    sweep_summary.txt
─────────────────────────────────────────────────────────────────────────────
"""

import os
import csv
import time
import random
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================
# Seed / Config
# ============================================================

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@dataclass
class Config:
    seed: int = 42
    device: str = "cpu"
    input_dim: int = 2
    output_dim: int = 1
    overlap_std: float = 0.9
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
    transition_steps: int = 8
    num_regimes: int = 3
    # hybrid delta
    ema_decay: float = 0.80
    err_baseline_momentum: float = 0.85
    w_env: float = 1.0
    w_err: float = 2.0
    # structure loss (backbone용)
    gamma_diversity: float = 0.08
    lambda_sep: float = 0.08
    lambda_cons: float = 0.03
    lambda_load: float = 0.03
    tau_k_min: int = 3
    tau_k_penalty: float = 0.05
    # phi
    phi_scale_env: float = 1.0
    phi_scale_err: float = 1.5
    phi_scale_explain: float = 2.0
    phi_scale_gap: float = 1.0
    temp_stable: float = 0.30
    temp_transition: float = 1.00
    use_hard_switch: bool = True
    phi_hard_threshold: float = 0.35
    # dynamic tau
    tau_min: float = 2.0
    tau_max: float = 8.0
    tau_var_scale: float = 6.0
    tau_var_window: int = 8
    # RL
    rl_alpha_sync:            float = 2.0
    rl_beta_dogma:            float = 0.3
    rl_gamma_nomad:           float = 0.5
    rl_baseline_momentum:     float = 0.95
    rl_entropy_coef:          float = 0.15
    rl_policy_lr:             float = 2e-4
    rl_policy_hidden_dim:     int   = 64
    rl_policy_mix_weight:     float = 0.25
    rl_clip_grad:             float = 1.0
    rl_transition_phi_threshold: float = 0.90
    save_dir: str = "outputs_rl_sweep"


# ============================================================
# Data
# ============================================================

REGIME_TO_ID = {"A": 0, "B": 1, "C": 2}

def sample_regime_x(regime, n, std, device):
    centers = {"A": (2.5, 2.5), "B": (-2.5, -2.5), "C": (2.5, -2.5)}
    c = torch.tensor(centers[regime], device=device)
    return std * torch.randn(n, 2, device=device) + c

def regime_function(x, regime):
    x1, x2 = x[:, 0], x[:, 1]
    if regime == "A": return (x1 + x2).unsqueeze(-1)
    if regime == "B": return (x1 - x2).unsqueeze(-1)
    if regime == "C": return (-x1 + 0.5 * x2).unsqueeze(-1)

def generate_phase_sequence(cfg, cycles):
    active = ["A", "B", "C"]
    xs, ys, rs, tags = [], [], [], []
    for _ in range(cycles):
        for i, curr in enumerate(active):
            nxt = active[(i + 1) % 3]
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
                xs.append(xm); ys.append(ym)
                rs.append(torch.full((cfg.phase_batch_size,), REGIME_TO_ID[curr if alpha < 0.5 else nxt],
                                      dtype=torch.long, device=cfg.device))
                tags.extend([f"transition_{curr}_to_{nxt}"] * cfg.phase_batch_size)
    return torch.cat(xs), torch.cat(ys), torch.cat(rs), tags

def iter_batches(X, Y, R, bs):
    n = X.size(0)
    for s in range(0, n, bs):
        yield X[s:min(s+bs,n)], Y[s:min(s+bs,n)], R[s:min(s+bs,n)]


# ============================================================
# Models
# ============================================================

class Expert(nn.Module):
    def __init__(self, in_d, h, out_d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d,h), nn.Tanh(),
                                  nn.Linear(h,h),    nn.Tanh(),
                                  nn.Linear(h,out_d))
    def forward(self, x): return self.net(x)

class GateNet(nn.Module):
    def __init__(self, in_d, h, n_exp):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d+2,h), nn.ReLU(),
                                  nn.Linear(h,h),      nn.ReLU(),
                                  nn.Linear(h,n_exp))
    def forward(self, x, dh, de, temp):
        logits = self.net(torch.cat([x, dh, de], dim=-1))
        return F.softmax(logits / temp, dim=-1), logits

class PolicyNetRL(nn.Module):
    def __init__(self, in_d, h, n_exp):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_d+5,h), nn.ReLU(),
                                     nn.Linear(h,h),      nn.ReLU())
        self.ss_head   = nn.Linear(h, 2)
        self.tgt_head  = nn.Linear(h, n_exp)
        self.mode_head = nn.Linear(h, 2)

    def forward(self, x):
        h = self.shared(x)
        return self.ss_head(h), self.tgt_head(h), self.mode_head(h)

    def sample_action(self, pi):
        ss_l, tg_l, md_l = self.forward(pi)
        ss_d = Categorical(logits=ss_l)
        tg_d = Categorical(logits=tg_l)
        md_d = Categorical(logits=md_l)
        ss_a = ss_d.sample(); tg_a = tg_d.sample(); md_a = md_d.sample()
        lp  = ss_d.log_prob(ss_a) + tg_d.log_prob(tg_a) + md_d.log_prob(md_a)
        ent = (ss_d.entropy() + tg_d.entropy() + md_d.entropy()) / 3.0
        return (ss_a, tg_a, md_a), lp, ent

class NomadicMoE_RL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.experts = nn.ModuleList([Expert(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
                                       for _ in range(cfg.num_experts)])
        self.gate   = GateNet(cfg.input_dim, cfg.gate_hidden_dim, cfg.num_experts)
        self.policy = PolicyNetRL(cfg.input_dim, cfg.rl_policy_hidden_dim, cfg.num_experts)

    def forward(self, x, dh, de, temp):
        gp, gl = self.gate(x, dh, de, temp)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        return (gp.unsqueeze(-1) * eo).sum(1), gp, gl, eo

class StandardMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.experts = nn.ModuleList([Expert(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
                                       for _ in range(cfg.num_experts)])
        self.gate = nn.Sequential(nn.Linear(cfg.input_dim, cfg.gate_hidden_dim), nn.ReLU(),
                                   nn.Linear(cfg.gate_hidden_dim, cfg.gate_hidden_dim), nn.ReLU(),
                                   nn.Linear(cfg.gate_hidden_dim, cfg.num_experts))
    def forward(self, x):
        gp = F.softmax(self.gate(x), dim=-1)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        return (gp.unsqueeze(-1) * eo).sum(1), gp


# ============================================================
# Utilities
# ============================================================

class HybridDeltaTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.recent = deque(maxlen=cfg.tau_var_window)
        self.reset()

    def reset(self):
        self.prev_mean = None; self.err_ema = None; self.err_base = None
        self.recent.clear()

    def compute(self, x, mse):
        xm = x.mean(0, keepdim=True)
        de = 0.0 if self.prev_mean is None else float(torch.norm(xm - self.prev_mean).item())
        self.prev_mean = xm.detach(); self.recent.append(de)
        err = float(mse.detach().item())
        if self.err_ema is None:
            self.err_ema = err; self.err_base = err; derr = 0.0
        else:
            self.err_ema  = self.cfg.ema_decay * self.err_ema  + (1-self.cfg.ema_decay) * err
            self.err_base = self.cfg.err_baseline_momentum * self.err_base + \
                            (1-self.cfg.err_baseline_momentum) * self.err_ema
            derr = max(0.0, self.err_ema - self.err_base)
        dh_val = float(torch.tanh(torch.tensor(self.cfg.w_env*de + self.cfg.w_err*derr)).item())
        sig2   = float(np.var(list(self.recent))) if len(self.recent) >= 2 else 0.0
        tau    = self.cfg.tau_min + (self.cfg.tau_max-self.cfg.tau_min)/(1+self.cfg.tau_var_scale*sig2)
        dh_t = torch.full((x.size(0),1), dh_val, device=x.device)
        de_t = torch.full((x.size(0),1), derr,   device=x.device)
        return dh_t, de, derr, dh_val, sig2, tau

def explanation_signals(yb, yhat, eo, gp):
    per = ((eo - yb.unsqueeze(1))**2).mean(-1)
    top1_err = per.gather(1, gp.argmax(-1, keepdim=True)).mean()
    best_err  = per.min(1).values.mean()
    return F.mse_loss(yhat, yb), torch.relu(top1_err - best_err)

def phi_signal(de, derr, expl, gap, cfg):
    return torch.tanh(cfg.phi_scale_env  * torch.tensor(de,   device=expl.device) +
                       cfg.phi_scale_err  * torch.tensor(derr, device=expl.device) +
                       cfg.phi_scale_explain * expl.detach() +
                       cfg.phi_scale_gap  * gap.detach())

def adaptive_temp(phi, cfg):
    return cfg.temp_stable + (cfg.temp_transition - cfg.temp_stable) * float(phi.mean().item())

def build_policy_input(xb, dh, de_t, phi, sig2, tau):
    phi_t = torch.full((xb.size(0),1), float(phi.mean().item()), device=xb.device)
    s2_t  = torch.full((xb.size(0),1), float(np.tanh(sig2*10)),      device=xb.device)
    tau_t = torch.full((xb.size(0),1), float(np.tanh((tau-5)/5)),    device=xb.device)
    return torch.cat([xb.mean(0,keepdim=True).expand(xb.size(0),-1), dh, de_t, phi_t, s2_t, tau_t], dim=-1)

def gate_entropy(gp):
    return -(gp * (gp+1e-8).log()).sum(-1)

def load_balance_loss(gp):
    E = gp.size(-1); mg = gp.mean(0)
    t1 = torch.tensor([(gp.argmax(-1)==i).float().mean() for i in range(E)], device=gp.device)
    return E * (t1 * mg).sum()

def diversity_loss(eo):
    E = eo.size(1); loss = 0.0; n = 0
    for i in range(E):
        for j in range(i+1,E):
            loss += F.cosine_similarity(eo[:,i], eo[:,j], dim=-1).mean(); n+=1
    return loss / max(n,1)

def regime_gate_stats(gp, rb, n_reg=3):
    means = [gp[(rb==r)].mean(0) for r in range(n_reg) if (rb==r).sum()>0]
    if len(means)<2:
        return torch.tensor(0.,device=gp.device), torch.tensor(0.,device=gp.device)
    sep = sum(F.cosine_similarity(means[i].unsqueeze(0), means[j].unsqueeze(0)).mean()
              for i in range(len(means)) for j in range(i+1,len(means)))
    sep /= max(len(means)*(len(means)-1)//2, 1)
    cons = sum(gp[(rb==r)].var(0).sum() for r in range(n_reg) if (rb==r).sum()>1)
    return sep, cons

class DwellReg:
    def __init__(self, cfg):
        self.pen = cfg.tau_k_penalty; self.cur = None; self.cnt = 0
    def reset(self): self.cur = None; self.cnt = 0
    def compute(self, gp, tau):
        dom = int(torch.bincount(gp.argmax(-1), minlength=gp.size(-1)).argmax().item())
        self.cnt = (self.cnt+1) if dom==self.cur else 1; self.cur = dom
        ent = gate_entropy(gp).mean()
        if self.cnt <= float(tau): return -self.pen * ent
        return min((self.cnt-tau)*self.pen, self.pen*10) * ent

class RunningBaseline:
    def __init__(self, m=0.95): self.m = m; self.v = None
    def update(self, r):
        self.v = r if self.v is None else self.m*self.v + (1-self.m)*r
    def advantage(self, r): return r - (self.v or 0.0)


# ============================================================
# R_total
# ============================================================

def compute_r_total(mse_after, mr, phi_val, cfg):
    r_sync  = 1.0 / (1.0 + mse_after)
    p_dogma = float((mr.mean(0)**2).sum().item())
    h_gate  = float(gate_entropy(mr).mean().item())
    is_trans = float(phi_val > cfg.rl_transition_phi_threshold)
    r_nomad  = h_gate * is_trans
    r_total  = cfg.rl_alpha_sync*r_sync - cfg.rl_beta_dogma*p_dogma + cfg.rl_gamma_nomad*r_nomad
    return r_total, {"r_sync": r_sync, "p_dogma": p_dogma,
                     "r_nomad": r_nomad, "r_total": r_total, "is_transition": is_trans}


# ============================================================
# Train
# ============================================================

def train_stdmoe(cfg, Xtr, Ytr, Rtr):
    model = StandardMoE(cfg).to(cfg.device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb, _ in iter_batches(Xtr, Ytr, Rtr, cfg.phase_batch_size):
            opt.zero_grad(); yh, _ = model(xb)
            F.mse_loss(yh, yb).backward(); opt.step()
    return model


def train_nomadic_rl(cfg, Xtr, Ytr, Rtr) -> Tuple[nn.Module, dict]:
    model = NomadicMoE_RL(cfg).to(cfg.device)
    p_ids = set(id(p) for p in model.policy.parameters())
    bb_p  = [p for p in model.parameters() if id(p) not in p_ids]
    opt_bb  = torch.optim.Adam(bb_p, lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_pol = torch.optim.Adam(model.policy.parameters(), lr=cfg.rl_policy_lr)
    baseline = RunningBaseline(cfg.rl_baseline_momentum)
    dwell    = DwellReg(cfg)

    history = {k: [] for k in ["r_total","r_sync","p_dogma","r_nomad",
                                "policy_entropy","mse","is_transition_ratio","advantage"]}

    for ep in range(cfg.epochs):
        model.train()
        tracker = HybridDeltaTracker(cfg); tracker.reset(); dwell.reset()
        ep_r, ep_ent, ep_mse, ep_tr, ep_adv = [], [], [], [], []

        for xb, yb, rb in iter_batches(Xtr, Ytr, Rtr, cfg.phase_batch_size):
            # 1. Δx
            z = torch.zeros(xb.size(0),1,device=cfg.device)
            with torch.no_grad():
                wy,_,_,_ = model(xb,z,z,cfg.temperature)
                wmse = F.mse_loss(wy,yb)
            dh, de, derr, dh_val, sig2, tau = tracker.compute(xb, wmse)
            de_t = torch.full((xb.size(0),1), derr, device=cfg.device)

            # 2. phi
            with torch.no_grad():
                py,pgp,_,peo = model(xb,dh,de_t,cfg.temperature)
            ee, gap = explanation_signals(yb,py,peo,pgp)
            phi     = phi_signal(de,derr,ee,gap,cfg)
            phi_val = float(phi.mean().item())
            tmp     = adaptive_temp(phi,cfg)

            # 3. action 샘플링
            pi   = build_policy_input(xb,dh,de_t,phi,sig2,tau)
            (ss_a,tgt_a,mode_a), lp, pol_ent = model.policy.sample_action(pi[:1])

            # 4. routing 적용
            _,gp,_,eo = model(xb,dh,de_t,tmp)
            eff_mix = cfg.rl_policy_mix_weight * float(ss_a.item())
            if eff_mix > 0:
                toh = F.one_hot(tgt_a.view(-1)[0], cfg.num_experts).float() \
                               .unsqueeze(0).expand(xb.size(0),-1)
                mr  = (1-eff_mix)*gp + eff_mix*((toh-gp).detach()+gp)
            else:
                mr = gp
            if (mode_a.item()==1 and dh_val<=cfg.phi_hard_threshold and cfg.use_hard_switch):
                mr = F.one_hot(mr.argmax(-1), cfg.num_experts).float()

            yh    = (mr.unsqueeze(-1)*eo).sum(1)
            mse_l = F.mse_loss(yh,yb)

            # 5. R_total
            r_total, rbd = compute_r_total(float(mse_l.detach().item()), mr.detach(), phi_val, cfg)

            # 6. advantage
            baseline.update(r_total)
            adv = baseline.advantage(r_total)
            pol_loss = (-adv*lp - cfg.rl_entropy_coef*pol_ent).mean()

            # 7. backbone loss
            _,gap2 = explanation_signals(yb,yh,eo,mr)
            sl,cl  = regime_gate_stats(mr,rb,cfg.num_regimes)
            bb_loss = (mse_l
                       + cfg.gamma_diversity * diversity_loss(eo)
                       + cfg.lambda_sep      * sl
                       + cfg.lambda_cons     * cl
                       + cfg.lambda_load     * load_balance_loss(mr)
                       - dwell.compute(mr,tau))

            # 8. 분리 update
            opt_bb.zero_grad()
            bb_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(bb_p, cfg.rl_clip_grad)
            opt_bb.step()

            opt_pol.zero_grad()
            pol_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.policy.parameters()), cfg.rl_clip_grad)
            opt_pol.step()

            ep_r.append(r_total); ep_ent.append(float(pol_ent.mean().item()))
            ep_mse.append(float(mse_l.detach().item()))
            ep_tr.append(rbd["is_transition"]); ep_adv.append(adv)

        history["r_total"].append(float(np.mean(ep_r)))
        history["r_sync"].append(float(np.mean([1/(1+m) for m in ep_mse])))
        history["p_dogma"].append(float("nan"))
        history["r_nomad"].append(float("nan"))
        history["policy_entropy"].append(float(np.mean(ep_ent)))
        history["mse"].append(float(np.mean(ep_mse)))
        history["is_transition_ratio"].append(float(np.mean(ep_tr)))
        history["advantage"].append(float(np.mean(ep_adv)))

    return model, history


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def eval_seq(model, X, Y, R, tags, cfg, is_nomadic: bool):
    model.eval()
    ys, phase_ents = [], []
    tracker = HybridDeltaTracker(cfg) if is_nomadic else None
    if tracker: tracker.reset()
    for bi, (xb,yb,rb) in enumerate(iter_batches(X,Y,R,cfg.phase_batch_size)):
        if is_nomadic:
            z = torch.zeros(xb.size(0),1,device=cfg.device)
            wy,_,_,_ = model(xb,z,z,cfg.temperature)
            wmse = F.mse_loss(wy,yb)
            dh,de,derr,_,sig2,tau = tracker.compute(xb,wmse)
            de_t = torch.full((xb.size(0),1),derr,device=cfg.device)
            py,pgp,_,peo = model(xb,dh,de_t,cfg.temperature)
            ee,gap = explanation_signals(yb,py,peo,pgp)
            phi    = phi_signal(de,derr,ee,gap,cfg)
            tmp    = adaptive_temp(phi,cfg)
            yh,gp,_,_ = model(xb,dh,de_t,tmp)
        else:
            yh, gp = model(xb)
        ys.append(yh)
        phase_ents.append((tags[bi*cfg.phase_batch_size], float(gate_entropy(gp).mean().item())))
    Yhat = torch.cat(ys)
    mse  = float(F.mse_loss(Yhat,Y).item())
    s_h  = [e for t,e in phase_ents if t.startswith("stable_")]
    tr_h = [e for t,e in phase_ents if t.startswith("transition_")]
    return mse, (float(np.mean(s_h)) if s_h else float("nan")), (float(np.mean(tr_h)) if tr_h else float("nan"))


# ============================================================
# Run one condition (RL only + optional StdMoE)
# ============================================================

def run_rl_condition(cfg: Config, seeds: List[int],
                     run_std: bool = True) -> dict:
    """
    RL 전용 실행.
    run_std=True 이면 StdMoE도 재실행 (SL CSV 없을 때 사용).
    """
    std_mses, rl_mses, rl_shs, rl_ths, rl_hists = [], [], [], [], []

    for seed in seeds:
        set_seed(seed)
        cfg_s = Config(**{k: v for k, v in cfg.__dict__.items()}); cfg_s.seed = seed
        Xtr,Ytr,Rtr,_    = generate_phase_sequence(cfg_s, cfg.phase_train_cycles)
        Xte,Yte,Rte,tags = generate_phase_sequence(cfg_s, cfg.phase_test_cycles)

        if run_std:
            std = train_stdmoe(cfg_s, Xtr, Ytr, Rtr)
            std_mse,_,_ = eval_seq(std,Xte,Yte,Rte,tags,cfg_s,is_nomadic=False)
            std_mses.append(std_mse)

        rl, hist = train_nomadic_rl(cfg_s, Xtr, Ytr, Rtr)
        rl_mse,rl_sh,rl_th = eval_seq(rl,Xte,Yte,Rte,tags,cfg_s,is_nomadic=True)
        rl_mses.append(rl_mse); rl_shs.append(rl_sh)
        rl_ths.append(rl_th);   rl_hists.append(hist)

    rl_m = float(np.mean(rl_mses))
    std_m = float(np.mean(std_mses)) if std_mses else None
    dh_vals = [th-sh for th,sh in zip(rl_ths,rl_shs) if not (np.isnan(th) or np.isnan(sh))]

    return {
        "rl_mse_mean":  rl_m,
        "rl_mse_std":   float(np.std(rl_mses)),
        "std_mse_mean": std_m,
        "rl_impr_pct":  (std_m-rl_m)/max(std_m,1e-9)*100 if std_m else float("nan"),
        "rl_dh_mean":   float(np.mean(dh_vals)) if dh_vals else float("nan"),
        "rl_dh_std":    float(np.std(dh_vals))  if dh_vals else float("nan"),
        "rl_stable_h":  float(np.mean(rl_shs)),
        "rl_trans_h":   float(np.mean(rl_ths)),
        "rl_histories": rl_hists,
    }


# ============================================================
# CSV helpers
# ============================================================

def load_sl_csv(path: str) -> List[dict]:
    """기존 SL/StdMoE sweep 결과 CSV 읽기 (run_pressure_sweep.py 출력 포맷)"""
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if v not in ('', 'nan') else float("nan")
                         for k,v in row.items()})
    print(f"  [SL CSV] loaded {len(rows)} rows from {path}")
    return rows


def save_csv(rows: list, path: str, key_col: str):
    if not rows: return
    all_keys = list(rows[0].keys())
    fieldnames = [key_col] + [k for k in all_keys if k != key_col]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k,v in r.items()})
    print(f"  → saved: {path}")


# ============================================================
# Merge RL result with SL CSV row
# ============================================================

def merge_row_sigma(sigma: float, rl_result: dict, sl_rows: List[dict]) -> dict:
    """RL 결과 + SL CSV에서 해당 sigma 행을 찾아 병합"""
    sl_row = next((r for r in sl_rows if abs(r.get("sigma",999)-sigma)<1e-6), None)

    row = {"sigma": sigma,
           "rl_mse_mean":  rl_result["rl_mse_mean"],
           "rl_mse_std":   rl_result["rl_mse_std"],
           "rl_dh_mean":   rl_result["rl_dh_mean"],
           "rl_dh_std":    rl_result["rl_dh_std"],
           "rl_stable_h":  rl_result["rl_stable_h"],
           "rl_trans_h":   rl_result["rl_trans_h"],
           }
    if sl_row:
        # run_pressure_sweep.py의 컬럼명 매핑
        row["std_mse_mean"] = sl_row.get("std_mse_mean", rl_result.get("std_mse_mean") or float("nan"))
        row["sl_mse_mean"]  = sl_row.get("nom_mse_mean", float("nan"))
        row["sl_impr_pct"]  = sl_row.get("mse_impr_pct", float("nan"))
        row["sl_dh_mean"]   = sl_row.get("dh_mean",      float("nan"))
        row["sl_stable_h"]  = sl_row.get("stable_h_mean",float("nan"))
        std_m = row["std_mse_mean"]
        rl_m  = row["rl_mse_mean"]
        row["rl_impr_pct"]  = (std_m-rl_m)/max(std_m,1e-9)*100 if not np.isnan(std_m) else float("nan")
        sl_m  = row["sl_mse_mean"]
        row["rl_vs_sl_pct"] = (sl_m-rl_m)/max(sl_m,1e-9)*100 if not np.isnan(sl_m) else float("nan")
    else:
        std_m = rl_result.get("std_mse_mean")
        row["std_mse_mean"] = std_m if std_m else float("nan")
        row["sl_mse_mean"]  = float("nan")
        row["sl_impr_pct"]  = float("nan")
        row["sl_dh_mean"]   = float("nan")
        row["sl_stable_h"]  = float("nan")
        row["rl_impr_pct"]  = rl_result["rl_impr_pct"]
        row["rl_vs_sl_pct"] = float("nan")
    return row


def merge_row_trans(steps: int, rl_result: dict, sl_rows: List[dict]) -> dict:
    sl_row = next((r for r in sl_rows if abs(r.get("transition_steps",999)-steps)<1e-6), None)
    row = {"transition_steps": steps,
           "rl_mse_mean":  rl_result["rl_mse_mean"],
           "rl_mse_std":   rl_result["rl_mse_std"],
           "rl_dh_mean":   rl_result["rl_dh_mean"],
           "rl_dh_std":    rl_result["rl_dh_std"],
           }
    if sl_row:
        row["std_mse_mean"] = sl_row.get("std_mse_mean", float("nan"))
        row["sl_mse_mean"]  = sl_row.get("nom_mse_mean", float("nan"))
        row["sl_impr_pct"]  = sl_row.get("mse_impr_pct", float("nan"))
        row["sl_dh_mean"]   = sl_row.get("dh_mean",      float("nan"))
        std_m = row["std_mse_mean"]; rl_m = row["rl_mse_mean"]
        row["rl_impr_pct"]  = (std_m-rl_m)/max(std_m,1e-9)*100 if not np.isnan(std_m) else float("nan")
        sl_m = row["sl_mse_mean"]
        row["rl_vs_sl_pct"] = (sl_m-rl_m)/max(sl_m,1e-9)*100 if not np.isnan(sl_m) else float("nan")
    else:
        std_m = rl_result.get("std_mse_mean")
        row["std_mse_mean"] = std_m if std_m else float("nan")
        row["sl_mse_mean"]  = float("nan"); row["sl_impr_pct"] = float("nan")
        row["sl_dh_mean"]   = float("nan"); row["rl_impr_pct"] = rl_result["rl_impr_pct"]
        row["rl_vs_sl_pct"] = float("nan")
    return row


# ============================================================
# Sweeps
# ============================================================

SIGMA_VALUES      = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0]
TRANSITION_VALUES = [2, 4, 8, 16, 24]


def run_sigma_sweep(cfg_base: Config, seeds: List[int],
                    sl_rows: List[dict], quick: bool) -> List[dict]:
    sigmas = [0.3, 0.9, 1.5] if quick else SIGMA_VALUES
    rows = []
    all_histories = {}  # sigma → histories (figure용)

    for σ in sigmas:
        print(f"  [σ={σ:.1f}] RL training ... ", end="", flush=True)
        t0  = time.time()
        cfg = Config(**{k: v for k, v in cfg_base.__dict__.items()})
        cfg.overlap_std = σ
        if quick: cfg.epochs = 80

        # SL CSV에 해당 σ가 있으면 StdMoE 재실행 생략
        has_sl = any(abs(r.get("sigma",999)-σ)<1e-6 for r in sl_rows)
        result = run_rl_condition(cfg, seeds, run_std=not has_sl)

        row = merge_row_sigma(σ, result, sl_rows)
        rows.append(row)
        all_histories[σ] = result["rl_histories"]

        elapsed = time.time()-t0
        rl_vs = f"RL_vs_SL={row['rl_vs_sl_pct']:+.1f}%" if not np.isnan(row["rl_vs_sl_pct"]) else "SL_N/A"
        print(f"RL_MSE={result['rl_mse_mean']:.4f}  RL_ΔH={result['rl_dh_mean']:.3f}  "
              f"{rl_vs}  ({elapsed:.0f}s)")

    return rows, all_histories


def run_transition_sweep(cfg_base: Config, seeds: List[int],
                         sl_rows: List[dict], quick: bool) -> List[dict]:
    steps_list = [2, 8, 24] if quick else TRANSITION_VALUES
    rows = []
    all_histories = {}

    for ts in steps_list:
        print(f"  [steps={ts}] RL training ... ", end="", flush=True)
        t0  = time.time()
        cfg = Config(**{k: v for k, v in cfg_base.__dict__.items()})
        cfg.overlap_std = 0.9; cfg.transition_steps = ts
        if quick: cfg.epochs = 80

        has_sl = any(abs(r.get("transition_steps",999)-ts)<1e-6 for r in sl_rows)
        result = run_rl_condition(cfg, seeds, run_std=not has_sl)

        row = merge_row_trans(ts, result, sl_rows)
        rows.append(row)
        all_histories[ts] = result["rl_histories"]

        elapsed = time.time()-t0
        rl_vs = f"RL_vs_SL={row['rl_vs_sl_pct']:+.1f}%" if not np.isnan(row["rl_vs_sl_pct"]) else "SL_N/A"
        print(f"RL_MSE={result['rl_mse_mean']:.4f}  RL_ΔH={result['rl_dh_mean']:.3f}  "
              f"{rl_vs}  ({elapsed:.0f}s)")

    return rows, all_histories


# ============================================================
# Figures
# ============================================================

CLR = {"std": "#888888", "sl": "#4A7BA7", "rl": "#E07B54", "dh_sl": "#4A7BA7", "dh_rl": "#E07B54"}


def plot_sigma_sweep(rows: list, save_path: str, cfg: Config):
    sigmas = [r["sigma"] for r in rows]
    has_sl = not all(np.isnan(r["sl_mse_mean"]) for r in rows)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Row 0: 성능 패널 ─────────────────────────────────────

    # Panel (0,0): MSE
    ax = fig.add_subplot(gs[0,0])
    ax.plot(sigmas, [r["rl_mse_mean"] for r in rows], "s-",
            color=CLR["rl"], lw=2, ms=7, label="Nomadic-RL")
    if has_sl:
        ax.plot(sigmas, [r["sl_mse_mean"] for r in rows], "o-",
                color=CLR["sl"], lw=2, ms=7, label="Nomadic-SL")
    ax.plot(sigmas, [r["std_mse_mean"] for r in rows if not np.isnan(r["std_mse_mean"])],
            "^--", color=CLR["std"], lw=1.5, ms=6, label="StdMoE",
            alpha=0.7)
    ax.set_xlabel("Noise σ", fontsize=10); ax.set_ylabel("Seq MSE", fontsize=9)
    ax.set_title("MSE vs Noise Level", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel (0,1): MSE improvement
    ax = fig.add_subplot(gs[0,1])
    ax.plot(sigmas, [r["rl_impr_pct"] for r in rows], "s-",
            color=CLR["rl"], lw=2, ms=7, label="RL vs StdMoE")
    if has_sl:
        ax.plot(sigmas, [r["sl_impr_pct"] for r in rows], "o-",
                color=CLR["sl"], lw=2, ms=7, label="SL vs StdMoE")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Noise σ", fontsize=10); ax.set_ylabel("MSE Improvement (%)", fontsize=9)
    ax.set_title("Improvement over StdMoE", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel (0,2): RL vs SL delta
    ax = fig.add_subplot(gs[0,2])
    rl_vs_sl = [r["rl_vs_sl_pct"] for r in rows]
    if has_sl and not all(np.isnan(v) for v in rl_vs_sl):
        colors = [CLR["rl"] if v>=0 else "#888" for v in rl_vs_sl]
        ax.bar(range(len(sigmas)), rl_vs_sl, color=colors, edgecolor="white")
        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([f"{s}" for s in sigmas], fontsize=9)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        for i,v in enumerate(rl_vs_sl):
            if not np.isnan(v):
                ax.text(i, v+(abs(v)*0.05 if v>=0 else -abs(v)*0.1),
                        f"{v:+.1f}%", ha="center", fontsize=8)
    else:
        ax.text(0.5,0.5,"SL data not available", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_xlabel("Noise σ", fontsize=10); ax.set_ylabel("RL gain over SL (%)", fontsize=9)
    ax.set_title("RL vs SL Delta", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ── Row 1: ΔH 패널 ──────────────────────────────────────

    # Panel (1,0): ΔH
    ax = fig.add_subplot(gs[1,0])
    rl_dh     = [r["rl_dh_mean"] for r in rows]
    rl_dh_std = [r["rl_dh_std"]  for r in rows]
    ax.plot(sigmas, rl_dh, "s-", color=CLR["dh_rl"], lw=2, ms=7, label="RL ΔH", zorder=3)
    ax.fill_between(sigmas, [d-e for d,e in zip(rl_dh,rl_dh_std)],
                             [d+e for d,e in zip(rl_dh,rl_dh_std)],
                    alpha=0.15, color=CLR["dh_rl"])
    if has_sl:
        sl_dh = [r["sl_dh_mean"] for r in rows]
        ax.plot(sigmas, sl_dh, "o--", color=CLR["dh_sl"], lw=1.5, ms=6,
                label="SL ΔH", alpha=0.8)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Noise σ", fontsize=10); ax.set_ylabel("ΔH (Trans H − Stable H)", fontsize=9)
    ax.set_title("Homeomorphic Fixation\nvs Noise Level", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel (1,1): Stable H
    ax = fig.add_subplot(gs[1,1])
    ax.plot(sigmas, [r["rl_stable_h"] for r in rows], "s-",
            color=CLR["rl"], lw=2, ms=7, label="RL Stable H")
    if has_sl:
        ax.plot(sigmas, [r["sl_stable_h"] for r in rows], "o--",
                color=CLR["sl"], lw=1.5, ms=6, label="SL Stable H", alpha=0.8)
    import math
    ax.axhline(math.log(3), color="#ccc", lw=0.8, ls=":", label="log(3) uniform")
    ax.set_xlabel("Noise σ", fontsize=10); ax.set_ylabel("Stable-phase H", fontsize=9)
    ax.set_title("Fixation Depth\nvs Noise Level", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel (1,2): Trans H
    ax = fig.add_subplot(gs[1,2])
    ax.plot(sigmas, [r["rl_trans_h"] for r in rows], "s-",
            color=CLR["rl"], lw=2, ms=7, label="RL Trans H")
    ax.set_xlabel("Noise σ", fontsize=10); ax.set_ylabel("Transition-phase H", fontsize=9)
    ax.set_title("Transition Entropy\nvs Noise Level", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"RL Sigma Sweep — R_total={cfg.rl_alpha_sync}·R_sync − {cfg.rl_beta_dogma}·P_dogma + {cfg.rl_gamma_nomad}·R_nomad\n"
        f"phi_threshold={cfg.rl_transition_phi_threshold}  entropy_coef={cfg.rl_entropy_coef}  policy_lr={cfg.rl_policy_lr}",
        fontsize=10, y=1.01
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → saved: {save_path}")


def plot_transition_sweep(rows: list, save_path: str):
    steps  = [r["transition_steps"] for r in rows]
    has_sl = not all(np.isnan(r.get("sl_mse_mean", float("nan"))) for r in rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(steps, [r["rl_mse_mean"] for r in rows], "s-",
            color=CLR["rl"], lw=2, ms=7, label="RL")
    if has_sl:
        ax.plot(steps, [r["sl_mse_mean"] for r in rows], "o-",
                color=CLR["sl"], lw=2, ms=7, label="SL")
    ax.set_xlabel("Transition Steps", fontsize=10); ax.set_ylabel("Seq MSE", fontsize=9)
    ax.set_title("MSE vs Transition Speed", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps, [r["rl_impr_pct"] for r in rows], "s-",
            color=CLR["rl"], lw=2, ms=7, label="RL vs StdMoE")
    if has_sl:
        ax.plot(steps, [r["sl_impr_pct"] for r in rows], "o-",
                color=CLR["sl"], lw=2, ms=7, label="SL vs StdMoE")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Transition Steps", fontsize=10); ax.set_ylabel("MSE Improvement (%)", fontsize=9)
    ax.set_title("Improvement over StdMoE", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    rl_dh = [r["rl_dh_mean"] for r in rows]
    ax.plot(steps, rl_dh, "s-", color=CLR["rl"], lw=2, ms=7, label="RL ΔH")
    if has_sl:
        ax.plot(steps, [r["sl_dh_mean"] for r in rows], "o--",
                color=CLR["sl"], lw=1.5, ms=6, label="SL ΔH", alpha=0.8)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Transition Steps", fontsize=10); ax.set_ylabel("ΔH", fontsize=9)
    ax.set_title("Homeomorphic Fixation\nvs Transition Speed", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("RL Transition Sweep (σ=0.9 fixed)\n"
                 "Reference: §4.8 Gradual (steps=24) as structural failure mode",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → saved: {save_path}")


def plot_reward_dynamics(all_histories: dict, sweep_key: str, save_path: str):
    """
    sweep_key별 R_total / policy_entropy / is_transition_ratio 학습 곡선.
    각 key별로 seeds를 평균낸 single 커브.
    """
    keys_list = sorted(all_histories.keys())
    n = len(keys_list)
    if n == 0: return

    cmap   = plt.cm.viridis(np.linspace(0.1, 0.9, n))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics   = ["r_total", "policy_entropy", "is_transition_ratio"]
    titles    = ["R_total (mean across seeds)", "Policy Entropy", "Transition Ratio"]

    for ax, metric, title in zip(axes, metrics, titles):
        for i, key in enumerate(keys_list):
            hists = all_histories[key]  # list of seed dicts
            # seeds 평균
            max_ep = max(len(h[metric]) for h in hists)
            arr    = np.full((len(hists), max_ep), np.nan)
            for j, h in enumerate(hists):
                v = h[metric]
                arr[j, :len(v)] = v
            mean_v = np.nanmean(arr, axis=0)
            std_v  = np.nanstd(arr, axis=0)
            epochs = list(range(1, max_ep+1))
            label  = f"{sweep_key}={key}"
            ax.plot(epochs, mean_v, color=cmap[i], lw=1.8, label=label)
            ax.fill_between(epochs, mean_v-std_v, mean_v+std_v,
                            alpha=0.12, color=cmap[i])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"RL Training Dynamics — {sweep_key} sweep", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → saved: {save_path}")


# ============================================================
# Summary text
# ============================================================

def write_sweep_summary(sigma_rows, trans_rows, cfg, save_path):
    lines = ["="*70, "RL Sweep Summary", "="*70, "",
             f"R_total = {cfg.rl_alpha_sync}·R_sync − {cfg.rl_beta_dogma}·P_dogma + {cfg.rl_gamma_nomad}·R_nomad",
             f"phi_threshold={cfg.rl_transition_phi_threshold}  entropy_coef={cfg.rl_entropy_coef}  policy_lr={cfg.rl_policy_lr}",
             ""]

    if sigma_rows:
        lines += ["Sigma Sweep (transition_steps=8)", "-"*60,
                  f"{'σ':>5}  {'StdMSE':>8}  {'SL_MSE':>8}  {'RL_MSE':>8}  "
                  f"{'SL%':>7}  {'RL%':>7}  {'RL-SL%':>8}  {'SL_ΔH':>7}  {'RL_ΔH':>7}"]
        for r in sigma_rows:
            def f(k): return f"{r.get(k,float('nan')):8.4f}" if not np.isnan(r.get(k,float('nan'))) else "     n/a"
            def g(k): return f"{r.get(k,float('nan')):+7.1f}%" if not np.isnan(r.get(k,float('nan'))) else "    n/a"
            lines.append(f"{r['sigma']:>5.2f}  {f('std_mse_mean')}  {f('sl_mse_mean')}  {f('rl_mse_mean')}  "
                         f"{g('sl_impr_pct')}  {g('rl_impr_pct')}  {g('rl_vs_sl_pct')}  "
                         f"{r.get('sl_dh_mean',float('nan')):7.3f}  {r.get('rl_dh_mean',float('nan')):7.3f}")
        best_dh = max(sigma_rows, key=lambda r: r.get("rl_dh_mean",0))
        lines += ["", f"Peak RL ΔH at σ={best_dh['sigma']:.2f} (ΔH={best_dh['rl_dh_mean']:.3f})", ""]

    if trans_rows:
        lines += ["Transition Sweep (σ=0.9 fixed)", "-"*60,
                  f"{'steps':>6}  {'StdMSE':>8}  {'RL_MSE':>8}  {'RL%':>7}  {'RL_ΔH':>7}"]
        for r in trans_rows:
            def f(k): return f"{r.get(k,float('nan')):8.4f}" if not np.isnan(r.get(k,float('nan'))) else "     n/a"
            lines.append(f"{int(r['transition_steps']):>6}  {f('std_mse_mean')}  {f('rl_mse_mean')}  "
                         f"{r.get('rl_impr_pct',float('nan')):+7.1f}%  {r.get('rl_dh_mean',float('nan')):7.3f}")
        lines += [""]

    lines += ["="*70,
              "해석 가이드:",
              "  RL ΔH > SL ΔH → R_total이 fixation을 더 잘 유도",
              "  RL ΔH < SL ΔH → imitation이 transition 패턴 학습에 유리",
              "  RL_vs_SL > 0   → 해당 σ에서 RL이 SL 대비 MSE 개선",
              "  RL_vs_SL < 0   → 해당 σ에서 SL이 MSE 최적화에 유리",
              "="*70]
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Nomadic RL — Sweep (RL only)")
    parser.add_argument("--sweep",   choices=["sigma","transition","both"], default="sigma")
    parser.add_argument("--quick",   action="store_true",
                        help="Fast check: 1 seed, 80 epoch, 3 values")
    parser.add_argument("--device",  default="cpu", choices=["cpu","cuda"])
    # SL 결과 CSV (없으면 StdMoE 재실행, SL 비교 없음)
    parser.add_argument("--sl_sigma_csv", default="",
                        help="기존 SL sigma_sweep_results.csv 경로 (run_pressure_sweep.py 출력)")
    parser.add_argument("--sl_trans_csv", default="",
                        help="기존 SL transition_sweep_results.csv 경로")
    # RL 하이퍼파라미터
    parser.add_argument("--alpha_sync",       type=float, default=2.0)
    parser.add_argument("--beta_dogma",        type=float, default=0.3)
    parser.add_argument("--gamma_nomad",       type=float, default=0.5)
    parser.add_argument("--policy_lr",         type=float, default=2e-4)
    parser.add_argument("--entropy_coef",      type=float, default=0.15)
    parser.add_argument("--phi_threshold",     type=float, default=0.90)
    parser.add_argument("--epochs",            type=int,   default=150)
    args = parser.parse_args()

    SEEDS = [42] if args.quick else [42, 123, 456]
    os.makedirs("outputs_rl_sweep", exist_ok=True)

    cfg = Config(device=args.device, epochs=args.epochs)
    cfg.rl_alpha_sync            = args.alpha_sync
    cfg.rl_beta_dogma            = args.beta_dogma
    cfg.rl_gamma_nomad           = args.gamma_nomad
    cfg.rl_policy_lr             = args.policy_lr
    cfg.rl_entropy_coef          = args.entropy_coef
    cfg.rl_transition_phi_threshold = args.phi_threshold

    print(f"\n{'='*65}")
    print(f"Nomadic RL Sweep — RL only (SL from CSV if provided)")
    print(f"  mode={args.sweep}  seeds={SEEDS}  device={args.device}  quick={args.quick}")
    print(f"  R_total = {cfg.rl_alpha_sync}·R_sync − {cfg.rl_beta_dogma}·P_dogma + {cfg.rl_gamma_nomad}·R_nomad")
    print(f"  phi_threshold={cfg.rl_transition_phi_threshold}  entropy_coef={cfg.rl_entropy_coef}")
    print(f"{'='*65}\n")

    sigma_rows, trans_rows = [], []
    sigma_hists, trans_hists = {}, {}

    if args.sweep in ("sigma", "both"):
        sl_rows = load_sl_csv(args.sl_sigma_csv)
        print("▶ Sigma Sweep")
        sigma_rows, sigma_hists = run_sigma_sweep(cfg, SEEDS, sl_rows, args.quick)
        save_csv(sigma_rows, "outputs_rl_sweep/rl_sigma_sweep.csv", "sigma")
        plot_sigma_sweep(sigma_rows, "outputs_rl_sweep/fig_rl_sigma_sweep.png", cfg)
        plot_reward_dynamics(sigma_hists, "sigma",
                             "outputs_rl_sweep/fig_rl_sigma_dynamics.png")

    if args.sweep in ("transition", "both"):
        sl_rows = load_sl_csv(args.sl_trans_csv)
        print("▶ Transition Sweep")
        trans_rows, trans_hists = run_transition_sweep(cfg, SEEDS, sl_rows, args.quick)
        save_csv(trans_rows, "outputs_rl_sweep/rl_transition_sweep.csv", "transition_steps")
        plot_transition_sweep(trans_rows, "outputs_rl_sweep/fig_rl_transition_sweep.png")
        plot_reward_dynamics(trans_hists, "steps",
                             "outputs_rl_sweep/fig_rl_trans_dynamics.png")

    write_sweep_summary(sigma_rows, trans_rows, cfg, "outputs_rl_sweep/sweep_summary.txt")

    print("\n✅ Done. outputs_rl_sweep/")
    print("  rl_sigma_sweep.csv            ← 전체 결과 (SL 병합 포함)")
    print("  rl_transition_sweep.csv")
    print("  fig_rl_sigma_sweep.png        ← 6-panel: MSE / improvement / ΔH / entropy")
    print("  fig_rl_sigma_dynamics.png     ← R_total / entropy / trans_ratio 학습곡선")
    print("  fig_rl_transition_sweep.png")
    print("  sweep_summary.txt")


if __name__ == "__main__":
    main()
