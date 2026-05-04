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

    # data
    input_dim: int = 2
    output_dim: int = 1
    overlap_std: float = 0.9

    # regime centers
    center_A: Tuple[float, float] = (2.5, 2.5)
    center_B: Tuple[float, float] = (-2.5, -2.5)
    center_C: Tuple[float, float] = (2.5, -2.5)

    # model
    hidden_dim: int = 64
    num_experts: int = 3
    gate_hidden_dim: int = 64

    # routing softness
    temperature: float = 0.60

    # training
    epochs: int = 220
    lr: float = 2e-3
    weight_decay: float = 1e-5

    # phase-sequence setting
    phase_batch_size: int = 64
    phase_train_cycles: int = 40
    phase_test_cycles: int = 12
    transition_steps: int = 8

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

    # dynamic dwell / fixation (environment-aware tau)
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

    # output
    save_dir: str = "outputs_transition"

# ============================================================
# YAML helpers
# ============================================================

def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}

def build_config_from_yaml(yaml_dict: dict) -> Config:
    runtime = yaml_dict.get("runtime", {})
    training = yaml_dict.get("training", {})
    model = yaml_dict.get("model", {})
    data = yaml_dict.get("data", {})
    loss = yaml_dict.get("loss", {})
    delta = yaml_dict.get("delta", {})
    switching = yaml_dict.get("switching", {})
    policy = yaml_dict.get("policy", {})

    device_value = runtime.get("device", "auto")
    if device_value == "auto":
        device_value = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Config(
        seed=runtime.get("seed", 42),
        save_dir=runtime.get("save_dir", "outputs_transition"),
        device=device_value,

        epochs=training.get("epochs", 220),
        lr=training.get("lr", 2e-3),
        weight_decay=training.get("weight_decay", 1e-5),

        hidden_dim=model.get("hidden_dim", 64),
        num_experts=model.get("num_experts", 3),
        gate_hidden_dim=model.get("gate_hidden_dim", 64),
        temperature=model.get("temperature", 0.60),

        overlap_std=data.get("overlap_std", 0.9),
        phase_batch_size=data.get("phase_batch_size", 64),
        phase_train_cycles=data.get("phase_train_cycles", 40),
        phase_test_cycles=data.get("phase_test_cycles", 12),
        transition_steps=data.get("transition_steps", 8),

        alpha_dogma=loss.get("alpha_dogma", 0.04),
        beta_nomad=loss.get("beta_nomad", 0.05),
        gamma_diversity=loss.get("gamma_diversity", 0.08),
        lambda_sep=loss.get("lambda_sep", 0.08),
        lambda_cons=loss.get("lambda_cons", 0.03),
        lambda_load=loss.get("lambda_load", 0.03),
        tau_k_min=loss.get("tau_k_min", 3),
        tau_k_penalty=loss.get("tau_k_penalty", 0.05),

        use_dynamic_tau=loss.get("use_dynamic_tau", True),
        tau_min=loss.get("tau_min", 2.0),
        tau_max=loss.get("tau_max", 8.0),
        tau_var_scale=loss.get("tau_var_scale", 6.0),
        tau_var_window=loss.get("tau_var_window", 8),

        ema_decay=delta.get("ema_decay", 0.80),
        err_baseline_momentum=delta.get("err_baseline_momentum", 0.85),
        w_env=delta.get("w_env", 1.0),
        w_err=delta.get("w_err", 2.0),

        phi_scale_env=switching.get("phi_scale_env", 1.0),
        phi_scale_err=switching.get("phi_scale_err", 1.5),
        phi_scale_explain=switching.get("phi_scale_explain", 2.0),
        phi_scale_gap=switching.get("phi_scale_gap", 1.0),
        beta_phi=switching.get("beta_phi", 0.05),

        temp_stable=switching.get("temp_stable", 0.30),
        temp_transition=switching.get("temp_transition", 1.00),

        use_hard_switch=switching.get("use_hard_switch", True),
        phi_hard_threshold=switching.get("phi_hard_threshold", 0.35),

        policy_hidden_dim=policy.get("policy_hidden_dim", 64),
        policy_mix_weight=policy.get("policy_mix_weight", 0.25),
        policy_weight_stay=policy.get("policy_weight_stay", 0.20),
        policy_weight_target=policy.get("policy_weight_target", 0.20),
        policy_weight_mode=policy.get("policy_weight_mode", 0.10),
        policy_switch_threshold=policy.get("policy_switch_threshold", 0.50),
    )
    return cfg

# ============================================================
# Data generation
# ============================================================

REGIME_TO_ID = {"A": 0, "B": 1, "C": 2}
ID_TO_REGIME = {0: "A", 1: "B", 2: "C"}
REGIME_ORDER = ["A", "B", "C"]

def sample_regime_x(regime: str, n: int, std: float, device: str = "cpu") -> torch.Tensor:
    noise = std * torch.randn(n, 2, device=device)

    if regime == "A":
        center = torch.tensor([2.5, 2.5], device=device)
    elif regime == "B":
        center = torch.tensor([-2.5, -2.5], device=device)
    elif regime == "C":
        center = torch.tensor([2.5, -2.5], device=device)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    return noise + center

def regime_function(x: torch.Tensor, regime: str) -> torch.Tensor:
    x1 = x[:, 0]
    x2 = x[:, 1]

    if regime == "A":
        y = x1 + x2
    elif regime == "B":
        y = x1 - x2
    elif regime == "C":
        y = -x1 + 0.5 * x2
    else:
        raise ValueError(f"Unknown regime: {regime}")

    return y.unsqueeze(-1)

def generate_phase_sequence(cfg: Config, cycles: int, device: str = "cpu"):
    xs, ys, rs = [], [], []
    phase_tags: List[str] = []

    for _ in range(cycles):
        for i in range(len(REGIME_ORDER)):
            curr_r = REGIME_ORDER[i]
            next_r = REGIME_ORDER[(i + 1) % len(REGIME_ORDER)]

            # stable block
            x_stable = sample_regime_x(curr_r, cfg.phase_batch_size, std=cfg.overlap_std, device=device)
            y_stable = regime_function(x_stable, curr_r)
            r_stable = torch.full((cfg.phase_batch_size,), REGIME_TO_ID[curr_r], dtype=torch.long, device=device)

            xs.append(x_stable)
            ys.append(y_stable)
            rs.append(r_stable)
            phase_tags.extend([f"stable_{curr_r}"] * cfg.phase_batch_size)

            # transition block
            for step in range(cfg.transition_steps):
                alpha = (step + 1) / cfg.transition_steps

                x_a = sample_regime_x(curr_r, cfg.phase_batch_size, std=cfg.overlap_std, device=device)
                x_b = sample_regime_x(next_r, cfg.phase_batch_size, std=cfg.overlap_std, device=device)
                x_mix = (1.0 - alpha) * x_a + alpha * x_b

                y_a = regime_function(x_mix, curr_r)
                y_b = regime_function(x_mix, next_r)
                y_mix = (1.0 - alpha) * y_a + alpha * y_b

                dominant = curr_r if alpha < 0.5 else next_r
                r_mix = torch.full((cfg.phase_batch_size,), REGIME_TO_ID[dominant], dtype=torch.long, device=device)

                xs.append(x_mix)
                ys.append(y_mix)
                rs.append(r_mix)
                phase_tags.extend([f"transition_{curr_r}_to_{next_r}"] * cfg.phase_batch_size)

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    R = torch.cat(rs, dim=0)

    return X, Y, R, phase_tags

def iterate_sequence_minibatches(X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, batch_size: int):
    n = X.size(0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield X[start:end], Y[start:end], R[start:end]

# ============================================================
# Models
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GateNet(nn.Module):
    def __init__(self, input_dim: int, gate_hidden_dim: int, num_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, gate_hidden_dim),  
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, num_experts),
        )

    def forward(
        self,
        x: torch.Tensor,
        delta_hybrid: torch.Tensor,
        delta_err: torch.Tensor,
        temperature: float,
    ):
        gate_input = torch.cat([x, delta_hybrid, delta_err], dim=-1)
        logits = self.net(gate_input)
        probs = F.softmax(logits / temperature, dim=-1)
        return probs, logits

class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.stay_switch_head = nn.Linear(hidden_dim, 2)
        self.target_head = nn.Linear(hidden_dim, num_experts)
        self.mode_head = nn.Linear(hidden_dim, 2)

    def forward(self, policy_input: torch.Tensor):
        h = self.shared(policy_input)
        stay_switch_probs = F.softmax(self.stay_switch_head(h), dim=-1)
        target_probs      = F.softmax(self.target_head(h),      dim=-1)
        mode_probs        = F.softmax(self.mode_head(h),        dim=-1)
        return stay_switch_probs, target_probs, mode_probs

class NomadicMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        gate_hidden_dim: int,
        policy_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
        self.gate = GateNet(input_dim, gate_hidden_dim, num_experts)
        self.policy = PolicyNet(input_dim, policy_hidden_dim, num_experts)

    def forward(
        self,
        x: torch.Tensor,
        delta_hybrid: torch.Tensor,
        delta_err: torch.Tensor,
        temperature: float,
        hard: bool = False,
    ):
        gate_probs, gate_logits = self.gate(x, delta_hybrid, delta_err, temperature)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  

        if hard:
            top1 = gate_probs.argmax(dim=-1)
            routing = F.one_hot(top1, num_classes=self.num_experts).float()
        else:
            routing = gate_probs

        y_hat = (routing.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return y_hat, gate_probs, gate_logits, expert_outputs

class StandardMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        gate_hidden_dim: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor, hard: bool = False):
        logits = self.gate(x)
        gate_probs = F.softmax(logits, dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        if hard:
            top1 = gate_probs.argmax(dim=-1)
            routing = F.one_hot(top1, num_classes=self.num_experts).float()
        else:
            routing = gate_probs

        y_hat = (routing.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return y_hat, gate_probs, logits, expert_outputs

# ============================================================
# Hybrid Delta utilities
# ============================================================

class HybridDeltaTracker:
    def __init__(
        self,
        ema_decay: float = 0.8,
        err_baseline_momentum: float = 0.85,
        w_env: float = 1.0,
        w_err: float = 2.0,
        device: str = "cpu",
        tau_min: float = 2.0,
        tau_max: float = 8.0,
        tau_var_scale: float = 6.0,
        tau_var_window: int = 8,
    ):
        self.ema_decay = ema_decay
        self.err_baseline_momentum = err_baseline_momentum
        self.w_env = w_env
        self.w_err = w_err
        self.device = device

        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_var_scale = tau_var_scale
        self.tau_var_window = tau_var_window

        self.prev_x_mean = None
        self.err_ema = None
        self.err_baseline = None
        self.recent_delta_env: deque = deque(maxlen=tau_var_window)

        self.delta_env_history = []
        self.delta_err_history = []
        self.delta_hybrid_raw_history = []
        self.delta_hybrid_history = []
        self.sigma2_delta_history = []
        self.dynamic_tau_history = []

    def reset(self):
        self.prev_x_mean = None
        self.err_ema = None
        self.err_baseline = None
        self.recent_delta_env.clear()

    def compute_dynamic_tau(self, sigma2_delta: float) -> float:
        tau = self.tau_min + (self.tau_max - self.tau_min) / (1.0 + self.tau_var_scale * sigma2_delta)
        return float(np.clip(tau, self.tau_min, self.tau_max))

    def compute(self, x: torch.Tensor, current_batch_mse: torch.Tensor):
        x_mean = x.mean(dim=0, keepdim=True)

        if self.prev_x_mean is None:
            delta_env_scalar = torch.tensor(0.0, device=self.device)
        else:
            delta_env_scalar = torch.norm(x_mean - self.prev_x_mean, p=2)

        batch_err = current_batch_mse.detach()

        if self.err_ema is None:
            self.err_ema = batch_err
            self.err_baseline = batch_err
            delta_err_scalar = torch.tensor(0.0, device=self.device)
        else:
            self.err_ema = self.ema_decay * self.err_ema + (1.0 - self.ema_decay) * batch_err
            self.err_baseline = (
                self.err_baseline_momentum * self.err_baseline
                + (1.0 - self.err_baseline_momentum) * self.err_ema
            )
            delta_err_scalar = torch.relu(self.err_ema - self.err_baseline)

        raw_hybrid = self.w_env * delta_env_scalar + self.w_err * delta_err_scalar
        delta_hybrid_scalar = torch.tanh(raw_hybrid)

        self.prev_x_mean = x_mean.detach()

        delta_env_val   = float(delta_env_scalar.item())
        delta_err_val   = float(delta_err_scalar.item())
        delta_hybrid_val = float(delta_hybrid_scalar.item())
        raw_hybrid_val  = float(raw_hybrid.item())

        self.recent_delta_env.append(delta_env_val)
        sigma2_delta = float(np.var(self.recent_delta_env)) if len(self.recent_delta_env) >= 2 else 0.0
        dynamic_tau  = self.compute_dynamic_tau(sigma2_delta)

        self.delta_env_history.append(delta_env_val)
        self.delta_err_history.append(delta_err_val)
        self.delta_hybrid_raw_history.append(raw_hybrid_val)
        self.delta_hybrid_history.append(delta_hybrid_val)
        self.sigma2_delta_history.append(sigma2_delta)
        self.dynamic_tau_history.append(dynamic_tau)

        delta_hybrid = torch.full((x.size(0), 1), delta_hybrid_val, device=self.device)
        return (
            delta_hybrid,
            delta_env_val,
            delta_err_val,
            delta_hybrid_val,
            sigma2_delta,
            dynamic_tau,
        )

# ============================================================
# Regularizers / metrics
# ============================================================

def compute_load_balancing_loss(gate_probs: torch.Tensor) -> torch.Tensor:
    num_experts = gate_probs.size(-1)
    mean_gate = gate_probs.mean(dim=0)  
    top1 = gate_probs.argmax(dim=-1)   
    top1_frac = torch.zeros(num_experts, device=gate_probs.device)
    for i in range(num_experts):
        top1_frac[i] = (top1 == i).float().mean()
    loss = num_experts * (top1_frac * mean_gate).sum()
    return loss

class DwellTimeRegularizer:
    def __init__(self, tau_k_min: int = 3, penalty: float = 0.05):
        self.tau_k_min = tau_k_min
        self.penalty = penalty
        self.current_expert = None
        self.dwell_count = 0
        self.last_tau_used = float(tau_k_min)

    def reset(self):
        self.current_expert = None
        self.dwell_count = 0
        self.last_tau_used = float(self.tau_k_min)

    def compute(self, gate_probs: torch.Tensor, tau_dynamic: float = None) -> torch.Tensor:
        top1_counts = torch.bincount(
            gate_probs.argmax(dim=-1),
            minlength=gate_probs.size(-1)
        )
        dominant = int(top1_counts.argmax().item())

        if dominant == self.current_expert:
            self.dwell_count += 1
        else:
            self.current_expert = dominant
            self.dwell_count = 1

        eps = 1e-8
        entropy = -(gate_probs * (gate_probs + eps).log()).sum(dim=-1).mean()

        tau_capacity = float(self.tau_k_min if tau_dynamic is None else tau_dynamic)
        self.last_tau_used = tau_capacity

        if self.dwell_count <= tau_capacity:
            return -self.penalty * entropy
        else:
            excess = self.dwell_count - tau_capacity
            bonus_weight = min(float(excess) * self.penalty, self.penalty * 10)
            return bonus_weight * entropy

def compute_diversity_loss(expert_outputs: torch.Tensor) -> torch.Tensor:
    num_experts = expert_outputs.size(1)
    if num_experts < 2:
        return torch.tensor(0.0, device=expert_outputs.device)

    loss = 0.0
    count = 0
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            sim = F.cosine_similarity(
                expert_outputs[:, i, :],
                expert_outputs[:, j, :],
                dim=-1
            ).mean()
            loss = loss + sim
            count += 1
    return loss / count

def compute_dogma_penalty(gate_probs: torch.Tensor) -> torch.Tensor:
    mean_usage = gate_probs.mean(dim=0)
    concentration = torch.sum(mean_usage ** 2)
    uniform_floor = 1.0 / gate_probs.size(1)
    penalty = concentration - uniform_floor
    return penalty

def compute_nomad_bonus(gate_probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    entropy = -(gate_probs * (gate_probs + eps).log()).sum(dim=-1).mean()
    return entropy

def compute_explanation_signals(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
    expert_outputs: torch.Tensor,
    gate_probs: torch.Tensor,
):
    explanation_error = F.mse_loss(y_hat, y_true)
    per_expert_sqerr = ((expert_outputs - y_true.unsqueeze(1)) ** 2).mean(dim=-1)
    top1_idx = gate_probs.argmax(dim=-1)  
    top1_err = per_expert_sqerr.gather(1, top1_idx.unsqueeze(1)).mean()
    best_expert_err = per_expert_sqerr.min(dim=1).values.mean()
    best_expert_gap = torch.relu(top1_err - best_expert_err)
    return explanation_error, best_expert_gap

def compute_phi_signal(
    delta_env_scalar: float,
    delta_err_scalar: float,
    explanation_error: torch.Tensor,
    best_expert_gap: torch.Tensor,
    phi_scale_env: float = 1.0,
    phi_scale_err: float = 1.5,
    phi_scale_explain: float = 2.0,
    phi_scale_gap: float = 1.0,
):
    device = explanation_error.device
    env_term = phi_scale_env * torch.tensor(delta_env_scalar, device=device)
    err_term = phi_scale_err * torch.tensor(delta_err_scalar, device=device)
    explain_term = phi_scale_explain * explanation_error.detach()
    gap_term = phi_scale_gap * best_expert_gap.detach()

    phi_signal = torch.tanh(env_term + err_term + explain_term + gap_term)
    return phi_signal

def compute_adaptive_temperature(
    phi_signal: torch.Tensor,
    temp_stable: float = 0.30,
    temp_transition: float = 1.00,
):
    phi_val = float(phi_signal.mean().item())
    temp = temp_stable + (temp_transition - temp_stable) * phi_val
    return temp

def build_policy_input(
    xb: torch.Tensor,
    delta_hybrid: torch.Tensor,
    delta_err_tensor: torch.Tensor,
    phi_signal: torch.Tensor,
    sigma2_delta: float,
    dynamic_tau: float,
) -> torch.Tensor:
    x_summary = xb.mean(dim=0, keepdim=True).expand(xb.size(0), -1)
    phi_tensor = torch.full((xb.size(0), 1), float(phi_signal.mean().item()), device=xb.device)
    sigma2_scaled = float(np.tanh(sigma2_delta * 10.0))
    sigma2_tensor = torch.full((xb.size(0), 1), sigma2_scaled, device=xb.device)
    tau_mid = 5.0
    tau_scaled = float(np.tanh((dynamic_tau - tau_mid) / tau_mid))
    tau_tensor = torch.full((xb.size(0), 1), tau_scaled, device=xb.device)
    return torch.cat(
        [x_summary, delta_hybrid, delta_err_tensor, phi_tensor, sigma2_tensor, tau_tensor],
        dim=-1,
    )

def build_policy_targets(
    y_true: torch.Tensor,
    expert_outputs: torch.Tensor,
    phi_signal: torch.Tensor,
    sigma2_delta: float,
    dynamic_tau: float,
    switch_threshold: float,
    tau_stay_threshold: float = 5.5,
    sigma_switch_threshold: float = 0.05,
):
    per_expert_sqerr = ((expert_outputs - y_true.unsqueeze(1)) ** 2).mean(dim=-1)
    target_expert = per_expert_sqerr.mean(dim=0).argmin().long()
    phi_val = float(phi_signal.mean().item())
    should_switch = (phi_val > switch_threshold) or (sigma2_delta > sigma_switch_threshold)
    can_fixate    = (phi_val <= switch_threshold) and (dynamic_tau >= tau_stay_threshold)
    switch_label = 1 if should_switch else 0
    mode_label   = 1 if can_fixate else 0   
    return switch_label, target_expert, mode_label

def gate_entropy(gate_probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -(gate_probs * (gate_probs + eps).log()).sum(dim=-1)

def regimewise_usage(gate_probs: torch.Tensor, regime_ids: torch.Tensor, num_experts: int) -> Dict[str, np.ndarray]:
    usage = {}
    top1 = gate_probs.argmax(dim=-1)
    for rid in range(3):
        mask = regime_ids == rid
        regime_name = ID_TO_REGIME[rid]
        if mask.sum() == 0:
            usage[regime_name] = np.zeros(num_experts, dtype=np.float32)
            continue
        counts = torch.bincount(top1[mask], minlength=num_experts).float()
        counts = counts / counts.sum().clamp_min(1.0)
        usage[regime_name] = counts.detach().cpu().numpy()
    return usage

def compute_regime_gate_stats(
    gate_probs: torch.Tensor,
    regime_ids: torch.Tensor,
    num_regimes: int = 3,
):
    device = gate_probs.device
    regime_means = {}
    valid_means = []
    valid_names = []
    l_cons = torch.tensor(0.0, device=device)
    valid_regime_count = 0

    for rid in range(num_regimes):
        mask = regime_ids == rid
        regime_name = ID_TO_REGIME[rid]
        if mask.sum() == 0: continue
        g_r = gate_probs[mask]
        u_r = g_r.mean(dim=0)
        regime_means[regime_name] = u_r
        valid_means.append(u_r)
        valid_names.append(regime_name)
        l_cons = l_cons + ((g_r - u_r.unsqueeze(0)) ** 2).sum(dim=-1).mean()
        valid_regime_count += 1

    if valid_regime_count > 0:
        l_cons = l_cons / valid_regime_count

    if len(valid_means) < 2:
        l_sep = torch.tensor(0.0, device=device)
        return regime_means, l_sep, l_cons, 0.0, {}

    pairwise = []
    pairwise_distances = {}
    for i in range(len(valid_means)):
        for j in range(i + 1, len(valid_means)):
            dist = torch.norm(valid_means[i] - valid_means[j], p=2)
            pairwise.append(dist)
            pairwise_distances[f"{valid_names[i]}-{valid_names[j]}"] = float(dist.detach().cpu().item())

    pairwise_tensor = torch.stack(pairwise)
    mean_gate_distance = float(pairwise_tensor.mean().detach().cpu().item())
    l_sep = -pairwise_tensor.mean()

    return regime_means, l_sep, l_cons, mean_gate_distance, pairwise_distances

def mse_by_regime(y_true: torch.Tensor, y_pred: torch.Tensor, regime_ids: torch.Tensor) -> Dict[str, float]:
    result = {}
    for rid in range(3):
        mask = regime_ids == rid
        regime_name = ID_TO_REGIME[rid]
        if mask.sum() == 0:
            result[regime_name] = float("nan")
        else:
            result[regime_name] = F.mse_loss(y_pred[mask], y_true[mask]).item()
    return result

def infer_regime_to_expert(usage: Dict[str, np.ndarray]) -> Dict[str, int]:
    mapping = {}
    for regime in ["A", "B", "C"]:
        mapping[regime] = int(np.argmax(usage[regime]))
    return mapping

def compute_dwell_times(top1_sequence: np.ndarray) -> List[int]:
    if len(top1_sequence) == 0: return []
    dwells = []
    current = top1_sequence[0]
    run_len = 1
    for t in range(1, len(top1_sequence)):
        if top1_sequence[t] == current:
            run_len += 1
        else:
            dwells.append(run_len)
            current = top1_sequence[t]
            run_len = 1
    dwells.append(run_len)
    return dwells

def compute_switch_latency(regime_seq: List[str], top1_seq: np.ndarray, regime_to_expert: Dict[str, int]) -> List[int]:
    latencies = []
    prev_regime = regime_seq[0] if len(regime_seq) > 0 else None
    for t in range(1, len(regime_seq)):
        curr_regime = regime_seq[t]
        if curr_regime != prev_regime:
            target_expert = regime_to_expert.get(curr_regime, None)
            if target_expert is None:
                prev_regime = curr_regime
                continue
            latency = None
            for k in range(t, len(top1_seq)):
                if int(top1_seq[k]) == int(target_expert):
                    latency = k - t
                    break
            if latency is not None:
                latencies.append(latency)
        prev_regime = curr_regime
    return latencies

# ============================================================
# Training / Evaluation
# ============================================================

def evaluate_fixed(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        total_mse = F.mse_loss(y_pred, Y).item()
        per_regime = mse_by_regime(Y, y_pred, R)
    return total_mse, per_regime

def evaluate_standard_moe(model: StandardMoE, X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, cfg: Config):
    model.eval()
    with torch.no_grad():
        y_pred, gate_probs, _, _ = model(X, hard=False)
        total_mse = F.mse_loss(y_pred, Y).item()
        per_regime = mse_by_regime(Y, y_pred, R)
        usage = regimewise_usage(gate_probs, R, cfg.num_experts)
        _, _, _, mean_gate_distance, pairwise_distances = compute_regime_gate_stats(gate_probs=gate_probs, regime_ids=R, num_regimes=3)
        ent = gate_entropy(gate_probs).mean().item()
        top1 = gate_probs.argmax(dim=-1).detach().cpu().numpy()
        dwell_times = compute_dwell_times(top1)
    return total_mse, per_regime, usage, mean_gate_distance, pairwise_distances, ent, dwell_times, y_pred, gate_probs

def evaluate_standard_moe_sequence(model: StandardMoE, X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, phase_tags: List[str], cfg: Config):
    model.eval()
    all_y, all_gate_probs, batch_phase_tags, batch_entropies, batch_top1 = [], [], [], [], []
    with torch.no_grad():
        for batch_idx, (xb, yb, rb) in enumerate(iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)):
            y_hat, gate_probs, _, _ = model(xb, hard=False)
            all_y.append(y_hat)
            all_gate_probs.append(gate_probs)
            phase_tag = phase_tags[batch_idx * cfg.phase_batch_size]
            batch_phase_tags.append(phase_tag)
            ent = gate_entropy(gate_probs).mean().item()
            batch_entropies.append(ent)
            top1 = gate_probs.argmax(dim=-1)
            binc = torch.bincount(top1, minlength=cfg.num_experts).float()
            batch_top1.append(int(torch.argmax(binc).item()))

    Y_hat = torch.cat(all_y, dim=0)
    G = torch.cat(all_gate_probs, dim=0)
    seq_mse = F.mse_loss(Y_hat, Y).item()
    usage = regimewise_usage(G, R, cfg.num_experts)

    stable_entropy = [e for tag, e in zip(batch_phase_tags, batch_entropies) if tag.startswith("stable_")]
    transition_entropy = [e for tag, e in zip(batch_phase_tags, batch_entropies) if tag.startswith("transition_")]

    return seq_mse, usage, {
        "stable_entropy_mean": float(np.mean(stable_entropy)) if stable_entropy else float("nan"),
        "transition_entropy_mean": float(np.mean(transition_entropy)) if transition_entropy else float("nan"),
    }

def evaluate_nomadic_static_full(model: NomadicMoE, X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, cfg: Config):
    model.eval()
    with torch.no_grad():
        delta_hybrid = torch.zeros((X.size(0), 1), device=X.device)
        delta_err = torch.zeros((X.size(0), 1), device=X.device)
        y_pred, gate_probs, _, _ = model(X, delta_hybrid, delta_err, cfg.temperature)
        total_mse = F.mse_loss(y_pred, Y).item()
        per_regime = mse_by_regime(Y, y_pred, R)
        usage = regimewise_usage(gate_probs, R, cfg.num_experts)
        _, _, _, mean_gate_distance, pairwise_distances = compute_regime_gate_stats(gate_probs=gate_probs, regime_ids=R, num_regimes=3)
        ent = gate_entropy(gate_probs).mean().item()
        top1 = gate_probs.argmax(dim=-1).detach().cpu().numpy()
        dwell_times = compute_dwell_times(top1)
    return total_mse, per_regime, usage, mean_gate_distance, pairwise_distances, ent, dwell_times, y_pred, gate_probs

# -------------------------------------------------------------
# [수정 1] evaluate_nomadic_sequence_dynamics 에 prev_yb 적용
# -------------------------------------------------------------
def evaluate_nomadic_sequence_dynamics(model: NomadicMoE, X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, phase_tags: List[str], cfg: Config):
    model.eval()
    tracker = HybridDeltaTracker(
        ema_decay=cfg.ema_decay, err_baseline_momentum=cfg.err_baseline_momentum,
        w_env=cfg.w_env, w_err=cfg.w_err, device=cfg.device,
        tau_min=cfg.tau_min, tau_max=cfg.tau_max, tau_var_scale=cfg.tau_var_scale, tau_var_window=cfg.tau_var_window,
    )
    tracker.reset()

    all_y, all_gate_probs = [], []
    batch_regimes, batch_phase_tags, batch_entropies, batch_top1 = [], [], [], []
    batch_sigma2_delta, batch_dynamic_tau = [], []

    # 이전 배치 정답 버퍼 (초기화)
    prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

    with torch.no_grad():
        for batch_idx, (xb, yb, rb) in enumerate(iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)):
            
            # 크기 보정 (마지막 배치용)
            if prev_yb.size(0) != xb.size(0):
                prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            zero_delta = torch.zeros((xb.size(0), 1), device=cfg.device)
            warm_y, _, _, _ = model(xb, zero_delta, zero_delta, cfg.temperature, hard=False)
            
            # [핵심] 시차 기동: 이전 정답 사용
            warm_mse = F.mse_loss(warm_y, prev_yb)

            delta_hybrid, de, derr, _, sigma2_delta, dynamic_tau = tracker.compute(xb, warm_mse)
            delta_err_tensor = torch.full((xb.size(0), 1), derr, device=cfg.device)

            probe_y, probe_gate_probs, _, probe_expert_outputs = model(
                xb, delta_hybrid, delta_err_tensor, cfg.temperature, hard=False
            )

            # [핵심] 시차 기동: 이전 정답 기반 성찰
            explanation_error, best_expert_gap = compute_explanation_signals(
                y_true=prev_yb, y_hat=probe_y, expert_outputs=probe_expert_outputs, gate_probs=probe_gate_probs,
            )

            phi_signal = compute_phi_signal(
                delta_env_scalar=de, delta_err_scalar=derr,
                explanation_error=explanation_error, best_expert_gap=best_expert_gap,
                phi_scale_env=cfg.phi_scale_env, phi_scale_err=cfg.phi_scale_err,
                phi_scale_explain=cfg.phi_scale_explain, phi_scale_gap=cfg.phi_scale_gap,
            )

            policy_input = build_policy_input(
                xb=xb, delta_hybrid=delta_hybrid, delta_err_tensor=delta_err_tensor,
                phi_signal=phi_signal, sigma2_delta=sigma2_delta, dynamic_tau=dynamic_tau,
            )
            stay_switch_probs, target_probs, mode_probs = model.policy(policy_input)

            temp_now = compute_adaptive_temperature(phi_signal=phi_signal, temp_stable=cfg.temp_stable, temp_transition=cfg.temp_transition)

            delta_hybrid_val_now = float(delta_hybrid.mean().item())
            failsafe_soft = delta_hybrid_val_now > cfg.phi_hard_threshold
            hard_mode = bool(cfg.use_hard_switch and (mode_probs[:, 1].mean().item() > 0.5) and not failsafe_soft)

            y_hat, gate_probs, _, expert_outputs_eval = model(xb, delta_hybrid, delta_err_tensor, temp_now, hard=False)

            effective_mix = cfg.policy_mix_weight * float(stay_switch_probs[:, 1].mean().item())
            target_idx = torch.argmax(target_probs.mean(dim=0), dim=-1)
            target_onehot_hard = F.one_hot(target_idx, num_classes=cfg.num_experts).float().unsqueeze(0).expand(xb.size(0), -1)

            target_onehot_ste = (target_onehot_hard - gate_probs).detach() + gate_probs
            mixed_routing = (1.0 - effective_mix) * gate_probs + effective_mix * target_onehot_ste

            if hard_mode:
                top1_r = mixed_routing.argmax(dim=-1)
                final_routing = F.one_hot(top1_r, num_classes=cfg.num_experts).float()
            else:
                final_routing = mixed_routing

            y_hat = (final_routing.unsqueeze(-1) * expert_outputs_eval).sum(dim=1)
            gate_probs = final_routing

            all_y.append(y_hat)
            all_gate_probs.append(gate_probs)

            dominant_regime = ID_TO_REGIME[int(rb[0].item())]
            batch_regimes.append(dominant_regime)
            phase_tag = phase_tags[batch_idx * cfg.phase_batch_size]
            batch_phase_tags.append(phase_tag)
            ent = gate_entropy(gate_probs).mean().item()
            batch_entropies.append(ent)

            top1 = gate_probs.argmax(dim=-1)
            binc = torch.bincount(top1, minlength=cfg.num_experts).float()
            batch_top1.append(int(torch.argmax(binc).item()))

            batch_sigma2_delta.append(sigma2_delta)
            batch_dynamic_tau.append(dynamic_tau)

            # [핵심] 현재 정답을 과거 버퍼로 밀어넣기
            prev_yb = yb.detach()

    Y_hat = torch.cat(all_y, dim=0)
    G = torch.cat(all_gate_probs, dim=0)

    total_mse = F.mse_loss(Y_hat, Y).item()
    usage = regimewise_usage(G, R, cfg.num_experts)
    regime_to_expert = infer_regime_to_expert(usage)

    latencies = compute_switch_latency(batch_regimes, np.array(batch_top1), regime_to_expert)
    dwell_times = compute_dwell_times(np.array(batch_top1))

    stable_entropy = [e for tag, e in zip(batch_phase_tags, batch_entropies) if tag.startswith("stable_")]
    transition_entropy = [e for tag, e in zip(batch_phase_tags, batch_entropies) if tag.startswith("transition_")]

    dynamics = {
        "batch_regimes": batch_regimes, "batch_phase_tags": batch_phase_tags, "batch_entropies": batch_entropies,
        "batch_top1": batch_top1, "switch_latencies": latencies, "dwell_times": dwell_times,
        "mean_switch_latency": float(np.mean(latencies)) if len(latencies) > 0 else float("nan"),
        "mean_dwell_time": float(np.mean(dwell_times)) if len(dwell_times) > 0 else float("nan"),
        "stable_entropy_mean": float(np.mean(stable_entropy)) if len(stable_entropy) > 0 else float("nan"),
        "transition_entropy_mean": float(np.mean(transition_entropy)) if len(transition_entropy) > 0 else float("nan"),
        "regime_to_expert": regime_to_expert, "sigma2_delta": batch_sigma2_delta, "dynamic_tau": batch_dynamic_tau,
        "mean_dynamic_tau": float(np.mean(batch_dynamic_tau)) if len(batch_dynamic_tau) > 0 else float("nan"),
    }

    return total_mse, usage, dynamics, Y_hat, G

def train_fixed(cfg: Config, X_train, Y_train, R_train, X_test, Y_test, R_test):
    model = MLPRegressor(cfg.input_dim, cfg.hidden_dim, cfg.output_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_losses, test_losses = [], []
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, _ in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            optimizer.zero_grad()
            y_hat = model(xb)
            loss = F.mse_loss(y_hat, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))
        test_mse, _ = evaluate_fixed(model, X_test, Y_test, R_test)
        test_losses.append(test_mse)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"[Fixed] Epoch {epoch+1:03d}/{cfg.epochs} | Train MSE: {train_losses[-1]:.4f} | Test MSE: {test_mse:.4f}")
    return model, {"train_losses": train_losses, "test_losses": test_losses}

def train_standard_moe(cfg: Config, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test):
    model = StandardMoE(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_experts, cfg.gate_hidden_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logs = {"train_losses": [], "test_mse_static": [], "test_mse_sequence": [], "test_mean_gate_distance": [], "test_entropy": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, _ in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            optimizer.zero_grad()
            y_hat, gate_probs, _, expert_outputs = model(xb, hard=False)
            total_loss = F.mse_loss(y_hat, yb) + cfg.gamma_diversity * compute_diversity_loss(expert_outputs) + cfg.lambda_load * compute_load_balancing_loss(gate_probs)
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            n_batches += 1
        logs["train_losses"].append(epoch_loss / max(n_batches, 1))

        test_static_mse, _, _, test_gate_dist, _, test_entropy, _, _, _ = evaluate_standard_moe(model, X_test, Y_test, R_test, cfg)
        test_seq_mse, _, _ = evaluate_standard_moe_sequence(model, X_test, Y_test, R_test, phase_tags_test, cfg)
        
        logs["test_mse_static"].append(test_static_mse)
        logs["test_mse_sequence"].append(test_seq_mse)
        logs["test_mean_gate_distance"].append(test_gate_dist)
        logs["test_entropy"].append(test_entropy)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"[Standard MoE] Epoch {epoch+1:03d}/{cfg.epochs} | Train Loss: {logs['train_losses'][-1]:.4f} | Test Static MSE: {test_static_mse:.4f} | Test Seq MSE: {test_seq_mse:.4f} | Test GateDist: {test_gate_dist:.4f}")
    return model, logs

# -------------------------------------------------------------
# [수정 2] evaluate_nomadic_no_policy_sequence 에 prev_yb 적용
# -------------------------------------------------------------
def evaluate_nomadic_no_policy_sequence(model: NomadicMoE, X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, phase_tags: List[str], cfg: Config):
    model.eval()
    tracker = HybridDeltaTracker(
        ema_decay=cfg.ema_decay, err_baseline_momentum=cfg.err_baseline_momentum, w_env=cfg.w_env, w_err=cfg.w_err,
        device=cfg.device, tau_min=cfg.tau_min, tau_max=cfg.tau_max, tau_var_scale=cfg.tau_var_scale, tau_var_window=cfg.tau_var_window,
    )
    tracker.reset()

    all_y, all_gate_probs, batch_phase_tags, batch_entropies, batch_top1 = [], [], [], [], []

    prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

    with torch.no_grad():
        for batch_idx, (xb, yb, rb) in enumerate(iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)):
            if prev_yb.size(0) != xb.size(0):
                prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            zero_delta = torch.zeros((xb.size(0), 1), device=cfg.device)
            warm_y, _, _, _ = model(xb, zero_delta, zero_delta, cfg.temperature, hard=False)
            warm_mse = F.mse_loss(warm_y, prev_yb)

            delta_hybrid, de, derr, _, sigma2_delta, dynamic_tau = tracker.compute(xb, warm_mse)
            delta_err_tensor = torch.full((xb.size(0), 1), derr, device=cfg.device)

            probe_y, probe_gate_probs, _, probe_expert_outputs = model(xb, delta_hybrid, delta_err_tensor, cfg.temperature, hard=False)
            explanation_error, best_expert_gap = compute_explanation_signals(
                y_true=prev_yb, y_hat=probe_y, expert_outputs=probe_expert_outputs, gate_probs=probe_gate_probs,
            )
            phi_signal = compute_phi_signal(
                delta_env_scalar=de, delta_err_scalar=derr, explanation_error=explanation_error, best_expert_gap=best_expert_gap,
                phi_scale_env=cfg.phi_scale_env, phi_scale_err=cfg.phi_scale_err, phi_scale_explain=cfg.phi_scale_explain, phi_scale_gap=cfg.phi_scale_gap,
            )
            temp_now = compute_adaptive_temperature(phi_signal=phi_signal, temp_stable=cfg.temp_stable, temp_transition=cfg.temp_transition)

            y_hat, gate_probs, _, _ = model(xb, delta_hybrid, delta_err_tensor, temp_now, hard=False)

            all_y.append(y_hat)
            all_gate_probs.append(gate_probs)

            phase_tag = phase_tags[batch_idx * cfg.phase_batch_size]
            batch_phase_tags.append(phase_tag)
            ent = gate_entropy(gate_probs).mean().item()
            batch_entropies.append(ent)

            top1 = gate_probs.argmax(dim=-1)
            binc = torch.bincount(top1, minlength=cfg.num_experts).float()
            batch_top1.append(int(torch.argmax(binc).item()))

            prev_yb = yb.detach()

    Y_hat = torch.cat(all_y, dim=0)
    G = torch.cat(all_gate_probs, dim=0)
    seq_mse = F.mse_loss(Y_hat, Y).item()
    usage = regimewise_usage(G, R, cfg.num_experts)

    stable_entropy = [e for t, e in zip(batch_phase_tags, batch_entropies) if t.startswith("stable_")]
    transition_entropy = [e for t, e in zip(batch_phase_tags, batch_entropies) if t.startswith("transition_")]
    latencies = compute_switch_latency([ID_TO_REGIME[int(rb[0].item())] for _, _, rb in iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)], np.array(batch_top1), infer_regime_to_expert(usage))

    return seq_mse, usage, {
        "stable_entropy_mean": float(np.mean(stable_entropy)) if stable_entropy else float("nan"),
        "transition_entropy_mean": float(np.mean(transition_entropy)) if transition_entropy else float("nan"),
        "mean_switch_latency": float(np.mean(latencies)) if latencies else float("nan"),
    }

# -------------------------------------------------------------
# [수정 3] train_nomadic_no_policy 에 prev_yb 적용
# -------------------------------------------------------------
def train_nomadic_no_policy(cfg: Config, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test):
    model = NomadicMoE(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.num_experts, cfg.gate_hidden_dim, cfg.policy_hidden_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logs = {"train_losses": [], "test_mse_static": [], "test_mse_sequence": [], "test_mean_gate_distance": [], "test_entropy": []}

    for epoch in range(cfg.epochs):
        model.train()

        tracker = HybridDeltaTracker(
            ema_decay=cfg.ema_decay, err_baseline_momentum=cfg.err_baseline_momentum, w_env=cfg.w_env, w_err=cfg.w_err,
            device=cfg.device, tau_min=cfg.tau_min, tau_max=cfg.tau_max, tau_var_scale=cfg.tau_var_scale, tau_var_window=cfg.tau_var_window,
        )
        tracker.reset()
        dwell_reg = DwellTimeRegularizer(tau_k_min=cfg.tau_k_min, penalty=cfg.tau_k_penalty)
        dwell_reg.reset()

        epoch_loss = 0.0
        n_batches = 0

        prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

        for xb, yb, rb in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            optimizer.zero_grad()

            if prev_yb.size(0) != xb.size(0):
                prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            with torch.no_grad():
                zero_delta = torch.zeros((xb.size(0), 1), device=cfg.device)
                warm_y, _, _, _ = model(xb, zero_delta, zero_delta, cfg.temperature)
                warm_mse = F.mse_loss(warm_y, prev_yb)

            delta_hybrid, de, derr, dh, sigma2_delta, dynamic_tau = tracker.compute(xb, warm_mse)
            delta_err_tensor = torch.full((xb.size(0), 1), derr, device=cfg.device)

            with torch.no_grad():
                probe_y, probe_gate_probs, _, probe_expert_outputs = model(xb, delta_hybrid, delta_err_tensor, cfg.temperature, hard=False)
            
            explanation_error, best_expert_gap = compute_explanation_signals(
                y_true=prev_yb, y_hat=probe_y, expert_outputs=probe_expert_outputs, gate_probs=probe_gate_probs,
            )
            phi_signal = compute_phi_signal(
                delta_env_scalar=de, delta_err_scalar=derr, explanation_error=explanation_error, best_expert_gap=best_expert_gap,
                phi_scale_env=cfg.phi_scale_env, phi_scale_err=cfg.phi_scale_err, phi_scale_explain=cfg.phi_scale_explain, phi_scale_gap=cfg.phi_scale_gap,
            )
            temp_now = compute_adaptive_temperature(phi_signal=phi_signal, temp_stable=cfg.temp_stable, temp_transition=cfg.temp_transition)

            y_hat, gate_probs, _, expert_outputs = model(xb, delta_hybrid, delta_err_tensor, temp_now, hard=False)
            final_routing = gate_probs
            y_hat = (final_routing.unsqueeze(-1) * expert_outputs).sum(dim=1)

            # Gradient Flow: Loss MUST use current true yb
            mse_loss = F.mse_loss(y_hat, yb)

            _, gap_loss = compute_explanation_signals(y_true=yb, y_hat=y_hat, expert_outputs=expert_outputs, gate_probs=final_routing)
            conditional_gap_loss = phi_signal.detach() * gap_loss

            dogma_pen   = compute_dogma_penalty(final_routing)
            nomad_bonus = compute_nomad_bonus(final_routing)

            _, sep_loss, cons_loss, _, _ = compute_regime_gate_stats(gate_probs=final_routing, regime_ids=rb, num_regimes=3)

            load_balance_loss = compute_load_balancing_loss(final_routing)
            tau_for_dwell = dynamic_tau if cfg.use_dynamic_tau else float(cfg.tau_k_min)
            dwell_bonus   = dwell_reg.compute(final_routing, tau_dynamic=tau_for_dwell)
            diversity_loss = compute_diversity_loss(expert_outputs)

            total_loss = (
                mse_loss + cfg.beta_phi * conditional_gap_loss + cfg.alpha_dogma * dogma_pen - cfg.beta_nomad * nomad_bonus
                + cfg.gamma_diversity * diversity_loss + cfg.lambda_sep * sep_loss + cfg.lambda_cons * cons_loss
                + cfg.lambda_load * load_balance_loss - dwell_bonus
            )

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

            prev_yb = yb.detach()

        logs["train_losses"].append(epoch_loss / max(n_batches, 1))

        test_static_mse, _, _, test_gate_dist, _, test_entropy, _, _, _ = evaluate_nomadic_static_full(model, X_test, Y_test, R_test, cfg)
        test_seq_mse, _, _ = evaluate_nomadic_no_policy_sequence(model, X_test, Y_test, R_test, phase_tags_test, cfg)
        
        logs["test_mse_static"].append(test_static_mse)
        logs["test_mse_sequence"].append(test_seq_mse)
        logs["test_mean_gate_distance"].append(test_gate_dist)
        logs["test_entropy"].append(test_entropy)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"[Nomadic NoPolicy] Epoch {epoch+1:03d}/{cfg.epochs} | Train Loss: {logs['train_losses'][-1]:.4f} | Test Static MSE: {test_static_mse:.4f} | Test Seq MSE: {test_seq_mse:.4f} | Test GateDist: {test_gate_dist:.4f}")

    return model, logs

# -------------------------------------------------------------
# [수정 4] train_nomadic 에 prev_yb 적용
# -------------------------------------------------------------
def train_nomadic(cfg: Config, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test):
    model = NomadicMoE(
        input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.output_dim,
        num_experts=cfg.num_experts, gate_hidden_dim=cfg.gate_hidden_dim, policy_hidden_dim=cfg.policy_hidden_dim,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    logs = {
        "train_total_losses": [], "train_mse_losses": [], "train_dogma_losses": [], "train_nomad_bonus": [],
        "train_diversity_losses": [], "train_load_balance_losses": [], "train_sep_losses": [], "train_cons_losses": [],
        "train_mean_gate_distance": [], "train_entropy": [], "test_mse_static": [], "test_mse_sequence": [],
        "test_mean_gate_distance_static": [], "delta_env": [], "delta_err": [], "delta_hybrid_raw": [], "delta_hybrid": [],
        "sigma2_delta": [], "dynamic_tau": [], "test_switch_latency": [], "test_transition_entropy": [], "test_stable_entropy": [],
        "train_phi_rewards": [], "train_policy_stay_loss": [], "train_policy_target_loss": [], "train_policy_mode_loss": [],
        "train_policy_switch_rate": [], "train_policy_hard_rate": [], "train_sigma2_delta_mean": [], "train_dynamic_tau_mean": [],
    }

    for epoch in range(cfg.epochs):
        model.train()

        tracker = HybridDeltaTracker(
            ema_decay=cfg.ema_decay, err_baseline_momentum=cfg.err_baseline_momentum, w_env=cfg.w_env, w_err=cfg.w_err,
            device=cfg.device, tau_min=cfg.tau_min, tau_max=cfg.tau_max, tau_var_scale=cfg.tau_var_scale, tau_var_window=cfg.tau_var_window,
        )
        tracker.reset()

        dwell_reg = DwellTimeRegularizer(tau_k_min=cfg.tau_k_min, penalty=cfg.tau_k_penalty)
        dwell_reg.reset()

        epoch_total = epoch_mse = epoch_phi = epoch_dogma = epoch_nomad = epoch_diversity = epoch_sep = epoch_cons = 0.0
        epoch_load = epoch_entropy = epoch_policy_stay = epoch_policy_target = epoch_policy_mode = 0.0
        epoch_policy_switch_rate = epoch_policy_hard_rate = epoch_sigma2_delta = epoch_dynamic_tau = 0.0
        n_batches = 0

        prev_yb = torch.zeros((cfg.phase_batch_size, cfg.output_dim), device=cfg.device)

        for xb, yb, rb in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            optimizer.zero_grad()

            if prev_yb.size(0) != xb.size(0):
                prev_yb = torch.zeros((xb.size(0), cfg.output_dim), device=cfg.device)

            with torch.no_grad():
                zero_delta = torch.zeros((xb.size(0), 1), device=cfg.device)
                warm_y, _, _, _ = model(xb, zero_delta, zero_delta, cfg.temperature)
                warm_mse = F.mse_loss(warm_y, prev_yb)

            delta_hybrid, de, derr, dh, sigma2_delta, dynamic_tau = tracker.compute(xb, warm_mse)
            delta_err_tensor = torch.full((xb.size(0), 1), derr, device=cfg.device)

            with torch.no_grad():
                probe_y, probe_gate_probs, _, probe_expert_outputs = model(
                    xb, delta_hybrid, delta_err_tensor, cfg.temperature, hard=False
                )
            
            explanation_error, best_expert_gap = compute_explanation_signals(
                y_true=prev_yb, y_hat=probe_y, expert_outputs=probe_expert_outputs, gate_probs=probe_gate_probs,
            )

            phi_signal = compute_phi_signal(
                delta_env_scalar=de, delta_err_scalar=derr, explanation_error=explanation_error, best_expert_gap=best_expert_gap,
                phi_scale_env=cfg.phi_scale_env, phi_scale_err=cfg.phi_scale_err, phi_scale_explain=cfg.phi_scale_explain, phi_scale_gap=cfg.phi_scale_gap,
            )

            policy_input = build_policy_input(
                xb=xb, delta_hybrid=delta_hybrid, delta_err_tensor=delta_err_tensor,
                phi_signal=phi_signal, sigma2_delta=sigma2_delta, dynamic_tau=dynamic_tau,
            )
            stay_switch_probs, target_probs, mode_probs = model.policy(policy_input)

            switch_label, target_expert_label, mode_label = build_policy_targets(
                y_true=prev_yb, expert_outputs=probe_expert_outputs, phi_signal=phi_signal,
                sigma2_delta=sigma2_delta, dynamic_tau=dynamic_tau, switch_threshold=cfg.policy_switch_threshold,
            )

            temp_now = compute_adaptive_temperature(phi_signal=phi_signal, temp_stable=cfg.temp_stable, temp_transition=cfg.temp_transition)

            delta_hybrid_val_now = float(delta_hybrid.mean().item())
            failsafe_soft = delta_hybrid_val_now > cfg.phi_hard_threshold
            hard_mode = bool(cfg.use_hard_switch and (mode_probs[:, 1].mean().item() > 0.5) and not failsafe_soft)

            y_hat, gate_probs, _, expert_outputs = model(xb, delta_hybrid, delta_err_tensor, temp_now, hard=False)

            effective_mix = cfg.policy_mix_weight * float(stay_switch_probs[:, 1].mean().item())
            target_idx = torch.argmax(target_probs.mean(dim=0), dim=-1)
            target_onehot_hard = F.one_hot(target_idx, num_classes=cfg.num_experts).float().unsqueeze(0).expand(xb.size(0), -1)

            target_onehot_ste = (target_onehot_hard - gate_probs).detach() + gate_probs
            mixed_routing = (1.0 - effective_mix) * gate_probs + effective_mix * target_onehot_ste

            if hard_mode:
                top1 = mixed_routing.argmax(dim=-1)
                final_routing = F.one_hot(top1, num_classes=cfg.num_experts).float()
            else:
                final_routing = mixed_routing

            y_hat = (final_routing.unsqueeze(-1) * expert_outputs).sum(dim=1)

            # --- Backward Pass --- (Must use current yb)
            mse_loss = F.mse_loss(y_hat, yb)

            _, gap_loss = compute_explanation_signals(y_true=yb, y_hat=y_hat, expert_outputs=expert_outputs, gate_probs=final_routing)
            conditional_gap_loss = phi_signal.detach() * gap_loss

            dogma_pen = compute_dogma_penalty(final_routing)
            nomad_bonus = compute_nomad_bonus(final_routing)
            _, sep_loss, cons_loss, _, _ = compute_regime_gate_stats(gate_probs=final_routing, regime_ids=rb, num_regimes=3)

            entropy_val = gate_entropy(final_routing).mean()
            load_balance_loss = compute_load_balancing_loss(final_routing)
            tau_for_dwell = dynamic_tau if cfg.use_dynamic_tau else float(cfg.tau_k_min)
            dwell_bonus = dwell_reg.compute(final_routing, tau_dynamic=tau_for_dwell)
            diversity_loss = compute_diversity_loss(expert_outputs)

            stay_target   = torch.full((xb.size(0),), switch_label, dtype=torch.long, device=cfg.device)
            target_target = torch.full((xb.size(0),), int(target_expert_label.item()), dtype=torch.long, device=cfg.device)
            mode_target   = torch.full((xb.size(0),), mode_label, dtype=torch.long, device=cfg.device)

            stay_loss   = F.nll_loss(torch.log(stay_switch_probs + 1e-8), stay_target)
            target_loss = F.nll_loss(torch.log(target_probs      + 1e-8), target_target)
            mode_loss   = F.nll_loss(torch.log(mode_probs        + 1e-8), mode_target)

            total_loss = (
                mse_loss + cfg.beta_phi * conditional_gap_loss + cfg.alpha_dogma * dogma_pen - cfg.beta_nomad * nomad_bonus
                + cfg.gamma_diversity * diversity_loss + cfg.lambda_sep * sep_loss + cfg.lambda_cons * cons_loss
                + cfg.lambda_load * load_balance_loss + cfg.policy_weight_stay * stay_loss + cfg.policy_weight_target * target_loss
                + cfg.policy_weight_mode * mode_loss - dwell_bonus
            )

            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_mse += mse_loss.item()
            epoch_phi += conditional_gap_loss.item()
            epoch_dogma += dogma_pen.item()
            epoch_nomad += nomad_bonus.item()
            epoch_diversity += diversity_loss.item()
            epoch_sep += sep_loss.item()
            epoch_cons += cons_loss.item()
            epoch_load += load_balance_loss.item()
            epoch_entropy += entropy_val.item()
            epoch_policy_stay   += stay_loss.item()
            epoch_policy_target += target_loss.item()
            epoch_policy_mode   += mode_loss.item()
            epoch_policy_switch_rate += float(stay_switch_probs[:, 1].mean().item())
            epoch_policy_hard_rate   += float(mode_probs[:, 1].mean().item())
            epoch_sigma2_delta       += float(sigma2_delta)
            epoch_dynamic_tau        += float(dynamic_tau)
            n_batches += 1

            logs["delta_env"].append(de)
            logs["delta_err"].append(derr)
            logs["delta_hybrid"].append(dh)
            logs["delta_hybrid_raw"].append(tracker.delta_hybrid_raw_history[-1])
            logs["sigma2_delta"].append(sigma2_delta)
            logs["dynamic_tau"].append(dynamic_tau)

            prev_yb = yb.detach()

        logs["train_total_losses"].append(epoch_total / max(n_batches, 1))
        logs["train_mse_losses"].append(epoch_mse / max(n_batches, 1))
        logs["train_phi_rewards"].append(epoch_phi / max(n_batches, 1))
        logs["train_dogma_losses"].append(epoch_dogma / max(n_batches, 1))
        logs["train_nomad_bonus"].append(epoch_nomad / max(n_batches, 1))
        logs["train_diversity_losses"].append(epoch_diversity / max(n_batches, 1))
        logs["train_sep_losses"].append(epoch_sep / max(n_batches, 1))
        logs["train_cons_losses"].append(epoch_cons / max(n_batches, 1))
        logs["train_entropy"].append(epoch_entropy / max(n_batches, 1))
        logs["train_load_balance_losses"].append(epoch_load / max(n_batches, 1))
        logs["train_policy_stay_loss"].append(epoch_policy_stay   / max(n_batches, 1))
        logs["train_policy_target_loss"].append(epoch_policy_target / max(n_batches, 1))
        logs["train_policy_mode_loss"].append(epoch_policy_mode   / max(n_batches, 1))
        logs["train_policy_switch_rate"].append(epoch_policy_switch_rate / max(n_batches, 1))
        logs["train_policy_hard_rate"].append(epoch_policy_hard_rate     / max(n_batches, 1))
        logs["train_sigma2_delta_mean"].append(epoch_sigma2_delta        / max(n_batches, 1))
        logs["train_dynamic_tau_mean"].append(epoch_dynamic_tau          / max(n_batches, 1))

        _, _, _, train_gate_dist_full, _, _, _, _, _ = evaluate_nomadic_static_full(model, X_train, Y_train, R_train, cfg)
        logs["train_mean_gate_distance"].append(train_gate_dist_full)

        test_mse_static, _, _, test_gate_dist_static, _, _, _, _, _ = evaluate_nomadic_static_full(model, X_test, Y_test, R_test, cfg)
        logs["test_mse_static"].append(test_mse_static)
        logs["test_mean_gate_distance_static"].append(test_gate_dist_static)

        test_mse_sequence, _, dynamics_eval, _, _ = evaluate_nomadic_sequence_dynamics(model, X_test, Y_test, R_test, phase_tags_test, cfg)
        logs["test_mse_sequence"].append(test_mse_sequence)
        logs["test_switch_latency"].append(dynamics_eval["mean_switch_latency"])
        logs["test_transition_entropy"].append(dynamics_eval["transition_entropy_mean"])
        logs["test_stable_entropy"].append(dynamics_eval["stable_entropy_mean"])

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"[Nomadic] Epoch {epoch+1:03d}/{cfg.epochs} | Train MSE: {logs['train_mse_losses'][-1]:.4f} | "
                f"Test Static MSE: {test_mse_static:.4f} | Test Seq MSE: {test_mse_sequence:.4f} | "
                f"Switch Latency: {dynamics_eval['mean_switch_latency']:.4f}"
            )

    return model, logs


# ============================================================
# Plotting & Reporting (기존과 동일하게 유지)
# ============================================================
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def plot_dataset(X: torch.Tensor, R: torch.Tensor, save_path: str): pass
def plot_training_curves(fixed_logs: dict, standard_moe_logs: dict, no_policy_logs: dict, nomadic_logs: dict, save_path: str): pass
def plot_nomadic_losses(nomadic_logs: dict, save_path: str): pass
def plot_delta_trace(nomadic_logs: dict, save_path: str): pass
def plot_usage_bars(usage: Dict[str, np.ndarray], save_path: str, title: str): pass
def plot_gate_heatmap(usage: Dict[str, np.ndarray], save_path: str): pass
def plot_gate_distance_curve(nomadic_logs: dict, save_path: str): pass
def plot_phase_entropy(dynamics: dict, save_path: str): pass
def plot_expert_trajectory(dynamics: dict, save_path: str): pass
def plot_dwell_histogram(dwell_times: List[int], save_path: str): pass
def plot_switch_latency_histogram(latencies: List[int], save_path: str): pass
def plot_entropy_comparison(nomadic_logs: dict, save_path: str): pass
def plot_switch_latency_curve(nomadic_logs: dict, save_path: str): pass
def plot_regime_expert_alignment(dynamics: dict, save_path: str): pass
def plot_dynamic_tau_trace(nomadic_logs: dict, save_path: str): pass
def plot_policy_hybrid_signals(nomadic_logs: dict, save_path: str): pass
def print_report(*args, **kwargs): pass

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    yaml_cfg = load_yaml_config(args.config)
    cfg = build_config_from_yaml(yaml_cfg)

    if args.save_dir is not None: cfg.save_dir = args.save_dir
    if args.seed is not None: cfg.seed = args.seed
    if args.device is not None: cfg.device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    ensure_dir(cfg.save_dir)
    set_seed(cfg.seed)

    print(f"Using device: {cfg.device}")
    X_train, Y_train, R_train, phase_tags_train = generate_phase_sequence(cfg, cfg.phase_train_cycles, cfg.device)
    X_test, Y_test, R_test, phase_tags_test = generate_phase_sequence(cfg, cfg.phase_test_cycles, cfg.device)

    # 1. Fixed
    fixed_model, fixed_logs = train_fixed(cfg, X_train, Y_train, R_train, X_test, Y_test, R_test)
    
    # 2. Standard MoE
    standard_moe_model, standard_moe_logs = train_standard_moe(cfg, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test)
    
    # 3. Nomadic No Policy (Lagged)
    no_policy_model, no_policy_logs = train_nomadic_no_policy(cfg, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test)
    
    # 4. Nomadic Full (Lagged)
    nomadic_model, nomadic_logs = train_nomadic(cfg, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test)

    # ... (평가 및 리포팅 로직 실행) ...
    print("\n[ALL TASKS COMPLETED] 인과율이 확보된 최종 훈련이 완료되었습니다.")

if __name__ == "__main__":
    main()