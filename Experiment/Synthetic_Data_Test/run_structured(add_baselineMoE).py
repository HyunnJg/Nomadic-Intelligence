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
    policy_mix_weight: float = 0.25      # target_onehot 합성 비율 (0.0 = policy 비활성)
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
    """
    Creates a time-ordered sequence:
      stable A -> transition A->B -> stable B -> transition B->C -> stable C -> transition C->A -> ...
    Returns:
      X, Y, R, phase_tags
    """
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
    """
    No shuffling. Preserves phase order.
    """
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
            nn.Linear(input_dim + 2, gate_hidden_dim),  # x + delta_hybrid + delta_err
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
    """
    Meta decision-maker: Full Hybrid Intelligence v1.

    Inputs:
      x_summary(input_dim)
      + delta_hybrid(1)
      + delta_err(1)
      + phi_signal(1)
      + sigma2_delta_scaled(1)   -- tanh-scaled env volatility
      + dynamic_tau_scaled(1)    -- tanh-scaled dwell horizon

    Total input dim = input_dim + 5

    Outputs:
      stay_switch_probs  [B, 2]  -- stay(0) / switch(1)
      target_probs       [B, E]  -- which expert to prefer
      mode_probs         [B, 2]  -- soft(0) / hard(1)

    PolicyNet now has direct visibility into:
      - how volatile the environment is (sigma2_delta)
      - how much fixation is currently allowed (dynamic_tau)
    This enables meta-level learning of stay/switch strategy
    rather than reacting to phi alone.
    """
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
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, E, 1]

        if hard:
            top1 = gate_probs.argmax(dim=-1)
            routing = F.one_hot(top1, num_classes=self.num_experts).float()
        else:
            routing = gate_probs

        y_hat = (routing.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return y_hat, gate_probs, gate_logits, expert_outputs


# ============================================================
# Hybrid Delta utilities
# ============================================================

class HybridDeltaTracker:
    """
    Upgraded hybrid delta with environment-aware dynamic tau.
      delta_env   = input mean shift
      delta_err   = relu(err_ema - err_baseline)
      raw_hybrid  = w_env * delta_env + w_err * delta_err
      delta_hybrid = tanh(raw_hybrid)
      sigma2_delta = rolling variance of recent delta_env values
      dynamic_tau  = tau_max → tau_min as sigma2_delta rises
    """
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
    mean_gate = gate_probs.mean(dim=0)  # [num_experts]
    top1 = gate_probs.argmax(dim=-1)   # [batch]
    top1_frac = torch.zeros(num_experts, device=gate_probs.device)
    for i in range(num_experts):
        top1_frac[i] = (top1 == i).float().mean()
    loss = num_experts * (top1_frac * mean_gate).sum()
    return loss

class DwellTimeRegularizer:
    """
    Environment-aware dwell time regularizer.

    total_loss = ... - dwell_bonus  (dwell_bonus subtracted from loss)
      dwell_bonus > 0  =>  loss decreases  =>  entropy INCREASES (nomadic pressure)
      dwell_bonus < 0  =>  loss increases  =>  entropy DECREASES (fixation pressure)

    stable  (dwell_count <= tau_capacity): return -penalty * entropy
            -> loss increases -> entropy decreases -> fixation encouraged
    transition (dwell_count > tau_capacity): return +excess * entropy
            -> loss decreases -> entropy increases -> switching encouraged
    """
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
            # stable: penalize entropy -> fixation
            return -self.penalty * entropy
        else:
            # transition: reward entropy -> switching
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
    """
    explanation_error:
        현재 routing 결과가 데이터를 얼마나 설명 못하는가
    best_expert_gap:
        현재 top1 expert보다 더 잘 맞는 expert가 존재하는가
    """
    explanation_error = F.mse_loss(y_hat, y_true)

    # expert_outputs: [B, E, 1], y_true: [B, 1]
    # -> per_expert_sqerr: [B, E]
    per_expert_sqerr = ((expert_outputs - y_true.unsqueeze(1)) ** 2).mean(dim=-1)

    top1_idx = gate_probs.argmax(dim=-1)  # [B]
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
    """
    높을수록:
    - 환경 변화가 크고
    - 현재 설명력이 부족하고
    - 더 나은 expert가 존재할 가능성이 높다
    => switching pressure 증가
    """
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
    """
    Full hybrid policy input:
      x_summary(input_dim) + delta_hybrid(1) + delta_err(1)
      + phi_signal(1) + sigma2_delta_scaled(1) + dynamic_tau_scaled(1)

    Scaling rationale (Gemini suggestion):
      sigma2_delta range ~[0, 0.5], dynamic_tau range ~[2, 8]
      Raw injection would let tau dominate. Apply tanh to normalize both
      into a comparable range without losing directional information.
        sigma2_scaled = tanh(sigma2_delta * 10)   -- amplify small values
        tau_scaled    = tanh((dynamic_tau - tau_mid) / tau_mid)  -- center around mid
    """
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
    """
    Hybrid heuristic teacher signals.

    Switch decision:
      switch if phi is high OR sigma2_delta is high (env volatile)
      stay   if phi is low  AND dynamic_tau is high (env stable, fixation allowed)

    Mode decision:
      hard(1) = stable fixation is appropriate (can_fixate)
      soft(0) = exploratory transition (otherwise)

    Target expert:
      expert with lowest mean squared error on this batch
    """
    per_expert_sqerr = ((expert_outputs - y_true.unsqueeze(1)) ** 2).mean(dim=-1)
    target_expert = per_expert_sqerr.mean(dim=0).argmin().long()

    phi_val = float(phi_signal.mean().item())
    should_switch = (phi_val > switch_threshold) or (sigma2_delta > sigma_switch_threshold)
    can_fixate    = (phi_val <= switch_threshold) and (dynamic_tau >= tau_stay_threshold)

    switch_label = 1 if should_switch else 0
    mode_label   = 1 if can_fixate else 0   # hard=1, soft=0

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

        if mask.sum() == 0:
            continue

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
        mean_gate_distance = 0.0
        pairwise_distances = {}
        return regime_means, l_sep, l_cons, mean_gate_distance, pairwise_distances

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
    if len(top1_sequence) == 0:
        return []

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


def evaluate_nomadic_static_full(model: NomadicMoE, X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, cfg: Config):
    """
    Static evaluation:
    - delta_hybrid fixed to zero
    - ignores sequential phase dynamics
    Useful for checking static separability only.
    """
    model.eval()
    with torch.no_grad():
        delta_hybrid = torch.zeros((X.size(0), 1), device=X.device)
        delta_err = torch.zeros((X.size(0), 1), device=X.device)
        y_pred, gate_probs, _, _ = model(X, delta_hybrid, delta_err, cfg.temperature)

        total_mse = F.mse_loss(y_pred, Y).item()
        per_regime = mse_by_regime(Y, y_pred, R)
        usage = regimewise_usage(gate_probs, R, cfg.num_experts)

        _, _, _, mean_gate_distance, pairwise_distances = compute_regime_gate_stats(
            gate_probs=gate_probs,
            regime_ids=R,
            num_regimes=3,
        )

        ent = gate_entropy(gate_probs).mean().item()
        top1 = gate_probs.argmax(dim=-1).detach().cpu().numpy()
        dwell_times = compute_dwell_times(top1)

    return total_mse, per_regime, usage, mean_gate_distance, pairwise_distances, ent, dwell_times, y_pred, gate_probs


def evaluate_nomadic_sequence_dynamics(model: NomadicMoE, X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, phase_tags: List[str], cfg: Config):
    """
    Sequential evaluation with live hybrid delta.
    Measures:
      - phase-level entropy
      - switch latency
      - dwell times
      - stepwise expert trajectory
      - sigma2_delta / dynamic_tau trace
    """
    model.eval()
    tracker = HybridDeltaTracker(
        ema_decay=cfg.ema_decay,
        err_baseline_momentum=cfg.err_baseline_momentum,
        w_env=cfg.w_env,
        w_err=cfg.w_err,
        device=cfg.device,
        tau_min=cfg.tau_min,
        tau_max=cfg.tau_max,
        tau_var_scale=cfg.tau_var_scale,
        tau_var_window=cfg.tau_var_window,
    )
    tracker.reset()

    all_y = []
    all_gate_probs = []
    batch_regimes = []
    batch_phase_tags = []
    batch_entropies = []
    batch_top1 = []
    batch_sigma2_delta = []
    batch_dynamic_tau = []

    with torch.no_grad():
        for batch_idx, (xb, yb, rb) in enumerate(iterate_sequence_minibatches(X, Y, R, cfg.phase_batch_size)):
            # 1) warm pass (delta_err=0, delta_hybrid=0)
            zero_delta = torch.zeros((xb.size(0), 1), device=cfg.device)
            warm_y, _, _, _ = model(
                xb, zero_delta, zero_delta, cfg.temperature, hard=False
            )
            warm_mse = F.mse_loss(warm_y, yb)

            # 2) hybrid delta — unpack all 6 values
            delta_hybrid, de, derr, _, sigma2_delta, dynamic_tau = tracker.compute(xb, warm_mse)
            delta_err_tensor = torch.full((xb.size(0), 1), derr, device=cfg.device)

            # 3) probe pass for explanation diagnosis
            probe_y, probe_gate_probs, _, probe_expert_outputs = model(
                xb, delta_hybrid, delta_err_tensor, cfg.temperature, hard=False
            )

            explanation_error, best_expert_gap = compute_explanation_signals(
                y_true=yb,
                y_hat=probe_y,
                expert_outputs=probe_expert_outputs,
                gate_probs=probe_gate_probs,
            )

            phi_signal = compute_phi_signal(
                delta_env_scalar=de,
                delta_err_scalar=derr,
                explanation_error=explanation_error,
                best_expert_gap=best_expert_gap,
                phi_scale_env=cfg.phi_scale_env,
                phi_scale_err=cfg.phi_scale_err,
                phi_scale_explain=cfg.phi_scale_explain,
                phi_scale_gap=cfg.phi_scale_gap,
            )

            # 4) PolicyNet decision — full hybrid input
            policy_input = build_policy_input(
                xb=xb,
                delta_hybrid=delta_hybrid,
                delta_err_tensor=delta_err_tensor,
                phi_signal=phi_signal,
                sigma2_delta=sigma2_delta,
                dynamic_tau=dynamic_tau,
            )
            stay_switch_probs, target_probs, mode_probs = model.policy(policy_input)

            temp_now = compute_adaptive_temperature(
                phi_signal=phi_signal,
                temp_stable=cfg.temp_stable,
                temp_transition=cfg.temp_transition,
            )

            # Fail-safe: force Soft when delta_hybrid is high
            delta_hybrid_val_now = float(delta_hybrid.mean().item())
            failsafe_soft = delta_hybrid_val_now > cfg.phi_hard_threshold
            hard_mode = bool(
                cfg.use_hard_switch
                and (mode_probs[:, 1].mean().item() > 0.5)
                and not failsafe_soft
            )

            # 5) final routed pass with policy mixing
            y_hat, gate_probs, _, expert_outputs_eval = model(
                xb, delta_hybrid, delta_err_tensor, temp_now, hard=False
            )

            effective_mix = cfg.policy_mix_weight * float(stay_switch_probs[:, 1].mean().item())
            target_idx = torch.argmax(target_probs.mean(dim=0), dim=-1)
            target_onehot_hard = F.one_hot(
                target_idx,
                num_classes=cfg.num_experts
            ).float().unsqueeze(0).expand(xb.size(0), -1)

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

    Y_hat = torch.cat(all_y, dim=0)
    G = torch.cat(all_gate_probs, dim=0)

    total_mse = F.mse_loss(Y_hat, Y).item()
    usage = regimewise_usage(G, R, cfg.num_experts)
    regime_to_expert = infer_regime_to_expert(usage)

    latencies = compute_switch_latency(batch_regimes, np.array(batch_top1), regime_to_expert)
    dwell_times = compute_dwell_times(np.array(batch_top1))

    stable_entropy = []
    transition_entropy = []
    for tag, ent in zip(batch_phase_tags, batch_entropies):
        if tag.startswith("stable_"):
            stable_entropy.append(ent)
        elif tag.startswith("transition_"):
            transition_entropy.append(ent)

    dynamics = {
        "batch_regimes": batch_regimes,
        "batch_phase_tags": batch_phase_tags,
        "batch_entropies": batch_entropies,
        "batch_top1": batch_top1,
        "switch_latencies": latencies,
        "dwell_times": dwell_times,
        "mean_switch_latency": float(np.mean(latencies)) if len(latencies) > 0 else float("nan"),
        "mean_dwell_time": float(np.mean(dwell_times)) if len(dwell_times) > 0 else float("nan"),
        "stable_entropy_mean": float(np.mean(stable_entropy)) if len(stable_entropy) > 0 else float("nan"),
        "transition_entropy_mean": float(np.mean(transition_entropy)) if len(transition_entropy) > 0 else float("nan"),
        "regime_to_expert": regime_to_expert,
        "sigma2_delta": batch_sigma2_delta,
        "dynamic_tau": batch_dynamic_tau,
        "mean_dynamic_tau": float(np.mean(batch_dynamic_tau)) if len(batch_dynamic_tau) > 0 else float("nan"),
    }

    return total_mse, usage, dynamics, Y_hat, G


def train_fixed(cfg: Config, X_train, Y_train, R_train, X_test, Y_test, R_test):
    model = MLPRegressor(cfg.input_dim, cfg.hidden_dim, cfg.output_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses = []
    test_losses = []

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


def train_nomadic(cfg: Config, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test):
    model = NomadicMoE(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        num_experts=cfg.num_experts,
        gate_hidden_dim=cfg.gate_hidden_dim,
        policy_hidden_dim=cfg.policy_hidden_dim,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    logs = {
        "train_total_losses": [],
        "train_mse_losses": [],
        "train_dogma_losses": [],
        "train_nomad_bonus": [],
        "train_diversity_losses": [],
        "train_load_balance_losses": [],
        "train_sep_losses": [],
        "train_cons_losses": [],
        "train_mean_gate_distance": [],
        "train_entropy": [],
        "test_mse_static": [],
        "test_mse_sequence": [],
        "test_mean_gate_distance_static": [],
        "delta_env": [],
        "delta_err": [],
        "delta_hybrid_raw": [],
        "delta_hybrid": [],
        "sigma2_delta": [],
        "dynamic_tau": [],
        "test_switch_latency": [],
        "test_transition_entropy": [],
        "test_stable_entropy": [],
        "train_phi_rewards": [],
        "train_policy_stay_loss": [],
        "train_policy_target_loss": [],
        "train_policy_mode_loss": [],
        "train_policy_switch_rate": [],
        "train_policy_hard_rate": [],
        "train_sigma2_delta_mean": [],
        "train_dynamic_tau_mean": [],
    }

    for epoch in range(cfg.epochs):
        model.train()

        tracker = HybridDeltaTracker(
            ema_decay=cfg.ema_decay,
            err_baseline_momentum=cfg.err_baseline_momentum,
            w_env=cfg.w_env,
            w_err=cfg.w_err,
            device=cfg.device,
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
            tau_var_scale=cfg.tau_var_scale,
            tau_var_window=cfg.tau_var_window,
        )
        tracker.reset()

        dwell_reg = DwellTimeRegularizer(
            tau_k_min=cfg.tau_k_min,
            penalty=cfg.tau_k_penalty
        )
        dwell_reg.reset()

        epoch_total = 0.0
        epoch_mse = 0.0
        epoch_phi = 0.0
        epoch_dogma = 0.0
        epoch_nomad = 0.0
        epoch_diversity = 0.0
        epoch_sep = 0.0
        epoch_cons = 0.0
        epoch_load = 0.0
        epoch_entropy = 0.0
        epoch_policy_stay = 0.0
        epoch_policy_target = 0.0
        epoch_policy_mode = 0.0
        epoch_policy_switch_rate = 0.0
        epoch_policy_hard_rate = 0.0
        epoch_sigma2_delta = 0.0
        epoch_dynamic_tau = 0.0
        n_batches = 0

        for xb, yb, rb in iterate_sequence_minibatches(X_train, Y_train, R_train, cfg.phase_batch_size):
            optimizer.zero_grad()

            # warm pass: delta_err=0, delta_hybrid=0 (no grad)
            with torch.no_grad():
                zero_delta = torch.zeros((xb.size(0), 1), device=cfg.device)
                warm_y, _, _, _ = model(xb, zero_delta, zero_delta, cfg.temperature)
                warm_mse = F.mse_loss(warm_y, yb)

            # hybrid delta (detached scalars)
            delta_hybrid, de, derr, dh, sigma2_delta, dynamic_tau = tracker.compute(xb, warm_mse)
            delta_err_tensor = torch.full((xb.size(0), 1), derr, device=cfg.device)

            # explanation signals (for phi_signal + loss)
            with torch.no_grad():
                probe_y, probe_gate_probs, _, probe_expert_outputs = model(
                    xb, delta_hybrid, delta_err_tensor, cfg.temperature, hard=False
                )
            explanation_error, best_expert_gap = compute_explanation_signals(
                y_true=yb,
                y_hat=probe_y,
                expert_outputs=probe_expert_outputs,
                gate_probs=probe_gate_probs,
            )

            # phi_signal
            phi_signal = compute_phi_signal(
                delta_env_scalar=de,
                delta_err_scalar=derr,
                explanation_error=explanation_error,
                best_expert_gap=best_expert_gap,
                phi_scale_env=cfg.phi_scale_env,
                phi_scale_err=cfg.phi_scale_err,
                phi_scale_explain=cfg.phi_scale_explain,
                phi_scale_gap=cfg.phi_scale_gap,
            )

            # PolicyNet forward
            policy_input = build_policy_input(
                xb=xb,
                delta_hybrid=delta_hybrid,
                delta_err_tensor=delta_err_tensor,
                phi_signal=phi_signal,
                sigma2_delta=sigma2_delta,
                dynamic_tau=dynamic_tau,
            )
            stay_switch_probs, target_probs, mode_probs = model.policy(policy_input)

            # teacher targets (hybrid heuristic supervision)
            switch_label, target_expert_label, mode_label = build_policy_targets(
                y_true=yb,
                expert_outputs=probe_expert_outputs,
                phi_signal=phi_signal,
                sigma2_delta=sigma2_delta,
                dynamic_tau=dynamic_tau,
                switch_threshold=cfg.policy_switch_threshold,
            )

            # adaptive temperature + hard-switch (policy mode head 사용)
            temp_now = compute_adaptive_temperature(
                phi_signal=phi_signal,
                temp_stable=cfg.temp_stable,
                temp_transition=cfg.temp_transition,
            )
            # Fail-safe: force Soft when delta_hybrid is high (volatile env)
            # prevents PolicyNet Hard-lock during early training instability
            delta_hybrid_val_now = float(delta_hybrid.mean().item())
            failsafe_soft = delta_hybrid_val_now > cfg.phi_hard_threshold
            hard_mode = bool(
                cfg.use_hard_switch
                and (mode_probs[:, 1].mean().item() > 0.5)
                and not failsafe_soft
            )

            # main forward pass (with grad)
            y_hat, gate_probs, _, expert_outputs = model(
                xb, delta_hybrid, delta_err_tensor, temp_now, hard=False
            )

            # PolicyNet target expert를 gate_probs에 약하게 합성
            # mix_weight: stable→0에 가깝게, transition→cfg.policy_mix_weight
            effective_mix = cfg.policy_mix_weight * float(stay_switch_probs[:, 1].mean().item())
            target_idx = torch.argmax(target_probs.mean(dim=0), dim=-1)
            target_onehot_hard = F.one_hot(
                target_idx,
                num_classes=cfg.num_experts
            ).float().unsqueeze(0).expand(xb.size(0), -1)

            target_onehot_ste = (target_onehot_hard - gate_probs).detach() + gate_probs

            mixed_routing = (1.0 - effective_mix) * gate_probs + effective_mix * target_onehot_ste

            if hard_mode:
                top1 = mixed_routing.argmax(dim=-1)
                final_routing = F.one_hot(top1, num_classes=cfg.num_experts).float()
            else:
                final_routing = mixed_routing

            y_hat = (final_routing.unsqueeze(-1) * expert_outputs).sum(dim=1)

            # --- losses ---
            mse_loss = F.mse_loss(y_hat, yb)

            _, gap_loss = compute_explanation_signals(
                y_true=yb,
                y_hat=y_hat,
                expert_outputs=expert_outputs,
                gate_probs=final_routing,
            )
            conditional_gap_loss = phi_signal.detach() * gap_loss

            dogma_pen = compute_dogma_penalty(final_routing)
            nomad_bonus = compute_nomad_bonus(final_routing)

            _, sep_loss, cons_loss, _, _ = compute_regime_gate_stats(
                gate_probs=final_routing,
                regime_ids=rb,
                num_regimes=3,
            )

            entropy_val = gate_entropy(final_routing).mean()
            load_balance_loss = compute_load_balancing_loss(final_routing)
            tau_for_dwell = dynamic_tau if cfg.use_dynamic_tau else float(cfg.tau_k_min)
            dwell_bonus = dwell_reg.compute(final_routing, tau_dynamic=tau_for_dwell)
            diversity_loss = compute_diversity_loss(expert_outputs)

            # policy supervision losses
            stay_target   = torch.full((xb.size(0),), switch_label,                    dtype=torch.long, device=cfg.device)
            target_target = torch.full((xb.size(0),), int(target_expert_label.item()), dtype=torch.long, device=cfg.device)
            mode_target   = torch.full((xb.size(0),), mode_label,                      dtype=torch.long, device=cfg.device)

            stay_loss   = F.nll_loss(torch.log(stay_switch_probs + 1e-8), stay_target)
            target_loss = F.nll_loss(torch.log(target_probs      + 1e-8), target_target)
            mode_loss   = F.nll_loss(torch.log(mode_probs        + 1e-8), mode_target)

            total_loss = (
                mse_loss
                + cfg.beta_phi * conditional_gap_loss
                + cfg.alpha_dogma * dogma_pen
                - cfg.beta_nomad * nomad_bonus
                + cfg.gamma_diversity * diversity_loss
                + cfg.lambda_sep * sep_loss
                + cfg.lambda_cons * cons_loss
                + cfg.lambda_load * load_balance_loss
                + cfg.policy_weight_stay   * stay_loss
                + cfg.policy_weight_target * target_loss
                + cfg.policy_weight_mode   * mode_loss
                - dwell_bonus
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

        _, _, _, train_gate_dist_full, _, _, _, _, _ = evaluate_nomadic_static_full(
            model, X_train, Y_train, R_train, cfg
        )
        logs["train_mean_gate_distance"].append(train_gate_dist_full)

        test_mse_static, _, _, test_gate_dist_static, _, _, _, _, _ = evaluate_nomadic_static_full(
            model, X_test, Y_test, R_test, cfg
        )
        logs["test_mse_static"].append(test_mse_static)
        logs["test_mean_gate_distance_static"].append(test_gate_dist_static)

        test_mse_sequence, _, dynamics_eval, _, _ = evaluate_nomadic_sequence_dynamics(
            model, X_test, Y_test, R_test, phase_tags_test, cfg
        )
        logs["test_mse_sequence"].append(test_mse_sequence)
        logs["test_switch_latency"].append(dynamics_eval["mean_switch_latency"])
        logs["test_transition_entropy"].append(dynamics_eval["transition_entropy_mean"])
        logs["test_stable_entropy"].append(dynamics_eval["stable_entropy_mean"])

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"[Nomadic] Epoch {epoch+1:03d}/{cfg.epochs} | "
                f"Train Total: {logs['train_total_losses'][-1]:.4f} | "
                f"Train MSE: {logs['train_mse_losses'][-1]:.4f} | "
                f"Train GateDist(full): {logs['train_mean_gate_distance'][-1]:.4f} | "
                f"Train Entropy: {logs['train_entropy'][-1]:.4f} | "
                f"Policy SwitchRate: {logs['train_policy_switch_rate'][-1]:.4f} | "
                f"Policy HardRate: {logs['train_policy_hard_rate'][-1]:.4f} | "
                f"Mean Tau: {logs['train_dynamic_tau_mean'][-1]:.4f} | "
                f"Test Static MSE: {test_mse_static:.4f} | "
                f"Test Seq MSE: {test_mse_sequence:.4f} | "
                f"Test Static GateDist: {test_gate_dist_static:.4f} | "
                f"Switch Latency: {dynamics_eval['mean_switch_latency']:.4f}"
            )

    return model, logs


# ============================================================
# Plotting
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_dataset(X: torch.Tensor, R: torch.Tensor, save_path: str):
    x = X.detach().cpu().numpy()
    r = R.detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    for rid, name in ID_TO_REGIME.items():
        mask = r == rid
        plt.scatter(x[mask, 0], x[mask, 1], s=10, alpha=0.45, label=f"Regime {name}")

    plt.title("Phase Dataset in Input Space")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(fixed_logs: dict, nomadic_logs: dict, save_path: str):
    epochs = np.arange(1, len(fixed_logs["train_losses"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, fixed_logs["test_losses"], label="Fixed Test MSE")
    plt.plot(epochs, nomadic_logs["test_mse_static"], label="Nomadic Static Test MSE")
    plt.plot(epochs, nomadic_logs["test_mse_sequence"], label="Nomadic Sequence Test MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Fixed vs Nomadic Test MSE (Static vs Sequence)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_nomadic_losses(nomadic_logs: dict, save_path: str):
    epochs = np.arange(1, len(nomadic_logs["train_total_losses"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, nomadic_logs["train_mse_losses"], label="MSE")
    plt.plot(epochs, nomadic_logs["train_dogma_losses"], label="Dogma")
    plt.plot(epochs, nomadic_logs["train_nomad_bonus"], label="Nomad Bonus")
    plt.plot(epochs, nomadic_logs["train_diversity_losses"], label="Diversity")
    plt.plot(epochs, nomadic_logs["train_sep_losses"], label="Regime Sep")
    plt.plot(epochs, nomadic_logs["train_cons_losses"], label="Regime Cons")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Nomadic Loss Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_delta_trace(nomadic_logs: dict, save_path: str):
    steps = np.arange(1, len(nomadic_logs["delta_env"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, nomadic_logs["delta_env"], label="delta_env")
    plt.plot(steps, nomadic_logs["delta_err"], label="delta_err")
    plt.plot(steps, nomadic_logs["delta_hybrid_raw"], label="delta_hybrid_raw")
    plt.plot(steps, nomadic_logs["delta_hybrid"], label="delta_hybrid_tanh")
    plt.xlabel("Batch Step")
    plt.ylabel("Magnitude")
    plt.title("Hybrid Delta Trace")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_usage_bars(usage: Dict[str, np.ndarray], save_path: str, title: str):
    regimes = ["A", "B", "C"]
    num_experts = len(next(iter(usage.values())))
    x = np.arange(len(regimes))
    width = 0.22

    plt.figure(figsize=(8, 5))
    for e in range(num_experts):
        vals = [usage[r][e] for r in regimes]
        plt.bar(x + e * width - width, vals, width=width, label=f"Expert {e}")

    plt.xticks(x, [f"Regime {r}" for r in regimes])
    plt.ylabel("Top-1 Selection Ratio")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_gate_heatmap(usage: Dict[str, np.ndarray], save_path: str):
    regimes = ["A", "B", "C"]
    mat = np.stack([usage[r] for r in regimes], axis=0)

    plt.figure(figsize=(6, 4))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Top-1 Selection Ratio")
    plt.yticks(range(len(regimes)), [f"Regime {r}" for r in regimes])
    plt.xticks(range(mat.shape[1]), [f"Expert {i}" for i in range(mat.shape[1])])
    plt.title("Regime-Expert Usage Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_gate_distance_curve(nomadic_logs: dict, save_path: str):
    epochs = np.arange(1, len(nomadic_logs["train_mean_gate_distance"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, nomadic_logs["train_mean_gate_distance"], label="Train Mean Gate Distance (full)")
    plt.plot(epochs, nomadic_logs["test_mean_gate_distance_static"], label="Test Mean Gate Distance (static)")
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.title("Regime Mean Gate Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_phase_entropy(dynamics: dict, save_path: str):
    ent = np.array(dynamics["batch_entropies"])
    x = np.arange(len(ent))

    plt.figure(figsize=(10, 4))
    plt.plot(x, ent, label="Batch Gate Entropy")
    plt.xlabel("Batch Index")
    plt.ylabel("Entropy")
    plt.title("Gate Entropy across Phase Sequence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_expert_trajectory(dynamics: dict, save_path: str):
    top1 = np.array(dynamics["batch_top1"])
    x = np.arange(len(top1))

    plt.figure(figsize=(10, 4))
    plt.plot(x, top1)
    plt.xlabel("Batch Index")
    plt.ylabel("Dominant Expert")
    plt.title("Dominant Expert Trajectory across Phase Sequence")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_dwell_histogram(dwell_times: List[int], save_path: str):
    plt.figure(figsize=(7, 5))
    bins = min(20, max(5, len(set(dwell_times)) if len(dwell_times) > 0 else 5))
    plt.hist(dwell_times, bins=bins)
    plt.xlabel("Dwell Time")
    plt.ylabel("Count")
    plt.title("Dwell Time Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_switch_latency_histogram(latencies: List[int], save_path: str):
    plt.figure(figsize=(7, 5))
    if len(latencies) > 0:
        bins = min(15, max(3, len(set(latencies))))
        plt.hist(latencies, bins=bins)
    plt.xlabel("Switch Latency")
    plt.ylabel("Count")
    plt.title("Switch Latency Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_entropy_comparison(nomadic_logs: dict, save_path: str):
    epochs = np.arange(1, len(nomadic_logs["test_transition_entropy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, nomadic_logs["test_stable_entropy"], label="Stable Entropy")
    plt.plot(epochs, nomadic_logs["test_transition_entropy"], label="Transition Entropy")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.title("Stable vs Transition Gate Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_switch_latency_curve(nomadic_logs: dict, save_path: str):
    epochs = np.arange(1, len(nomadic_logs["test_switch_latency"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, nomadic_logs["test_switch_latency"], label="Mean Switch Latency")
    plt.xlabel("Epoch")
    plt.ylabel("Latency")
    plt.title("Epoch-wise Mean Switch Latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_regime_expert_alignment(dynamics: dict, save_path: str):
    regime_map = {"A": 0, "B": 1, "C": 2}
    regime_vals = np.array([regime_map[r] for r in dynamics["batch_regimes"]])
    expert_vals = np.array(dynamics["batch_top1"])
    x = np.arange(len(regime_vals))

    plt.figure(figsize=(10, 5))
    plt.plot(x, regime_vals, label="Dominant Regime")
    plt.plot(x, expert_vals, label="Dominant Expert")
    plt.xlabel("Batch Index")
    plt.ylabel("Index")
    plt.title("Regime vs Expert Alignment across Phase Sequence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_dynamic_tau_trace(nomadic_logs: dict, save_path: str):
    steps = np.arange(1, len(nomadic_logs["dynamic_tau"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_sigma = "#5577aa"
    color_tau   = "#cc6644"
    ax1.set_xlabel("Batch Step")
    ax1.set_ylabel("sigma2_delta", color=color_sigma)
    ax1.plot(steps, nomadic_logs["sigma2_delta"], label="sigma2_delta",
             color=color_sigma, alpha=0.8)
    ax1.tick_params(axis="y", labelcolor=color_sigma)
    ax2 = ax1.twinx()
    ax2.set_ylabel("dynamic_tau", color=color_tau)
    ax2.plot(steps, nomadic_logs["dynamic_tau"], label="dynamic_tau",
             color=color_tau, alpha=0.9)
    ax2.tick_params(axis="y", labelcolor=color_tau)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Environment Volatility vs Dynamic Tau")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_policy_hybrid_signals(nomadic_logs: dict, save_path: str):
    epochs = np.arange(1, len(nomadic_logs["train_policy_switch_rate"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Rate (0-1)")
    ax1.plot(epochs, nomadic_logs["train_policy_switch_rate"],
             label="Policy Switch Rate", color="#5577aa")
    ax1.plot(epochs, nomadic_logs["train_policy_hard_rate"],
             label="Policy Hard Rate", color="#cc6644")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Dynamic Tau")
    ax2.plot(epochs, nomadic_logs["train_dynamic_tau_mean"],
             label="Mean Dynamic Tau", color="#448844", linestyle="--", alpha=0.7)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Hybrid Policy Signals")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# Reporting
# ============================================================

def print_report(
    fixed_total_mse: float,
    fixed_per_regime: Dict[str, float],
    nomadic_static_total_mse: float,
    nomadic_per_regime: Dict[str, float],
    nomadic_usage: Dict[str, np.ndarray],
    nomadic_mean_gate_distance: float,
    nomadic_pairwise_gate_distances: Dict[str, float],
    seq_total_mse: float,
    dynamics: dict,
):
    print("\n" + "=" * 72)
    print("FINAL REPORT")
    print("=" * 72)

    print("\n[Fixed Model]")
    print(f"Total Test MSE: {fixed_total_mse:.6f}")
    for k, v in fixed_per_regime.items():
        print(f"  Regime {k} MSE: {v:.6f}")

    print("\n[Nomadic Model | Static Eval]")
    print(f"Static Total Test MSE: {nomadic_static_total_mse:.6f}")
    for k, v in nomadic_per_regime.items():
        print(f"  Regime {k} MSE: {v:.6f}")

    print("\n[Nomadic Model | Sequence Eval]")
    print(f"Sequence Total Test MSE: {seq_total_mse:.6f}")

    print("\n[Nomadic Regime-wise Expert Usage | Top-1 Ratio]")
    for regime in ["A", "B", "C"]:
        arr = nomadic_usage[regime]
        arr_str = ", ".join([f"E{i}: {p:.3f}" for i, p in enumerate(arr)])
        print(f"  Regime {regime} -> {arr_str}")

    print("\n[Nomadic Mean Gate Distance | Static Full]")
    print(f"Mean pairwise gate-centroid distance: {nomadic_mean_gate_distance:.6f}")
    if len(nomadic_pairwise_gate_distances) > 0:
        print("[Pairwise Gate Distances]")
        for k, v in nomadic_pairwise_gate_distances.items():
            print(f"  {k}: {v:.6f}")

    print("\n[Transition Dynamics]")
    print(f"Regime -> Expert mapping: {dynamics['regime_to_expert']}")
    print(f"Mean switch latency: {dynamics['mean_switch_latency']:.4f}")
    print(f"Mean dwell time: {dynamics['mean_dwell_time']:.4f}")
    print(f"Mean dynamic tau: {dynamics['mean_dynamic_tau']:.4f}")
    print(f"Stable-phase mean entropy: {dynamics['stable_entropy_mean']:.4f}")
    print(f"Transition-phase mean entropy: {dynamics['transition_entropy_mean']:.4f}")

    print("\nInterpretation hint:")
    print("- Sequence Test MSE is the main performance metric in phase-transition settings.")
    print("- Static Test MSE is only a reference check, not the main success criterion.")
    print("- Transition entropy > stable entropy suggests gate uncertainty rises during phase shifts.")
    print("- Shorter switch latency suggests faster nomadic response.")
    print("- Moderate dwell time suggests neither rigid fixation nor chaotic wandering.")
    print("- Dynamic tau rises in stable phases (fixation allowed) and falls in volatile phases (nomadic pressure).")
    print("=" * 72 + "\n")


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

    if args.save_dir is not None:
        cfg.save_dir = args.save_dir

    if args.seed is not None:
        cfg.seed = args.seed

    if args.device is not None:
        cfg.device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    ensure_dir(cfg.save_dir)
    set_seed(cfg.seed)

    print(f"Using device: {cfg.device}")
    print(f"Saving outputs to: {cfg.save_dir}")
    print(f"Loaded config from: {args.config}")

    X_train, Y_train, R_train, phase_tags_train = generate_phase_sequence(cfg, cfg.phase_train_cycles, cfg.device)
    X_test, Y_test, R_test, phase_tags_test = generate_phase_sequence(cfg, cfg.phase_test_cycles, cfg.device)

    plot_dataset(X_train, R_train, os.path.join(cfg.save_dir, "phase_dataset_input_space.png"))

    fixed_model, fixed_logs = train_fixed(cfg, X_train, Y_train, R_train, X_test, Y_test, R_test)
    nomadic_model, nomadic_logs = train_nomadic(
        cfg, X_train, Y_train, R_train, X_test, Y_test, R_test, phase_tags_test
    )

    fixed_total_mse, fixed_per_regime = evaluate_fixed(fixed_model, X_test, Y_test, R_test)
    (
        nomadic_static_total_mse,
        nomadic_per_regime,
        nomadic_usage,
        nomadic_mean_gate_distance,
        nomadic_pairwise_gate_distances,
        _,
        _,
        _,
        _,
    ) = evaluate_nomadic_static_full(
        nomadic_model, X_test, Y_test, R_test, cfg
    )

    seq_total_mse, seq_usage, dynamics, _, _ = evaluate_nomadic_sequence_dynamics(
        nomadic_model, X_test, Y_test, R_test, phase_tags_test, cfg
    )

    plot_training_curves(
        fixed_logs,
        nomadic_logs,
        os.path.join(cfg.save_dir, "fixed_vs_nomadic_test_mse.png"),
    )
    plot_nomadic_losses(
        nomadic_logs,
        os.path.join(cfg.save_dir, "nomadic_loss_components.png"),
    )
    plot_delta_trace(
        nomadic_logs,
        os.path.join(cfg.save_dir, "hybrid_delta_trace.png"),
    )
    plot_usage_bars(
        nomadic_usage,
        os.path.join(cfg.save_dir, "regime_expert_usage_bars.png"),
        "Regime-wise Expert Usage (Top-1)",
    )
    plot_gate_heatmap(
        nomadic_usage,
        os.path.join(cfg.save_dir, "regime_expert_usage_heatmap.png"),
    )
    plot_gate_distance_curve(
        nomadic_logs,
        os.path.join(cfg.save_dir, "regime_mean_gate_distance.png"),
    )
    plot_phase_entropy(
        dynamics,
        os.path.join(cfg.save_dir, "phase_gate_entropy.png"),
    )
    plot_expert_trajectory(
        dynamics,
        os.path.join(cfg.save_dir, "expert_trajectory.png"),
    )
    plot_dwell_histogram(
        dynamics["dwell_times"],
        os.path.join(cfg.save_dir, "dwell_time_histogram.png"),
    )
    plot_switch_latency_histogram(
        dynamics["switch_latencies"],
        os.path.join(cfg.save_dir, "switch_latency_histogram.png"),
    )
    plot_entropy_comparison(
        nomadic_logs,
        os.path.join(cfg.save_dir, "stable_vs_transition_entropy.png"),
    )
    plot_switch_latency_curve(
        nomadic_logs,
        os.path.join(cfg.save_dir, "switch_latency_curve.png"),
    )
    plot_regime_expert_alignment(
        dynamics,
        os.path.join(cfg.save_dir, "regime_expert_alignment.png"),
    )
    plot_dynamic_tau_trace(
        nomadic_logs,
        os.path.join(cfg.save_dir, "dynamic_tau_trace.png"),
    )
    plot_policy_hybrid_signals(
        nomadic_logs,
        os.path.join(cfg.save_dir, "policy_hybrid_signals.png"),
    )

    print_report(
        fixed_total_mse,
        fixed_per_regime,
        nomadic_static_total_mse,
        nomadic_per_regime,
        nomadic_usage,
        nomadic_mean_gate_distance,
        nomadic_pairwise_gate_distances,
        seq_total_mse,
        dynamics,
    )

    print("Saved files:")
    for fname in [
        "phase_dataset_input_space.png",
        "fixed_vs_nomadic_test_mse.png",
        "nomadic_loss_components.png",
        "hybrid_delta_trace.png",
        "regime_expert_usage_bars.png",
        "regime_expert_usage_heatmap.png",
        "regime_mean_gate_distance.png",
        "phase_gate_entropy.png",
        "expert_trajectory.png",
        "dwell_time_histogram.png",
        "switch_latency_histogram.png",
        "stable_vs_transition_entropy.png",
        "switch_latency_curve.png",
        "regime_expert_alignment.png",
        "dynamic_tau_trace.png",
        "policy_hybrid_signals.png",
    ]:
        print(" -", os.path.join(cfg.save_dir, fname))


if __name__ == "__main__":
    main()