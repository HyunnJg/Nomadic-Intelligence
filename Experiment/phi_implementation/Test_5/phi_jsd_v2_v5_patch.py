    def compute(self, delta_env, delta_err, explanation_error, best_expert_gap,
                gate_probs, stay_switch_probs=None, cfg=None):
        device = gate_probs.device

        per_jsd = self._per_sample_jsd(gate_probs)  # [B], GPU tensor

        # ★ v5 fix: std_term / mean_term을 detach하여 gradient graph 차단
        # v4에서 std_term이 graph에 연결된 채로 phi_raw → tanh → loss backward 경로가
        # 열려 Φ 신호 자체가 training 중 왜곡됨. PhiEMA와 동일하게 detach 처리.
        std_term  = per_jsd.std().detach()   # scalar GPU tensor, no grad
        mean_term = per_jsd.mean().detach()  # scalar GPU tensor, no grad

        # EMA 업데이트 — detach된 값으로 누적 (v4와 동일)
        if self.ema_mean_jsd is None:
            self.ema_mean_jsd = mean_term
        else:
            self.ema_mean_jsd = (self.ema_decay * self.ema_mean_jsd
                                 + (1.0 - self.ema_decay) * mean_term)

        phi_raw = self.s_div * std_term + self.s_ema * self.ema_mean_jsd
        return torch.tanh(phi_raw)
