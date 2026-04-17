# §4.10 Recurrent Gate + Oracle Baseline — Complete Results

**실험 목적**: Nomadic Full의 성능이 "temporal structure가 있으면 충분"한 것인지,
아니면 explicit transition dynamics control이 필요한지를 검증.

**실험 버전**: v2 (GRU 과적합 방지 + Oracle 재설계)

| 수정 사항 | 내용 |
|---|---|
| GRU train shuffle | 매 cycle마다 regime 순서 랜덤 permutation (패턴 암기 방지) |
| GRU regularization | Dropout(p=0.15) + weight_decay × 5 |
| Oracle 재설계 | Nomadic expert 재사용 → regime-specialist MLP 독립 학습 + label-conditioned routing |

**Seeds**: 42, 123, 456 | **Epochs**: 220 | **GPU**: L4

---

## 최종 결과 (3-seed mean ± std)

| Model | Params | Seq MSE | MSE std | ΔH | ΔH std | Stable H | Trans H | Switch Lat |
|---|---|---|---|---|---|---|---|---|
| Standard MoE | 17,798 | 0.415 | 0.002 | 0.033 | 0.017 | 0.979 | 1.012 | — |
| GRU MoE | 26,502 | **0.105** | 0.039 | 0.354 | 0.034 | 0.365 | 0.719 | 2.287 |
| **Nomadic Full** | **23,053** | 0.215 | 0.028 | **0.755** | 0.100 | **0.176** | **0.931** | 1.232 |
| Oracle | 3×4,417 | 0.330 | 0.004 | — | — | — | — | — |

## Per-seed 상세

| Model | Seed | Seq MSE | Stable H | Trans H | ΔH | Switch Lat |
|---|---|---|---|---|---|---|
| StdMoE | 42 | 0.418 | 0.982 | 1.014 | 0.033 | — |
| StdMoE | 123 | 0.412 | 0.931 | 0.943 | 0.012 | — |
| StdMoE | 456 | 0.415 | 1.026 | 1.079 | 0.053 | — |
| GRU | 42 | 0.159 | 0.404 | 0.751 | 0.347 | 1.833 |
| GRU | 123 | 0.070 | 0.354 | 0.752 | 0.398 | 2.000 |
| GRU | 456 | 0.086 | 0.338 | 0.653 | 0.315 | 3.028 |
| Nomadic | 42 | 0.192 | 0.286 | 0.909 | 0.623 | 0.472 |
| Nomadic | 123 | 0.199 | 0.158 | 0.935 | 0.778 | 3.111 |
| Nomadic | 456 | 0.255 | 0.083 | 0.948 | 0.865 | 0.111 |
| Oracle | 42 | 0.329 | — | — | — | — |
| Oracle | 123 | 0.325 | — | — | — | — |
| Oracle | 456 | 0.335 | — | — | — | — |

## Oracle Expert 학습 결과

| Seed | Expert A (y=x₁+x₂) | Expert B (y=x₁-x₂) | Expert C (y=-x₁+0.5x₂) |
|---|---|---|---|
| 42 | 0.344 | 0.107 | 0.479 |
| 123 | 0.338 | 0.106 | 0.476 |
| 456 | 0.341 | 0.110 | 0.475 |
| **mean** | **0.341** | **0.108** | **0.477** |

---

## 핵심 발견 분석

### 1. GRU MSE < Nomadic MSE — 두 가지 이유의 복합

GRU(0.105)가 Nomadic(0.215)보다 낮은 MSE를 기록했다. 이는 두 가지 구분된 이유의 복합이다.

**첫째, GRU의 genuine temporal learning.** Shuffle + regularization 이후에도 GRU가
StdMoE(0.415) 대비 75% MSE 감소를 달성했다. ΔH=0.354(StdMoE 0.033 대비 10.7×)는
GRU가 실제로 temporal context를 활용하고 있음을 보여준다.

**둘째, GRU의 transition 배치 처리 우위.** 현재 태스크에서 transition 배치의 레이블은
두 regime 함수의 선형 보간(`y_mix = (1-α)·f_A + α·f_B`)이다. GRU hidden state는
모든 배치 타입을 학습에서 보기 때문에 interpolated target에도 적응한다. 반면 Oracle
expert는 stable 데이터만으로 학습되어 transition 배치에서 구조적으로 불리하다.
이 때문에 Oracle MSE(0.330)가 Nomadic(0.215)보다도 높게 나왔으며,
Oracle을 absolute upper bound로 해석할 수 없다.

### 2. ΔH에서 Nomadic의 명확한 우위 — 2.14×

| 지표 | GRU | Nomadic | 비율 |
|---|---|---|---|
| ΔH | 0.354 | **0.755** | **2.14×** |
| Stable H | 0.365 | **0.176** | 0.48× (낮을수록 좋음) |
| Trans H | 0.719 | **0.931** | 1.29× |

GRU의 Stable Entropy(0.365)는 Nomadic(0.176)의 두 배 이상이다. GRU는 temporal
context를 통해 partial entropy reduction을 달성하지만, Nomadic의 near-deterministic
stable-phase fixation(homeomorphic fixation)에는 도달하지 못한다.

이는 논문의 핵심 구분을 실증한다:
- **implicit temporal learning (GRU)**: MSE 개선 O, homeomorphic fixation X
- **explicit transition dynamics control (Nomadic)**: MSE 개선 O, homeomorphic fixation O

### 3. Switch Latency 역전 — Nomadic이 더 빠른 전환

| 모델 | Switch Lat (mean) |
|---|---|
| GRU | 2.287 |
| Nomadic | **1.232** |

Nomadic이 GRU 대비 46% 빠른 switch latency를 보인다. Δx 기반 명시적 전환 감지가
GRU의 hidden state 갱신보다 환경 변화에 더 빠르게 반응함을 의미한다.

### 4. Oracle 한계 — absolute upper bound 아님

Oracle의 설계 의도는 "label을 알고 있을 때의 이론적 상한"이었으나, regime-specialist
expert가 transition 배치(선형 보간 레이블)에 적응하지 못하는 구조적 한계로 인해
Oracle MSE(0.330)가 Nomadic(0.215)보다 높게 나왔다.

올바른 해석: Oracle은 stable 구간에서의 이론적 전문화 수준을 나타내며,
"transition을 포함한 전체 시퀀스에서의 최적 upper bound"가 아니다.
이 한계는 §5 Discussion에서 future work으로 언급한다
(transition-aware Oracle: α-weighted expert mixture로 transition 배치를 처리하는 버전).

---

## 이론적 함의

**MSE vs ΔH의 분리**: 이 실험은 MSE와 behavioral signature(ΔH)가 서로 다른 최적화
목표임을 명확히 보여준다. GRU는 MSE를 낮추는 데 효과적이지만 구조화된 전환 행동
(structured adaptive fixation)을 학습하지 못한다. Nomadic의 설계 목표는 MSE 최소화가
아니라 stable-phase fixation과 transition-phase exploration의 분리이며,
ΔH가 이를 측정하는 주된 지표다.

**Temporal structure의 충분조건 문제**: GRU 결과는 "temporal structure가 있으면 MSE
개선에 충분하다"는 주장을 지지하지만, "homeomorphic fixation을 달성하기에는 충분하지
않다"는 주장도 동시에 지지한다. 이 구분이 Nomadic의 contribution을 명확히 한다.

---

## 다음 단계

**완료**: §4.10 결과 → PAPER.md 반영 예정.

**잔여 실험**:
- §4.11 Real non-stationary time series (ETT / Weather dataset)
- §4.12 LLM experiment (ground-truth phase label 기반 rigorous ver.)

**Future work으로 남겨둘 실험**:
- Transition-aware Oracle (α-weighted expert mixture)
- GRU + explicit Δx 결합 버전 (hybrid baseline)
