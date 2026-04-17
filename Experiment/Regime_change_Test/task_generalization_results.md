# §4.9 Task Generalization — Complete Results

**실험 목적**: §4.1 결과가 선형/gradual/Gaussian 태스크 설계에 종속되지 않음을 검증.
3모델(Fixed, Standard MoE, Nomadic Full) × 3seeds(42, 123, 456) × 5variants.

**기준값 (§4.1 원본)**:

| Model | Seq MSE | ΔH | Stable Ent |
|---|---|---|---|
| Fixed | 0.409 | — | — |
| Standard MoE | 0.410 | 0.017 | 0.952 |
| Nomadic Full | 0.165 | 0.781 | 0.091 |

---

## 최종 결과 (3-seed mean ± std)

| Variant | Model | Seq MSE | Seq MSE std | ΔH | ΔH std | Stable Ent | Trans Ent | Switch Lat |
|---|---|---|---|---|---|---|---|---|
| nonlinear | Fixed | 0.221 | 0.006 | — | — | — | — | — |
| nonlinear | Standard MoE | 0.280 | 0.048 | 0.071 | 0.012 | 0.977 | 1.048 | — |
| nonlinear | **Nomadic Full** | **0.174** | 0.030 | **0.341** | 0.167 | 0.634 | 0.975 | 2.000 |
| abrupt | Fixed | 0.307 | 0.011 | — | — | — | — | — |
| abrupt | Standard MoE | 0.317 | 0.009 | 0.012 | 0.001 | 1.061 | 1.073 | — |
| abrupt | **Nomadic Full** | **0.160** | 0.079 | −0.009 | 0.036 | 0.985 | 0.976 | 0.361 |
| gradual | Fixed | 0.436 | 0.048 | — | — | — | — | — |
| gradual | Standard MoE | 0.416 | 0.008 | 0.074 | 0.004 | 0.984 | 1.059 | — |
| gradual | Nomadic Full | 0.424 | 0.152 | 0.111 | 0.019 | 0.775 | 0.886 | 1.639 |
| heavy_tail | Fixed | 5.657 | 2.270 | — | — | — | — | — |
| heavy_tail | Standard MoE | 6.160 | 2.152 | 0.068 | 0.033 | 0.443 | 0.511 | — |
| heavy_tail | **Nomadic Full** | **4.415** | 1.825 | 0.006 | 0.037 | 0.896 | 0.902 | 0.315 |
| combined | Fixed | 7,771 | 8,299 | — | — | — | — | — |
| combined | Standard MoE | 16,846 | 22,849 | 0.006 | 0.009 | 0.519 | 0.525 | — |
| combined | Nomadic Full | 17,769 | 23,870 | 0.025 | 0.038 | 0.401 | 0.426 | 0.694 |

---

## Per-seed 상세

### Nonlinear
| Model | Seed | Seq MSE | Stable Ent | Trans Ent | ΔH | Switch Lat |
|---|---|---|---|---|---|---|
| Fixed | 42 | 0.226 | — | — | — | — |
| Fixed | 123 | 0.225 | — | — | — | — |
| Fixed | 456 | 0.212 | — | — | — | — |
| StdMoE | 42 | 0.220 | 0.974 | 1.051 | 0.077 | — |
| StdMoE | 123 | 0.282 | 1.014 | 1.069 | 0.055 | — |
| StdMoE | 456 | 0.337 | 0.943 | 1.025 | 0.082 | — |
| Nomadic | 42 | 0.207 | 0.621 | 0.972 | 0.351 | 3.306 |
| Nomadic | 123 | 0.181 | 0.443 | 0.984 | 0.541 | 1.333 |
| Nomadic | 456 | 0.136 | 0.839 | 0.970 | 0.132 | 1.361 |

### Abrupt (steps=2)
| Model | Seed | Seq MSE | Stable Ent | Trans Ent | ΔH | Switch Lat |
|---|---|---|---|---|---|---|
| Fixed | 42 | 0.322 | — | — | — | — |
| Fixed | 123 | 0.297 | — | — | — | — |
| Fixed | 456 | 0.303 | — | — | — | — |
| StdMoE | 42 | 0.324 | 1.059 | 1.070 | 0.012 | — |
| StdMoE | 123 | 0.305 | 1.065 | 1.078 | 0.013 | — |
| StdMoE | 456 | 0.322 | 1.061 | 1.072 | 0.011 | — |
| Nomadic | 42 | 0.269 | 0.983 | 0.950 | −0.032 | 0.583 |
| Nomadic | 123 | 0.123 | 1.007 | 0.972 | −0.035 | 0.000 |
| Nomadic | 456 | 0.087 | 0.964 | 1.006 | 0.042 | 0.500 |

### Gradual (steps=24)
| Model | Seed | Seq MSE | Stable Ent | Trans Ent | ΔH | Switch Lat |
|---|---|---|---|---|---|---|
| Fixed | 42 | 0.406 | — | — | — | — |
| Fixed | 123 | 0.399 | — | — | — | — |
| Fixed | 456 | 0.503 | — | — | — | — |
| StdMoE | 42 | 0.426 | 0.980 | 1.059 | 0.079 | — |
| StdMoE | 123 | 0.406 | 0.989 | 1.059 | 0.069 | — |
| StdMoE | 456 | 0.417 | 0.984 | 1.058 | 0.074 | — |
| Nomadic | 42 | 0.575 | 0.798 | 0.934 | 0.135 | 2.778 |
| Nomadic | 123 | 0.217 | 0.775 | 0.865 | 0.090 | 0.361 |
| Nomadic | 456 | 0.482 | 0.753 | 0.860 | 0.107 | 1.778 |

### Heavy-tail (Student-T df=2)
| Model | Seed | Seq MSE | Stable Ent | Trans Ent | ΔH | Switch Lat |
|---|---|---|---|---|---|---|
| Fixed | 42 | 3.157 | — | — | — | — |
| Fixed | 123 | 5.163 | — | — | — | — |
| Fixed | 456 | 8.650 | — | — | — | — |
| StdMoE | 42 | 3.131 | 0.391 | 0.413 | 0.022 | — |
| StdMoE | 123 | 7.930 | 0.539 | 0.627 | 0.087 | — |
| StdMoE | 456 | 7.419 | 0.399 | 0.495 | 0.096 | — |
| Nomadic | 42 | 1.853 | 0.872 | 0.865 | −0.006 | 0.000 |
| Nomadic | 123 | 5.965 | 0.924 | 0.979 | 0.056 | 0.944 |
| Nomadic | 456 | 5.425 | 0.893 | 0.861 | −0.032 | 0.000 |

### Combined (Nonlinear + Abrupt + Heavy-tail)
| Model | Seed | Seq MSE | Stable Ent | Trans Ent | ΔH | Switch Lat |
|---|---|---|---|---|---|---|
| Fixed | 42 | 19,404 | — | — | — | — |
| Fixed | 123 | 3,305 | — | — | — | — |
| Fixed | 456 | 604 | — | — | — | — |
| StdMoE | 42 | 49,160 | 0.004 | 0.004 | −0.001 | — |
| StdMoE | 123 | 666 | 0.693 | 0.711 | 0.018 | — |
| StdMoE | 456 | 712 | 0.859 | 0.859 | 0.000 | — |
| Nomadic | 42 | 51,525 | 0.581 | 0.660 | 0.079 | 0.778 |
| Nomadic | 123 | 845 | 0.002 | 0.002 | −0.000 | 0.000 |
| Nomadic | 456 | 935 | 0.620 | 0.616 | −0.003 | 1.306 |

---

## 핵심 발견 분석

### 1. Nonlinear — 강건성 확인 (부분적)

Nomadic Full이 Seq MSE에서 StdMoE를 0.105 차이로 앞서고(0.174 vs 0.280),
ΔH도 0.341 vs 0.071로 4.8배 우위를 유지. §4.1 대비 ΔH가 0.781→0.341로 감소했지만,
Stable Entropy(0.634)가 여전히 StdMoE(0.977)보다 낮아 fixation 패턴은 보존됨.
**결론**: 비선형 함수 변형에서 MSE 우위와 방향성(ΔH>0)은 유지. 절대적 fixation 깊이는
감소하며, 이는 Expert(Tanh activation)의 비선형 근사 부담이 증가한 결과.

### 2. Abrupt (steps=2) — MSE 우위 유지, ΔH 붕괴

Nomadic이 Seq MSE에서 0.160으로 StdMoE(0.317) 대비 49% 개선 — §4.1 패턴 완전 보존.
그러나 ΔH = −0.009로 사실상 0이며, Stable Entropy(0.985) ≈ Transition Entropy(0.976).
이는 transition window가 steps=2로 극도로 좁아지면서 stable/transition 구간의
배치 수가 균형을 잃어 ΔH 측정 자체가 의미를 잃기 때문. 단, Nomadic이 MSE 목표를
달성하는 방식은 다름 — seed 123(0.123), seed 456(0.087)에서 현저한 low MSE를
달성하면서도 ΔH ≈ 0을 보임: 빠른 전환 환경에서는 fixation 없이 빠른 switching으로
적응하는 전략으로 수렴했을 가능성.
**결론**: Abrupt에서 MSE 개선은 보존되나, ΔH는 측정 구조상 유의미하지 않음.

### 3. Gradual (steps=24) — 전반적 성능 저하

모든 모델의 MSE가 §4.1 대비 증가. Nomadic(0.424)이 StdMoE(0.416)보다 약간 높고
seed 분산이 0.152로 매우 큼(seed 42: 0.575, seed 123: 0.217, seed 456: 0.482).
ΔH는 0.111로 StdMoE(0.074) 대비 1.5배이나 §4.1(0.781) 대비 대폭 감소.
원인 분석: transition_steps=24에서 전체 시퀀스 중 transition 구간 비중이
steps=8(23%)에서 steps=24(50%)로 증가. Δx_env 신호가 전체 시퀀스의 절반 이상에서
중간값(transition 상태)을 유지하여 stable과 transition의 구분이 약해짐. DwellReg가
항상 중간 국면으로 판단하게 되어 fixation이 어려워짐.
**결론**: Gradual은 현재 설계의 실제 한계 조건. §5.3 limitation으로 기술하는 것이 적합.

### 4. Heavy-tail (Student-T df=2) — MSE 우위, ΔH 붕괴

Nomadic이 MSE에서 4.415로 StdMoE(6.160), Fixed(5.657) 모두를 앞섬. 절대 MSE가
크게 높아진 것은 Student-T의 극단값이 loss를 직접 올리기 때문 — 상대적 순위는
Nomadic > Fixed > StdMoE. 그러나 ΔH = 0.006으로 사실상 0. Stable Entropy(0.896)와
Transition Entropy(0.902)가 거의 동일. 원인: 극단값 샘플이 배치마다 임의로 포함되어
Δx_err(error EMA-baseline)가 일시적 급등을 regime 전환으로 오인하는 노이즈를 도입.
Φ 신호가 노이즈에 반응하여 constantly mid-range로 유지됨으로써 stable/transition 분리
실패. MSE는 낮추지만 fixation 패턴은 상실.
**결론**: Heavy-tail에서 절대 성능 우위는 보존. ΔH 붕괴는 Δx_err의 노이즈 민감도를
드러내며 §5.3 limitation 확장 및 노이즈-robust Δx 설계의 future work으로 연결.

### 5. Combined — 전면적 붕괴

비선형 + abrupt + heavy-tail 동시 적용 시 모든 모델의 MSE가 수천~수만 단위로 폭증,
seed간 분산이 극대화(Fixed: 8299, StdMoE: 22849, Nomadic: 23870).
Nomadic이 StdMoE보다 오히려 높은 MSE를 보이는 유일한 케이스.
상세 분석: Seed 42에서 StdMoE(49,160) ≈ Nomadic(51,525)이나 Seed 123/456에서는
StdMoE(666/712) vs Nomadic(845/935)으로 Nomadic이 약간 더 나쁨.
핵심 원인: 비선형 함수(큰 y 스케일) + Student-T 극단값의 결합이 gradient를 불안정하게
만들어 PolicyNet teacher signal이 의미를 잃음. L_policy loss가 오히려 학습을 방해.
**결론**: Combined는 현재 아키텍처의 안정성 한계를 드러내며, gradient clipping/
output normalization 또는 robust loss function 도입의 필요성을 시사. 논문에서는
§5.3 limitation으로 기술하고, 향후 실험의 동기로 활용.

---

## 이론적 함의 요약

| Variant | MSE 우위 | ΔH 보존 | 해석 |
|---|---|---|---|
| nonlinear | ✅ Nomadic +0.105 | △ 0.341 (감소) | 비선형 함수에서도 핵심 메커니즘 작동 |
| abrupt | ✅ Nomadic +0.157 | ✗ ≈0 (구조적 이유) | MSE 우위 보존; ΔH는 측정 한계 |
| gradual | ✗ ≈동등 (+0.008) | △ 0.111 (감소) | 긴 transition이 stable/transition 구분 약화 |
| heavy_tail | ✅ Nomadic +1.745 | ✗ ≈0 | Δx_err의 노이즈 민감도 노출 |
| combined | ✗ Nomadic −923 | ✗ ≈0 | 복합 스트레스에서 안정성 한계 |

**전체 결론**: Nomadic Intelligence의 MSE 우위는 비선형(nonlinear), 급전환(abrupt),
heavy-tail noise 단일 조건에서 유지된다. ΔH(entropy differentiation) 시그니처는
비선형 조건에서만 부분적으로 보존되며, abrupt/heavy-tail/gradual 조건에서 약화된다.
이는 현재 Δx 신호 설계가 transition window 폭과 noise 스케일에 대한 robustness
개선이 필요함을 시사하며, §4.9의 결과를 future work의 구체적 방향으로 연결한다.

---

## 다음 단계

**완료**: §4.9 Task Generalization → PAPER.md 반영 예정.

**진행 예정**:
- §4.10 Recurrent Gate + Oracle baseline (Nomadic Full vs GRU-gate vs Oracle)
- §4.11 Real non-stationary time series (ETT / Weather dataset)
- §4.12 LLM experiment (rigorous, ground-truth phase label 기반)

**설계 개선 방향 (future work)**:
- Gradual 조건 대응: transition ratio에 따른 Φ 스케일 자동 조정
- Heavy-tail 대응: Δx_err에 robust estimator(Huber loss 기반 EMA) 도입
- Combined 대응: output normalization + gradient clipping + policy loss weighting 재조정
