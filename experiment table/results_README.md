# Results

실험 전 과정의 원본 데이터 및 분석 테이블.

---

## 파일 구조

| 파일 | 내용 |
|------|------|
| `experiment_progression.md` | 결과1~17 전체 진행 표 — 단계별 패치와 지표 변화 |
| `ablation_study.md` | 결과18 — 4개 모델군 × 3 seed 비교 (최종 ablation) |
| `beta_phi_sweep_gatenet.md` | β_φ sweep (GateNet 단계, 결과8~10) |
| `beta_phi_sweep_dynamic_tau.md` | β_φ sweep (Dynamic τ 단계, 결과14~16) |
| `raw_logs/` | 실험 원본 출력 로그 전체 |

---

## 실험 흐름 요약

```
결과1  → 최초 구현
결과2~4 → Load Balancing 도입 및 λ_load 튜닝
결과5  → τₖ Lower Bound 구현
결과6~7 → Φ (Will to Resonance) 도입
결과8~10 → GateNet 구조 + β_φ sweep
결과11~13 → PolicyNet 도입 + β_φ 재탐색
결과14~16 → Dynamic τₖ + β_φ sweep
결과17 → Full Hybrid (Dynamic τₖ + PolicyNet 통합)
결과18 → 4-model Ablation Study (최종 검증)
```

---

## 핵심 수치 (최종)

| 모델 | Seq MSE (avg) | Stable H (avg) | ΔH (avg) |
|------|:---:|:---:|:---:|
| Fixed | ~0.412 | — | — |
| Standard MoE | 0.410 | 0.951 | +0.033 |
| Nomadic NoPolicy | 0.255 | 0.556 | +0.394 |
| **Nomadic Full** | **0.162** | **0.108** | **+0.788** |

모든 원본 로그는 `raw_logs/` 폴더에서 확인 가능.
