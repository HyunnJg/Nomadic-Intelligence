# LLM Phase 4 Results — PolicyNet Training + LoRA Expert Switching

**Run ID**: 20260412_122334  
**Model**: Gemma-4-E2B (4-bit NF4, Colab T4)

---

## 전체 결과 요약

| 단계 | 핵심 지표 | 결과 |
|------|---------|------|
| Phase 1 | Δx 신호 추출 | ✅ 작동 |
| Phase 2 | 동적 온도 제어 | ✅ 작동 |
| Phase 3 | PolicyNet (untrained) | ΔH +0.731 |
| **Phase 4a** | **PolicyNet 학습 후** | **ΔH +0.984** |
| **Phase 4b** | **LoRA Expert Switching** | **54.2% switch rate, 3 experts 모두 활성** |

---

## PolicyNet 학습 결과

### 학습 전 → 후 비교

| 지표 | 학습 전 (untrained) | 학습 후 | 변화 |
|------|:------------------:|:-------:|:----:|
| Stable H | 1.806 | **1.249** | −0.557 ↓ |
| Trans H | 2.537 | **2.234** | −0.303 ↓ |
| **ΔH** | **+0.731** | **+0.984** | +0.253 ↑ |
| Stable switch prob | ~0.55 | 0.9999 | — |
| Trans switch prob | ~0.55 | 0.9999 | — |

**핵심 관찰:**
- ΔH가 +0.731 → +0.984로 증가. 원 논문 합성 실험(+0.788)을 초과.
- Stable Entropy 감소폭(−0.557)이 Trans Entropy 감소폭(−0.303)보다 큼 → 안정 맥락에서 더 결정론적이 된다는 방향이 맞음.

**한계 — switch_prob 포화:**
- stable/transition 맥락 모두 switch_prob ≈ 1.0000 으로 포화.
- PolicyNet이 "항상 switch"를 학습한 상태. stay vs switch 구분이 아직 미완성.
- 원인: 학습 데이터 클래스 불균형 또는 학습률 과다 가능성.
- 개선 방향: stable 프롬프트 비중 증가, class_weight 적용, lr 감소.

### 학습 곡선

- Epoch 0: loss 0.692
- Epoch 1: 0.636 (급락)
- Epoch 2~30: 0.632~0.635 (수렴)

Epoch 1에서 급락 후 수렴. 수렴값 0.63은 이진 분류 random baseline(0.693)보다 낮으나,  
포화된 switch_prob를 감안하면 실질적인 stay/switch 구분 학습은 부분적.

---

## LoRA Expert Switching 결과

### Expert 구성

| Expert | 역할 | LoRA r | 학습 데이터 특성 |
|--------|------|:------:|--------------|
| stable | 안정적, 사실적 생성 | 4 | 수학, 사실, 정의 |
| transition | 전환, 유연한 생성 | 4 | 대비, 역설, 변화 |
| creative | 창의적, 탐색적 생성 | 4 | 가정, 상상, 개방형 |

### 라우팅 규칙

```
Δx_hybrid < 0.20  AND switch_prob < 0.5  → stable
Δx_hybrid ≥ 0.45  OR  switch_prob ≥ 0.6  → creative
그 외                                     → transition
```

### 스위칭 통계 (3개 프롬프트, 120 스텝)

| 지표 | 값 |
|------|:--:|
| 총 스텝 | 120 |
| 총 전환 수 | 65 |
| 전환율 | **54.2%** |
| stable 사용 | 26.7% |
| transition 사용 | 37.5% |
| creative 사용 | 35.8% |

세 expert 모두 활성화됨. transition과 creative가 합계 73.3%로 다수 — Δx_hybrid가 대부분의 스텝에서 0.2 이상을 유지하는 LLM 생성의 특성 반영.

### 그래프 해석 (lora_expert_switching.png)

- **상단 막대그래프**: Δx_hybrid 값, 색깔 = 활성 expert (녹색=stable, 주황=transition, 빨강=creative)
- **하단 점 그래프**: 스텝별 active expert
- **파란 수직선**: expert 전환이 발생한 시점

Step 1(stable, Δx=0.034) → Step 2(creative, Δx=0.474) 전환이 가장 뚜렷한 패턴.  
Δx가 creative threshold(0.45) 이상으로 올라갈 때마다 creative expert가 활성화되고,  
Δx가 낮아지면 stable 또는 transition으로 복귀.

---

## 원 논문 합성 실험과의 비교

| 지표 | 합성 MoE (원 논문) | LLM Phase 3 (untrained) | LLM Phase 4 (trained) |
|------|:-----------------:|:----------------------:|:--------------------:|
| Stable H | 0.108 | 1.806 | **1.249** |
| Trans H | 0.896 | 2.537 | **2.234** |
| ΔH | +0.788 | +0.731 | **+0.984** |
| Expert switching | ✅ | ❌ | **✅** |
| PolicyNet 학습 | ✅ | ❌ | **⚠️ 부분** |

---

## 논문 §4.5 업데이트 요약

이 결과로 추가 가능한 내용:

1. **ΔH 개선**: untrained +0.731 → trained +0.984. 원 논문 +0.788 초과.
2. **LoRA Expert Switching 작동 확인**: 3개 expert가 Δx 신호 기반으로 실시간 전환.
3. **한계 명시**: PolicyNet switch_prob 포화 문제 — 추가 학습 필요.

---

## 다음 단계

| 항목 | 우선순위 | 예상 효과 |
|------|---------|---------|
| PolicyNet 재학습 (class_weight + lr 조정) | 높음 | switch_prob 포화 해소 |
| LoRA r=8로 증가 (A100) | 중간 | Expert 표현력 향상 |
| 정량적 태스크 평가 | 높음 | 논문 Table 4 완성 |
| Baseline benchmark | 높음 | standard 디코딩 대비 비교 |
