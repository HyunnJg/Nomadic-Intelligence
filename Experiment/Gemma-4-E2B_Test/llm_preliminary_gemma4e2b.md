# LLM Preliminary Experiment — Gemma-4-E2B (결과20)

Nomadic Intelligence의 신호 계층(Δx, τ_dynamic, PolicyNet)을 실제 LLM에 이식 가능한지 확인하는 **개념 증명 실험**.  
합성 MoE 태스크를 넘어 실제 언어 모델에서도 핵심 메커니즘이 작동하는지 검증한다.

> **하드웨어**: Google Colab T4 (15GB VRAM)  
> **모델**: Gemma-4-E2B (4-bit NF4 양자화)  
> **상태**: 예비 실험 (preliminary). PolicyNet은 untrained 상태.

---

## 실험 단계

### Phase 1 — Δx 신호 추출 확인

LLM의 마지막 레이어 hidden state를 `current_x`로 사용.  
HybridDeltaTracker를 그대로 이식하여 토큰 생성 중 Δx 신호를 실시간 측정.

```
프롬프트: "유목지능의 핵심 철학에 대해 설명해줘."

[Nomadic Signal Detected]
Δx_env:    0.0000  (첫 스텝 — 이전 상태 없음, 정상)
Δx_err:    0.0010
Δx_hybrid: 0.0010
```

**해석:** 단일 추론에서는 이전 상태가 없어 Δx_env=0. 정상 동작.

---

### Phase 2 — 동적 온도 제어 루프

Δx_hybrid를 실시간으로 계산해 라우팅 온도를 제어.  
토큰 생성마다 모델 불확실성(1 − top1_prob)을 Δx_err 입력으로 사용.

```
프롬프트: "미래의 인공지능은 생물학적 한계를"

Step 01 | '뛰어'   | 에너지(Δx): 0.0338 | 온도: 0.1372
Step 02 | '넘'     | 에너지(Δx): 0.3773 | 온도: 0.5151  ← 급등
Step 03 | '을'     | 에너지(Δx): 0.4694 | 온도: 0.6164
Step 04 | '수'     | 에너지(Δx): 0.4314 | 온도: 0.5745
Step 07 | '?'      | 에너지(Δx): 0.4416 | 온도: 0.5858
Step 08 | '\n'     | 에너지(Δx): 0.2084 | 온도: 0.3293  ← 하강
...
```

**생성 결과:**
> 미래의 인공지능은 생물학적 한계를 뛰어넘을 수 있을까?  
> 인간의 뇌를 모방한 인공지능 로봇이 인간을 능가할 수

**핵심 관찰:**
- Step 01(0.034) → Step 02(0.377): "뛰어"→"넘" 전환에서 Δx 급등. hidden state가 크게 변화하는 시점을 정확히 포착.
- 줄바꿈('\n') 직전 Step 08에서 0.208로 하강 — 문장 구조가 안정되는 구간에서 에너지가 낮아지는 패턴.
- **Δx가 LLM의 내부 표현 변화를 에너지로 읽는 핵심 아이디어가 작동함을 확인.**

---

### Phase 3 — PolicyNet + Dynamic τ 연동

AdvancedNomadicTracker(σ², τ_dynamic 포함)와 LightweightPolicyNet을 Gemma에 연결.  
PolicyNet 입력: `[h_proj(64), d_hybrid, d_err, d_hybrid, σ²_scaled, τ_scaled]`

```
프롬프트: "기술의 발전은 궁극적으로 인류에게"

Step 01 | '이'    | Switch 압력: 0.53 | Hard 모드: 0.43 | τ_dyn: 8.00
Step 02 | '점을'  | Switch 압력: 0.52 | Hard 모드: 0.47 | τ_dyn: 6.57
Step 06 | '지만'  | Switch 압력: 0.51 | Hard 모드: 0.46 | τ_dyn: 7.11
Step 16 | '르게'  | Switch 압력: 0.57 | Hard 모드: 0.42 | τ_dyn: 7.80
Step 21 | '.'     | Switch 압력: 0.51 | Hard 모드: 0.45 | τ_dyn: 7.83
Step 30 | 'AI'    | Switch 압력: 0.57 | Hard 모드: 0.40 | τ_dyn: 7.35
```

**생성 결과:**
> 기술의 발전은 궁극적으로 인류에게 이점을 가져다주지만, 그 이점은 모든 이에게 고르게 돌아가는 것은 아니다.  
> 최근 여러 연구에 따르면, AI

**핵심 관찰:**
- τ_dynamic이 Step 01(8.00) → Step 02(6.57) → Step 15+ (7.4~7.8) 로 수렴. σ²_Δ가 안정화될수록 τ가 τ_max 방향으로 올라가는 설계대로 작동.
- PolicyNet Switch 압력이 0.49~0.58로 0.5 근방에 수렴 — **untrained 상태의 랜덤 가중치 결과. 학습이 필요한 구간.**
- PolicyNet이 Gemma hidden dim(1536)을 오류 없이 처리함을 확인.

---

## 무엇이 작동했고 무엇이 아직 미완성인가

| 컴포넌트 | 상태 | 비고 |
|---------|------|------|
| HybridDeltaTracker (Δx_env, Δx_err, Δx_hybrid) | ✅ 작동 | LLM hidden state에서 의미 있는 신호 추출 |
| Dynamic τ_k (σ² 기반) | ✅ 작동 | 토큰 생성 안정화에 따라 τ가 반응 |
| 적응적 온도 제어 | ✅ 작동 | Δx → 온도 조절 → 생성 다양성 제어 |
| PolicyNet (구조) | ✅ 작동 | 1536차원 입력 처리, 오류 없음 |
| PolicyNet (학습) | ❌ 미완성 | 랜덤 가중치 상태, 의미 있는 결정 아직 없음 |
| Expert Switching | ❌ 미구현 | 현재는 단일 LLM + 온도 조절. LoRA adapter 전환이 다음 단계 |

---

## 다음 단계 (Colab 완성 목표)

**Step 1 — LoRA Expert 구현**  
각 레짐(안정/전환/탐색)에 대응하는 LoRA adapter 3개를 준비.  
Δx 신호로 어느 adapter를 활성화할지 라우팅.

**Step 2 — PolicyNet 학습**  
Δx가 높은 구간(전환기)에서는 switch=1, 낮은 구간(안정기)에서는 stay=1 이 되도록  
heuristic teacher signal로 supervised fine-tuning.

**Step 3 — Entropy Differentiation 측정**  
원 논문의 핵심 지표인 Stable Entropy vs Transition Entropy를  
LLM 토큰 생성 맥락에서 측정. 원 논문 결과(ΔH=+0.788)와 비교.

**Step 4 — 태스크 기반 평가**  
레짐 전환이 명확한 태스크(예: 문체 전환, 도메인 전환 QA)에서  
Nomadic routing과 standard 디코딩의 성능 비교.

---

## 논문 기여 가능성

이 실험들이 완성되면 논문에 추가될 수 있는 내용:

- **§4.5 LLM Applicability**: 합성 MoE → 실제 LLM으로의 신호 이식 가능성 실증
- **Abstract 업데이트**: "합성 환경에서 검증됨"에서 "LLM 예비 실험에서도 핵심 신호 작동 확인"으로
- Limitations §5.3의 "합성 환경만 평가" 항목 완화

---

## 하드웨어 메모

- GTX 1660 Super (6GB): 이 실험 불가 (Gemma-4-E2B 4bit만 해도 ~6GB 필요)
- **Colab T4 (15GB)**: Phase 1~3 모두 가능. 권장.
- Colab A100 (40GB): LoRA expert switching 실험 가능. 다음 단계 권장.
