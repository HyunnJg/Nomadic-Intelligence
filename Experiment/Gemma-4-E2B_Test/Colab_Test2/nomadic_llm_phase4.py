# ============================================================
# STEP 0: 드라이브 마운트 + 경로 설정
# 세션 끊겨도 결과 보존
# ============================================================
from google.colab import drive
import os, json, time
drive.mount('/content/drive', force_remount=True)

DRIVE_BASE = '/content/drive/MyDrive/nomadic_llm_results'
MODEL_PATH = '/content/drive/MyDrive/gemma-4-E2B'
RUN_ID = time.strftime('%Y%m%d_%H%M%S')
RUN_DIR = os.path.join(DRIVE_BASE, f'run_phase4_{RUN_ID}')
os.makedirs(RUN_DIR, exist_ok=True)

print(f'✅ 드라이브 마운트 완료')
print(f'📁 저장 경로: {RUN_DIR}')


# ============================================================
# STEP 1: 패키지 설치
# ============================================================
# !pip install -q transformers accelerate bitsandbytes peft


# ============================================================
# STEP 2: 임포트
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim import AdamW

print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')


# ============================================================
# STEP 3: 모델 로드
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

print('Gemma 로드 중...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True
)
base_model.eval()

dummy = tokenizer('test', return_tensors='pt').input_ids.to(base_model.device)
with torch.no_grad():
    out = base_model(dummy, output_hidden_states=True)
HIDDEN_DIM = out.hidden_states[-1].shape[-1]
print(f'✅ 모델 로드 완료 | hidden_dim={HIDDEN_DIM}')


# ============================================================
# STEP 4: Nomadic 컴포넌트 정의 (이전과 동일)
# ============================================================
class HybridDeltaTracker:
    def __init__(self, alpha=0.8, beta=0.85,
                 tau_min=2.0, tau_max=8.0,
                 tau_var_scale=6.0, tau_window=8):
        self.alpha = alpha; self.beta = beta
        self.tau_min = tau_min; self.tau_max = tau_max
        self.tau_var_scale = tau_var_scale; self.tau_window = tau_window
        self.reset()

    def reset(self):
        self.prev_x = None
        self.ema_err = 0.0; self.baseline_err = 0.0
        self.recent_delta_env = deque(maxlen=self.tau_window)
        self.history = []

    def compute(self, current_x, current_err):
        if self.prev_x is None:
            delta_env = 0.0
        else:
            cos_sim = F.cosine_similarity(
                current_x.float().flatten(),
                self.prev_x.float().flatten(), dim=0)
            delta_env = float(1.0 - cos_sim.item())

        self.prev_x = current_x.detach().clone()
        self.recent_delta_env.append(delta_env)

        self.ema_err = self.alpha * self.ema_err + (1-self.alpha) * current_err
        self.baseline_err = self.beta * self.baseline_err + (1-self.beta) * current_err
        delta_err = max(0.0, self.ema_err - self.baseline_err)

        delta_hybrid = float(torch.tanh(torch.tensor(delta_env + delta_err)).item())
        sigma2 = float(np.var(self.recent_delta_env)) if len(self.recent_delta_env) >= 2 else 0.0
        tau_dyn = self.tau_min + (self.tau_max - self.tau_min) / (1.0 + self.tau_var_scale * sigma2)

        rec = dict(delta_env=delta_env, delta_err=delta_err,
                   delta_hybrid=delta_hybrid, sigma2=sigma2, tau_dyn=tau_dyn)
        self.history.append(rec)
        return rec


class NomadicPolicyNet(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.stay_switch_head = nn.Linear(hidden_dim, 2)
        self.mode_head = nn.Linear(hidden_dim, 2)

    def forward(self, hidden_state, meta_signals):
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(1)
        h = F.relu(self.proj(hidden_state.float()))
        inp = torch.cat([h, meta_signals.float()], dim=-1)
        h = self.shared(inp)
        return (F.softmax(self.stay_switch_head(h), dim=-1),
                F.softmax(self.mode_head(h), dim=-1))


def build_meta_signals(rec, device):
    sigma2_scaled = float(np.tanh(rec['sigma2'] * 10.0))
    tau_scaled = float(np.tanh((rec['tau_dyn'] - 5.0) / 5.0))
    return torch.tensor([[
        rec['delta_hybrid'], rec['delta_err'], rec['delta_hybrid'],
        sigma2_scaled, tau_scaled
    ]], dtype=torch.float32, device=device)

print('✅ Nomadic 컴포넌트 정의 완료')


# ============================================================
# STEP 5: PolicyNet 학습 데이터 생성
#
# 핵심 아이디어:
#   안정 맥락 프롬프트 → switch=0 (stay), hard=1 (집중)
#   전환 맥락 프롬프트 → switch=1 (switch), hard=0 (탐색)
#
# 실제 Δx 신호를 측정한 뒤 threshold로 레이블 결정
# ============================================================

# --- 학습용 프롬프트 ---
STABLE_PROMPTS = [
    # 수학/사실: 예측 가능하고 확실한 맥락
    "수학의 기본 원리는 공리계로부터 출발한다.",
    "The capital of France is Paris. The capital of Germany is",
    "2 더하기 2는 4이다. 3 더하기 3은",
    "물은 100도에서 끓는다. 얼음은 0도에서",
    "피타고라스 정리에 따르면 직각삼각형에서",
    "The Earth revolves around the Sun. The Moon revolves around",
    "사과는 과일이다. 바나나는",
    "Python is a programming language. Java is",
    "1 곱하기 1은 1이다. 2 곱하기 2는",
    "The sky is blue. The grass is",
]

TRANSITION_PROMPTS = [
    # 전환/역설/대비: 예측하기 어렵고 맥락이 급변하는 상황
    "처음에는 안정적이었지만 갑자기 모든 것이 바뀌었다. 그 순간",
    "Although science seemed to have all the answers, suddenly",
    "과거에는 옳았던 것이 이제는 틀렸다. 왜냐하면",
    "그는 평생 믿어온 신념을 버리기로 결심했다. 그 이유는",
    "The situation changed dramatically when",
    "예상과 달리 결과는 완전히 반대였다. 구체적으로",
    "Once upon a time everything was clear, but then unexpectedly",
    "규칙이 바뀌었다. 이제부터는",
    "What seemed impossible became reality when",
    "기술의 발전이 모든 것을 바꿔놓았다. 특히",
]

def collect_policy_training_data(model, tokenizer, tracker,
                                 stable_prompts, transition_prompts,
                                 steps_per_prompt=15,
                                 switch_threshold=0.25):
    """
    각 프롬프트에서 실제 Δx를 측정하고
    heuristic teacher signal로 레이블을 생성한다.

    switch_threshold: Δx_hybrid < 이 값이면 stable(stay), 이상이면 transition(switch)
    """
    data = []  # (hidden, meta_signals, switch_label, mode_label)

    def run_prompt(prompt, expected_switch):
        tracker.reset()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)

        for step in range(steps_per_prompt):
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            logits = out.logits[:, -1, :]
            hidden = out.hidden_states[-1][:, -1, :]
            probs = F.softmax(logits, dim=-1)
            err = 1.0 - probs.max().item()

            rec = tracker.compute(hidden, err)

            # 실제 Δx 기반으로 레이블 결정
            # expected_switch는 프롬프트 종류에서 오고
            # 실제 Δx로 미세 조정
            if expected_switch:
                # 전환 맥락: Δx가 낮으면 여전히 switch, 높으면 확실히 switch
                switch_label = 1
                mode_label = 0  # soft (탐색)
            else:
                # 안정 맥락: Δx가 높으면 switch, 낮으면 stay
                if rec['delta_hybrid'] > switch_threshold:
                    switch_label = 1
                    mode_label = 0
                else:
                    switch_label = 0
                    mode_label = 1  # hard (집중)

            meta = build_meta_signals(rec, model.device)
            data.append({
                'hidden': hidden.detach().cpu(),
                'meta': meta.detach().cpu(),
                'switch_label': switch_label,
                'mode_label': mode_label,
                'delta_hybrid': rec['delta_hybrid'],
            })

            # 다음 토큰 (greedy)
            next_token = probs.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    print('학습 데이터 수집 중...')
    for i, prompt in enumerate(stable_prompts):
        run_prompt(prompt, expected_switch=False)
        print(f'  안정 프롬프트 {i+1}/{len(stable_prompts)} 완료', end='\r')

    print()
    for i, prompt in enumerate(transition_prompts):
        run_prompt(prompt, expected_switch=True)
        print(f'  전환 프롬프트 {i+1}/{len(transition_prompts)} 완료', end='\r')

    print(f'\n✅ 총 {len(data)}개 샘플 수집')
    switch_count = sum(d['switch_label'] for d in data)
    print(f'   switch=1: {switch_count} | stay=0: {len(data)-switch_count}')
    return data

tracker_train = HybridDeltaTracker()
train_data = collect_policy_training_data(
    base_model, tokenizer, tracker_train,
    STABLE_PROMPTS, TRANSITION_PROMPTS,
    steps_per_prompt=15
)


# ============================================================
# STEP 6: PolicyNet 학습
# ============================================================
policy_net = NomadicPolicyNet(input_dim=HIDDEN_DIM, hidden_dim=64)
policy_net = policy_net.to(base_model.device)

optimizer = AdamW(policy_net.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 30
BATCH_SIZE = 16
print(f'PolicyNet 학습 시작 | epochs={EPOCHS} | batch={BATCH_SIZE}')
print(f'파라미터 수: {sum(p.numel() for p in policy_net.parameters()):,}')

loss_history = []

for epoch in range(EPOCHS):
    # 셔플
    indices = torch.randperm(len(train_data))
    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, len(train_data), BATCH_SIZE):
        batch_idx = indices[start:start+BATCH_SIZE]
        batch = [train_data[i] for i in batch_idx]

        hidden_batch = torch.cat([b['hidden'] for b in batch], dim=0).to(base_model.device)
        meta_batch   = torch.cat([b['meta'] for b in batch], dim=0).to(base_model.device)
        switch_labels = torch.tensor([b['switch_label'] for b in batch],
                                     dtype=torch.long, device=base_model.device)
        mode_labels   = torch.tensor([b['mode_label'] for b in batch],
                                     dtype=torch.long, device=base_model.device)

        optimizer.zero_grad()
        stay_switch_probs, mode_probs = policy_net(hidden_batch, meta_batch)

        # loss = stay/switch 분류 + mode 분류
        loss_ss   = criterion(stay_switch_probs, switch_labels)
        loss_mode = criterion(mode_probs, mode_labels)
        loss = loss_ss + 0.5 * loss_mode

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        # 정확도 계산
        with torch.no_grad():
            all_hidden = torch.cat([d['hidden'] for d in train_data]).to(base_model.device)
            all_meta   = torch.cat([d['meta'] for d in train_data]).to(base_model.device)
            all_switch = torch.tensor([d['switch_label'] for d in train_data],
                                      device=base_model.device)
            ss_probs, _ = policy_net(all_hidden, all_meta)
            pred = ss_probs.argmax(dim=-1)
            acc = (pred == all_switch).float().mean().item()
        print(f'Epoch {epoch+1:02d}/{EPOCHS} | loss={avg_loss:.4f} | switch acc={acc:.3f}')

# 학습 곡선 저장
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(loss_history, 'b-o', ms=3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('PolicyNet Training Loss')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(RUN_DIR, 'policy_training_loss.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f'💾 학습 곡선 저장: {fig_path}')

# PolicyNet 가중치 저장
policy_path = os.path.join(RUN_DIR, 'policy_net.pt')
torch.save(policy_net.state_dict(), policy_path)
print(f'💾 PolicyNet 가중치 저장: {policy_path}')


# ============================================================
# STEP 7: 학습된 PolicyNet으로 엔트로피 재측정
# 학습 전 결과와 비교
# ============================================================
def measure_entropy_with_trained_policy(prompts_stable, prompts_transition,
                                        max_steps=20):
    def get_entropy_trace(prompt, steps):
        tracker = HybridDeltaTracker()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)
        entropies = []
        switch_probs = []

        for _ in range(steps):
            with torch.no_grad():
                out = base_model(input_ids, output_hidden_states=True)
            logits = out.logits[:, -1, :]
            hidden = out.hidden_states[-1][:, -1, :]
            probs = F.softmax(logits, dim=-1)
            err = 1.0 - probs.max().item()

            rec = tracker.compute(hidden, err)
            meta = build_meta_signals(rec, base_model.device)

            with torch.no_grad():
                ss, _ = policy_net(hidden, meta)
            switch_prob = ss[0, 1].item()

            topk_probs = probs.topk(50).values
            topk_probs = topk_probs / topk_probs.sum()
            H = -(topk_probs * topk_probs.log()).sum().item()

            entropies.append(H)
            switch_probs.append(switch_prob)

            next_token = probs.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        return entropies, switch_probs

    stable_H, trans_H = [], []
    stable_sw, trans_sw = [], []

    for p in prompts_stable:
        h, sw = get_entropy_trace(p, max_steps)
        stable_H.extend(h); stable_sw.extend(sw)
    for p in prompts_transition:
        h, sw = get_entropy_trace(p, max_steps)
        trans_H.extend(h); trans_sw.extend(sw)

    return {
        'stable_H': np.mean(stable_H),
        'trans_H': np.mean(trans_H),
        'delta_H': np.mean(trans_H) - np.mean(stable_H),
        'stable_switch_prob': np.mean(stable_sw),
        'trans_switch_prob': np.mean(trans_sw),
    }

print('\n학습된 PolicyNet으로 엔트로피 재측정...')
result_trained = measure_entropy_with_trained_policy(
    STABLE_PROMPTS, TRANSITION_PROMPTS
)

print('\n=== 결과 비교 ===')
print(f'{"":20} {"학습 전":>12} {"학습 후":>12}')
print(f'{"Stable H":20} {"1.806":>12} {result_trained["stable_H"]:>12.4f}')
print(f'{"Trans H":20} {"2.537":>12} {result_trained["trans_H"]:>12.4f}')
print(f'{"ΔH":20} {"+0.731":>12} {result_trained["delta_H"]:>+12.4f}')
print(f'{"Stable switch prob":20} {"~0.55":>12} {result_trained["stable_switch_prob"]:>12.4f}')
print(f'{"Trans switch prob":20} {"~0.55":>12} {result_trained["trans_switch_prob"]:>12.4f}')

entropy_path = os.path.join(RUN_DIR, 'entropy_after_training.json')
with open(entropy_path, 'w') as f:
    json.dump(result_trained, f, indent=2)
print(f'💾 결과 저장: {entropy_path}')


# ============================================================
# STEP 8: LoRA Expert 3개 정의 및 학습
#
# Expert 0 (Stable)    : 안정 맥락 — 정확하고 집중적인 생성
# Expert 1 (Transition): 전환 맥락 — 유연하고 탐색적인 생성
# Expert 2 (Creative)  : 고창의 맥락 — 최대 다양성
#
# 라우팅 규칙 (Δx 기반):
#   Δx_hybrid < 0.2  → Expert 0
#   0.2 ≤ Δx < 0.45  → Expert 1
#   Δx ≥ 0.45        → Expert 2
# ============================================================

# LoRA 설정 (T4 15GB 기준 r=4, A100이면 r=8 가능)
LORA_R = 4  # T4에서는 4, A100에서는 8

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_R * 2,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj'],
    bias='none',
)

# Expert별 학습 데이터 (프롬프트 + 완성 텍스트)
EXPERT_DATA = {
    'stable': [
        # 사실적, 정확한 내용 위주
        ("수학의 기본 원리는", "공리계로부터 출발하며, 논리적 추론을 통해 정리를 증명한다."),
        ("The capital of France is", "Paris, which has been the capital since the 12th century."),
        ("물은 100도에서", "끓기 시작하며, 이는 표준 대기압 조건에서의 수치이다."),
        ("Python is a programming language that", "emphasizes code readability and simplicity."),
        ("태양은 지구로부터", "약 1억 5천만 킬로미터 떨어진 항성이다."),
        ("The speed of light is approximately", "300,000 kilometers per second in a vacuum."),
        ("한국의 수도는", "서울이며, 약 천만 명의 인구가 거주한다."),
        ("Photosynthesis is the process by which", "plants convert sunlight into chemical energy."),
    ],
    'transition': [
        # 전환, 대비, 예상 밖의 전개
        ("처음에는 단순해 보였지만", "예상치 못한 복잡성이 드러나기 시작했다."),
        ("Although everything seemed stable,", "the situation changed dramatically overnight."),
        ("과거의 방식이 통하지 않을 때", "새로운 접근법을 찾아야 한다는 것을 깨달았다."),
        ("The old rules no longer applied when", "the environment shifted fundamentally."),
        ("안정적이던 상황이 갑자기", "급격한 변화를 맞이하게 되었다."),
        ("What was once certain became", "uncertain as new information emerged."),
        ("기존의 틀을 깨고 나서야", "새로운 가능성이 보이기 시작했다."),
        ("The transition happened so quickly that", "nobody had time to prepare for it."),
    ],
    'creative': [
        # 창의적, 상상력, 열린 결말
        ("만약 중력이 없다면", "우주는 완전히 다른 형태로 진화했을 것이다."),
        ("In a world where time flows backwards,", "memories would become prophecies."),
        ("AI가 감정을 느낀다면", "그것이 인류와 공존하는 방식은 근본적으로 달라질 것이다."),
        ("Imagine a civilization that never", "discovered the concept of linear time."),
        ("의식이 디지털로 이전된다면", "개인의 정체성은 어떻게 정의될까?"),
        ("If language had never evolved,", "human thought would be fundamentally different."),
        ("무한한 에너지가 존재한다면", "사회의 모든 갈등 구조가 바뀔 것이다."),
        ("A society built entirely on", "the principle of constant change would look like this."),
    ]
}

def train_lora_expert(base_model, tokenizer, expert_name, train_pairs,
                      lora_cfg, epochs=3, lr=2e-4):
    """
    단일 LoRA expert 학습.
    train_pairs: [(prompt, completion), ...]
    """
    print(f'\nExpert {expert_name} 학습 중...')

    # LoRA 적용
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=lr)

    loss_history = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for prompt, completion in train_pairs:
            full_text = prompt + ' ' + completion
            inputs = tokenizer(full_text, return_tensors='pt',
                               max_length=128, truncation=True).to(base_model.device)

            # causal LM loss: prompt 부분은 -100으로 마스킹
            prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids
            prompt_len = prompt_ids.shape[1]

            labels = inputs.input_ids.clone()
            labels[:, :prompt_len] = -100  # prompt 부분 무시

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(train_pairs)
        loss_history.append(avg)
        print(f'  Epoch {epoch+1}/{epochs} | loss={avg:.4f}')

    model.eval()

    # LoRA 가중치 저장
    save_path = os.path.join(RUN_DIR, f'lora_expert_{expert_name}')
    model.save_pretrained(save_path)
    print(f'  💾 저장: {save_path}')

    # base_model로 돌아오기 위해 LoRA 제거
    model = model.merge_and_unload()

    return save_path, loss_history

# 3개 expert 순차 학습
expert_paths = {}
all_loss_histories = {}

for expert_name, data_pairs in EXPERT_DATA.items():
    path, losses = train_lora_expert(
        base_model, tokenizer,
        expert_name, data_pairs,
        lora_cfg, epochs=5, lr=2e-4
    )
    expert_paths[expert_name] = path
    all_loss_histories[expert_name] = losses

print('\n✅ 모든 Expert 학습 완료')
print(expert_paths)


# ============================================================
# STEP 9: LoRA Expert Switching 생성 루프
#
# Δx_hybrid 값에 따라 세 Expert 중 하나를 선택
# PolicyNet의 switch 판단과 결합
# ============================================================

def select_expert(delta_hybrid, switch_prob,
                  threshold_stable=0.2, threshold_creative=0.45):
    """
    Δx와 PolicyNet switch 압력으로 expert 선택.
    """
    if delta_hybrid < threshold_stable and switch_prob < 0.5:
        return 'stable'
    elif delta_hybrid >= threshold_creative or switch_prob >= 0.6:
        return 'creative'
    else:
        return 'transition'


def nomadic_generate_with_lora(prompt, expert_paths, policy_net,
                                max_steps=50, T_stable=0.1, T_transition=1.2):
    """
    단계별로 Expert를 전환하면서 생성.
    expert_paths: {'stable': path, 'transition': path, 'creative': path}
    """
    tracker = HybridDeltaTracker()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(base_model.device)

    log = []
    current_expert_name = 'stable'
    current_model = PeftModel.from_pretrained(base_model, expert_paths['stable'])
    current_model.eval()

    print(f'\n--- Nomadic LoRA Generation ---')
    print(f'프롬프트: "{prompt}"\n')

    for step in range(max_steps):
        with torch.no_grad():
            out = current_model(input_ids, output_hidden_states=True)

        logits = out.logits[:, -1, :]
        hidden = out.hidden_states[-1][:, -1, :]
        probs = F.softmax(logits, dim=-1)
        err = 1.0 - probs.max().item()

        rec = tracker.compute(hidden, err)
        meta = build_meta_signals(rec, base_model.device)

        with torch.no_grad():
            ss, mode = policy_net(hidden, meta)
        switch_prob = ss[0, 1].item()

        # Expert 선택
        new_expert = select_expert(rec['delta_hybrid'], switch_prob)

        # Expert 전환 (필요한 경우만)
        expert_switched = False
        if new_expert != current_expert_name:
            current_expert_name = new_expert
            current_model = PeftModel.from_pretrained(
                base_model, expert_paths[new_expert]
            )
            current_model.eval()
            expert_switched = True

        # 온도 제어
        temp = T_stable + (T_transition - T_stable) * rec['delta_hybrid']

        next_token = torch.multinomial(
            F.softmax(logits / max(temp, 0.01), dim=-1),
            num_samples=1
        )
        word = tokenizer.decode(next_token[0])
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        switch_marker = ' ← SWITCH' if expert_switched else ''
        print(f"Step {step+1:02d} | [{current_expert_name:<10}] | '{word:<8}' "
              f"| Δx={rec['delta_hybrid']:.3f} | sw={switch_prob:.2f}"
              f" | τ={rec['tau_dyn']:.2f}{switch_marker}")

        log.append(dict(
            step=step+1, token=word,
            expert=current_expert_name,
            delta_hybrid=rec['delta_hybrid'],
            switch_prob=switch_prob,
            switched=expert_switched,
            temp=temp, tau_dyn=rec['tau_dyn']
        ))

        if next_token.item() == tokenizer.eos_token_id:
            break

    result = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return result, log


# 생성 테스트
TEST_PROMPTS = [
    '기술의 발전은 궁극적으로 인류에게',
    '처음에는 모든 것이 안정적이었지만 갑자기',
    '만약 AI가 감정을 가진다면',
]

all_results = {}
for prompt in TEST_PROMPTS:
    result, log = nomadic_generate_with_lora(
        prompt, expert_paths, policy_net, max_steps=40
    )
    all_results[prompt] = {'result': result, 'log': log}
    print(f'\n--- 최종 생성 ---\n{result}\n')

# 드라이브 저장
results_path = os.path.join(RUN_DIR, 'lora_generation_results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f'💾 생성 결과 저장: {results_path}')


# ============================================================
# STEP 10: 최종 지표 측정 및 시각화
# ============================================================
def measure_final_metrics(all_results):
    """expert switching 패턴 분석"""
    total_steps = 0
    total_switches = 0
    expert_usage = {'stable': 0, 'transition': 0, 'creative': 0}

    for prompt, data in all_results.items():
        for rec in data['log']:
            total_steps += 1
            expert_usage[rec['expert']] += 1
            if rec['switched']:
                total_switches += 1

    switch_rate = total_switches / total_steps if total_steps > 0 else 0

    print('\n=== Expert Switching 분석 ===')
    print(f'총 스텝: {total_steps}')
    print(f'총 전환: {total_switches} ({switch_rate:.1%})')
    print(f'Expert 사용 비율:')
    for exp, count in expert_usage.items():
        print(f'  {exp}: {count} ({count/total_steps:.1%})')

    return dict(
        total_steps=total_steps,
        total_switches=total_switches,
        switch_rate=switch_rate,
        expert_usage={k: v/total_steps for k, v in expert_usage.items()}
    )

metrics = measure_final_metrics(all_results)

# 시각화: 첫 번째 프롬프트의 스텝별 Δx + expert
first_prompt = TEST_PROMPTS[0]
first_log = all_results[first_prompt]['log']

steps = [r['step'] for r in first_log]
deltas = [r['delta_hybrid'] for r in first_log]
experts = [r['expert'] for r in first_log]
expert_colors = {'stable': 'green', 'transition': 'orange', 'creative': 'red'}
colors = [expert_colors[e] for e in experts]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].bar(steps, deltas, color=colors, alpha=0.7)
axes[0].axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='stable threshold')
axes[0].axhline(0.45, color='red', linestyle='--', alpha=0.5, label='creative threshold')
axes[0].set_ylabel('Δx_hybrid')
axes[0].set_title(f'LoRA Expert Switching — "{first_prompt}"')
axes[0].legend()

# expert 레이블
expert_numeric = [{'stable': 0, 'transition': 1, 'creative': 2}[e] for e in experts]
axes[1].scatter(steps, expert_numeric, c=colors, s=60, zorder=5)
axes[1].plot(steps, expert_numeric, 'k-', alpha=0.3)
axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(['stable', 'transition', 'creative'])
axes[1].set_xlabel('Generation Step')
axes[1].set_ylabel('Active Expert')

# 전환 시점 표시
for rec in first_log:
    if rec['switched']:
        axes[0].axvline(rec['step'], color='blue', alpha=0.3, linewidth=1)
        axes[1].axvline(rec['step'], color='blue', alpha=0.3, linewidth=1)

plt.tight_layout()
fig_path = os.path.join(RUN_DIR, 'lora_expert_switching.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f'💾 시각화 저장: {fig_path}')

# 최종 요약 저장
final_summary = {
    'run_id': RUN_ID,
    'phases': ['policy_training', 'lora_experts', 'nomadic_generation'],
    'entropy_before': {'stable_H': 1.806, 'trans_H': 2.537, 'delta_H': 0.731},
    'entropy_after_policy': result_trained,
    'switching_metrics': metrics,
    'expert_paths': expert_paths,
}
with open(os.path.join(RUN_DIR, 'final_summary.json'), 'w') as f:
    json.dump(final_summary, f, indent=2)

print('\n=== 실행 완료 ===')
print(f'저장 위치: {RUN_DIR}')
print('드라이브에서 확인 가능한 파일들:')
for fname in sorted(os.listdir(RUN_DIR)):
    size = os.path.getsize(os.path.join(RUN_DIR, fname))
    print(f'  {fname} ({size:,} bytes)')
