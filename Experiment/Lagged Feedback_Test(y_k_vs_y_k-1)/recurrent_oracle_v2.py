# -*- coding: utf-8 -*-
"""recurrent_oracle_v3_lagged.ipynb

Nomadic Intelligence — §4.10 Recurrent Gate + Oracle Baseline (v3)
* Temporal Lag (시차 기동) 적용 완료: 
  현재 배치(k)의 기동을 결정할 때 현재의 정답(y_k)을 훔쳐보지 않고,
  이전 배치(k-1)의 오차(prev_expl)와 설명력(prev_gap)만을 사용하여 인과율을 엄격히 준수함.
"""

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

import os, random, math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams['figure.dpi'] = 120

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS  = [42, 123, 456]

@dataclass
class Config:
    seed: int = 42; device: str = DEVICE
    input_dim: int = 2; output_dim: int = 1; overlap_std: float = 0.9
    hidden_dim: int = 64; num_experts: int = 3; gate_hidden_dim: int = 64
    temperature: float = 0.60
    epochs: int = 220; lr: float = 2e-3; weight_decay: float = 1e-5
    phase_batch_size: int = 64; phase_train_cycles: int = 40
    phase_test_cycles: int = 12; transition_steps: int = 8
    ema_decay: float = 0.80; err_baseline_momentum: float = 0.85
    w_env: float = 1.0; w_err: float = 2.0
    alpha_dogma: float = 0.04; beta_nomad: float = 0.05; beta_phi: float = 0.02
    gamma_diversity: float = 0.08; lambda_sep: float = 0.08
    lambda_cons: float = 0.03; lambda_load: float = 0.03
    tau_k_min: int = 3; tau_k_penalty: float = 0.05
    use_dynamic_tau: bool = True; tau_min: float = 2.0; tau_max: float = 8.0
    tau_var_scale: float = 6.0; tau_var_window: int = 8
    phi_scale_env: float = 1.0; phi_scale_err: float = 1.5
    phi_scale_explain: float = 1.5; phi_scale_gap: float = 0.8
    temp_stable: float = 0.35; temp_transition: float = 0.90
    use_hard_switch: bool = True; phi_hard_threshold: float = 0.30
    policy_hidden_dim: int = 64; policy_mix_weight: float = 0.25
    policy_weight_stay: float = 0.20; policy_weight_target: float = 0.20
    policy_weight_mode: float = 0.10; policy_switch_threshold: float = 0.50

def count_params(m): return sum(p.numel() for p in m.parameters())

REGIME_TO_ID = {'A':0,'B':1,'C':2}
ID_TO_REGIME = {0:'A',1:'B',2:'C'}
REGIME_ORDER = ['A','B','C']

def sample_regime_x(regime, n, cfg):
    centers = {'A':(2.5,2.5),'B':(-2.5,-2.5),'C':(2.5,-2.5)}
    c = torch.tensor(centers[regime], device=cfg.device)
    return cfg.overlap_std * torch.randn(n, 2, device=cfg.device) + c

def regime_function(x, regime):
    x1, x2 = x[:,0], x[:,1]
    if regime=='A':   y = x1 + x2
    elif regime=='B': y = x1 - x2
    elif regime=='C': y = -x1 + 0.5*x2
    return y.unsqueeze(-1)

def generate_phase_sequence(cfg, cycles, shuffle_regimes=False, rng_seed=None):
    xs,ys,rs,tags = [],[],[],[]
    rng = np.random.RandomState(rng_seed) if rng_seed is not None else np.random.RandomState()
    for cyc in range(cycles):
        order = rng.permutation(REGIME_ORDER).tolist() if shuffle_regimes else REGIME_ORDER
        for i, curr_r in enumerate(order):
            next_r = order[(i+1)%3]
            x_s = sample_regime_x(curr_r, cfg.phase_batch_size, cfg)
            xs.append(x_s); ys.append(regime_function(x_s,curr_r))
            rs.append(torch.full((cfg.phase_batch_size,),REGIME_TO_ID[curr_r],dtype=torch.long,device=cfg.device))
            tags.extend([f'stable_{curr_r}']*cfg.phase_batch_size)
            for step in range(cfg.transition_steps):
                alpha = (step+1)/cfg.transition_steps
                xa = sample_regime_x(curr_r,cfg.phase_batch_size,cfg)
                xb = sample_regime_x(next_r,cfg.phase_batch_size,cfg)
                xm = (1-alpha)*xa + alpha*xb
                ym = (1-alpha)*regime_function(xm,curr_r) + alpha*regime_function(xm,next_r)
                dom = curr_r if alpha<0.5 else next_r
                xs.append(xm); ys.append(ym)
                rs.append(torch.full((cfg.phase_batch_size,),REGIME_TO_ID[dom],dtype=torch.long,device=cfg.device))
                tags.extend([f'transition_{curr_r}_to_{next_r}']*cfg.phase_batch_size)
    return torch.cat(xs),torch.cat(ys),torch.cat(rs),tags

def iterate_minibatches(X,Y,R,bs):
    n=X.size(0)
    for s in range(0,n,bs): yield X[s:min(s+bs,n)],Y[s:min(s+bs,n)],R[s:min(s+bs,n)]

# ============================================================
# Model Definitions
# ============================================================

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,output_dim),
        )
    def forward(self,x): return self.net(x)

class StandardMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, gate_hidden):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim,hidden_dim,output_dim) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(input_dim,gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden,gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden,num_experts),
        )
    def forward(self, x, hard=False):
        gp = F.softmax(self.gate(x), dim=-1)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        r  = F.one_hot(gp.argmax(-1),self.num_experts).float() if hard else gp
        return (r.unsqueeze(-1)*eo).sum(1), gp, eo

class GRUMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, dropout=0.15):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim  = hidden_dim
        self.experts     = nn.ModuleList([Expert(input_dim,hidden_dim,output_dim) for _ in range(num_experts)])
        self.gru         = nn.GRUCell(input_dim, hidden_dim)
        self.gate_head   = nn.Linear(hidden_dim, num_experts)
        self.dropout     = nn.Dropout(p=dropout)

    def forward(self, x, h, temperature=1.0, hard=False):
        x_mean = x.mean(0, keepdim=True)
        h_new  = self.gru(x_mean, h)
        h_drop = self.dropout(h_new) if self.training else h_new
        logits = self.gate_head(h_drop)
        gp_batch = F.softmax(logits / temperature, dim=-1)
        gp = gp_batch.expand(x.size(0), -1)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        r  = F.one_hot(gp.argmax(-1),self.num_experts).float() if hard else gp
        return (r.unsqueeze(-1)*eo).sum(1), gp, eo, h_new

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_dim, device=device)

class NomadicMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts,
                 gate_hidden_dim, policy_hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim,hidden_dim,output_dim) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(input_dim+2,gate_hidden_dim), nn.ReLU(),
            nn.Linear(gate_hidden_dim,gate_hidden_dim), nn.ReLU(),
            nn.Linear(gate_hidden_dim,num_experts),
        )
        self.policy_shared = nn.Sequential(
            nn.Linear(input_dim+5,policy_hidden_dim), nn.ReLU(),
            nn.Linear(policy_hidden_dim,policy_hidden_dim), nn.ReLU(),
        )
        self.stay_head   = nn.Linear(policy_hidden_dim,2)
        self.target_head = nn.Linear(policy_hidden_dim,num_experts)
        self.mode_head   = nn.Linear(policy_hidden_dim,2)

    def gate_forward(self, x, dh, de, temperature):
        return F.softmax(self.gate(torch.cat([x,dh,de],dim=-1)) / temperature, dim=-1)

    def policy_forward(self, pi):
        h = self.policy_shared(pi)
        return (F.softmax(self.stay_head(h),dim=-1),
                F.softmax(self.target_head(h),dim=-1),
                F.softmax(self.mode_head(h),dim=-1))

    def forward(self, x, dh, de, temperature, hard=False):
        gp = self.gate_forward(x,dh,de,temperature)
        eo = torch.stack([e(x) for e in self.experts], dim=1)
        r  = F.one_hot(gp.argmax(-1),self.num_experts).float() if hard else gp
        return (r.unsqueeze(-1)*eo).sum(1), gp, eo

# ============================================================
# Utilities
# ============================================================

class HybridDeltaTracker:
    def __init__(self,cfg):
        self.cfg=cfg; self.prev_x_mean=None; self.err_ema=None
        self.err_baseline=None; self.recent_de=deque(maxlen=cfg.tau_var_window)
    def reset(self):
        self.prev_x_mean=None; self.err_ema=None
        self.err_baseline=None; self.recent_de.clear()
    def compute_dynamic_tau(self,s2):
        return float(np.clip(self.cfg.tau_min+(self.cfg.tau_max-self.cfg.tau_min)/(1+self.cfg.tau_var_scale*s2),
                             self.cfg.tau_min,self.cfg.tau_max))
    def compute(self, x, current_batch_mse):
        xm = x.mean(0, keepdim=True)
        de = 0.0 if self.prev_x_mean is None else float(torch.norm(xm-self.prev_x_mean, p=2).item())
        be = current_batch_mse.detach()
        if self.err_ema is None:
            self.err_ema = be; self.err_baseline = be; derr = 0.0
        else:
            self.err_ema = self.cfg.ema_decay*self.err_ema + (1-self.cfg.ema_decay)*be
            self.err_baseline = self.cfg.err_baseline_momentum*self.err_baseline + (1-self.cfg.err_baseline_momentum)*self.err_ema
            derr = float(torch.relu(self.err_ema-self.err_baseline).item())
        dh = float(torch.tanh(torch.tensor(self.cfg.w_env*de + self.cfg.w_err*derr)).item())
        self.prev_x_mean = xm.detach()
        self.recent_de.append(de)
        s2 = float(np.var(self.recent_de)) if len(self.recent_de) >= 2 else 0.0
        return torch.full((x.size(0),1), dh, device=self.cfg.device), de, derr, dh, s2, self.compute_dynamic_tau(s2)

class DwellTimeRegularizer:
    def __init__(self,cfg): self.cfg=cfg; self.cur=None; self.cnt=0
    def reset(self): self.cur=None; self.cnt=0
    def compute(self,gp,tau=None):
        dom=int(torch.bincount(gp.argmax(-1),minlength=gp.size(-1)).argmax().item())
        if dom==self.cur: self.cnt+=1
        else: self.cur=dom; self.cnt=1
        eps=1e-8; ent=-(gp*(gp+eps).log()).sum(-1).mean()
        tc=float(tau if tau is not None else self.cfg.tau_k_min)
        if self.cnt<=tc: return -self.cfg.tau_k_penalty*ent
        else: return min((self.cnt-tc)*self.cfg.tau_k_penalty,self.cfg.tau_k_penalty*10)*ent

def gate_entropy(gp):
    eps=1e-8; return -(gp*(gp+eps).log()).sum(-1)

def compute_load_balancing_loss(gp):
    K=gp.size(-1); mg=gp.mean(0); top1=gp.argmax(-1)
    return K*(torch.bincount(top1,minlength=K).float()/top1.size(0)*mg).sum()

def compute_diversity_loss(eo):
    K=eo.size(1)
    if K<2: return eo.new_zeros(1).squeeze()
    ii,jj=zip(*[(i,j) for i in range(K) for j in range(i+1,K)])
    return F.cosine_similarity(eo[:,ii,:],eo[:,jj,:],dim=-1).mean()

def compute_explanation_signals(yt,yh,eo,gp):
    expl=F.mse_loss(yh,yt)
    pe=((eo-yt.unsqueeze(1))**2).mean(-1)
    t1=pe.gather(1,gp.argmax(-1).unsqueeze(1)).mean()
    return expl,torch.relu(t1-pe.min(1).values.mean())

def compute_phi(de,derr,expl,gap,cfg):
    d=expl.device
    return torch.tanh(cfg.phi_scale_env*torch.tensor(de,device=d)+cfg.phi_scale_err*torch.tensor(derr,device=d)
                      +cfg.phi_scale_explain*expl.detach()+cfg.phi_scale_gap*gap.detach())

def compute_temp(phi,cfg):
    return cfg.temp_stable+(cfg.temp_transition-cfg.temp_stable)*float(phi.mean().item())

def build_policy_input(xb,dh_t,de_t,phi,s2,dtau):
    xs=xb.mean(0,keepdim=True).expand(xb.size(0),-1)
    pt=torch.full((xb.size(0),1),float(phi.mean().item()),device=xb.device)
    s2t=torch.full((xb.size(0),1),float(np.tanh(s2*10)),device=xb.device)
    tt=torch.full((xb.size(0),1),float(np.tanh((dtau-5)/5)),device=xb.device)
    return torch.cat([xs,dh_t,de_t,pt,s2t,tt],dim=-1)

def build_policy_targets(yb,eo,phi,s2,dtau,cfg):
    pe=((eo-yb.unsqueeze(1))**2).mean(-1)
    tgt=pe.mean(0).argmin().long()
    pv=float(phi.mean().item())
    sw=1 if(pv>cfg.policy_switch_threshold or s2>0.05) else 0
    mod=1 if(pv<=cfg.policy_switch_threshold and dtau>=5.5) else 0
    return sw,tgt,mod

def compute_regime_gate_stats(gp,rb):
    dev=gp.device; vm=[]; lc=torch.tensor(0.,device=dev); cnt=0
    for rid in range(3):
        mask=rb==rid
        if mask.sum()==0: continue
        gr=gp[mask]; ur=gr.mean(0); vm.append(ur)
        lc=lc+((gr-ur.unsqueeze(0))**2).sum(-1).mean(); cnt+=1
    if cnt>0: lc=lc/cnt
    if len(vm)<2: return torch.tensor(0.,device=dev),lc
    pw=[torch.norm(vm[i]-vm[j],p=2) for i in range(len(vm)) for j in range(i+1,len(vm))]
    return -torch.stack(pw).mean(),lc

def regimewise_usage(gp,rb,K):
    top1=gp.argmax(-1); usage={}
    for rid in range(3):
        mask=rb==rid; name=ID_TO_REGIME[rid]
        if mask.sum()==0: usage[name]=np.zeros(K); continue
        c=torch.bincount(top1[mask],minlength=K).float()
        usage[name]=(c/c.sum().clamp_min(1)).cpu().numpy()
    return usage

def infer_r2e(usage): return {r:int(np.argmax(usage[r])) for r in ['A','B','C']}

def compute_switch_latency(reg_seq,top1_seq,r2e):
    lats=[]; prev=reg_seq[0] if reg_seq else None
    for t in range(1,len(reg_seq)):
        curr=reg_seq[t]
        if curr!=prev:
            tgt=r2e.get(curr)
            if tgt is not None:
                for k in range(t,len(top1_seq)):
                    if int(top1_seq[k])==int(tgt): lats.append(k-t); break
        prev=curr
    return lats


# ============================================================
# Evaluations & Training (Peeking Bias Removed)
# ============================================================

def eval_stdmoe(model,X,Y,R,phase_tags,cfg):
    model.eval(); all_y,all_g,tags,ents=[],[],[],[]
    with torch.no_grad():
        for bi,(xb,yb,rb) in enumerate(iterate_minibatches(X,Y,R,cfg.phase_batch_size)):
            yh,gp,_=model(xb); all_y.append(yh); all_g.append(gp)
            tags.append(phase_tags[bi*cfg.phase_batch_size])
            ents.append(gate_entropy(gp).mean().item())
    seq_mse=F.mse_loss(torch.cat(all_y),Y).item()
    sh=[e for t,e in zip(tags,ents) if t.startswith('stable_')]
    th=[e for t,e in zip(tags,ents) if t.startswith('transition_')]
    return seq_mse,{'sh':float(np.mean(sh)) if sh else float('nan'),
                    'th':float(np.mean(th)) if th else float('nan'),
                    'dh':float(np.mean(th)-np.mean(sh)) if(sh and th) else float('nan')}


def eval_gru(model,X,Y,R,phase_tags,cfg):
    model.eval(); h=model.init_hidden(cfg.device)
    all_y,all_g,tags,ents,top1s,regs=[],[],[],[],[],[]
    with torch.no_grad():
        for bi,(xb,yb,rb) in enumerate(iterate_minibatches(X,Y,R,cfg.phase_batch_size)):
            yh,gp,_,h=model(xb,h,cfg.temperature)
            all_y.append(yh); all_g.append(gp)
            tags.append(phase_tags[bi*cfg.phase_batch_size])
            ents.append(gate_entropy(gp).mean().item())
            top1s.append(int(torch.bincount(gp.argmax(-1),minlength=cfg.num_experts).argmax().item()))
            regs.append(ID_TO_REGIME[int(rb[0].item())])
    seq_mse=F.mse_loss(torch.cat(all_y),Y).item()
    G=torch.cat(all_g); usage=regimewise_usage(G,R,cfg.num_experts)
    r2e=infer_r2e(usage)
    lats=compute_switch_latency(regs,np.array(top1s),r2e)
    sh=[e for t,e in zip(tags,ents) if t.startswith('stable_')]
    th=[e for t,e in zip(tags,ents) if t.startswith('transition_')]
    return seq_mse,{'sh':float(np.mean(sh)) if sh else float('nan'),
                    'th':float(np.mean(th)) if th else float('nan'),
                    'dh':float(np.mean(th)-np.mean(sh)) if(sh and th) else float('nan'),
                    'lat':float(np.mean(lats)) if lats else float('nan')}


def eval_nomadic(model,X,Y,R,phase_tags,cfg):
    """
    [핵심 수정] 시차 기동(Temporal Lag) 적용
    - 과거의 오차(prev_expl, prev_gap)만 사용하여 현재 기동 결정
    """
    model.eval(); tracker=HybridDeltaTracker(cfg); tracker.reset()
    all_y,all_g,tags,ents,top1s,regs=[],[],[],[],[],[]
    
    # 0턴 초기화 (미래를 보지 않음)
    prev_expl = torch.tensor(0.0, device=cfg.device)
    prev_gap  = torch.tensor(0.0, device=cfg.device)
    
    with torch.no_grad():
        for bi,(xb,yb,rb) in enumerate(iterate_minibatches(X,Y,R,cfg.phase_batch_size)):
            
            # 1) 환경 변화 감지 (과거의 에러를 tracker에 피드백)
            dh_t,de,derr,dh,s2,dtau = tracker.compute(xb, prev_expl)
            de_t = torch.full((xb.size(0),1), derr, device=cfg.device)
            
            # 2) 과거의 설명력으로 현재의 기동 신호(phi) 계산
            phi = compute_phi(de, derr, prev_expl, prev_gap, cfg)
            temp = compute_temp(phi, cfg)
            
            # 3) PolicyNet 결심 (오직 과거 기억 + 현재 입력 X의 변화량)
            pi = build_policy_input(xb, dh_t, de_t, phi, s2, dtau)
            sw, tp, mp = model.policy_forward(pi)
            
            # 4) Forward Pass 및 라우팅 확정
            yh, gp, eo = model(xb, dh_t, de_t, temp)
            em = cfg.policy_mix_weight * float(sw[:,1].mean().item())
            ti = tp.mean(0).argmax()
            toh = (F.one_hot(ti, cfg.num_experts).float().unsqueeze(0).expand(xb.size(0),-1) - gp).detach() + gp
            mx = (1-em)*gp + em*toh
            fs = dh > cfg.phi_hard_threshold
            hm = cfg.use_hard_switch and (mp[:,1].mean().item() > 0.5) and not fs
            fr = F.one_hot(mx.argmax(-1), cfg.num_experts).float() if hm else mx
            yh = (fr.unsqueeze(-1)*eo).sum(1)
            gp = fr
            
            # 5) 다음 턴(k+1)을 위해 현재(k)의 실제 결과물을 사후 저장
            curr_expl, curr_gap = compute_explanation_signals(yb, yh, eo, gp)
            prev_expl = curr_expl.detach()
            prev_gap  = curr_gap.detach()

            all_y.append(yh); all_g.append(gp)
            tags.append(phase_tags[bi*cfg.phase_batch_size])
            ents.append(gate_entropy(gp).mean().item())
            top1s.append(int(torch.bincount(gp.argmax(-1),minlength=cfg.num_experts).argmax().item()))
            regs.append(ID_TO_REGIME[int(rb[0].item())])
            
    seq_mse=F.mse_loss(torch.cat(all_y),Y).item()
    G=torch.cat(all_g); usage=regimewise_usage(G,R,cfg.num_experts)
    r2e=infer_r2e(usage)
    lats=compute_switch_latency(regs,np.array(top1s),r2e)
    sh=[e for t,e in zip(tags,ents) if t.startswith('stable_')]
    th=[e for t,e in zip(tags,ents) if t.startswith('transition_')]
    return seq_mse,{'sh':float(np.mean(sh)) if sh else float('nan'),
                    'th':float(np.mean(th)) if th else float('nan'),
                    'dh':float(np.mean(th)-np.mean(sh)) if(sh and th) else float('nan'),
                    'lat':float(np.mean(lats)) if lats else float('nan')}

# ----------------- 오라클 학습/평가 함수 그대로 유지 -----------------
def train_oracle_experts(cfg, Xtr, Ytr, Rtr):
    experts = []
    for rid in range(cfg.num_experts):
        regime_name = ID_TO_REGIME[rid]
        mask = Rtr == rid
        if mask.sum() == 0:
            experts.append(Expert(cfg.input_dim, cfg.hidden_dim, cfg.output_dim).to(cfg.device))
            continue
        Xr = Xtr[mask]; Yr = Ytr[mask]
        e = Expert(cfg.input_dim, cfg.hidden_dim, cfg.output_dim).to(cfg.device)
        opt = torch.optim.Adam(e.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n = Xr.size(0)
        for ep in range(cfg.epochs):
            e.train()
            perm = torch.randperm(n, device=cfg.device)
            for s in range(0, n, cfg.phase_batch_size):
                idx = perm[s:min(s+cfg.phase_batch_size, n)]
                opt.zero_grad()
                F.mse_loss(e(Xr[idx]), Yr[idx]).backward()
                opt.step()
        e.eval()
        with torch.no_grad():
            train_mse = F.mse_loss(e(Xr), Yr).item()
        experts.append(e)
    return experts

def eval_oracle(experts, X, Y, R, phase_tags, cfg):
    [e.eval() for e in experts]
    all_y, tags, ents, top1s, regs = [], [], [], [], []
    with torch.no_grad():
        for bi, (xb, yb, rb) in enumerate(iterate_minibatches(X, Y, R, cfg.phase_batch_size)):
            tag = phase_tags[bi * cfg.phase_batch_size]
            dom_regime = int(rb[0].item())
            yh = experts[dom_regime](xb)
            K = len(experts)
            gp = F.one_hot(torch.full((xb.size(0),), dom_regime, dtype=torch.long, device=cfg.device), K).float()
            all_y.append(yh); tags.append(tag); ents.append(0.0)
            top1s.append(dom_regime); regs.append(ID_TO_REGIME[dom_regime])
    seq_mse = F.mse_loss(torch.cat(all_y), Y).item()
    r2e = {r: i for i, r in enumerate(['A', 'B', 'C'])}
    lats = compute_switch_latency(regs, np.array(top1s), r2e)
    return seq_mse, {'sh': 0.0, 'th': 0.0, 'dh': 0.0, 'lat': float(np.mean(lats)) if lats else float('nan')}


# ----------------- Training Loops -----------------
def train_stdmoe(cfg,Xtr,Ytr,Rtr,Xte,Yte,Rte,tags_te):
    m=StandardMoE(cfg.input_dim,cfg.hidden_dim,cfg.output_dim,cfg.num_experts,cfg.gate_hidden_dim).to(cfg.device)
    opt=torch.optim.Adam(m.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    mse_log,dyn_log=[],[]
    for ep in range(cfg.epochs):
        m.train()
        for xb,yb,_ in iterate_minibatches(Xtr,Ytr,Rtr,cfg.phase_batch_size):
            opt.zero_grad()
            yh,gp,eo=m(xb)
            (F.mse_loss(yh,yb)+cfg.gamma_diversity*compute_diversity_loss(eo)
             +cfg.lambda_load*compute_load_balancing_loss(gp)).backward()
            opt.step()
        mse,dyn=eval_stdmoe(m,Xte,Yte,Rte,tags_te,cfg)
        mse_log.append(mse); dyn_log.append(dyn)
    return m,mse_log,dyn_log


def train_gru(cfg,Xtr,Ytr,Rtr,Xte,Yte,Rte,tags_te):
    m=GRUMoE(cfg.input_dim,cfg.hidden_dim,cfg.output_dim,cfg.num_experts,dropout=0.15).to(cfg.device)
    opt=torch.optim.Adam(m.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay*5)
    mse_log,dyn_log=[],[]
    for ep in range(cfg.epochs):
        m.train()
        h=m.init_hidden(cfg.device)
        for xb,yb,rb in iterate_minibatches(Xtr,Ytr,Rtr,cfg.phase_batch_size):
            opt.zero_grad()
            yh,gp,eo,h_new=m(xb,h.detach(),cfg.temperature)
            h=h_new.detach()
            sep_l,cons_l=compute_regime_gate_stats(gp,rb)
            (F.mse_loss(yh,yb)+cfg.gamma_diversity*compute_diversity_loss(eo)
             +cfg.lambda_load*compute_load_balancing_loss(gp)+cfg.lambda_sep*sep_l+cfg.lambda_cons*cons_l).backward()
            opt.step()
        mse,dyn=eval_gru(m,Xte,Yte,Rte,tags_te,cfg)
        mse_log.append(mse); dyn_log.append(dyn)
    return m,mse_log,dyn_log


def train_nomadic(cfg,Xtr,Ytr,Rtr,Xte,Yte,Rte,tags_te):
    """
    [핵심 수정] Train 단계 시차 기동 적용
    """
    m=NomadicMoE(cfg.input_dim,cfg.hidden_dim,cfg.output_dim,cfg.num_experts,
                 cfg.gate_hidden_dim,cfg.policy_hidden_dim).to(cfg.device)
    opt=torch.optim.Adam(m.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    mse_log,dyn_log=[],[]
    for ep in range(cfg.epochs):
        m.train(); tracker=HybridDeltaTracker(cfg); tracker.reset()
        dwr=DwellTimeRegularizer(cfg); dwr.reset()
        
        # 0턴 초기화
        prev_expl = torch.tensor(0.0, device=cfg.device)
        prev_gap  = torch.tensor(0.0, device=cfg.device)
        
        for xb,yb,rb in iterate_minibatches(Xtr,Ytr,Rtr,cfg.phase_batch_size):
            opt.zero_grad()
            
            # 1) 환경 추적 (과거의 오차 활용)
            dh_t, de, derr, dh, s2, dtau = tracker.compute(xb, prev_expl)
            de_t = torch.full((xb.size(0),1), derr, device=cfg.device)
            
            # 2) 기동 신호 계산 (과거의 설명력 활용)
            phi = compute_phi(de, derr, prev_expl, prev_gap, cfg)
            temp = compute_temp(phi, cfg)
            
            # 3) Policy 결심 및 Forward
            pi = build_policy_input(xb, dh_t, de_t, phi, s2, dtau)
            sw, tp, mp = m.policy_forward(pi)
            
            yh, gp, eo = m(xb, dh_t, de_t, temp)
            em = cfg.policy_mix_weight * float(sw[:,1].mean().item())
            ti = tp.mean(0).argmax()
            toh = (F.one_hot(ti, cfg.num_experts).float().unsqueeze(0).expand(xb.size(0),-1) - gp).detach() + gp
            mx = (1-em)*gp + em*toh
            fs = dh > cfg.phi_hard_threshold
            hm = cfg.use_hard_switch and (mp[:,1].mean().item() > 0.5) and not fs
            fr = F.one_hot(mx.argmax(-1), cfg.num_experts).float() if hm else mx
            yh = (fr.unsqueeze(-1)*eo).sum(1)
            
            # 4) [중요] 사후 강평 (현재의 정답 yb 확인)
            curr_expl, curr_gap = compute_explanation_signals(yb, yh, eo, fr)
            
            # 다음 턴을 위한 저장
            prev_expl = curr_expl.detach()
            prev_gap  = curr_gap.detach()
            
            # 5) Loss 계산 및 모델 업데이트
            sl, cl = compute_regime_gate_stats(fr, rb)
            td = dtau if cfg.use_dynamic_tau else float(cfg.tau_k_min)
            dw = dwr.compute(fr, tau=td)
            
            # Policy Targets (사후적 관점에서 "무엇이 최선이었나"를 학습)
            t_sw, t_tg, t_md = build_policy_targets(yb, eo, phi, s2, dtau, cfg)
            sw_t = torch.full((xb.size(0),), t_sw, dtype=torch.long, device=cfg.device)
            tg_t = torch.full((xb.size(0),), int(t_tg), dtype=torch.long, device=cfg.device)
            md_t = torch.full((xb.size(0),), t_md, dtype=torch.long, device=cfg.device)
            
            loss = (F.mse_loss(yh, yb)
                  + cfg.beta_phi * (phi.detach() * curr_gap)
                  + cfg.alpha_dogma * (gp.mean(0).pow(2).sum() - 1/cfg.num_experts)
                  - cfg.beta_nomad * (-(gp*(gp+1e-8).log()).sum(-1).mean())
                  + cfg.gamma_diversity * compute_diversity_loss(eo)
                  + cfg.lambda_sep * sl + cfg.lambda_cons * cl
                  + cfg.lambda_load * compute_load_balancing_loss(fr)
                  + cfg.policy_weight_stay * F.nll_loss(torch.log(sw+1e-8), sw_t)
                  + cfg.policy_weight_target * F.nll_loss(torch.log(tp+1e-8), tg_t)
                  + cfg.policy_weight_mode * F.nll_loss(torch.log(mp+1e-8), md_t)
                  - dw)
            loss.backward(); opt.step()
            
        mse,dyn=eval_nomadic(m,Xte,Yte,Rte,tags_te,cfg)
        mse_log.append(mse); dyn_log.append(dyn)
        if(ep+1)%55==0 or ep==0:
            print(f'  [Nomadic] Ep{ep+1:03d} MSE={mse:.4f} ΔH={dyn["dh"]:.3f} StH={dyn["sh"]:.3f}')
    return m,mse_log,dyn_log

import time

if __name__ == '__main__':
    all_results = {m:{} for m in ['StdMoE','GRU','Nomadic','Oracle']}

    for seed in SEEDS:
        t0=time.time()
        print(f'\n========== Seed {seed} ==========')
        set_seed(seed); cfg=Config(seed=seed)
        Xtr,Ytr,Rtr,tags_tr=generate_phase_sequence(cfg,cfg.phase_train_cycles, shuffle_regimes=True, rng_seed=seed)
        Xte,Yte,Rte,tags_te=generate_phase_sequence(cfg,cfg.phase_test_cycles, shuffle_regimes=False)

        print('--- Standard MoE ---')
        sm,sm_mse,sm_dyn=train_stdmoe(cfg,Xtr,Ytr,Rtr,Xte,Yte,Rte,tags_te)
        all_results['StdMoE'][seed]={'mse_log':sm_mse,'dyn_log':sm_dyn}

        print('--- GRU MoE ---')
        gm,gm_mse,gm_dyn=train_gru(cfg,Xtr,Ytr,Rtr,Xte,Yte,Rte,tags_te)
        all_results['GRU'][seed]={'mse_log':gm_mse,'dyn_log':gm_dyn}

        print('--- Nomadic Full (Lagged) ---')
        nm,nm_mse,nm_dyn=train_nomadic(cfg,Xtr,Ytr,Rtr,Xte,Yte,Rte,tags_te)
        all_results['Nomadic'][seed]={'mse_log':nm_mse,'dyn_log':nm_dyn}

        print('--- Oracle (Label-conditioned) ---')
        set_seed(seed)
        oracle_experts = train_oracle_experts(cfg, Xtr, Ytr, Rtr)
        oracle_mse, oracle_dyn = eval_oracle(oracle_experts, Xte, Yte, Rte, tags_te, cfg)
        all_results['Oracle'][seed]={'mse_log':[oracle_mse],'dyn_log':[oracle_dyn]}
        print(f'  Seed {seed} done ({time.time()-t0:.0f}s)')

    print('\n=== All seeds complete ===')

    rows=[]
    for mn in ['StdMoE','GRU','Nomadic','Oracle']:
        mse_v,dh_v,sh_v,th_v,lat_v=[],[],[],[],[]
        for seed in SEEDS:
            r=all_results[mn][seed]
            mse_v.append(r['mse_log'][-1])
            d=r['dyn_log'][-1]
            dh_v.append(d['dh']); sh_v.append(d['sh']); th_v.append(d['th'])
        rows.append({
            'Model':mn,
            'Seq MSE': np.mean(mse_v),
            'ΔH':  np.mean(dh_v),
            'Stable H': np.mean(sh_v), 
            'Trans H': np.mean(th_v)
        })

    df=pd.DataFrame(rows)
    print('\n'+'='*75)
    print('§4.10 RECURRENT GATE + ORACLE BASELINE (Lagged v3)')
    print('='*75)
    print(df.to_string(float_format=lambda x: f'{x:.4f}' if isinstance(x,float) else str(x),index=False))

    nm_mse = df[df['Model']=='Nomadic']['Seq MSE'].values[0]
    gm_mse = df[df['Model']=='GRU']['Seq MSE'].values[0]
    or_mse = df[df['Model']=='Oracle']['Seq MSE'].values[0]
    
    print('\n--- 핵심 비교 ---')
    print(f'  GRU MoE vs Nomadic: MSE {gm_mse:.4f} vs {nm_mse:.4f}')
    print(f'  Nomadic / Oracle ratio:  {nm_mse/or_mse:.3f}  (1.0 = Oracle)')