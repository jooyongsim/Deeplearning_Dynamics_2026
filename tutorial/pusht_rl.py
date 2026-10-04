# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reinforcement Learning PushT — 처음부터 vs 모방 학습에서 시작
#
# `pusht_imitation.py` 에서는 시연을 **따라 하도록** 학습했습니다 (supervised).
# 강화학습 (RL) 은 시연 없이 **보상 (reward)** 만 보고, 직접 해 보면서 (trial and error) 더 많은 보상을 받는 행동을 찾습니다.
#
# $$
# \boxed{
# s_t \xrightarrow{\ \pi_\phi(a\mid s)\ } a_t
# \xrightarrow{\ \text{물리}\ } s_{t+1},\ r_t
# \qquad
# \max_\phi\ \mathbb E\Big[\sum_t \gamma^t r_t\Big]
# }
# $$
#
# 이 파일에서 두 가지를 비교합니다.
#
# | 실험 | 시작점 | 결과 (같은 랜덤 과제, 성공 기준 0.15 m / 10°) |
# |---|---|---|
# | **A. RL from scratch** | 무작위 신경망 | 1,640만 step (2.5시간) 학습 후에도 **성공 0%** — T를 goal 쪽으로 조금 옮기는 것까지만 배움 |
# | **B. BC → RL fine-tuning** | `pusht_imitation.py` 의 MLP-BC policy (성공 약 42%) | 아래 14절의 학습 로그 참고 |
#
# A가 실패하는 이유가 이 강의의 중요한 교훈입니다: PushT의 성공은 "접촉 위치를 고르고 → 돌아가서 → 미는" 긴 행동 순서 끝에만 옵니다.
# 무작위로 움직여서는 그 순서를 우연히 해낼 확률이 거의 0이라 (**exploration 문제**), 성공 보상을 한 번도 보지 못합니다.
# B는 시연에서 배운 policy로 이미 절반 가까이 성공하므로, RL은 "이미 되는 행동을 더 잘하게" 다듬기만 하면 됩니다.
#
# **실행 방법**
#
# ```bash
# python pusht_rl.py scratch      # A: RL from scratch   -> checkpoints/rl/scratch/
# python pusht_rl.py bc           # B: BC -> RL          -> checkpoints/rl/bc_ppo/   (먼저 pusht_imitation.py 로 MLP-BC 학습)
# python pusht_rl.py eval         # 저장된 policy들의 성공률
# python pusht_rl_realtime.py     # 실시간 창: script / RL scratch / BC / BC+RL (중간) / BC+RL (최종)
# ```
#
# 이 파일은 `import pusht_rl` 로 불러도 학습을 시작하지 않습니다 (정의만). `pusht_rl_realtime.py` 가 그렇게 씁니다.

# %%
import ast
import contextlib
import io
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, HERE)
CKPT_DIR = os.path.join(HERE, "checkpoints")
RL_DIR = os.path.join(CKPT_DIR, "rl")

# %% [markdown]
# ## 1. 환경과 expert는 `pusht_imitation.py` 의 것을 그대로
#
# `pusht_imitation.py` 는 위에서부터 실행하는 튜토리얼이라 `import` 하면 데이터 수집과 학습까지 돌아갑니다.
# 그래서 Python의 `ast` 모듈로 파일을 읽어 **필요한 `class` / `def` 정의만** 골라 실행합니다 (그림, 학습 셀은 실행하지 않음).
# 같은 환경, 같은 랜덤 과제 (seed), 같은 성공 기준을 써야 imitation과 RL을 공정하게 비교할 수 있기 때문입니다.

# %%
_NEEDED = {"circle_vs_T_fast", "wrap_angle", "PushTEnv", "ScriptedExpert", "T_outline", "run_episode",
           "to_goal_frame", "from_goal_frame", "action_to_model", "action_from_model", "make_obs",
           "MLP", "Normalized", "BCPolicy", "LearnedAgent", "cosine_betas", "DiffusionPolicy"}


def load_imitation_defs(path=os.path.join(HERE, "pusht_imitation.py")):
    tree = ast.parse(open(path, encoding="utf-8").read())
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _NEEDED)
            or (isinstance(n, ast.Assign) and isinstance(n.value, (ast.Constant, ast.Tuple))      # NAME = constant only
                and all(isinstance(t, (ast.Name, ast.Tuple)) for t in n.targets))]
    had = "PUSHT_NO_GUI" in os.environ
    os.environ["PUSHT_NO_GUI"] = "1"
    with contextlib.redirect_stdout(io.StringIO()):
        import pusht_keyboard_sim as ksim
    if not had:
        del os.environ["PUSHT_NO_GUI"]
    ns = {"__name__": "pusht_imitation_defs", "ksim": ksim, "DEVICE": "cpu"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    ksim.circle_vs_T = ns["circle_vs_T"] = ns["circle_vs_T_fast"]       # fast collision, same math
    return ns


IM = load_imitation_defs()
PushTEnv, ScriptedExpert, run_episode = IM["PushTEnv"], IM["ScriptedExpert"], IM["run_episode"]
wrap_angle, make_obs, action_from_model = IM["wrap_angle"], IM["make_obs"], IM["action_from_model"]
BCPolicy, LearnedAgent = IM["BCPolicy"], IM["LearnedAgent"]
MAX_STEPS = 300                                     # 30 s, same as the imitation evaluation
RL_SEED0 = 300_000                                  # training tasks; evaluation uses seeds 0, 1, 2, ...

# %% [markdown]
# ## 2. 보상 (reward) 설계
#
# 과제의 "남은 거리" 를 하나의 숫자로 정의합니다 (각도 1 rad ≈ 0.3 m 로 환산):
#
# $$
# c(s)=\lVert\mathbf x_G-\mathbf x_g\rVert+0.3\,\lvert\theta-\theta_g\rvert
# $$
#
# 매 step의 보상:
#
# $$
# r_t=\underbrace{10\,\big(c(s_t)-c(s_{t+1})\big)}_{\text{goal에 가까워진 만큼}}
# \;-\;\underbrace{0.02\,\max(\lVert\mathbf x_P-\mathbf x_G\rVert-1,\,0)}_{\text{T에서 멀리 있으면 벌점}}
# \;-\;\underbrace{0.01}_{\text{시간}}
# \;+\;\underbrace{10\cdot\mathbb 1[\text{성공}]}_{\text{성공 보너스}}
# $$
#
# 첫 항은 **potential-based shaping** 입니다: 한 episode 동안 더하면 $10\,(c(s_0)-c(s_T))$ 가 되어, 중간 경로와 상관없이 "얼마나 가까이 갔나" 만 남습니다.
# 그래서 보상 모양을 바꿔도 최적 policy는 바뀌지 않으면서 학습 신호는 매 step 생깁니다.
#
# 성공하면 episode가 끝나고 (terminal), 300 step이 지나면 잘립니다 (truncation). 잘린 경우는 "끝난 것" 이 아니므로
# 마지막 상태의 가치 $V(s_T)$ 로 이어 붙여 줍니다 (bootstrap).

# %%
def task_cost(sim):
    return float(np.linalg.norm(sim.T.pos - sim.goal_pos) + 0.3 * abs(wrap_angle(sim.T.angle - sim.goal_angle)))


def reward(c_before, c_after, sim, success):
    dist_pt = float(np.linalg.norm(sim.pusher.pos - sim.T.pos))
    return 10.0 * (c_before - c_after) - 0.02 * max(dist_pt - 1.0, 0.0) - 0.01 + (10.0 if success else 0.0)

# %% [markdown]
# ## 3. Observation 과 action — 두 실험이 다른 점
#
# 둘 다 imitation 튜토리얼과 같은 **goal 좌표계** ($\tilde{\mathbf p}=R(\theta_g)^\top(\mathbf p-\mathbf x_g)$) 를 씁니다.
#
# | | A. scratch | B. BC → RL |
# |---|---|---|
# | actor 입력 | $\tilde{\mathbf x}_P,\tilde{\mathbf x}_G,\cos\tilde\theta,\sin\tilde\theta,\tilde{\mathbf v}_G,\omega,\tilde{\mathbf v}_P,\tilde{\mathbf x}_G-\tilde{\mathbf x}_P$ (13) | BC와 같은 $(o_{t-1},o_t)$ (12) |
# | critic 입력 | actor와 같음 | $(o_{t-1},o_t)$ + 속도·시간 (18) |
# | action | $\mathbf u\in[-1,1]^2$, target $=\mathbf x_P+0.15\,R(\theta_g)\mathbf u$ | BC chunk의 첫 action (pusher 기준 offset, m) |
#
# B의 actor는 **BC 네트워크 그 자체** 입니다. 그래서 학습 0 step의 B = imitation policy (한 step씩 실행).
# critic (가치 함수 $V(s)$) 은 시연에 없던 것이라 새로 만들고, 처음 10번의 update 동안은 critic만 학습시켜 actor가 엉뚱한 방향으로 밀리지 않게 합니다.

# %%
ACT_SCALE = 0.15


def goal_rot(state):
    c, s = math.cos(state[7]), math.sin(state[7])
    return np.array([[c, s], [-s, c]])                      # R(theta_g)^T


def obs_scratch(env):
    s, sim = env.state(), env.sim
    Rt = goal_rot(s)
    p, T = Rt @ (s[0:2] - s[5:7]), Rt @ (s[2:4] - s[5:7])
    d = s[4] - s[7]
    vT, vP = Rt @ sim.T.vel, Rt @ sim.pusher.vel
    return np.array([p[0], p[1], T[0], T[1], math.cos(d), math.sin(d), vT[0], vT[1], sim.T.omega,
                     vP[0], vP[1], T[0] - p[0], T[1] - p[1]], dtype=np.float32)


def target_scratch(env, u):
    s = env.state()
    u = np.clip(u, -1, 1) * ACT_SCALE
    return s[0:2] + goal_rot(s).T @ u


def obs_bc(env, prev_obs):
    s, sim = env.state(), env.sim
    Rt = goal_rot(s)
    vT, vP = Rt @ sim.T.vel, Rt @ sim.pusher.vel
    extra = np.array([vT[0], vT[1], sim.T.omega, vP[0], vP[1], env.t / MAX_STEPS])
    return np.concatenate([prev_obs, make_obs(s), extra]).astype(np.float32)


def target_bc(env, a):
    return action_from_model(np.asarray(a, float), env.state())

# %% [markdown]
# ## 4. 병렬 시뮬레이션
#
# 우리 물리는 순수 Python이라 1 step에 수 ms가 걸립니다. PPO는 수백만 step이 필요하므로
# **16개 프로세스 × 프로세스당 4개 환경 = 64개 환경**을 동시에 돌립니다 (`multiprocessing`).
# 메인 프로세스가 64개 observation에 대해 신경망을 한 번에 계산해 action을 보내고, 각 프로세스가 물리를 진행해 결과를 돌려줍니다.
# 끝난 환경은 그 자리에서 새 랜덤 과제로 reset 됩니다.

# %%
def _worker(remote, seed, k, kind):
    rng = np.random.default_rng(seed)
    envs = [PushTEnv(max_steps=MAX_STEPS) for _ in range(k)]
    cost, prev = [0.0] * k, [None] * k

    def reset(i):
        envs[i].reset(int(rng.integers(RL_SEED0, 10**9)))
        cost[i] = task_cost(envs[i].sim)
        prev[i] = make_obs(envs[i].state())

    def obs(i):
        return obs_scratch(envs[i]) if kind == "scratch" else obs_bc(envs[i], prev[i])

    for i in range(k):
        reset(i)
    while True:
        cmd, data = remote.recv()
        if cmd == "obs":
            remote.send([obs(i) for i in range(k)])
        elif cmd == "step":
            out = []
            for i, env in enumerate(envs):
                prev[i] = make_obs(env.state())
                env.step(target_scratch(env, data[i]) if kind == "scratch" else target_bc(env, data[i]))
                c_after = task_cost(env.sim)
                success = env.is_success()
                r = reward(cost[i], c_after, env.sim, success)
                truncated = (not success) and env.t >= MAX_STEPS
                last = obs(i)
                cost[i] = c_after
                if success or truncated:
                    reset(i)
                out.append((obs(i), r, success, truncated, last))
            remote.send(out)
        else:
            break


class ParallelEnvs:
    def __init__(self, kind, n_procs=16, envs_per_proc=4, seed=0):
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        self.k, self.remotes = envs_per_proc, []
        for i in range(n_procs):
            a, b = ctx.Pipe()
            ctx.Process(target=_worker, args=(b, seed + 1000 * i, envs_per_proc, kind), daemon=True).start()
            self.remotes.append(a)
        self.n = n_procs * envs_per_proc

    def obs(self):
        for r in self.remotes:
            r.send(("obs", None))
        return np.array([x for r in self.remotes for x in r.recv()])

    def step(self, actions):
        for j, r in enumerate(self.remotes):
            r.send(("step", actions[j * self.k:(j + 1) * self.k]))
        o, rew, term, trunc, last = zip(*[x for r in self.remotes for x in r.recv()])
        return np.array(o), np.array(rew, np.float32), np.array(term), np.array(trunc), np.array(last)

    def close(self):
        for r in self.remotes:
            r.send(("close", None))

# %% [markdown]
# ## 5. Actor-critic 네트워크
#
# PPO의 policy는 Gaussian 입니다: $\ \pi_\phi(a\mid s)=\mathcal N\big(\mu_\phi(s),\ \operatorname{diag}(\sigma^2)\big)$.
# 학습 중에는 여기서 sampling해서 탐험하고, 평가·시연할 때는 평균 $\mu_\phi(s)$ 를 씁니다.
#
# - **A (scratch)** : actor $\mu_\phi$ 와 critic $V_\psi$ 모두 2층 256 tanh MLP, $\sigma$ 의 처음 값 $e^{-0.5}\approx0.6$ (크게 탐험).
# - **B (BC → RL)** : actor = MLP-BC 네트워크 (폭 512), $\sigma$ 의 처음 값 0.03 m (BC 행동 근처만 조금씩 탐험).
#
# critic 입력은 학습 중 모은 평균·분산으로 정규화합니다 (running mean/std).

# %%
class RunningNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("var", torch.ones(d))
        self.register_buffer("cnt", torch.tensor(1e-4))

    def norm(self, x):
        return ((x - self.mu) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)

    def update_norm(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d, tot = bm - self.mu, self.cnt + bc
        self.mu += d * bc / tot
        self.var = (self.var * self.cnt + bv * bc + d ** 2 * self.cnt * bc / tot) / tot
        self.cnt = tot


class ScratchActorCritic(RunningNorm):
    kind = "scratch"

    def __init__(self, d=13, h=256):
        super().__init__(d)
        self.pi = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 2))
        self.v = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def mean(self, o):
        return self.pi(self.norm(o))

    def value(self, o):
        return self.v(self.norm(o)).squeeze(-1)

    def actor_params(self):
        return list(self.pi.parameters()) + [self.log_std]


class BCActorCritic(RunningNorm):
    kind = "bc"

    def __init__(self, bc_state=None, width=512, log_std=math.log(0.03)):
        super().__init__(18)
        self.bc = BCPolicy(12, 16, width=width)
        for k, shape in (("x_mu", 12), ("x_sd", 12), ("y_mu", 16), ("y_sd", 16)):
            self.bc.register_buffer(k, bc_state[k] if bc_state is not None else torch.zeros(shape))
        if bc_state is not None:
            self.bc.load_state_dict(bc_state)
        self.log_std = nn.Parameter(torch.full((2,), log_std))
        self.v = nn.Sequential(nn.Linear(18, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))

    def mean(self, o):                                  # first action of the BC chunk, in metres
        return self.bc.dy(self.bc.f(self.bc.nx(o[:, :12])))[:, :2]

    def value(self, o):
        return self.v(self.norm(o)).squeeze(-1)

    def actor_params(self):
        return list(self.bc.parameters()) + [self.log_std]


def policy_dist(model, o):
    return torch.distributions.Normal(model.mean(o), model.log_std.exp())


def load_policy(path):
    sd = torch.load(path, map_location="cpu")
    model = BCActorCritic() if any(k.startswith("bc.") for k in sd) else ScratchActorCritic()
    model.load_state_dict(sd)
    return model.eval()

# %% [markdown]
# ## 6. PPO (Proximal Policy Optimization)
#
# 한 번의 반복:
#
# 1. **Rollout** : 64개 환경에서 128 step씩 = 8192개의 $(s_t,a_t,r_t)$ 수집.
# 2. **Advantage (GAE)** : "평소보다 얼마나 좋았나" 를 계산합니다.
#
# $$
# \delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\qquad
# \hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l},\qquad \gamma=0.99,\ \lambda=0.95
# $$
#
# 3. **Update** (10 epoch, minibatch) : 확률 비 $\rho=\pi_\phi(a\mid s)/\pi_{\text{old}}(a\mid s)$ 가 $[0.8,1.2]$ 를 벗어나면 더 밀지 않는 clipped objective
#
# $$
# L^{\text{actor}}=-\,\mathbb E\Big[\min\big(\rho\,\hat A,\ \operatorname{clip}(\rho,0.8,1.2)\,\hat A\big)\Big],\qquad
# L^{\text{critic}}=\mathbb E\big[(V_\psi(s)-\hat R)^2\big]
# $$
#
# clip 덕분에 한 번의 update로 policy가 너무 크게 바뀌지 않습니다. B에서는 이것이 특히 중요합니다 — 이미 잘하는 BC policy를 망가뜨리지 않아야 하므로
# actor learning rate도 $3\times10^{-5}$ 로 작게 둡니다 (A는 $3\times10^{-4}$).

# %%
def train_ppo(kind, total_steps, out_dir, actor_lr, critic_warmup=0, save_every=25, bc_ckpt=None):
    os.makedirs(out_dir, exist_ok=True)
    envs = ParallelEnvs(kind)
    if kind == "scratch":
        model = ScratchActorCritic()
    else:
        model = BCActorCritic(torch.load(bc_ckpt, map_location="cpu")["model"])
        torch.save(model.state_dict(), os.path.join(out_dir, "bcppo_0k.pt"))     # step 0 = pure imitation
    prefix = "ppo_scratch" if kind == "scratch" else "bcppo"
    opt_actor = torch.optim.Adam(model.actor_params(), lr=actor_lr)
    opt_critic = torch.optim.Adam(model.v.parameters(), lr=3e-4)
    gamma, lam, n_steps, batch = 0.99, 0.95, 128, 1024
    o = envs.obs()
    steps, it, t0, ep_success = 0, 0, time.time(), []
    log = open(os.path.join(out_dir, "train_log.txt"), "a")
    while steps < total_steps:
        O, A, LP, R, V, DONE = [], [], [], [], [], []
        torch.set_num_threads(1)                         # small batches: one thread is fastest
        for _ in range(n_steps):
            ot = torch.tensor(o)
            with torch.no_grad():
                d = policy_dist(model, ot)
                a = d.sample()
                lp, v = d.log_prob(a).sum(-1), model.value(ot)
            o, r, term, trunc, last = envs.step(a.numpy())
            r = r.copy()
            if trunc.any():                              # time limit: bootstrap with V(last state)
                with torch.no_grad():
                    r[trunc] += gamma * model.value(torch.tensor(last[trunc])).numpy()
            O.append(ot); A.append(a); LP.append(lp); R.append(torch.tensor(r)); V.append(v)
            DONE.append(torch.tensor(term | trunc, dtype=torch.float32))
            ep_success += [bool(term[i]) for i in np.where(term | trunc)[0]]
        torch.set_num_threads(4)

        with torch.no_grad():                            # GAE
            v_next = model.value(torch.tensor(o))
        adv, g = torch.zeros(n_steps, envs.n), torch.zeros(envs.n)
        for t in reversed(range(n_steps)):
            nv = v_next if t == n_steps - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - DONE[t]) - V[t]
            g = delta + gamma * lam * (1 - DONE[t]) * g
            adv[t] = g
        Ob, Ab, LPb = torch.cat(O), torch.cat(A), torch.cat(LP)
        ADV = adv.reshape(-1)
        RET = ADV + torch.cat(V)
        model.update_norm(Ob)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        for _ in range(10):                              # PPO epochs
            perm = torch.randperm(len(Ob))
            for i in range(0, len(Ob), batch):
                idx = perm[i:i + batch]
                lv = ((model.value(Ob[idx]) - RET[idx]) ** 2).mean()
                opt_critic.zero_grad(); lv.backward()
                nn.utils.clip_grad_norm_(model.v.parameters(), 0.5); opt_critic.step()
                if it >= critic_warmup:
                    ratio = (policy_dist(model, Ob[idx]).log_prob(Ab[idx]).sum(-1) - LPb[idx]).exp()
                    la = -torch.min(ratio * ADV[idx], ratio.clamp(0.8, 1.2) * ADV[idx]).mean()
                    opt_actor.zero_grad(); la.backward()
                    nn.utils.clip_grad_norm_(model.actor_params(), 0.5); opt_actor.step()
        steps += len(Ob)
        it += 1
        if it % 5 == 0:
            rate = np.mean(ep_success[-300:]) if ep_success else 0.0
            line = (f"steps {steps / 1e6:6.2f}M  episodes {len(ep_success):6d}  success(last 300) {rate:.2f}  "
                    f"std {model.log_std.exp().detach().numpy().round(3)}  {(time.time() - t0) / 60:.1f} min")
            print(line, flush=True)
            log.write(line + "\n"); log.flush()
        if it % save_every == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"{prefix}_{steps // 1000}k.pt"))
    torch.save(model.state_dict(), os.path.join(out_dir, f"{prefix}_final.pt"))
    envs.close()
    return model

# %% [markdown]
# ## 7. 학습된 policy를 agent로 — 평가와 실시간 창에서 사용
#
# expert, imitation agent와 같은 `reset()` / `act(state)` 인터페이스입니다. 평가와 시연에서는 탐험 noise 없이 평균 action을 씁니다.
# 속도 정보가 필요하므로 agent는 자신이 조종할 `env` 를 알고 있습니다.

# %%
class RLAgent:
    def __init__(self, env, model):
        self.env, self.model = env, model
        self.reset()

    def reset(self):
        self.prev = None

    @torch.no_grad()
    def act(self, state):
        env = self.env
        if self.model.kind == "scratch":
            o = obs_scratch(env)
            u = self.model.mean(torch.tensor(o[None]))[0].numpy()
            return target_scratch(env, u)
        cur = make_obs(state)
        o = obs_bc(env, cur if self.prev is None else self.prev)
        self.prev = cur
        a = self.model.mean(torch.tensor(o[None]))[0].numpy()
        return target_bc(env, a)


def evaluate(make_agent, n=50, label=""):
    env = PushTEnv(max_steps=MAX_STEPS)
    agent = make_agent(env)
    ok = [run_episode(env, agent, seed)[2] for seed in range(n)]
    print(f"{label:28s} success {np.mean(ok) * 100:5.1f} %  ({n} unseen tasks)")
    return float(np.mean(ok))


def checkpoints(folder, prefix):
    """Sorted [(steps, path)] of the checkpoints in `folder` (final = largest)."""
    out = []
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.startswith(prefix) and f.endswith(".pt"):
                tag = f[len(prefix) + 1:-3]
                k = 10**9 if tag == "final" else int(tag.rstrip("k")) if tag.rstrip("k").isdigit() else None
                if k is not None:
                    out.append((k, os.path.join(folder, f)))
    return sorted(out)

# %% [markdown]
# ## 8. 실행
#
# - `scratch` : A, 2천만 step 예산 (16코어 CPU에서 약 3시간 — 우리는 1,640만 step에서 멈췄습니다)
# - `bc` : B, 6백만 step (약 75분). `checkpoints/` 의 MLP-BC (`pusht_imitation.py` 의 결과) 가 필요합니다.
# - `eval` : 저장된 policy들을 평가 seed 0–49 에서 비교

# %%
BC_CKPT = os.path.join(CKPT_DIR, "MLP-BC_expert_180ep_28626st_w512_30000.pt")

if __name__ == "__main__" and len(sys.argv) > 1:
    mode = sys.argv[1]
    if mode == "scratch":
        train_ppo("scratch", 20e6, os.path.join(RL_DIR, "scratch"), actor_lr=3e-4)
    elif mode == "bc":
        assert os.path.exists(BC_CKPT), f"train MLP-BC first (pusht_imitation.py): {BC_CKPT}"
        train_ppo("bc", 6e6, os.path.join(RL_DIR, "bc_ppo"), actor_lr=3e-5, critic_warmup=10, bc_ckpt=BC_CKPT)
    elif mode == "eval":
        torch.set_num_threads(4)
        evaluate(lambda e: ScriptedExpert(e), label="script expert")
        for folder, prefix, name in (("scratch", "ppo_scratch", "RL from scratch"), ("bc_ppo", "bcppo", "BC -> RL")):
            ck = checkpoints(os.path.join(RL_DIR, folder), prefix)
            for k, path in ([ck[0], ck[len(ck) // 2], ck[-1]] if len(ck) >= 3 else ck):
                m = load_policy(path)
                evaluate(lambda e, m=m: RLAgent(e, m), label=f"{name} {os.path.basename(path)}")
