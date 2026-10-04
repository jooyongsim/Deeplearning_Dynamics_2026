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
# # Imitation Learning PushT — 사람의 시연으로 policy 학습하기
#
# 앞의 튜토리얼에서
#
# - `pusht_mini_sim.py` : 물리 (impulse, friction, restitution) 를 직접 구현했고
# - `pusht_keyboard_sim.py` : **사람이 policy 역할**을 하며 키보드로 T를 밀어 보았습니다.
#
# 이번에는 사람이 한 일을 **신경망이 따라 하도록** 만듭니다. 이것이 imitation learning (모방 학습) 입니다.
#
# $$
# \boxed{
# \underbrace{\text{사람 시연}}_{\mathcal D=\{(o_t,a_t)\}}
# \;\rightarrow\;
# \underbrace{\text{데이터셋}}_{(o,\ \mathbf a_{t:t+H})}
# \;\rightarrow\;
# \underbrace{\text{학습}}_{\pi_\phi(\mathbf a\mid o)}
# \;\rightarrow\;
# \underbrace{\text{closed-loop 평가}}_{\text{success rate}}
# }
# $$
#
# **이 튜토리얼에서 답할 두 가지 질문**
#
# 1. **시연(demonstration)을 몇 개나 모아야 하나?** → 6절 (문헌 기준) + 13절 (직접 실험)
# 2. **어떤 모델을 써야 하나?** → 8절 (multimodality 실험) + 9–12절 (MLP-BC vs Diffusion Policy)
#
# **전체 단계**
#
# | 절 | 내용 |
# |---|---|
# | 1–2 | 물리 불러오기, 랜덤 초기 위치 + 랜덤 target pose 환경 |
# | 3 | 시연 데이터 형식 (episode, `.npz`) |
# | 4 | **사람 시연 수집 GUI** (키보드 / 마우스) |
# | 5 | Scripted expert — 강의의 impulse 식으로 만든 "가상의 사람" (빠른 대량 수집용) |
# | 6 | 몇 개나 모아야 하나? |
# | 7 | 데이터셋: goal 좌표계, observation history, action chunk, 정규화 |
# | 8 | 어떤 모델? — multimodality 장난감 실험 |
# | 9–10 | Model 1: MLP Behavior Cloning, Model 2: Diffusion Policy |
# | 11–12 | 학습, closed-loop 평가 |
# | 13 | 데이터 수 vs 성공률 실험 |
# | 14 | 학습된 policy를 창에서 보기, 해 볼 것 |
#
# **실행 방법**
#
# - 전체 튜토리얼: `python pusht_imitation.py` (그림은 `figures_imitation/` 에 저장)
# - **사람 시연 수집만**: `python pusht_imitation.py collect` → 키보드 창이 열림
# - VS Code Interactive / Jupyter: 위에서부터 셀 실행. 4절의 `COLLECT_HUMAN = True` 로 바꾸면 수집 창이 열립니다.
# - notebook으로 바꾸기: `python ../tools/py2ipynb.py pusht_imitation.py`
#
# 필요한 것: `numpy`, `matplotlib`, `torch` (CPU로 충분, GPU가 있으면 자동 사용), 그리고 같은 폴더의 `pusht_keyboard_sim.py`.

# %%
import contextlib
import io
import math
import os
import sys
import time

import numpy as np

try:
    get_ipython()                                   # noqa: F821  (Jupyter / VS Code Interactive)
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, HERE)
MODE = sys.argv[1] if (not IN_NOTEBOOK and len(sys.argv) > 1) else "tutorial"   # "tutorial" | "collect"
NO_GUI = bool(os.environ.get("PUSHT_NO_GUI"))

FIG_DIR = os.path.join(HERE, "figures_imitation")
DEMO_DIR = os.path.join(HERE, "demos")
CKPT_DIR = os.path.join(HERE, "checkpoints")
for d in (FIG_DIR, DEMO_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

import matplotlib
if not IN_NOTEBOOK and MODE == "tutorial":
    matplotlib.use("Agg")                           # script: save figures instead of blocking windows
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def show(fig, name):
    """Notebook: draw inline.  Script: save to figures_imitation/<name>.png."""
    if IN_NOTEBOOK:
        plt.show()
    else:
        path = os.path.join(FIG_DIR, name + ".png")
        fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print("figure ->", os.path.relpath(path, HERE))

# %% [markdown]
# ## 1. 물리 불러오기 — `pusht_keyboard_sim.py` 를 그대로 재사용
#
# 키보드 튜토리얼의 `KeyboardPushT` (PD pusher + impulse contact + 벽) 를 import 합니다.
# import 할 때 그 파일이 창을 열지 않도록 `PUSHT_NO_GUI=1` 을 잠시 켭니다.
#
# **속도 개선 한 가지.** 데이터를 수백 episode 만들고 평가하려면 물리를 수십만 번 돌려야 합니다.
# 가장 오래 걸리는 곳은 `circle_vs_T()` 인데, T의 8개 변에 대해 Python `for` 문으로
#
# $$
# t_i=\operatorname{clip}\!\left(\frac{(\mathbf c-\mathbf a_i)\cdot(\mathbf b_i-\mathbf a_i)}{\lVert\mathbf b_i-\mathbf a_i\rVert^2},\,0,\,1\right),\qquad
# \mathbf q_i=\mathbf a_i+t_i(\mathbf b_i-\mathbf a_i),\qquad
# i^{*}=\arg\min_i\lVert\mathbf q_i-\mathbf c\rVert
# $$
#
# 를 계산합니다. **같은 식**을 numpy 배열 연산으로 8개 변을 한 번에 계산하도록 바꾸고 (결과는 동일), 모듈의 함수를 교체합니다.

# %%
_had_flag = "PUSHT_NO_GUI" in os.environ
os.environ["PUSHT_NO_GUI"] = "1"                    # import the physics only, no window
with contextlib.redirect_stdout(io.StringIO()):     # silence that file's own self-check prints
    import pusht_keyboard_sim as ksim
if not _had_flag:
    del os.environ["PUSHT_NO_GUI"]

from pusht_keyboard_sim import ARENA, KEYS, KeyboardDemo, KeyboardPushT, cross2, rotation_matrix

_circle_vs_T_loop = ksim.circle_vs_T


def circle_vs_T_fast(circle_pos, radius, body_pos, body_angle, vertices):
    """Same as pusht_keyboard_sim.circle_vs_T, all 8 edges at once with numpy."""
    R = rotation_matrix(body_angle)
    c = R.T @ (circle_pos - body_pos)                         # world -> T local frame
    A = vertices
    AB = np.roll(vertices, -1, axis=0) - A
    t = np.clip(((c - A) * AB).sum(1) / (AB * AB).sum(1), 0.0, 1.0)
    Q = A + t[:, None] * AB                                   # closest point on every edge
    d2 = ((Q - c) ** 2).sum(1)
    i = int(np.argmin(d2))
    d = math.sqrt(d2[i])

    xi, yi = A[:, 0], A[:, 1]                                 # ray casting, vectorised
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    crosses = (yi > c[1]) != (yj > c[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        x_hit = (xj - xi) * (c[1] - yi) / (yj - yi) + xi
    inside = bool(np.count_nonzero(crosses & (c[0] < x_hit)) % 2)
    if not inside and d >= radius:
        return None

    direction = (Q[i] - c) / d if d > 1e-12 else np.array([1.0, 0.0])
    n_local, pen = (-direction, radius + d) if inside else (direction, radius - d)
    n = R @ n_local
    n /= np.linalg.norm(n) + 1e-12
    return body_pos + R @ Q[i], n, pen


# check: identical to the loop version on random configurations
_rng = np.random.default_rng(0)
_V = KeyboardPushT().T.vertices
for _ in range(2000):
    args = (_rng.uniform(-2, 2, 2), 0.18, _rng.uniform(-1, 1, 2), _rng.uniform(-4, 4), _V)
    r_loop, r_fast = _circle_vs_T_loop(*args), circle_vs_T_fast(*args)
    assert (r_loop is None) == (r_fast is None)
    assert r_loop is None or all(np.allclose(x, y) for x, y in zip(r_loop, r_fast))

ksim.circle_vs_T = circle_vs_T_fast                   # KeyboardPushT.substep() now uses the fast one
circle_vs_T = circle_vs_T_fast
print("circle_vs_T_fast == circle_vs_T on 2000 random cases -> patched")

# %% [markdown]
# ## 2. 과제: 랜덤 초기 위치 + 랜덤 target pose
#
# 키보드 튜토리얼에서는 T와 goal이 항상 같은 곳에서 시작했습니다. 하나의 장면만 외운 policy는 쓸모가 없으므로,
# episode마다 다음을 **무작위로** 뽑습니다.
#
# | 무엇 | 분포 |
# |---|---|
# | T 초기 pose $(x_G,y_G,\theta)$ | $x\sim\mathcal U(-2,2),\ y\sim\mathcal U(-1,1),\ \theta\sim\mathcal U(-\pi,\pi)$ |
# | goal pose $(x_g,y_g,\theta_g)$ | 같은 분포 (너무 쉬운 경우 — 거리 $<1$ **이고** 각도 차 $<45^\circ$ — 는 다시 뽑음) |
# | pusher 초기 위치 | arena 안 균일, T의 COM에서 $1.6$ 이상 떨어진 곳 |
#
# 범위를 arena ($8\times6$) 보다 좁게 잡은 이유: T의 크기가 $1.8$ 이라, goal이 벽에 붙어 있으면 pusher가 T와 벽 사이로 들어갈 수 없는 경우가 생깁니다.
#
# **Seed** 하나가 한 episode의 초기 조건을 완전히 결정합니다. 그래서
#
# - 시연 수집 seed : `100000 +` (expert), `200000 +` (사람)
# - **평가 seed : `0, 1, 2, …`** (학습에 한 번도 나오지 않은 초기 조건)
#
# 로 나눠 두면 "처음 보는 상황에서도 되는가" 를 공정하게 잴 수 있습니다.
#
# **Control loop.** policy는 10 Hz로 action을 냅니다. action $a_t$ = pusher의 **target 위치** $(x^{*},y^{*})$ (키보드 튜토리얼의 × 표시).
# `step(a)` 한 번 = target을 $a$ 로 두고 물리를 $\Delta t_{\text{control}}=0.1$ s (substep 20번) 진행.
#
# **State** (8차원, 사람이 화면에서 보는 것과 같은 정보):
#
# $$
# s_t=\big(\underbrace{x_P,\,y_P}_{\text{pusher}},\ \underbrace{x_G,\,y_G,\,\theta}_{\text{T}},\ \underbrace{x_g,\,y_g,\,\theta_g}_{\text{goal}}\big)
# $$
#
# **성공 조건** (키보드 튜토리얼과 같음): $\ \lVert\mathbf x_G-\mathbf x_g\rVert<0.15\ $ 이고 $\ \lvert\theta-\theta_g\rvert<10^\circ$.

# %%
STATE_NAMES = ["pusher_x", "pusher_y", "T_x", "T_y", "T_theta", "goal_x", "goal_y", "goal_theta"]
EXPERT_SEED0, HUMAN_SEED0 = 100_000, 200_000


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PushTEnv:
    """KeyboardPushT + random initial / goal poses + a 10 Hz control step."""
    control_dt = 0.1

    def __init__(self, max_steps=300, pos_tol=0.15, ang_tol_deg=10.0):
        self.sim = KeyboardPushT()
        self.max_steps, self.pos_tol, self.ang_tol = max_steps, pos_tol, ang_tol_deg
        self.reset(0)

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        sim = self.sim
        sim.reset()

        def pose():
            return np.array([rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.0)]), rng.uniform(-math.pi, math.pi)

        while True:                                   # reject tasks that are already almost solved
            p, a = pose()
            g, ga = pose()
            if np.linalg.norm(p - g) > 1.0 or abs(wrap_angle(a - ga)) > math.radians(45):
                break
        while True:                                   # pusher somewhere away from the T
            q = np.array([rng.uniform(-3.6, 3.6), rng.uniform(-2.6, 2.6)])
            if np.linalg.norm(q - p) > 1.6:
                break
        sim.T.pos[:], sim.T.angle = p, a
        sim.goal_pos, sim.goal_angle = g, ga
        sim.pusher.pos[:] = q
        sim.target = q.copy()
        self.seed, self.t = seed, 0
        return self.state()

    def state(self):
        s = self.sim
        return np.array([*s.pusher.pos, *s.T.pos, s.T.angle, *s.goal_pos, s.goal_angle])

    def is_success(self):
        d, a = self.sim.goal_error()
        return d < self.pos_tol and a < self.ang_tol

    def step(self, action):
        x_min, x_max, y_min, y_max = ARENA
        self.sim.target = np.clip(np.asarray(action, dtype=float), [x_min, y_min], [x_max, y_max])
        self.sim.advance(self.control_dt)
        self.t += 1
        success = self.is_success()
        return self.state(), success, success or self.t >= self.max_steps


def T_outline(pos, angle, vertices):
    return pos + vertices @ rotation_matrix(angle).T


def draw_task(ax, env, state=None, color="#2774B8", alpha=1.0, goal=True):
    s = env.state() if state is None else state
    V = env.sim.T.vertices
    x_min, x_max, y_min, y_max = ARENA
    if goal:
        ax.add_patch(Polygon([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                             closed=True, fill=False, ec="#404040", lw=2))
        ax.add_patch(Polygon(T_outline(s[5:7], s[7], V), closed=True, fill=False, ec="#3D8C54", lw=2, ls="--"))
    ax.add_patch(Polygon(T_outline(s[2:4], s[4], V), closed=True, fc="#D9EEFC", ec=color, lw=1.5, alpha=alpha))
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


env = PushTEnv()
fig, axes = plt.subplots(2, 4, figsize=(16, 6.4))
for ax, seed in zip(axes.flat, range(8)):
    s = env.reset(seed)
    draw_task(ax, env)
    ax.add_patch(plt.Circle(s[0:2], env.sim.pusher.radius, fc="#FFEBC9", ec="#E97820", lw=2))
    d, a = env.sim.goal_error()
    ax.set_title(f"seed {seed}: {d:.2f} m, {a:.0f} deg to go", fontsize=10)
fig.suptitle("Random initial T (blue), random goal (green dashed), pusher (orange)")
show(fig, "01_random_tasks")

# %% [markdown]
# ## 3. 시연 데이터의 형식
#
# 한 episode는 길이 $T_e$ 의 state–action 쌍입니다.
#
# $$
# \tau=\big((s_0,a_0),(s_1,a_1),\dots,(s_{T_e-1},a_{T_e-1})\big)
# $$
#
# 여러 episode를 하나로 이어 붙이고, 각 episode가 어디서 끝나는지 `episode_ends` 로 기록합니다.
# (Diffusion Policy / LeRobot의 PushT 데이터셋과 같은 방식.)
#
# ```text
# states   (N_total, 8)    s_t
# actions  (N_total, 2)    a_t = target (x*, y*)
# episode_ends (E,)        e.g. [143, 301, 455, ...]   -> episode k = rows [ends[k-1], ends[k])
# seeds    (E,)            초기 조건을 다시 만들 수 있도록
# ```
#
# **성공한 episode만 저장합니다.** 실패한 시연을 따라 하면 실패를 배우기 때문입니다 (filtered behavior cloning).

# %%
def save_demos(path, episodes, seeds):
    """episodes: list of (states (T,8), actions (T,2))."""
    np.savez_compressed(
        path,
        states=np.concatenate([S for S, _ in episodes]).astype(np.float32),
        actions=np.concatenate([A for _, A in episodes]).astype(np.float32),
        episode_ends=np.cumsum([len(S) for S, _ in episodes]),
        seeds=np.asarray(seeds),
    )


def load_demos(path):
    z = np.load(path)
    starts = np.r_[0, z["episode_ends"][:-1]]
    episodes = [(z["states"][a:b].astype(float), z["actions"][a:b].astype(float))
                for a, b in zip(starts, z["episode_ends"])]
    return episodes, list(z["seeds"])


def run_episode(env, agent, seed):
    """Roll out `agent` (anything with reset() and act(state) -> target) from `seed`."""
    s = env.reset(seed)
    agent.reset()
    S, A = [], []
    while True:
        a = agent.act(s)
        S.append(s); A.append(np.asarray(a, dtype=float))
        s, success, done = env.step(a)
        if done:
            return np.array(S), np.array(A), success

# %% [markdown]
# ## 4. 사람 시연 수집 GUI
#
# 키보드 튜토리얼의 `KeyboardDemo` 를 상속해서, 화면은 그대로 쓰고 **기록 기능**만 더합니다.
#
# | 키 | 동작 |
# |---|---|
# | 방향키 / W A S D, 마우스 드래그 | target 이동 (키보드 튜토리얼과 같음) |
# | (goal 도달) | **자동 저장** → 1초 뒤 새 랜덤 episode |
# | N | 이번 episode 버리고 새 랜덤 episode |
# | R | 같은 초기 조건으로 다시 (지금까지 기록 버림) |
# | Backspace | 마지막으로 저장한 episode 삭제 (실수했을 때) |
# | Space | 일시정지 |
# | Q / Esc | 종료 (저장은 매 episode마다 이미 됨) |
#
# **데이터 품질을 위한 세 가지 설계**
#
# 1. **고정된 10 Hz 기록.** 화면은 20 fps (0.05 s) 로 그리지만, target은 0.1 s마다 한 번만 바꾸고 그 순간 $(s_t,a_t)$ 를 기록합니다.
#    그래서 사람 데이터의 한 step이 `PushTEnv.step()` 의 한 step과 **정확히 같은 의미**가 됩니다.
#    (실제 시간 대신 고정 시간으로 진행 — 컴퓨터가 느려도 데이터는 같음.)
# 2. **처음 움직이기 전의 정지 구간은 기록하지 않음.** 사람이 화면을 보고 생각하는 동안의 "가만히 있기" 를 배우면,
#    policy도 시작할 때 가만히 서 있게 됩니다.
# 3. **F / E (마찰, 반발계수 전환) 는 끔.** 물리 파라미터가 시연마다 다르면 같은 상황에서 다른 행동이 나와 학습이 어려워집니다.
#
# 저장 위치: `demos/human_demos.npz` (episode가 성공할 때마다 전체 파일을 다시 저장 — 중간에 꺼도 안전).
# 다시 실행하면 기존 파일을 불러와 **이어서** 모읍니다.

# %%
HUMAN_PATH = os.path.join(DEMO_DIR, "human_demos.npz")


class DemoRecorder(KeyboardDemo):

    frame_dt = 0.05                                 # draw at 20 fps, record every 2nd frame (10 Hz)

    def __init__(self, path=HUMAN_PATH):
        self.env = PushTEnv(max_steps=10**9)
        super().__init__(self.env.sim, target_speed=0.0)   # target moved here, once per control step
        self.speed = 1.5                            # m/s while a key is held (same as the keyboard demo)
        self.path = path
        self.episodes, self.seeds = load_demos(path) if os.path.exists(path) else ([], [])
        self.rng = np.random.default_rng()
        self.mouse = None
        self.message, self.message_until = "", 0.0
        self.new_episode()

    # ------------------------------------------------------------ episodes
    def new_episode(self, seed=None):
        self.seed = int(HUMAN_SEED0 + self.rng.integers(10**6)) if seed is None else seed
        self.env.reset(self.seed)
        self.goal_patch.set_xy(self._world(self.sim.goal_pos, self.sim.goal_angle))
        self.S, self.A = [], []
        self.started, self.frame, self.restart_at = False, 0, None

    def flash(self, text, seconds=1.5):
        self.message, self.message_until = text, time.perf_counter() + seconds

    def save(self):
        if self.episodes:
            save_demos(self.path, self.episodes, self.seeds)
        elif os.path.exists(self.path):
            os.remove(self.path)

    # ------------------------------------------------------------- events
    def on_press(self, event):
        key = (event.key or "").lower()
        if key in ("f", "e", "r"):                  # physics stays fixed while recording; R = restart
            if key == "r":
                self.new_episode(self.seed)
                self.flash("restart (same seed)")
            return
        if key == "n":
            self.new_episode()
            self.flash("skipped -> new random episode")
        elif key == "backspace" and self.episodes:
            self.episodes.pop(); self.seeds.pop()
            self.save()
            self.flash(f"deleted last demo -> {len(self.episodes)} left")
        else:
            super().on_press(event)                 # arrows/WASD, space, q/esc

    def on_motion(self, event):
        if self.dragging and event.inaxes is self.ax and event.xdata is not None:
            self.mouse = np.array([event.xdata, event.ydata])

    def on_button_release(self, event):
        super().on_button_release(event)
        self.mouse = None

    # -------------------------------------------------------------- frame
    def control_tick(self):
        """Every 0.1 s: read the human input -> new target -> record (s_t, a_t)."""
        now = time.perf_counter()
        move = np.zeros(2)
        for key, d in KEYS.items():
            if self._held(key, now):
                move += d
        target = self.sim.target.copy()
        if self.mouse is not None:
            step = self.mouse - target
            n = np.linalg.norm(step)
            target = target + (step if n < 0.3 else step / n * 0.3)     # pusher can't follow faster anyway
        elif move.any():
            target = target + self.speed * self.env.control_dt * move / np.linalg.norm(move)
        x_min, x_max, y_min, y_max = ARENA
        target = np.clip(target, [x_min, y_min], [x_max, y_max])

        if not self.started and np.linalg.norm(target - self.sim.target) > 1e-9:
            self.started = True                     # skip the idle "thinking" time before the first move
        if self.started:
            self.S.append(self.env.state())
            self.A.append(target.copy())
        self.sim.target = target

    def update(self, _frame=None, frame_dt=None):
        if not self.paused and self.restart_at is None:
            if self.frame % 2 == 0:
                self.control_tick()
            self.frame += 1
        if self.restart_at is not None and time.perf_counter() > self.restart_at:
            self.new_episode()
        out = super().update(frame_dt=self.frame_dt if self.restart_at is None else 0.0)

        if self.restart_at is None and self.started and self.env.is_success() and self.frame % 2 == 0:
            self.episodes.append((np.array(self.S), np.array(self.A)))
            self.seeds.append(self.seed)
            self.save()
            self.flash(f"saved demo #{len(self.episodes)}  ({len(self.S)} steps, {len(self.S) * 0.1:.1f} s)")
            self.restart_at = time.perf_counter() + 1.0

        n_steps = sum(len(S) for S, _ in self.episodes)
        status = self.message if time.perf_counter() < self.message_until else \
            f"recording {len(self.S) * 0.1:5.1f} s" if self.started else "move to start recording"
        self.title.set_text(f"demos: {len(self.episodes)}  ({n_steps} steps)   |   {status}")
        self.title.set_color("#3D8C54" if "saved" in status else "#203864")
        return out


def open_recorder():
    if IN_NOTEBOOK:
        get_ipython().run_line_magic("matplotlib", "tk")    # noqa: F821  separate window takes keys
    else:
        matplotlib.use("TkAgg", force=True)
    rec = DemoRecorder()
    rec.fig.texts[-1].set_text("\n".join([
        "arrows / WASD : move target", "mouse drag    : target", "goal reached  : auto-save",
        "N             : new episode", "R             : restart", "Backspace     : delete last",
        "space         : pause", "Q / Esc       : quit"]))
    rec.run()
    return rec


COLLECT_HUMAN = False          # True -> open the recording window here (notebook)

if MODE == "collect":
    open_recorder()
    raise SystemExit
if COLLECT_HUMAN and not NO_GUI:
    recorder = open_recorder()

# %% [markdown]
# > **Notebook에서 수집했다면**: 창을 닫은 뒤 다음 셀에서 `%matplotlib inline` 을 실행해 그림을 다시 notebook 안에 그리게 하세요.

# %%
if COLLECT_HUMAN and IN_NOTEBOOK:
    get_ipython().run_line_magic("matplotlib", "inline")   # noqa: F821

# %% [markdown]
# ## 5. Scripted expert — 강의의 impulse 식으로 만든 "가상의 사람"
#
# 사람 시연 200개를 모으려면 1–2시간이 걸립니다. 수업 시간에 학습 과정을 바로 보고, 또 "데이터가 몇 개 필요한가" 를 실험하려면
# **시연을 자동으로 만들어 주는 expert** 가 있으면 편합니다. 아래 expert는 사람처럼 화면의 정보(state)를 보고 target을 움직이며,
# 판단에는 강의에서 유도한 **impulse–momentum 식**을 씁니다.
#
# **(1) 어디를 밀까?** T 둘레의 후보점 $C_k$ (8개 변 × 3점) 에서 안쪽 법선 $\mathbf n_k$ 방향으로 크기 1의 impulse를 준다고 하면
#
# $$
# \Delta\mathbf v_G=\frac{\mathbf n_k}{m},\qquad
# \Delta\omega=\frac{\mathbf r_k\times\mathbf n_k}{I}
# $$
#
# 원하는 움직임은 goal 쪽으로의 이동과 회전입니다:
#
# $$
# \mathbf d=\big(x_g-x_G,\ y_g-y_G,\ L\,(\theta_g-\theta)\big),\qquad
# \mathbf c_k=\Big(\frac{\mathbf n_k}{m},\ L\,\frac{\mathbf r_k\times\mathbf n_k}{I}\Big)
# $$
#
# ($L=0.8$ : 각도와 길이의 단위를 맞추는 길이 scale). 두 벡터의 방향이 가장 비슷한 후보를 고릅니다.
#
# $$
# k^{*}=\arg\max_k\ \frac{\mathbf c_k\cdot\mathbf d}{\lVert\mathbf c_k\rVert\,\lVert\mathbf d\rVert}
# $$
#
# 여기에 몇 가지 상식을 더합니다: 접촉 중인 벽 쪽으로 미는 후보는 감점, 지금 밀던 후보는 가산점 (자꾸 바꾸지 않게),
# pusher가 들어갈 자리가 없는 후보 (벽 밖, T 안) 는 제외.
#
# **(2) 어떻게 갈까?** 고른 접촉점 바로 바깥 (standoff) 으로 이동합니다. 직선 경로가 T에 막히면 T 둘레를 돌아서 (orbit) 갑니다 —
# 시계/반시계 중 arena 안에 머무는 짧은 쪽.
#
# **(3) 밀기.** 접촉점 뒤에 도착하면 $-\mathbf n$... 즉 $\mathbf n_k$ 방향으로 target을 옮깁니다. 속도는 남은 오차에 비례하게 줄입니다:
# T는 damping $e^{-c_v t}$ 로만 멈추므로, $v$ 로 밀다 놓으면 $\approx v/c_v$ 만큼 더 미끄러지기 때문입니다.
#
# 이 expert는 완벽하지 않습니다 (약 90% 성공). 사람도 가끔 실패하고, 실패한 시연은 저장하지 않으므로 상관없습니다.

# %%
class ScriptedExpert:
    L = 0.8                                          # length scale that makes rad comparable to m

    def __init__(self, env):
        self.env = env
        sim = env.sim
        V = sim.T.vertices
        self.cands = []                              # (local point, local outward normal)
        for i in range(len(V)):
            a, b = V[i], V[(i + 1) % len(V)]
            d = b - a
            out = np.array([-d[1], d[0]]) / np.linalg.norm(d)      # vertices are clockwise
            for f in (0.15, 0.5, 0.85):
                self.cands.append((a + f * d, out))
        # clearance ring around the T in its own frame: how far out the pusher must stay at each angle
        rp = sim.pusher.radius
        self.phis = np.linspace(-math.pi, math.pi, 73)[:-1]
        self.rhos = []
        for phi in self.phis:
            u = np.array([math.cos(phi), math.sin(phi)])
            rho = 0.4
            while circle_vs_T(rho * u, rp + 0.04, np.zeros(2), 0.0, V) is not None:
                rho += 0.05
            self.rhos.append(rho + 0.15)
        self.rhos = np.array(self.rhos)
        self.reset()

    def reset(self):
        self.cur = None

    def act(self, state):
        sim = self.env.sim
        T, P = sim.T, sim.pusher
        R = rotation_matrix(T.angle)
        rp = P.radius
        x_min, x_max, y_min, y_max = ARENA

        e = sim.goal_pos - T.pos
        d = np.array([e[0], e[1], self.L * wrap_angle(sim.goal_angle - T.angle)])
        d_norm = np.linalg.norm(d)

        world = T.pos + T.vertices @ R.T             # walls the T is touching (inward normals)
        walls = [n for gap, n in ((world[:, 0].min() - x_min, (1, 0)), (x_max - world[:, 0].max(), (-1, 0)),
                                  (world[:, 1].min() - y_min, (0, 1)), (y_max - world[:, 1].max(), (0, -1)))
                 if gap < 0.3]

        # (1) choose the contact point
        best = None
        for k, (c_local, out_local) in enumerate(self.cands):
            cw, out = T.pos + R @ c_local, R @ out_local
            n_in, r = -out, cw - T.pos
            c = np.array([n_in[0] / T.mass, n_in[1] / T.mass, self.L * cross2(r, n_in) / T.inertia])
            score = c @ d / (np.linalg.norm(c) * d_norm + 1e-9)
            stand = cw + out * (rp + 0.12)
            if not (x_min + rp < stand[0] < x_max - rp and y_min + rp < stand[1] < y_max - rp):
                continue
            if circle_vs_T(stand, rp, T.pos, T.angle, T.vertices) is not None:
                continue
            for n_wall in walls:
                if n_in @ np.array(n_wall) < -0.3:
                    score -= 0.6                     # would push the T further into that wall
            if k == self.cur:
                score += 0.1                         # hysteresis
            if best is None or score > best[0]:
                best = (score, k, cw, out, stand)
        if best is None:
            return sim.target.copy()
        _, self.cur, cw, out, stand = best

        # (3) behind the contact point -> push along n_in, slower when close to the goal
        rel = P.pos - cw
        along, lateral = rel @ out, abs(rel @ np.array([-out[1], out[0]]))
        if lateral < 0.12 and 0.0 < along < rp + 0.25:
            step = float(np.clip(0.3 * d_norm, 0.04, 0.12))
            return P.pos - out * (step + 0.04)

        # (2) otherwise travel to the standoff point: straight if clear, else around the T
        goal_t = stand if self._clear(P.pos, stand) else self._orbit(P.pos, stand)
        delta = goal_t - sim.target
        n = np.linalg.norm(delta)
        return sim.target + (delta if n < 0.2 else delta / n * 0.2)

    # ---------------------------------------------------------------- path helpers
    def _ring(self, phi):
        T = self.env.sim.T
        rho = np.interp(wrap_angle(phi - T.angle), self.phis, self.rhos, period=2 * math.pi)
        return T.pos + rho * np.array([math.cos(phi), math.sin(phi)])

    def _inside_arena(self, p):
        x_min, x_max, y_min, y_max = ARENA
        rp = self.env.sim.pusher.radius
        return x_min + rp <= p[0] <= x_max - rp and y_min + rp <= p[1] <= y_max - rp

    def _orbit(self, p, stand):
        T = self.env.sim.T
        a0 = math.atan2(*(p - T.pos)[::-1])
        a1 = math.atan2(*(stand - T.pos)[::-1])
        here = self._ring(a0)
        if np.linalg.norm(p - T.pos) < np.linalg.norm(here - T.pos) - 0.1:
            return here                              # first step out, away from the T
        best = None
        for sgn in (1, -1):                          # counter-clockwise / clockwise
            arc = (sgn * (a1 - a0)) % (2 * math.pi)
            pts = [self._ring(a0 + sgn * f) for f in np.arange(0.25, arc, 0.25)]
            if all(self._inside_arena(q) for q in pts) and (best is None or arc < best[0]):
                best = (arc, pts[0] if pts else stand)
        return best[1] if best else here

    def _clear(self, a, b):
        T, rp = self.env.sim.T, self.env.sim.pusher.radius
        return all(circle_vs_T(a + f * (b - a), rp + 0.05, T.pos, T.angle, T.vertices) is None
                   for f in np.linspace(0, 1, 12))


expert = ScriptedExpert(env)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, seed in zip(axes, (0, 1, 2)):
    S, A, ok = run_episode(env, expert, seed)
    draw_task(ax, env, S[0])
    for k in range(0, len(S), 25):
        draw_task(ax, env, S[k], color=plt.cm.viridis(k / len(S)), alpha=0.5, goal=False)
    draw_task(ax, env, env.state(), color="#E97820", goal=False)
    ax.plot(S[:, 0], S[:, 1], "-", color="#E97820", lw=0.8)
    ax.set_title(f"seed {seed}: {'success' if ok else 'fail'} in {len(S)} steps ({len(S) * 0.1:.1f} s)", fontsize=10)
fig.suptitle("Scripted expert (orange line = pusher path, colour = time)")
show(fig, "02_expert_rollouts")

# %% [markdown]
# ### Expert 시연 대량 수집
#
# 학습용 seed (`100000 +`) 에서 expert를 돌려 **성공한 episode만** 저장합니다. 결과는 `demos/expert_demos.npz` 로 저장되어, 다음 실행부터는 바로 불러옵니다.
#
# 같은 expert를 평가 seed (`0, 1, …`) 에서 돌린 성공률은 학습된 policy가 도달할 수 있는 **대략의 상한선**입니다.

# %%
N_EXPERT_DEMOS = 30
EXPERT_PATH = os.path.join(DEMO_DIR, "expert_demos.npz")


def collect_expert_demos(n, path=EXPERT_PATH):
    env_c = PushTEnv(max_steps=400)
    exp = ScriptedExpert(env_c)
    episodes, seeds, tried = [], [], 0
    t0 = time.perf_counter()
    while len(episodes) < n:
        seed = EXPERT_SEED0 + tried
        tried += 1
        S, A, ok = run_episode(env_c, exp, seed)
        if ok:
            episodes.append((S, A)); seeds.append(seed)
        if tried % 50 == 0:
            print(f"  tried {tried:4d}, kept {len(episodes):4d}   ({time.perf_counter() - t0:.0f} s)")
    save_demos(path, episodes, seeds)
    print(f"kept {len(episodes)} / {tried} episodes -> {os.path.relpath(path, HERE)}")
    return episodes, seeds


if os.path.exists(EXPERT_PATH) and len(load_demos(EXPERT_PATH)[0]) >= N_EXPERT_DEMOS:
    expert_episodes, expert_seeds = load_demos(EXPERT_PATH)
    print(f"loaded {len(expert_episodes)} expert demos from {os.path.relpath(EXPERT_PATH, HERE)}")
else:
    expert_episodes, expert_seeds = collect_expert_demos(N_EXPERT_DEMOS)

lengths = np.array([len(S) for S, _ in expert_episodes])
print(f"episode length: mean {lengths.mean():.0f} steps ({lengths.mean() * 0.1:.1f} s), "
      f"min {lengths.min()}, max {lengths.max()},  total {lengths.sum()} (s, a) pairs")

# %% [markdown]
# ### 어떤 데이터로 학습할까?
#
# `DATA_SOURCE`
#
# - `"human"` : 4절에서 모은 사람 시연 (`demos/human_demos.npz`)
# - `"expert"` : 위의 scripted expert
# - `"auto"` : 사람 시연이 `MIN_HUMAN_DEMOS` 개 이상 있으면 사람, 아니면 expert
#
# 사람 시연이 적을 때 둘을 섞는 것도 가능하지만 (`"both"`), 두 시연자의 스타일이 다르면 같은 상황에서 다른 행동이 섞여 **multimodality가 커집니다** (8절).

# %%
DATA_SOURCE = "auto"
MIN_HUMAN_DEMOS = 50

human_episodes = load_demos(HUMAN_PATH)[0] if os.path.exists(HUMAN_PATH) else []
print(f"human demos available: {len(human_episodes)}")
if DATA_SOURCE == "auto":
    DATA_SOURCE = "human" if len(human_episodes) >= MIN_HUMAN_DEMOS else "expert"
episodes = {"human": human_episodes, "expert": expert_episodes,
            "both": human_episodes + expert_episodes}[DATA_SOURCE]
print(f"training data: {DATA_SOURCE}, {len(episodes)} episodes, {sum(len(S) for S, _ in episodes)} steps")

# %% [markdown]
# ## 6. 시연을 몇 개나 모아야 하나?
#
# 정답은 하나가 아니고 **과제의 다양성 × 원하는 성공률 × 모델** 에 따라 달라집니다. 참고할 숫자:
#
# | 출처 | 과제 | 시연 수 |
# |---|---|---|
# | Diffusion Policy (Chi et al., 2023) | PushT, **goal 고정**, 초기 위치만 랜덤 | 사람 시연 200개 (약 2.5만 step) |
# | LeRobot `pusht` 데이터셋 | 같은 과제 | 206 episode, 25,650 frame |
# | ACT / ALOHA (Zhao et al., 2023) | 실제 양팔 로봇 조작 | 과제당 50개 |
#
# 우리 과제는 **goal까지 랜덤**이라 원래 PushT보다 어렵습니다. 다만 7절의 goal 좌표계 변환으로 대부분을 "goal 고정 문제" 로 되돌릴 수 있습니다.
#
# **권장 계획 (사람 시연, 한 개에 15–30 s)**
#
# | 단계 | 개수 | 시간 | 목적 |
# |---|---|---|---|
# | 파이프라인 확인 | 10–20 | 10분 | 저장/불러오기/학습 코드가 도는지, loss가 내려가는지 |
# | 첫 결과 | 50–100 | 30–60분 | 성공이 나오기 시작. 실패 양상 관찰 |
# | 쓸 만한 policy | 200 이상 | 1.5–2시간 | Diffusion Policy 논문 수준의 데이터 양 |
#
# 개수보다 중요한 것: **초기 조건이 골고루 퍼져 있을 것** (seed가 랜덤이므로 자동), **같은 상황에서 일관된 전략**, **실패 시연은 버리기**.
#
# "정말 그만큼 필요한가?" 는 13절에서 시연 수를 바꿔 가며 직접 확인합니다.

# %% [markdown]
# ## 7. 데이터셋 만들기
#
# ### 7.1 Goal 좌표계 — 랜덤 goal을 "고정 goal"로 바꾸기
#
# goal이 매번 다르면, 같은 "T를 오른쪽으로 0.5 m 밀기" 라도 world 좌표에서는 무한히 많은 모양으로 나타납니다. 네트워크가 이것을 다 외우려면 데이터가 훨씬 많이 필요합니다.
#
# 모든 점을 **goal에 붙은 좌표계** 로 옮기면 goal은 항상 원점, 각도 0이 됩니다:
#
# $$
# \tilde{\mathbf p}=R(\theta_g)^{\!\top}\,(\mathbf p-\mathbf x_g),\qquad
# \tilde\theta=\theta-\theta_g
# $$
#
# 이것은 강의의 **상대 운동 (relative motion)** 과 같은 생각입니다 — 관찰자를 goal에 태우는 것.
# 물리 법칙은 회전·평행이동에 대해 변하지 않으므로 (벽만 예외), goal 좌표계에서의 올바른 행동은 goal 위치와 무관합니다.
#
# **Observation** (6차원, 각도는 $\cos,\sin$ 으로 — $\pm\pi$ 에서 끊어지지 않도록):
#
# $$
# o_t=\big(\tilde x_P,\ \tilde y_P,\ \tilde x_G,\ \tilde y_G,\ \cos\tilde\theta,\ \sin\tilde\theta\big)
# $$
#
# **Action** 도 goal 좌표계로: $\ \tilde{\mathbf a}=R(\theta_g)^{\!\top}(\mathbf a-\mathbf x_g)$. 실행할 때는 거꾸로 $\ \mathbf a=\mathbf x_g+R(\theta_g)\,\tilde{\mathbf a}$.
#
# ### 7.2 Observation history 와 action chunk
#
# Diffusion Policy와 같은 구조를 씁니다.
#
# - **입력** : 최근 $T_o=2$ 개의 observation $(o_{t-1},o_t)$ — 두 개를 보면 속도 정보 (T가 미끄러지는 중인지) 가 생깁니다.
# - **출력** : 앞으로 $H=8$ 개의 action $\mathbf a_{t:t+H}=(a_t,\dots,a_{t+7})$ = 0.8 s 계획 (**action chunk**).
# - **실행** : 그 중 앞의 $T_a=4$ 개만 실행하고 다시 예측 (**receding horizon**).
#
# 한 step씩만 예측하면 매 step 결정이 바뀌며 떨리기 쉽습니다. chunk는 "한 번 정한 동작을 잠시 밀고 나가는" 일관성을 줍니다.
#
# episode의 시작/끝에서 창이 밖으로 나가면 첫/마지막 값을 반복해서 채웁니다 (padding).
#
# ### 7.3 정규화와 train/val 분할
#
# 각 차원을 $z=(x-\mu)/\sigma$ 로 정규화합니다 (통계는 training set에서만). 그리고 **episode 단위로** train/val을 나눕니다.
# 같은 episode의 이웃한 step들은 거의 같으므로, step 단위로 섞어 나누면 val loss가 실제보다 좋게 나옵니다 (data leakage).

# %%
OBS_HORIZON, ACT_HORIZON, EXEC_HORIZON = 2, 8, 4


def to_goal_frame(points, state):
    """World points (..., 2) -> goal frame of `state` (8,)."""
    c, s = math.cos(state[7]), math.sin(state[7])
    v = points - state[5:7]
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def from_goal_frame(points, state):
    c, s = math.cos(state[7]), math.sin(state[7])
    v = np.stack([c * points[..., 0] - s * points[..., 1], s * points[..., 0] + c * points[..., 1]], -1)
    return v + state[5:7]


def make_obs(state):
    """One state (8,) -> observation (6,) in the goal frame."""
    p = to_goal_frame(state[0:2], state)
    T = to_goal_frame(state[2:4], state)
    dth = state[4] - state[7]
    return np.array([p[0], p[1], T[0], T[1], math.cos(dth), math.sin(dth)])


OBS_DIM = 6


def make_samples(episodes):
    """All (obs window, action chunk) pairs: X (N, T_o*6), Y (N, H*2)."""
    X, Y = [], []
    for S, A in episodes:
        O = np.array([make_obs(s) for s in S])
        n = len(S)
        for t in range(n):
            io = [max(0, t - k) for k in range(OBS_HORIZON - 1, -1, -1)]     # t-1, t (padded at the start)
            ia = [min(n - 1, t + k) for k in range(ACT_HORIZON)]              # t ... t+7 (padded at the end)
            X.append(O[io].ravel())
            Y.append(to_goal_frame(A[ia], S[t]).ravel())                      # chunk in the goal frame of s_t
    return np.array(X, np.float32), np.array(Y, np.float32)


def split_episodes(episodes, val_frac=0.1, seed=0):
    idx = np.random.default_rng(seed).permutation(len(episodes))
    n_val = max(1, int(round(val_frac * len(episodes))))
    return [episodes[i] for i in idx[n_val:]], [episodes[i] for i in idx[:n_val]]


train_eps, val_eps = split_episodes(episodes)
X_train, Y_train = make_samples(train_eps)
X_val, Y_val = make_samples(val_eps)
print(f"train: {len(train_eps)} episodes -> X {X_train.shape}, Y {Y_train.shape}")
print(f"val  : {len(val_eps)} episodes -> X {X_val.shape}")

# one sample, drawn in the goal frame
S0, A0 = train_eps[0]
t = 40
fig, ax = plt.subplots(figsize=(6, 4.5))
V = env.sim.T.vertices
ax.add_patch(Polygon(V, closed=True, fill=False, ec="#3D8C54", lw=2, ls="--"))       # goal = origin, 0 deg
o = make_obs(S0[t])
ax.add_patch(Polygon(T_outline(o[2:4], math.atan2(o[5], o[4]), V), closed=True, fc="#D9EEFC", ec="#2774B8"))
ax.plot(*o[0:2], "o", color="#E97820", ms=10, label="pusher $o_t$")
o_prev = make_obs(S0[t - 1])
ax.plot(*o_prev[0:2], "o", color="#E97820", ms=6, alpha=0.4, label="pusher $o_{t-1}$")
chunk = to_goal_frame(A0[t:t + ACT_HORIZON], S0[t])
ax.plot(chunk[:EXEC_HORIZON, 0], chunk[:EXEC_HORIZON, 1], "x-", color="#FF0000", label="executed $a_{t:t+4}$")
ax.plot(chunk[EXEC_HORIZON - 1:, 0], chunk[EXEC_HORIZON - 1:, 1], "x--", color="#FF0000", alpha=0.4,
        label="predicted, not executed")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
ax.set_title("One training sample in the goal frame (goal = origin, 0 deg)")
show(fig, "03_sample_goal_frame")

# %% [markdown]
# ## 8. 어떤 모델을 써야 하나? — multimodality
#
# 가장 단순한 방법은 **Behavior Cloning (BC)**: 지도학습으로 $o\mapsto a$ 를 회귀합니다.
#
# $$
# \min_\phi\ \mathbb E_{(o,a)\sim\mathcal D}\ \big\lVert \pi_\phi(o)-a\big\rVert^2
# $$
#
# 문제는 사람 시연이 **multimodal** 이라는 것입니다. 같은 상황에서 어떤 때는 T의 왼쪽으로 돌아가고, 어떤 때는 오른쪽으로 돌아갑니다.
# MSE의 최적해는 조건부 **평균** $\ \pi^{*}(o)=\mathbb E[a\mid o]\ $ 이므로, 두 모드의 한가운데 — 즉 **T에 정면으로 부딪히는** 행동 — 을 배웁니다.
#
# 대안은 평균 대신 **분포** $p(a\mid o)$ 를 배우고 거기서 하나를 뽑는 생성 모델입니다.
#
# | 모델 | 출력 | multimodal? | 특징 |
# |---|---|---|---|
# | MLP-BC (MSE) | 한 점 | ✗ (평균) | 가장 빠르고 단순. 기준선(baseline) |
# | MLP-BC + action chunk | $H$ 개의 점 | ✗ | 시간적 일관성 ↑ |
# | GMM / MDN | 가우시안 $K$개 혼합 | △ | 모드 수 $K$ 를 정해야 함 |
# | VQ-BeT | 이산 코드 + offset | ○ | Transformer |
# | ACT (CVAE + Transformer) | chunk | ○ | 실제 로봇 50개 시연으로 학습 |
# | **Diffusion Policy** | chunk, 분포에서 sampling | ○ | PushT의 표준 기준, 학습 안정 |
#
# 이 튜토리얼의 선택: **MLP-BC (chunk) 를 기준선으로, Diffusion Policy를 주 모델로** 학습해 비교합니다.
# 둘 다 같은 MLP 몸통을 쓰므로 차이는 오직 **"평균을 배우느냐, 분포를 배우느냐"** 입니다.
#
# 먼저 장난감 문제로 차이를 눈으로 봅니다: 장애물 앞에서 시연자가 반반 확률로 위($a=+1$) 또는 아래($a=-1$) 로 피합니다.

# %%
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
print("torch", torch.__version__, "device:", DEVICE)

# %% [markdown]
# ### 두 모델 정의
#
# **공통 몸통** : 3층 MLP (폭 256, Mish 활성함수).
#
# **Model 1 — MLP-BC** : $\ \hat{\mathbf a}=f_\phi(o)$, loss $\ \lVert\hat{\mathbf a}-\mathbf a\rVert^2$.
#
# **Model 2 — Diffusion Policy (DDPM)** : 깨끗한 action chunk $\mathbf a^0$ 에 단계적으로 noise를 섞는 과정을 정의하고
#
# $$
# \mathbf a^k=\sqrt{\bar\alpha_k}\,\mathbf a^0+\sqrt{1-\bar\alpha_k}\,\boldsymbol\epsilon,\qquad
# \boldsymbol\epsilon\sim\mathcal N(\mathbf 0,I),\qquad k=1,\dots,K
# $$
#
# 네트워크 $\epsilon_\phi(\mathbf a^k,k,o)$ 가 섞인 noise를 맞히도록 학습합니다:
#
# $$
# \mathcal L=\mathbb E_{k,\boldsymbol\epsilon}\ \big\lVert\epsilon_\phi(\mathbf a^k,k,o)-\boldsymbol\epsilon\big\rVert^2
# $$
#
# 실행할 때는 순수한 noise $\mathbf a^K\sim\mathcal N(\mathbf 0,I)$ 에서 출발해 noise를 조금씩 빼 나갑니다 ($k=K,\dots,1$):
#
# $$
# \hat{\mathbf a}^0=\frac{\mathbf a^k-\sqrt{1-\bar\alpha_k}\,\epsilon_\phi}{\sqrt{\bar\alpha_k}},\qquad
# \mathbf a^{k-1}=\underbrace{\frac{\sqrt{\bar\alpha_{k-1}}\,\beta_k}{1-\bar\alpha_k}\hat{\mathbf a}^0+\frac{\sqrt{\alpha_k}\,(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}\mathbf a^k}_{\text{posterior mean}}+\sigma_k\mathbf z
# $$
#
# 출발 noise가 매번 달라서 **매번 다른 sample** 이 나오며, 그 sample들이 시연의 여러 모드를 따라갑니다.
# 여기서 $\beta_k$ 는 cosine schedule, $\alpha_k=1-\beta_k$, $\bar\alpha_k=\prod_{i\le k}\alpha_i$, $K=50$ 입니다.
#
# 두 모델 모두 정규화 통계 $(\mu,\sigma)$ 를 `buffer` 로 품고 있어서, checkpoint 하나로 저장/불러오기가 됩니다.

# %%
class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=256, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Mish()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, d_out))

    def forward(self, x):
        return self.net(x)


class Normalized(nn.Module):
    """Holds z-score statistics of X and Y (fitted on the training set)."""

    def set_stats(self, X, Y):
        self.register_buffer("x_mu", torch.tensor(X.mean(0)))
        self.register_buffer("x_sd", torch.tensor(X.std(0) + 1e-6))
        self.register_buffer("y_mu", torch.tensor(Y.mean(0)))
        self.register_buffer("y_sd", torch.tensor(Y.std(0) + 1e-6))
        return self

    def nx(self, x): return (x - self.x_mu) / self.x_sd
    def ny(self, y): return (y - self.y_mu) / self.y_sd
    def dy(self, y): return y * self.y_sd + self.y_mu


class BCPolicy(Normalized):
    name = "MLP-BC"

    def __init__(self, d_obs, d_act):
        super().__init__()
        self.f = MLP(d_obs, d_act)

    def loss(self, x, y):
        return ((self.f(self.nx(x)) - self.ny(y)) ** 2).mean()

    @torch.no_grad()
    def predict(self, x):
        return self.dy(self.f(self.nx(x)))


def cosine_betas(K, s=0.008):
    f = lambda k: np.cos((k / K + s) / (1 + s) * math.pi / 2) ** 2
    return np.clip([1 - f(k + 1) / f(k) for k in range(K)], 1e-4, 0.999)


class DiffusionPolicy(Normalized):
    name = "Diffusion"

    def __init__(self, d_obs, d_act, K=50, t_emb=32):
        super().__init__()
        self.K, self.d_act, self.t_emb = K, d_act, t_emb
        self.eps = MLP(d_obs + d_act + t_emb, d_act)
        betas = torch.tensor(cosine_betas(K), dtype=torch.float32)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("abar", torch.cumprod(1 - betas, 0))

    def embed(self, k):                             # sinusoidal embedding of the noise level k
        freqs = torch.exp(-math.log(1000.0) * torch.arange(self.t_emb // 2, device=k.device) / (self.t_emb // 2))
        e = k[:, None].float() * freqs[None]
        return torch.cat([torch.sin(e), torch.cos(e)], -1)

    def eps_hat(self, a_k, k, xn):
        return self.eps(torch.cat([a_k, xn, self.embed(k)], -1))

    def loss(self, x, y):
        a0 = self.ny(y)
        k = torch.randint(0, self.K, (len(a0),), device=a0.device)
        eps = torch.randn_like(a0)
        ab = self.abar[k][:, None]
        a_k = ab.sqrt() * a0 + (1 - ab).sqrt() * eps          # forward (noising) process
        return ((self.eps_hat(a_k, k, self.nx(x)) - eps) ** 2).mean()

    @torch.no_grad()
    def predict(self, x):
        xn = self.nx(x)
        a = torch.randn(len(xn), self.d_act, device=xn.device)  # a^K ~ N(0, I)
        for k in reversed(range(self.K)):
            kk = torch.full((len(a),), k, device=a.device)
            ab = self.abar[k]
            a0 = ((a - (1 - ab).sqrt() * self.eps_hat(a, kk, xn)) / ab.sqrt()).clamp(-5, 5)
            if k == 0:
                a = a0
                break
            ab_prev = self.abar[k - 1]
            mean = (ab_prev.sqrt() * self.betas[k] / (1 - ab)) * a0 + \
                   (self.alphas[k].sqrt() * (1 - ab_prev) / (1 - ab)) * a
            var = self.betas[k] * (1 - ab_prev) / (1 - ab)
            a = mean + var.sqrt() * torch.randn_like(a)
        return self.dy(a)


def train(model, X, Y, X_val=None, Y_val=None, steps=20_000, batch=256, lr=1e-3, log_every=2000, seed=0):
    torch.manual_seed(seed)
    model.to(DEVICE).train()
    X, Y = torch.tensor(X, device=DEVICE), torch.tensor(Y, device=DEVICE)
    if X_val is not None:
        X_val, Y_val = torch.tensor(X_val, device=DEVICE), torch.tensor(Y_val, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist, run, t0 = {"step": [], "train": [], "val": []}, [], time.perf_counter()
    for it in range(1, steps + 1):
        idx = torch.randint(0, len(X), (batch,), device=DEVICE)
        loss = model.loss(X[idx], Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(loss.item())
        if it % log_every == 0 or it == steps:
            v = float("nan")
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    torch.manual_seed(1)                    # same noise every time -> comparable val loss
                    v = model.loss(X_val, Y_val).item()
                model.train()
            hist["step"].append(it); hist["train"].append(float(np.mean(run))); hist["val"].append(v)
            print(f"  [{model.name:9s}] step {it:6d}  train {np.mean(run):.4f}  val {v:.4f}  "
                  f"({time.perf_counter() - t0:.0f} s)")
            run = []
    model.eval()
    return hist

# %% [markdown]
# ### 장난감 실험: 장애물 앞에서 위로 피할까, 아래로 피할까
#
# 관측 $o\in[-1,1]$ (장애물까지 거리), 시연 행동 $a=\pm1+0.1\,\epsilon$ (반반). 정답 분포는 두 봉우리입니다.

# %%
rng = np.random.default_rng(0)
o_toy = rng.uniform(-1, 1, (4000, 1)).astype(np.float32)
a_toy = (rng.choice([-1.0, 1.0], (4000, 1)) + 0.1 * rng.standard_normal((4000, 1))).astype(np.float32)

toy_bc = BCPolicy(1, 1).set_stats(o_toy, a_toy)
toy_dp = DiffusionPolicy(1, 1).set_stats(o_toy, a_toy)
train(toy_bc, o_toy, a_toy, steps=300, log_every=300)
train(toy_dp, o_toy, a_toy, steps=300, log_every=300)

o_q = torch.tensor(np.linspace(-1, 1, 400, dtype=np.float32)[:, None], device=DEVICE)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
axes[0].plot(o_toy[:800, 0], a_toy[:800, 0], ".", ms=3, color="#7F7F7F")
axes[0].set_title("demonstrations: up or down (50 / 50)")
axes[1].plot(o_q.cpu()[:, 0], toy_bc.predict(o_q).cpu()[:, 0], ".", ms=3, color="#2774B8")
axes[1].set_title("MLP-BC (MSE) -> the mean: straight into the obstacle")
axes[2].plot(o_q.cpu()[:, 0], toy_dp.predict(o_q).cpu()[:, 0], ".", ms=3, color="#E97820")
axes[2].set_title("Diffusion -> samples from both modes")
for ax in axes:
    ax.axhline(0, color="#FF0000", lw=0.8, ls=":"); ax.set_xlabel("observation $o$"); ax.grid(alpha=0.3)
axes[0].set_ylabel("action $a$")
show(fig, "04_multimodality_toy")

# %% [markdown]
# MLP-BC는 $a\approx0$ — 어느 쪽으로도 피하지 않는 행동 — 을 배웁니다. Diffusion은 매번 위 또는 아래 중 하나를 뽑습니다.
#
# PushT에서도 같은 일이 생깁니다: pusher가 T를 어느 쪽으로 돌아갈지, 어느 면을 밀지 시연마다 다르면 MLP-BC는 그 사이 어딘가에서 머뭇거립니다.

# %% [markdown]
# ## 9–11. PushT policy 학습
#
# 두 모델을 같은 데이터, 같은 step 수로 학습합니다.
#
# - batch 256, AdamW, learning rate $10^{-3}$ → cosine으로 0까지
# - `TRAIN_STEPS` 회 gradient step (CPU 기준 모델 하나에 수 분, GPU면 더 빠름)
#
# 학습이 끝나면 `checkpoints/` 에 저장하고, 다음 실행에서 **같은 데이터**면 다시 학습하지 않고 불러옵니다 (`RETRAIN = True` 로 강제 재학습).

# %%
TRAIN_STEPS = 1000
RETRAIN = False


def data_tag(eps):
    return f"{DATA_SOURCE}_{len(eps)}ep_{sum(len(S) for S, _ in eps)}st"


def fit_or_load(cls, X, Y, Xv, Yv, tag, steps=TRAIN_STEPS):
    path = os.path.join(CKPT_DIR, f"{cls.name}_{tag}_{steps}.pt")
    model = cls(X.shape[1], Y.shape[1]).set_stats(X, Y)
    if os.path.exists(path) and not RETRAIN:
        ck = torch.load(path, map_location="cpu")
        model.load_state_dict(ck["model"])
        print(f"loaded {os.path.relpath(path, HERE)}")
        return model.to(DEVICE).eval(), ck["hist"]
    hist = train(model, X, Y, Xv, Yv, steps=steps)
    torch.save({"model": model.state_dict(), "hist": hist}, path)
    return model, hist


tag = data_tag(train_eps)
bc, hist_bc = fit_or_load(BCPolicy, X_train, Y_train, X_val, Y_val, tag)
dp, hist_dp = fit_or_load(DiffusionPolicy, X_train, Y_train, X_val, Y_val, tag)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (m, h) in zip(axes, ((bc, hist_bc), (dp, hist_dp))):
    ax.semilogy(h["step"], h["train"], "o-", label="train")
    ax.semilogy(h["step"], h["val"], "s--", label="val (held-out episodes)")
    ax.set_title(f"{m.name}: " + ("MSE of the action chunk" if m.name == "MLP-BC" else "noise-prediction MSE"))
    ax.set_xlabel("gradient step"); ax.grid(alpha=0.3, which="both"); ax.legend()
show(fig, "05_loss_curves")

# %% [markdown]
# **loss만 보고 판단하지 마세요.** 두 loss는 서로 다른 양이라 크기를 비교할 수 없고, 같은 모델 안에서도
# loss가 낮다고 closed-loop 성공률이 높은 것은 아닙니다. 학습할 때는 항상 **시연 데이터 위**의 상태만 보지만,
# 실행할 때는 policy 자신의 작은 실수가 쌓여 시연에 없던 상태로 들어가기 때문입니다 (**compounding error / covariate shift**).
# 그래서 반드시 시뮬레이터에서 직접 돌려 봐야 합니다.
#
# ## 12. Closed-loop 평가
#
# 학습된 모델을 `reset()` / `act(state)` 를 가진 agent로 감쌉니다 (expert와 같은 인터페이스).
#
# 1. 최근 $T_o=2$ 개의 state → observation window
# 2. 모델이 goal 좌표계의 chunk $\tilde{\mathbf a}_{t:t+8}$ 예측 → world 좌표로 되돌림
# 3. 앞의 $T_a=4$ 개를 차례로 실행, 다 쓰면 다시 예측
#
# 평가 seed `0 … N_EVAL-1` 은 학습 데이터 (`100000+`, `200000+`) 에 없는 초기 조건입니다. 같은 seed에서 expert의 성공률도 함께 잽니다.

# %%
class LearnedAgent:
    def __init__(self, model, exec_horizon=EXEC_HORIZON):
        self.model, self.exec_horizon = model, exec_horizon
        self.reset()

    def reset(self):
        self.history, self.queue, self.plan = [], [], None

    def act(self, state):
        self.history.append(state)
        if not self.queue:
            window = self.history[-OBS_HORIZON:]
            window = [window[0]] * (OBS_HORIZON - len(window)) + window            # pad at the start
            x = np.concatenate([make_obs(s) for s in window]).astype(np.float32)
            y = self.model.predict(torch.tensor(x[None], device=DEVICE))[0].cpu().numpy()
            self.plan = from_goal_frame(y.reshape(ACT_HORIZON, 2), state)        # back to the world frame
            self.queue = list(self.plan[:self.exec_horizon])
        return self.queue.pop(0)


N_EVAL, EVAL_MAX_STEPS = 4, 100


def evaluate(agent, n=N_EVAL, max_steps=EVAL_MAX_STEPS, label=""):
    env_e = PushTEnv(max_steps=max_steps)
    if agent == "expert":
        agent = ScriptedExpert(env_e)
    res = {"success": [], "steps": [], "pos_err": [], "ang_err": [], "seed": []}
    t0 = time.perf_counter()
    for seed in range(n):
        S, A, ok = run_episode(env_e, agent, seed)
        d, a = env_e.sim.goal_error()
        for k, v in zip(res, (ok, len(S), d, a, seed)):
            res[k].append(v)
    res = {k: np.array(v) for k, v in res.items()}
    print(f"{label:10s} success {res['success'].mean() * 100:5.1f} %   "
          f"median final error {np.median(res['pos_err']):.2f} m, {np.median(res['ang_err']):.0f} deg   "
          f"({time.perf_counter() - t0:.0f} s)")
    return res


results = {
    "expert": evaluate("expert", label="expert"),
    bc.name: evaluate(LearnedAgent(bc), label=bc.name),
    dp.name: evaluate(LearnedAgent(dp), label=dp.name),
}

# %% [markdown]
# ### 결과 보기
#
# 왼쪽: 성공률 막대. 오른쪽: 각 평가 seed의 최종 오차 (점선 상자 = 성공 영역).

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
names = list(results)
colors = ["#7F7F7F", "#2774B8", "#E97820"]
rates = [results[k]["success"].mean() * 100 for k in names]
axes[0].bar(names, rates, color=colors)
for i, r in enumerate(rates):
    axes[0].text(i, r + 1, f"{r:.0f}%", ha="center")
axes[0].set_ylim(0, 105); axes[0].set_ylabel("success rate on unseen seeds [%]")
axes[0].set_title(f"{N_EVAL} unseen random tasks, {len(train_eps)} training demos ({DATA_SOURCE})")
for k, c in zip(names, colors):
    axes[1].scatter(results[k]["pos_err"], results[k]["ang_err"], s=18, color=c, label=k, alpha=0.8)
axes[1].add_patch(plt.Rectangle((0, 0), 0.15, 10, fill=False, ls="--", ec="#3D8C54"))
axes[1].set_xscale("symlog", linthresh=0.2); axes[1].set_yscale("symlog", linthresh=10)
axes[1].set_xlabel("final position error [m]"); axes[1].set_ylabel("final angle error [deg]")
axes[1].legend(); axes[1].grid(alpha=0.3)
show(fig, "06_success_rates")


def plot_rollouts(model, seeds, name):
    env_p = PushTEnv(max_steps=EVAL_MAX_STEPS)
    agent = LearnedAgent(model)
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 3.6))
    for ax, seed in zip(axes, seeds):
        S, A, ok = run_episode(env_p, agent, seed)
        draw_task(ax, env_p, S[0])
        for k in range(0, len(S), 25):
            draw_task(ax, env_p, S[k], color=plt.cm.viridis(k / max(len(S), 1)), alpha=0.5, goal=False)
        draw_task(ax, env_p, env_p.state(), color="#E97820", goal=False)
        ax.plot(S[:, 0], S[:, 1], "-", color="#E97820", lw=0.8)
        ax.set_title(f"seed {seed}: {'success' if ok else 'fail'} ({len(S)} steps)", fontsize=9)
    fig.suptitle(f"{model.name} rollouts on unseen seeds")
    show(fig, name)


plot_rollouts(bc, range(4), "07_rollouts_bc")
plot_rollouts(dp, range(4), "08_rollouts_diffusion")

# %% [markdown]
# ## 13. 시연을 몇 개 모아야 하나 — 직접 실험
#
# 학습 데이터 수 $N$ 만 바꾸고 (나머지는 동일, gradient step 수도 동일) 평가 성공률을 봅니다.
# expert 시연으로 하는 이유는 같은 품질의 시연을 원하는 만큼 만들 수 있기 때문입니다.
#
# 시간이 꽤 걸리므로 (`N` 하나당 모델 2개 학습 + 평가) 기본값은 꺼져 있습니다: `RUN_SCALING = True`.

# %%
RUN_SCALING = True
SCALING_NS = [10, 20]

if RUN_SCALING:
    pool_train, pool_val = split_episodes(expert_episodes)
    Xv_s, Yv_s = make_samples(pool_val)
    scaling = {"MLP-BC": [], "Diffusion": []}
    for n in SCALING_NS:
        sub = pool_train[:n]
        Xs, Ys = make_samples(sub)
        print(f"--- N = {n} demos ({len(Xs)} samples)")
        for cls in (BCPolicy, DiffusionPolicy):
            m, _ = fit_or_load(cls, Xs, Ys, Xv_s, Yv_s, f"expert_{n}ep_{len(Xs)}st")
            scaling[cls.name].append(evaluate(LearnedAgent(m), label=f"{cls.name} N={n}")["success"].mean() * 100)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (k, v), c in zip(scaling.items(), ("#2774B8", "#E97820")):
        ax.plot(SCALING_NS, v, "o-", color=c, label=k)
    ax.axhline(results["expert"]["success"].mean() * 100, color="#7F7F7F", ls="--", label="expert")
    ax.set_xscale("log"); ax.set_xticks(SCALING_NS); ax.set_xticklabels(SCALING_NS)
    ax.set_xlabel("number of demonstrations"); ax.set_ylabel("success rate [%]")
    ax.grid(alpha=0.3); ax.legend()
    show(fig, "09_data_scaling")

# %% [markdown]
# ## 14. 학습된 policy를 창에서 보기
#
# 키보드 데모 창을 그대로 쓰되, target을 사람이 아니라 policy가 정합니다. **빨간 점 = 모델이 예측한 action chunk** (실행할 4개는 진하게).
#
# | 키 | 동작 |
# |---|---|
# | N | 새 랜덤 과제 |
# | 방향키 / 마우스 | **사람이 개입** (누르는 동안 policy 대신 사람) — policy가 막혔을 때 도와줘 보세요 |
# | Space | 일시정지,  Q / Esc 종료 |

# %%
class PolicyViewer(DemoRecorder):

    def __init__(self, model):
        self.agent = LearnedAgent(model)
        super().__init__(path=os.devnull + ".npz")        # reuse the window, never save
        self.plan_dots, = self.ax.plot([], [], ".", color="#FF0000", ms=7, alpha=0.4)
        self.exec_dots, = self.ax.plot([], [], "o", color="#FF0000", ms=5)
        self.name = model.name

    def new_episode(self, seed=None):
        super().new_episode(seed)
        self.agent.reset()

    def save(self):
        pass

    def control_tick(self):
        now = time.perf_counter()
        human = self.mouse is not None or any(self._held(k, now) for k in KEYS)
        if human:
            super().control_tick()
            self.agent.reset()                           # replan from scratch after the human lets go
        else:
            self.sim.target = np.asarray(self.agent.act(self.env.state()), dtype=float)
        if self.agent.plan is not None:
            self.plan_dots.set_data(self.agent.plan[:, 0], self.agent.plan[:, 1])
            q = np.array(self.agent.queue) if self.agent.queue else np.zeros((0, 2))
            self.exec_dots.set_data(q[:, 0], q[:, 1])

    def update(self, _frame=None, frame_dt=None):
        if self.restart_at is not None and time.perf_counter() > self.restart_at:
            self.new_episode()
        if not self.paused and self.restart_at is None:
            if self.frame % 2 == 0:
                self.control_tick()
            self.frame += 1
        out = KeyboardDemo.update(self, frame_dt=self.frame_dt if self.restart_at is None else 0.0)
        if self.restart_at is None and self.env.is_success():
            self.restart_at = time.perf_counter() + 1.5
        self.title.set_text(f"{self.name} policy   t = {self.sim.time:5.1f} s   "
                            + ("SUCCESS" if self.env.is_success() else "(N: new task, arrows: help)"))
        return out


def watch_policy(model):
    if IN_NOTEBOOK:
        get_ipython().run_line_magic("matplotlib", "tk")    # noqa: F821
    else:
        matplotlib.use("TkAgg", force=True)
    viewer = PolicyViewer(model)
    viewer.fig.texts[-1].set_text("\n".join([
        "N             : new task", "arrows / mouse: take over", "space         : pause", "Q / Esc       : quit",
        "", "red dots: predicted chunk", "solid  : to be executed"]))
    viewer.run()
    return viewer


best = max((bc, dp), key=lambda m: results[m.name]["success"].mean())
print("best model:", best.name)
if not NO_GUI:
    viewer = watch_policy(best)

# %% [markdown]
# ## 15. 해 볼 것
#
# 1. **사람 시연으로 학습.** `python pusht_imitation.py collect` 로 50개 → 학습 → 100개 → 학습. 성공률이 어떻게 바뀌나요?
#    expert 시연과 비교해 MLP-BC와 Diffusion의 **차이가 더 커지는지** 보세요 (사람 시연이 더 multimodal 하기 때문).
# 2. **Goal 좌표계를 끄면?** `make_obs` 를 world 좌표 (state 8개, 각도는 cos/sin) 로 바꾸고, action도 world 좌표로 두고 다시 학습해 보세요.
#    같은 데이터로 성공률이 얼마나 떨어지나요? 이것이 "좋은 표현 = 데이터 절약" 입니다.
# 3. **Action chunk 길이.** `ACT_HORIZON` = 1, 4, 8, 16 / `EXEC_HORIZON` 을 바꿔 보세요. 1이면 매 step 결정이 바뀌어 떨림이 생깁니다.
# 4. **Observation history.** `OBS_HORIZON = 1` 이면 속도를 알 수 없습니다. T가 미끄러지는 중인지 모르면 어떤 실수가 생기나요?
# 5. **DAgger.** 학습된 policy를 돌리다가 사람이 개입한 구간 (14절의 창에서 방향키) 을 기록해 데이터에 더하면, policy가 스스로 만든 실수 상황에서의 회복 방법을 배웁니다 (covariate shift 해결).
# 6. **물리 파라미터가 바뀌면?** 학습은 $\mu=0.4$ 로 했습니다. 평가할 때만 `env.sim.friction = 0.2` 로 바꾸면 성공률은? 데이터 기반 policy의 한계와 domain randomization을 생각해 보세요.
# 7. **Diffusion sampling step 줄이기.** $K=50$ 번의 denoising 대신 DDIM으로 10번만 하면 속도와 성공률이 어떻게 변하나요?
