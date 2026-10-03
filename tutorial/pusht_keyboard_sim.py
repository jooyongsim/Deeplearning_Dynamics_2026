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
# # Keyboard PushT — 실시간으로 직접 밀어 보는 2D rigid-body contact
#
# 앞의 튜토리얼 (`pusht_mini_sim.py`) 에서 만든 물리를 그대로 쓰되, 이번에는 **사람이 policy 역할**을 합니다.
# 키보드(또는 마우스)로 target을 움직이면 pusher가 PD controller로 따라가고, T-block과의 충돌이 실시간으로 계산됩니다.
#
# $$
# \boxed{
# \text{keyboard}
# \rightarrow
# \text{target } (x^{*},y^{*})
# \rightarrow
# \text{PD pusher}
# \rightarrow
# \text{contact}
# \rightarrow
# J
# \rightarrow
# (\mathbf v_G,\omega)
# \rightarrow
# (\mathbf x_G,\theta)
# }
# $$
#
# **조작법**
#
# | 키 | 동작 |
# |---|---|
# | 방향키 / W A S D | target 이동 (누르고 있는 동안 계속) |
# | 마우스 왼쪽 드래그 | target을 마우스 위치로 |
# | Space | 일시정지 / 재개 |
# | R | 초기 상태로 reset |
# | F | 마찰 on/off ($\mu = 0.4 \leftrightarrow 0$) |
# | E | 반발계수 전환 ($e = 0 \leftrightarrow 0.5$) |
# | Q / Esc | 종료 |
#
# **실행 방법**
#
# - 터미널: `python pusht_keyboard_sim.py` → 별도 창이 열립니다.
# - VS Code Interactive / Jupyter: 셀을 위에서부터 실행하면 마지막 셀에서 **별도 Tk 창**이 열립니다
#   (inline 그림은 키보드 입력을 받을 수 없기 때문에 `%matplotlib tk` 로 전환합니다).
# - 창을 한 번 클릭해서 **포커스를 준 뒤** 키를 누르세요.
#
# 필요한 것: `numpy`, `matplotlib` (Tk backend — 표준 Python에 포함된 `tkinter`).
#
# notebook으로 바꾸기: `python ../tools/py2ipynb.py pusht_keyboard_sim.py`

# %%
import math
import os
import time
from dataclasses import dataclass

import numpy as np

# %% [markdown]
# ## 1. 물리 core — 이전 튜토리얼과 같은 식
#
# 자세한 유도는 `pusht_mini_sim.py` 에 있고, 여기서는 핵심 식만 다시 적습니다.
#
# **Contact point 속도와 상대속도**
#
# $$
# \mathbf v_C=\mathbf v_G+\boldsymbol\omega\times\mathbf r,\qquad
# v_n=(\mathbf v_C-\mathbf v_{\text{other}})\cdot\mathbf n
# $$
#
# **Normal impulse** ($v_n<0$, 즉 서로 접근할 때만)
#
# $$
# K_n=\frac1m+\frac{(\mathbf r\times\mathbf n)^2}{I},\qquad
# j_n=-\frac{(1+e)\,v_n}{K_n}
# $$
#
# **속도 변화**
#
# $$
# \Delta\mathbf v_G=\frac{\mathbf J}{m},\qquad
# \Delta\omega=\frac{\mathbf r\times\mathbf J}{I}
# $$
#
# **Coulomb friction** : $\ j_t^{*}=-v_t/K_t,\quad \lvert j_t\rvert\le\mu\,j_n$
#
# 이전 코드와 한 가지만 다릅니다. `resolve_impulse()` 가 pusher 객체 대신 **상대 물체의 속도** `other_vel` 을 받습니다.
# 그래서 같은 함수로 pusher ($\mathbf v_{\text{other}}=\mathbf v_P$) 와 벽 ($\mathbf v_{\text{other}}=\mathbf 0$) 을 모두 처리합니다.

# %%
def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def cross2(a, b):
    """2D cross product a x b -> scalar (z component)."""
    return float(a[0] * b[1] - a[1] * b[0])


def omega_cross_r(omega, r):
    """[0, 0, omega] x [rx, ry, 0] = [-omega*ry, omega*rx]"""
    return np.array([-omega * r[1], omega * r[0]])


def make_T_geometry(W=1.8, H=1.8, stem_w=0.5, top_t=0.5, mass=1.0):
    """T = top rectangle + stem rectangle. Returns COM-centred vertices and I about the COM."""
    top_area = W * top_t
    stem_h = H - top_t
    stem_area = stem_w * stem_h
    total_area = top_area + stem_area

    y_top, y_stem = H / 2 - top_t / 2, -top_t / 2
    y_com = (top_area * y_top + stem_area * y_stem) / total_area

    m_top = mass * top_area / total_area
    m_stem = mass * stem_area / total_area
    # rectangle about its own centre + parallel-axis shift to the T's COM
    I_top = m_top * (W**2 + top_t**2) / 12 + m_top * (y_top - y_com) ** 2
    I_stem = m_stem * (stem_w**2 + stem_h**2) / 12 + m_stem * (y_stem - y_com) ** 2

    vertices = np.array([
        [-W / 2, H / 2], [W / 2, H / 2], [W / 2, H / 2 - top_t], [stem_w / 2, H / 2 - top_t],
        [stem_w / 2, -H / 2], [-stem_w / 2, -H / 2], [-stem_w / 2, H / 2 - top_t], [-W / 2, H / 2 - top_t],
    ], dtype=float)
    vertices[:, 1] -= y_com          # body origin = centre of mass
    return vertices, I_top + I_stem


def closest_point_segment(p, a, b):
    ab = b - a
    den = np.dot(ab, ab)
    if den < 1e-12:
        return a.copy()
    t = np.clip(np.dot(p - a, ab) / den, 0.0, 1.0)
    return a + t * ab


def point_in_polygon(p, vertices):
    """Ray casting — works for the concave T."""
    x, y = p
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def circle_vs_T(circle_pos, radius, body_pos, body_angle, vertices):
    """Return (contact_point, normal pusher->T, penetration) or None."""
    R = rotation_matrix(body_angle)
    c = R.T @ (circle_pos - body_pos)                 # world -> T local frame

    best_q, best_d2 = None, float("inf")
    for i in range(len(vertices)):
        q = closest_point_segment(c, vertices[i], vertices[(i + 1) % len(vertices)])
        d2 = float(np.dot(q - c, q - c))
        if d2 < best_d2:
            best_q, best_d2 = q, d2
    d = math.sqrt(best_d2)
    inside = point_in_polygon(c, vertices)
    if not inside and d >= radius:
        return None

    direction = (best_q - c) / d if d > 1e-12 else np.array([1.0, 0.0])
    if inside:                                         # centre already inside the T
        n_local, pen = -direction, radius + d
    else:
        n_local, pen = direction, radius - d
    n = R @ n_local
    n /= np.linalg.norm(n) + 1e-12
    return body_pos + R @ best_q, n, pen


@dataclass
class TBody:
    pos: np.ndarray
    angle: float
    vel: np.ndarray
    omega: float
    mass: float
    inertia: float
    vertices: np.ndarray


@dataclass
class Pusher:
    pos: np.ndarray
    vel: np.ndarray
    radius: float


def resolve_impulse(body, other_vel, contact, restitution=0.0, friction=0.4):
    """Impulse on `body` from a kinematic partner moving at `other_vel` (pusher or wall).

    contact = (point, normal pointing INTO the body, penetration).  Returns (j_n, j_t).
    """
    point, n, pen = contact
    r = point - body.pos
    v_rel = body.vel + omega_cross_r(body.omega, r) - other_vel
    v_n = float(np.dot(v_rel, n))

    j_n = j_t = 0.0
    if v_n < 0:                                        # approaching
        K_n = 1.0 / body.mass + cross2(r, n) ** 2 / body.inertia
        j_n = -(1.0 + restitution) * v_n / K_n
        J = j_n * n
        body.vel += J / body.mass
        body.omega += cross2(r, J) / body.inertia

        # friction, using the velocity after the normal impulse
        t = np.array([-n[1], n[0]])
        v_rel = body.vel + omega_cross_r(body.omega, r) - other_vel
        v_t = float(np.dot(v_rel, t))
        K_t = 1.0 / body.mass + cross2(r, t) ** 2 / body.inertia
        j_t = float(np.clip(-v_t / K_t, -friction * j_n, friction * j_n))
        Jt = j_t * t
        body.vel += Jt / body.mass
        body.omega += cross2(r, Jt) / body.inertia

    # push apart whatever overlap is left (penetration correction)
    body.pos += 0.8 * max(pen - 1e-4, 0.0) * n
    return j_n, j_t

# %% [markdown]
# ## 2. 벽 (arena wall) — "속도가 0인 kinematic body"
#
# 이전 예제에서는 T가 한 번 맞으면 멀리 미끄러져 나갔습니다. 여기서는 사각형 arena에 벽을 둡니다.
#
# 벽 충돌은 새로운 물리가 아닙니다. 벽은 움직이지 않는 kinematic body이므로
#
# $$
# \mathbf v_{\text{other}}=\mathbf 0,\qquad \frac1{m_{\text{wall}}}=0
# $$
#
# 로 두면 pusher와 **완전히 같은 식** 으로 impulse가 계산됩니다.
#
# T의 꼭짓점 $\mathbf p_i$ 가 벽 밖으로 나가면 그 꼭짓점이 contact point입니다. 예를 들어 왼쪽 벽 $x=x_{\min}$ 이면
#
# $$
# \delta = x_{\min}-p_{i,x} > 0,\qquad \mathbf n=(1,0)\ \ (\text{벽} \rightarrow T)
# $$
#
# (concave T의 안쪽 꼭짓점은 바깥 꼭짓점보다 먼저 벽에 닿을 수 없으므로 모든 꼭짓점을 검사해도 문제없습니다.)

# %%
ARENA = (-4.0, 4.0, -3.0, 3.0)        # x_min, x_max, y_min, y_max


def wall_contacts(body, arena=ARENA):
    """Contacts of the T's vertices with the four arena walls: list of (point, normal, penetration)."""
    x_min, x_max, y_min, y_max = arena
    world = body.pos + body.vertices @ rotation_matrix(body.angle).T
    contacts = []
    for p in world:
        for depth, n in ((x_min - p[0], (1.0, 0.0)), (p[0] - x_max, (-1.0, 0.0)),
                         (y_min - p[1], (0.0, 1.0)), (p[1] - y_max, (0.0, -1.0))):
            if depth > 0:
                contacts.append((p.copy(), np.array(n), depth))
    return contacts

# %% [markdown]
# ## 3. 실시간 simulator
#
# 화면은 약 50 fps로 갱신되지만, 물리는 그보다 잘게 나눠 계산합니다.
# 한 frame 동안 실제로 흐른 시간 $\Delta t_{\text{frame}}$ 을 재서, 그만큼을 $\Delta t_{\text{physics}}=0.005$ s 단위로 나누어 적분합니다.
#
# $$
# n_{\text{sub}} = \operatorname{round}\!\left(\frac{\Delta t_{\text{frame}}}{\Delta t_{\text{physics}}}\right)
# $$
#
# 한 substep의 순서 (이전 튜토리얼의 `step()` 과 같음):
#
# 1. **Pusher PD control** : $\ \mathbf a_P=K_p(\mathbf x^{*}-\mathbf x_P)-K_d\mathbf v_P$, 속도 제한 후 적분
# 2. **T damping** : $\ \mathbf v_G\leftarrow\mathbf v_G\,e^{-c_v\Delta t},\ \ \omega\leftarrow\omega\,e^{-c_\omega\Delta t}$ (바닥 마찰 흉내)
# 3. **T 적분** : $\ \mathbf x_G\leftarrow\mathbf x_G+\mathbf v_G\Delta t,\ \ \theta\leftarrow\theta+\omega\Delta t$
# 4. **Contact solver** (3회 반복) : pusher–T, 벽–T
#
# 마지막 contact의 위치·normal을 저장하고, 한 frame 동안의 impulse 합을 시간으로 나눠 **평균 contact force** 로 표시합니다.
#
# $$
# \mathbf F_{\text{contact}}\approx\frac{1}{\Delta t_{\text{frame}}}\sum_{\text{substeps}}\mathbf J
# $$
#
# impulse는 힘을 시간에 대해 적분한 것($\mathbf J=\int\mathbf F\,dt$)이므로, 짧은 구간의 impulse 합을 그 시간으로 나누면 평균 힘이 됩니다.
# substep 하나의 $j_n$ 은 0.005 s 동안의 impulse라서 아주 작지만, 힘으로 바꾸면 크기를 비교하기 쉽습니다.
#
# **Goal** : 화면의 초록 점선 T 위치·각도에 맞추는 것이 PushT의 과제입니다.
# 위치 오차 $\lVert\mathbf x_G-\mathbf x_{\text{goal}}\rVert<0.15$, 각도 오차 $<10^\circ$ 이면 성공으로 표시합니다.

# %%
class KeyboardPushT:

    def __init__(self):
        self.physics_dt = 0.005
        self.kp, self.kd, self.max_speed = 80.0, 18.0, 3.0
        self.linear_damping, self.angular_damping = 3.0, 3.0
        self.restitution, self.friction = 0.0, 0.4
        self.goal_pos, self.goal_angle = np.array([2.0, -0.6]), math.radians(45)
        self.reset()

    def reset(self):
        vertices, inertia = make_T_geometry(mass=1.0)
        self.T = TBody(pos=np.array([-0.5, 0.3]), angle=0.0, vel=np.zeros(2), omega=0.0,
                       mass=1.0, inertia=inertia, vertices=vertices)
        self.pusher = Pusher(pos=np.array([-2.6, 0.6]), vel=np.zeros(2), radius=0.18)
        self.target = self.pusher.pos.copy()
        self.time = 0.0
        self.last_contact = None          # (point, normal) of the latest pusher contact
        self.contact_force = (0.0, 0.0)   # (F_n, F_t) = impulses summed over the last frame / frame time
        self._j_sum = np.zeros(2)

    # ----------------------------------------------------------- one substep
    def substep(self, dt):
        P, T = self.pusher, self.T

        # 1. pusher PD control toward the target
        acc = self.kp * (self.target - P.pos) - self.kd * P.vel
        P.vel += acc * dt
        speed = np.linalg.norm(P.vel)
        if speed > self.max_speed:
            P.vel *= self.max_speed / speed
        P.pos += P.vel * dt
        x_min, x_max, y_min, y_max = ARENA                     # keep the pusher in the arena
        P.pos = np.clip(P.pos, [x_min + P.radius, y_min + P.radius], [x_max - P.radius, y_max - P.radius])

        # 2. damping  3. integrate the T
        T.vel *= math.exp(-self.linear_damping * dt)
        T.omega *= math.exp(-self.angular_damping * dt)
        T.pos += T.vel * dt
        T.angle += T.omega * dt

        # 4. contact solver: pusher-T and wall-T, a few iterations
        touching = False
        for _ in range(3):
            c = circle_vs_T(P.pos, P.radius, T.pos, T.angle, T.vertices)
            if c is not None:
                jn, jt = resolve_impulse(T, P.vel, c, self.restitution, self.friction)
                touching = True
                self.last_contact = (c[0], c[1])
                self._j_sum += (jn, jt)
            for w in wall_contacts(T):
                resolve_impulse(T, np.zeros(2), w, self.restitution, self.friction)
        if not touching:
            self.last_contact = None
        self.time += dt

    def advance(self, frame_dt):
        n = max(1, int(round(frame_dt / self.physics_dt)))
        self._j_sum[:] = 0.0
        for _ in range(n):
            self.substep(self.physics_dt)
        F = self._j_sum / (n * self.physics_dt)      # impulse per time = average force over the frame
        self.contact_force = (float(F[0]), float(F[1]))

    def goal_error(self):
        d = float(np.linalg.norm(self.T.pos - self.goal_pos))
        a = (self.T.angle - self.goal_angle + math.pi) % (2 * math.pi) - math.pi
        return d, math.degrees(abs(a))

# %% [markdown]
# ## 4. 확인: GUI 없이 "키를 누르고 있는" 상황을 재현
#
# 실제 창을 열기 전에, target을 오른쪽으로 2초 동안 움직였을 때 T가 밀리는지 숫자로 확인합니다.
# (키보드 입력은 결국 target을 매 frame 조금씩 옮기는 것뿐이므로, 같은 일을 코드로 할 수 있습니다.)

# %%
sim = KeyboardPushT()
TARGET_SPEED = 1.5           # m/s, how fast a held key moves the target
frame_dt = 0.02
start = sim.T.pos.copy()
n_contact = 0
for _ in range(int(2.0 / frame_dt)):
    sim.target = sim.target + np.array([TARGET_SPEED * frame_dt, 0.0])   # "holding →"
    sim.advance(frame_dt)
    n_contact += sim.last_contact is not None

print(f"T moved      : {np.round(sim.T.pos - start, 3)}  (rotated {math.degrees(sim.T.angle):.1f} deg)")
print(f"frames in contact: {n_contact} / {int(2.0 / frame_dt)}")
print(f"contact force (F_n, F_t) in the last frame: {tuple(round(v, 3) for v in sim.contact_force)} N")
print(f"goal error   : {sim.goal_error()[0]:.2f} m, {sim.goal_error()[1]:.1f} deg")

# %% [markdown]
# ## 5. 실시간 키보드 데모
#
# matplotlib의 이벤트를 씁니다.
#
# - `key_press_event` / `key_release_event` : 어떤 키가 눌려 있는지 기록
# - `button_press_event` / `motion_notify_event` : 마우스 드래그로 target 지정
# - `FuncAnimation` : 약 20 ms마다 `update()` 호출 → 실제 경과 시간만큼 물리를 진행하고 다시 그림
#
# **키를 "누르고 있는" 판정** : 운영체제의 key auto-repeat 때문에, 누르고 있어도 press/release 이벤트가 짧은 간격으로 반복해서 들어옵니다.
# 그래서 "release를 받았다"가 아니라 **"마지막 press가 0.12 s 이내이거나 아직 release되지 않았다"** 를 누르고 있는 상태로 봅니다.
#
# 화면에 그리는 것:
#
# - 파란 T = 현재 상태, 초록 점선 T = goal, 주황 원 = pusher, × = target
# - **빨간 화살표** = contact normal $\mathbf n$, 길이 $\propto F_n$ (T가 받는 평균 contact force)
# - 회색 화살표 = 질량중심 속도 $\mathbf v_G$

# %%
KEYS = {
    "up": (0, 1), "w": (0, 1), "down": (0, -1), "s": (0, -1),
    "left": (-1, 0), "a": (-1, 0), "right": (1, 0), "d": (1, 0),
}


class KeyboardDemo:

    def __init__(self, sim, target_speed=TARGET_SPEED):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrow, Polygon

        self.plt, self.Polygon, self.Circle, self.FancyArrow = plt, Polygon, Circle, FancyArrow
        self.sim, self.target_speed = sim, target_speed
        self.down, self.press_t = set(), {}
        self.paused, self.dragging, self.running = False, False, True
        self.last_wall = time.perf_counter()

        # matplotlib binds many single keys (s = save, f = fullscreen, r = home, ...) — free them
        for k in list(plt.rcParams):
            if k.startswith("keymap."):
                plt.rcParams[k] = []

        self.fig, self.ax = plt.subplots(figsize=(12.5, 7))
        self.fig.subplots_adjust(left=0.02, right=0.76, top=0.93, bottom=0.07)
        ax = self.ax
        x_min, x_max, y_min, y_max = ARENA
        ax.set_xlim(x_min - 0.2, x_max + 0.2)
        ax.set_ylim(y_min - 0.2, y_max + 0.2)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Polygon([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                             closed=True, fill=False, ec="#404040", lw=3))

        self.goal_patch = Polygon(self._world(sim.goal_pos, sim.goal_angle), closed=True, fill=False,
                                  ec="#3D8C54", lw=2, ls="--")
        self.T_patch = Polygon(self._world(sim.T.pos, sim.T.angle), closed=True, fc="#D9EEFC", ec="#2774B8", lw=3)
        self.P_patch = Circle(sim.pusher.pos, sim.pusher.radius, fc="#FFEBC9", ec="#E97820", lw=3)
        for p in (self.goal_patch, self.T_patch, self.P_patch):
            ax.add_patch(p)
        self.com_dot, = ax.plot([], [], "o", color="#111111", ms=4)
        self.target_mark, = ax.plot([], [], "x", color="#E97820", ms=10, mew=2)
        self.arrows = []
        self.hud = self.fig.text(0.775, 0.90, "", va="top", ha="left", family="monospace", fontsize=10,
                                 bbox=dict(fc="white", ec="#B4C3D6", boxstyle="square,pad=0.6"))
        self.title = ax.set_title("")
        help_text = "\n".join([
            "arrows / WASD : move target",
            "mouse drag    : target",
            "space         : pause",
            "R             : reset",
            "F             : friction on/off",
            "E             : restitution",
            "Q / Esc       : quit",
        ])
        self.fig.text(0.775, 0.36, help_text,
                      va="top", ha="left", family="monospace", fontsize=9, color="#404040")

        c = self.fig.canvas
        c.mpl_connect("key_press_event", self.on_press)
        c.mpl_connect("key_release_event", self.on_release)
        c.mpl_connect("button_press_event", self.on_button)
        c.mpl_connect("button_release_event", self.on_button_release)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("close_event", self.on_close)

    # ------------------------------------------------------------ helpers
    def _world(self, pos, angle):
        return pos + self.sim.T.vertices @ rotation_matrix(angle).T

    def _held(self, key, now):
        return key in self.down or now - self.press_t.get(key, -1.0) < 0.12

    # ------------------------------------------------------------- events
    def on_press(self, event):
        key = (event.key or "").lower()
        now = time.perf_counter()
        if key in KEYS:
            self.down.add(key)
            self.press_t[key] = now
        elif key == " ":
            self.paused = not self.paused
        elif key == "r":
            self.sim.reset()
        elif key == "f":
            self.sim.friction = 0.0 if self.sim.friction > 0 else 0.4
        elif key == "e":
            self.sim.restitution = 0.5 if self.sim.restitution == 0 else 0.0
        elif key in ("q", "escape"):
            self.plt.close(self.fig)

    def on_release(self, event):
        self.down.discard((event.key or "").lower())

    def on_button(self, event):
        if event.button == 1 and event.inaxes is self.ax:
            self.dragging = True
            self.on_motion(event)

    def on_button_release(self, event):
        self.dragging = False

    def on_motion(self, event):
        if self.dragging and event.inaxes is self.ax and event.xdata is not None:
            self.sim.target = np.array([event.xdata, event.ydata])

    def on_close(self, event):
        self.running = False

    # ------------------------------------------------------------- frame
    def update(self, _frame=None, frame_dt=None):
        now = time.perf_counter()
        if frame_dt is None:
            frame_dt = min(now - self.last_wall, 0.05)    # real elapsed time, capped
        self.last_wall = now
        sim = self.sim

        if not self.paused:
            move = np.zeros(2)
            for key, d in KEYS.items():
                if self._held(key, now):
                    move += d
            if move.any():
                x_min, x_max, y_min, y_max = ARENA
                sim.target = np.clip(sim.target + self.target_speed * frame_dt * move / np.linalg.norm(move),
                                     [x_min, y_min], [x_max, y_max])
            sim.advance(frame_dt)

        # ---- draw
        self.T_patch.set_xy(self._world(sim.T.pos, sim.T.angle))
        self.P_patch.center = tuple(sim.pusher.pos)
        self.com_dot.set_data([sim.T.pos[0]], [sim.T.pos[1]])
        self.target_mark.set_data([sim.target[0]], [sim.target[1]])
        for a in self.arrows:
            a.remove()
        self.arrows = []
        v = sim.T.vel
        if np.linalg.norm(v) > 0.02:
            self.arrows.append(self.ax.add_patch(self.FancyArrow(
                *sim.T.pos, *(0.5 * v), width=0.03, head_width=0.12, color="#A5A5A5", length_includes_head=True)))
        Fn, Ft = sim.contact_force
        if sim.last_contact is not None:
            q, n = sim.last_contact
            L = min(0.2 + 0.25 * Fn, 1.5)
            self.arrows.append(self.ax.add_patch(self.FancyArrow(
                *q, *(L * n), width=0.035, head_width=0.13, color="#FF0000", length_includes_head=True)))

        d_err, a_err = sim.goal_error()
        self.hud.set_text(
            f"t        {sim.time:7.2f} s\n"
            f"x_G      ({sim.T.pos[0]:+.2f}, {sim.T.pos[1]:+.2f})\n"
            f"theta    {math.degrees(sim.T.angle):+7.1f} deg\n"
            f"v_G      ({v[0]:+.2f}, {v[1]:+.2f})\n"
            f"omega    {sim.T.omega:+7.2f} rad/s\n"
            f"contact  {'YES' if sim.last_contact is not None else 'no'}\n"
            f"F_n      {Fn:7.2f} N\n"
            f"F_t      {Ft:+7.2f} N\n"
            f"mu = {sim.friction:.1f}   e = {sim.restitution:.1f}\n"
            f"goal err {d_err:.2f} m, {a_err:.1f} deg")
        if d_err < 0.15 and a_err < 10:
            self.title.set_text("Goal reached!"); self.title.set_color("#3D8C54")
        else:
            self.title.set_text("PAUSED" if self.paused else "Keyboard PushT — push the T onto the dashed goal")
            self.title.set_color("#FF0000" if self.paused else "#203864")
        return []

    def run(self):
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self.update, interval=20, blit=False, cache_frame_data=False)
        self.plt.show()
        return self.anim

# %% [markdown]
# ### 실행
#
# 아래 셀이 창을 엽니다.
#
# - Jupyter / VS Code Interactive에서는 `%matplotlib tk` 로 backend를 바꿔 별도 창을 띄웁니다.
# - 터미널에서는 `TkAgg` backend를 씁니다.
# - 환경변수 `PUSHT_NO_GUI=1` 이면 창을 열지 않습니다 (자동 테스트용).

# %%
def open_window():
    try:
        ip = get_ipython()                     # Jupyter / VS Code Interactive
        ip.run_line_magic("matplotlib", "tk")
    except NameError:                          # plain `python pusht_keyboard_sim.py`
        import matplotlib
        matplotlib.use("TkAgg")
    demo = KeyboardDemo(KeyboardPushT())
    demo.run()
    return demo


if not os.environ.get("PUSHT_NO_GUI"):
    demo = open_window()

# %% [markdown]
# ## 6. 해 볼 것
#
# 1. **COM 높이를 밀기 vs 끝을 밀기** — stem 아래쪽을 밀 때와 top bar 끝을 밀 때 HUD의 $\omega$ 를 비교해 보세요.
#    $\lvert\mathbf r\times\mathbf n\rvert$ 가 클수록 같은 push가 회전으로 더 많이 바뀝니다 ($K_n$ 의 회전항).
# 2. **F 키로 마찰 끄기** — $\mu=0$ 이면 impulse가 항상 normal 방향뿐이라 비스듬히 밀 때 pusher가 표면을 미끄러집니다.
# 3. **E 키로 반발계수 0.5** — $v_n^{+}=-e\,v_n^{-}$ 이므로 T가 pusher와 벽에서 튕겨 나갑니다. PushT가 왜 $e\approx0$ 을 쓰는지 느껴 보세요.
# 4. **벽에 대고 돌리기** — 벽도 같은 impulse 식을 쓰는 kinematic body입니다. 벽에 한쪽을 대고 반대쪽을 밀면 회전시키기 쉬워집니다.
# 5. **Goal 맞추기** — 초록 점선 T에 위치 0.15 m, 각도 $10^\circ$ 이내로 맞춰 보세요. 사람이 하는 이 일을 학습하는 것이 PushT policy (예: Diffusion Policy) 의 과제입니다.
