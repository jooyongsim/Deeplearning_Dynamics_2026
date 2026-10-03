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
# # Mini PushT Simulator — Pymunk 없이 직접 구현하는 2D rigid-body contact
#
# 이 파일은 **jupytext "percent" 형식**의 Python 파일입니다.
#
# - `# %% [markdown]` 셀 = 설명 (Markdown + LaTeX)
# - `# %%` 셀 = 실행 코드
#
# 그대로 `python pusht_mini_sim.py` 로 실행할 수도 있고, notebook으로 바꿀 수도 있습니다.
#
# ```bash
# python ../tools/py2ipynb.py pusht_mini_sim.py      # 의존성 없음 -> pusht_mini_sim.ipynb
# jupytext --to notebook pusht_mini_sim.py           # jupytext가 설치되어 있다면
# ```
#
# 목표는 정확한 production physics가 아니라, 아래 연결 고리가 **실제 코드에서 어떻게 이어지는지** 보는 것입니다.
#
# $$
# \boxed{
# \text{action}
# \rightarrow
# \text{pusher movement}
# \rightarrow
# \text{collision}
# \rightarrow
# J
# \rightarrow
# (v,\omega)
# \rightarrow
# (x,\theta)
# }
# $$
#
# 강의(Engineering Dynamics)에서 다룬 식들이 그대로 등장합니다.
#
# | 강의 내용 | 이 코드에서 |
# |---|---|
# | Mass moment of inertia, parallel-axis theorem | `make_T_geometry()` |
# | Relative velocity $\mathbf v_C = \mathbf v_G + \boldsymbol\omega\times\mathbf r$ | `resolve_impulse()` |
# | Impulse–momentum (linear + angular) | `resolve_impulse()` |
# | Coefficient of restitution $e$ | `restitution` |
# | Coulomb friction $\lvert j_t\rvert \le \mu j_n$ | `friction` |

# %% [markdown]
# ## 1. PushT에서 1 step은 정확히 무엇인가?
#
# Policy가 action
#
# $$
# a_t=(x_{\text{target}},\,y_{\text{target}})
# $$
#
# 을 하나 출력했다고 하겠습니다. 한 control step 안에서 physics를 여러 번 계산합니다.
#
# ```text
# action = target pusher position
#              ↓
#        PD controller
#              ↓
#        pusher velocity
#              ↓
#       pusher position
#              ↓
# circle ↔ T collision detection
#              ↓
#  contact point, normal, penetration
#              ↓
#         impulse J
#              ↓
#      T의 v, ω 변경
#              ↓
#        x, y, θ 적분
# ```
#
# 예를 들어 control frequency가 10 Hz이고 physics가 100 Hz이면
#
# $$
# \Delta t_{\text{control}}=0.1,\qquad \Delta t_{\text{physics}}=0.01
# $$
#
# 이므로 하나의 `step()`에서 physics를 10번 수행합니다.
#
# 아래 코드는 `numpy`만 있으면 실행됩니다 (그림은 `matplotlib`).

# %%
import math
from dataclasses import dataclass

import numpy as np

# %% [markdown]
# ## 2. Basic 2D math
#
# 2D에서 각속도는 $z$축 성분 하나뿐인 벡터 $\boldsymbol\omega=(0,0,\omega)$ 입니다.
#
# **2D cross product** (결과는 scalar, $z$ 성분):
#
# $$
# \mathbf a\times\mathbf b = a_x b_y - a_y b_x
# $$
#
# **$\boldsymbol\omega\times\mathbf r$** (강체 위 한 점의 회전 속도):
#
# $$
# (0,0,\omega)\times(r_x,r_y,0) = (-\omega r_y,\ \omega r_x)
# $$
#
# **회전 행렬** — body local 좌표를 world 좌표로:
#
# $$
# R(\theta)=\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}
# $$

# %%
# ============================================================
# 1. Basic 2D math
# ============================================================

def rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s],
        [s,  c]
    ], dtype=float)


def cross2(a, b):
    """
    2D vector cross product
    a x b -> scalar
    """
    return float(
        a[0] * b[1]
        - a[1] * b[0]
    )


def omega_cross_r(omega, r):
    """
    [0,0,omega] x [rx, ry, 0]

    = [-omega*ry, omega*rx]
    """
    return np.array([
        -omega * r[1],
         omega * r[0]
    ])

# %% [markdown]
# ## 3. T polygon + mass moment of inertia
#
# T를 **top rectangle + stem rectangle** 두 개로 나누어 질량중심(COM)과 관성모멘트를 구합니다.
# collision 자체는 8개 꼭짓점을 가진 concave T polygon으로 처리합니다.
#
# **질량중심** (면적 가중 평균, 균일 밀도):
#
# $$
# y_{\text{com}}=\frac{A_{\text{top}}\,y_{\text{top}}+A_{\text{stem}}\,y_{\text{stem}}}{A_{\text{top}}+A_{\text{stem}}}
# $$
#
# **직사각형의 중심축 관성모멘트** (폭 $w$, 높이 $h$):
#
# $$
# I_c=\frac{m\,(w^2+h^2)}{12}
# $$
#
# **Parallel-axis theorem** — 각 직사각형을 T 전체의 COM 기준으로 옮김:
#
# $$
# I = I_c + m\,d^2,\qquad
# I_{\text{T}} = I_{\text{top}} + I_{\text{stem}}
# $$
#
# 마지막으로 꼭짓점 좌표를 $y_{\text{com}}$ 만큼 내려서 **body local origin = COM** 이 되게 합니다.
# 이렇게 해야 이후 $\mathbf r$ (COM → contact) 를 바로 쓸 수 있습니다.

# %%
# ============================================================
# 2. Make T polygon + moment of inertia
# ============================================================

def make_T_geometry(
    W=1.8,
    H=1.8,
    stem_w=0.5,
    top_t=0.5,
    mass=1.0
):
    """
    T shape:

        --------------
        |            |
        --------------
             ||
             ||
             ||
             ||

    T를
    top rectangle + stem rectangle
    으로 생각해서 COM / inertia 계산.

    collision 자체는 아래의 concave T polygon으로 처리.
    """

    # -----------------------
    # areas
    # -----------------------

    top_area = W * top_t

    stem_h = H - top_t
    stem_area = stem_w * stem_h

    total_area = top_area + stem_area


    # -----------------------
    # rectangle centers
    # -----------------------

    y_top = H / 2 - top_t / 2
    y_stem = -top_t / 2


    # -----------------------
    # center of mass
    # -----------------------

    y_com = (
        top_area * y_top
        + stem_area * y_stem
    ) / total_area


    # -----------------------
    # mass split
    # -----------------------

    m_top = mass * top_area / total_area
    m_stem = mass * stem_area / total_area


    # -----------------------
    # inertia
    #
    # rectangle center:
    #
    # I = m(w²+h²)/12
    #
    # parallel axis:
    #
    # I = I_c + m d²
    # -----------------------

    I_top_center = (
        m_top
        * (W**2 + top_t**2)
        / 12
    )

    I_top = (
        I_top_center
        + m_top * (y_top - y_com)**2
    )


    I_stem_center = (
        m_stem
        * (stem_w**2 + stem_h**2)
        / 12
    )

    I_stem = (
        I_stem_center
        + m_stem * (y_stem - y_com)**2
    )


    inertia = I_top + I_stem


    # ========================================================
    # T polygon boundary
    #
    # coordinates before COM correction
    # ========================================================

    vertices = np.array([
        [-W/2,      H/2],

        [ W/2,      H/2],

        [ W/2,      H/2 - top_t],

        [ stem_w/2, H/2 - top_t],

        [ stem_w/2, -H/2],

        [-stem_w/2, -H/2],

        [-stem_w/2, H/2 - top_t],

        [-W/2,      H/2 - top_t],

    ], dtype=float)


    # body local origin = center of mass
    vertices[:, 1] -= y_com


    return vertices, inertia

# %% [markdown]
# ### 확인: 관성모멘트를 수치적으로 검증
#
# 정의 $I=\int r^2\,dm$ 을 Monte Carlo로 근사해서, 위의 parallel-axis 계산과 같은지 봅니다.
# T 내부에 균일하게 점을 뿌리면 각 점의 질량은 $m/N$ 이므로
#
# $$
# I \approx \frac{m}{N}\sum_{k=1}^{N}\lVert\mathbf r_k\rVert^2
# $$
#
# (COM이 원점이므로 $\mathbf r_k$ 는 점의 좌표 그 자체입니다.)

# %%
vertices, inertia = make_T_geometry(mass=1.0)

rng = np.random.default_rng(0)
W, H, stem_w, top_t = 1.8, 1.8, 0.5, 0.5
y_shift = vertices[0, 1] - H / 2          # = -y_com (vertices were shifted by this)

pts = rng.uniform([-W / 2, -H / 2], [W / 2, H / 2], size=(400_000, 2))
in_top = pts[:, 1] >= H / 2 - top_t
in_stem = np.abs(pts[:, 0]) <= stem_w / 2
inside = pts[in_top | in_stem]
inside[:, 1] += y_shift                     # same COM-centred frame as `vertices`

print("COM of samples (should be ~0):", inside.mean(axis=0).round(4))
print("I  parallel-axis :", round(inertia, 5))
print("I  Monte Carlo   :", round(float(np.mean(np.sum(inside**2, axis=1))), 5))

# %% [markdown]
# ## 4. Geometry helpers
#
# ### 4.1 Point → segment 최소거리
#
# 각 polygon edge를 $A\rightarrow B$, circle center를 $P$ 라고 하면 segment 위 가장 가까운 점은
#
# $$
# t=\frac{(P-A)\cdot(B-A)}{\lVert B-A\rVert^2},\qquad t\in[0,1]\ \text{로 clamp}
# $$
#
# $$
# \boxed{Q=A+t\,(B-A)}
# $$
#
# 이것을 T polygon의 모든 edge에 대해 계산하고, 가장 작은 $\lVert P-Q\rVert$ 를 찾습니다.
#
# ### 4.2 Point in polygon (ray casting)
#
# $P$ 에서 $+x$ 방향으로 반직선을 쏴서 polygon edge와 몇 번 교차하는지 셉니다.
# 홀수면 내부, 짝수면 외부입니다. T처럼 **concave polygon**에도 그대로 동작합니다.

# %%
# ============================================================
# 3. Geometry helper
# ============================================================

def closest_point_segment(p, a, b):
    """
    point p에서 line segment a-b로의 가장 가까운 점.
    """

    ab = b - a

    denominator = np.dot(ab, ab)

    if denominator < 1e-12:
        return a.copy()

    t = np.dot(p - a, ab) / denominator

    t = np.clip(t, 0.0, 1.0)

    q = a + t * ab

    return q


def point_in_polygon(p, vertices):
    """
    Ray casting algorithm.

    p가 concave polygon 내부에 있는지 확인.
    """

    x, y = p

    inside = False

    j = len(vertices) - 1

    for i in range(len(vertices)):

        xi, yi = vertices[i]
        xj, yj = vertices[j]

        crossing = (
            (yi > y) != (yj > y)
        )

        if crossing:

            x_intersection = (
                (xj - xi)
                * (y - yi)
                / (yj - yi)
                + xi
            )

            if x < x_intersection:
                inside = not inside

        j = i

    return inside

# %% [markdown]
# ## 5. Circle vs T polygon collision detection
#
# ### 5.1 World → body local frame
#
# T는 회전해 있을 수 있습니다. 그래서 먼저 circle center를 **T의 local coordinate**로 옮깁니다.
#
# $$
# \boxed{\mathbf p_C^{\text{local}} = R^T(\theta)\,\left(\mathbf p_C-\mathbf p_G\right)}
# $$
#
# 이렇게 하면 T가 $\theta=45^\circ$ 회전해 있어도, 항상 **고정된 T와 circle의 관계**만 보면 됩니다.
# robotics / physics simulation에서 아주 자주 쓰는 변환입니다 (world → body local frame).
#
# ### 5.2 Collision 조건과 penetration
#
# circle radius가 $R$, polygon boundary까지 최소거리가 $d$ 이면
#
# $$
# \boxed{d<R}\ \Rightarrow\ \text{collision},\qquad
# \boxed{\delta=R-d}\ \ (\text{penetration depth})
# $$
#
# ```text
#           T
#     ┌──────────
#     │
#     │  Q ●
#     │    ←----- d -----○ P
#     │
# ```
#
# $P$ 는 circle center, $Q$ 는 polygon의 가장 가까운 점입니다.
#
# ### 5.3 Contact normal
#
# normal은 **circle center → T boundary** (= pusher → T) 방향, 즉 T가 받는 normal impulse의 방향입니다.
#
# $$
# \boxed{\mathbf n=\frac{Q-P}{\lVert Q-P\rVert}}
# $$
#
# local에서 구한 normal은 마지막에 $R(\theta)$ 를 곱해 world로 되돌립니다.
# circle center가 T **내부**까지 들어간 비정상 상황이면 방향을 뒤집고 $\delta = R + d$ 로 둡니다.

# %%
# ============================================================
# 4. Circle vs T polygon collision
# ============================================================

def circle_vs_T(
    circle_pos,
    radius,
    body_pos,
    body_angle,
    vertices
):
    """
    return:

        contact_point
        normal
        penetration

    normal은

        pusher -> T

    방향.

    즉 T가 받는 normal impulse 방향.
    """

    R = rotation_matrix(body_angle)


    # --------------------------------------------------------
    # World circle coordinate
    # -> T local coordinate
    # --------------------------------------------------------

    circle_local = (
        R.T
        @ (circle_pos - body_pos)
    )


    # --------------------------------------------------------
    # Polygon의 모든 edge를 조사하여
    # circle center와 가장 가까운 boundary point 찾기
    # --------------------------------------------------------

    best_point = None
    best_dist2 = float("inf")

    N = len(vertices)

    for i in range(N):

        a = vertices[i]
        b = vertices[(i + 1) % N]

        q = closest_point_segment(
            circle_local,
            a,
            b
        )

        delta = q - circle_local

        dist2 = np.dot(delta, delta)

        if dist2 < best_dist2:

            best_dist2 = dist2
            best_point = q


    distance = math.sqrt(best_dist2)


    # --------------------------------------------------------
    # circle center가 T 내부인가?
    # --------------------------------------------------------

    inside = point_in_polygon(
        circle_local,
        vertices
    )


    # circle outside이고
    # boundary까지 거리가 radius보다 크면 collision 없음
    if not inside and distance >= radius:
        return None


    # --------------------------------------------------------
    # Normal direction
    # --------------------------------------------------------

    if distance > 1e-12:

        direction_to_boundary = (
            best_point - circle_local
        ) / distance

    else:

        direction_to_boundary = np.array([
            1.0,
            0.0
        ])


    if inside:

        # 매우 깊게 penetration한 비정상 상황.
        #
        # polygon을 circle로부터 밖으로 밀기 위한 방향.
        normal_local = -direction_to_boundary

        penetration = radius + distance

    else:

        # circle -> polygon
        normal_local = direction_to_boundary

        penetration = radius - distance


    # local normal -> world normal

    normal_world = R @ normal_local

    normal_world /= (
        np.linalg.norm(normal_world)
        + 1e-12
    )


    # contact point:
    #
    # polygon boundary 위의 nearest point

    contact_world = (
        body_pos
        + R @ best_point
    )


    return (
        contact_world,
        normal_world,
        penetration
    )

# %% [markdown]
# ### 확인: collision detection 동작
#
# T를 원점에 두고, circle을 몇 군데 놓아 봅니다.
# 같은 위치라도 T를 $45^\circ$ 돌리면 결과가 달라지는 것도 확인합니다.

# %%
radius = 0.18
for label, c, ang in [
    ("left of stem, touching", np.array([-0.40, -0.30]), 0.0),
    ("far away",               np.array([-1.50,  0.00]), 0.0),
    ("same point, T at 45 deg", np.array([-0.40, -0.30]), math.pi / 4),
]:
    hit = circle_vs_T(c, radius, np.zeros(2), ang, vertices)
    if hit is None:
        print(f"{label:26s} -> no contact")
    else:
        q, n, pen = hit
        print(f"{label:26s} -> contact {q.round(3)}, normal {n.round(3)}, penetration {pen:.3f}")

# %% [markdown]
# ## 6. Dynamic state
#
# - `TBody` : dynamic body — 위치 $\mathbf x_G$, 각도 $\theta$, 속도 $\mathbf v_G$, 각속도 $\omega$, 질량 $m$, 관성모멘트 $I$
# - `Pusher` : **kinematic body** — PD controller가 속도를 정하고, 충돌 impulse를 받아도 속도가 변하지 않습니다.

# %%
# ============================================================
# 5. Dynamic state
# ============================================================

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

# %% [markdown]
# ## 7. Collision impulse — PushT physics의 핵심
#
# ### 7.1 Contact point의 속도
#
# 질량중심 속도가 $\mathbf v_G$ 라도 contact point의 속도는 다릅니다. Meriam에서 본 relative velocity:
#
# $$
# \boxed{\mathbf v_C=\mathbf v_G+\boldsymbol\omega\times\mathbf r},
# \qquad \mathbf r = \mathbf x_{\text{contact}}-\mathbf x_G
# $$
#
# ```text
#        contact
#           ●
#          /
#         / r
#        /
#       G
# ```
#
# ### 7.2 상대속도와 normal 성분
#
# Pusher 속도가 $\mathbf v_P$ 이면
#
# $$
# \boxed{\mathbf v_{\text{rel}}=\mathbf v_C-\mathbf v_P},\qquad
# \boxed{v_n=\mathbf v_{\text{rel}}\cdot\mathbf n}
# $$
#
# normal을 pusher → T 방향으로 정의했으므로 $v_n<0$ 이면 두 물체가 **서로 접근**하고 있다는 뜻입니다.
#
# ### 7.3 Collision 조건 (restitution)
#
# 충돌 뒤에는 서로 계속 파고들면 안 됩니다. Coefficient of restitution $e$ 를 쓰면
#
# $$
# \boxed{v_n^{+}=-e\,v_n^{-}}
# $$
#
# PushT처럼 거의 비탄성 충돌이면 $e\approx0$ 이고, 충돌 뒤 $v_n^{+}\approx0$ — normal 방향으로 더 이상 파고들지 않습니다.
#
# ### 7.4 Impulse가 속도를 바꾸는 방식
#
# Normal impulse를 $\mathbf J = j\,\mathbf n$ 이라 하면, **linear impulse–momentum**:
#
# $$
# m\mathbf v^{+}=m\mathbf v^{-}+\mathbf J
# \quad\Rightarrow\quad
# \boxed{\Delta\mathbf v_G=\frac{\mathbf J}{m}}
# $$
#
# Impulse가 COM을 지나지 않으면 **angular impulse** 가 생깁니다. 2D에서 $L=I\omega$ 이므로
#
# $$
# I\,\Delta\omega=\mathbf r\times\mathbf J
# \quad\Rightarrow\quad
# \boxed{\Delta\omega=\frac{\mathbf r\times\mathbf J}{I}}
# $$
#
# 이 한 줄이 바로 "pusher가 T의 옆부분을 밀면 T가 회전하는 이유" 입니다.
#
# ### 7.5 Scalar $j$ 계산 — effective inverse mass
#
# Impulse를 가하면 $\mathbf v_G$ 뿐만 아니라 $\omega$ 도 변합니다. 둘을 모두 고려하면 contact normal 방향의 effective inverse mass는
#
# $$
# \boxed{K_n=\frac1m+\frac{(\mathbf r\times\mathbf n)^2}{I}}
# $$
#
# 이고, $v_n^{+}=-e\,v_n^{-}$ 을 만족하는 impulse 크기는
#
# $$
# \boxed{j=-\frac{(1+e)\,v_n}{K_n}}
# $$
#
# **왜 분모에 회전항이 들어가는가?** contact가 COM 근처면 $\mathbf r\times\mathbf n\approx0$, $K_n\approx 1/m$ 이고 대부분 translation이 됩니다.
# 멀리 떨어진 곳을 밀면 $\lvert\mathbf r\times\mathbf n\rvert$ 가 커져서 impulse의 일부가 회전으로 들어갑니다.
#
# $$
# \boxed{\text{same push}+\text{different contact position}\Rightarrow\text{different translation / rotation}}
# $$
#
# 이것이 PushT task의 본질 중 하나입니다 (아래 실험 8에서 직접 확인).
#
# ### 7.6 Pusher도 dynamic body라면?
#
# 일반적으로 두 rigid body A, B가 충돌하면
#
# $$
# K_n=\frac1{m_A}+\frac1{m_B}+\frac{(\mathbf r_A\times\mathbf n)^2}{I_A}+\frac{(\mathbf r_B\times\mathbf n)^2}{I_B}
# $$
#
# 하지만 kinematic pusher는 $1/m_P = 0$, $1/I_P=0$ 처럼 취급하므로 T의 항만 남습니다.
#
# ### 7.7 Friction impulse $j_t$
#
# Normal impulse만 쓰면 pusher가 T 표면에서 너무 쉽게 미끄러집니다. tangent를
#
# $$
# \mathbf n=(n_x,n_y)\ \Rightarrow\ \boxed{\mathbf t=(-n_y,\ n_x)}
# $$
#
# 로 정의하고, 상대속도의 tangent 성분 $v_t=\mathbf v_{\text{rel}}\cdot\mathbf t$ 을 없애는 impulse는
#
# $$
# j_t^{*}=-\frac{v_t}{K_t},\qquad K_t=\frac1m+\frac{(\mathbf r\times\mathbf t)^2}{I}
# $$
#
# 그러나 **Coulomb friction** 때문에 무한히 큰 마찰은 쓸 수 없으므로 clamp 합니다.
#
# $$
# \boxed{\lvert j_t\rvert\le\mu\,j_n}
# $$
#
# ### 7.8 Position correction
#
# 이론적인 충돌 순간에는 penetration $=0$ 이어야 하지만, discrete simulation은 $t\rightarrow t+\Delta t$ 로 점프하기 때문에 이미 겹친 상태가 생깁니다.
#
# ```text
# previous          next
#
# ○   | T            ○|T
#                      ↑
#                    already overlapping
# ```
#
# velocity impulse만으로는 이미 생긴 겹침이 남으므로, normal 방향으로 직접 조금 밀어냅니다 (**penetration correction**).
#
# $$
# \Delta\mathbf x_G=\beta\,\max(\delta-\text{slop},\,0)\,\mathbf n
# $$

# %%
# ============================================================
# 6. Collision impulse
# ============================================================

def resolve_impulse(
    body,
    pusher,
    contact,
    restitution=0.0,
    friction=0.4
):

    contact_point, normal, penetration = contact


    # ========================================================
    # Contact position relative to T center of mass
    # ========================================================

    r = (
        contact_point
        - body.pos
    )


    # ========================================================
    # velocity of contact point on T
    #
    # v_c = v_G + omega x r
    # ========================================================

    v_contact_T = (
        body.vel
        + omega_cross_r(
            body.omega,
            r
        )
    )


    # ========================================================
    # relative velocity
    #
    # T contact velocity
    # -
    # pusher velocity
    # ========================================================

    v_relative = (
        v_contact_T
        - pusher.vel
    )


    # normal relative velocity

    v_normal = float(
        np.dot(
            v_relative,
            normal
        )
    )


    j_normal = 0.0
    j_tangent = 0.0


    # ========================================================
    # vn < 0 means approaching
    # ========================================================

    if v_normal < 0:

        # ====================================================
        # Effective inverse mass
        #
        # K =
        # 1/m
        # +
        # (r x n)^2 / I
        #
        # pusher = kinematic body
        # -> pusher inverse mass = 0
        # ====================================================

        K_normal = (
            1.0 / body.mass
            +
            cross2(r, normal)**2
            / body.inertia
        )


        # ====================================================
        # normal impulse magnitude
        #
        # j =
        # -(1+e) vn / K
        # ====================================================

        j_normal = (
            -(1.0 + restitution)
            * v_normal
            / K_normal
        )


        impulse_normal = (
            j_normal
            * normal
        )


        # ====================================================
        # linear velocity change
        #
        # Δv = J / m
        # ====================================================

        body.vel += (
            impulse_normal
            / body.mass
        )


        # ====================================================
        # angular velocity change
        #
        # Δω = (r x J) / I
        # ====================================================

        body.omega += (
            cross2(
                r,
                impulse_normal
            )
            / body.inertia
        )


        # ====================================================
        # friction impulse
        # ====================================================

        # normal impulse 적용 후
        # 새로운 contact velocity 계산

        v_contact_T = (
            body.vel
            + omega_cross_r(
                body.omega,
                r
            )
        )

        v_relative = (
            v_contact_T
            - pusher.vel
        )


        tangent = np.array([
            -normal[1],
             normal[0]
        ])


        v_tangent = float(
            np.dot(
                v_relative,
                tangent
            )
        )


        K_tangent = (
            1.0 / body.mass
            +
            cross2(r, tangent)**2
            / body.inertia
        )


        j_tangent_free = (
            -v_tangent
            / K_tangent
        )


        # Coulomb friction
        #
        # |Jt| <= mu * Jn

        j_tangent = float(
            np.clip(
                j_tangent_free,

                -friction * j_normal,

                 friction * j_normal
            )
        )


        impulse_tangent = (
            j_tangent
            * tangent
        )


        body.vel += (
            impulse_tangent
            / body.mass
        )


        body.omega += (
            cross2(
                r,
                impulse_tangent
            )
            / body.inertia
        )


    # ========================================================
    # Position correction
    #
    # 이미 물체가 조금 겹친 경우
    # 바로 분리해준다.
    # ========================================================

    slop = 1e-4
    beta = 0.8

    correction = (
        beta
        * max(
            penetration - slop,
            0.0
        )
    )


    body.pos += (
        correction
        * normal
    )


    return (
        j_normal,
        j_tangent
    )

# %% [markdown]
# ## 8. 실험: 같은 push, 다른 contact 높이
#
# 7.5의 주장을 숫자로 확인합니다. 정지한 T의 **stem 왼쪽 면** ($x=-0.25$) 을 pusher가 같은 속도 $\mathbf v_P=(1,0)$ 로 밀고,
# 접촉 높이 $y$ 만 바꿉니다. 마찰은 끄고 ($\mu=0$), $e=0$ 입니다.
#
# 이때 $\mathbf n=(1,0)$, $\mathbf r=(-0.25,\,y)$ 이므로 $\mathbf r\times\mathbf n=-y$ 이고
#
# $$
# K_n=\frac1m+\frac{y^2}{I},\qquad j=\frac{1}{K_n},\qquad
# \Delta v_x=\frac{j}{m},\qquad \Delta\omega=\frac{-y\,j}{I}
# $$
#
# - $y=0$ (COM 높이) : 회전 없이 순수 translation, $\Delta v_x = 1$
# - $\lvert y\rvert$ 가 클수록 : $j$ 와 $\Delta v_x$ 는 줄고, 그만큼 $\lvert\Delta\omega\rvert$ 가 커짐

# %%
stem_left = -0.25
y_bottom = vertices[:, 1].min()             # bottom of the stem (COM frame)
y_stem_top = vertices[3, 1]                 # where the stem meets the top bar

print(f"{'contact y':>10s} {'K_n':>8s} {'j_n':>8s} {'dv_x':>8s} {'d_omega':>9s}")
for y in np.sort(np.r_[np.linspace(y_bottom + 0.05, y_stem_top - 0.05, 5), 0.0]):   # include y = 0 (COM height)
    body = TBody(pos=np.zeros(2), angle=0.0, vel=np.zeros(2), omega=0.0,
                 mass=1.0, inertia=inertia, vertices=vertices)
    pusher = Pusher(pos=np.array([stem_left - 0.18, y]), vel=np.array([1.0, 0.0]), radius=0.18)
    contact = (np.array([stem_left, y]), np.array([1.0, 0.0]), 0.0)   # point, normal, penetration
    K_n = 1.0 / body.mass + cross2(contact[0] - body.pos, contact[1]) ** 2 / body.inertia
    jn, _ = resolve_impulse(body, pusher, contact, restitution=0.0, friction=0.0)
    print(f"{y:10.3f} {K_n:8.3f} {jn:8.3f} {body.vel[0]:8.3f} {body.omega:9.3f}")

# %% [markdown]
# ## 9. Mini PushT simulator
#
# 한 번의 `step(action)` 이 하는 일 (substep마다 반복):
#
# **A. Pusher PD control**
#
# $$
# \mathbf a_P = K_p\,(\mathbf x^{*}-\mathbf x_P) - K_d\,\mathbf v_P,\qquad
# \mathbf v_P \leftarrow \mathbf v_P+\mathbf a_P\Delta t,\qquad
# \mathbf x_P \leftarrow \mathbf x_P+\mathbf v_P\Delta t
# $$
#
# (속도는 `max_speed` 로 제한)
#
# **B. T damping** — 바닥 마찰을 흉내 낸 지수 감쇠
#
# $$
# \mathbf v_G \leftarrow \mathbf v_G\,e^{-c_v\Delta t},\qquad
# \omega \leftarrow \omega\,e^{-c_\omega\Delta t}
# $$
#
# **C. T 적분** (semi-implicit Euler)
#
# $$
# \mathbf x_G \leftarrow \mathbf x_G+\mathbf v_G\Delta t,\qquad
# \theta \leftarrow \theta+\omega\Delta t
# $$
#
# **D. Contact solver** — `circle_vs_T` → `resolve_impulse` 를 최대 3번 반복
#
# 왜 여러 번 돌리나? 한 번의 impulse로 충돌이 완벽하게 해결되지 않는 경우가 많기 때문입니다
# (friction, rotation, penetration, 여러 contact, corner contact).
# Pymunk, Box2D 같은 physics engine도 훨씬 정교한 **iterative constraint solver** 를 사용합니다.

# %%
# ============================================================
# 7. Mini PushT simulator
# ============================================================

class MiniPushT:

    def __init__(self):

        mass = 1.0

        vertices, inertia = make_T_geometry(
            mass=mass
        )


        self.T = TBody(

            pos=np.array([
                0.0,
                0.0
            ]),

            angle=0.0,

            vel=np.zeros(2),

            omega=0.0,

            mass=mass,

            inertia=inertia,

            vertices=vertices
        )


        # pusher starts close to T
        # so that one control step can create collision

        self.pusher = Pusher(

            pos=np.array([
                -1.15,
                 0.30
            ]),

            vel=np.zeros(2),

            radius=0.18
        )


        # ----------------------------------------------------
        # simulation frequency
        # ----------------------------------------------------

        self.control_dt = 0.1

        self.physics_dt = 0.01

        self.substeps = int(
            self.control_dt
            / self.physics_dt
        )


        # ----------------------------------------------------
        # pusher PD controller
        # ----------------------------------------------------

        self.kp = 80.0

        self.kd = 18.0

        self.max_speed = 3.0


        # ----------------------------------------------------
        # T damping
        # ----------------------------------------------------

        self.linear_damping = 1.5

        self.angular_damping = 1.5


        # ----------------------------------------------------
        # collision properties
        # ----------------------------------------------------

        self.restitution = 0.0

        self.friction = 0.4


    def observation(self):

        return np.array([
            self.pusher.pos[0],
            self.pusher.pos[1],

            self.T.pos[0],
            self.T.pos[1],

            self.T.angle
        ])


    def step(self, action):

        """
        action:

            [target_x, target_y]

        One RL/control step.
        """

        target = np.asarray(
            action,
            dtype=float
        )


        collision_impulses = []


        # ====================================================
        # One control step
        #
        # = multiple physics steps
        # ====================================================

        for _ in range(
            self.substeps
        ):

            dt = self.physics_dt


            # =================================================
            # A. Pusher PD control
            # =================================================

            acceleration = (

                self.kp
                * (
                    target
                    - self.pusher.pos
                )

                -

                self.kd
                * self.pusher.vel
            )


            self.pusher.vel += (
                acceleration
                * dt
            )


            # speed limit

            speed = np.linalg.norm(
                self.pusher.vel
            )


            if speed > self.max_speed:

                self.pusher.vel *= (
                    self.max_speed
                    / speed
                )


            # pusher integration

            self.pusher.pos += (
                self.pusher.vel
                * dt
            )


            # =================================================
            # B. T damping
            # =================================================

            self.T.vel *= math.exp(
                -self.linear_damping
                * dt
            )


            self.T.omega *= math.exp(
                -self.angular_damping
                * dt
            )


            # =================================================
            # C. Integrate T
            # =================================================

            self.T.pos += (
                self.T.vel
                * dt
            )


            self.T.angle += (
                self.T.omega
                * dt
            )


            # =================================================
            # D. Contact solver
            #
            # repeated a few times for stability
            # =================================================

            for _ in range(3):

                contact = circle_vs_T(

                    self.pusher.pos,

                    self.pusher.radius,

                    self.T.pos,

                    self.T.angle,

                    self.T.vertices
                )


                if contact is None:
                    break


                jn, jt = resolve_impulse(

                    self.T,

                    self.pusher,

                    contact,

                    restitution=
                    self.restitution,

                    friction=
                    self.friction
                )


                collision_impulses.append(
                    (jn, jt)
                )


        return (
            self.observation(),
            collision_impulses
        )

# %% [markdown]
# ## 10. Test: one control step
#
# Pusher는 T의 왼쪽 ($x=-1.15$, $y=0.30$) 에서 시작하고, target을 오른쪽 ($x=0$) 으로 줍니다.
# $y=0.30$ 은 COM보다 위이므로, push와 함께 회전도 생겨야 합니다.

# %%
# ============================================================
# 8. Test one step
# ============================================================

sim = MiniPushT()


print(
    "Before:"
)

print(
    "obs =",
    sim.observation()
)

print(
    "T velocity =",
    sim.T.vel
)

print(
    "T omega =",
    sim.T.omega
)


# pusher target:
# move to the right

action = np.array([
    0.0,
    0.30
])


obs, impulses = sim.step(
    action
)


print("\nAfter one step:")

print(
    "obs =",
    obs
)

print(
    "T velocity =",
    sim.T.vel
)

print(
    "T omega =",
    sim.T.omega
)

print(
    "impulses =",
    impulses
)

# %% [markdown]
# ## 11. 실험: 여러 step rollout과 시각화
#
# 같은 target을 15 step (1.5 s) 동안 유지하면서 T와 pusher를 그립니다.
# 색이 진해질수록 나중 시점입니다. T가 오른쪽으로 밀리면서 동시에 회전하는 것이 보여야 합니다.

# %%
import matplotlib.pyplot as plt

sim = MiniPushT()
frames = [(sim.T.pos.copy(), sim.T.angle, sim.pusher.pos.copy())]
for _ in range(15):
    sim.step(np.array([0.6, 0.30]))
    frames.append((sim.T.pos.copy(), sim.T.angle, sim.pusher.pos.copy()))

fig, ax = plt.subplots(figsize=(7, 5))
for k, (pos, ang, ppos) in enumerate(frames):
    if k % 3:
        continue
    shade = 0.25 + 0.75 * k / (len(frames) - 1)
    world = pos + vertices @ rotation_matrix(ang).T
    ax.fill(*np.vstack([world, world[:1]]).T, fc=(0.85, 0.93, 0.99, 0.35), ec=(0.15, 0.45, 0.72, shade), lw=2)
    ax.add_patch(plt.Circle(ppos, sim.pusher.radius, color=(0.91, 0.47, 0.13, shade)))
    ax.plot(*pos, "k.", ms=4)
path = np.array([f[2] for f in frames])
ax.plot(path[:, 0], path[:, 1], "--", color="0.5", lw=1, label="pusher path")
ax.set_aspect("equal")
ax.set_title("Mini PushT: 15 control steps toward target (0.6, 0.3)")
ax.legend(loc="lower right")
plt.show()

print("final T position:", sim.T.pos.round(3), " angle (deg):", round(math.degrees(sim.T.angle), 1))

# %% [markdown]
# ## 12. 정리 — PushT 1 step을 수식으로
#
# 1. **Action** : $a_t=(x^{*},y^{*})$
#
# 2. **Pusher controller**
#
# $$
# \mathbf a_P=K_p(\mathbf x^{*}-\mathbf x_P)-K_d\mathbf v_P
# $$
#
# 3. **Collision detection**
#
# $$
# (\text{circle},\,T)\rightarrow(\mathbf q,\,\mathbf n,\,\delta)
# $$
#
# 4. **Contact velocity / normal velocity**
#
# $$
# \mathbf v_C=\mathbf v_G+\boldsymbol\omega\times\mathbf r,\qquad
# v_n=(\mathbf v_C-\mathbf v_P)\cdot\mathbf n
# $$
#
# 5. **Impulse**
#
# $$
# j_n=-\frac{(1+e)\,v_n}{K_n},\qquad \mathbf J_n=j_n\mathbf n
# $$
#
# 6. **Velocity update**
#
# $$
# \mathbf v_G\leftarrow\mathbf v_G+\frac{\mathbf J_n}{m},\qquad
# \omega\leftarrow\omega+\frac{\mathbf r\times\mathbf J_n}{I}
# $$
#
# 7. **Integration**
#
# $$
# \mathbf x_G\leftarrow\mathbf x_G+\mathbf v_G\Delta t,\qquad
# \theta\leftarrow\theta+\omega\Delta t
# $$
#
# ### 가장 중요한 4개의 식
#
# $$
# \boxed{\mathbf v_C=\mathbf v_G+\boldsymbol\omega\times\mathbf r}
# $$
#
# $$
# \boxed{j=-\frac{(1+e)\,v_n}{1/m+(\mathbf r\times\mathbf n)^2/I}}
# $$
#
# $$
# \boxed{\mathbf v_G^{+}=\mathbf v_G^{-}+\frac{\mathbf J}{m}}
# $$
#
# $$
# \boxed{\omega^{+}=\omega^{-}+\frac{\mathbf r\times\mathbf J}{I}}
# $$
#
# 이 네 식이 collision detection에서 구한 `(contact point, normal)` 을 실제 T-block의 translation과 rotation으로 연결하는 다리입니다.
#
# 이 예제는 의도적으로 단순화했습니다. Pymunk / Box2D 수준으로 가면 multiple contact manifold, warm starting,
# static/dynamic friction, constraint stabilization, continuous collision detection 등이 추가됩니다.
# 하지만 PushT의 물리를 이해하고 작은 simulator를 직접 만드는 단계에서는 이 정도가 가장 좋은 출발점입니다.
