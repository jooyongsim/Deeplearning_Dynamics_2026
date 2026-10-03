가능합니다. 아래 코드는 **Pymunk 없이 PushT의 핵심 물리만 직접 구현한 교육용 mini-simulator**입니다. 핵심 목적은 정확한 production physics보다,

\[
\boxed{
action
\rightarrow
pusher\ movement
\rightarrow
collision
\rightarrow
J
\rightarrow
(v,\omega)
\rightarrow
(x,\theta)
}
\]

가 실제 코드에서 어떻게 연결되는지를 보는 것입니다.

---

# 1. PushT에서 1 step은 정확히 무엇인가?

Policy가 action

\[
a_t=(x_{\text{target}},y_{\text{target}})
\]

을 하나 출력했다고 하겠습니다.

한 control step 안에서 physics를 여러 번 계산합니다.

```text
action = target pusher position
             ↓
       PD controller
             ↓
       pusher velocity
             ↓
      pusher position
             ↓
circle ↔ T collision detection
             ↓
 contact point, normal, penetration
             ↓
        impulse J
             ↓
     T의 v, ω 변경
             ↓
       x, y, θ 적분
```

예를 들어 control frequency가 10 Hz이고 physics가 100 Hz이면:

\[
\Delta t_{control}=0.1
\]

\[
\Delta t_{physics}=0.01
\]

이므로 하나의 `step()`에서 physics를 10번 수행합니다.

---

# 2. 전체 최소 구현 코드

아래 코드는 `numpy`만 있으면 실행할 수 있습니다.

```python
import math
from dataclasses import dataclass

import numpy as np


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
```

이제 이 코드에서 가장 중요한 부분만 하나씩 보면 됩니다.

---

# 3. \(J\)는 어떻게 구하는가?

여기가 PushT physics의 핵심입니다.

충돌 직전에 T의 contact point 속도를 먼저 구합니다.

질량중심 속도가

\[
\mathbf v_G
\]

라고 해도 contact point의 속도는 다릅니다.

Meriam에서 본

\[
\boxed{
\mathbf v_C
=
\mathbf v_G
+
\boldsymbol\omega\times\mathbf r
}
\]

를 사용합니다.

코드에서는:

```python
v_contact_T = (
    body.vel
    + omega_cross_r(
        body.omega,
        r
    )
)
```

여기서

```python
r = contact_point - body.pos
```

입니다.

즉:

```text
       contact
          ●
         /
        / r
       /
      G
```

---

## 상대속도

Pusher의 속도가

\[
\mathbf v_P
\]

라면

\[
\boxed{
\mathbf v_{rel}
=
\mathbf v_C-\mathbf v_P
}
\]

입니다.

normal 방향만 보면

\[
\boxed{
v_n
=
\mathbf v_{rel}\cdot\mathbf n
}
\]

입니다.

코드:

```python
v_relative = (
    v_contact_T
    - pusher.vel
)

v_normal = np.dot(
    v_relative,
    normal
)
```

우리 코드에서는 normal을

\[
\boxed{\text{pusher}\rightarrow T}
\]

방향으로 정의했습니다.

따라서

\[
v_n<0
\]

이면 두 물체가 서로 접근하고 있다는 뜻입니다.

---

# 4. Collision impulse 조건

충돌 뒤에는 물체가 서로 계속 파고들면 안 됩니다.

restitution \(e\)를 사용하면

\[
\boxed{
v_n^+
=
-e v_n^-
}
\]

라고 놓습니다.

PushT처럼 거의 비탄성 충돌이면 보통

\[
e\approx0
\]

로 둘 수 있습니다.

그러면 충돌 뒤에는

\[
v_n^+\approx0
\]

가 됩니다.

즉 normal 방향으로 더 이상 서로 파고들지 않습니다.

---

# 5. Impulse가 속도를 어떻게 바꾸나?

Normal impulse를

\[
\boxed{
\mathbf J
=
j\mathbf n
}
\]

이라고 합시다.

그러면 linear momentum 관계로

\[
m\mathbf v^+
=
m\mathbf v^-
+
\mathbf J
\]

따라서

\[
\boxed{
\Delta\mathbf v_G
=
\frac{\mathbf J}{m}
}
\]

입니다.

코드:

```python
body.vel += (
    impulse_normal
    / body.mass
)
```

---

# 6. 그런데 PushT에서는 회전도 발생한다

Impulse가 COM을 정확히 지나지 않는다면 torque impulse가 생깁니다.

\[
\boxed{
\Delta L
=
\mathbf r\times\mathbf J
}
\]

2D에서는

\[
L=I\omega
\]

이므로

\[
I\Delta\omega
=
\mathbf r\times\mathbf J
\]

따라서

\[
\boxed{
\Delta\omega
=
\frac{
\mathbf r\times\mathbf J
}{
I
}
}
\]

입니다.

코드:

```python
body.omega += (
    cross2(
        r,
        impulse_normal
    )
    / body.inertia
)
```

이 한 줄이 바로

> “pusher가 T의 옆부분을 밀면 T가 회전하는 이유”

입니다.

---

# 7. 그러면 scalar \(j\) 자체는 어떻게 계산하나?

Impulse를 가하면

\[
\mathbf v_G
\]

뿐만 아니라

\[
\omega
\]

도 변합니다.

이 둘을 모두 고려하면 contact normal 방향의 effective inverse mass가

\[
\boxed{
K_n
=
\frac1m
+
\frac{
(\mathbf r\times\mathbf n)^2
}{I}
}
\]

가 됩니다.

그래서

\[
\boxed{
j
=
-\frac{
(1+e)v_n
}{
K_n
}
}
\]

입니다.

코드가 바로:

```python
K_normal = (
    1.0 / body.mass
    +
    cross2(r, normal)**2
    / body.inertia
)

j_normal = (
    -(1.0 + restitution)
    * v_normal
    / K_normal
)
```

입니다.

이 식은 정말 중요합니다.

---

# 8. 왜 denominator에 회전항이 들어가는가?

만약 contact가 COM 근처라면

\[
r\times n\approx0
\]

입니다.

그러면

\[
K_n\approx\frac1m
\]

이고 대부분 translation이 발생합니다.

반대로 멀리 떨어진 곳을 밀면

\[
|r\times n|
\]

이 커집니다.

그래서 impulse의 일부가 회전운동으로 들어갑니다.

즉

\[
\boxed{
\text{same push}
+
\text{different contact position}
\Rightarrow
\text{different translation/rotation}
}
\]

입니다.

이게 PushT task의 본질 중 하나입니다.

---

# 9. Pusher도 dynamic body라면?

우리 예제에서는 pusher를 **kinematic body**로 봅니다.

그래서 pusher는 impulse를 받아도 속도가 변하지 않습니다.

일반적으로 두 rigid body가 충돌한다면 denominator는

\[
\boxed{
K_n
=
\frac1{m_A}
+
\frac1{m_B}
+
\frac{(r_A\times n)^2}{I_A}
+
\frac{(r_B\times n)^2}{I_B}
}
\]

입니다.

하지만 kinematic pusher는

\[
\frac1{m_P}=0
\]

처럼 취급합니다.

따라서 우리 PushT에서는

\[
K_n
=
\frac1{m_T}
+
\frac{(r_T\times n)^2}{I_T}
\]

만 남습니다.

---

# 10. Circle과 T polygon collision detection은 어떻게 하는가?

이 부분도 중요합니다.

T는 concave polygon입니다.

그래서 단순 convex SAT보다 여기서는 더 직관적인 방법을 썼습니다.

먼저 circle center를 T의 local coordinate로 변환합니다.

World:

\[
\mathbf p_C
\]

T position:

\[
\mathbf p_G
\]

T orientation:

\[
R(\theta)
\]

이면

\[
\boxed{
\mathbf p_C^{local}
=
R^T
(
\mathbf p_C-\mathbf p_G
)
}
\]

입니다.

코드:

```python
R = rotation_matrix(body_angle)

circle_local = (
    R.T
    @ (circle_pos - body_pos)
)
```

이렇게 하면 회전된 T를 다룰 필요가 없어집니다.

즉 항상 이런 고정된 T와 circle의 관계만 보면 됩니다.

```text
----------------
|              |
------    ------
     |    |
     |    |       ○ circle
     |    |
     |    |
     ------
```

---

# 11. Circle center와 polygon edge의 최소거리

각 polygon edge를

\[
A\rightarrow B
\]

라고 합시다.

circle center를 \(P\)라고 하면 segment 상 가장 가까운 점은

\[
t=
\frac{
(P-A)\cdot(B-A)
}{
\|B-A\|^2
}
\]

입니다.

그리고

\[
t\in[0,1]
\]

로 clamp합니다.

\[
\boxed{
Q=A+t(B-A)
}
\]

코드는:

```python
def closest_point_segment(p, a, b):

    ab = b - a

    t = (
        np.dot(p - a, ab)
        /
        np.dot(ab, ab)
    )

    t = np.clip(
        t,
        0.0,
        1.0
    )

    q = a + t * ab

    return q
```

이걸 T polygon의 모든 edge에 대해 합니다.

그리고 가장 작은

\[
\|P-Q\|
\]

를 찾습니다.

---

# 12. Collision 조건

circle radius가 \(R\)이고 polygon boundary까지 최소거리가 \(d\)라면

\[
\boxed{
d<R
}
\]

이면 collision입니다.

penetration depth는

\[
\boxed{
\delta=R-d
}
\]

입니다.

예를 들어:

```text
          T
    ┌──────────
    │
    │  Q ●
    │    ←----- d -----○ P
    │
```

\(P\)는 circle center,

\(Q\)는 polygon의 가장 가까운 점입니다.

---

# 13. Contact normal

우리 normal은

\[
\boxed{
\text{circle center}
\rightarrow
\text{T boundary}
}
\]

방향입니다.

따라서

\[
\boxed{
\mathbf n
=
\frac{
Q-P
}{
\|Q-P\|
}
}
\]

입니다.

코드:

```python
direction_to_boundary = (
    best_point
    - circle_local
) / distance
```

그리고 world coordinate로:

```python
normal_world = (
    R @ direction_to_boundary
)
```

합니다.

---

# 14. T가 회전한 경우에도 같은 collision algorithm이 가능한 이유

예를 들어 T가

\[
\theta=45^\circ
\]

회전했다고 합시다.

World coordinate에서는:

```text
     /
    / ----
   / |
     |
```

처럼 복잡해집니다.

하지만 circle을 T local frame으로 변환하면 다시

```text
--------------
|            |
--------------
      ||
      ||
      ||
```

형태가 됩니다.

그래서 robotics/physics simulation에서는 굉장히 자주

\[
\boxed{
\text{world}
\rightarrow
\text{body local frame}
}
\]

변환을 사용합니다.

---

# 15. Friction impulse \(J_t\)

Normal impulse만 사용하면 pusher가 T 표면에서 너무 쉽게 미끄러집니다.

그래서 tangent를 정의합니다.

normal이

\[
n=(n_x,n_y)
\]

라면

\[
\boxed{
t=(-n_y,n_x)
}
\]

입니다.

코드:

```python
tangent = np.array([
    -normal[1],
     normal[0]
])
```

상대속도의 tangent component는

\[
v_t
=
v_{rel}\cdot t
\]

입니다.

이를 없애기 위한 impulse는

\[
j_t^*
=
-\frac{v_t}{K_t}
\]

입니다.

그러나 Coulomb friction 때문에 무한히 큰 friction은 사용할 수 없습니다.

\[
\boxed{
|j_t|
\le
\mu j_n
}
\]

그래서 코드에서:

```python
j_tangent = np.clip(
    j_tangent_free,
    -friction * j_normal,
     friction * j_normal
)
```

합니다.

---

# 16. 결국 하나의 contact에서 일어나는 일

이 부분만 확실히 이해하면 됩니다.

Pusher가

```text
                    T
               ┌─────────
Pusher ○ ----> │
               │
               │
```

이렇게 밀면 contact normal이 거의 오른쪽입니다.

Impulse:

\[
\mathbf J=j_n\mathbf n
\]

가 T에 들어갑니다.

그러면

\[
\boxed{
\Delta v
=
\frac Jm
}
\]

으로 T가 오른쪽으로 움직이고,

접촉점이 COM 위쪽이면

\[
r\times J\neq0
\]

이므로

\[
\boxed{
\Delta\omega
=
\frac{r\times J}{I}
}
\]

가 생겨 회전합니다.

즉 PushT의 핵심 물리가 딱 이것입니다:

\[
\boxed{
\mathbf J
\Rightarrow
\begin{cases}
\Delta\mathbf v
=
\mathbf J/m \\[4pt]
\Delta\omega
=
(\mathbf r\times\mathbf J)/I
\end{cases}
}
\]

---

# 17. 왜 한 step에서 collision solver를 여러 번 돌렸는가?

코드에:

```python
for _ in range(3):

    contact = circle_vs_T(...)

    if contact is None:
        break

    resolve_impulse(...)
```

가 있습니다.

실제 collision은 한 번 impulse를 줬다고 완벽하게 해결되지 않는 경우가 많습니다.

특히

- friction
- rotation
- penetration
- 여러 contact
- corner contact

때문에 반복해서 조금씩 correction합니다.

Pymunk, Box2D 같은 physics engine도 훨씬 정교한 **iterative constraint solver**를 사용합니다.

---

# 18. Position correction은 왜 필요한가?

이론적인 collision 순간에는

\[
penetration=0
\]

이어야 합니다.

그런데 discrete simulation에서는

\[
t
\rightarrow
t+\Delta t
\]

로 점프하기 때문에 다음처럼 될 수 있습니다.

```text
previous

○   | T


next

   ○|T
     ↑
   already overlapping
```

그래서 velocity impulse만 적용하면 이미 생긴 겹침이 남습니다.

따라서

```python
body.pos += (
    correction
    * normal
)
```

처럼 직접 조금 밀어냅니다.

이것을 **penetration correction**이라고 보면 됩니다.

---

# 19. PushT 1 step을 수식으로 다시 쓰면

Action:

\[
\boxed{
a_t=(x^*,y^*)
}
\]

Pusher controller:

\[
\boxed{
a_P
=
K_p(x^*-x_P)
-
K_dv_P
}
\]

Pusher update:

\[
v_P
\leftarrow
v_P+a_P\Delta t
\]

\[
x_P
\leftarrow
x_P+v_P\Delta t
\]

Collision detection:

\[
\boxed{
(circle,T)
\rightarrow
(q,n,\delta)
}
\]

Contact velocity:

\[
\boxed{
v_C
=
v_G+\omega\times r
}
\]

Relative velocity:

\[
\boxed{
v_{rel}=v_C-v_P
}
\]

Normal velocity:

\[
\boxed{
v_n=v_{rel}\cdot n
}
\]

Impulse:

\[
\boxed{
j_n
=
-\frac{(1+e)v_n}{
1/m+(r\times n)^2/I
}
}
\]

\[
\boxed{
J_n=j_nn
}
\]

Velocity update:

\[
\boxed{
v_G
\leftarrow
v_G+\frac{J_n}{m}
}
\]

\[
\boxed{
\omega
\leftarrow
\omega+
\frac{r\times J_n}{I}
}
\]

마지막으로:

\[
x_G
\leftarrow
x_G+v_G\Delta t
\]

\[
\theta
\leftarrow
\theta+\omega\Delta t
\]

입니다.

---

## 가장 중요한 4개의 식

PushT simulator를 직접 구현한다면 아래 4개만 먼저 완전히 이해하는 것이 좋습니다.

\[
\boxed{
v_C=v_G+\omega\times r
}
\]

\[
\boxed{
j
=
-\frac{(1+e)v_n}{
1/m+(r\times n)^2/I
}
}
\]

\[
\boxed{
v_G^+
=
v_G^-+\frac{J}{m}
}
\]

\[
\boxed{
\omega^+
=
\omega^-+
\frac{r\times J}{I}
}
\]

이 네 식이 **collision detection에서 구한 `(contact point, normal)`을 실제 T-block의 translation과 rotation으로 연결하는 다리**입니다.

그리고 위 예제는 의도적으로 단순화했습니다. 실제 Pymunk/Box2D 수준으로 가면 `multiple contact manifold`, `warm starting`, `static/dynamic friction`, `constraint stabilization`, `continuous collision detection` 등이 추가됩니다. 하지만 **PushT의 물리를 이해하고 직접 작은 simulator를 만드는 단계에서는 지금 코드 정도가 가장 좋은 출발점**입니다.