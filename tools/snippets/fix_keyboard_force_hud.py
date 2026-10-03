# -*- coding: utf-8 -*-
"""Keyboard tutorial: show contact force (sum of impulses / frame time) and move the HUD beside the arena."""
import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "tutorial", "pusht_keyboard_sim.py")
s = open(p, encoding="utf-8").read()
R = [
# --- markdown: explain force display
("""# 마지막 contact의 위치·normal·impulse를 저장해 두었다가 화면에 화살표로 그립니다.""",
"""# 마지막 contact의 위치·normal을 저장하고, 한 frame 동안의 impulse 합을 시간으로 나눠 **평균 contact force** 로 표시합니다.
#
# $$
# \mathbf F_{\text{contact}}\approx\frac{1}{\Delta t_{\text{frame}}}\sum_{\text{substeps}}\mathbf J
# $$
#
# impulse는 힘을 시간에 대해 적분한 것($\mathbf J=\int\mathbf F\,dt$)이므로, 짧은 구간의 impulse 합을 그 시간으로 나누면 평균 힘이 됩니다.
# substep 하나의 $j_n$ 은 0.005 s 동안의 impulse라서 아주 작지만, 힘으로 바꾸면 크기를 비교하기 쉽습니다."""),
# --- sim state
("""        self.last_contact = None          # (point, normal, j_n, j_t) of the latest pusher contact
        self.last_impulse = (0.0, 0.0)""",
"""        self.last_contact = None          # (point, normal) of the latest pusher contact
        self.contact_force = (0.0, 0.0)   # (F_n, F_t) = impulses summed over the last frame / frame time
        self._j_sum = np.zeros(2)"""),
("""                if jn > 0:
                    self.last_contact = (c[0], c[1], jn, jt)
                    self.last_impulse = (jn, jt)""",
"""                self.last_contact = (c[0], c[1])
                self._j_sum += (jn, jt)"""),
("""    def advance(self, frame_dt):
        n = max(1, int(round(frame_dt / self.physics_dt)))
        for _ in range(n):
            self.substep(self.physics_dt)""",
"""    def advance(self, frame_dt):
        n = max(1, int(round(frame_dt / self.physics_dt)))
        self._j_sum[:] = 0.0
        for _ in range(n):
            self.substep(self.physics_dt)
        F = self._j_sum / (n * self.physics_dt)      # impulse per time = average force over the frame
        self.contact_force = (float(F[0]), float(F[1]))"""),
# --- headless check prints
("""print(f"last impulse (j_n, j_t): {tuple(round(v, 4) for v in sim.last_impulse)}")""",
"""print(f"contact force (F_n, F_t) in the last frame: {tuple(round(v, 3) for v in sim.contact_force)} N")"""),
# --- demo layout: HUD in a panel to the right
("""        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        ax = self.ax""",
"""        self.fig, self.ax = plt.subplots(figsize=(11.5, 7))
        self.fig.subplots_adjust(left=0.02, right=0.76, top=0.93, bottom=0.07)
        ax = self.ax"""),
("""        self.hud = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                           family="monospace", fontsize=9,
                           bbox=dict(fc="white", ec="#B4C3D6", alpha=0.9))""",
"""        self.hud = self.fig.text(0.775, 0.90, "", va="top", ha="left", family="monospace", fontsize=10,
                                 bbox=dict(fc="white", ec="#B4C3D6", boxstyle="square,pad=0.6"))"""),
("""        ax.text(0.5, -0.02, "arrows/WASD: move   mouse drag: target   space: pause   "
                            "R: reset   F: friction   E: restitution   Q: quit",
                transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#404040")""",
"""        self.fig.text(0.775, 0.36, "arrows / WASD : move target\nmouse drag    : target\nspace         : pause\n"
                                   "R             : reset\nF             : friction on/off\nE             : restitution\n"
                                   "Q / Esc       : quit",
                      va="top", ha="left", family="monospace", fontsize=9, color="#404040")"""),
# --- arrow + HUD text
("""        if sim.last_contact is not None:
            q, n, jn, jt = sim.last_contact
            L = min(0.25 + 12.0 * jn, 1.4)""",
"""        Fn, Ft = sim.contact_force
        if sim.last_contact is not None:
            q, n = sim.last_contact
            L = min(0.2 + 0.25 * Fn, 1.5)"""),
("""        jn, jt = sim.last_impulse
        self.hud.set_text(""", """        self.hud.set_text("""),
("""# - **빨간 화살표** = contact normal $\mathbf n$, 길이 $\propto j_n$ (T가 받는 impulse)""",
"""# - **빨간 화살표** = contact normal $\mathbf n$, 길이 $\propto F_n$ (T가 받는 평균 contact force)"""),
]
for a, b in R:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
