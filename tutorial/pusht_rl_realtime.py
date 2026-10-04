"""Real-time PushT demo: several policies side by side on the same random task.

    python pusht_rl_realtime.py                    # all panels below
    python pusht_rl_realtime.py script             # only the scripted expert
    python pusht_rl_realtime.py script bc_final    # any subset, in that order

Panels
    script     ScriptedExpert (pusht_imitation.py section 5)
    scratch    PPO from scratch            checkpoints/rl/scratch/ppo_scratch_<largest>.pt
    bc         MLP-BC imitation policy     checkpoints/rl/bc_ppo/bcppo_0k.pt     (= RL step 0)
    bc_mid     BC -> PPO, halfway          checkpoints/rl/bc_ppo/bcppo_<middle>.pt
    bc_final   BC -> PPO, last checkpoint  checkpoints/rl/bc_ppo/bcppo_<largest>.pt

Keys: space pause | N next task | + / - simulation speed | Q / Esc quit
All panels start from the same seed; when every panel is done (success or 30 s), a new random task starts.
"""
import math
import os
import sys
import time

import matplotlib

if "PUSHT_NO_GUI" not in os.environ:
    try:
        get_ipython().run_line_magic("matplotlib", "tk")      # noqa: F821  Jupyter / VS Code Interactive
    except NameError:
        matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Polygon

import pusht_rl as R

ARENA = R.IM["ARENA"]                # pusht_keyboard_sim was already imported by pusht_rl (without its window)

ALL_PANELS = ["script", "scratch", "bc", "bc_mid", "bc_final"]
TITLES = {"script": "Script (expert)", "scratch": "RL from scratch", "bc": "Imitation (BC), RL step 0",
          "bc_mid": "BC -> RL, halfway", "bc_final": "BC -> RL, final"}
FRAME_DT = 0.05                     # draw every 0.05 s of simulated time; control at 10 Hz = every 2 frames


def make_agent_factory(name):
    """Returns (factory env -> agent, subtitle) or (None, reason) if the weights are missing."""
    if name == "script":
        return (lambda env: R.ScriptedExpert(env)), "hand-written, impulse-based"
    if name == "scratch":
        ck = R.checkpoints(os.path.join(R.RL_DIR, "scratch"), "ppo_scratch")
        pick = ck[-1] if ck else None
    else:
        ck = R.checkpoints(os.path.join(R.RL_DIR, "bc_ppo"), "bcppo")
        pick = None if not ck else {"bc": ck[0], "bc_mid": ck[len(ck) // 2], "bc_final": ck[-1]}[name]
    if pick is None:
        return None, "no checkpoint yet - run pusht_rl.py"
    model = R.load_policy(pick[1])
    return (lambda env, m=model: R.RLAgent(env, m)), os.path.basename(pick[1])


class Panel:
    def __init__(self, ax, name):
        self.ax, self.name = ax, name
        self.factory, self.subtitle = make_agent_factory(name)
        self.env = R.PushTEnv(max_steps=R.MAX_STEPS)
        self.agent = self.factory(self.env) if self.factory else None
        self.wins = self.runs = 0
        V = self.env.sim.T.vertices
        x0, x1, y0, y1 = ARENA
        ax.add_patch(Polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], closed=True, fill=False, ec="#404040", lw=2))
        self.goal = ax.add_patch(Polygon(V, closed=True, fill=False, ec="#3D8C54", lw=2, ls="--"))
        self.T = ax.add_patch(Polygon(V, closed=True, fc="#D9EEFC", ec="#2774B8", lw=2))
        self.P = ax.add_patch(Circle((0, 0), self.env.sim.pusher.radius, fc="#FFEBC9", ec="#E97820", lw=2))
        self.trail, = ax.plot([], [], "-", color="#2774B8", lw=0.8, alpha=0.5)
        self.target, = ax.plot([], [], "x", color="#E97820", ms=7, mew=2)
        self.status = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)
        ax.set_xlim(x0 - 0.1, x1 + 0.1); ax.set_ylim(y0 - 0.1, y1 + 0.1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if self.agent is None:
            ax.text(0.5, 0.5, self.subtitle, transform=ax.transAxes, ha="center", va="center", color="#FF0000")

    def start(self, seed):
        self.done, self.success, self.sub = self.agent is None, False, 0
        self.env.reset(seed)
        if self.agent is not None:
            self.agent.reset()
        s = self.env.sim
        self.goal.set_xy(R.IM["T_outline"](s.goal_pos, s.goal_angle, s.T.vertices))
        self.path = [s.T.pos.copy()]

    def frame(self):
        """Advance FRAME_DT of simulated time (half a control step)."""
        if self.done:
            return
        env, sim = self.env, self.env.sim
        if self.sub == 0:                                   # new action every 0.1 s, held for 2 frames
            x0, x1, y0, y1 = ARENA
            sim.target = np.clip(np.asarray(self.agent.act(env.state()), float), [x0, y0], [x1, y1])
        sim.advance(FRAME_DT)
        self.sub ^= 1
        if self.sub == 0:
            env.t += 1
            self.path.append(sim.T.pos.copy())
            if env.is_success() or env.t >= R.MAX_STEPS:
                self.done, self.success = True, env.is_success()
                self.runs += 1
                self.wins += self.success

    def draw(self):
        s = self.env.sim
        V = s.T.vertices
        self.T.set_xy(R.IM["T_outline"](s.T.pos, s.T.angle, V))
        self.P.center = tuple(s.pusher.pos)
        self.target.set_data([s.target[0]], [s.target[1]])
        p = np.array(self.path)
        self.trail.set_data(p[:, 0], p[:, 1])
        d, a = s.goal_error()
        state = ("SUCCESS" if self.success else "time out") if self.done and self.agent else f"t {self.env.t / 10:4.1f} s"
        self.status.set_text(f"{state}\nerr {d:.2f} m {a:4.0f} deg")
        self.status.set_color("#3D8C54" if self.success else "#FF0000" if self.done else "#203864")
        self.T.set_edgecolor("#3D8C54" if self.success else "#2774B8")
        rate = f"   success {self.wins}/{self.runs}" if self.runs else ""
        self.ax.set_title(f"{TITLES[self.name]}{rate}\n{self.subtitle}", fontsize=10)


class RealtimeDemo:
    def __init__(self, names):
        torch.set_num_threads(1)
        for k in list(plt.rcParams):                      # free single-letter keys (s = save, ...)
            if k.startswith("keymap."):
                plt.rcParams[k] = []
        n = len(names)
        cols = min(n, 3)
        rows = math.ceil(n / cols)
        scale = 1.8 if n == 1 else 1.0                    # one panel -> a bigger window
        self.fig, axes = plt.subplots(rows, cols, figsize=(scale * 5.2 * cols + 0.4, scale * 4.3 * rows + 0.8),
                                      squeeze=False)
        for ax in axes.flat[n:]:
            ax.axis("off")
        self.panels = [Panel(ax, name) for ax, name in zip(axes.flat, names)]
        self.header = self.fig.suptitle("", fontsize=11)
        self.rng = np.random.default_rng()
        self.speed, self.paused, self.next_at = 1, False, None
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.new_task()
        self.update()                                     # titles exist before the layout is computed
        self.fig.tight_layout(rect=(0, 0, 1, 0.93))

    def new_task(self):
        self.seed = int(self.rng.integers(0, 10**6))
        for p in self.panels:
            p.start(self.seed)
        self.next_at = None

    def on_key(self, event):
        key = (event.key or "").lower()
        if key == " ":
            self.paused = not self.paused
        elif key == "n":
            self.new_task()
        elif key in ("+", "="):
            self.speed = min(self.speed * 2, 16)
        elif key == "-":
            self.speed = max(self.speed // 2, 1)
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def update(self, _frame=None):
        if not self.paused:
            if self.next_at is not None and time.perf_counter() > self.next_at:
                self.new_task()
            for _ in range(self.speed):
                for p in self.panels:
                    p.frame()
            if self.next_at is None and all(p.done for p in self.panels):
                self.next_at = time.perf_counter() + 1.5          # show the end state, then a new task
        for p in self.panels:
            p.draw()
        self.header.set_text(f"PushT  seed {self.seed}   speed x{self.speed}{'   PAUSED' if self.paused else ''}\n"
                             "[space] pause   [N] next task   [+/-] speed   [Q] quit")
        return []

    def run(self):
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self.update, interval=int(FRAME_DT * 1000), blit=False,
                                  cache_frame_data=False)
        plt.show()
        return self.anim


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if a in ALL_PANELS] or ALL_PANELS
    unknown = [a for a in sys.argv[1:] if a not in ALL_PANELS]
    if unknown:
        print("unknown panel(s):", unknown, " choose from:", ALL_PANELS)
    demo = RealtimeDemo(names)
    if not os.environ.get("PUSHT_NO_GUI"):
        demo.run()
