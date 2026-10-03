"""Drive tutorial/pusht_imitation.py's DemoRecorder without a window (Agg backend).

Runs the tutorial only up to the end of section 4 (no torch needed), feeds synthetic
key events, forces a goal-reached state and checks what lands in the .npz file.
Snapshot -> review/imitation_recorder.png

    python test_imitation_recorder.py
"""
import os
import sys
import tempfile
import types

import matplotlib

matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TUT = os.path.join(HERE, "..", "tutorial", "pusht_imitation.py")
OUT = os.path.join(HERE, "review")
os.makedirs(OUT, exist_ok=True)

os.environ["PUSHT_NO_GUI"] = "1"
code = open(TUT, encoding="utf-8").read()
code = code[:code.index("COLLECT_HUMAN = False")]          # sections 0-4 only
ns = {"__name__": "pusht_imitation", "__file__": os.path.abspath(TUT)}
_stdout, sys.stdout = sys.stdout, open(os.devnull, "w", encoding="utf-8")
exec(compile(code, TUT, "exec"), ns)
sys.stdout = _stdout

path = os.path.join(tempfile.mkdtemp(), "human_demos.npz")
rec = ns["DemoRecorder"](path=path)
ev = lambda **kw: types.SimpleNamespace(**kw)


def frames(n):
    for _ in range(n):
        rec.update()


# 1. idle frames are not recorded
frames(10)
print("idle 10 frames      : recorded", len(rec.S), "(should be 0)")

# 2. hold right for 40 frames (2 s) -> 20 control steps at 10 Hz, target moves 1.5 m/s
t0 = rec.sim.target.copy()
rec.on_press(ev(key="right"))
frames(40)
rec.on_release(ev(key="right"))
print("hold right 40 frames: recorded", len(rec.S), "(should be 20)  target moved",
      (rec.sim.target - t0).round(2), "(should be ~ +3.0 in x, or clipped at the wall)")
A = np.array(rec.A)
print("action step sizes   :", np.unique(np.linalg.norm(np.diff(A, axis=0), axis=1).round(3)))

# 3. f / e must not change the physics while recording
rec.on_press(ev(key="f")); rec.on_press(ev(key="e"))
print("f / e ignored       : mu =", rec.sim.friction, " e =", rec.sim.restitution)

# 4. reach the goal -> auto-save
rec.sim.T.pos[:] = rec.sim.goal_pos
rec.sim.T.angle = rec.sim.goal_angle
rec.sim.T.vel[:] = 0.0
rec.sim.pusher.pos[:] = rec.sim.goal_pos + np.array([3.0, 0.0])
rec.sim.target = rec.sim.pusher.pos.copy()
frames(2)
eps, seeds = ns["load_demos"](path)
print("goal reached        : saved", len(eps), "episode(s), lengths", [len(S) for S, _ in eps],
      " seed", seeds, " title:", rec.title.get_text())
rec.fig.savefig(os.path.join(OUT, "imitation_recorder.png"), dpi=70)

# 5. env.reset(seed) reproduces the first recorded state's T and goal
env = ns["PushTEnv"]()
s0 = env.reset(seeds[0])
print("seed reproduces     :", bool(np.allclose(s0[2:], eps[0][0][0][2:])), "(T and goal at the first step)")

# 6. N -> new random episode, Backspace -> delete the saved one
old_seed = rec.seed
rec.restart_at = None
rec.on_press(ev(key="n"))
print("N                   : new seed", rec.seed != old_seed, " recorded", len(rec.S))
rec.on_press(ev(key="backspace"))
print("Backspace           : episodes", len(rec.episodes), " file exists", os.path.exists(path))
