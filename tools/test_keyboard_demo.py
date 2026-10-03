"""Drive tutorial/pusht_keyboard_sim.py's KeyboardDemo without a window (Agg backend).

Feeds synthetic key / mouse events and advances frames by hand, then saves snapshots
to review/keyboard_*.png so the drawing can be checked.

    python test_keyboard_demo.py
"""
import os
import sys
import types

import matplotlib

matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
TUT = os.path.join(HERE, "..", "tutorial", "pusht_keyboard_sim.py")
OUT = os.path.join(HERE, "review")
os.makedirs(OUT, exist_ok=True)

os.environ["PUSHT_NO_GUI"] = "1"
ns = {"__name__": "pusht_keyboard_sim"}
code = open(TUT, encoding="utf-8").read()
_stdout, sys.stdout = sys.stdout, open(os.devnull, "w", encoding="utf-8")   # silence the tutorial's own prints
exec(compile(code, TUT, "exec"), ns)
sys.stdout = _stdout

demo = ns["KeyboardDemo"](ns["KeyboardPushT"]())
sim = demo.sim
ev = lambda **kw: types.SimpleNamespace(**kw)


def frames(n, dt=0.02):
    for _ in range(n):
        demo.update(frame_dt=dt)


# 1. hold "right" for 1.6 s (press once, no release) -> target and pusher move right, T gets pushed
x0 = sim.T.pos.copy()
demo.on_press(ev(key="right"))
frames(80)
demo.on_release(ev(key="right"))
print("after holding right : target", sim.target.round(2), " T moved", (sim.T.pos - x0).round(3),
      " contact", sim.last_contact is not None, " F_n", round(sim.contact_force[0], 2))
demo.fig.savefig(os.path.join(OUT, "keyboard_1_push.png"), dpi=80)

# 2. auto-repeat pattern: press/release pairs every 33 ms must still count as "held"
t0 = sim.target.copy()
for _ in range(30):
    demo.on_press(ev(key="up")); demo.on_release(ev(key="up")); frames(1)
print("auto-repeat 'up'    : target moved", (sim.target - t0).round(3), "(should be ~ +0.9 in y)")

# 3. released key stops after the 0.12 s grace window (real time is used for that check)
import time
time.sleep(0.15)
t1 = sim.target.copy(); frames(10)
print("after release       : target moved", (sim.target - t1).round(4), "(should be 0)")

# 4. toggles, pause, mouse drag, reset
demo.on_press(ev(key="f")); demo.on_press(ev(key="e"))
print("toggles             : mu =", sim.friction, " e =", sim.restitution)
demo.on_press(ev(key=" ")); p = sim.T.pos.copy(); frames(5)
print("paused              : T frozen", bool((sim.T.pos == p).all()), " title:", demo.title.get_text())
demo.on_press(ev(key=" "))
demo.on_button(ev(button=1, inaxes=demo.ax, xdata=1.0, ydata=-2.0))
demo.on_motion(ev(inaxes=demo.ax, xdata=1.5, ydata=-2.2))
demo.on_button_release(ev())
print("mouse drag          : target", sim.target)
frames(40)
demo.fig.savefig(os.path.join(OUT, "keyboard_2_after.png"), dpi=80)
demo.on_press(ev(key="r")); frames(1)
print("reset               : T at", sim.T.pos, " t =", round(sim.time, 3))

# 5. wall: drive the T into the right wall and check it stays inside the arena
sim.T.pos[:] = (3.0, 0.0); sim.T.vel[:] = (6.0, 0.0)
x_max = ns["ARENA"][1]
peak = -1e9
for _ in range(60):
    frames(1)
    world = sim.T.pos + sim.T.vertices @ ns["rotation_matrix"](sim.T.angle).T
    peak = max(peak, world[:, 0].max())
print("wall                : peak vertex x = %.3f over the run (wall at %.1f), final v_x = %+.2f"
      % (peak, x_max, sim.T.vel[0]))
