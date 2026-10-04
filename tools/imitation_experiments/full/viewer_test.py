import os, io, contextlib, types
os.environ["PUSHT_NO_GUI"] = "1"
import matplotlib; matplotlib.use("Agg")
full = open("pusht_imitation.py", encoding="utf-8").read()
code = full[:full.index("TRAIN_STEPS = 30_000")].replace("train(toy_bc", "pass  # ").replace("train(toy_dp", "pass  # ")
extra = full[full.index("class LearnedAgent:"):full.index("results = {")]
viewer = full[full.index("class PolicyViewer"):full.index("best = max(")]
ns = {"__name__": "x", "__file__": os.path.abspath("pusht_imitation.py")}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, "t", "exec"), ns); exec(compile(extra, "t2", "exec"), ns); exec(compile(viewer, "t3", "exec"), ns)
import torch
X, Y = ns["X_train"], ns["Y_train"]
m = ns["DiffusionPolicy"](X.shape[1], Y.shape[1]).set_stats(X, Y)
m.load_state_dict(torch.load("s2_diff_30000_0.003_0.999.pt")); m.eval()
v = ns["PolicyViewer"](m)
v.new_episode(0)
ev = lambda **kw: types.SimpleNamespace(**kw)
for i in range(60): v.update()
print("after 60 frames: t =", round(v.sim.time, 2), " plan dots", len(v.plan_dots.get_xdata()), " queue", len(v.agent.queue))
v.fig.savefig("viewer_1.png", dpi=70)
# human takeover
t0 = v.sim.target.copy(); v.on_press(ev(key="up"))
for i in range(10): v.update()
v.on_release(ev(key="up"))
print("takeover up: target dy =", round(v.sim.target[1] - t0[1], 2), " agent reset ->", v.agent.queue == [])
import time; time.sleep(0.15)
for i in range(400):
    v.update()
    if v.restart_at is not None: break
print("success reached:", v.env.is_success(), " t =", round(v.sim.time, 1), " title:", v.title.get_text())
v.on_press(ev(key="n")); print("N -> new seed", v.seed)
v.fig.savefig("viewer_2.png", dpi=70)
