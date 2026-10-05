import os, sys, time, types
os.environ["PUSHT_NO_GUI"] = "1"
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, "/home/cosmos/claude/04_DeepL_Dyna_Lecture/Deeplearning_Dynamics_2026/tutorial")
import pusht_rl_realtime as RT
for names in (["script"], RT.ALL_PANELS):
    d = RT.RealtimeDemo(names)
    t = time.perf_counter()
    for i in range(120): d.update()
    dt = (time.perf_counter() - t) / 120
    print(names, "ms/update", round(dt * 1000, 1), [(p.name, p.subtitle, p.env.t, p.done) for p in d.panels])
    d.fig.savefig(f"rt_{len(names)}.png", dpi=60); d.on_key(types.SimpleNamespace(key="+")); d.on_key(types.SimpleNamespace(key="n"))
    print("speed", d.speed, "new seed", d.seed)
    d.fig.savefig(f"rt_{len(names)}.png", dpi=60)
