import os, sys, time, io, contextlib
os.environ["PUSHT_NO_GUI"] = "1"
import matplotlib; matplotlib.use("Agg")
kind, N, STEPS, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
full = open("pusht_imitation.py", encoding="utf-8").read()
code = full[:full.index("TRAIN_STEPS = 30_000")]
code = code.replace("train(toy_bc", "pass  # train(toy_bc").replace("train(toy_dp", "pass  # train(toy_dp")
extra = full[full.index("class LearnedAgent:"):full.index("results = {")]
ns = {"__name__": "x", "__file__": os.path.abspath("pusht_imitation.py")}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, "t", "exec"), ns); exec(compile(extra, "t2", "exec"), ns)
import torch; torch.set_num_threads(4)
eps, _ = ns["load_demos"]("demos/expert_400.npz")
tr, va = ns["split_episodes"](eps)
X, Y = ns["make_samples"](tr[:N]); Xv, Yv = ns["make_samples"](va)
cls = ns["BCPolicy"] if kind == "mlp" else ns["DiffusionPolicy"]
m = cls(X.shape[1], Y.shape[1], width=W).set_stats(X, Y)
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    h = ns["train"](m, X, Y, Xv, Yv, steps=STEPS)
torch.save(m.state_dict(), f"scale2_{kind}_{N}_{STEPS}_{W}.pt")
r = ns["evaluate"](ns["LearnedAgent"](m), n=100, label="x")
print(f"RESULT {kind} N={N} steps={STEPS} width={W}: success {r['success'].mean()*100:.0f}%  med err {sorted(r['pos_err'])[50]:.2f} m {sorted(r['ang_err'])[50]:.0f} deg  val {h['val'][-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
