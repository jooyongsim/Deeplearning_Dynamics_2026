import os, sys, copy, time
os.environ["PUSHT_NO_GUI"] = "1"
import matplotlib; matplotlib.use("Agg")
kind, steps, ema, lr, K = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
code = open("pusht_imitation.py", encoding="utf-8").read()
code = code[:code.index("TRAIN_STEPS = 30_000")]
code = code.replace("train(toy_bc", "pass  # train(toy_bc").replace("train(toy_dp", "pass  # train(toy_dp")
full = open("pusht_imitation.py", encoding="utf-8").read()
extra = full[full.index("class LearnedAgent:"):full.index("results = {")]
ns = {"__name__": "x", "__file__": os.path.abspath("pusht_imitation.py")}
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, "t", "exec"), ns)
    exec(compile(extra, "t2", "exec"), ns)
import torch; torch.set_num_threads(4)
X, Y, Xv, Yv = ns["X_train"], ns["Y_train"], ns["X_val"], ns["Y_val"]
cls = ns["BCPolicy"] if kind == "mlp" else ns["DiffusionPolicy"]
m = cls(X.shape[1], Y.shape[1], **({} if kind == "mlp" else {"K": K})).set_stats(X, Y)
# EMA training loop (copy of tutorial train with EMA)
torch.manual_seed(0)
Xt, Yt = torch.tensor(X), torch.tensor(Y)
opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-6)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
avg = copy.deepcopy(m)
t0 = time.time()
for it in range(1, steps + 1):
    idx = torch.randint(0, len(Xt), (256,))
    loss = m.loss(Xt[idx], Yt[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if ema > 0:
        with torch.no_grad():
            for pa, pm in zip(avg.parameters(), m.parameters()):
                pa.mul_(ema).add_(pm, alpha=1 - ema)
final = avg if ema > 0 else m
final.eval()
with torch.no_grad():
    torch.manual_seed(1); v = final.loss(torch.tensor(Xv), torch.tensor(Yv)).item()
ns["DEVICE"] = "cpu"
torch.save(final.state_dict(), f"study_{kind}_{steps}_{ema}_{lr}_{K}.pt")
r = ns["evaluate"](ns["LearnedAgent"](final), n=50, label="x")
print(f"RESULT {kind} steps={steps} ema={ema} lr={lr} K={K}: success {r['success'].mean()*100:.0f}%  med err {sorted(r['pos_err'])[25]:.2f} m {sorted(r['ang_err'])[25]:.0f} deg  val {v:.4f}  train {time.time()-t0:.0f}s", flush=True)
