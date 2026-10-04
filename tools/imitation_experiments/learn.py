import sys, time, math
import torch as _t; _t.set_num_threads(4)
import numpy as np, torch, torch.nn as nn
from proto import *

def collect(n, seed0=1000):
    env = PushTEnv(max_steps=400); eps = []; seed = seed0
    while len(eps) < n:
        env.reset(seed); ex = Expert(env.sim); seed += 1
        S, A = [], []
        while True:
            s = env.sim
            S.append(np.r_[s.pusher.pos, s.T.pos, s.T.angle, s.goal_pos, s.goal_angle])
            a = ex.act(); A.append(a)
            _, ok, done = env.step(a)
            if done: break
        if ok: eps.append((np.array(S), np.array(A)))
    return eps

def feat(S):  # S (..., 8) raw -> features
    p, T, th, g, gth = S[..., 0:2], S[..., 2:4], S[..., 4:5], S[..., 5:7], S[..., 7:8]
    return np.concatenate([p, T, np.cos(th), np.sin(th), g, np.cos(gth), np.sin(gth)], -1)

OH, AH = 2, 8
def windows(eps):
    O, Y = [], []
    for S, A in eps:
        F = feat(S); n = len(S)
        for t in range(n):
            idx_o = [max(0, t - k) for k in range(OH - 1, -1, -1)]
            idx_a = [min(n - 1, t + k) for k in range(AH)]
            O.append(F[idx_o].ravel()); Y.append(A[idx_a].ravel())
    return np.array(O, np.float32), np.array(Y, np.float32)

class MLP(nn.Module):
    def __init__(s, i, o, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(i, h), nn.Mish(), nn.Linear(h, h), nn.Mish(), nn.Linear(h, h), nn.Mish(), nn.Linear(h, o))
    def forward(s, x): return s.net(x)

class Denoiser(nn.Module):
    def __init__(s, od, ad, h=256):
        super().__init__(); s.ad = ad
        s.net = nn.Sequential(nn.Linear(od + ad + 32, h), nn.Mish(), nn.Linear(h, h), nn.Mish(), nn.Linear(h, h), nn.Mish(), nn.Linear(h, ad))
    def forward(s, x, k, o):
        f = torch.exp(-math.log(1000) * torch.arange(16) / 16)
        e = k[:, None].float() * f[None]
        return s.net(torch.cat([x, o, torch.sin(e), torch.cos(e)], -1))

K_STEPS = 50
betas = torch.linspace(1e-4, 0.02, K_STEPS) * (100 / K_STEPS)  # ~ same total noise
betas = torch.tensor(np.clip(1 - np.cos((np.arange(K_STEPS) + 1) / K_STEPS * math.pi / 2) ** 2 /
                             np.maximum(np.cos(np.arange(K_STEPS) / K_STEPS * math.pi / 2) ** 2, 1e-8), 1e-4, 0.999), dtype=torch.float32)
alphas = 1 - betas; abar = torch.cumprod(alphas, 0)

def train(kind, O, Y, epochs=200, bs=256):
    torch.manual_seed(0)
    om, os_ = O.mean(0), O.std(0) + 1e-6; ym, ys = Y.mean(0), Y.std(0) + 1e-6
    On = torch.tensor((O - om) / os_); Yn = torch.tensor((Y - ym) / ys)
    m = MLP(On.shape[1], Yn.shape[1]) if kind == "mlp" else Denoiser(On.shape[1], Yn.shape[1])
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-6)
    steps = epochs * (len(On) // bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(On)); tot = 0
        for i in range(0, len(On) - bs + 1, bs):
            o, y = On[perm[i:i + bs]], Yn[perm[i:i + bs]]
            if kind == "mlp":
                loss = ((m(o) - y) ** 2).mean()
            else:
                k = torch.randint(0, K_STEPS, (len(o),)); eps = torch.randn_like(y)
                ab = abar[k][:, None]
                loss = ((m(ab.sqrt() * y + (1 - ab).sqrt() * eps, k, o) - eps) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if ep % 50 == 0 or ep == epochs - 1: print(kind, ep, tot / (len(On) // bs), f"{time.time()-t0:.0f}s", flush=True)
    m.eval()
    def policy(o):
        on = torch.tensor(((o - om) / os_)[None], dtype=torch.float32)
        with torch.no_grad():
            if kind == "mlp": yn = m(on)
            else:
                x = torch.randn(1, Yn.shape[1])
                for k in reversed(range(K_STEPS)):
                    kk = torch.full((1,), k)
                    eps = m(x, kk, on)
                    x0 = (x - (1 - abar[k]).sqrt() * eps) / abar[k].sqrt()
                    x0 = x0.clamp(-4, 4)
                    if k > 0:
                        # DDPM posterior mean
                        ab_prev = abar[k - 1]
                        mean = (ab_prev.sqrt() * betas[k] / (1 - abar[k])) * x0 + (alphas[k].sqrt() * (1 - ab_prev) / (1 - abar[k])) * x
                        var = betas[k] * (1 - ab_prev) / (1 - abar[k])
                        x = mean + var.sqrt() * torch.randn_like(x)
                    else: x = x0
                yn = x
        return (yn.numpy()[0] * ys + ym).reshape(AH, 2)
    return policy

def evaluate(policy, n=50, seed0=0, exec_h=4, max_steps=300):
    env = PushTEnv(max_steps=max_steps); succ = []
    for seed in range(seed0, seed0 + n):
        env.reset(seed); s = env.sim
        hist = [np.r_[s.pusher.pos, s.T.pos, s.T.angle, s.goal_pos, s.goal_angle]] * OH
        done = False
        while not done:
            o = feat(np.array(hist[-OH:])).ravel()
            plan = policy(o)
            for a in plan[:exec_h]:
                _, ok, done = env.step(a)
                hist.append(np.r_[s.pusher.pos, s.T.pos, s.T.angle, s.goal_pos, s.goal_angle])
                if done: break
        succ.append(ok)
    return float(np.mean(succ))

if __name__ == "__main__":
    import pickle, os
    N = int(sys.argv[1]); kinds = sys.argv[2].split(",")
    if os.path.exists("eps.pkl"): eps = pickle.load(open("eps.pkl", "rb"))
    else:
        t = time.time(); eps = collect(300); print("collect", time.time() - t); pickle.dump(eps, open("eps.pkl", "wb"))
    O, Y = windows(eps[:N]); print("samples", len(O))
    for kind in kinds:
        pol = train(kind, O, Y, epochs=int(sys.argv[3]) if len(sys.argv) > 3 else 200)
        t = time.time(); print(kind, "N", N, "success", evaluate(pol), f"eval {time.time()-t:.0f}s", flush=True)
