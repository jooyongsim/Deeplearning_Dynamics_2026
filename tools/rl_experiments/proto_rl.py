"""PPO prototype on the PushT env of pusht_imitation.py (definitions only, via ast)."""
import ast, math, os, sys, time, io, contextlib
import multiprocessing as mp
import numpy as np

TUT = "/home/cosmos/claude/04_DeepL_Dyna_Lecture/Deeplearning_Dynamics_2026/tutorial"
sys.path.insert(0, TUT)


def load_defs(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    want = {"circle_vs_T_fast", "wrap_angle", "PushTEnv", "ScriptedExpert", "T_outline"}
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in want)
            or (isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant))
            or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Tuple) for t in n.targets) and isinstance(n.value, ast.Tuple))]
    os.environ["PUSHT_NO_GUI"] = "1"
    with contextlib.redirect_stdout(io.StringIO()):
        import pusht_keyboard_sim as ksim
    ns = {"__name__": "pusht_defs", "ksim": ksim, "__file__": path}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    ksim.circle_vs_T = ns["circle_vs_T_fast"]
    ns["circle_vs_T"] = ns["circle_vs_T_fast"]
    return ns


D = load_defs(os.path.join(TUT, "pusht_imitation.py"))
PushTEnv, wrap_angle, rotation_matrix = D["PushTEnv"], D["wrap_angle"], D["rotation_matrix"]

ACT_SCALE = 0.15
MAX_STEPS = int(os.environ.get("MAX_STEPS", 300))


def cost(sim):
    d = np.linalg.norm(sim.T.pos - sim.goal_pos)
    a = abs(wrap_angle(sim.T.angle - sim.goal_angle))
    return d + 0.3 * a


def rl_obs(env):
    s = env.state(); sim = env.sim
    c, sn = math.cos(s[7]), math.sin(s[7]); Rt = np.array([[c, sn], [-sn, c]])
    p = Rt @ (s[0:2] - s[5:7]); T = Rt @ (s[2:4] - s[5:7]); d = s[4] - s[7]
    vT = Rt @ sim.T.vel; vP = Rt @ sim.pusher.vel
    rel = T - p
    return np.array([p[0], p[1], T[0], T[1], math.cos(d), math.sin(d), vT[0], vT[1], sim.T.omega, vP[0], vP[1],
                     rel[0], rel[1]], dtype=np.float32)


OBS_DIM = 13


def apply_action(env, a):
    s = env.state(); c, sn = math.cos(s[7]), math.sin(s[7])
    a = np.clip(a, -1, 1) * ACT_SCALE
    return s[0:2] + np.array([c * a[0] - sn * a[1], sn * a[0] + c * a[1]])


def worker(remote, seed0, k):
    rng = np.random.default_rng(seed0)
    envs = [PushTEnv(max_steps=MAX_STEPS) for _ in range(k)]
    costs = [0.0] * k
    def reset(i):
        envs[i].reset(int(rng.integers(300_000, 10**9)))
        costs[i] = cost(envs[i].sim)
        return rl_obs(envs[i])
    for i in range(k):
        reset(i)
    while True:
        cmd, data = remote.recv()
        if cmd != "step":
            break
        out = []
        for i, env in enumerate(envs):
            env.step(apply_action(env, data[i]))
            c1 = cost(env.sim)
            dist_pt = np.linalg.norm(env.sim.pusher.pos - env.sim.T.pos)
            r = 10.0 * (costs[i] - c1) - 0.02 * max(dist_pt - 1.0, 0.0) - 0.01
            success = env.is_success()
            trunc = (not success) and env.t >= MAX_STEPS
            if success:
                r += 10.0
            last = rl_obs(env)
            costs[i] = c1
            o2 = reset(i) if (success or trunc) else last
            out.append((o2, r, success, trunc, last, success))
        remote.send(out)


class VecEnv:
    def __init__(self, n, k=4):
        ctx = mp.get_context("fork")
        self.remotes, self.procs, self.k = [], [], k
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, 1234 + i, k), daemon=True); p.start()
            self.remotes.append(a); self.procs.append(p)
        self.n = n * k

    def step(self, acts):
        for j, r in enumerate(self.remotes): r.send(("step", acts[j * self.k:(j + 1) * self.k]))
        out = [x for r in self.remotes for x in r.recv()]
        o, rew, term, trunc, last, succ = zip(*out)
        return np.array(o), np.array(rew, np.float32), np.array(term), np.array(trunc), np.array(last), np.array(succ)


import torch, torch.nn as nn
torch.set_num_threads(4)


class AC(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 2))
        self.v = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.register_buffer("mu", torch.zeros(d)); self.register_buffer("var", torch.ones(d)); self.register_buffer("cnt", torch.tensor(1e-4))

    def norm(self, x): return ((x - self.mu) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)

    def update_norm(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mu; tot = self.cnt + bc
        self.mu += d * bc / tot
        self.var = (self.var * self.cnt + bv * bc + d ** 2 * self.cnt * bc / tot) / tot
        self.cnt = tot

    def dist(self, x):
        x = self.norm(x)
        return torch.distributions.Normal(self.pi(x), self.log_std.exp()), self.v(x).squeeze(-1)


def train(total_steps, n_envs=16, n_steps=128, out="ckpt", tag=""):
    os.makedirs(out, exist_ok=True)
    venv = VecEnv(n_envs, 4); n_envs = venv.n
    ac = AC(OBS_DIM); opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
    for r in venv.remotes: pass
    # initial obs: step with zero action once
    o, *_ = venv.step(np.zeros((n_envs, 2)))
    gamma, lam = 0.99, 0.95
    steps, it, t0 = 0, 0, time.time()
    ep_succ, ep_ret, cur_ret = [], [], np.zeros(n_envs)
    while steps < total_steps:
        O, A, LP, R, V, DN = [], [], [], [], [], []
        ot = torch.tensor(o)
        torch.set_num_threads(1)
        for t in range(n_steps):
            with torch.no_grad():
                d, v = ac.dist(ot); a = d.sample(); lp = d.log_prob(a).sum(-1)
            o2, r, term, trunc, last, succ = venv.step(a.numpy())
            r = r.copy()
            if trunc.any():  # bootstrap timeouts
                with torch.no_grad():
                    _, vl = ac.dist(torch.tensor(last[trunc]))
                r[trunc] += gamma * vl.numpy()
            O.append(ot); A.append(a); LP.append(lp); R.append(torch.tensor(r)); V.append(v); DN.append(torch.tensor(term | trunc, dtype=torch.float32))
            cur_ret += r
            for i in np.where(term | trunc)[0]:
                ep_succ.append(bool(succ[i])); ep_ret.append(cur_ret[i]); cur_ret[i] = 0
            ot = torch.tensor(o2); o = o2
        with torch.no_grad():
            _, v_last = ac.dist(ot)
        adv = torch.zeros(n_steps, n_envs); g = torch.zeros(n_envs)
        for t in reversed(range(n_steps)):
            nv = v_last if t == n_steps - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - DN[t]) - V[t]
            g = delta + gamma * lam * (1 - DN[t]) * g; adv[t] = g
        Ob = torch.cat(O); Ab = torch.cat(A); LPb = torch.cat(LP); ADV = adv.reshape(-1); RET = ADV + torch.cat(V)
        torch.set_num_threads(4)
        ac.update_norm(Ob)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        N = len(Ob)
        for ep in range(10):
            perm = torch.randperm(N)
            for i in range(0, N, 512):
                idx = perm[i:i + 512]
                d, v = ac.dist(Ob[idx]); lp = d.log_prob(Ab[idx]).sum(-1); ratio = (lp - LPb[idx]).exp()
                pl = -torch.min(ratio * ADV[idx], ratio.clamp(0.8, 1.2) * ADV[idx]).mean()
                vl = ((v - RET[idx]) ** 2).mean()
                loss = pl + 0.5 * vl - 0.0 * d.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
        steps += N; it += 1
        if it % 5 == 0:
            rs = ep_succ[-200:]
            print(f"{tag} steps {steps/1e6:5.2f}M  eps {len(ep_succ):6d}  succ(last200) {np.mean(rs) if rs else 0:.2f}  "
                  f"ret {np.mean(ep_ret[-200:]) if ep_ret else 0:7.2f}  std {ac.log_std.exp().detach().numpy().round(2)}  "
                  f"{(time.time()-t0)/60:.1f} min  {steps/(time.time()-t0):.0f} sps", flush=True)
        if it % 50 == 0:
            torch.save(ac.state_dict(), f"{out}/ac_{steps//1000}k.pt")
    torch.save(ac.state_dict(), f"{out}/ac_final.pt")


if __name__ == "__main__":
    train(float(sys.argv[1]), n_envs=int(sys.argv[2]) if len(sys.argv) > 2 else 16, out=sys.argv[3] if len(sys.argv) > 3 else "ckpt")
