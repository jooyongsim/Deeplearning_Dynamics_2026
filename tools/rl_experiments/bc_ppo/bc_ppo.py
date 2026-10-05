"""PPO fine-tuning that starts from the MLP-BC imitation policy (first action of its chunk)."""
import math, os, sys, time
import multiprocessing as mp
import numpy as np
import torch, torch.nn as nn
from defs import load_defs

D = load_defs()
PushTEnv, wrap_angle, make_obs, action_from_model = D["PushTEnv"], D["wrap_angle"], D["make_obs"], D["action_from_model"]
BC_CKPT = "/home/cosmos/claude/04_DeepL_Dyna_Lecture/Deeplearning_Dynamics_2026/tutorial/checkpoints/MLP-BC_expert_180ep_28626st_w512_30000.pt"
MAX_STEPS = 300


def cost(sim):
    return np.linalg.norm(sim.T.pos - sim.goal_pos) + 0.3 * abs(wrap_angle(sim.T.angle - sim.goal_angle))


def extra_obs(env):
    """Velocities in the goal frame (for the critic only)."""
    s = env.state(); sim = env.sim
    c, sn = math.cos(s[7]), math.sin(s[7]); Rt = np.array([[c, sn], [-sn, c]])
    vT, vP = Rt @ sim.T.vel, Rt @ sim.pusher.vel
    return np.array([vT[0], vT[1], sim.T.omega, vP[0], vP[1], env.t / MAX_STEPS])


def worker(remote, seed0, k):
    rng = np.random.default_rng(seed0)
    envs = [PushTEnv(max_steps=MAX_STEPS) for _ in range(k)]
    costs, prev = [0.0] * k, [None] * k

    def reset(i):
        envs[i].reset(int(rng.integers(300_000, 10**9)))
        costs[i] = cost(envs[i].sim)
        prev[i] = make_obs(envs[i].state())

    def obs(i):
        cur = make_obs(envs[i].state())
        return np.concatenate([prev[i], cur, extra_obs(envs[i])]).astype(np.float32), cur

    for i in range(k):
        reset(i)
    while True:
        cmd, data = remote.recv()
        if cmd == "obs":
            remote.send([obs(i)[0] for i in range(k)])
            continue
        if cmd != "step":
            break
        out = []
        for i, env in enumerate(envs):
            prev[i] = make_obs(env.state())
            env.step(action_from_model(np.asarray(data[i], float), env.state()))
            c1 = cost(env.sim)
            dist_pt = np.linalg.norm(env.sim.pusher.pos - env.sim.T.pos)
            r = 10.0 * (costs[i] - c1) - 0.02 * max(dist_pt - 1.0, 0.0) - 0.01
            success = env.is_success()
            trunc = (not success) and env.t >= MAX_STEPS
            if success:
                r += 10.0
            last, _ = obs(i)
            costs[i] = c1
            if success or trunc:
                reset(i)
            out.append((obs(i)[0], r, success, trunc, last))
        remote.send(out)


class VecEnv:
    def __init__(self, n, k=4):
        ctx = mp.get_context("fork")
        self.remotes, self.k = [], k
        for i in range(n):
            a, b = ctx.Pipe()
            ctx.Process(target=worker, args=(b, 777 + i, k), daemon=True).start()
            self.remotes.append(a)
        self.n = n * k

    def obs(self):
        for r in self.remotes: r.send(("obs", None))
        return np.array([x for r in self.remotes for x in r.recv()])

    def step(self, acts):
        for j, r in enumerate(self.remotes): r.send(("step", acts[j * self.k:(j + 1) * self.k]))
        out = [x for r in self.remotes for x in r.recv()]
        o, rew, term, trunc, last = zip(*out)
        return np.array(o), np.array(rew, np.float32), np.array(term), np.array(trunc), np.array(last)


class BCActorCritic(nn.Module):
    """Actor = the BC network (first action of its chunk) + learnable std.  Critic = separate MLP."""

    def __init__(self, bc_state, width=512, log_std=math.log(0.03)):
        super().__init__()
        self.bc = D["BCPolicy"](12, 16, width=width)
        for k in ("x_mu", "x_sd", "y_mu", "y_sd"):
            self.bc.register_buffer(k, bc_state[k])
        self.bc.load_state_dict(bc_state)
        self.log_std = nn.Parameter(torch.full((2,), log_std))
        self.v = nn.Sequential(nn.Linear(18, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
        self.register_buffer("mu", torch.zeros(18)); self.register_buffer("var", torch.ones(18))
        self.register_buffer("cnt", torch.tensor(1e-4))

    def mean(self, o):                               # o: (N, 18); actor sees the 12-d BC window
        y = self.bc.dy(self.bc.f(self.bc.nx(o[:, :12])))
        return y[:, :2]

    def value(self, o):
        return self.v(((o - self.mu) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)).squeeze(-1)

    def dist(self, o):
        return torch.distributions.Normal(self.mean(o), self.log_std.exp()), self.value(o)

    def update_norm(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mu; tot = self.cnt + bc
        self.mu += d * bc / tot
        self.var = (self.var * self.cnt + bv * bc + d ** 2 * self.cnt * bc / tot) / tot
        self.cnt = tot


def train(total_steps, out, n_workers=16, n_steps=128, actor_lr=3e-5, critic_warmup=10, save_every=25):
    os.makedirs(out, exist_ok=True)
    venv = VecEnv(n_workers, 4); n_envs = venv.n
    ac = BCActorCritic(torch.load(BC_CKPT, map_location="cpu")["model"])
    torch.save(ac.state_dict(), f"{out}/bcppo_0k.pt")
    opt_a = torch.optim.Adam(list(ac.bc.parameters()) + [ac.log_std], lr=actor_lr)
    opt_c = torch.optim.Adam(ac.v.parameters(), lr=3e-4)
    gamma, lam = 0.99, 0.95
    o = venv.obs()
    steps, it, t0 = 0, 0, time.time()
    ep_succ, cur = [], np.zeros(n_envs)
    while steps < total_steps:
        O, A, LP, R, V, DN = [], [], [], [], [], []
        torch.set_num_threads(1)
        for t in range(n_steps):
            ot = torch.tensor(o)
            with torch.no_grad():
                d, v = ac.dist(ot); a = d.sample(); lp = d.log_prob(a).sum(-1)
            o2, r, term, trunc, last = venv.step(a.numpy())
            r = r.copy()
            if trunc.any():
                with torch.no_grad():
                    r[trunc] += gamma * ac.value(torch.tensor(last[trunc])).numpy()
            O.append(ot); A.append(a); LP.append(lp); R.append(torch.tensor(r)); V.append(v)
            DN.append(torch.tensor(term | trunc, dtype=torch.float32))
            for i in np.where(term | trunc)[0]:
                ep_succ.append(bool(term[i]))
            o = o2
        torch.set_num_threads(4)
        with torch.no_grad():
            v_last = ac.value(torch.tensor(o))
        adv = torch.zeros(n_steps, n_envs); g = torch.zeros(n_envs)
        for t in reversed(range(n_steps)):
            nv = v_last if t == n_steps - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - DN[t]) - V[t]
            g = delta + gamma * lam * (1 - DN[t]) * g; adv[t] = g
        Ob, Ab, LPb = torch.cat(O), torch.cat(A), torch.cat(LP)
        ADV = adv.reshape(-1); RET = ADV + torch.cat(V)
        ac.update_norm(Ob)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        N = len(Ob)
        for ep in range(10):
            perm = torch.randperm(N)
            for i in range(0, N, 1024):
                idx = perm[i:i + 1024]
                d, v = ac.dist(Ob[idx])
                vl = ((v - RET[idx]) ** 2).mean()
                opt_c.zero_grad(); vl.backward(); nn.utils.clip_grad_norm_(ac.v.parameters(), 0.5); opt_c.step()
                if it >= critic_warmup:                     # let the critic catch up before moving the BC actor
                    d, _ = ac.dist(Ob[idx])
                    ratio = (d.log_prob(Ab[idx]).sum(-1) - LPb[idx]).exp()
                    pl = -torch.min(ratio * ADV[idx], ratio.clamp(0.8, 1.2) * ADV[idx]).mean()
                    opt_a.zero_grad(); pl.backward()
                    nn.utils.clip_grad_norm_(list(ac.bc.parameters()) + [ac.log_std], 0.5); opt_a.step()
        steps += N; it += 1
        if it % 5 == 0:
            rs = ep_succ[-300:]
            print(f"steps {steps/1e6:5.2f}M  eps {len(ep_succ):6d}  succ(last300) {np.mean(rs) if rs else 0:.2f}  "
                  f"std {ac.log_std.exp().detach().numpy().round(3)}  {(time.time()-t0)/60:.1f} min", flush=True)
        if it % save_every == 0:
            torch.save(ac.state_dict(), f"{out}/bcppo_{steps//1000}k.pt")
    torch.save(ac.state_dict(), f"{out}/bcppo_final.pt")


if __name__ == "__main__":
    train(float(sys.argv[1]), sys.argv[2])
