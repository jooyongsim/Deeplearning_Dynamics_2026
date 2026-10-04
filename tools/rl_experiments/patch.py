p='proto_rl.py'; s=open(p).read()
a=s.index("def worker(remote, seed0):"); b=s.index("class VecEnv:")
s=s[:a]+'''def worker(remote, seed0, k):
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


'''+s[b:]
old='''    def __init__(self, n):
        ctx = mp.get_context("fork")
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, 1234 + i), daemon=True); p.start()
            self.remotes.append(a); self.procs.append(p)
        self.n = n

    def step(self, acts):
        for r, a in zip(self.remotes, acts): r.send(("step", a))
        out = [r.recv() for r in self.remotes]'''
assert old in s
s=s.replace(old,'''    def __init__(self, n, k=4):
        ctx = mp.get_context("fork")
        self.remotes, self.procs, self.k = [], [], k
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, 1234 + i, k), daemon=True); p.start()
            self.remotes.append(a); self.procs.append(p)
        self.n = n * k

    def step(self, acts):
        for j, r in enumerate(self.remotes): r.send(("step", acts[j * self.k:(j + 1) * self.k]))
        out = [x for r in self.remotes for x in r.recv()]''')
for a_,b_ in [("def train(total_steps, n_envs=16, n_steps=256,","def train(total_steps, n_envs=16, n_steps=128,"),
              ("    venv = VecEnv(n_envs)\n","    venv = VecEnv(n_envs, 4); n_envs = venv.n\n"),
              ("        ot = torch.tensor(o)\n        for t in range(n_steps):","        ot = torch.tensor(o)\n        torch.set_num_threads(1)\n        for t in range(n_steps):"),
              ("        ac.update_norm(Ob)","        torch.set_num_threads(4)\n        ac.update_norm(Ob)")]:
    assert a_ in s, a_; s=s.replace(a_,b_)
open(p,'w').write(s)
print("patched")
