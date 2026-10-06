"""Paired evaluation of Scripted expert / BC / BC+PPO on the same unseen seeds, and a PPO parallelism benchmark.

Evaluation / benchmark code only: physics, training code and checkpoints are not changed.

    python pusht_rl_eval.py                       # paired evaluation, seeds 0-99  -> eval_results/
    python pusht_rl_eval.py --bc-exec 1           # BC executed one action per step (= the PPO start point)
    python pusht_rl_eval.py --benchmark           # PPO parallelism / throughput benchmark (short run, temp dir)

Every policy: same PushTEnv.reset(seed), max 300 control steps (30 s), same PushTEnv.is_success()
(position < 0.15 m and angle < 10 deg), deterministic actions (expert rules, BC regression output,
PPO Gaussian mean; no exploration noise).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time

import multiprocessing as mp
import numpy as np
import torch

import pusht_rl as R

HERE = R.HERE
OUT_DIR = os.path.join(HERE, "eval_results")
DT = R.PushTEnv.control_dt                         # 0.1 s per control step
BC_CKPT = R.BC_CKPT
PPO_FINAL = os.path.join(R.RL_DIR, "bc_ppo", "bcppo_final.pt")


# ----------------------------------------------------------------------------- policies
def load_bc(path=BC_CKPT, width=512):
    sd = torch.load(path, map_location="cpu")["model"]
    m = R.BCPolicy(12, 16, width=width)
    for k in ("x_mu", "x_sd", "y_mu", "y_sd"):
        m.register_buffer(k, sd[k])
    m.load_state_dict(sd)
    return m.eval()


def file_id(path):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    return {"path": os.path.relpath(path, HERE), "sha256_12": h, "mtime": time.ctime(os.path.getmtime(path))}


def build_policies(bc_ckpt, ppo_ckpt, bc_exec):
    """{name: factory(env) -> agent}  plus a description of what was loaded."""
    bc = load_bc(bc_ckpt)
    ppo = R.load_policy(ppo_ckpt)
    assert ppo.kind == "bc", f"{ppo_ckpt} is not a BC+PPO checkpoint"
    pols = {
        "scripted": lambda env: R.ScriptedExpert(env),
        "bc": lambda env: R.LearnedAgent(bc, exec_horizon=bc_exec),
        "bc_ppo": lambda env: R.RLAgent(env, ppo),
    }
    info = {"scripted": "ScriptedExpert (rules, deterministic)",
            "bc": {"checkpoint": file_id(bc_ckpt), "obs_horizon": 2, "pred_horizon": 8, "exec_horizon": bc_exec},
            "bc_ppo": {"checkpoint": file_id(ppo_ckpt), "action": "Gaussian mean (no sampling)"}}
    return pols, info


# ----------------------------------------------------------------------------- paired evaluation
_AGENTS = {}


def _episode(job):
    name, seed = job
    torch.set_num_threads(1)
    env, agent = _AGENTS[name]
    S, A, ok = R.run_episode(env, agent, seed)           # env.reset(seed) inside; stops at success or 300 steps
    d, a = env.sim.goal_error()
    steps = len(S)
    return {"seed": seed, "policy": name, "success": int(ok),
            "steps_to_success": steps if ok else "",
            "episode_steps": steps,
            "simulated_time": round(steps * DT, 1),
            "final_pos_err": round(float(d), 4), "final_ang_err_deg": round(float(a), 2)}


def paired_eval(policies, seeds, procs):
    _AGENTS.clear()
    for name, make in policies.items():
        env = R.PushTEnv(max_steps=R.MAX_STEPS)
        _AGENTS[name] = (env, make(env))
    jobs = [(name, s) for s in seeds for name in policies]
    with mp.get_context("fork").Pool(procs) as pool:       # agents inherited by fork; each job resets its env
        rows = pool.map(_episode, jobs, chunksize=2)
    return sorted(rows, key=lambda r: (r["seed"], list(policies).index(r["policy"])))


def summarize(rows, names):
    out = {}
    for name in names:
        rs = [r for r in rows if r["policy"] == name]
        ok = np.array([r["success"] for r in rs], bool)
        st = np.array([r["episode_steps"] for r in rs], float)[ok]
        pe = np.array([r["final_pos_err"] for r in rs])
        ae = np.array([r["final_ang_err_deg"] for r in rs])
        p = ok.mean()
        q = lambda f: float(f(st)) if ok.any() else float("nan")
        out[name] = {"n": len(rs), "successes": int(ok.sum()), "success_rate": float(p),
                     "success_rate_se": math.sqrt(p * (1 - p) / len(rs)),
                     "steps_median": q(np.median), "steps_mean": q(np.mean),
                     "steps_p90": q(lambda x: np.percentile(x, 90)),
                     "time_median_s": q(np.median) * DT, "time_mean_s": q(np.mean) * DT,
                     "time_p90_s": q(lambda x: np.percentile(x, 90)) * DT,
                     "final_pos_err_median": float(np.median(pe)), "final_ang_err_median_deg": float(np.median(ae))}
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_diff(rows, a, b):
    by = {}
    for r in rows:
        by.setdefault(r["seed"], {})[r["policy"]] = r
    seeds = sorted(by)
    sa = {s: by[s][a]["success"] for s in seeds}
    sb = {s: by[s][b]["success"] for s in seeds}
    both = [s for s in seeds if sa[s] and sb[s]]
    only_a = [s for s in seeds if sa[s] and not sb[s]]
    only_b = [s for s in seeds if sb[s] and not sa[s]]
    neither = [s for s in seeds if not sa[s] and not sb[s]]
    t_a = np.array([by[s][a]["episode_steps"] for s in both], float)
    t_b = np.array([by[s][b]["episode_steps"] for s in both], float)
    res = {"both_success": len(both), f"only_{a}": only_a, f"only_{b}": only_b, "both_fail": neither,
           "mcnemar_exact_p": mcnemar_exact(len(only_a), len(only_b))}
    if both:
        d = t_b - t_a                                       # negative = b faster
        res.update({"on_both_success": {
            f"{a}_median_steps": float(np.median(t_a)), f"{b}_median_steps": float(np.median(t_b)),
            "median_diff_steps(b-a)": float(np.median(d)), "mean_diff_steps(b-a)": float(d.mean()),
            f"{b}_faster_on": int((d < 0).sum()), f"{a}_faster_on": int((d > 0).sum()), "ties": int((d == 0).sum())}})
    return res


def print_report(summary, diff, a, b, info, seeds):
    print(f"\nPaired evaluation on unseen seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} tasks), "
          f"max {R.MAX_STEPS} steps, deterministic actions")
    for k, v in info.items():
        print(f"  {k:9s}: {v}")
    hdr = (f"{'policy':10s} {'success':>12s} | {'completion steps (successes only)':^34s} | "
           f"{'time [s]':^22s} | {'median final err':^18s}")
    print("\n" + hdr)
    print(f"{'':10s} {'':>12s} | {'median':>10s} {'mean':>11s} {'p90':>11s} | {'median':>6s} {'mean':>7s} {'p90':>7s} |"
          f" {'pos [m]':>8s} {'ang[deg]':>9s}")
    print("-" * len(hdr))
    for name, s in summary.items():
        print(f"{name:10s} {s['successes']:3d}/{s['n']:<3d} {s['success_rate'] * 100:5.1f}% | "
              f"{s['steps_median']:10.1f} {s['steps_mean']:11.1f} {s['steps_p90']:11.1f} | "
              f"{s['time_median_s']:6.1f} {s['time_mean_s']:7.1f} {s['time_p90_s']:7.1f} | "
              f"{s['final_pos_err_median']:8.3f} {s['final_ang_err_median_deg']:9.2f}")
    print(f"\nSame-seed comparison: {a} vs {b}")
    print(f"  both succeed      : {diff['both_success']}")
    print(f"  only {a:12s} : {len(diff['only_' + a]):3d}   seeds {diff['only_' + a]}")
    print(f"  only {b:12s} : {len(diff['only_' + b]):3d}   seeds {diff['only_' + b]}")
    print(f"  both fail         : {len(diff['both_fail']):3d}   seeds {diff['both_fail']}")
    print(f"  exact McNemar p (success difference) = {diff['mcnemar_exact_p']:.3f}")
    if "on_both_success" in diff:
        print(f"  on the {diff['both_success']} seeds both solve: {diff['on_both_success']}")


# ----------------------------------------------------------------------------- parallelism benchmark
def benchmark(iters, bc_ckpt):
    """Runs the real R.train_ppo('bc', ...) for `iters` PPO iterations in a temp dir and times its phases.

    train_ppo calls torch.set_num_threads(1) when a rollout starts and torch.set_num_threads(4) when the
    PPO update starts; those calls are used as phase markers (the training code itself is not modified).
    ParallelEnvs.step is wrapped to time the simulation part of the rollout.
    """
    marks, env_time, info = [], [0.0], {}
    real_set = torch.set_num_threads
    real_step, real_init = R.ParallelEnvs.step, R.ParallelEnvs.__init__

    def set_threads(n):
        marks.append((time.perf_counter(), n))
        real_set(n)

    def init(self, *a, **kw):
        t = time.perf_counter()
        real_init(self, *a, **kw)
        kids = mp.active_children()
        info.update({"backend": "CPU multiprocessing (multiprocessing.Pipe workers, start method "
                                f"'{mp.get_start_method(allow_none=True) or 'fork'}')",
                     "processes": len(self.remotes), "envs_per_process": self.k, "total_envs": self.n,
                     "worker_pids": [p.pid for p in kids][:4] + (["..."] if len(kids) > 4 else []),
                     "cpu_cores": os.cpu_count(), "spawn_time_s": time.perf_counter() - t})

    def step(self, actions):
        t = time.perf_counter()
        out = real_step(self, actions)
        env_time[0] += time.perf_counter() - t
        return out

    tmp = tempfile.mkdtemp(prefix="ppo_bench_")
    torch.set_num_threads, R.ParallelEnvs.step, R.ParallelEnvs.__init__ = set_threads, step, init
    try:
        t0 = time.perf_counter()
        model = R.train_ppo("bc", iters * 64 * 128, tmp, actor_lr=3e-5, critic_warmup=0, save_every=10**9,
                            bc_ckpt=bc_ckpt)
        wall = time.perf_counter() - t0
    finally:
        torch.set_num_threads, R.ParallelEnvs.step, R.ParallelEnvs.__init__ = real_set, real_step, real_init
        shutil.rmtree(tmp, ignore_errors=True)
    t_end = t0 + wall
    roll, upd = [], []
    for (t, n), nxt in zip(marks, marks[1:] + [(t_end, None)]):
        (roll if n == 1 else upd).append(nxt[0] - t)
    steps = iters * 64 * 128
    info.update({
        "model_device": str(next(model.parameters()).device),
        "ppo_iterations": iters, "env_steps": steps, "steps_per_iteration": 64 * 128,
        "wall_clock_s": wall,
        "rollout_s_per_iter": float(np.mean(roll)), "ppo_update_s_per_iter": float(np.mean(upd)),
        "env_step_s_per_iter (inside ParallelEnvs.step)": env_time[0] / iters,
        "env_steps_per_s (rollout only)": steps / sum(roll),
        "env_steps_per_s (overall incl. update)": steps / (sum(roll) + sum(upd)),
        "update_fraction_of_time": sum(upd) / (sum(roll) + sum(upd)),
    })
    return info


def logged_wallclock():
    """Wall-clock of the real training runs, from their logs.

    bc_ppo/train_log.txt holds run 1 (prototype log format, '... eps ...') followed by the lines that the
    resumed run appended ('... episodes ...'); the resumed run is also in train_log_more.txt.
    """
    import re
    pat = re.compile(r"steps\s+([\d.]+)M.*?([\d.]+) min")
    runs = (("RL from scratch", "scratch/train_log.txt", " eps "),
            ("BC+PPO run 1 (0 -> 6M)", "bc_ppo/train_log.txt", " eps "),
            ("BC+PPO resumed (6M -> )", "bc_ppo/train_log_more.txt", " episodes "))
    out = {}
    for name, rel, key in runs:
        p = os.path.join(R.RL_DIR, rel)
        if not os.path.exists(p):
            continue
        hits = [pat.search(l) for l in open(p) if key in l]
        hits = [h for h in hits if h]
        if len(hits) < 2:
            continue
        s0, s1 = float(hits[0].group(1)), float(hits[-1].group(1))
        m0, m1 = float(hits[0].group(2)), float(hits[-1].group(2))
        out[name] = {"logged_steps_M": [s0, s1], "wall_clock_min": m1,
                     "avg_env_steps_per_s_incl_updates": (s1 - s0) * 1e6 / ((m1 - m0) * 60)}
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-99", help="unseen seeds, e.g. 0-99")
    ap.add_argument("--bc-ckpt", default=BC_CKPT)
    ap.add_argument("--ppo-ckpt", default=PPO_FINAL)
    ap.add_argument("--bc-exec", type=int, default=4, help="BC actions executed per prediction (tutorial: 4)")
    ap.add_argument("--procs", type=int, default=8, help="evaluation processes")
    ap.add_argument("--benchmark", action="store_true", help="run the PPO parallelism benchmark instead")
    ap.add_argument("--bench-iters", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.benchmark:
        res = {"short_benchmark": benchmark(args.bench_iters, args.bc_ckpt), "from_training_logs": logged_wallclock()}
        path = os.path.join(OUT_DIR, "ppo_parallel_benchmark.json")
        json.dump(res, open(path, "w"), indent=2)
        print(json.dumps(res, indent=2))
        print("->", os.path.relpath(path, HERE))
        return

    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    pols, info = build_policies(args.bc_ckpt, args.ppo_ckpt, args.bc_exec)
    t0 = time.perf_counter()
    rows = paired_eval(pols, seeds, args.procs)
    names = list(pols)
    summary = summarize(rows, names)
    diff = paired_diff(rows, "scripted", "bc_ppo")
    print_report(summary, diff, "scripted", "bc_ppo", info, seeds)

    ppo_tag = os.path.splitext(os.path.basename(args.ppo_ckpt))[0]
    if ppo_tag == "bcppo_final":                         # name the file after the step count it really is
        d = os.path.dirname(args.ppo_ckpt)
        fin = torch.load(args.ppo_ckpt, map_location="cpu")
        def same(f):                                      # equal weights (torch files differ in their archive name)
            o = torch.load(os.path.join(d, f), map_location="cpu")
            return o.keys() == fin.keys() and all(torch.equal(o[k], fin[k]) for k in fin)
        numbered = sorted((f for f in os.listdir(d) if f.startswith("bcppo_") and f.endswith("k.pt")),
                          key=lambda f: -int(f[6:-4]))
        ppo_tag = next((os.path.splitext(f)[0] for f in numbered if same(f)), ppo_tag)
    tag = f"seeds{lo}-{hi}_{ppo_tag}" + ("" if args.bc_exec == 4 else f"_bcexec{args.bc_exec}")
    csv_path = os.path.join(OUT_DIR, f"paired_eval_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json_path = os.path.join(OUT_DIR, f"paired_summary_{tag}.json")
    json.dump({"seeds": [lo, hi], "max_steps": R.MAX_STEPS, "control_dt_s": DT,
               "success_criterion": "pos err < 0.15 m and angle err < 10 deg (PushTEnv.is_success)",
               "policies": info, "summary": summary, "scripted_vs_bc_ppo": diff,
               "eval_wall_clock_s": time.perf_counter() - t0}, open(json_path, "w"), indent=2)
    print(f"\n-> {os.path.relpath(csv_path, HERE)}\n-> {os.path.relpath(json_path, HERE)}"
          f"   ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
