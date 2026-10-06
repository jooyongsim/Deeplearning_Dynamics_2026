"""Figures for the imitation-learning / RL lecture deck (ADL_2026_ImitationRL.html).

    ../.venv/bin/python make_il_figures.py          # -> ../slides/assets/figures/il_*.png

Numbers come from the logged experiments, not re-run here:
  scaling table      imitation_experiments/full/sc4_*.txt   (also pusht_imitation.py, section 6)
  RL training curves rl_experiments/log_a.txt, tutorial/checkpoints/rl/bc_ppo/train_log*.txt
  paired evaluation  tutorial/eval_results/paired_summary_seeds0-99_*.json
The tutorial figures (random tasks, rollouts, toy multimodality) are copied from
imitation_experiments/final/figures_imitation/. The realtime snapshot drives
tutorial/pusht_rl_realtime.py headless on one fixed seed.
"""
import json
import os
import re
import shutil
import sys

os.environ.setdefault("PUSHT_NO_GUI", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TUT = os.path.join(ROOT, "tutorial")
OUT = os.path.join(ROOT, "slides", "assets", "figures")

NAVY, BLUE, ORANGE, GREEN, PURPLE, GREY, RED = "#203864", "#2774B8", "#E97820", "#3D8C54", "#6D237C", "#7F7F7F", "#FF0000"
plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "Arimo", "DejaVu Sans"], "font.size": 15,
    "axes.edgecolor": "#404040", "axes.labelcolor": "#111111", "xtick.color": "#404040", "ytick.color": "#404040",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E7E6E6",
    "grid.linewidth": 1.0, "axes.axisbelow": True, "savefig.dpi": 200, "legend.frameon": False,
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


def scaling():
    ns = [25, 50, 100, 180, 360]
    mlp = [9, 26, 32, 48, 59]
    dif = [20, 48, 61, 63, 68]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(90, color=GREY, ls="--", lw=1.6)
    ax.text(26, 92, "scripted expert ≈ 90%", color=GREY, fontsize=13)
    ax.plot(ns, dif, "-o", color=ORANGE, lw=3, ms=8, label="Diffusion Policy")
    ax.plot(ns, mlp, "-s", color=BLUE, lw=3, ms=8, label="MLP-BC")
    for x, a, b in zip(ns, dif, mlp):
        ax.annotate(f"{a}", (x, a), textcoords="offset points", xytext=(0, 9), ha="center", color=ORANGE, fontsize=12)
        ax.annotate(f"{b}", (x, b), textcoords="offset points", xytext=(0, -19), ha="center", color=BLUE, fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    ax.set_ylim(0, 100); ax.set_xlim(20, 450)
    ax.set_xlabel("number of training demonstrations N")
    ax.set_ylabel("success on 100 unseen tasks [%]")
    ax.legend(loc="lower right")
    save(fig, "il_scaling.png")


def _parse(path, key):
    xs, ys = [], []
    for line in open(path, encoding="utf-8"):
        m = re.search(r"steps\s+([\d.]+)M.*?" + key + r"\s+([\d.]+)", line)
        if m:
            xs.append(float(m.group(1))); ys.append(float(m.group(2)))
    return xs, ys


def rl_curves():
    sx, sy = _parse(os.path.join(HERE, "rl_experiments", "log_a.txt"), r"succ\(last200\)")
    bx, by = _parse(os.path.join(TUT, "checkpoints", "rl", "bc_ppo", "train_log.txt"), r"succ(?:ess)?\(last ?300\)")
    mx, my = _parse(os.path.join(TUT, "checkpoints", "rl", "bc_ppo", "train_log_more.txt"), r"success\(last 300\)")
    pts = sorted(set(zip(bx + mx, by + my)))
    bx, by = [p[0] for p in pts], [p[1] for p in pts]

    def smooth(y, k=9):
        out = []
        for i in range(len(y)):
            w = y[max(0, i - k + 1): i + 1]
            out.append(sum(w) / len(w))
        return out

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(88, color=GREY, ls="--", lw=1.6)
    ax.text(0.2, 90, "scripted expert (88%)", color=GREY, fontsize=13)
    ax.plot(bx, [100 * v for v in by], color=ORANGE, lw=0.8, alpha=0.35)
    ax.plot(bx, [100 * v for v in smooth(by)], color=ORANGE, lw=3, label="BC → RL (PPO fine-tuning)")
    ax.plot(sx, [100 * v for v in sy], color=BLUE, lw=3, label="RL from scratch")
    ax.axvline(6.0, color=NAVY, lw=1.2, ls=":")
    ax.text(6.1, 8, "resume\n(6M → 12M)", color=NAVY, fontsize=12)
    ax.set_xlim(0, 16.5); ax.set_ylim(-3, 100)
    ax.set_xlabel("environment steps [million]")
    ax.set_ylabel("training success, last episodes [%]")
    ax.legend(loc="center right")
    save(fig, "il_rl_curves.png")


def paired():
    d6 = json.load(open(os.path.join(TUT, "eval_results", "paired_summary_seeds0-99_bcppo_6004k.json")))
    d12 = json.load(open(os.path.join(TUT, "eval_results", "paired_summary_seeds0-99_bcppo_12008k.json")))
    rows = [("Scripted\nexpert", d6["summary"]["scripted"], GREY),
            ("BC\n(MLP-BC)", d6["summary"]["bc"], BLUE),
            ("BC+PPO\n6M", d6["summary"]["bc_ppo"], ORANGE),
            ("BC+PPO\n12M", d12["summary"]["bc_ppo"], "#F2A65E")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    xs = range(len(rows))
    for i, (lab, s, c) in enumerate(rows):
        r, se = 100 * s["success_rate"], 100 * s["success_rate_se"]
        a1.bar(i, r, color=c, width=0.62, yerr=se, capsize=5, error_kw={"lw": 1.4, "ecolor": "#404040"})
        a1.text(i, r + se + 2, f"{r:.0f}%", ha="center", fontsize=14, fontweight="bold", color="#111111")
        a2.bar(i, s["time_median_s"], color=c, width=0.62)
        a2.text(i, s["time_median_s"] + 0.4, f"{s['time_median_s']:.1f} s", ha="center", fontsize=14,
                fontweight="bold", color="#111111")
    for a in (a1, a2):
        a.set_xticks(list(xs)); a.set_xticklabels([r[0] for r in rows], fontsize=13)
        a.grid(axis="x", visible=False)
    a1.set_ylim(0, 105); a1.set_ylabel("success rate [%]  (±1 s.e.)")
    a2.set_ylim(0, 19); a2.set_ylabel("median time to success [s]")
    fig.tight_layout(w_pad=3)
    save(fig, "il_paired.png")


def realtime_snapshot(seed=30, max_frames=640, at_frame=None):
    sys.path.insert(0, TUT)
    cwd = os.getcwd()
    os.chdir(TUT)
    try:
        import pusht_rl_realtime as RT
        demo = RT.RealtimeDemo(RT.ALL_PANELS)
        demo.seed = seed
        for p in demo.panels:
            p.start(seed)
        for k in range(max_frames):
            for p in demo.panels:
                p.frame()
            if at_frame is not None and k + 1 >= at_frame:
                break
            if all(p.done for p in demo.panels):
                break
        for p in demo.panels:
            p.draw()
        demo.header.set_text(f"pusht_rl_realtime.py — all five policies on the same unseen task (seed {seed})")
        demo.fig.set_size_inches(12.0, 6.9)
        demo.fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.6)
        save(demo.fig, "il_realtime.png")
        print({p.name: (p.success, p.env.t) for p in demo.panels})
    finally:
        os.chdir(cwd)


def copy_tutorial_figs():
    src = os.path.join(HERE, "imitation_experiments", "final", "figures_imitation")
    for f, dst in [("01_random_tasks.png", "il_random_tasks.png"), ("02_expert_rollouts.png", "il_expert_rollouts.png"),
                   ("03_sample_goal_frame.png", "il_goal_frame.png"), ("04_multimodality_toy.png", "il_toy.png"),
                   ("07_rollouts_bc.png", "il_rollouts_bc.png"), ("08_rollouts_diffusion.png", "il_rollouts_diff.png")]:
        shutil.copy(os.path.join(src, f), os.path.join(OUT, dst))
        print("copied", dst)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1:] or ["copy", "scaling", "rl", "paired", "realtime"]
    if "copy" in what:
        copy_tutorial_figs()
    if "scaling" in what:
        scaling()
    if "rl" in what:
        rl_curves()
    if "paired" in what:
        paired()
    if "realtime" in what:
        seed = int(os.environ.get("SNAP_SEED", "30"))
        realtime_snapshot(seed=seed)
