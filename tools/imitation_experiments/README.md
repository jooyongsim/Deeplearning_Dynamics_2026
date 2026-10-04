# imitation_experiments — scratch work behind `tutorial/pusht_imitation.py`

Prototypes and experiments used to choose the tutorial's design and the numbers quoted in it.
Copied from the session scratchpad as-is (checkpoints `*.pt`, `*.pkl`, the venv and all `*.npz` except `full/demos/expert_400.npz` were left out).
`full/`, `smoke/`, `final/` each contain a **snapshot** of `pusht_imitation.py` from the time they were run, because the
scripts there read `pusht_imitation.py` from the current directory — they are not the current tutorial.

## Top level — prototypes (before the tutorial existed)

| File | What it is | Result files |
|---|---|---|
| `proto.py` | Random-task env + scripted expert prototype; `python proto.py 40` prints expert success rate | `fail.png` (failure plots) |
| `fastcol.py` | Vectorised `circle_vs_T` (became `circle_vs_T_fast`) | — |
| `learn.py` | First BC / diffusion trainer, world-frame features → 0% success; collects `eps.pkl` | — |
| `learn2.py` | Same with goal-frame features; argv 4 = `goal_abs` or `goal_rel` (action representation) | `r_*_goal_*.txt` → 7.1 table (26/36/20/46%) |
| `diag.py`, `diag2.py` | Rollout plot of the MLP; train-seed vs test-seed success (50% vs 30%) | `diag.png` |

## `full/` — experiments on the tutorial code (run from inside `full/`)

| File | What it is | Result files → where quoted |
|---|---|---|
| `run1.log` | First full tutorial run (width 256, lr 1e-3, no EMA): expert 88%, MLP 32%, Diffusion 8% | section 9–11 table row 1 |
| `study.py` | `kind steps ema lr K` — EMA / steps / lr / K sweep | `st_*.txt` → section 9–11 table |
| `study2.py` | `kind steps lr ema exec_list` — uses the tutorial's `train()` (with EMA) | `s2_*.txt` (exec horizon 2/4/8, lr) |
| `collect400.py` | 400 expert demos → `demos/expert_400.npz` (included: 400 episodes, 63,441 steps, seeds 100000+) | — |
| `scale.py` | `kind N` — width 256, 30k steps, eval 100 | `sc_*.txt` (first scaling run) |
| `scale2.py` | `kind N steps width` | `sc2_*.txt`, `sc3_*.txt` (longer / wider), `sc4_*.txt` |
| `run_scale512.sh` | All N ∈ {25, 50, 100, 180, 360} × {mlp, diff} with `scale2.py`, width 512, 30k steps | **`sc4_*.txt` → section 6 table** |
| `viewer_test.py` | Headless test of `PolicyViewer` | `viewer_1.png`, `viewer_2.png` |

## `smoke/`, `final/`

- `smoke/` — tutorial run with tiny settings (30 demos, 1000 steps) to catch errors.
- `final/` — the tutorial with its default settings, end to end: `run.log` (expert 88%, MLP-BC 38%, Diffusion 54%) and the figures.

## Re-running

Paths are as they were in the scratchpad:

- `proto.py` adds `/home/cosmos/claude/04_DeepL_Dyna_Lecture/Deeplearning_Dynamics_2026/tutorial` to `sys.path`.
- `run_scale512.sh` calls `../venv/bin/python` — replace with your Python (needs numpy, matplotlib, torch).
- `scale*.py` read `demos/expert_400.npz`, which is included in `full/demos/`. `collect400.py` regenerates the same file (fixed seeds, about 4 minutes).
