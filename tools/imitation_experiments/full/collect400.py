import os, io, contextlib
os.environ["PUSHT_NO_GUI"] = "1"
import matplotlib; matplotlib.use("Agg")
full = open("pusht_imitation.py", encoding="utf-8").read()
code = full[:full.index("expert = ScriptedExpert(env)")]
ns = {"__name__": "x", "__file__": os.path.abspath("pusht_imitation.py")}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(code, "t", "exec"), ns)
ns["N_EXPERT_DEMOS"] = 400; ns["EXPERT_PATH"] = "demos/expert_400.npz"
full2 = full[full.index("def collect_expert_demos"):full.index("if os.path.exists(EXPERT_PATH)")].replace("XX","XX") if False else full[full.index("def collect_expert_demos"):full.index("if os.path.exists(EXPERT_PATH)")]
exec(compile(full2, "t2", "exec"), ns)
ns["collect_expert_demos"](400, path="demos/expert_400.npz")
