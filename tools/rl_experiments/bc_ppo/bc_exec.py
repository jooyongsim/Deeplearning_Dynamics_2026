import sys, torch, numpy as np, time
from defs import load_defs
D = load_defs(); torch.set_num_threads(4)
ck = torch.load(D["__name__"] and "/home/cosmos/claude/04_DeepL_Dyna_Lecture/Deeplearning_Dynamics_2026/tutorial/checkpoints/MLP-BC_expert_180ep_28626st_w512_30000.pt", map_location="cpu")
sd = ck["model"]
m = D["BCPolicy"](12, 16, width=512)
for k in ("x_mu","x_sd","y_mu","y_sd"): m.register_buffer(k, sd[k])
m.load_state_dict(sd); m.eval()
eh = int(sys.argv[1])
env = D["PushTEnv"](max_steps=300); ag = D["LearnedAgent"](m, exec_horizon=eh); s = []
t = time.time()
for seed in range(50):
    S, A, ok = D["run_episode"](env, ag, seed); s.append(ok)
print(f"MLP-BC exec={eh}: success {np.mean(s)*100:.0f}%  ({time.time()-t:.0f}s)")
