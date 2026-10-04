import sys, time, math, pickle
import torch; torch.set_num_threads(4)
import learn
from learn import *

MODE = sys.argv[4] if len(sys.argv) > 4 else "goal_abs"   # goal_abs | goal_rel

def to_goal(S):
    """raw (...,8) -> R_g^T (x - g) frame"""
    g, gth = S[..., 5:7], S[..., 7]
    c, s = np.cos(gth), np.sin(gth)
    def rot(v):  # R^T v
        v = v - g
        return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)
    return rot(S[..., 0:2]), rot(S[..., 2:4]), S[..., 4] - gth

def feat_g(S):
    p, T, d = to_goal(S)
    return np.concatenate([p, T, np.cos(d)[..., None], np.sin(d)[..., None]], -1)

def act_to_goal(S_t, A):   # A (...,2) world targets -> goal frame (abs or relative to pusher at time t)
    g, gth = S_t[5:7], S_t[7]; c, s = math.cos(gth), math.sin(gth)
    v = A - (S_t[0:2] if MODE == "goal_rel" else g)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)

def act_from_goal(S_t, Ag):
    g, gth = S_t[5:7], S_t[7]; c, s = math.cos(gth), math.sin(gth)
    v = np.stack([c * Ag[..., 0] - s * Ag[..., 1], s * Ag[..., 0] + c * Ag[..., 1]], -1)
    return v + (S_t[0:2] if MODE == "goal_rel" else g)

def windows(eps):
    O, Y = [], []
    for S, A in eps:
        F = feat_g(S); n = len(S)
        for t in range(n):
            io = [max(0, t - k) for k in range(OH - 1, -1, -1)]
            ia = [min(n - 1, t + k) for k in range(AH)]
            O.append(F[io].ravel()); Y.append(act_to_goal(S[t], A[ia]).ravel())
    return np.array(O, np.float32), np.array(Y, np.float32)

def evaluate(policy, n=50, seed0=0, exec_h=4, max_steps=300):
    env = PushTEnv(max_steps=max_steps); succ = []; errs = []
    for seed in range(seed0, seed0 + n):
        env.reset(seed); s = env.sim
        raw = lambda: np.r_[s.pusher.pos, s.T.pos, s.T.angle, s.goal_pos, s.goal_angle]
        hist = [raw()] * OH; done = False
        while not done:
            St = hist[-1]
            plan = act_from_goal(St, policy(feat_g(np.array(hist[-OH:])).ravel()))
            for a in plan[:exec_h]:
                _, ok, done = env.step(a); hist.append(raw())
                if done: break
        succ.append(ok); errs.append(env.sim.goal_error())
    errs = np.array(errs)
    return float(np.mean(succ)), np.median(errs, 0).round(2)

if __name__ == "__main__":
    N = int(sys.argv[1]); kinds = sys.argv[2].split(",")
    eps = pickle.load(open("eps.pkl", "rb"))
    O, Y = windows(eps[:N]); print("samples", len(O), MODE)
    for kind in kinds:
        pol = train(kind, O, Y, epochs=int(sys.argv[3]))
        t = time.time(); print(kind, "N", N, "success / median err", evaluate(pol), f"eval {time.time()-t:.0f}s", flush=True)
