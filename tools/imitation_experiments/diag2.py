import sys, pickle
sys.argv = ["x", "0", "mlp", "0", "goal_rel"]
import learn2
from learn2 import *
eps = pickle.load(open("eps.pkl", "rb"))
O, Y = windows(eps[:200])
# which seeds were the training eps? collect() started at 1000 and kept successes
env = PushTEnv(max_steps=400); train_seeds = []
seed = 1000
for S, A in eps[:200]:
    while True:
        env.reset(seed); s = env.sim
        if np.allclose(s.T.pos, S[0, 2:4]): train_seeds.append(seed); seed += 1; break
        seed += 1
print("train seeds", train_seeds[:5])
pol = train("mlp", O, Y, epochs=int(sys.argv_epochs) if False else 300)
def ev(seeds):
    res = []
    for sd in seeds:
        res.append(evaluate(pol, n=1, seed0=sd)[0])
    return np.mean(res)
print("train-seed success", ev(train_seeds[:30]), " test-seed success", evaluate(pol, n=30))
