import pickle, sys, torch
torch.set_num_threads(8)
from learn import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
eps = pickle.load(open("eps.pkl","rb"))
O, Y = windows(eps[:200])
pol = train("mlp", O, Y, epochs=40)
fig, axs = plt.subplots(1, 4, figsize=(22, 5))
env = PushTEnv(max_steps=300)
for ax, seed in zip(axs, range(4)):
    env.reset(seed); s = env.sim
    W = lambda p, a: p + s.T.vertices @ rotation_matrix(a).T
    ax.add_patch(Polygon(W(s.goal_pos, s.goal_angle), fill=False, ec='g', ls='--', lw=2))
    hist = [np.r_[s.pusher.pos, s.T.pos, s.T.angle, s.goal_pos, s.goal_angle]] * OH; ps=[]; tg=[]
    done=False; t=0
    while not done:
        plan = pol(feat(np.array(hist[-OH:])).ravel())
        for a in plan[:4]:
            _, ok, done = env.step(a); tg.append(a); ps.append(s.pusher.pos.copy()); t+=1
            hist.append(np.r_[s.pusher.pos, s.T.pos, s.T.angle, s.goal_pos, s.goal_angle])
            if t % 30 == 0: ax.add_patch(Polygon(W(s.T.pos, s.T.angle), fill=False, ec=plt.cm.viridis(t/300)))
            if done: break
    ps=np.array(ps); tg=np.array(tg); ax.plot(ps[:,0],ps[:,1],'orange'); ax.plot(tg[:,0],tg[:,1],'r.',ms=2)
    ax.set_xlim(-4,4); ax.set_ylim(-3,3); ax.set_aspect('equal'); ax.set_title(f"{seed} {np.round(s.goal_error(),2)}")
fig.savefig("diag.png", dpi=55)
