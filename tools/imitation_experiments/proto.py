import math, os, sys, time
import numpy as np
os.environ["PUSHT_NO_GUI"] = "1"
sys.path.insert(0, "/home/cosmos/claude/04_DeepL_Dyna_Lecture/Deeplearning_Dynamics_2026/tutorial")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import pusht_keyboard_sim as K
from pusht_keyboard_sim import KeyboardPushT, ARENA, rotation_matrix, cross2
from fastcol import circle_vs_T_fast as circle_vs_T
K.circle_vs_T = circle_vs_T

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

class PushTEnv:
    control_dt = 0.1
    def __init__(self, max_steps=300, pos_tol=0.15, ang_tol=10.0):
        self.sim = KeyboardPushT(); self.max_steps=max_steps; self.pos_tol=pos_tol; self.ang_tol=ang_tol
    def reset(self, seed):
        rng = np.random.default_rng(seed)
        s = self.sim; s.reset()
        def pose(): return np.array([rng.uniform(-2.0,2.0), rng.uniform(-1.0,1.0)]), rng.uniform(-math.pi, math.pi)
        while True:
            p, a = pose(); g, ga = pose()
            if np.linalg.norm(p-g) > 1.0 or abs(wrap(a-ga)) > math.radians(45): break
        s.T.pos[:] = p; s.T.angle = a; s.goal_pos = g; s.goal_angle = ga
        while True:
            q = np.array([rng.uniform(-3.6,3.6), rng.uniform(-2.6,2.6)])
            if np.linalg.norm(q-p) > 1.6: break
        s.pusher.pos[:] = q; s.target = q.copy(); self.t = 0
        return self.obs()
    def obs(self):
        s=self.sim; return dict(pusher=s.pusher.pos.copy(), T=s.T.pos.copy(), angle=s.T.angle, goal=s.goal_pos.copy(), goal_angle=s.goal_angle)
    def step(self, target):
        x0,x1,y0,y1 = ARENA
        self.sim.target = np.clip(np.asarray(target,float), [x0,y0],[x1,y1])
        self.sim.advance(self.control_dt); self.t += 1
        d,a = self.sim.goal_error()
        success = d < self.pos_tol and a < self.ang_tol
        return self.obs(), success, success or self.t >= self.max_steps

class Expert:
    L = 0.8
    def __init__(self, sim, rng=None):
        self.sim = sim; V = sim.T.vertices; self.cands=[]
        for i in range(len(V)):
            a,b = V[i], V[(i+1)%len(V)]; d=b-a; out=np.array([-d[1],d[0]])/np.linalg.norm(d)
            for f in (0.15,0.5,0.85): self.cands.append((a+f*d, out))
        self.cur=None
        rp=sim.pusher.radius; self.phis=np.linspace(-math.pi, math.pi, 73)[:-1]; self.rhos=[]
        for phi in self.phis:
            u=np.array([math.cos(phi), math.sin(phi)])
            for rho in np.arange(0.4, 1.8, 0.05):
                if circle_vs_T(rho*u, rp+0.04, np.zeros(2), 0.0, V) is None: break
            self.rhos.append(rho+0.15)
        self.rhos=np.array(self.rhos)
    def act(self):
        s=self.sim; T=s.T; P=s.pusher; R=rotation_matrix(T.angle)
        e = s.goal_pos - T.pos; dth = wrap(s.goal_angle - T.angle)
        D = np.array([e[0], e[1], self.L*dth]); Dn = np.linalg.norm(D)
        rp = P.radius; x0,x1,y0,y1=ARENA
        world = T.pos + T.vertices @ R.T
        walls = [nw for dist, nw in ((world[:,0].min()-x0, (1,0)), (x1-world[:,0].max(), (-1,0)),
                                     (world[:,1].min()-y0, (0,1)), (y1-world[:,1].max(), (0,-1))) if dist < 0.3]
        best=None
        for k,(c,out) in enumerate(self.cands):
            cw = T.pos + R@c; ow = R@out; nin=-ow; r = cw - T.pos
            C = np.array([nin[0]/T.mass, nin[1]/T.mass, self.L*cross2(r,nin)/T.inertia])
            sc = C@D/(np.linalg.norm(C)*Dn+1e-9)
            stand = cw + ow*(rp+0.12)
            if not (x0+rp < stand[0] < x1-rp and y0+rp < stand[1] < y1-rp): continue
            if circle_vs_T(stand, rp, T.pos, T.angle, T.vertices) is not None: continue
            for nw in walls:
                if nin @ np.array(nw) < -0.3: sc -= 0.6       # would push the T into a wall it is touching
            if self.cur==k: sc += 0.1
            if best is None or sc>best[0]: best=(sc,k,cw,ow,stand)
        if best is None: return P.pos.copy()
        sc,k,cw,ow,stand = best; self.cur=k
        dist = np.linalg.norm(P.pos - stand)
        # are we in pushing position? pusher near the contact line behind cw
        rel = P.pos - cw; along = rel@ow; lat = abs(rel@np.array([-ow[1],ow[0]]))
        if lat < 0.12 and 0.0 < along < rp+0.25:
            step = float(np.clip(0.3*Dn, 0.04, 0.12))       # slower when close: the T keeps sliding
            return P.pos - ow*(step + 0.04)
        elif self._clear(P.pos, stand):
            goal_t = stand
        else:
            goal_t = self._orbit(P.pos, stand)
        d = goal_t - s.target
        n = np.linalg.norm(d)
        return s.target + (d if n < 0.2 else d/n*0.2)
    def _ring(self, phi):
        """Closest point at angle phi (around the COM) where the pusher clears the T, plus a margin."""
        T=self.sim.T; u=np.array([math.cos(phi), math.sin(phi)])
        rho=np.interp(wrap(phi-T.angle), self.phis, self.rhos, period=2*math.pi)
        return T.pos + rho*u
    def _inside(self, p):
        x0,x1,y0,y1=ARENA; rp=self.sim.pusher.radius
        return x0+rp <= p[0] <= x1-rp and y0+rp <= p[1] <= y1-rp
    def _orbit(self, p, stand):
        T=self.sim.T
        v=p-T.pos; a0=math.atan2(v[1],v[0]); vs=stand-T.pos; a1=math.atan2(vs[1],vs[0])
        here=self._ring(a0)
        if np.linalg.norm(p-T.pos) < np.linalg.norm(here-T.pos)-0.1:
            return here                              # first step out, away from the T
        best=None
        for sgn in (1,-1):
            da=(sgn*(a1-a0))%(2*math.pi)             # arc length (rad) going this way
            pts=[self._ring(a0+sgn*f) for f in np.arange(0.25, da, 0.25)]
            if all(self._inside(q) for q in pts) and (best is None or da<best[0]):
                best=(da, pts[0] if pts else stand)
        return best[1] if best else here
    def _clear(self, a, b):
        T=self.sim.T; rp=self.sim.pusher.radius
        for f in np.linspace(0,1,12):
            if circle_vs_T(a+f*(b-a), rp+0.05, T.pos, T.angle, T.vertices) is not None: return False
        return True

if __name__=="__main__":
    env=PushTEnv(max_steps=400); succ=[]; lens=[]; t0=time.perf_counter()
    N=int(sys.argv[1]) if len(sys.argv)>1 else 30
    for seed in range(N):
        env.reset(seed); ex=Expert(env.sim)
        while True:
            _,ok,done = env.step(ex.act())
            if done: break
        succ.append(ok); lens.append(env.t)
        d,a = env.sim.goal_error()
        print(seed, ok, env.t, f"{d:.2f} {a:.1f}")
    print("success", np.mean(succ), "mean len succ", np.mean([l for l,s in zip(lens,succ) if s]) if any(succ) else 0, "time", time.perf_counter()-t0)
