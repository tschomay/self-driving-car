"""Phase 4.7 after E3: how well does the out-lap map line up with the true track? The map-following teacher drives
the first 12 selection loops (start 0, no noise); prints odometer minus true position, the shift that best lines the
mapped curvature up with the true one, and their correlation.  python -m sdc.phase47.map_align"""
import json, numpy as np
from sdc.phase47 import where as W, e2_prep
from sdc.jumps import env as je
pol, env = W._driver("mapteacher")
courses = json.loads(e2_prep.OUT.read_text())["split"]["select"][:12]
lags_all=[]; err=[]
for c in courses:
    b = je.JumpBatch(1, courses=[c], **env); car=b.cars[0]; b._obs=np.zeros((1,b.obs_dim),np.float32); b._obs[0]=car.reset(c,x0=0.0)
    xs,kt,km,ss=[],[],[],[]
    done=False
    while not done:
        a=np.asarray(pol(b),np.float32); st=car.state()
        xs.append(st["x"]); kt.append(st["kappa"]); km.append(car.map_kappa(0.0)); ss.append(car.dead.s)
        _,_,t,tr,inf=b.step(a); done=bool(t[0] or tr[0])
    xs,kt,km,ss=map(np.array,(xs,kt,km,ss))
    # true kappa as function of distance; map kappa at odometer: offset = s - x
    off=ss-xs
    # best shift: true kappa at x+d vs map kappa
    best=[]
    for d in np.arange(-6,6.01,0.25):
        kd=np.interp(xs+d,xs,kt); best.append((np.mean((kd-km)**2),d))
    print(c, inf[0]["episode"]["reason"], "len",round(xs[-1],1),"odo-x start/mid/end",np.round(off[[0,len(off)//2,-1]],2),"best shift",min(best)[1],
          "corr",round(np.corrcoef(kt,km)[0,1],2), "kmax true/map",round(np.abs(kt).max(),2),round(np.abs(km).max(),2))
