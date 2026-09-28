"""Fit a SIREN signed distance function to the Stanford bunny.

Caches the weights next to this script so examples_diffcpm/ex7 can load them.
About five minutes on four CPU cores. The printed geometric error is the honest
measure of the fit: how far the network's zero level set sits from the mesh,
over the band points the CPM solve will actually use.
"""

import os
import sys
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIREN_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bunny_siren.pkl")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from diffcpm.mesh import load_bunny
from diffcpm.sdf import (adam, siren_init, siren_apply, cp_level_set,
                         level_set_residual_norm, level_set_residuals)
from diffcpm.grid import make_grid1d, band_from_cp

m = load_bunny()
rng = np.random.default_rng(0)

# ---- samples: area-weighted surface points, jittered along the normal, plus box
tri = m.tri
area = 0.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)
prob = area/area.sum()
def surface_samples(n):
    f = rng.choice(len(tri), size=n, p=prob)
    u = rng.random((n,1)); v = rng.random((n,1))
    over = (u+v)>1; u[over]=1-u[over]; v[over]=1-v[over]
    a,b,c = tri[f,0], tri[f,1], tri[f,2]
    return a + u*(b-a) + v*(c-a)

t0=time.time()
NS=90000
S = surface_samples(NS)
jit_scales = np.array([0.01,0.03,0.08,0.20])
J = np.concatenate([S[i::len(jit_scales)] + rng.normal(scale=s, size=S[i::len(jit_scales)].shape)
                    for i,s in enumerate(jit_scales)])
B = rng.uniform(-1.25,1.25,size=(30000,3)); B[:,2]*=0.85
X = np.concatenate([J,B])
print("generated %d samples in %.1fs"%(len(X),time.time()-t0), flush=True)
t0=time.time(); Y = m.signed_distance(X, k=24); print("signed distance in %.1fs  range [%.3f, %.3f]"%(time.time()-t0,Y.min(),Y.max()), flush=True)
Xj, Yj = jnp.asarray(X), jnp.asarray(Y)

def make_loss(batch, eikonal=0.0):
    """Signed-distance regression, optionally with an eikonal penalty.

    Plain regression on sampled distances gets the rms error low while leaving a
    tail: on a first attempt the fit reached 0.008 rms and still put its zero
    level set up to 0.7 away from the mesh at a few points, which is larger than
    a CPM band half-width and made the downstream inverse problem useless.

    The eikonal term penalises |grad f| - 1. It is what makes the level sets
    parallel and the surface well located between samples, and it is also the
    condition the closest point Newton relies on: where |grad f| collapses, the
    Gauss-Newton step blows up. So this term targets both failure modes at once.
    """
    grad_f = jax.grad(siren_apply, argnums=1)

    def loss(p, k):
        i = jax.random.randint(k, (batch,), 0, Xj.shape[0])
        pred = jax.vmap(siren_apply, in_axes=(None, 0))(p, Xj[i])
        data = jnp.mean((pred - Yj[i])**2)
        if eikonal == 0.0:
            return data
        g = jax.vmap(grad_f, in_axes=(None, 0))(p, Xj[i])
        eik = jnp.mean((jnp.linalg.norm(g, axis=1) - 1.0)**2)
        return data + eikonal * eik
    return loss

# ---- timing probe to pick a net we can actually train
for widths in ((64,64,64),(128,128,128)):
    p0 = siren_init(jax.random.PRNGKey(0), 3, widths, w0_first=8.0, w0=8.0)
    lg = jax.value_and_grad(make_loss(4096))
    t0=time.time(); pr,_ = adam(lg, p0, 50, lr=1e-3, key=jax.random.PRNGKey(1)); jax.block_until_ready(pr['layers'][0][0])
    print("widths %s: 50 steps in %.1fs -> %.3f s/step"%(str(widths), time.time()-t0, (time.time()-t0)/50), flush=True)

# ---- the real fit
WIDTHS=(128,128,128); STEPS=9000; BATCH=4096; EIKONAL=0.1
p0 = siren_init(jax.random.PRNGKey(0), 3, WIDTHS, w0_first=8.0, w0=8.0)
lg = jax.value_and_grad(make_loss(BATCH, eikonal=EIKONAL))
t0=time.time()
params, fl = adam(lg, p0, STEPS, lr=2e-3, key=jax.random.PRNGKey(2))
jax.block_until_ready(params['layers'][0][0])
print("fit %s for %d steps (eikonal %.2f) in %.0fs, last-batch loss %.3e"%(str(WIDTHS),STEPS,EIKONAL,time.time()-t0,fl), flush=True)
full = float(jnp.mean((jax.vmap(siren_apply,in_axes=(None,0))(params, Xj[:40000]) - Yj[:40000])**2))
print("held-in mse over 40k samples: %.3e  (rms %.4f)"%(full, full**0.5), flush=True)
gg = jax.vmap(jax.grad(siren_apply,argnums=1), in_axes=(None,0))(params, Xj[:20000])
gn = np.asarray(jnp.linalg.norm(gg,axis=1))
print("|grad f| over 20k samples: mean %.4f  p5 %.4f  p95 %.4f  (1.0 is an exact SDF)"%(gn.mean(), np.percentile(gn,5), np.percentile(gn,95)), flush=True)
with open(SIREN_OUT,'wb') as fh:
    pickle.dump(jax.tree_util.tree_map(lambda a: np.asarray(a), params), fh)

# ---- geometric error of the learned surface, on a band
dx=0.09
x1d=[make_grid1d(-1.25,1.25,dx), make_grid1d(-1.25,1.25,dx), make_grid1d(-1.0,1.0,dx)]
g=band_from_cp(x1d, lambda p: m.closest_point(p,k=24,exact=False), p=3, stenrad=1, chunk=60000)
xg=jnp.asarray(g.xg)
t0=time.time()
cp_s = cp_level_set(siren_apply, params, xg, projection_steps=8,
                    newton_iters=25, damping=1e-10, n_backtrack=6, max_step=4*dx)
jax.block_until_ready(cp_s)
print("siren closest points for %d band pts in %.0fs"%(g.n,time.time()-t0), flush=True)
fmax,sinmax = level_set_residual_norm(siren_apply, params, xg, cp_s)
cp_m = m.closest_point(g.xg, k=24, exact=True)
err = np.linalg.norm(np.asarray(cp_s)-cp_m, axis=1)
print("newton residuals |f|=%.2e sin=%.2e"%(fmax,sinmax), flush=True)
fv, sn = level_set_residuals(siren_apply, params, xg, cp_s)
conv = (np.asarray(fv)<=1e-8)&(np.asarray(sn)<=1e-6)
print("closest point newton: %.2f%% of band points failed to converge"%(100*(~conv).mean()), flush=True)
print("geometric error |cp_siren - cp_mesh| on converged points:", flush=True)
print("   median %.4f  p95 %.4f  p99 %.4f  max %.4f   (dx=%.2f)"%(
    np.median(err[conv]), np.percentile(err[conv],95), np.percentile(err[conv],99), err[conv].max(), dx), flush=True)
print("The max matters more than the median here: a band half-width is %.3f, so a"%(4.124*dx), flush=True)
print("tail beyond that puts the learned surface outside a band built for the mesh.", flush=True)

