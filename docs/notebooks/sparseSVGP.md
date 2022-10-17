---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: 'Python 3.9.13 (''.venv'': poetry)'
    language: python
    name: python3
---

# SVGP with efficient parametrisation of the variational distribution

This notebook illustrates how to use the equivalent observation framework to get an efficient parametrisation of an GP classification models.

First, let's load some required packages


TODO: 
 * add routines for initialising the sparseSVGP (e.g. from prior, from submodel, from ensemble)
 * Experiment: 
    - find dataset
    - proper timing (are they faster to train?)
    - ablation study: with and without warm start for all methods

```python
import numpy as np
import gpflow
import guepard
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

from gpflow.utilities import print_summary

# The lines below are specific to the notebook format
%matplotlib inline
plt.rcParams["figure.figsize"] = (12, 6)
from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: 150em; }</style>"));

M = 50 # number of inducing points
P = 5  # number of models in ensemble  
```

We now load the banana dataset: This is a binary classification problem with two classes. 

```python
data = sio.loadmat('../../data/banana.mat')
Y = data['banana_Y']
X = data['banana_X']
N = X.shape[0]

x1_lim = [-3.5, 3.5]
x2_lim = [-3.5, 3.5]

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], 'o', color="C0", ms=3, alpha=0.3, label="$y=0$")
ax.plot(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], 'o', color="C1", ms=3, alpha=0.3, label="$y=1$")

ax.set_xlim(x1_lim)
ax.set_ylim(x2_lim)

ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.tight_layout()
```

We then split the data in subsets

```python
Z, label_x = scipy.cluster.vq.kmeans2(X, M, minit='points')
_, label_z = scipy.cluster.vq.kmeans2(Z, P, minit='points') 
Z_init = Z.copy()

masks_x = np.zeros((N, P), dtype="bool")
masks_z = np.zeros((M, P), dtype="bool")
for i in range(P):
    masks_x[:, i] = label_z[label_x] == i
    masks_z[:, i] = label_z == i

fig, ax = plt.subplots(figsize=(6, 6))
for i in range(P):
    ax.plot(X[masks_x[:, i], 0], X[masks_x[:, i], 1], marker="o", linewidth=0, color=f"C{i}", ms=3, alpha=0.3, label="datapoint class")
    ax.plot(Z[label_z == i, 0], Z[label_z == i, 1], marker="x", linewidth=0, color=f"k", ms=5, mew=2,  alpha=1., label="inducing location")
     
```

We build an SVGP model for each data subset, with 15 inducing variables for each of them. Note that all submodels share the same kernel and that the kernel parameters are fixed.

```python
kernel = gpflow.kernels.Matern32(variance=50., lengthscales=[3., 3.])
gpflow.set_trainable(kernel, False)
lik = gpflow.likelihoods.Bernoulli()
gpflow.set_trainable(lik, False)
mean_function = gpflow.mean_functions.Zero()
```

```python
M = []
for mask_x, mask_z in zip(masks_x.T, masks_z.T):
    X_ = X[mask_x, :]
    Y_ = Y[mask_x, :]
    Z_ = Z[mask_z, :]

    m = gpflow.models.SVGP(inducing_variable=Z_, likelihood=lik, kernel=kernel, mean_function=mean_function, whiten=False)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss_closure((X_, Y_)), m.trainable_variables);
    M += [m]
    
```

We can now use the equivalent observation framework to merge these four submodels

```python
Zs, q_mus, q_sqrts = guepard.utilities.init_ssvgp_with_ensemble(M) 
m_ssvgp = guepard.SparseSVGP(kernel, lik, Zs, q_mus, q_sqrts, whiten=False)
m_ssvgp
```

```python
x1_grid = np.linspace(*x1_lim, 50)
x2_grid = np.linspace(*x2_lim, 50)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid) 
Xtest = np.hstack([X1_grid.reshape(-1, 1), X2_grid.reshape(-1, 1)])

Ytest = m_ssvgp.predict_y(Xtest)[0]
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
cs = ax.contour(X1_grid, X2_grid, np.reshape(Ytest, (50, 50)), linewidths=1, colors=["C0", "C0", "grey","C1", "C1"],
                        levels=[0.05, 0.25, 0.5, 0.75, 0.95], zorder=0)
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')

plt.plot(m_ssvgp.inducing_variable.Z[:, 0], m_ssvgp.inducing_variable.Z[:, 1], "ko", ms=2., alpha=.4)

ax.set_xlim(x1_lim)
ax.set_ylim(x2_lim)
ax.set_xticks(np.arange(-3, 4))
ax.set_yticks(np.arange(-3, 4))
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([]);
```

Compare with an SVGP initialised with the ensemble predictions at Z (they should have the same prediction)

```python
m_ens = guepard.EquivalentObsEnsemble(M)
Z = m_ssvgp.inducing_variable.Z

q_m, q_v = m_ens.predict_f(Z, full_cov=True)
q_s = np.linalg.cholesky(q_v)
m_svgp = gpflow.models.SVGP(inducing_variable=Z, likelihood=lik, kernel=kernel, mean_function=mean_function, q_mu=q_m, q_sqrt=q_s, whiten=False)
   
Ytest_ = m_svgp.predict_y(Xtest)[0]
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
cs = ax.contour(X1_grid, X2_grid, np.reshape(Ytest_, (50, 50)), linewidths=1, colors=["C0", "C0", "grey","C1", "C1"],
                        levels=[0.05, 0.25, 0.5, 0.75, 0.95], zorder=0, alpha=.2)

cs = ax.contour(X1_grid, X2_grid, np.reshape(Ytest, (50, 50)), linewidths=1, colors=["C0", "C0", "grey","C1", "C1"],
                        levels=[0.05, 0.25, 0.5, 0.75, 0.95], zorder=0)
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')

[plt.plot(m.inducing_variable.Z[:, 0], m.inducing_variable.Z[:, 1], "ko", ms=2., alpha=.4) for m in m_ens.models]
ax.set_xlim(x1_lim)
ax.set_ylim(x2_lim)
ax.set_xticks(np.arange(-3, 4))
ax.set_yticks(np.arange(-3, 4))
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([]);
```

```python
q_ms, q_vs = m_ssvgp.predict_f(Xtest, full_cov=False)
q_mo, q_vo = m_ens.predict_f(Xtest, full_cov=False)

print(np.max(np.abs(q_ms - q_mo)))
print(np.max(np.abs(q_vs - q_vo)))
```

```python
loss = m_ssvgp.training_loss_closure((X, Y), compile=True)
print(loss())

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(loss, m_ssvgp.trainable_variables);

print(loss())
```

```python
Ytest = m_ssvgp.predict_y(Xtest)[0]
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
cs = ax.contour(X1_grid, X2_grid, np.reshape(Ytest, (50, 50)), linewidths=1, colors=["C0", "C0", "grey","C1", "C1"],
                        levels=[0.05, 0.25, 0.5, 0.75, 0.95], zorder=0)
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')

plt.plot(m_ssvgp.inducing_variable.Z[:, 0], m_ssvgp.inducing_variable.Z[:, 1], "ko", ms=2., alpha=.4)

ax.set_xlim(x1_lim)
ax.set_ylim(x2_lim)
ax.set_xticks(np.arange(-3, 4))
ax.set_yticks(np.arange(-3, 4))
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([]);
```

```python
q_mu, q_sigma = m_ssvgp.predict_f(m_ssvgp.inducing_variable.Z, full_cov=True)
q_sqrt = np.linalg.cholesky(q_sigma)

m_fsvgp = gpflow.models.SVGP(
            m_ssvgp.kernel,
            m_ssvgp.likelihood,
            inducing_variable=m_ssvgp.inducing_variable.Z,
            mean_function=m_ssvgp.mean_function,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=False,
        )

closure = m_fsvgp.training_loss_closure((X, Y))
print(closure())

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(closure, m_fsvgp.trainable_variables)
print(closure())

Yfsvgp = m_fsvgp.predict_y(Xtest)[0]

```

```python
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], 'o', color="C0", ms=3, alpha=0.05, label="$y=0$")
ax.plot(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], 'o', color="C1", ms=3, alpha=0.05, label="$y=1$")

cs = ax.contour(X1_grid, X2_grid, np.reshape(Yfsvgp, (50, 50)), linewidths=1, colors=["C0", "C0", "grey","C1", "C1"],
                        levels=[0.05, 0.25, 0.5, 0.75, 0.95], zorder=0, alpha=.5)

cs = ax.contour(X1_grid, X2_grid, np.reshape(Ytest, (50, 50)), linewidths=1, colors=["C0", "C0", "grey","C1", "C1"],
                        levels=[0.05, 0.25, 0.5, 0.75, 0.95], zorder=0)

ax.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')

plt.plot(m_ssvgp.inducing_variable.Z[:, 0], m_ssvgp.inducing_variable.Z[:, 1], "ko", ms=2., alpha=.4)
plt.plot(m_fsvgp.inducing_variable.Z[:, 0], m_fsvgp.inducing_variable.Z[:, 1], "ko", ms=2., alpha=.2)
ax.set_xlabel("$x_1$", fontsize=14)
ax.set_ylabel("$x_2$", fontsize=14)
ax.set_xlim(x1_lim)
ax.set_ylim(x2_lim)
ax.axes.xaxis.set_ticks([-3, 0, 3])
ax.axes.yaxis.set_ticks([-3, 0, 3])

plt.tight_layout()
```

Train full SVGP without warm start

```python
m_ffsvgp = gpflow.models.SVGP(
            m_ssvgp.kernel,
            m_ssvgp.likelihood,
            inducing_variable=Z_init,
            mean_function=m_ssvgp.mean_function,
            whiten=False,
        )

closure = m_ffsvgp.training_loss_closure((X, Y))
print(closure())

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(closure, m_ffsvgp.trainable_variables)
print(closure())

Yffsvgp = m_ffsvgp.predict_y(Xtest)[0]

```

```python
Z = m_ssvgp.inducing_variable.Z
q_m, q_v = m_ffsvgp.predict_f(Z, full_cov=True)
inv_noise = np.linalg.inv(q_v[0])-np.linalg.inv(kernel(Z))
plt.matshow(inv_noise, vmin=-10, vmax=10)

Z = m_ssvgp.inducing_variable.Z
q_ms, q_vs = m_ssvgp.predict_f(Z, full_cov=True)
inv_noises = np.linalg.inv(q_vs[0])-np.linalg.inv(kernel(Z))
plt.matshow(inv_noises, vmin=-10, vmax=10)

```

```python

```
