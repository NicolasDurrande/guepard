---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: 'Python 3.9.13 (''.venv'': poetry)'
    language: python
    name: python3
---

# Merging SVGP sub-models using PAPL

This notebook illustrates how to use PAPL (Posterior Aggregation using Pseudo-Likelihood) to train an ensemble of GP classification models and to make predictions with it.

First, let's load some required packages

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
```

We now load the banana dataset: This is a binary classification problem with two classes. 

```python
data = sio.loadmat('../../data/banana.mat')
Y = data['banana_Y']
X = data['banana_X']

#x1_lim = [np.min(X[:, 0])-0.1, np.max(X[:, 0])+0.1]
#x2_lim = [np.min(X[:, 1])-0.1, np.max(X[:, 1])+0.1]
x1_lim = [-3.5, 3.5]
x2_lim = [-3.5, 3.5]

fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(0, color='k', linestyle="dashed", alpha=0.5, linewidth=.5)
ax.axvline(0, color='k', linestyle="dashed", alpha=0.5, linewidth=.5)
ax.plot(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], 'o', color="C0", ms=3, alpha=0.3, label="$y=0$")
ax.plot(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], 'o', color="C1", ms=3, alpha=0.3, label="$y=1$")

ax.set_xlim(x1_lim)
ax.set_ylim(x2_lim)

ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.tight_layout()
#plt.savefig("plots/banana_data.pdf")
```

We then split the dataset in four, with one subset per quadrant of the input space:

```python
# Compute masks for the four quadrants
maskNE = np.logical_and(X[:, 0] < 0, X[:, 1] >= 0) 
maskNW = np.logical_and(X[:, 0] >= 0, X[:, 1] >= 0) 
maskSE = np.logical_and(X[:, 0] < 0, X[:, 1] < 0) 
maskSW = np.logical_and(X[:, 0] >= 0, X[:, 1] < 0) 
masks = [maskNE, maskNW, maskSE, maskSW]

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
for i in range(2):
    for j in range(2):
        k = 2 * i + j
        
        axes[i,j].axhline(0, color='k', linestyle="dashed", alpha=0.5, linewidth=.5)
        axes[i,j].axvline(0, color='k', linestyle="dashed", alpha=0.5, linewidth=.5)

        X_ = X[masks[k], :]
        Y_ = Y[masks[k], :]
        axes[i, j].plot(X_[Y_[:, 0] == 0, 0], X_[Y_[:, 0] == 0, 1], 'o', color="C0", ms=3, alpha=0.13, label="$y=0$")
        axes[i, j].plot(X_[Y_[:, 0] == 1, 0], X_[Y_[:, 0] == 1, 1], 'o', color="C1", ms=3, alpha=0.13, label="$y=1$")

        axes[i, j].plot(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], 'o', color="C0", ms=3, alpha=0.02, label="$y=0$")
        axes[i, j].plot(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], 'o', color="C1", ms=3, alpha=0.02, label="$y=1$")

        axes[i, j].set_xticks(np.arange(-3, 4))
        axes[i, j].set_yticks(np.arange(-3, 4))
        axes[i, j].axes.xaxis.set_ticklabels([])
        axes[i, j].axes.yaxis.set_ticklabels([])

plt.tight_layout()
#plt.savefig("plots/banana_subdata.pdf")
```

We build an SVGP model for each data subset, with 15 inducing variables for each of them. Note that all submodels share the same kernel and that the kernel parameters are fixed.

```python
kernel = gpflow.kernels.Matern32(variance=50., lengthscales=[3., 3.])
gpflow.set_trainable(kernel, False)
lik = gpflow.likelihoods.Bernoulli()
mean_function = gpflow.mean_functions.Zero()

M = []
for mask in masks:
    X_ = X[mask, :]
    Y_ = Y[mask, :]
    Z = scipy.cluster.vq.kmeans(X_, 15)[0] # the locations of the inducing variables are initialised with k-means

    m = gpflow.models.SVGP(inducing_variable=Z, likelihood=lik, kernel=kernel, mean_function=mean_function)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss_closure((X_, Y_)), m.trainable_variables);
    M += [m]
    
```

Let's plot the submodels predictions in the data space.

```python
x1_grid = np.linspace(*x1_lim, 50)
x2_grid = np.linspace(*x2_lim, 50)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid) 
Xtest = np.hstack([X1_grid.reshape(-1, 1), X2_grid.reshape(-1, 1)])

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
for i in range(2):
    for j in range(2):
        k = 2 * i + j
        
        axes[i,j].axhline(0, color='k', linestyle="dashed", alpha=0.2, linewidth=.5)
        axes[i,j].axvline(0, color='k', linestyle="dashed", alpha=0.2, linewidth=.5)

        X_ = X[masks[k], :]
        Y_ = Y[masks[k], :]
        axes[i, j].plot(X_[Y_[:, 0] == 0, 0], X_[Y_[:, 0] == 0, 1], 'o', color="C0", ms=3, alpha=0.05, label="$y=0$")
        axes[i, j].plot(X_[Y_[:, 0] == 1, 0], X_[Y_[:, 0] == 1, 1], 'o', color="C1", ms=3, alpha=0.05, label="$y=1$")
        
        Z = M[k].inducing_variable.Z
        axes[i, j].plot(Z[:, 0], Z[:, 1], "ko", ms=2., alpha=.4)

        Ytest, _ = M[k].predict_y(Xtest)
        cs = axes[i, j].contour(X1_grid, X2_grid, np.reshape(Ytest, (50, 50)), linewidths=1, colors=["C0", "C0", "grey","C1", "C1"],
                        levels=[0.05, 0.25, 0.5, 0.75, 0.95], zorder=0)
        axes[i, j].clabel(cs, inline=1, fontsize=10, fmt='%1.2f')


        axes[i, j].set_xticks(np.arange(-3, 4))
        axes[i, j].set_yticks(np.arange(-3, 4))
        axes[i, j].axes.xaxis.set_ticklabels([])
        axes[i, j].axes.yaxis.set_ticklabels([])

plt.tight_layout()
#plt.savefig("plots/banana_submodels.pdf")
```

We can also plot the submodel predictions in the latent space

```python
def plot_latent(Mtest, Vtest_full, X1_grid, X2_grid, ax, num_sample=100):
    Vtest_full = Vtest_full.numpy()[0, :, :]
    Vtest = np.diag(Vtest_full).reshape((50, 50))

    # plot samples
    if num_sample != 0:
        Fp = np.random.multivariate_normal(Mtest.numpy().flatten(), Vtest_full, num_sample).T
        for k in range(num_sample):
            ax.contour(X1_grid, X2_grid, np.reshape(Fp[:, k], (50, 50)),
                       levels=np.arange(-4, 4)*2, alpha=0.1,
                       linewidths= .5)

    # plot mean
    cs = ax.contour(X1_grid, X2_grid, np.reshape(Mtest, (50, 50)),
                           levels=np.arange(-4, 4)*2,
                           linewidths= 1)
    ax.clabel(cs, inline=1, fontsize=10, fmt='%1.f')

    # add transparency where predictions are uncertain
    alphas = np.minimum(1., Vtest/np.max(Vtest))
    ax.imshow(np.ones((50, 50)), cmap="binary", alpha=alphas, zorder=2, extent=(*x1_lim, *x2_lim), interpolation="bilinear")

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
for i in range(2):
    for j in range(2):
        k = 2 * i + j
        
        Z = M[k].inducing_variable.Z
        axes[i, j].plot(Z[:, 0], Z[:, 1], "ko", ms=2., alpha=.4)
        
        Ftest, Vtest_full = M[k].predict_f(Xtest, full_cov=True)
        plot_latent(Ftest, Vtest_full, X1_grid, X2_grid, axes[i,j], num_sample=0) # The figure in paper has num_sample=50

        axes[i,j].axhline(0, color='k', linestyle="dashed", alpha=0.2, linewidth=.5)
        axes[i,j].axvline(0, color='k', linestyle="dashed", alpha=0.2, linewidth=.5)

        axes[i, j].set_xticks(np.arange(-3, 4))
        axes[i, j].set_yticks(np.arange(-3, 4))
        axes[i, j].axes.xaxis.set_ticklabels([])
        axes[i, j].axes.yaxis.set_ticklabels([])

plt.tight_layout()
#plt.savefig("plots/banana_sublatents.pdf")
```

We can now use the equivalent observation framework to merge these four submodels

```python
m_agg = guepard.SparseGuepard(M)
m_agg.predict_f = m_agg.predict_foo
Ftest, Vtest_full = m_agg.predict_f(Xtest, full_cov=True)
Ytest = m_agg.predict_y(Xtest)[0]

```

```python
fig, ax = plt.subplots(figsize=(6, 6))

plt.plot(m_agg.inducing_variable.Z[:, 0], m_agg.inducing_variable.Z[:, 1], "ko", ms=2., alpha=.4)

plot_latent(Ftest, Vtest_full, X1_grid, X2_grid, ax, num_sample=0)  # the figure in the paper uses num_sample=100 

ax.axhline(0, color='k', linestyle="dashed", alpha=0.5, linewidth=.5)
ax.axvline(0, color='k', linestyle="dashed", alpha=0.5, linewidth=.5)

ax.set_xlim(x1_lim)
ax.set_ylim(x2_lim)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.tight_layout()
#plt.savefig("plots/banana_latents.pdf")

```

For comparison we fit an SVGP model with the same kernel, same inducing location Z, but an optimised distribution for the inducing variables.

```python
m_svgp = m_agg.get_fully_parameterized_svgp()
gpflow.set_trainable(m_svgp.inducing_variable, False)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_svgp.training_loss_closure((X, Y)), m_svgp.trainable_variables)

Ysvgp = m_svgp.predict_y(Xtest)[0]
```

```python
fig, ax = plt.subplots(figsize=(6, 6))

ax.axhline(0, color='k', linestyle="dashed", alpha=0.5, linewidth=.5)
ax.axvline(0, color='k', linestyle="dashed", alpha=0.5, linewidth=.5)

ax.plot(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], 'o', color="C0", ms=3, alpha=0.05, label="$y=0$")
ax.plot(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], 'o', color="C1", ms=3, alpha=0.05, label="$y=1$")

cs = ax.contour(X1_grid, X2_grid, np.reshape(Ysvgp, (50, 50)), linewidths=1, colors=["C0", "C0", "grey","C1", "C1"],
                        levels=[0.05, 0.25, 0.5, 0.75, 0.95], zorder=0, alpha=.5)

cs = ax.contour(X1_grid, X2_grid, np.reshape(Ytest, (50, 50)), linewidths=1, colors=["C0", "C0", "grey","C1", "C1"],
                        levels=[0.05, 0.25, 0.5, 0.75, 0.95], zorder=0)

ax.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')

plt.plot(m_agg.inducing_variable.Z[:, 0], m_agg.inducing_variable.Z[:, 1], "ko", ms=2., alpha=.4)
ax.set_xlabel("$x_1$", fontsize=14)
ax.set_ylabel("$x_2$", fontsize=14)
ax.set_xlim(x1_lim)
ax.set_ylim(x2_lim)
ax.axes.xaxis.set_ticks([-3, 0, 3])
ax.axes.yaxis.set_ticks([-3, 0, 3])

plt.tight_layout()
# plt.savefig("plots/banana_models.pdf")
```

we can plot the absolute error

```python
error = (Ytest- Ysvgp).numpy().flatten()
print("max absolute error", np.max(np.abs(error)))


fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], 'o', color="C0", ms=3, alpha=0.05, label="$y=0$")
ax.plot(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], 'o', color="C1", ms=3, alpha=0.05, label="$y=1$")

cs = ax.contour(X1_grid, X2_grid, np.reshape(np.abs(error), (50, 50)), linewidths=1, levels=[0.01, 0.02, 0.03, 0.04, 0.05])
ax.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')

ax.set_xlabel("$x_1$", fontsize=14)
ax.set_ylabel("$x_2$", fontsize=14)
ax.set_xlim(x1_lim)
ax.set_ylim(x2_lim)
```

```python
from scipy.stats import qmc
sampler = qmc.Halton(d=2, scramble=False)
sample = sampler.random(n=2**7)[:, :1]

plt.plot(sample, 0 * sample, 'kx')
plt.xlim((0, 1))
np.max(sample)
sample.shape
```


