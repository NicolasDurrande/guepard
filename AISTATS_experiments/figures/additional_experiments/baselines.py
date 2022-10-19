# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.experimental.check_shapes import disable_check_shapes

import guepard
from guepard.baselines import Ensemble, EnsembleMethods, WeightingMethods
from guepard.utilities import get_gpr_submodels

# %%


def plot_mean_conf(x, mean, var, ax, color='C0'):
    ax.plot(x, mean, color, lw=2)
    ax.fill_between(
        x[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color=color,
        alpha=0.2,
    )

def plot_model(m, ax, x=np.linspace(-.1, 1.1, 101)[:, None], plot_data=True, color='C0'):
    if plot_data:
        X, Y = m.data
        ax.plot(X, Y, "kx", mew=1.)
    
    mean, var = m.predict_f(x)
    plot_mean_conf(x, mean, var, ax, color)


def get_data(num_data):
    X = np.linspace(0, 1, num_data)[:, None]
    F = gpflow.models.GPR((np.c_[-10.], np.c_[0.]), KERNEL, noise_variance=NOISE_VAR).predict_f_samples(X)
    Y = F + np.random.randn(*F.numpy().shape) * NOISE_VAR**.5
    return X, Y


# %%
NOISE_VAR = 1e-3
NUM_DATA = 300
KERNEL = gpflow.kernels.SquaredExponential(lengthscales=.1)
(X, Y) = get_data(NUM_DATA)

# %%

plt.plot(X, Y, "kx")

# %%
num_split = 100
x_list = np.array_split(X, num_split)  # list of num_split np.array
y_list = np.array_split(Y, num_split)  

# make submodels and aggregate them
datasets = list(zip(x_list, y_list))
# %%

submodels = get_gpr_submodels(datasets, KERNEL, mean_function=None, noise_variance=NOISE_VAR) # list of num_split GPR models
# fig, axes = plt.subplots(10, 10, figsize=(16, 16))
# axes = axes.flatten()
# [plot_model(m, ax) for ax, m in zip(axes, submodels)];
# [ax.plot(X, Y, 'kx', mew=1., alpha=.1) for ax, _ in zip(axes, submodels)];
# %%

full = gpflow.models.GPR((X, Y), KERNEL, noise_variance=NOISE_VAR)

# %%
nr, nc = len(EnsembleMethods), len(WeightingMethods)
fig, axes = plt.subplots(nr, nc, figsize=(nc * 4, nr * 4), squeeze=False, sharex=True, sharey=True)

# with disable_check_shapes():
for i, method in enumerate(EnsembleMethods):
    axes_row = axes[i]
    axes_row[0].set_ylabel(method.value)
    for j, weighting in enumerate(WeightingMethods):
        ax = axes_row[j]
        ax.set_ylim(-3.0, 3.0)
        if i == 0: ax.set_title(weighting.value)
        ensemble = Ensemble(submodels, method, weighting)
        xx = np.linspace(-.5, 1.5, 101)[:, None]
        plot_model(ensemble, ax, xx, plot_data=False)
        plot_model(full, ax, xx, color="C1")
        ax.plot(X, Y, "kx")

plt.savefig("ensemble.png", transparent=False, facecolor="white", dpi=200)
# %%
# %%
