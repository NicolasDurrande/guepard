# %%
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import gpflow

import guepard
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


def plot_model(m, ax, x=np.linspace(0, 1, 101)[:, None], plot_data=True, color='C0'):
    if plot_data:
        X, Y = m.data
        ax.plot(X, Y, "kx", mew=1.)
    
    mean, var = m.predict_f(x)
    plot_mean_conf(x, mean, var, ax, color)

# %%

def get_data(num_data, kernel):
    X = np.linspace(0, 1, num_data)[:, None]
    Y = gpflow.models.GPR((np.c_[-10.], np.c_[0.]), kernel, noise_variance=NOISE_VAR).predict_f_samples(X)
    return X, Y


def get_subset_of_data(A: np.ndarray, step: int) -> np.ndarray:
    """
    Returns a subarray of `A`. The elements in the subarray depend on `step`.
    For `step` equal to 0, the subarray consists of a single element which will be
    the the middle element of `A`. For `step` >= 1, the subarray contains the (2**step + 1)
    middle elements.
    """
    middle_index = len(A)//2
    if step == 0:
        return A[[middle_index]]
    else:
        pad = int(2 ** (step - 1))  # 1, 2, 4, 8, ...
        return A[middle_index - pad: middle_index + pad + 1]


def get_aggregate_model(X, Y, num_splits, kernel):
    x_list = np.array_split(X, num_splits)  # list of num_split np.array
    y_list = np.array_split(Y, num_splits)  
    datasets = list(zip(x_list, y_list))

    submodels = get_gpr_submodels(datasets, kernel, mean_function=None, noise_variance=NOISE_VAR) # list of num_split GPR models
    m_agg = guepard.EquivalentObsEnsemble(submodels)
    return m_agg


def kl_univariate_gaussians(mu_1, sigma_1, mu_2, sigma_2):
    """
    KL(p || q), where p = N(mu_1, sigma_1^2), and q = N(mu_2, sigma_2^2)
    """
    logs =  np.log(sigma_2) - np.log(sigma_1)
    a = (sigma_1**2 + (mu_1 - mu_2) **2) / (2. * sigma_2**2)
    return logs + a - 0.5


def compare_full_vs_agg(X, full, agg):
    mus = []
    sigmas = []
    len_x_subset = []

    for i in range(LN_NUM_DATA + 1):
        xx = get_subset_of_data(X, i)
        m, v = agg.predict_f(xx)
        mus.append(m.numpy().flatten()[len(xx) // 2])
        sigmas.append(v.numpy().flatten()[len(xx) // 2] ** 0.5)
        len_x_subset.append(len(xx))

    m, v = full.predict_f(get_subset_of_data(X, 0))
    m_full = m.numpy().flatten()[0]
    sigma_full = v.numpy().flatten()[0] ** 0.5

    kls = [
        kl_univariate_gaussians(m, s, m_full, sigma_full) for m, s in zip(mus, sigmas)
    ]

    return len_x_subset, kls

# %%
NOISE_VAR = 1e-1
LN_NUM_DATA = 5  # num_datapoints = 2 ** LN_NUM_DATA + 1
REPS_ITER = range(5)
NUM_SPLITS_ITER = range(2, 6)
KERNEL = gpflow.kernels.SquaredExponential(lengthscales=.1)

results = []

import itertools as it

import pandas as pd

for rep, num_splits in it.product(REPS_ITER, NUM_SPLITS_ITER):
    print(rep, num_splits)
    X, Y = get_data(2 ** LN_NUM_DATA + 1, KERNEL)
    full_gpr = gpflow.models.GPR((X, Y), KERNEL, noise_variance=NOISE_VAR)
    agg_gpr = get_aggregate_model(X, Y, num_splits, KERNEL)
    size, kl = compare_full_vs_agg(X, full_gpr, agg_gpr)
    results.extend({"rep": rep, "num_splits": num_splits, "kl": k, "size": s} for s,k in zip(size, kl))

#%%
df = pd.DataFrame(results)
df

def err(x):
    err = 1.96 * x.std() / np.sqrt(len(x))
    return err

# %%
df = df.groupby(["num_splits", "size"]).agg(
    {
        "rep": "count",
        "kl": ["mean", "std", err]
    }
)
df
# %%

for i, num_splits in enumerate(NUM_SPLITS_ITER):
    r = df.xs(num_splits, axis=0, level="num_splits")
    x, m, e = r.index.values, r["kl", "mean"].values, r["kl", "err"]
    plt.plot(x, m, f"C{i}x-", label=f"P = {num_splits}")
    plt.yscale('log')
    plt.fill_between(x, np.maximum(m - e, m), m + e, color=f"C{i}", alpha=.2,)

plt.legend(loc="lower left")
plt.ylabel("KL divergence")
plt.xlabel("Size X*")
plt.savefig("pred_acc__vs__test_size.png", transparent=False, facecolor="white")
# %%
