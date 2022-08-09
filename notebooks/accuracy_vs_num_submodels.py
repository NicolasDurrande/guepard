# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

import gpflow

import guepard
from guepard.utilities import get_gpr_submodels

# %%
NOISE_VAR = 1e-5
NUM_DATAPOINTS_PER_DATASET = 16

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
# sampler = qmc.Halton(d=2, scramble=False)
# sample = sampler.random(n=2**7)[:, :1]
# plt.plot(sample, np.zeros_like(sample), 'kx')
# sample.shape
# sample[:10]

# %%
X = np.linspace(0, 1, 2**6)[:, None]
kernel = gpflow.kernels.SquaredExponential(lengthscales=.1)
Y = gpflow.models.GPR((np.c_[-10.], np.c_[0.]), kernel, noise_variance=NOISE_VAR).predict_f_samples(X)

plt.plot(X, Y)
#%%
def get_subsets_data(data, num_datapoints_per_dataset):
    X, Y = data
    assert len(X) % num_datapoints_per_dataset == 0
    datasets = []
    num_datasets = len(X) // num_datapoints_per_dataset
    for i in range(num_datasets):
        X_ = X[i * num_datapoints_per_dataset: (i+1) * num_datapoints_per_dataset]
        Y_ = Y[i * num_datapoints_per_dataset: (i+1) * num_datapoints_per_dataset]
        datasets.append((X_, Y_))
    return datasets


datasets = get_subsets_data((X, Y), NUM_DATAPOINTS_PER_DATASET)
print(len(datasets))

for i, data in enumerate(datasets):
    X_, Y_ = data
    plt.plot(X_, Y_, f"C{i%7}x")

# %%

# mean_function = gpflow.mean_functions.Constant(0.5)
submodels = get_gpr_submodels(datasets, kernel, mean_function=None, noise_variance=NOISE_VAR) # list of num_split GPR models

# M is a list of GPR models, let's plot them
fig, axes = plt.subplots(1, len(datasets), figsize=(16, 4))
x = np.linspace(-.5, 1.5, 101)[:, None]
[plot_model(m, ax, x) for ax, m in zip(axes, submodels)];
[ax.plot(X, Y, 'kx', mew=1., alpha=.1) for ax, _ in zip(axes, submodels)];

# %%
m_agg = guepard.EquivalentObsEnsemble(submodels)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_model(m_agg, ax, plot_data=False)
plt.plot(X, Y, "kx")

# %%

Xnew = np.r_[.1, .3][:, None]
# %%
me, Se = submodels[2].predict_f(Xnew, full_cov=True)
Se = Se[0].numpy()
me = me.numpy()
print(me)

