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




# %%
X = np.linspace(0, 1, 2**5)[:, None]
kernel = gpflow.kernels.SquaredExponential(lengthscales=.1)
Y = gpflow.models.GPR((np.c_[-10.], np.c_[0.]), kernel, noise_variance=NOISE_VAR).predict_f_samples(X)

full_gpr = gpflow.models.GPR((X, Y), kernel, noise_variance=NOISE_VAR)

fig, ax = plt.subplots()
ax.plot(X, Y, "kx")
plot_model(full_gpr, ax=ax, plot_data=False)
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
plot_model(full_gpr, ax, plot_data=False)
plt.plot(X, Y, "kx")

# %%

def get_x_test(step: int) -> np.ndarray:
    """
    returns:

    step 1: [1/2],
    step 2: [1/4, 2/4, 3/4],
    step 3: [1/8, 2/8, ..., 7/8],
    ...

    importantly, the middle element always equals 0.5.
    """
    assert step > 0
    xx = np.arange(1, 2**step) / 2**step
    # assert xx[len(xx) // 2] == 0.5
    return xx


mus = []
sigmas = []

for i in range(1, 12):
    print("=" * 30, i)
    xx = get_x_test(i)
    m, v = m_agg.predict_f(xx[:, None])
    mus.append(m.numpy().flatten()[len(xx) // 2])
    sigmas.append(v.numpy().flatten()[len(xx) // 2] ** 0.5)
# %%

m, v = full_gpr.predict_f(get_x_test(1)[:, None])
m_full = m.numpy().flatten()[0]
sigma_full = v.numpy().flatten()[0] ** 0.5
# %%

def kl_univariate_gaussians(mu_1, sigma_1, mu_2, sigma_2):
    """
    KL(p || q), where p = N(mu_1, sigma_1^2), and q = N(mu_2, sigma_2^2)
    """
    logs =  np.log(sigma_2) - np.log(sigma_1)
    a = (sigma_1**2 + (mu_1 - mu_2) **2) / (2. * sigma_2**2)
    return logs + a - 0.5

kls = [
    kl_univariate_gaussians(m, s, m_full, sigma_full) for m, s in zip(mus, sigmas)
]

colors = cm.viridis(np.linspace(0, 1, len(kls)))

plt.scatter(np.arange(len(kls)), kls, color=colors)
plt.ylabel("KL divergence")
plt.xlabel("Size X*")
plt.xticks(range(11), [fr"$2^{{{step}}}$" for step in range(11)])
plt.yscale("log")
plt.savefig('acc_vs_test_size__kl_div.png', facecolor="white", transparent=False)

# %%
theta_1 = np.array(mus) / np.array(sigmas)**2.
theta_2 = 1. / (2 * np.array(sigmas)**2.)

import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0, 1, len(theta_1)))

plt.scatter(theta_1, theta_2, color=colors)
plt.plot(m_full / sigma_full**2, 1. / (2 * sigma_full**2), "kx", ms=10, label="GPR prediction at 0.5")
plt.loglog()
plt.legend(loc="upper left")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$-\theta_2$")
plt.savefig('acc_vs_test_size_parameter_space.png', facecolor="white", transparent=False)

# %%
