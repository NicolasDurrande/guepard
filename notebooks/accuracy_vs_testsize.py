# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

import gpflow

import guepard
from guepard.utilities import get_gpr_submodels


#%%
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
NOISE_VAR = 1e-1
NUM_SPLIT = 3
LN_NUM_DATA = 5  # num_datapoints = 2 ** LN_NUM_DATA + 1
# %%

X = np.linspace(0, 1, 2**LN_NUM_DATA + 1)[:, None]
print("num data", len(X))
kernel = gpflow.kernels.SquaredExponential(lengthscales=.1)
Y = gpflow.models.GPR((np.c_[-10.], np.c_[0.]), kernel, noise_variance=NOISE_VAR).predict_f_samples(X)
full_gpr = gpflow.models.GPR((X, Y), kernel, noise_variance=NOISE_VAR)
# %%
fig, ax = plt.subplots()
ax.plot(X, Y, "kx")
plot_model(full_gpr, ax=ax, plot_data=False)

#%%
x_list = np.array_split(X, NUM_SPLIT)  # list of num_split np.array
y_list = np.array_split(Y, NUM_SPLIT)  
datasets = list(zip(x_list, y_list))

for i, data in enumerate(datasets):
    X_, Y_ = data
    print(len(X_), len(Y_))
    plt.plot(X_, Y_, f"C{i%7}x")


# %%
# mean_function = gpflow.mean_functions.Constant(0.5)
submodels = get_gpr_submodels(datasets, kernel, mean_function=None, noise_variance=NOISE_VAR) # list of num_split GPR models

# M is a list of GPR models, let's plot them
fig, axes = plt.subplots(1, len(datasets), figsize=(16, 4))
if len(datasets) == 1:
    axes = [axes]

x = np.linspace(-.5, 1.5, 101)[:, None]
[plot_model(m, ax, x) for ax, m in zip(axes, submodels)];
[ax.plot(X, Y, 'kx', mew=1., alpha=.1) for ax, _ in zip(axes, submodels)];

# %%
m_agg = guepard.EquivalentObsEnsemble(submodels)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_model(m_agg, ax, plot_data=False, color="C0")
plot_model(full_gpr, ax, plot_data=False, color="C1")
ax.plot(X, Y, "kx")

# %%

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


for step in range(LN_NUM_DATA + 1):
    X_subset = get_subset_of_data(X, step)
    assert len(X_subset) == 1 if step == 0 else (2 ** step + 1)
    
# %%


mus = []
sigmas = []
len_x_subset = []

for i in range(LN_NUM_DATA + 1):
    print("=" * 30, i)
    xx = get_subset_of_data(X, i)
    m, v = m_agg.predict_f(xx)
    mus.append(m.numpy().flatten()[len(xx) // 2])
    sigmas.append(v.numpy().flatten()[len(xx) // 2] ** 0.5)
    len_x_subset.append(len(xx))
    print(len_x_subset[-1], mus[-1], sigmas[-1])
# %%

m, v = full_gpr.predict_f(get_subset_of_data(X, 0))
m_full = m.numpy().flatten()[0]
sigma_full = v.numpy().flatten()[0] ** 0.5
print(m_full, sigma_full)
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

plt.scatter(len_x_subset, kls, color=colors)
plt.ylabel("KL divergence")
plt.xlabel("size X*")
# plt.xticks(range(11), [fr"$2^{{{step}}}$" for step in range(11)])
plt.yscale("log")
plt.savefig('acc_vs_test_size__kl_div.png', facecolor="white", transparent=False)

# %%
theta_1 = np.array(mus) / np.array(sigmas)**2.
theta_2 = 1. / (2 * np.array(sigmas)**2.)


colors = cm.viridis(np.linspace(0, 1, len(theta_1)))

plt.scatter(theta_1, theta_2, color=colors)
plt.plot(m_full / sigma_full**2, 1. / (2 * sigma_full**2), "kx", ms=10, label="GPR prediction at 0.5")
plt.loglog()
plt.legend(loc="upper left")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$-\theta_2$")
plt.savefig('acc_vs_test_size_parameter_space.png', facecolor="white", transparent=False)

# %%
