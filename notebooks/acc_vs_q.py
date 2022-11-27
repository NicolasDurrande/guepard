#%%
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gpflow

import guepard
from guepard.utilities import get_gpr_submodels


def get_test_locations(inputs: np.ndarray, size: int, strided=False) -> List[np.ndarray]:
    assert len(inputs) % size == 0
    num_groups = len(inputs) // size
    if not strided:
        return [
            inputs[size*i:(i+1)*size] for i in range(num_groups)
        ]
    else:
        return [
            inputs[i::num_groups] for i in range(num_groups)
        ]


X = np.linspace(0, 1, 32)[:, None]
fig, axes = plt.subplots(2, 1, sharex=True)
for e in range(6):
    q = int(2**e)
    test_locations_strided = get_test_locations(X, q, strided=True)
    test_locations = get_test_locations(X, q, strided=False)
    for test_set_strided, test_set in zip(test_locations_strided, test_locations):
        axes[0].plot(test_set_strided, np.ones_like(test_set) * e, 'o')
        axes[1].plot(test_set, np.ones_like(test_set) * e, 'o')

axes[0].set_ylabel('log(q)')
axes[0].set_title('strided')
axes[1].set_ylabel('log(q)')
axes[1].set_title('non strided')
plt.savefig('acc_vs_q__test_locations.png')
# %%

def get_data(num_data, kernel):
    X = np.linspace(0, 1, num_data)[:, None]
    Y = gpflow.models.GPR((np.c_[-10.], np.c_[0.]), kernel, noise_variance=NOISE_VAR).predict_f_samples(X)
    return X, Y

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


STRIDED = False
NOISE_VAR = 1e-3
NUM_DATA = 32
data = []

KERNEL = gpflow.kernels.SquaredExponential(lengthscales=.25)

for rep in range(5):
    print("*****", rep)
    X, Y = get_data(NUM_DATA, KERNEL)
    full_gpr = gpflow.models.GPR((X, Y), KERNEL, noise_variance=NOISE_VAR)

    for exp_p in range(1, 5):
        P = int(2 ** exp_p)
        agg_gpr = get_aggregate_model(X, Y, P, KERNEL)
        print("P", P)
        for exp_q in range(6):
            Q = int(2**exp_q)
            print("Q", Q)
            test_sets = get_test_locations(X, Q, strided=STRIDED)
            means, variances = [], []
            for test_set in test_sets:
                m, v = agg_gpr.predict_f(test_set)
                means.append(m)
                variances.append(v)
            means = np.concatenate(means, axis=0)
            variances = np.concatenate(variances, axis=0)

            x_test = np.concatenate(test_sets, axis=0)
            mean_full, variance_full = full_gpr.predict_f(x_test)
            mean_full, variance_full = np.array(mean_full), np.array(variance_full)

            plt.figure()
            plt.errorbar(x_test.flatten(), mean_full.flatten(), variance_full.flatten() ** .5, label="GPR")
            plt.errorbar(x_test.flatten() + .002, means.flatten(), variances.flatten() ** .5, label="guepard")
            plt.title(f"P = {P}, Q = {Q}")
            plt.savefig(f"acc_vs_q_plots/predictions_{rep}_p_{P}_q_{Q}.png")
            plt.close()

            kl = sum([
                kl_univariate_gaussians(m, v**.5, m2, v2**.5) for m, v, m2, v2 in zip(
                    means.flatten(), variances.flatten(), mean_full.flatten(), variance_full.flatten()
                )
            ])
            data.append({
                'P': P,
                'Q': Q,
                'kl': kl,
                'rep': rep
            })


# %%
df = pd.DataFrame(data)

# %%
# df2 = df.groupby(['P', 'Q']).agg({'kl': ['mean', 'std']}).reset_index()
# df2.columns = df2.columns.map('_'.join).str.strip('_')

fig, ax = plt.subplots()
pd.plotting.boxplot(df, column='kl', by=['Q', 'P'], rot=90, ax=ax)
fig.suptitle('')
ax.set_yscale('log')
ax.set_ylabel('KL')
if STRIDED:
    ax.set_title('Cummulative KL at 32 test locations. Strided.')
else:
    ax.set_title('Cummulative KL at 32 test locations. Non strided.')

if STRIDED:
    plt.savefig('acc_vs_q__p_strided.png')
else:
    plt.savefig('acc_vs_q__p_non_strided.png')

# %%
