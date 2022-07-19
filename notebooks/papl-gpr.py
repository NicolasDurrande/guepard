# %% [markdown]
# # Merging GP regression sub-models using PAPL
# 
# This notebook illustrates how to use PAPL (Posterior Aggregation using Pseudo-Likelihood) to train an ensemble of Gaussian process models and to make predictions with it.
# 
# First, let's load some required packages

import gpflow
import matplotlib.pyplot as plt

# %%
import numpy as np
from gpflow.utilities import print_summary

import guepard
from guepard.gpr_submodels import get_gpr_submodels

# The lines below are specific to the notebook format
# %matplotlib inline
plt.rcParams["figure.figsize"] = (12, 6)

from IPython.core.display import HTML, display

display(HTML("<style>div.output_scroll { height: 150em; }</style>"))

# %% [markdown]
# We now define a couple of helper functions, and generate a dataset

# %%
noise_var = 0.01


def f(x):
    return np.sin(10 * x[:, :1]) + 3. * x[:, :1]


X = np.linspace(0, 1, 101)[:, None]
#np.random.shuffle(X)
Y = f(X) + np.sqrt(noise_var) * np.random.normal(size=X.shape)

plt.plot(X, Y, 'kx')

# %% [markdown]
# We now split the dataset in three, and build a GPR model for each of them

# %%
num_split = 3

Xl = np.array_split(X, num_split)  # list of num_split np.array
Yl = np.array_split(Y, num_split)  

kernel = gpflow.kernels.Matern32()

# make submodels and aggregate them
M = get_gpr_submodels(zip(Xl, Yl), kernel, noise_variance=noise_var) # list of num_split GPR models

m_agg = guepard.PAPL(M)

# %% [markdown]
# `M` is a list of GPR models, let's plot them

# %%
# define plotting helper functions

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
    
    mean, var = m.predict_f(x)[:2]
    plot_mean_conf(x, mean, var, ax, color)
    
# plot predictions
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
x = np.linspace(0, 2, 101)[:, None]
[plot_model(m, axes[i], x) for i, m in enumerate(M)];
[axes[i].plot(X, Y, 'kx', mew=1., alpha=.1) for i, _ in enumerate(M)];

# %% [markdown]
# We can now aggregate the three sub-models using PAPL

# %%
m_papl = guepard.PAPL(M)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_model(m_papl, ax, plot_data=False)

# %% [markdown]
# Guepard models inherit from GPflow GPmodels, it is thus possible to interact with them like any other GPflow models:

# %%
# print the model parameter summary
gpflow.utilities.print_summary(m_papl)

# Set the value of one parameter
m_papl.kernel.lengthscales.assign(0.3)

# %% [markdown]
# ## PAPL Model training
# 
# Guepard models can be trained like any other GPflow model

# %%
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_papl.training_loss, m_papl.trainable_variables, options=dict(maxiter=100))
print_summary(m_papl)

# %% [markdown]
# ## Comparison with a GPR model based on the full dataset

# %%
# make a GPR model as baseline
m_gpr = gpflow.models.GPR((X, Y), kernel, noise_variance=noise_var)
opt_logs = opt.minimize(m_gpr.training_loss, m_gpr.trainable_variables, options=dict(maxiter=100))
print_summary(m_papl)

# Check "good" match between aggregated model and gpr at training points
mean_papl, var_papl = m_papl.predict_f(X)
mean_gpr, var_gpr = m_gpr.predict_f(X)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_model(m_papl, axes[0], plot_data=False)
plot_model(m_gpr, axes[0], plot_data=False, color='C1')

axes[1].plot(x, mean_papl - mean_gpr, label="error in mean prediction")
axes[1].plot(x, np.sqrt(var_papl) - np.sqrt(var_gpr), label="error in variance prediction")
plt.legend()
plt.tight_layout()

# %% [markdown]
# On this simple example, predictions from PAPL are extremely close to the ground truth despite requiring to store and invert matrices that are 1/3rd of the size of a full model.  

# %% [markdown]
# 


