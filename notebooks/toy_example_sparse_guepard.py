# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow

from guepard.sparse import SparseGuepard, get_svgp_submodels

# %%
FIGURE_PATH = Path(os.getcwd()) / "plots"
FIGURE_PATH.mkdir(exist_ok=True)
SAVE = True
DEBUG = False


NOISE_VAR = 0.01
# %%

def get_data():

    def f(x):
        return np.sin(10 * x[:, :1]) + 3. * x[:, :1]

    X = np.linspace(0, 1, 101)[:, None]
    Y = f(X) + np.sqrt(NOISE_VAR) * np.random.normal(size=X.shape)
    return X, Y

X, Y = get_data()
# %%
num_split = 3

x_list = np.array_split(X, num_split)  # list of num_split np.array
y_list = np.array_split(Y, num_split)  

kernel = gpflow.kernels.Matern32()
num_inducing_list = [3] * num_split

# make submodels and aggregate them
data_list = list(zip(x_list, y_list))
models = get_svgp_submodels(
    data_list,
    num_inducing_list,
    kernel,
    noise_variance=NOISE_VAR,
    maxiter=-1  # no separate taining of submodels
) # list of num_split GPR models

sparse_guepard = SparseGuepard(models)
# %%


submodels_training_closure = tf.function(lambda: sparse_guepard.training_loss_submodels(data_list))
print(submodels_training_closure())
gpflow.optimizers.scipy.Scipy().minimize(
    submodels_training_closure,
    sparse_guepard.trainable_variables,
    options={"disp": DEBUG, "maxiter": 100},
)

# %%
# define plotting helper functions

def plot_mean_var(x, mean, var, ax, color='C0'):
        ax.plot(x, mean, color, lw=2)
        ax.fill_between(
            x[:, 0],
            mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
            mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
            color=color,
            alpha=0.2,
        )


def plot_model(model: gpflow.models.SVGP, data, x=np.linspace(0, 1, 101)[:, None], plot_data=True, color='C0', ax = None):
    if ax is None:
        fig, ax = plt.subplots()

    if plot_data:
        X, Y = data
        ax.plot(X, Y, "kx", mew=1.)
    
    q_mu = model.q_mu.numpy().flatten()
    q_sqrt_diag = np.diag(model.q_sqrt.numpy()[0]).flatten()
    z_loc = model.inducing_variable.Z.numpy().flatten()
    
    ax.errorbar(x=z_loc, y=q_mu, yerr=q_sqrt_diag, color=color, fmt='o')
    
    mean, var = model.predict_f(x)
    plot_mean_var(x, mean, var, ax, color)

    return ax
    
# plot predictions
models = sparse_guepard.models
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
x_plot = np.linspace(-1.25, 1.25, 101)[:, None]
[plot_model(model, data, x=x_plot, ax=axes[i]) for i, (model, data) in enumerate(zip(models, data_list))];
[axes[i].plot(X, Y, 'kx', mew=1., alpha=.1) for i, _ in enumerate(models)];
if SAVE:
    plt.tight_layout()
    plt.savefig(str( FIGURE_PATH / "sparse_submodels.pdf"))
# %%

print("ELBO at INIT:", sparse_guepard.elbo((X, Y)))
fig, ax = plt.subplots()
plot_model(sparse_guepard, (X, Y), x=x_plot, ax=ax)
ax.set_title("ELBO: {:.2f}".format(sparse_guepard.elbo((X, Y))))
if SAVE:
    plt.tight_layout()
    plt.savefig(str( FIGURE_PATH / "sparse_guepard_at_init.pdf"))

# %%
gpflow.optimizers.scipy.Scipy().minimize(
    sparse_guepard.training_loss_closure((X, Y)),
    sparse_guepard.trainable_variables,
    options={"disp": DEBUG, "maxiter": 100},
)
# %%

print("ELBO post training:", sparse_guepard.elbo((X, Y)))
fig, ax = plt.subplots()
plot_model(sparse_guepard, (X, Y), x=x_plot, ax=ax)
ax.set_title("ELBO: {:.2f}".format(sparse_guepard.elbo((X, Y))))

if SAVE:
    plt.tight_layout()
    plt.savefig(str( FIGURE_PATH / "sparse_guepard_post_training.pdf"))
# %%

fully_parameterized_svgp = sparse_guepard.get_fully_parameterized_svgp()
print("FULL @ INIT", fully_parameterized_svgp.elbo((X, Y)))
# %%

gpflow.optimizers.scipy.Scipy().minimize(
    fully_parameterized_svgp.training_loss_closure((X, Y)),
    fully_parameterized_svgp.trainable_variables,
    options={"disp": DEBUG, "maxiter": 100},
)
# %%

print("FULL @ INIT", fully_parameterized_svgp.elbo((X, Y)))
ax = plot_model(fully_parameterized_svgp, (X, Y), x=x_plot)
ax.set_title("ELBO: {:.2f}".format(fully_parameterized_svgp.elbo((X, Y))))
if SAVE:
    plt.tight_layout()
    plt.savefig(str( FIGURE_PATH / "sparse_guepard_fully_parameterized.pdf"))
# %%
