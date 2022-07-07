# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow

from guepard.sparse import SparsePapl, get_svgp_submodels

# %%
noise_var = 0.01


def f(x):
    return np.sin(10 * x[:, :1]) + 3. * x[:, :1]


X = np.linspace(0, 1, 101)[:, None]
#np.random.shuffle(X)
Y = f(X) + np.sqrt(noise_var) * np.random.normal(size=X.shape)

plt.plot(X, Y, 'kx')
# %%
num_split = 3

x_list = np.array_split(X, num_split)  # list of num_split np.array
y_list = np.array_split(Y, num_split)  

kernel = gpflow.kernels.Matern32()
num_inducing_list = [3] * num_split

# make submodels and aggregate them
data_list = list(zip(x_list, y_list))
models = get_svgp_submodels(data_list,num_inducing_list, kernel, noise_variance=noise_var, maxiter=-1) # list of num_split GPR models
papl = SparsePapl(models)
# %%
print(papl.trainable_variables)
# %%

submodels_training_closure = tf.function(lambda: papl.training_loss_submodels(data_list))
print(submodels_training_closure())
gpflow.optimizers.scipy.Scipy().minimize(
    submodels_training_closure,
    papl.trainable_variables,
    options={"disp": True, "maxiter": 100},
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
    print(q_sqrt_diag)
    z_loc = model.inducing_variable.Z.numpy().flatten()
    
    ax.errorbar(x=z_loc, y=q_mu, yerr=q_sqrt_diag, color=color, fmt='o')
    
    mean, var = model.predict_f(x)
    plot_mean_var(x, mean, var, ax, color)

    return ax
    
# plot predictions
models = papl.models
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
x_plot = np.linspace(-1.25, 1.25, 101)[:, None]
[plot_model(model, data, x=x_plot, ax=axes[i]) for i, (model, data) in enumerate(zip(models, data_list))];
[axes[i].plot(X, Y, 'kx', mew=1., alpha=.1) for i, _ in enumerate(models)];
plt.tight_layout()
plt.savefig("toy_submodels_sparse.pdf")
# %%


ensemble = papl.get_ensemble_svgp()
# %%

ax = plot_model(ensemble, (X, Y), x=x_plot)
ax.set_title("ELBO: {:.2f}".format(ensemble.elbo((X, Y))))
plt.tight_layout()
plt.savefig("agg_sparse_init.pdf")
# %%

gpflow.optimizers.scipy.Scipy().minimize(
    ensemble.training_loss_closure((X, Y)),
    ensemble.trainable_variables,
    options={"disp": True, "maxiter": 100},
)
# %%

ax = plot_model(ensemble, (X, Y), x=x_plot)
ax.set_title("ELBO: {:.2f}".format(ensemble.elbo((X, Y))))
plt.tight_layout()
plt.savefig("agg_sparse.pdf")
# %%
