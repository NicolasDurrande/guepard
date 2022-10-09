# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: 'Python 3.9.13 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # non-conjugate SVGP with 'efficient' parametrisation...
#

# %%
import numpy as np
import tensorflow as tf
import gpflow
import guepard
import matplotlib.pyplot as plt
import scipy
import time

rng = np.random.RandomState(123)
tf.random.set_seed(42)

D = 5                   # input space dimension
N = 1000                # number of observation points
M = 100                 # number of inducing points
P = int(np.sqrt(M))     # number of blocks in the SSVGP param (ie num of submodels in the GP ensemble)
M = P**2
num_rep = 5

var = 3.
lengthscale = M**(-1/D)  # inducing points are roughly 1 lengthscale apart

kernel =  gpflow.kernels.Matern32(variance=var, lengthscales=[lengthscale]*D)  
gpflow.set_trainable(kernel, False)
gpflow.utilities.print_summary(kernel)


lik = gpflow.likelihoods.Bernoulli()
mean_function = gpflow.mean_functions.Zero()

# The lines below are specific to the notebook format
# %matplotlib inline
plt.rcParams["figure.figsize"] = (12, 6)
from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: 150em; }</style>"));


# %%

# %% [markdown]
# We generate samples from the prior distribution as test function.
#

# %%
def make_data(num_rep, N, D, P):
    X = np.random.uniform(size=(num_rep, N, D))
    Z = np.empty((num_rep, M, D))
    Y = np.empty((num_rep, N, 1))
    for i in range(num_rep):
        dummy_data = np.array([[100.]]*D).T, np.array([[0.]])
        F = gpflow.models.GPR(dummy_data, kernel).predict_f_samples(X[i], 1).numpy()[0, :, :]
        Y[i] = np.random.binomial(1, lik.invlink(F))
        Z[i], _ = scipy.cluster.vq.kmeans2(X[i], M, minit='points')
    return X, Y, Z

X, Y, Z = make_data(num_rep, N, D, P)
plt.plot(X[0, :, 0], Y[0, :, 0], "kx")

# %% [markdown]
# We fit a classic SVGP model with 50 inducing points taken uniformly in the input space


# %%
maxiter = 1000

def callback(x):
    global times
    global traces
    times += [time.time()]
    traces += [training_loss()]

rng = np.random.RandomState(123)

Times = np.nan * np.zeros((maxiter, num_rep))
Traces = np.nan * np.zeros((maxiter, num_rep))

for i in range(num_rep):
    print(f"{i}/{num_rep}")
    _, label_x = scipy.cluster.vq.kmeans2(X[i], Z[i])
    var_init = np.random.uniform(var/2., var*2)
    lengthscales_init = np.random.uniform(lengthscale/2., lengthscale*2., size=D) 
    kernel =  gpflow.kernels.Matern32(variance=var_init, lengthscales=lengthscales_init)  # inducing points are roughly 1 lengthscale apart
    m = gpflow.models.SVGP(kernel, lik, Z[i], num_data=N, whiten=False)
    gpflow.set_trainable(m.inducing_variable, False)
    training_loss = m.training_loss_closure((X[i], Y[i]), compile=True)
    
    opt = gpflow.optimizers.Scipy()
    traces = []
    times = []
    start_time = time.time()
    opt.minimize(training_loss, m.trainable_variables, callback=callback, options={'maxiter':maxiter})
    Traces[:len(traces), i] = traces
    Times[:len(times), i] = [t - start_time for t in times]

Times_svgp = Times
Traces_svgp = Traces


# %%
m

# %%
Times = Times_svgp
Traces = Traces_svgp

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
for i in range(0, num_rep):
    plt.plot(Times[:, i], -Traces[:, i], label=i);
    plt.xlabel("wall clock time")
    plt.ylabel("ELBO")
    plt.title("SVGP")
    #plt.ylim(-500, 100)
plt.legend()
plt.title(f"SVGP: N = {N}, D = {D}, M = {M}")

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
for i in range(0, num_rep):
    plt.plot(-Traces[:, i], label=i);
    plt.xlabel("iterations")
    plt.ylabel("ELBO")
    plt.title("SVGP")
    #plt.ylim(-500, 100)
plt.legend()
plt.title(f"SVGP: N = {N}, D = {D}, M = {M}")


# %% [markdown]
# Now, let's fit 50 SVGP with one inducing point each, use equivalent observations to merge the ensemble predicions and use the result to warm start an SVGP optimisation...

# %%
Times = np.nan * np.zeros((maxiter, num_rep))
Traces = np.nan * np.zeros((maxiter, num_rep))

for i in range(num_rep):
    print(f"{i}/{num_rep}")
    _, label_x = scipy.cluster.vq.kmeans2(X[i], Z[i])
    _, label_z = scipy.cluster.vq.kmeans2(Z[i], P, minit='points') 

    start_time = time.time()

    var_init = np.random.uniform(var/2., var*2)
    lengthscales_init = np.random.uniform(lengthscale/2., lengthscale*2., size=D) 
    kernel =  gpflow.kernels.Matern32(variance=var_init, lengthscales=lengthscales_init)  # inducing points are roughly 1 lengthscale apart
    gpflow.set_trainable(kernel, False)
    
    m_ensemble = []
    for j in range(P):
        mask_x = label_z[label_x] == j
        mask_z = label_z == j
        X_ = X[i, mask_x, :]
        Y_ = Y[i, mask_x, :]
        Z_ = Z[i, mask_z, :]

        m = gpflow.models.SVGP(inducing_variable=Z_, likelihood=lik, kernel=kernel, mean_function=mean_function, whiten=False)
        gpflow.set_trainable(m.inducing_variable, False)
        opt = gpflow.optimizers.Scipy()
        training_loss = m.training_loss_closure((X_, Y_), compile=True)
        opt_logs = opt.minimize(training_loss, m.trainable_variables, options={'maxiter':10});
        gpflow.utilities.print_summary(m)
        m_ensemble += [m]

    Zs, q_mus, q_sqrts = guepard.utilities.init_ssvgp_with_ensemble(m_ensemble, add_jitter=True) 
    gpflow.set_trainable(kernel, True)
    m_ssvgp = guepard.SparseSVGP(kernel, lik, Zs, q_mus, q_sqrts, whiten=False)
    gpflow.set_trainable(m_ssvgp.inducing_variable, False)

    traces = []
    times = []
    training_loss = m_ssvgp.training_loss_closure((X[i], Y[i]), compile=True)
    opt = gpflow.optimizers.Scipy()

    # start_time = time.time()

    opt.minimize(training_loss, m_ssvgp.trainable_variables, callback=callback, options={'maxiter':maxiter})
    Traces[:len(traces), i] = traces
    Times[:len(times), i] = [t - start_time for t in times]

Times_ssvgp = Times
Traces_ssvgp = Traces


# %%
m_ssvgp

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
for i in range(0, num_rep):
    plt.plot(Times[:, i], -Traces[:, i], label=i);
    plt.xlabel("wall clock time")
    plt.ylabel("ELBO")
    plt.title("SSVGP")
    #plt.ylim(-500, 100)
plt.legend()
plt.title(f"SSVGP: N = {N}, D = {D}, M = {M}, P = {P}")

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
for i in range(0, num_rep):
    plt.plot(-Traces[:, i], label=i);
    plt.xlabel("iterations  (ignoring sub-models pre-training)")
    plt.ylabel("ELBO")
    #plt.ylim(-500, 100)
plt.legend()
plt.title(f"SSVGP: N = {N}, D = {D}, M = {M}, P = {P}")


# %%
SGPR_evidence = -np.nanmax(-Traces_ssvgp, axis=0)
ylim = (-5000, 500)

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
for i in range(0, num_rep):
    plt.plot(Times_ssvgp[:, i], -Traces_ssvgp[:, i] + SGPR_evidence[i], "C0", label="SSVGP" if i==0 else None);
    plt.plot(Times_svgp[:, i], -Traces_svgp[:, i] + SGPR_evidence[i], "C1", label="SVGP" if i==0 else None);
    plt.xlabel("wall clock time")
    plt.ylabel("$ELBO - max(ELBO_{SVGP})$")
    #plt.ylim(ylim)

plt.legend()
plt.title(f"N = {N}, D = {D}, M = {M}, P = {P}")

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
for i in range(0, num_rep):
    plt.plot(-Traces_ssvgp[:, i] + SGPR_evidence[i], "C0", label="SSVGP" if i==0 else None);
    plt.plot(-Traces_svgp[:, i] + SGPR_evidence[i], "C1", label="SVGP" if i==0 else None);
    plt.xlabel("iterations (ignoring sub-models pre-training)")
    plt.ylabel("$ELBO - max(ELBO_{SVGP})$")
    #plt.ylim(ylim)
plt.legend()
plt.title(f"N = {N}, D = {D}, M = {M}, P = {P}")


# %%
np.nanmax(Traces_ssvgp, axis=0)

# %% [markdown]
#
#
#
