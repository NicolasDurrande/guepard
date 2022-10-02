"""
Train and evaluate SVGP model on SUSY dataset.
"""
from collections import namedtuple

import numpy as np
from scipy.cluster import vq
from sklearn.metrics import auc, roc_curve

import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow
import gpflux
from tensorflow.python.ops.gen_batch_ops import batch


from data import susy


Dataset = namedtuple("Dataset", ["X", "Y", "X_test", "Y_test"])


class Config:
    num_inducing = 1000


def get_data():
    X, Y, XT, YT = susy(int(20e3))
    data = Dataset(X=X, Y=Y, X_test=XT, Y_test=YT)
    return data


def build_model(data):
    num_data, X_dim = data.X.shape

    kernel = gpflow.kernels.Matern32(lengthscales=[0.5] * X_dim)
    ind = np.random.permutation(num_data)[:10000]
    Z, _ = vq.kmeans(data.X[ind, :], Config.num_inducing)
    inducing_variable = gpflow.inducing_variables.InducingPoints(Z)

    gp_layer = gpflux.layers.GPLayer(
        kernel,
        inducing_variable,
        num_data=num_data,
        num_latent_gps=1,
        mean_function=gpflow.mean_functions.Zero(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Bernoulli())

    model = gpflux.models.DeepGP([gp_layer], likelihood_layer)
    training_model = model.as_training_model()
    training_model.compile(tf.optimizers.Adam(0.01))

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        "loss", factor=0.95, patience=3, min_lr=1e-6, verbose=1
    )

    history = training_model.fit(
        {"inputs": data.X, "targets": data.Y},
        epochs=int(1e1),
        verbose=1,
        batch_size=1024,
        callbacks=[reduce_lr],
    )

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(history.history["loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].plot(history.history["lr"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("LR")
    plt.savefig("model_training_loss_and_lr.png")

    return model


def evaluate(model: tf.keras.Model, data: Dataset):
    def predict_in_batches(X_eval, batch_size=512):
        num_data_test = len(X_eval)
        y_predict = []
        for k in range(0, num_data_test, batch_size):
            y_predict.append(model(X_eval[k : k + batch_size]).y_mean)
        y_predict = np.concatenate(y_predict).ravel()  # [num_data_test,]
        assert len(y_predict) == num_data_test
        return y_predict

    fig, (ax1, ax2) = plt.subplots(1, 2)
    y_test_true = data.Y_test.ravel()
    y_test_predict = predict_in_batches(data.X_test)
    auc_test = plot_roc(y_test_predict, y_test_true, name="SVGP", title="test data", ax=ax1)
    y_true = data.Y.ravel()
    y_predict = predict_in_batches(data.X)
    auc_train = plot_roc(y_predict, y_true, name="SVGP", title="train data", ax=ax2)
    plt.savefig("AUC.png")
    return {"auc_test": auc_test, "auc_train": auc_train}


def plot_roc(y_prob, y_true, ax=None, name=None, title=None, color="darkorange") -> float:
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc = auc(fpr, tpr)

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    lw = 2
    ax.plot(
        fpr,
        tpr,
        color=color,
        lw=lw,
        label="{name} (area = {auc:.3f})".format(name=name, auc=roc_auc),
    )
    ax.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--", alpha=0.1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="lower right")
    return roc_auc


if __name__ == "__main__":
    data = get_data()
    model = build_model(data)
    AUCs = evaluate(model.as_prediction_model(), data)
    print(AUCs)

