"""
Train and evaluate SVGP model on SUSY dataset.
"""
from collections import namedtuple
from typing import Callable

import numpy as np
from scipy.cluster import vq
from sklearn.metrics import auc, roc_curve

import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow

import guepard


from data import susy


Dataset = namedtuple("Dataset", ["X", "Y", "X_test", "Y_test"])


class Config:
    # Number of models in the ensemble
    num_models_in_ensemble = 500
    num_data = 1_000_000


def get_data():
    # X, Y, XT, YT = susy(int(20e3))
    X, Y, XT, YT = susy(Config.num_data)
    data = Dataset(X=X, Y=Y, X_test=XT, Y_test=YT)
    print("X", X.shape)
    print("Xt", Y.shape)
    print("Y", XT.shape)
    print("Yt", YT.shape)
    return data


def build_model(data) -> guepard.EquivalentObsEnsemble:
    num_data, X_dim = data.X.shape
    _, label_X = vq.kmeans2(data.X, Config.num_models_in_ensemble, minit="points")
    datasets = [
        (data.X[label_X == p, :], data.Y[label_X == p, :])
        for p in range(Config.num_models_in_ensemble)
    ]
    print("Max size:")
    print(max([len(x) for x,y in datasets]))

    kernel = gpflow.kernels.Matern32(lengthscales=np.ones(X_dim))
    submodels = guepard.utilities.get_gpr_submodels(
        datasets, kernel, mean_function=None, noise_variance=1e-3
    )
    ensemble = guepard.EquivalentObsEnsemble(submodels)

    gpflow.optimizers.scipy.Scipy().minimize(
        tf.function(lambda: ensemble.training_loss(datasets)),
        ensemble.trainable_variables,
        options={"disp": True, "maxiter": 20},
    )

    return ensemble


def evaluate(predict_y_func: Callable, data: Dataset, batch_size: int = 2048):
    def predict_in_batches(X_eval):
        num_data_test = len(X_eval)
        y_predict = []
        for k in range(0, num_data_test, batch_size):
            X_test_batch = X_eval[k : k + batch_size]
            y_pred, _ = predict_y_func(X_test_batch)
            y_predict.append(y_pred)
        y_predict = np.concatenate(y_predict).ravel()  # [num_data_test,]
        assert len(y_predict) == num_data_test
        return y_predict

    fig, (ax1, ax2) = plt.subplots(1, 2)
    y_test_true = data.Y_test.ravel()
    y_test_predict = predict_in_batches(data.X_test)
    auc_test = plot_roc(
        y_test_predict, y_test_true, name="Ensemble", title="test data", ax=ax1
    )
    y_true = data.Y.ravel()
    y_predict = predict_in_batches(data.X)
    auc_train = plot_roc(y_predict, y_true, name="Ensemble", title="train data", ax=ax2)
    plt.savefig("AUC.png")
    return {"auc_test": auc_test, "auc_train": auc_train}


def plot_roc(
    y_prob, y_true, ax=None, name=None, title=None, color="darkorange"
) -> float:
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
    print("Getting Data")
    data = get_data()
    print("Building")
    model = build_model(data)
    print("Testing")
    AUCs = evaluate(model.predict_y, data, batch_size=2048)
    print("Full")
    print(AUCs)

    AUCs = evaluate(model.predict_y_marginals, data, batch_size=2048)
    print("Marginals")
    print(AUCs)
