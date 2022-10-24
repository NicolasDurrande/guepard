"""
Train and evaluate SVGP model on SUSY dataset.
"""
import datetime
import json
from collections import namedtuple
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data import susy
from scipy.cluster import vq
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm, trange

import gpflow

import guepard

Dataset = namedtuple("Dataset", ["X", "Y", "X_test", "Y_test"])
_FILE_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Config:
    num_models_in_ensemble: int = 10
    only_pretrain: bool = True
    lr: float = 5e-4
    num_inducing: int = 500
    num_data: int = 20_000
    batch_size: int = 1024
    num_training_steps: int = 200
    num_pretraining_steps: int = 100
    log_freq: int = 20

_Config = Config()

def get_data(seed=None):
    # X, Y, XT, YT = susy(_Config.num_data, seed=seed)
    X, Y, XT, YT = susy(None, seed=seed)
    N = _Config.num_data
    data = Dataset(X=X[:N], Y=Y[:N], X_test=XT, Y_test=YT)
    return data


def estimate_kernel(X_train, Y_train):
    data = (X_train, Y_train)
    X_dim = data[0].shape[-1]
    kernel = gpflow.kernels.Matern32(lengthscales=np.ones((X_dim,)) * 5e-1)
    likelihood=gpflow.likelihoods.Bernoulli()
    model = gpflow.models.VGP(data, kernel, likelihood)
    gpflow.optimizers.scipy.Scipy().minimize(
        model.training_loss_closure(),
        model.trainable_variables,
        options={"disp": True, "maxiter": 500},
    )
    return model



def build_model(data) -> guepard.EquivalentObsEnsemble:
    num_data, X_dim = data.X.shape
    _, label_X = vq.kmeans2(data.X, _Config.num_models_in_ensemble, minit="points")
    data_list = [
        (data.X[label_X == p, :], data.Y[label_X == p, :])
        for p in range(_Config.num_models_in_ensemble)
    ]
    print("Num experts", _Config.num_models_in_ensemble)
    print("Max point per expert", max([len(t[0]) for t in data_list]))
    print("Min point per expert", min([len(t[0]) for t in data_list]))

    ells = np.array([0.31174366, 2.30417976, 5.94359217, 0.52124237, 3.01414852, 7.26889154, 0.26389877, 6.9380755 ])
    kernel = gpflow.kernels.Matern32(lengthscales=ells)
    kernel.variance.assign(3.05)
    gpflow.set_trainable(kernel, False)
    gpflow.utilities.print_summary(kernel)
    
    submodels = guepard.utilities.get_svgp_submodels(
        data_list=data_list,
        num_inducing_list=[_Config.num_inducing] * _Config.num_models_in_ensemble,
        kernel=kernel,
        mean_function=None,
        likelihood=gpflow.likelihoods.Bernoulli(),
        maxiter=_Config.num_pretraining_steps,
    )
    ensemble = guepard.EquivalentObsEnsemble(submodels)
    if _Config.only_pretrain:
        return ensemble

    gpflow.set_trainable(kernel, True)

    def _create_dataset(data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size=10_000)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=_Config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return iter(dataset)

    dataset_list = list(map(_create_dataset, data_list))

    @tf.function
    def step() -> None:
        batch_list = list(map(next, dataset_list))
        loss = lambda: ensemble.training_loss()
        opt.minimize(loss, ensemble.trainable_variables)

    opt = tf.keras.optimizers.Adam(_Config.lr)
    valid_data = list(map(next, dataset_list))
    tqdm_range = trange(_Config.num_training_steps)
    for i in tqdm_range:
        try:
            step()
        except KeyboardInterrupt as e:
            print("User stopped training...")
            break

        if i % _Config.log_freq == 0:
            l = ensemble.training_loss(valid_data).numpy()
            tqdm_range.set_description(f"{str(i).zfill(6)}: {l:.2f}")

    gpflow.utilities.print_summary(kernel)
    return ensemble


def evaluate(predict_y_func: Callable, data: Dataset, batch_size: int = 2048, name: str = 'auc'):
    def predict_in_batches(X_eval):
        num_data_test = len(X_eval)
        y_predict = []
        for k in tqdm(range(0, num_data_test, batch_size), total=num_data_test//batch_size):
            X_test_batch = X_eval[k : k + batch_size]
            y_pred, _ = predict_y_func(X_test_batch)
            y_predict.append(y_pred)
        y_predict = np.concatenate(y_predict).ravel()  # [num_data_test,]
        assert len(y_predict) == num_data_test
        return y_predict

    y_test_true = data.Y_test.ravel()
    y_test_predict = predict_in_batches(data.X_test)
    fpr, tpr, _ = roc_curve(y_test_true.ravel(), y_test_predict.ravel())
    roc_auc = auc(fpr, tpr)
    return {f"{name}_auc_test": roc_auc}


def main(seed: Optional[int] = 0):
    date = datetime.datetime.now().strftime("%b%d_%H%M%S")
    ext = "json"
    config = {**asdict(_Config), **{'seed': seed}}
    # Hashing config to get unique filename...
    filename = date + '_' + str(abs(hash(frozenset(config.items()))))[:7] + "." + ext
    outfile = _FILE_DIR / "tmp" / filename
    if outfile.exists():
        print("Experiment already exists. Quitting experiment.")
        return -1

    print("Getting Data")
    data = get_data(seed)
    print("Building")
    model = build_model(data)
    print("Testing")
    metrics = evaluate(model.predict_y_marginals, data, batch_size=2048, name="marginals")
    print(metrics)

    print("Saving results")
    results = {**config, **metrics}
    with open(outfile, "w") as outfile:
        json.dump(results, outfile, indent=4)


# if __name__ == "__main__":
#     import fire
#     fire.Fire(main)