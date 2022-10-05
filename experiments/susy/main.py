"""
Train and evaluate SVGP model on SUSY dataset.
"""
from typing import Optional
import datetime
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, asdict
from typing import Callable
from tqdm import tqdm, trange
import json
import fire


import numpy as np
from scipy.cluster import vq
from sklearn.metrics import auc, roc_curve

import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow

import guepard


from data import susy


Dataset = namedtuple("Dataset", ["X", "Y", "X_test", "Y_test"])
_FILE_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Config:
    num_models_in_ensemble: int = 10
    num_inducing: int = 1024
    num_data: int = None
    batch_size: int = 1024
    num_training_steps: int = 500
    log_freq: int = 25

_Config = Config()

def get_data(seed=None):
    X, Y, XT, YT = susy(_Config.num_data, seed=seed)
    data = Dataset(X=X, Y=Y, X_test=XT, Y_test=YT)
    return data


def build_model(data) -> guepard.EquivalentObsEnsemble:
    num_data, X_dim = data.X.shape
    _, label_X = vq.kmeans2(data.X, _Config.num_models_in_ensemble, minit="points")
    data_list = [
        (data.X[label_X == p, :], data.Y[label_X == p, :])
        for p in range(_Config.num_models_in_ensemble)
    ]

    kernel = gpflow.kernels.Matern32(lengthscales=np.ones((X_dim,)) * 1e-1)
    
    submodels = guepard.utilities.get_svgp_submodels(
        data_list=data_list,
        num_inducing_list=[_Config.num_inducing] * _Config.num_models_in_ensemble,
        kernel=kernel,
        mean_function=None,
        likelihood=gpflow.likelihoods.Bernoulli(),
        maxiter=0,
    )
    ensemble = guepard.EquivalentObsEnsemble(submodels)
    
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
        loss = lambda: ensemble.training_loss(batch_list) / (1.0 * num_data)
        opt.minimize(loss, ensemble.trainable_variables)

    opt = tf.keras.optimizers.Adam(1e-2)
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
    outfile = _FILE_DIR / "results" / filename
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


if __name__ == "__main__":
    fire.Fire(main)