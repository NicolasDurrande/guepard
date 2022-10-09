from typing import Callable, Tuple

import datetime
import json
import pprint
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
from bayesian_benchmarks.data import _ALL_REGRESSION_DATATSETS
from bayesian_benchmarks.data import Dataset
from sacred import Experiment
from scipy.stats import norm
from utils import ExperimentName

from sklearn_models import SVM, LinearRegressionModel
from models import GuepardRegression, gPoE_unif, gPoE_var, rBCM_entr, BAR_var, GPR


_THIS_DIR = Path(__file__).parent
_LOGS = _THIS_DIR / "tmp"
_EXPERIMENT = Experiment("UCI")


@_EXPERIMENT.config
def config():
    # Timestamp (None)
    date = datetime.datetime.now().strftime("%b%d_%H%M%S")
    # Dataset (None)
    dataset = "Yacht"
    # Dataset split (None)
    split = 0
    # Model name (None)
    model = "SVM"
    # Task (T)
    task = "reg"


@_EXPERIMENT.capture
def experiment_name(_config: Any) -> Optional[str]:
    config = _config.copy()
    del config["seed"]
    return ExperimentName(_EXPERIMENT, config).get()


@_EXPERIMENT.capture
def get_dataset_class(dataset) -> Type[Dataset]:
    assert dataset.lower() in _ALL_REGRESSION_DATATSETS.keys()
    return _ALL_REGRESSION_DATATSETS[dataset.lower()]


@_EXPERIMENT.capture
def get_data(split, dataset):
    data = get_dataset_class(dataset)(split=split)
    return data


@_EXPERIMENT.capture
def get_model(model):
    if model == "linear":
        return LinearRegressionModel()
    elif model == "SVM":
        return SVM()
    elif model == "guepard":
        return GuepardRegression()
    elif model == "gPoE_unif":
        return gPoE_unif()
    elif model == "gPoE_var":
        return gPoE_var()
    elif model == "rBCM_entr":
        return rBCM_entr()
    elif model == "BAR_var":
        return BAR_var()
    elif model == "gpr":
        return GPR()
    else:
        raise NotImplementedError(f"Unknown model type {model}")


PREDICT_Y_FN = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


def evaluate_model(predict_y: PREDICT_Y_FN, data_test):
    XT, YT = data_test
    y_mean, y_var = predict_y(XT)
    d = YT - y_mean
    l = norm.logpdf(YT, loc=y_mean, scale=y_var ** 0.5)
    mse = np.average(d ** 2)
    rmse = mse ** 0.5
    nlpd = -np.average(l)
    return dict(rmse=rmse, mse=mse, nlpd=nlpd)


@_EXPERIMENT.automain
def main(_config):
    data = get_data()

    # Build
    model = get_model() 

    failed = False
    try:
        # Train
        model.fit(data.X_train, data.Y_train)
        # Evaluate
        metrics = evaluate_model(model.predict, (data.X_test, data.Y_test))
    except Exception as e:
        failed = True
        metrics = dict(rmse=np.nan, mse=np.nan, nlpd=np.nan)
        print(e)

    # Save
    data_stats = {
        "num_data": get_dataset_class().N,
        "input_dim": get_dataset_class().D,
    }

    results = {**_config, **data_stats, **metrics, 'failed': failed}
    with open(f"{_LOGS}/{experiment_name()}.json", "w") as fp:
        json.dump(results, fp, indent=2)

    print("=" * 60)
    print(experiment_name())
    pprint.pprint(results)
    print("=" * 60)
