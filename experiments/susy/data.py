from typing import Optional

import pandas as pd
import numpy as np


def susy(N=None, test_percentage=0.1, feature_set="low_level", seed: Optional[int]=0):
    """
    :param N: int
        Total number of datapoints in train and test sets
        default None: complete dataset (5m points)
    :param test_percentage: float [0, 1]
        Fraction of the dataset used as test set
        defaults to 10%
    :param feature_set: str
        - low_level: 8 features
        - high_level: 10 features engineered from the low level ones
        - complete: 18 features. Union of low and high level
    
    :return X, Y, XT, YT
        features are normalized on [-1, 1]
    """
    low_level_features = [
        # low-level features:
        "lepton_1_pT",
        "lepton_1_eta",
        "lepton_1_phi",
        "lepton_2_pT",
        "lepton_2_eta",
        "lepton_2_phi",
        "missing_energy_magnitude",
        "missing_energy_phi",
    ]
    high_level_features = [
        # high-level features:
        "MET_rel",
        "axial_MET",
        "M_R",
        "M_TR_2",
        "R",
        "MT2",
        "S_R",
        "M_Delta_R",
        "dPhi_r_b",
        "cos_theta_r1"
    ]
    # the first column is target (1 for signal, 0 for background)
    # followed by the 18 features (8 low-level features then 10 high-level features)
    names = ["y", *low_level_features, *high_level_features]
    data = pd.read_csv("SUSY.csv", header=None, names=names)

    # Pick out the data
    if feature_set == "low_level":
        columns_of_interest = low_level_features
    elif feature_set == "high_level":
        columns_of_interest = high_level_features
    elif feature_set == "complete":
        columns_of_interest = [
            *low_level_features, 
            *high_level_features,
        ]
    else:
        raise NotImplementedError(
            "Unknown `feature_set` {}".format(feature_set))

    Y = data['y'].values
    X = data[columns_of_interest].values
    shuffled_indices = np.arange(len(X))

    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(shuffled_indices)
    else:
        print("Seed is None")

    X = X[shuffled_indices]
    Y = Y[shuffled_indices]
    
    n = len(X) if N is None else N
    n_train = n - int(test_percentage * n)

    XT = X[n_train:n]
    YT = Y[n_train:n]
    X = X[:n_train]
    Y = Y[:n_train]

    # Normalize X on [-1, 1]
    Xmin, Xmax = X.min(0), X.max(0)
    X = (X - Xmin) / (Xmax - Xmin)
    XT = (XT - Xmin) / (Xmax - Xmin)
    X = 2 * (X - 0.5)
    XT = 2 * (XT - 0.5)

    return X, Y.reshape(-1, 1), XT, YT.reshape(-1, 1)


if __name__ == "__main__":
    X, Y, XT, YT = susy()

    def arr_info(t, name):
        print("=" * 20)
        print(name)
        print("-" * 20)
        print("shape\t", t.shape)
        print("mean\t", t.mean(0))
        print("std\t", t.std(0))
        print("min\t", t.min(0))
        print("max\t", t.max(0))

    arr_info(X, "X")
    arr_info(Y, "Y")
    arr_info(XT, "X test")
    arr_info(YT, "Y test")
