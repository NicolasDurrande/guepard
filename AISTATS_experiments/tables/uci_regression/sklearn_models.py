import numpy as np
from sklearn import (
    ensemble,
    linear_model,
    naive_bayes,
    neighbors,
    neural_network,
    svm,
    tree,
)


def _regression_model(model):
    class SKLWrapperRegression(object):
        def __init__(self, is_test=False, seed=0):
            self.model = model

        def fit(self, X, Y):
            self.model.fit(X, Y.flatten())
            self.std = np.std(self.model.predict(X) - Y.flatten())

        def predict(self, Xs):
            pred_mean = self.model.predict(Xs)[:, None]
            return pred_mean, np.ones_like(pred_mean) * (self.std + 1e-6) ** 2

    return SKLWrapperRegression


LinearRegressionModel = _regression_model(linear_model.LinearRegression())
SVM = _regression_model(svm.SVR())