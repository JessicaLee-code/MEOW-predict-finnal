import os
from sklearn.linear_model import Ridge
from log import log


class MeowModel(object):
    def __init__(self, cacheDir):
        self.estimator = Ridge(
            alpha=0.5,
            random_state=None,
            fit_intercept=False,
            tol=1e-8
        )

    def fit(self, xdf, ydf):
        self.estimator.fit(
            X=xdf.to_numpy(),
            y=ydf.to_numpy(),
        )
        log.inf("Done fitting")

    def predict(self, xdf):
        return self.estimator.predict(xdf.to_numpy())
