"""Extremely randomized survival forest approximation with leaf Nelson-Aalen curves.

ExtraTrees splits minimize squared observed-time error, rather than ranger's survival
splitting criterion. Event indicators are used in leaf hazards. This distinction matters.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


class SurvivalForest:
    def __init__(self, n_trees=500, min_leaf=15, seed=0):
        self.model = ExtraTreesRegressor(n_estimators=n_trees, min_samples_leaf=min_leaf,
                                         bootstrap=True, max_features=1.0,
                                         random_state=seed, n_jobs=-1)

    def fit(self, observed):
        self.features = observed[["x1", "x2", "price"]].to_numpy(float)
        self.times = observed.sales.to_numpy(int)
        self.events = observed.delta.to_numpy(int)
        self.max_time = int(self.times.max())
        self.model.fit(self.features, self.times)
        return self

    def survival(self, features):
        features = np.asarray(features, float)
        output = np.zeros((len(features), self.max_time+1), float)
        for tree in self.model.estimators_:
            train_leaf = tree.apply(self.features)
            query_leaf = tree.apply(features)
            unique, inverse = np.unique(train_leaf, return_inverse=True)
            curves = np.empty((len(unique), self.max_time+1), float)
            for j in range(len(unique)):
                ix = inverse == j
                counts = np.bincount(self.times[ix], minlength=self.max_time+1)
                events = np.bincount(self.times[ix], weights=self.events[ix],
                                     minlength=self.max_time+1)
                risk = np.cumsum(counts[::-1])[::-1]
                hazard = np.divide(events, risk, out=np.zeros_like(events), where=risk > 0)
                curves[j] = np.exp(-np.cumsum(hazard))
            pos = np.searchsorted(unique, query_leaf)
            output += curves[pos]
        return output / len(self.model.estimators_)


def surrogate_reward(observed, forest, cost, mode="paper_continuous"):
    price = observed.price.to_numpy(float)
    sales = observed.sales.to_numpy(int)
    censored = observed.delta.to_numpy(int) == 0
    reward = price*sales
    if not censored.any():
        return reward, {"censored": 0, "small_denominators": 0,
                        "tail_cap": getattr(forest, "max_time", None)}
    sf = forest.survival(observed.loc[censored, ["x1", "x2", "price"]].to_numpy())
    s = sales[censored]
    # The paper truncates the survival tail at the largest observed sale.
    numer = np.array([row[t:forest.max_time].sum() for row, t in zip(sf, s)])
    if mode != "paper_continuous":
        raise ValueError("Unknown survival mode")
    denom = sf[np.arange(len(s)), np.minimum(s, forest.max_time)]
    small = denom < 1e-6
    extra = np.maximum(numer / np.maximum(denom, 1e-6), 0)
    reward[censored] -= cost*extra
    return reward, {"censored": int(censored.sum()),
                    "small_denominators": int(small.sum()), "tail_cap": forest.max_time,
                    "mean_estimated_excess": float(extra.mean())}
