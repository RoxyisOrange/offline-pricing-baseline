import warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
from .survival import SurvivalForest, surrogate_reward


class DenseNet:
    """Small NumPy MLP with explicit gradients and Adam updates."""
    def __init__(self, sizes, seed):
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, np.sqrt(2/a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(v) for v in self.weights+self.biases]
        self.v = [np.zeros_like(v) for v in self.weights+self.biases]
        self.step = 0

    def forward(self, x):
        activations = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.maximum(activations[-1]@w+b, 0))
        activations.append(activations[-1]@self.weights[-1]+self.biases[-1])
        return activations

    def backward(self, activations, dout):
        gw = [None]*len(self.weights)
        gb = [None]*len(self.biases)
        for j in reversed(range(len(self.weights))):
            gw[j] = activations[j].T@dout
            gb[j] = dout.sum(axis=0)
            dout = dout@self.weights[j].T
            if j:
                dout *= (activations[j] > 0)
        return gw+gb

    def update(self, grads, learning_rate, ridge=0):
        params = self.weights+self.biases
        self.step += 1
        norm = np.sqrt(sum(np.sum(g*g) for g in grads))
        factor = min(1., 20./(norm+1e-12))
        for j, (param, grad) in enumerate(zip(params, grads)):
            grad = grad*factor + (2*ridge*param if j < len(self.weights) else 0)
            self.m[j] = .9*self.m[j]+.1*grad
            self.v[j] = .999*self.v[j]+.001*grad*grad
            param -= learning_rate*(self.m[j]/(1-.9**self.step))/(np.sqrt(self.v[j]/(1-.999**self.step))+1e-8)


def _state(observed):
    return np.column_stack([observed.x1, observed.x2, observed.inventory/10.])


def _q_features(observed):
    return np.column_stack([_state(observed), observed.price])


def fit_density_oof(observed, cfg, seed):
    a = cfg["algorithm"]
    x = _state(observed)
    prices = observed.price.to_numpy(float)
    density = np.zeros(len(observed))
    folds = KFold(n_splits=a["cross_fitting_folds"], shuffle=True, random_state=seed)
    for fold, (train, held) in enumerate(folds.split(x)):
        model = DenseNet([3]+a["density_hidden"]+[2], seed+fold)
        rng = np.random.default_rng(seed+100+fold)
        for _ in range(a["density_epochs"]):
            for batch in np.array_split(rng.permutation(train),
                                        max(1, int(np.ceil(len(train)/a["density_batch_size"])))):
                acts = model.forward(x[batch])
                mu = acts[-1][:, 0]
                raw = acts[-1][:, 1]
                logsd = np.clip(raw, -4, 1)
                invvar = np.exp(-2*logsd)
                diff = mu-prices[batch]
                grad = np.column_stack([diff*invvar,
                                        (1-diff*diff*invvar)*((raw > -4)&(raw < 1))])/len(batch)
                model.update(model.backward(acts, grad), .001)
        out = model.forward(x[held])[-1]
        sd = np.exp(np.clip(out[:, 1], -4, 1))
        z = (prices[held]-out[:, 0])/sd
        density[held] = np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*sd)
    floor = a["density_floor"]
    cap = a["max_inverse_density"]
    inverse = np.minimum(1/np.maximum(density, floor), cap)
    return inverse, {"density_min_raw": float(density.min()),
                     "density_below_floor": int((density < floor).sum()),
                     "inverse_weight_at_cap": int((inverse >= cap).sum()),
                     "density_median": float(np.median(density))}


class OraclePricing:
    def __init__(self, config):
        self.config = config
        c = config["simulation"]
        step = config["evaluation"]["price_grid_step"]
        self.grid = np.linspace(c["price_min"], c["price_max"],
                                round((c["price_max"]-c["price_min"])/step)+1)

    def fit(self, observed_train_data=None):
        return self

    def predict_price(self, x1, x2, inventory):
        from .simulation import expected_profit
        x1, x2, inventory = np.broadcast_arrays(x1, x2, inventory)
        x1, x2, inventory = x1.ravel(), x2.ravel(), inventory.ravel()
        result = np.empty(len(x1))
        for start in range(0, len(x1), 100):
            sl = slice(start, start+100)
            q = expected_profit(x1[sl, None], x2[sl, None], inventory[sl, None],
                                self.grid[None, :], self.config)
            result[sl] = self.grid[np.argmax(q, axis=1)]
        return result.reshape(np.shape(x1))


class PricingPipeline:
    def __init__(self, config, reward_mode, seed=0):
        if reward_mode not in ("naive", "censoring_aware"):
            raise ValueError(reward_mode)
        self.config, self.reward_mode, self.seed = config, reward_mode, seed

    def fit(self, observed_train_data, density_cache=None):
        # Fail closed if oracle-only columns enter either learned method.
        expected = {"sample_id", "x1", "x2", "price", "inventory", "sales", "delta"}
        if set(observed_train_data.columns) != expected:
            raise ValueError("fit accepts only the seven observed-data columns")
        data = observed_train_data.reset_index(drop=True)
        a = self.config["algorithm"]
        n = len(data)
        self.reward = data.price.to_numpy()*data.sales.to_numpy()
        self.diagnostics = {}
        if self.reward_mode == "censoring_aware":
            forest = SurvivalForest(a["survival_forest_trees"], a["survival_min_leaf"], self.seed+7)
            forest.fit(data)
            self.reward, sd = surrogate_reward(data, forest,
                self.config["simulation"]["stockout_cost"], a["survival_mode"])
            self.diagnostics["survival"] = sd
        if density_cache is None:
            inverse, diagnostics = fit_density_oof(data, self.config, self.seed+23)
        else:
            inverse, diagnostics = density_cache
        self.density_cache = (inverse, diagnostics)
        self.diagnostics["density"] = diagnostics
        grid = np.linspace(0, 1, round(1/a["q_grid_step"])+1)
        q_smooth = np.empty((n, len(grid)))
        residual = np.empty(n)
        q_logged = np.empty(n)
        folds = KFold(n_splits=a["cross_fitting_folds"], shuffle=True, random_state=self.seed+23)
        qx = _q_features(data)
        state = _state(data)
        for fold, (train, held) in enumerate(folds.split(qx)):
            q_model = MLPRegressor(hidden_layer_sizes=tuple(a["q_hidden"]), solver="adam",
                                   max_iter=a["q_max_iter"], random_state=self.seed+fold+47,
                                   learning_rate_init=.003, early_stopping=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                q_model.fit(qx[train], self.reward[train])
            q_logged[held] = q_model.predict(qx[held])
            residual[held] = self.reward[held]-q_logged[held]
            batch = 100
            for start in range(0, len(held), batch):
                ids = held[start:start+batch]
                xx = np.column_stack([np.repeat(state[ids], len(grid), axis=0),
                                      np.tile(grid, len(ids))])
                q_grid = q_model.predict(xx).reshape(len(ids), len(grid))
                # Gaussian convolution approximates ∫_0^1 Q(p) K((p-π)/h) dp/h.
                q_smooth[ids] = gaussian_filter1d(q_grid, a["bandwidth"]/a["q_grid_step"],
                                                    axis=1, mode="constant", cval=0)
        self.q_logged = q_logged
        self.q_smooth = q_smooth
        self.grid = grid
        self.residual = residual
        self.inverse_density = inverse
        self._fit_policy(state, data.price.to_numpy(), q_smooth, residual, inverse)
        return self

    def _fit_policy(self, state, logged_price, q_smooth, residual, inverse):
        a = self.config["algorithm"]
        h = a["bandwidth"]
        network = DenseNet([3]+a["policy_hidden"]+[1], self.seed+1001)
        rng = np.random.default_rng(self.seed+1002)
        grid_step = self.grid[1]-self.grid[0]
        for _ in range(a["policy_epochs"]):
            for ids in np.array_split(rng.permutation(len(state)),
                                      max(1, int(np.ceil(len(state)/a["batch_size"])))):
                acts = network.forward(state[ids])
                pi = 1/(1+np.exp(-np.clip(acts[-1][:, 0], -30, 30)))
                left = np.minimum((pi/grid_step).astype(int), len(self.grid)-2)
                slope = (q_smooth[ids, left+1]-q_smooth[ids, left])/grid_step
                z = (logged_price[ids]-pi)/h
                kernel = np.exp(-.5*z*z)/(np.sqrt(2*np.pi)*h)
                dr_slope = inverse[ids]*residual[ids]*kernel*z/h
                dloss = -((slope+dr_slope)*pi*(1-pi)/len(ids))[:, None]
                network.update(network.backward(acts, dloss), a["policy_learning_rate"],
                               a["regularization"])
        self.policy = network
        self.diagnostics["residual_std"] = float(residual.std())
        self.diagnostics["training_mean_price"] = float(self.predict_price(
            state[:, 0], state[:, 1], state[:, 2]*10).mean())

    def predict_price(self, x1, x2, inventory):
        x1, x2, inventory = np.broadcast_arrays(x1, x2, inventory)
        x = np.column_stack([x1.ravel(), x2.ravel(), inventory.ravel()/10.])
        out = self.policy.forward(x)[-1][:, 0]
        return (1/(1+np.exp(-np.clip(out, -30, 30)))).reshape(x1.shape)
