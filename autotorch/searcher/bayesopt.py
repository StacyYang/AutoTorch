import warnings
import pickle
import numpy as np

from .searcher import BaseSearcher

__all__ = ['BayesOptSearcher']

def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class BayesOptSearcher(BaseSearcher):
    """A wrapper around BayesOpt

    Requires scikit-learn to be installed. You can install scikit-learn with the
    command: ``pip install scikit-learn``.

    Parameters
    ----------
    configspace: ConfigSpace.ConfigurationSpace
        The configuration space to sample from. It contains the full
        specification of the Hyperparameters with their priors
    lazy_configs: list of dict
        Mannual configurations to handle some special case. In some cases,
        not all configurations are valid in the space, and we need to pre-sample
        valid configurations with extra constraints.

    Examples
    --------
    >>> import numpy as np
    >>> import autotorch as at
    >>> @at.args(
    ...     lr=at.Real(1e-3, 1e-2, log=True),
    ...     wd=at.Real(1e-3, 1e-2))
    >>> def train_fn(args, reporter):
    ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
    ...     for e in range(10):
    ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
    ...         reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)
    >>> searcher = at.searcher.BayesOptSearcher(train_fn.cs)
    >>> searcher.get_config()
    {'lr': 0.0031622777, 'wd': 0.0055}
    """
    def __init__(self, configspace, lazy_configs=None, random_state=None,
                 ac_kind='ucb', **kwargs):
        super().__init__(configspace)
        self.ac_kind = ac_kind
        self.kappa = kwargs.get('kappa', 2.576)
        self.xi = kwargs.get('xi', 0.0)
        # fix order of hyperparams in configspace.
        hp_ordering = configspace.get_hyperparameter_names()
        pbounds = {}
        self._integer_spaces = []
        for hp in hp_ordering:
            hp_obj = configspace.get_hyperparameter(hp)
            hp_type = str(type(hp_obj)).lower()
            if 'integer' in hp_type:
                pbounds[hp] = (hp_obj.lower, hp_obj.upper+0.999)
                self._integer_spaces.append(hp)
            elif 'float' in hp_type:
                pbounds[hp] = (hp_obj.lower, hp_obj.upper)
            else:
                raise NotImplementedError

        self._space = TargetSpace(pbounds, random_state)
    
        # lazy_configs?
        self.lazy_space = LazySpace(pbounds, random_state)
        if lazy_configs is not None:
            for cfg in lazy_configs:
                self.lazy_space.register(cfg)

        self._random_state = ensure_rng(random_state)
        # Internal GP regressor
        from sklearn.gaussian_process.kernels import Matern
        from sklearn.gaussian_process import GaussianProcessRegressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=self._random_state,
        )

    def _format_configs(self, configs):
        configs = self._space.array_to_params(configs)
        for k in self._integer_spaces:
            configs[k] = int(configs[k])
        return configs
    
    def get_config(self, **kwargs):
        """Function to sample a new configuration

        Parameters
        ----------
        returns: config
            must return a valid configuration
        """
        # fit gp model
        if len(self._space) == 0:
            if len(self.lazy_space) > 0:
                idx = np.random.randint(0, len(self.lazy_space))
                new_config = self._format_configs(self.lazy_space.params[idx])
                with self.LOCK:
                    self.lazy_space.delete(idx)
            else:
                new_config = self._format_configs(self._space.random_sample())
            with self.LOCK:
                assert pickle.dumps(new_config) not in self._results.keys()
                self._results[pickle.dumps(new_config)] = self._reward_while_pending()
            return new_config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.LOCK:
                self._gp.fit(self._space.params, self._space.target)

        if len(self.lazy_space) > 0:
            # return the best in the pool
            ys = self.acquisition(self.lazy_space.params, self._space.target.max())
            x_max = self.lazy_space.params[ys.argmax()]
            with self.LOCK:
                self.lazy_space.delete(ys.argmax())
            suggestion = np.clip(x_max, self._space.bounds[:, 0], self._space.bounds[:, 1])
        else:
            # Finding argmax of the acquisition function.
            suggestion = self.acq_max()

        new_config = self._format_configs(suggestion)

        with self.LOCK:
            assert pickle.dumps(new_config) not in self._results.keys()
            self._results[pickle.dumps(new_config)] = self._reward_while_pending()
        return new_config

    def update(self, config, reward, **kwargs):
        """Update the searcher with the newest metric report
        """
        super().update(config, reward, **kwargs)
        is_done = kwargs.get('done', False)
        is_terminated = kwargs.get('terminated', False)
        # Only if evaluation is done or terminated (otherwise, it is an intermediate
        # result)
        if is_done or is_terminated:
            with self.LOCK:
                self._space.register(config, reward)

    def acq_max(self, n_warmup=10000, n_iter=20):
        """
        A function to find the maximum of the acquisition function
        It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
        optimization method. First by sampling `n_warmup` (1e5) points at random,
        and then running L-BFGS-B from `n_iter` (250) random starting points.
        Parameters
        ----------
        :param n_warmup:
            number of times to randomly sample the aquisition function
        :param n_iter:
            number of times to run scipy.minimize
        Returns
        -------
        :return: x_max, The arg max of the acquisition function.
        """
        y_max=self._space.target.max()
        bounds = self._space.bounds

        # Warm up with random points
        from scipy.optimize import minimize
        x_tries = self._random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(n_warmup, bounds.shape[0]))
        ys = self.acquisition(x_tries, y_max=y_max)
        x_max = x_tries[ys.argmax()]
        x_max = np.clip(x_max, bounds[:, 0], bounds[:, 1])
        max_acq = ys.max()

        # Explore the parameter space more throughly
        x_seeds = self._random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(n_iter, bounds.shape[0]))
        succeed = False
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: -self.acquisition(x.reshape(1, -1), y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")

            # See if success
            if not res.success:
                continue

            x_res = np.clip(res.x, bounds[:, 0], bounds[:, 1])
            # Store it if better than previous minimum(maximum).
            if (max_acq is None or -res.fun[0] >= max_acq) and \
                    pickle.dumps(self._format_configs(x_res)) not in self._results.keys():
                x_max = x_res
                max_acq = -res.fun[0]
                succeed = True

        if not succeed:
            idx = ys.argmax()
            while pickle.dumps(self._format_configs(x_max)) in self._results.keys():
                x_tries = self._random_state.uniform(bounds[:, 0], bounds[:, 1],
                                                     size=(n_warmup, bounds.shape[0]))
                ys = self.acquisition(x_tries, y_max=y_max)
                x_max = x_tries[ys.argmax()]
                x_max = np.clip(x_max, bounds[:, 0], bounds[:, 1])

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return x_max
    

    def acquisition(self, x, y_max):
        if self.ac_kind == 'ucb':
            return self._ucb(x, self._gp, self.kappa)
        if self.ac_kind == 'ei':
            return self._ei(x, self._gp, y_max, self.xi)
        if self.ac_kind == 'poi':
            return self._poi(x, self._gp, y_max, self.xi)
        else:
            raise NotImplementedError

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
  
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)


 
def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class LazySpace(object):
    def __init__(self, pbounds, random_state=None):
        self.random_state = ensure_rng(random_state)
        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        return len(self._params)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def delete(self, idx):
        self._params = np.delete(self._params, idx, 0)

    def register(self, params):
        """
        Append a point and its target value to the known data.
        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim
        Raises
        ------
        KeyError:
            if the point is not unique
        """
        x = self._as_array(params)
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = None
        self._params = np.concatenate([self._params, x.reshape(1, -1)])

    def random_sample(self):
        """
        Creates random points within the bounds of the space.
        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()

class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added
    Example
    """
    def __init__(self, pbounds, random_state=None):
        """
        Parameters
        ----------
        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.
        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state)


        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params, target):
        """
        Append a point and its target value to the known data.
        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim
        y : float
            target function value
        Raises
        ------
        KeyError:
            if the point is not unique
        Notes
        -----
        runs in ammortized constant time
        """
        x = self._as_array(params)
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def random_sample(self):
        """
        Creates random points within the bounds of the space.
        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()

    def max(self):
        """Get maximum target value found and corresponding parametes."""
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        """Get all target values found and corresponding parametes."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds
        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]
