import logging
import multiprocessing as mp
import pickle
from collections import OrderedDict

__all__ = ['BaseSearcher', 'RandomSearcher']

logger = logging.getLogger(__name__)


class BaseSearcher(object):
    """Base Searcher (A virtual class to inherit from)

    Parameters
    ----------
    configspace: ConfigSpace.ConfigurationSpace
        The configuration space to sample from. It contains the full
        specification of the Hyperparameters with their priors
    """
    LOCK = mp.Lock()

    def __init__(self, configspace):
        self.configspace = configspace
        self._results = OrderedDict()
        self._best_state_path = None

    @staticmethod
    def _reward_while_pending():
        """Defines the reward value which is assigned to config, while it is pending."""
        return float("-inf")

    def get_config(self, **kwargs):
        """Function to sample a new configuration

        This function is called inside TaskScheduler to query a new configuration

        Parameters
        ----------
        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        raise NotImplementedError(f'This function needs to be overwritten in {self.__class__.__name__}.')

    def update(self, config, reward, **kwargs):
        """Update the searcher with the newest metric report

        Note that for multi-fidelity schedulers (e.g., Hyperband), also
        intermediate results are reported. In this case, the time attribute is
        among **kwargs. We can also assume that if
        register_pending(config, ...) is received, then later on,
        the searcher receives update(config, ...) with milestone as time attribute.
        """
        is_done = kwargs.get('done', False)
        is_terminated = kwargs.get('terminated', False)
        # Only if evaluation is done or terminated (otherwise, it is an intermediate
        # result)
        if is_done or is_terminated:
            with self.LOCK:
                # Note: In certain versions of a scheduler, we may see 'terminated'
                # several times for the same config. In this case, we log the best
                # (largest) result here
                config_pkl = pickle.dumps(config)
                old_reward = self._results.get(config_pkl, reward)
                self._results[config_pkl] = max(reward, old_reward)
            logger.info(f'Finished Task with config: {config} and reward: {reward}')

    def register_pending(self, config, milestone=None):
        """
        Signals to searcher that evaluation for config has started, but not
        yet finished, which allows model-based searchers to register this
        evaluation as pending.
        For multi-fidelity schedulers, milestone is the next milestone the
        evaluation will attend, so that model registers (config, milestone)
        as pending.
        In general, the searcher may assume that update is called with that
        config at a later time.
        """
        pass

    def get_best_reward(self):
        with self.LOCK:
            if self._results:
                return max(self._results.values())
        return self._reward_while_pending()

    def get_reward(self, config):
        k = pickle.dumps(config)
        with self.LOCK:
            assert k in self._results
            return self._results[k]

    def get_best_config(self):
        with self.LOCK:
            if self._results:
                config_pkl = max(self._results, key=self._results.get)
                return pickle.loads(config_pkl)
            else:
                return dict()

    def get_best_config_reward(self):
        with self.LOCK:
            if self._results:
                config_pkl = max(self._results, key=self._results.get)
                return pickle.loads(config_pkl), self._results[config_pkl]
            else:
                return dict(), self._reward_while_pending()

    def get_topK_configs(self, k):
        results_sorted = {k: v for k, v in sorted(self._results.items(), key=lambda item: item[1])}
        keys = list(results_sorted.keys())
        k = min(k, len(keys))
        topK_cfgs = [pickle.loads(key) for key in keys[:k]]
        return topK_cfgs

    def __repr__(self):
        config, reward = self.get_best_config_reward()
        reprstr = (
                f'{self.__class__.__name__}(' +
                f'\nConfigSpace: {self.configspace}.' +
                f'\nNumber of Trials: {len(self._results)}.' +
                f'\nBest Config: {config}' +
                f'\nBest Reward: {reward}' +
                f')'
        )
        return reprstr


class RandomSearcher(BaseSearcher):
    """Random sampling Searcher for ConfigSpace

    Parameters
    ----------
    configspace: ConfigSpace.ConfigurationSpace
        The configuration space to sample from. It contains the full
        specification of the Hyperparameters with their priors

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
    >>> searcher = at.searcher.RandomSearcher(train_fn.cs)
    >>> searcher.get_config()
    {'lr': 0.0031622777, 'wd': 0.0055}
    """
    MAX_RETRIES = 100

    def get_config(self, **kwargs):
        """Function to sample a new configuration at random

        Parameters
        ----------
        returns: config
            must return a valid configuration
        """
        if not self._results:  # no hyperparams have been tried yet, first try default config
            new_config = self.configspace.get_default_configuration().get_dictionary()
        else:
            new_config = self.configspace.sample_configuration().get_dictionary()
        with self.LOCK:
            num_tries = 1
            while pickle.dumps(new_config) in self._results.keys():
                assert num_tries <= self.MAX_RETRIES, \
                    f"Cannot find new config in BaseSearcher, even after {self.MAX_RETRIES} trials"
                new_config = self.configspace.sample_configuration().get_dictionary()
                num_tries += 1
            self._results[pickle.dumps(new_config)] = self._reward_while_pending()
        return new_config
