__all__ = ['GridSearcher']

from .searcher import BaseSearcher
from ..core.space import Choice

class GridSearcher(BaseSearcher):
    """Grid Searcher, only search spaces :class:`autotorch.space.Choice`

    Requires scikit-learn to be installed. You can install scikit-learn with the
    command: ``pip install scikit-learn``.

    Examples
    --------
    >>> import autotorch as ag
    >>> @ag.args(
    >>>     x=ag.space.Choice(0, 1, 2),
    >>>     y=ag.space.Choice('a', 'b', 'c'))
    >>> def train_fn(args, reporter):
    ...     pass
    >>> searcher = ag.searcher.GridSearcher(train_fn.cs)
    >>> searcher.get_config()
    Number of configurations for grid search is 9
    {'x.choice': 2, 'y.choice': 2}
   
    """
    def __init__(self, configspace):
        super().__init__(configspace)
        param_grid = {}
        hp_ordering = configspace.get_hyperparameter_names()
        for hp in hp_ordering:
            hp_obj = configspace.get_hyperparameter(hp)
            hp_type = str(type(hp_obj)).lower()
            assert 'categorical' in hp_type, \
                'Only Choice is supported, but {} is {}'.format(hp, hp_type)
            param_grid[hp] = hp_obj.choices

        from sklearn.model_selection import ParameterGrid
        self._configs = list(ParameterGrid(param_grid))
        print('Number of configurations for grid search is {}'.format(len(self._configs)))

    def __len__(self):
        return len(self._configs)

    def get_config(self):
        return self._configs.pop()
