from .searcher import RandomSearcher
from .grid_searcher import GridSearcher
from .bayesopt import BayesOptSearcher

def searcher_factory(name, **kwargs):
    if name == 'random':
        return RandomSearcher(**kwargs)
    elif name == 'grid':
        return GridSearcher(**kwargs)
    elif name == 'bayesopt':
        return BayesOptSearcher(**kwargs)
    else:
        raise AssertionError("name = '{}' not supported".format(name))
