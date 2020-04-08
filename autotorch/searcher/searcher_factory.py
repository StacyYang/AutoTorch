from .searcher import RandomSearcher
from .grid_searcher import GridSearcher


def searcher_factory(name, **kwargs):
    if name == 'random':
        return RandomSearcher(**kwargs)
    elif name == 'grid':
        return GridSearcher(**kwargs)
    else:
        raise AssertionError("name = '{}' not supported".format(name))
