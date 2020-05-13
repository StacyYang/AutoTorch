import numpy as np
import autotorch as at

def test_bayesopt_searcher():
    @at.args(
        lr=at.Real(1e-3, 1e-2, log=True),
        wd=at.Real(1e-3, 1e-2))
    def train_fn(args, reporter):
        for e in range(10):
            dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
            reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

    random_searcher = at.searcher.RandomSearcher(train_fn.cs)
    lazy_configs = []
    for i in range(10):
        lazy_configs.append(random_searcher.get_config())
    searcher = at.searcher.BayesOptSearcher(train_fn.cs, lazy_configs=lazy_configs)
    config = searcher.get_config()
    for i in range(20):
        if i < 10:
            assert config in lazy_configs
        searcher.update(config, np.random.uniform(0.1, 0.9), done=True)
        config = searcher.get_config()
