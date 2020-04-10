import numpy as np
import autotorch as at
@at.args(
    lr=at.space.Real(1e-3, 1e-2, log=True),
    wd=at.space.Real(1e-3, 1e-2))
def train_fn(args, reporter):
    print('lr: {}, wd: {}'.format(args.lr, args.wd))
    for e in range(10):
        dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)
searcher = at.searcher.RandomSearcher(train_fn.cs)

g = at.Dict(
        key1=at.Choice('alpha', 'beta'),
        key2=at.Int(0, 3),
    )
print(g)

sequence = at.List(
    at.Choice('conv3x3', 'conv5x5', 'conv7x7'),
    at.Choice('BatchNorm', 'InstanceNorm'),
    at.Choice('relu', 'sigmoid'),
)
