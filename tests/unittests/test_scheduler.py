import numpy as np
import autotorch as at
from nose.plugins.attrib import attr

@at.args(
    lr=at.space.Real(1e-3, 1e-2, log=True),
    wd=at.space.Real(1e-3, 1e-2))
def train_fn(args, reporter):
    for e in range(10):
        dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

@at.args(
    lr=at.space.Choice(1e-3, 1e-2),
    wd=at.space.Choice(1e-3, 1e-2))
def rl_train_fn(args, reporter):
    for e in range(10):
        dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)


def test_fifo_scheduler():
    scheduler = at.scheduler.FIFOScheduler(train_fn,
                                           resource={'num_cpus': 2, 'num_gpus': 0},
                                           num_trials=10,
                                           reward_attr='accuracy',
                                           time_attr='epoch')
    scheduler.run()
    scheduler.join_jobs()

def test_hyperband_scheduler():
    scheduler = at.scheduler.HyperbandScheduler(train_fn,
                                                resource={'num_cpus': 2, 'num_gpus': 0},
                                                num_trials=10,
                                                reward_attr='accuracy',
                                                time_attr='epoch',
                                                grace_period=1)
    scheduler.run()
    scheduler.join_jobs()
