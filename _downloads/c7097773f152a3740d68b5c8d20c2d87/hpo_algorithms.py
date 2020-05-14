# -*- coding: utf-8 -*-
"""
Experiment Scheduler using HPO Algorithms
=========================================

In previous course, we learn how to define search spaces, construct a :class:`autotorch.searcher.RandomSearcher`, run a single trial using searcher suggested configurations.

In this course, we will construct a AutoTorch experiment scheduler and go through the overall system workflow using a toy example.

AutoTorch System Implementatin Logic
------------------------------------

.. image:: https://raw.githubusercontent.com/zhanghang1989/AutoGluonWebdata/master/doc/api/autogluon_system.png
    :width: 500px
    :alt: AutoTorch System Overview
    :align: center

Important components of the AutoTorch system include the Searcher, Scheduler and Resource Manager:

- The Searcher suggests hyperparameter configurations for the next training job.
- The Scheduler runs the training job when computation resources become available.

In this tutorial, we illustrate how various search algorithms work and
compare their performance via toy experiments.

FIFO Scheduling vs. Early Stopping
----------------------------------

In this section, we compare the different behaviors of a sequential First In, First Out (FIFO) scheduler using :class:`autotorch.scheduler.FIFOScheduler` vs. a preemptive scheduling algorithm
:class:`autotorch.scheduler.HyperbandScheduler` that early-terminates certain training jobs that do not appear promising during their early stages.

Create a Dummy Training Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import numpy as np
import autotorch as at

@at.args(
    lr=at.Real(1e-3, 1e-2, log=True),
    wd=at.Real(1e-3, 1e-2))
def train_fn(args, reporter):
    for e in range(10):
        dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

###############################################################
# - FIFO Scheduler
#
# This scheduler runs training trials in order. When there are more resources available than required for a single training job, multiple training jobs may be run in parallel.

scheduler = at.scheduler.FIFOScheduler(train_fn,
                                       resource={'num_cpus': 2, 'num_gpus': 0},
                                       num_trials=20,
                                       reward_attr='accuracy',
                                       time_attr='epoch')
scheduler.run()
scheduler.join_jobs()

###############################################################
# Visualize the results:

scheduler.get_training_curves(plot=True, use_legend=False)

###############################################################
# - Hyperband Scheduler
#
# The Hyperband Scheduler terminates training trials that don't appear promising during the early stages to free up compute resources for more promising hyperparameter configurations.

scheduler = at.scheduler.HyperbandScheduler(train_fn,
                                            resource={'num_cpus': 2, 'num_gpus': 0},
                                            num_trials=20,
                                            reward_attr='accuracy',
                                            time_attr='epoch',
                                            grace_period=1)
scheduler.run()
scheduler.join_jobs()

###############################################################
# Visualize the results:

scheduler.get_training_curves(plot=True, use_legend=False)

