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


###############################################################
# Random Search vs. Bayesian Optimization
# ----------------------------------------
#
# In this section, we demonstrate the behaviors of random search and Bayesian optimization
# in a simple simulation environment.
# 
# Create a Reward Function for Toy Experiments
# --------------------------------------------
# 
# - Import the packages:
# 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###############################################################
# - Generate the simulated reward as a mixture of 2 gaussians:
# 
# Input Space `x = [0: 99], y = [0: 99]`.
# The rewards is a combination of 2 gaussians as shown in the following figure:
# 

def gaussian2d(x, y, x0, y0, xalpha, yalpha, A): 
    return A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2) 

x, y = np.linspace(0, 99, 100), np.linspace(0, 99, 100) 
X, Y = np.meshgrid(x, y)

Z = np.zeros(X.shape) 
ps = [(20, 70, 35, 40, 1),
      (80, 40, 20, 20, 0.7)]
for p in ps:
    Z += gaussian2d(X, Y, *p)


###############################################################
# - Visualize the reward space:
#

fig = plt.figure()
ax = fig.gca(projection='3d') 
ax.plot_surface(X, Y, Z, cmap='plasma') 
ax.set_zlim(0,np.max(Z)+2)
plt.show()


###############################################################
# Create Training Function
# ------------------------
#
#
# We can simply define an AutoTorch searchable function with a decorator `at.gargs`.
# The `reporter` is used to communicate with AutoTorch search and scheduling algorithms.

@at.args(
    x=at.Int(0, 99),
    y=at.Int(0, 99),
)
def toy_simulation(args, reporter):
    x, y = args.x, args.y
    reporter(accuracy=Z[y][x])


###############################################################
# Random Search
# -------------
#

random_scheduler = at.scheduler.FIFOScheduler(toy_simulation,
                                              resource={'num_cpus': 1, 'num_gpus': 0},
                                              num_trials=30,
                                              reward_attr="accuracy",
                                              resume=False)
random_scheduler.run()
random_scheduler.join_jobs()
print('Best config: {}, best reward: {}'.format(random_scheduler.get_best_config(), random_scheduler.get_best_reward()))

###############################################################
# Bayesian Optimization
# ---------------------
#

bayesopt_scheduler = at.scheduler.FIFOScheduler(toy_simulation,
                                                searcher='bayesopt',
                                                resource={'num_cpus': 1, 'num_gpus': 0},
                                                num_trials=30,
                                                reward_attr="accuracy",
                                                resume=False)
bayesopt_scheduler.run()
bayesopt_scheduler.join_jobs()
print('Best config: {}, best reward: {}'.format(bayesopt_scheduler.get_best_config(), bayesopt_scheduler.get_best_reward()))

###############################################################
# Compare the performance
# -----------------------
#
# Get the result history:
#

results_bayes = [v[0]['accuracy'] for v in bayesopt_scheduler.training_history.values()]
results_random = [v[0]['accuracy'] for v in random_scheduler.training_history.values()]

plt.plot(range(len(results_random)), results_random, range(len(results_bayes)), results_bayes)

