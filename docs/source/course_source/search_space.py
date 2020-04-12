# -*- coding: utf-8 -*-
"""
Search Space and Decorator
==========================

This tutorial explains the supported search spaces and how to use them, including
simple search spaces (Int, Real, Bool and Choice) and nested search spaces
(Choice, List, Dict). Each search space describes the set of possible values for a hyperparameter, from which the searcher will try particular values during hyperparameter optimization. AutoTorch also enables search spaces in user-defined objects using the decorator
`at.obj` and user-defined functions using the decorator `at.func`.
"""

import autotorch as at

###############################################################
# Search Space
# ------------
# Simple Search Space
# ~~~~~~~~~~~~~~~~~~~
# - Integer Space :class:`autotorch.space.Int`
#
# An integer is chosen between lower and upper value during the searcher sampling.

a = at.Int(lower=0, upper=10)
print(a)

###############################################################
# Get default value:
print(a.default)

###############################################################
# Change default value, which is the first configuration that a random searcher
# :class:`autotorch.searcher.RandomSearcher` will try:

a = at.Int(lower=0, upper=10, default=2)
print(a.default)

###############################################################
# Pick a random value.
print(a.rand)

###############################################################
# - Real Space :class:`autotorch.space.Real`
#
# A real number is chosen between lower and upper value during the
# searcher sampling.

b = at.Real(lower=1e-4, upper=1e-2)
print(b)

###############################################################
# Real space in log scale:

c = at.Real(lower=1e-4, upper=1e-2, log=True)
print(c)

###############################################################
# - Choice Space :class:`autotorch.space.Choice`
#
# Choice Space chooses one value from all the possible values during the searcher sampling.

d = at.Choice('Monday', 'Tuesday', 'Wednesday')
print(d)

###############################################################
# Nested Search Space
# ~~~~~~~~~~~~~~~~~~~
# - Choice Space :class:`autotorch.space.Choice`
#
# Choice Space can also be used as a nested search space.
# For an example, see NestedExampleObj_.
#
# - List Space :class:`autotorch.space.List`
#
# List Space returns a list of sampled results.
# In this example, the first element of the list is a Int Space sampled
# from 0 to 3, and the second element is a Choice Space sampled from the choices of `'alpha'` and `'beta'`.

f = at.List(
        at.Int(0, 3),
        at.Choice('alpha', 'beta'),
    )
print(f)

###############################################################
# Get one example configuration:
print(f.rand)

###############################################################
# - Dict Space :class:`autotorch.space.Dict`
# 
# Dict Space returns a dict of sampled results.
# Similar to List Space, the resulting configuraton of Dict is
# a dict. In this example, the value of `'key1'` is sampled from
# a Choice Space with the choices of `'alpha'` and `'beta'`,
# and the value of `'key2'` is sampled from an Int Space between
# 0 and 3.

g = at.Dict(
        key1=at.Choice('alpha', 'beta'),
        key2=at.Int(0, 3),
        key3='constant'
    )
print(g)

###############################################################
# Get one example configuration:
print(g.rand)

###############################################################
# Decorators for Searchbale Object and Customized Training Scripts
# ----------------------------------------------------------------
# In this section, we show how to insert search space into customized objects and training functions.

###############################################################
# - Searchable Space in Customized Class :func:`autotorch.obj`
#
# In AutoTorch searchable object can be returned by a user defined class with a decorator.

@at.obj(
    name=at.Choice('auto', 'gluon'),
    static_value=10,
    rank=at.Int(2, 5),
)
class MyObj:
    def __init__(self, name, rank, static_value):
        self.name = name
        self.rank = rank
        self.static_value = static_value
    def __repr__(self):
        repr = 'MyObj -- name: {}, rank: {}, static_value: {}'.format(
                self.name, self.rank, self.static_value)
        return repr
h = MyObj()
print(h)

###############################################################
# Get one example random object:
print(h.rand)


###############################################################
# .. _NestedExampleObj:
# We can also use it within a Nested Space such as :class:`autotorch.space.Choice`.
# In this example, the resulting nested space will be sampled from: 

nested = at.Choice(
        at.Dict(
                obj1='1',
                obj2=at.Choice('a', 'b'),
            ),
        MyObj(),
    )

print(nested)

###############################################################
# Get an example output:

for _ in range(5):
    result = nested.rand
    assert (isinstance(result, dict) and result['obj2'] in ['a', 'b']) or hasattr(result, 'name')
    print(result)

###############################################################
# - Searchable Space in Customized Function :func:`autotorch.obj`
#
# We can also insert a searchable space in a customized function:

@at.func(
    framework=at.Choice('mxnet', 'pytorch'),
)
def myfunc(framework):
    return framework
i = myfunc()
print(i)

###############################################################
# We can also put a searchable space inside a nested space:
#

j = at.Dict(
        a=at.Real(0, 10),
        obj1=MyObj(),
        obj2=myfunc(),
    )
print(j)

###############################################################
# - Customized Train Script Using :func:`autotorch.args`
#
# `train_func` is where to put your model training script, which takes in various keyword `args` as its hyperparameters and reports the performance of the trained model using the provided `reporter`. Here, we show a dummy train_func that simply prints these objects.
#

@at.args(
    a=at.Int(1, 10),
    b=at.Real(1e-3, 1e-2),
    c=at.Real(1e-3, 1e-2, log=True),
    d=at.Choice('a', 'b', 'c', 'd'),
    e=at.Bool(),
    f=at.List(
            at.Int(1, 2),
            at.Choice(4, 5),
        ),
    g=at.Dict(
            a=at.Real(0, 10),
            obj=MyObj(),
        ),
    h=at.Choice('test', MyObj()),
    i = myfunc(),
)
def train_fn(args, reporter):
    print('args: {}'.format(args))

###############################################################
# Create Searcher and Run with a Configuration
# --------------------------------------------
# In this section, we create a Searcher object, which orchestrates a particular hyperparameter-tuning strategy.
#
# - Create a Searcher and Sample Configuration
#
searcher = at.searcher.RandomSearcher(train_fn.cs)
config = searcher.get_config()
print(config)

###############################################################
# - Run one training job with the sampled configuration:
#
train_fn(train_fn.args, config)

