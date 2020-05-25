import copy
import logging
import argparse
import functools
from collections import OrderedDict
import numpy as np
import multiprocessing as mp
import ConfigSpace as CS

from .space import *
from .space import _add_hp, _add_cs, _rm_hp, _strip_config_space
from ..utils import EasyDict as ezdict
from ..utils.deprecate import make_deprecate

__all__ = ['args', 'obj', 'func', 'sample_config']

logger = logging.getLogger(__name__)

def sample_config(args, config):
    args = copy.deepcopy(args)
    striped_keys = [k.split('.')[0] for k in config.keys()]
    if isinstance(args, (argparse.Namespace, argparse.ArgumentParser)):
        args_dict = vars(args)
    else:
        args_dict = args
    for k, v in args_dict.items():
        # handle different type of configurations
        if k in striped_keys:
            if isinstance(v, NestedSpace):
                sub_config = _strip_config_space(config, prefix=k)
                args_dict[k] = v.sample(**sub_config)
            else:
                if '.' in k: continue
                args_dict[k] = config[k]
        elif isinstance(v, AutoTorchObject):
            args_dict[k] = v.init()
    return args

class _autotorch_method(object):
    SEED = mp.Value('i', 0)
    LOCK = mp.Lock()
    def __init__(self, f):
        self.f = f
        self.args = ezdict()
        functools.update_wrapper(self, f)

    def __call__(self, args, config={}, **kwargs):
        new_config = copy.deepcopy(config)
        self._rand_seed()
        args = sample_config(args, new_config)
        from ..scheduler.reporter import FakeReporter
        if 'reporter' not in kwargs:
            logger.debug('Creating FakeReporter for test purpose.')
            kwargs['reporter'] = FakeReporter()

        output = self.f(args, **kwargs)
        logger.debug('Reporter Done!')
        kwargs['reporter'](done=True)
        return output
 
    def register_args(self, default={}, **kwvars):
        if isinstance(default, (argparse.Namespace, argparse.ArgumentParser)):
            default = vars(default)
        self.kwvars = {}
        self.args = ezdict()
        self.args.update(default)
        self.update(**kwvars)

    def update(self, **kwargs):
        """For searcher support ConfigSpace
        """
        self.kwvars.update(kwargs)
        for k, v in self.kwvars.items():
            if isinstance(v, (NestedSpace)):
                self.args.update({k: v})
            elif isinstance(v, Space):
                hp = v.get_hp(name=k)
                self.args.update({k: hp.default_value})
            else:
                self.args.update({k: v})

    @property
    def cs(self):
        cs = CS.ConfigurationSpace()
        for k, v in self.kwvars.items():
            if isinstance(v, NestedSpace):
                _add_cs(cs, v.cs, k)
            elif isinstance(v, Space):
                hp = v.get_hp(name=k)
                _add_hp(cs, hp)
            else:
                _rm_hp(cs, k)
        return cs

    @property
    def kwspaces(self):
        """For RL searcher/controller
        """
        kw_spaces = OrderedDict()
        for k, v in self.kwvars.items():
            if isinstance(v, NestedSpace):
                if isinstance(v, Choice):
                    kw_spaces['{}.choice'.format(k)] = v
                for sub_k, sub_v in v.kwspaces.items():
                    new_k = '{}.{}'.format(k, sub_k)
                    kw_spaces[new_k] = sub_v
            elif isinstance(v, Space):
                kw_spaces[k] = v
        return kw_spaces

    def _rand_seed(self):
        _autotorch_method.SEED.value += 1
        np.random.seed(_autotorch_method.SEED.value)

    def __repr__(self):
        return repr(self.f)


def args(default={}, **kwvars):
    r"""Decorator for customized training script, registering arguments or searchable spaces
    to the decorated function. The arguments should be python built-in objects,
    autotorch objects (see :func:`autotorch.obj` .), or autotorch search spaces
    (:class:`autotorch.space.Int`, :class:`autotorch.space.Real` ...).

    Examples
    --------
    >>> @at.args(batch_size=10, lr=at.Real(0.01, 0.1))
    >>> def my_train(args):
    ...     print('Batch size is {}, LR is {}'.format(args.batch_size, arg.lr))

    """
    kwvars['_default_config'] = default
    def registered_func(func):
        @_autotorch_method
        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            return func(*args, **kwargs)

        default = kwvars['_default_config']
        wrapper_call.register_args(default=default, **kwvars)
        return wrapper_call

    return registered_func


def func(**kwvars):
    """Register args or searchable spaces to the functions.

    Returns
    -------
    instance of :class:`autotorch.space.AutoTorchObject`:
        a lazy init object, which allows distributed training.

    Examples
    --------
    >>> torchvision.models as models
    >>> 
    >>> @at.func(pretrained=at.Bool())
    >>> def resnet18(pretrained):
    ...     return models.resnet18(pretrained=pretrained)
    """
    def _autotorch_kwargs_func(**kwvars):
        def registered_func(func):
            kwspaces = OrderedDict()
            @functools.wraps(func)
            def wrapper_call(*args, **kwargs):
                _kwvars = copy.deepcopy(kwvars)
                _kwvars.update(kwargs)
                for k, v in _kwvars.items():
                    if isinstance(v, NestedSpace):
                        kwspaces[k] = v
                        kwargs[k] = v
                    elif isinstance(v, Space):
                        kwspaces[k] = v
                        hp = v.get_hp(name=k)
                        kwargs[k] = hp.default_value
                    else:
                        kwargs[k] = v
                return func(*args, **kwargs)
            wrapper_call.kwspaces = kwspaces
            return wrapper_call
        return registered_func

    def registered_func(func):
        class autotorchobject(AutoTorchObject):
            @_autotorch_kwargs_func(**kwvars)
            def __init__(self, *args, **kwargs):
                self.func = func
                self.args = args
                self.kwargs = kwargs
                self._inited = False

            def sample(self, **config):
                kwargs = copy.deepcopy(self.kwargs)
                kwspaces = copy.deepcopy(autotorchobject.kwspaces)
                for k, v in kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], NestedSpace):
                        sub_config = _strip_config_space(config, prefix=k)
                        kwargs[k] = kwspaces[k].sample(**sub_config)
                    elif k in config:
                        kwargs[k] = config[k]
                        
                return self.func(*self.args, **kwargs)

        @functools.wraps(func)
        def wrapper_call(*args, **kwargs):
            _kwvars = copy.deepcopy(kwvars)
            _kwvars.update(kwargs)
            agobj = autotorchobject(*args, **kwargs)
            agobj.kwvars = _kwvars
            return agobj
        return wrapper_call
    return registered_func

def obj(**kwvars):
    """Register args or searchable spaces to the class.

    Returns
    -------
    instance of :class:`autotorch.space.AutoTorchObject`:
        a lazy init object, which allows distributed training.

    Examples
    --------
    >>> import autotorch as at
    >>> import torch
    >>> @at.obj(
    >>>     lr=at.Real(1e-4, 1e-1, log=True),
    >>>     weight_decay=at.Real(1e-4, 1e-1),
    >>> )
    >>> class Adam(torch.optim.Adam):
    >>>     pass

    """
    def _autotorch_kwargs_obj(**kwvars):
        def registered_func(func):
            kwspaces = OrderedDict()
            @functools.wraps(func)
            def wrapper_call(*args, **kwargs):
                kwvars.update(kwargs)
                for k, v in kwvars.items():
                    if isinstance(v, NestedSpace):
                        kwspaces[k] = v
                        kwargs[k] = v
                    elif isinstance(v, Space):
                        kwspaces[k] = v
                        hp = v.get_hp(name=k)
                        kwargs[k] = hp.default_value
                    else:
                        kwargs[k] = v
                return func(*args, **kwargs)
            wrapper_call.kwspaces = kwspaces
            wrapper_call.kwvars = kwvars
            return wrapper_call
        return registered_func

    def registered_class(Cls):
        class autotorchobject(AutoTorchObject):
            @_autotorch_kwargs_obj(**kwvars)
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self._inited = False

            def sample(self, **config):
                kwargs = copy.deepcopy(self.kwargs)
                kwspaces = copy.deepcopy(autotorchobject.kwspaces)
                for k, v in kwargs.items():
                    if k in kwspaces and isinstance(kwspaces[k], NestedSpace):
                        sub_config = _strip_config_space(config, prefix=k)
                        kwargs[k] = kwspaces[k].sample(**sub_config)
                    elif k in config:
                        kwargs[k] = config[k]

                args = self.args
                return Cls(*args, **kwargs)

            def __repr__(self):
                return 'AutoTorchObject -- ' + Cls.__name__

        autotorchobject.kwvars = autotorchobject.__init__.kwvars
        autotorchobject.__doc__ = Cls.__doc__
        autotorchobject.__name__ = Cls.__name__
        return autotorchobject

    return registered_class
