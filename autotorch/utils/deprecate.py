import warnings
from warnings import warn

__all__ = ['AutoTorchEarlyStop', 'AutoTorchWarning', 'make_deprecate',
           'DeprecationHelper']

class AutoTorchEarlyStop(ValueError):
    pass

class AutoTorchWarning(DeprecationWarning):
    pass

warnings.simplefilter('once', AutoTorchWarning)

def make_deprecate(meth, old_name):
    """TODO Add Docs
    """
    new_name = meth.__name__
    def deprecated_init(*args, **kwargs):
        warn("autotorch.{} is now deprecated in favor of autotorch.{}."
             .format(old_name, new_name), AutoTorchWarning)
        return meth(*args, **kwargs)

    deprecated_init.__doc__ = r"""
    {old_name}(...)
    .. warning::
        This method is now deprecated in favor of :func:`autotorch.{new_name}`. \
    See :func:`~autotorch.{new_name}` for details.""".format(
        old_name=old_name, new_name=new_name)
    deprecated_init.__name__ = old_name
    return deprecated_init


class DeprecationHelper(object):
    """TODO Add Docs
    """
    def __init__(self, new_class, new_name):
        self.new_class = new_class
        self.new_name = new_class.__name__
        self.old_name = new_name

    def _warn(self):
        warn("autotorch.{} is now deprecated in favor of autotorch.{}." \
             .format(self.old_name, self.new_name), AutoTorchWarning)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_class(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.new_class, attr)
