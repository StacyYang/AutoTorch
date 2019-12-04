# coding: utf-8
# pylint: disable=wrong-import-position
"""AutoGluon: AutoML toolkit with Gluon."""
from .version import __version__

from . import scheduler, searcher, utils
from .scheduler import get_cpu_count, get_gpu_count

from .utils import *
from .core import *
