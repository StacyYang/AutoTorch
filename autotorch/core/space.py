import copy
from collections import OrderedDict
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ..utils import DeprecationHelper, EasyDict, classproperty

__all__ = ['Space', 'SimpleSpace', 'NestedSpace', 'AutoTorchObject',
           'List', 'Dict', 'Choice', 'Real', 'Int', 'Bool']

class Space(object):
    """Basic Search Space
    """
    pass

class SimpleSpace(Space):
    """Non-nested Search Space
    """
    def __repr__(self):
        reprstr = self.__class__.__name__
        if hasattr(self, 'lower') and hasattr(self, 'upper'):
            reprstr += ': lower={}, upper={}'.format(self.lower, self.upper)
        if hasattr(self, 'value'):
            reprstr += ': value={}'.format(self.value)
        return reprstr

    def get_hp(self, name):
        raise NotImplemented

    @property
    def hp(self):
        return self.get_hp(name='')

    @property
    def default(self):
        default = self._default if self._default else self.hp.default_value
        return default

    @default.setter
    def default(self, value):
        self._default = value

    @property
    def rand(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(self.hp)
        return cs.sample_configuration().get_dictionary()['']

class NestedSpace(Space):
    """Nested Search Spaces
    """
    def sample(self, **config):
        pass

    @property
    def cs(self):
        raise NotImplemented

    @property
    def kwspaces(self):
        raise NotImplemented

    @property
    def default(self):
        config = self.cs.get_default_configuration().get_dictionary()
        return self.sample(**config)

    @property
    def rand(self):
        config = self.cs.sample_configuration().get_dictionary()
        return self.sample(**config)

class AutoTorchObject(NestedSpace):
    r"""Searchable Objects,
    created by decorating customized class or function using
    :func:`autotorch.obj` or :func:`autotorch.func` decorators.
    """
    def __call__(self, *args, **kwargs):
        """Convenience method for interacting with AutoTorchObject.
        """
        if not self._inited:
            self._inited = True
            self._instance = self.init()
        return self._instance.__call__(*args, **kwargs)

    def init(self):
        """Initiate a real instance for interacting with AutoTorchObject.
        """
        config = self.cs.get_default_configuration().get_dictionary()
        return self.sample(**config)

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

    @classproperty
    def kwspaces(cls):
        return cls.__init__.kwspaces

    def sample(self):
        raise NotImplemented

    def __repr__(self):
        return 'AutoTorchObject'

class List(NestedSpace):
    r"""A Searchable List (Nested Space)

    Parameters
    ----------

    args : list
        a list of search spaces.

    Examples
    --------
    >>> sequence = at.List(
    >>>     at.Choice('conv3x3', 'conv5x5', 'conv7x7'),
    >>>     at.Choice('BatchNorm', 'InstanceNorm'),
    >>>     at.Choice('relu', 'sigmoid'),
    >>> )
    >>> sequence.rand                                                                   
    ['conv3x3', 'InstanceNorm', 'relu']
    """
    def __init__(self, *args):
        self.data = [*args]

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, data):
        self.data[index] = data

    def __len__(self):
        return len(self.data)

    def __getstate__(self):
        return self.data

    def __setstate__(self, d):
        self.data = d

    def __getattribute__(self, s):
        try:    
            x = super(List, self).__getattribute__(s)
        except AttributeError:      
            pass
        else:
            return x
        x = self.data.__getattribute__(s)
        return x

    def sample(self, **config):
        ret = []
        kwspaces = self.kwspaces
        striped_keys = [k.split('.')[0] for k in config.keys()]
        for idx, obj in enumerate(self.data):
            if isinstance(obj, NestedSpace):
                sub_config = _strip_config_space(config, prefix=str(idx))
                ret.append(obj.sample(**sub_config))
            elif isinstance(obj, SimpleSpace):
                ret.append(config[str(idx)])
            else:
                ret.append(obj)
        return ret

    @property
    def cs(self):
        cs = CS.ConfigurationSpace()
        for k, v in enumerate(self.data):
            if isinstance(v, NestedSpace):
                _add_cs(cs, v.cs, str(k))
            elif isinstance(v, Space):
                hp = v.get_hp(name=str(k))
                _add_hp(cs, hp)
        return cs

    @property
    def kwspaces(self):
        kw_spaces = OrderedDict()
        for idx, obj in enumerate(self.data):
            k = str(idx)
            if isinstance(obj, NestedSpace):
                kw_spaces[k] = obj
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}.{}'.format(k, sub_k)
                    kw_spaces[new_k] = sub_v
            elif isinstance(obj, Space):
                kw_spaces[k] = obj
        return kw_spaces

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr

class Dict(NestedSpace):
    """A Searchable Dict (Nested Space)

    Examples
    --------
    >>> g = at.Dict(
    >>>         key1=at.Choice('alpha', 'beta'),
    >>>         key2=at.Int(0, 3),
    >>>     )
    >>> g.rand                                                                           
    {'key1': 'alpha', 'key2': 1}
    """
    def __init__(self, **kwargs):
        self.data = EasyDict(kwargs)

    def __getattribute__(self, s):
        try:    
            x = super(Dict, self).__getattribute__(s)
        except AttributeError:      
            pass
        else:
            return x
        x = self.data.__getattribute__(s)
        return x

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, data):
        self.data[key] = data

    def __getstate__(self):
        return self.data

    def __setstate__(self, d):
        self.data = d

    @property
    def cs(self):
        cs = CS.ConfigurationSpace()
        for k, v in self.data.items():
            if isinstance(v, NestedSpace):
                _add_cs(cs, v.cs, k)
            elif isinstance(v, Space):
                hp = v.get_hp(name=k)
                _add_hp(cs, hp)
        return cs

    @property
    def kwspaces(self):
        kw_spaces = OrderedDict()
        for k, obj in self.data.items():
            if isinstance(obj, NestedSpace):
                kw_spaces[k] = obj
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}.{}'.format(k, sub_k)
                    kw_spaces[new_k] = sub_v
                    kw_spaces[new_k] = sub_v
            elif isinstance(obj, Space):
                kw_spaces[k] = obj
        return kw_spaces

    def sample(self, **config):
        ret = {}
        ret.update(self.data)
        kwspaces = self.kwspaces
        kwspaces.update(config)
        striped_keys = [k.split('.')[0] for k in config.keys()]
        for k, v in kwspaces.items():
            if k in striped_keys:
                if isinstance(v, NestedSpace):
                    sub_config = _strip_config_space(config, prefix=k)
                    ret[k] = v.sample(**sub_config)
                else:
                    ret[k] = v
        return ret

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr

class Choice(NestedSpace):
    """Choice Search Space (Nested Space)
    Add example for conditional space.

    Parameters
    ----------
    data : Space or python built-in objects
        the choice candidates

    Examples
    --------
    >>> a = at.Choice('a', 'b', 'c', 'd')
    >>> a.rand
    'a'
    """
    def __init__(self, *data):
        self.data = [*data]

    def __iter__(self):
        for elem in self.data:
            yield elem

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, data):
        self.data[index] = data

    def __len__(self):
        return len(self.data)

    @property
    def cs(self):
        cs = CS.ConfigurationSpace()
        if len(self.data) == 0: 
            return CS.ConfigurationSpace()
        hp = CSH.CategoricalHyperparameter(name='choice', choices=range(len(self.data)))
        _add_hp(cs, hp)
        for i, v in enumerate(self.data):
            if isinstance(v, NestedSpace):
                _add_cs(cs, v.cs, str(i))
        return cs

    def sample(self, **config):
        choice = config.pop('choice')
        if isinstance(self.data[choice], NestedSpace):
            # nested space: Choice of AutoTorchobjects
            min_config = _strip_config_space(config, prefix=str(choice))
            return self.data[choice].sample(**min_config)
        else:
            return self.data[choice]

    @property
    def kwspaces(self):
        kw_spaces = OrderedDict()
        for idx, obj in enumerate(self.data):
            if isinstance(obj, NestedSpace):
                for sub_k, sub_v in obj.kwspaces.items():
                    new_k = '{}.{}'.format(idx, sub_k)
                    kw_spaces[new_k] = sub_v
        return kw_spaces

    def __repr__(self):
        reprstr = self.__class__.__name__ + str(self.data)
        return reprstr

class Real(SimpleSpace):
    """linear search space.

    Parameters
    ----------
    lower : float
        the lower bound of the search space
    upper : float
        the upper bound of the search space
    default : float (optional)
        default value
    log : (True/False)
        search space in log scale

    Examples
    --------
    >>> learning_rate = at.Real(0.01, 0.1, log=True)
    >>> learning_rate.rand                                                               
    0.013396492756434304
    """
    def __init__(self, lower, upper, default=None, log=False):
        self.lower = lower
        self.upper = upper
        self.log = log
        self._default = default

    def get_hp(self, name):
        return CSH.UniformFloatHyperparameter(name=name, lower=self.lower, upper=self.upper,
                                              default_value=self._default, log=self.log)

class Int(SimpleSpace):
    """integer search space.

    Parameters
    ----------
    lower : int
        the lower bound of the search space
    upper : int
        the upper bound of the search space
    default : int (optional)
        default value


    Examples
    --------
    >>> range = at.Int(0, 100)
    >>> range.rand                                                                       
    82
    """
    def __init__(self, lower, upper, default=None):
        self.lower = lower
        self.upper = upper
        self._default = default

    def get_hp(self, name):
        return CSH.UniformIntegerHyperparameter(name=name, lower=self.lower, upper=self.upper,
                                                default_value=self._default)

class Bool(Choice):
    """Bool Search Space

    Examples
    --------
    >>> pretrained = at.Bool()
    >>> pretrained.rand
    False
    """
    def __init__(self):
        super(Bool, self).__init__(True, False)

def _strip_config_space(config, prefix):
    # filter out the config with the corresponding prefix
    new_config = {}
    for k, v in config.items():
        if k.startswith(prefix):
            new_config[k[len(prefix)+1:]] = v
    return new_config

def _add_hp(cs, hp):
    if hp.name in cs._hyperparameters:
        cs._hyperparameters[hp.name] = hp
    else:
        cs.add_hyperparameter(hp)

def _add_cs(master_cs, sub_cs, prefix, delimiter='.', parent_hp=None):
    new_parameters = []
    for hp in sub_cs.get_hyperparameters():
        new_parameter = copy.deepcopy(hp)
        # Allow for an empty top-level parameter
        if new_parameter.name == '':
            new_parameter.name = prefix
        elif not prefix == '':
            new_parameter.name = "%s%s%s" % (prefix, '.', new_parameter.name)
        new_parameters.append(new_parameter)
    for hp in new_parameters:
        _add_hp(master_cs, hp)

def _rm_hp(cs, k):
    if k in cs._hyperparameters:
        cs._hyperparameters.pop(k)
    for hp in cs.get_hyperparameters():
        if  hp.name.startswith("%s."%(k)):
            cs._hyperparameters.pop(hp.name)
