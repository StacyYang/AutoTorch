##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2019
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import io
import os
import subprocess

from setuptools import setup, find_packages
import setuptools.command.develop 
import setuptools.command.install 

cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.0.1'
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
        cwd=cwd).decode('ascii').strip()
    version += '+' + sha[:7]
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'autogluon', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is autogluon version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

def try_and_install_mxnet():
    """Install MXNet is not detected
    """
    try:
        import mxnet as mx
    except ImportError:
        print("Automatically install MXNet cpu version.")
        subprocess.check_call("pip install mxnet".split())
    finally:
        import mxnet as mx
        print("MXNet {} detected.".format(mx.__version__))

def uninstall_legacy_dask():
    has_dask = True
    try:
        import dask
    except ImportError:
        has_dask = False
    finally:
        if has_dask:
            subprocess.check_call("pip uninstall -y dask".split())
    subprocess.check_call("pip install dask[complete]==2.6.0".split())
    has_dist = True
    try:
        import distributed
    except ImportError:
        has_dist = False
    finally:
        if has_dist:
            subprocess.check_call("pip uninstall -y distributed".split())
    subprocess.check_call("pip install distributed==2.6.0".split())

# run test scrip after installation
class install(setuptools.command.install.install):
    def run(self):
        create_version_file()
        try_and_install_mxnet()
        uninstall_legacy_dask()
        setuptools.command.install.install.run(self)

class develop(setuptools.command.develop.develop):
    def run(self):
        create_version_file()
        try_and_install_mxnet()
        uninstall_legacy_dask()
        setuptools.command.develop.develop.run(self)

try:
    import pypandoc
    readme = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    readme = open('README.md').read()

MIN_PYTHON_VERSION = '>=3.6.*'

requirements = [
    'tqdm',
    'numpy',
    'scipy',
    'cython',
    'requests',
    'matplotlib',
    'tornado',
    'paramiko==2.5.0',
    'ConfigSpace==0.4.10',
    'nose',
    'graphviz',
]

setup(
    name="AutoGluon",
    version=version,
    author="AutoGluon Community",
    url="https://github.com/dmlc/AutoGluon",
    description="AutoGluon Package",
    long_description=readme,
    license='MIT',
    install_requires=requirements,
<<<<<<< HEAD
    packages=find_packages(exclude=["docs", "tests", "examples"]),
=======
    python_requires=MIN_PYTHON_VERSION,
>>>>>>> 49ef052... Bug Bash Patch (#94)
    package_data={'autogluon': [
        'LICENSE',
    ]},
    cmdclass={
        'install': install,
        'develop': develop,
    },
)
