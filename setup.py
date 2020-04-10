import io
import os
import subprocess

from setuptools import setup, find_packages
import setuptools.command.develop 
import setuptools.command.install 

cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.0.1'
try:
    from datetime import date
    today = date.today()
    day = today.strftime("b%Y%m%d")
    version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'autotorch', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is autotorch version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

# run test scrip after installation
class install(setuptools.command.install.install):
    def run(self):
        create_version_file()
        setuptools.command.install.install.run(self)

class develop(setuptools.command.develop.develop):
    def run(self):
        create_version_file()
        setuptools.command.develop.develop.run(self)


MIN_PYTHON_VERSION = '>=3.6.*'

requirements = [
    'numpy',
    'cython',
    'requests',
    'matplotlib',
    'dask>=2.6.0',
    'tqdm>=4.38.0',
    'paramiko~=2.4',
    'tornado>=5.0.1',
    'distributed>=2.6.0',
    'ConfigSpace<=0.4.11',
    'nose',
]

setup(
    name="AutoTorch",
    version=version,
    author="AutoTorch Community",
    url="https://github.com/StacyYang/AutoTorch",
    description="AutoTorch Package",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    install_requires=requirements,
    packages=find_packages(exclude=["docs", "tests", "examples"]),
    package_data={'autotorch': [
        'LICENSE',
    ]},
    cmdclass={
        'install': install,
        'develop': develop,
    },
    entry_points={
        'console_scripts': [
            'agremote = autotorch.scheduler.remote.cli:main',
        ]
    },
)
