import io
import os
import subprocess

from setuptools import setup, find_packages

cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.0.2'
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

MIN_PYTHON_VERSION = '>=3.6.*'

requirements = [
    'torch>=1.0.0',
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

if __name__ == '__main__':
    create_version_file()
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
        entry_points={
            'console_scripts': [
                'agremote = autotorch.scheduler.remote.cli:main',
            ]
        },
    )
