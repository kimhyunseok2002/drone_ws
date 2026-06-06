## catkin python package setup (used by catkin_python_setup() in CMakeLists.txt)
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['drone_practice'],
    package_dir={'': 'src'},
)

setup(**setup_args)
