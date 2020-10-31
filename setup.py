import os
from setuptools import setup, find_packages

"""
First upload to test pypi:
    mktmpenv (Python version should not matter)
    pip install torch tqdm pandas numpy twine
    python setup.py sdist
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    
    do check if it is working properly or not.
    
    and then upload it to PyPi.
    
    enjoy!
    
    please add keywords and choose a name for your package search on pypi index first 
    to check if the name is allready taken or available.
"""


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="",  # add name of your package as the dsm name is already taken
    version="0.0.1",
    author="Chirag Nagpal",
    author_email="chiragn@cs.cmu.edu",
    maintainer="Prince Roshan",
    maintainer_email="princekrroshan01@gmail.com",
    url="https://github.com/autonlab/DeepSurvivalMachines",
    description=(
        "Python package dsm provides an API to train the Deep Survival Machines and associated models for problems in survival analysis."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPLv3+",
    packages=find_packages(exclude=["tests*"]),
    keywords=[""],  # add keywords
    python_requires=">=3.6",
    install_requires=["torch", "numpy", "pandas", "tqdm", "scikit-learn"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
)
