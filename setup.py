import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="dsm", 
    version="0.0.1",
    maintainer=["Chirag Nagpal","Prince Roshan"],
    maintainer_email=["chiragn@cs.cmu.edu","princekrroshan01@gmail.com"],
    url="https://github.com/autonlab/DeepSurvivalMachines",
    description=(
        "Python package dsm provides an API to train the Deep Survival Machines and associated models for problems in survival analysis."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPLv3+",
    packages=find_packages(exclude=["tests*"]),
    keywords=["dsm"],
    python_requires=">=3.6",
    install_requires=["torch", "numpy", "pandas", "tqdm", "scikit-learn"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
)
