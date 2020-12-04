import glob
import os

from setuptools import find_packages
from setuptools import setup

requirements = ["numpy", "jittor"]

setup(
    name="Detectron.jittor",
    version="0.1",
    description="instance segmentation based on jittor (converted from maskrcnn-benchmark)",
    packages=find_packages(exclude=("configs", "tests",)),
    install_requires=requirements,
)
