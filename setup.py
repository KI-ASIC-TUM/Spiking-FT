#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="s-ft",
    version="0.1",
    description="Spiking S-FT implementation on SpiNNaker for FMCW radar data",
    url="https://gitlab.com/ki-asic/s-ft",
    author="Technical University of Munich. AIR",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.1.2",
    ],
    include_package_data=True,
)
