from setuptools import setup, find_packages

setup(
    name="fruit_fly_cs2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "matplotlib",
        "networkx",
        "tqdm",
        "pyyaml",
        "requests"
    ],
)

