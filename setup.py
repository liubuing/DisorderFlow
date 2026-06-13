from setuptools import setup, find_packages

setup(
    name="disorderflow",
    version="1.0.0",
    description="disorderflow: High-Fidelity Fixed-Backbone Antibody Design via Discrete Bayesian Flow Networks",
    author="Yue Hu Lab",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "biopython",
        "pyyaml",
        "easydict",
        "tensorboard"
    ],
)
