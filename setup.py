from setuptools import setup, find_packages
from os import path


setup(name = 'LearnableEnvironment',
    packages=find_packages(),
    version = '0.1.7',
    description = 'Learnable environments for model-based RL in PyTorch',
    url = 'hhttps://github.com/rssalessio/torch-rl-learnable-environment',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    install_requires=['numpy', 'pydantic', 'scikit-learn', 'gym', 'matplotlib', 'torch', 'mujoco'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)