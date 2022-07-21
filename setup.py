from setuptools import setup, find_packages
from os import path


setup(name = 'LearnableEnvironment',
    packages=find_packages(),
    version = '0.0.3',
    description = 'Learnable environments for model-based RL in PyTorch',
    url = 'https://github.com/rssalessio/torch-model-based-rl',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    license='MIT',
    zip_safe=False,
    python_requires='>=3.7',
)