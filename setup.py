from setuptools import setup
from beyondml import __version__

with open('requirements.txt', 'r') as f:
    requirements = [
        line for line in f.read().splitlines() if line != ''
    ]

setup(
    name='beyondml',
    version=__version__,
    url='https://github.com/Beyond-ML-Labs/BeyondML',
    packages=['beyondml', 'beyondml.tflow', 'beyondml.tflow.layers',
              'beyondml.tflow.utils', 'beyondml.pt', 'beyondml.pt.layers',
              'beyondml.pt.utils'],
    author='The AI Squared Team',
    author_email='mann@squared.ai',
    description='Package containing utilities for implementing RSN2/MANN',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    install_requires=requirements
)
