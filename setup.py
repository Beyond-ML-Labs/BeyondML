from setuptools import setup
from mann import __version__

with open('requirements.txt', 'r') as f:
    install_requires = [line for line in f.read().splitlines() if len(line) > 0]

setup(
    name = 'mann',
    version = __version__,
    url = 'https://github.com/AISquaredInc/mann',
    packages = ['mann', 'mann.layers', 'mann.utils'],
    author = 'The AI Squared Team',
    author_email = 'mann@squared.ai',
    description = 'Package containing utilities for implementing RSN2/MANN',
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    license = 'Apache 2.0',
    install_requires = install_requires
)
