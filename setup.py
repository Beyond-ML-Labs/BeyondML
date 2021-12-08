from setuptools import setup
from mann import __version__

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
    install_requires = [
        'tensorflow>=2.4'
    ]
)
