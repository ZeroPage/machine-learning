from setuptools import setup, find_packages

setup(
    name='neural-style',
    version='0.1',
    description='Implementation of neural style',

    packages=find_packages(exclude=[]),
    install_requires=['tensorflow', 'numpy']
)
