from setuptools import setup, find_packages
import numpy

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='marcia',
    packages=find_packages(),
    version='0.1.0.dev0',
    install_requires=['numpy', 'scipy', 'emcee', 'toml', 'tqdm'],
    description='Multi tasking Gaussian Process for cosmological inference',
    author='Anto Idicherian Lonappan, Balakrishna Sandeep Haridasu',
    author_email='mail@antolonappan.me, sandeep.haridasu@sissa.it',
    url='https://github.com/antolonappan/marcia',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'marcia': ['params.ini','GPconfig.ini','constants.ini','Data/Pantheon+/*.dat', 'Data/Pantheon+/*.cov']},
    include_package_data=True,
)
