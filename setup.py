from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

files = ["marcia/*"]
setup(
        name = 'marcia',
        packages = ['marcia'],
        package_data = {'marcia' : files },
        version = 'beta-0.0.0',
        install_requires = ['numpy','scipy','emcee','toml'],
        description = 'Multi tasking Gaussian Process for cosmological inference',
        author = ['Balakrishna Sandeep Haridasu','Anto Idicherian Lonappan'], 
        author_email = ['mail@antolonappan.me','sandeep.haridasu@sissa.it'],
        url = 'https://github.com/antolonappan/pycachera',
        long_description=long_description,
        long_description_content_type="text/markdown",
        )
