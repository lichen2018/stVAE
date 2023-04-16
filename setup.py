from setuptools import Command, find_packages, setup

__lib_name__ = "stVAE"
__lib_version__ = "0.0.1"
__description__ = ""
__url__ = ""
__author__ = "Chen LI"
__author_email__ = ""
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "Variation auto-encoder"]
__requires__ = ["requests",]

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['stVAE'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__
)