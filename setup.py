import os
import subprocess
from setuptools import find_packages, setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Have a different version per commit to automatically update the package after each
# commit
with open("version.txt", "r") as file_handler:
    __version__ = file_handler.read().strip()

hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=".")
    .decode("ascii")
    .strip()
)


setup(
    name="svpg",
    author="SVPG team-SU",
    license="MIT",
    url="https://github.com/Anidwyd/pandroide-svpg.git",
    python_requires=">=3.7",
    packages=find_packages(),
    version=f"{__version__}.dev0+{hash}",
    long_description=read("README.md"),
)
