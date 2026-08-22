"""Minimal setup.py to force platform-specific wheel tags.

setuptools does not detect bundled native shared libraries (.so/.dylib/.dll)
as binary extensions by default — it treats them as data files and produces
a `py3-none-any` wheel, which is wrong because the shared library is
platform-specific. Overriding `has_ext_modules()` forces the correct
platform tag (e.g. `py3-none-manylinux_2_28_x86_64`).
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


setup(distclass=BinaryDistribution)
