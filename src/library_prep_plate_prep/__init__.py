import sys

from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from . import (
    costs,
    geometries,
    # metrics,
    problems,
    # simulate,
    solvers,
    utils
)
from .simulate import *


try:
    dist_name = "library-prep-plate-prep"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version
    del PackageNotFoundError
