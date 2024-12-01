import sys

import pandas as pd
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    dist_name = "library-prep-plate-prep"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


pd.set_option('future.no_silent_downcasting', True)