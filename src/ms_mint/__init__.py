import os
import logging
from .Mint import Mint
from . import _version

__all__ = ["__version__"]

MINT_DATA_PATH = os.path.abspath(os.path.join(__path__[0], "..", "static"))

__version__ = _version.get_versions()['version']

Mint.version = __version__

logging.info(Mint.version)



