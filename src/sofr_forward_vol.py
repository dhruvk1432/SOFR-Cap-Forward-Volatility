"""Backward-compatible public API for the SOFR forward-vol package.

New code should import from the smaller modules in this package.  This
shim preserves the original `src.sofr_forward_vol` import path used by
the notebook and older tests.
"""

from .artifacts import *
from .constants import *
from .curve import *
from .data import *
from .inference import *
from .plotting import *
from .pricing import *
from .strategy import *
from .validation import *
