from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_caplet_price(vol: float, expiry: float, strike: float, forward: float, discount: float) -> float:
    """Black caplet price per unit notional and unit accrual.

    The quarterly accrual factor is omitted because it multiplies every caplet
    in the bootstrap and therefore cancels in the implied-vol inversion.
    """

    expiry = max(float(expiry), 1e-12)
    if vol <= 1e-12:
        return discount * max(forward - strike, 0.0)
    if forward <= 0 or strike <= 0:
        raise ValueError("Black pricing requires positive forward and strike rates.")
    sigma_root_t = vol * np.sqrt(expiry)
    d1 = (np.log(forward / strike) + 0.5 * sigma_root_t**2) / sigma_root_t
    d2 = d1 - sigma_root_t
    return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def normal_caplet_price(vol: float, expiry: float, strike: float, forward: float, discount: float) -> float:
    """Bachelier caplet price per unit notional and unit accrual."""

    expiry = max(float(expiry), 1e-12)
    if vol <= 1e-12:
        return discount * max(forward - strike, 0.0)
    sigma_root_t = vol * np.sqrt(expiry)
    d = (forward - strike) / sigma_root_t
    return discount * ((forward - strike) * norm.cdf(d) + sigma_root_t * norm.pdf(d))
