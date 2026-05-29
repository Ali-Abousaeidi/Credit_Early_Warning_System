"""Forward-looking early-warning target construction.

The timing convention is deliberately strict:
features use data up to month t, while the label uses only months t+1 through
t+horizon.
"""

from __future__ import annotations

WARNING_HORIZON_MONTHS: int = 6
DETERIORATION_DPD_THRESHOLD: int = 30
