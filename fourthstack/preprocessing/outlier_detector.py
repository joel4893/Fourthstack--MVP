"""Outlier detection utilities.

Provides a small `OutlierDetector` class that computes IQR-based bounds for
fields and can clip or test values.
"""
from typing import Dict, Iterable, List, Tuple, Optional


def _percentile(values: List[float], p: float) -> float:
        """Compute the p-th percentile (0-100) of a list of numeric values.

        Uses linear interpolation between closest ranks. Expects `values` to be non-empty and sorted.
        """
        if not values:
                raise ValueError("Empty values for percentile")
        n = len(values)
        if n == 1:
                return values[0]
        # fractional rank
        rank = (p / 100.0) * (n - 1)
        lower = int(rank)
        upper = min(lower + 1, n - 1)
        weight = rank - lower
        return values[lower] * (1 - weight) + values[upper] * weight


class OutlierDetector:
        """Simple outlier detector using IQR rule.

        Usage:
                od = OutlierDetector()
                od.fit(data_rows, ['age','applied_credit_limit'])
                od.is_outlier(99, 'age')
                od.clip(120000, 'applied_credit_limit')
        """

        def __init__(self) -> None:
                # stats[field] = (low_bound, high_bound)
                self.stats: Dict[str, Tuple[float, float]] = {}

        def fit(self, data: Iterable[dict], fields: Iterable[str]) -> None:
                """Fit IQR-based bounds for each field in `fields` using `data`.

                `data` is an iterable of dict-like rows. Non-numeric or missing entries are ignored.
                """
                rows = list(data)
                for f in fields:
                        vals: List[float] = []
                        for row in rows:
                                v = row.get(f)
                                if v is None:
                                        continue
                                try:
                                        fv = float(v)
                                except (TypeError, ValueError):
                                        continue
                                vals.append(fv)

                        if not vals:
                                # no numeric data for this field; skip
                                continue

                        vals.sort()
                        q1 = _percentile(vals, 25.0)
                        q3 = _percentile(vals, 75.0)
                        iqr = q3 - q1
                        low = q1 - 1.5 * iqr
                        high = q3 + 1.5 * iqr
                        self.stats[f] = (low, high)

        def clip(self, value: float, field: str) -> Optional[float]:
                """Clip `value` to the learned bounds for `field`.

                Returns the clipped value, or None if no bounds exist for the field.
                """
                if field not in self.stats:
                        return None
                low, high = self.stats[field]
                try:
                        v = float(value)
                except (TypeError, ValueError):
                        return None
                return max(min(v, high), low)

        def is_outlier(self, value: float, field: str) -> Optional[bool]:
                """Return True if `value` is outside learned bounds for `field`.

                Returns None if no bounds exist for the field or value cannot be cast to float.
                """
                if field not in self.stats:
                        return None
                try:
                        v = float(value)
                except (TypeError, ValueError):
                        return None
                low, high = self.stats[field]
                return not (low <= v <= high)

