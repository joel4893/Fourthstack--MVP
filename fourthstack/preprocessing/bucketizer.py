"""Bucketizing utilities.

Provides a `Bucketizer` class for bucketing common numeric fields.
"""
from typing import List, Optional, Tuple


class Bucketizer:
        """Bucketizer for age and credit limit fields.

        Attributes:
                age_bins: list of (low, high) inclusive age buckets
                credit_bins: list of (low, high) inclusive credit limit buckets
        """

        def __init__(self) -> None:
                self.age_bins: List[Tuple[int, int]] = [
                        (18, 24),
                        (25, 30),
                        (31, 40),
                        (41, 50),
                        (51, 70),
                ]

                self.credit_bins: List[Tuple[int, int]] = [
                        (0, 50000),
                        (50000, 200000),
                        (200000, 500000),
                        (500000, 1000000),
                ]

        def bucket_age(self, age: Optional[float]) -> Optional[int]:
                """Return the index of the age bucket containing `age`, or None."""
                if age is None:
                        return None
                try:
                        a = float(age)
                except (TypeError, ValueError):
                        return None
                for i, (low, high) in enumerate(self.age_bins):
                        if low <= a <= high:
                                return i
                return None

        def bucket_credit_limit(self, limit: Optional[float]) -> Optional[int]:
                """Return the index of the credit limit bucket containing `limit`, or None."""
                if limit is None:
                        return None
                try:
                        v = float(limit)
                except (TypeError, ValueError):
                        return None
                for i, (low, high) in enumerate(self.credit_bins):
                        if low <= v <= high:
                                return i
                return None
