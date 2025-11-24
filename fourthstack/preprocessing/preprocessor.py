import json
from typing import Dict, Any, Iterable, List

from .bucketizer import Bucketizer
from .encoder import CategoricalEncoder
from .outlier_detector import OutlierDetector


class Preprocessor:
    """High-level preprocessing orchestration.

    Components:
      - Bucketizer for age/credit buckets
      - CategoricalEncoder for category fields
      - OutlierDetector for numeric clipping
    """

    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        self.bucketizer = Bucketizer()
        self.encoder = CategoricalEncoder(schema)
        self.outliers = OutlierDetector()
        self.schema = schema

    def fit_outliers(self, rows: Iterable[Dict[str, Any]]) -> None:
        """Fit outlier bounds for numeric fields using example rows."""
        numeric_fields = [
            f
            for f, r in self.schema.items()
            if r.get("type") in ("int", "integer", "float", "number")
        ]
        self.outliers.fit(rows, numeric_fields)

    def transform(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single row according to the schema.

        - Clips numeric fields using learned outlier bounds
        - Encodes categorical fields
        - Adds age_bucket and credit_bucket when source fields exist
        """
        processed: Dict[str, Any] = {}

        for field, rule in self.schema.items():
            val = row.get(field)
            ftype = rule.get("type")
            if ftype in ("int", "integer"):
                clipped = self.outliers.clip(val, field)
                processed[field] = int(clipped) if clipped is not None else None

            elif ftype in ("float", "number"):
                clipped = self.outliers.clip(val, field)
                processed[field] = float(clipped) if clipped is not None else None

            elif ftype == "category":
                # Leave None as-is
                if val is None:
                    processed[field] = None
                else:
                    processed[field] = self.encoder.encode(field, val)

            else:
                # Unknown types: copy through
                processed[field] = val

        # add bucketed fields if source values present
        if "age" in row:
            processed["age_bucket"] = self.bucketizer.bucket_age(row.get("age"))
        if "applied_credit_limit" in row:
            processed["credit_bucket"] = self.bucketizer.bucket_credit_limit(
                row.get("applied_credit_limit")
            )

        return processed

    def run(self, input_json: str, output_json: str) -> None:
        """Run preprocessing over a JSONL file (one JSON object per line)."""
        rows: List[Dict[str, Any]] = [
            json.loads(line) for line in open(input_json, "r")
        ]
        # Fit outlier bounds from data
        self.fit_outliers(rows)

        with open(output_json, "w") as out:
            for r in rows:
                pr = self.transform(r)
                out.write(json.dumps(pr) + "\n")

        print("Preprocessing complete.")
