import csv
import json
from .schema_validator import SchemaValidator


class IngestionService:
    def __init__(self, schema_path: str):
        self.validator = SchemaValidator(schema_path)

    def ingest(self, input_csv: str, output_json: str) -> None:
        """Read CSV, validate rows against schema, and write JSONL output."""
        valid_rows = []

        with open(input_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                casted = self._cast_types(row)
                # will raise on invalid rows
                self.validator.validate_row(casted)
                valid_rows.append(casted)

        with open(output_json, "w") as f:
            for r in valid_rows:
                f.write(json.dumps(r) + "\n")

        print("Ingestion complete. Rows validated:", len(valid_rows))

    def _cast_types(self, row: dict) -> dict:
        """Attempt to cast CSV string values to int/float where appropriate."""
        casted = {}
        for k, v in row.items():
            if v is None:
                casted[k] = None
                continue
            v = v.strip()
            if v == "":
                casted[k] = None
                continue

            # Try integer first (without decimal point)
            try:
                if "." not in v:
                    casted[k] = int(v)
                    continue
            except ValueError:
                pass

            # Try float
            try:
                casted[k] = float(v)
                continue
            except ValueError:
                pass

            # Fallback to original string
            casted[k] = v

        return casted
