import yaml
from typing import Any, Dict


class SchemaValidator:
    """Validate rows against a simple schema file.

    The schema file is expected to be YAML with a top-level `fields` mapping.
    Supported field types: `int`/`integer`, `float`/`number`, and `category`.
    """

    def __init__(self, schema_path: str):
        with open(schema_path, "r") as f:
            data = yaml.safe_load(f) or {}
        self.schema: Dict[str, Dict[str, Any]] = data.get("fields", {})
        if not isinstance(self.schema, dict):
            raise ValueError("schema 'fields' must be a mapping of field -> rules")

    def validate_row(self, row: Dict[str, Any]) -> bool:
        """Validate a single row (dict). Raises on validation errors.

        Returns True when the row passes all checks.
        """
        for field, rules in self.schema.items():
            if field not in row:
                raise ValueError(f"Missing required field: {field}")

            value = row[field]
            field_type = rules.get("type")

            if field_type in ("int", "integer"):
                # bool is subclass of int so explicitly exclude it
                if not (isinstance(value, int) and not isinstance(value, bool)):
                    raise TypeError(f"{field} must be int")
                self._check_range(field, value, rules)

            elif field_type in ("float", "number"):
                if not (
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                ):
                    raise TypeError(f"{field} must be float")
                self._check_range(field, value, rules)

            elif field_type == "category":
                allowed = rules.get("allowed")
                if allowed is None:
                    raise ValueError(f"{field} category has no 'allowed' list")
                if value not in allowed:
                    raise ValueError(f"{field}: '{value}' not in {allowed}")

            else:
                raise ValueError(
                    f"Unsupported field type '{field_type}' for field '{field}'"
                )

        return True

    def _check_range(self, field: str, value: Any, rules: Dict[str, Any]) -> None:
        if "min" in rules and value < rules["min"]:
            raise ValueError(f"{field} < min ({rules['min']})")
        if "max" in rules and value > rules["max"]:
            raise ValueError(f"{field} > max ({rules['max']})")
