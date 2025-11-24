class CategoricalEncoder:
    """Simple categorical encoder that maps allowed categories to integer ids.

    Example:
            enc = CategoricalEncoder(schema)
            enc.encode('occupation_type', 'salaried')  # -> 0
    """

    def __init__(self, schema: dict) -> None:
        self.mappings: dict = {}
        for field, rules in schema.items():
            if rules.get("type") == "category":
                allowed = rules.get("allowed", [])
                # Preserve order from schema; build mapping value->index
                self.mappings[field] = {val: i for i, val in enumerate(allowed)}

    def encode(self, field: str, value):
        """Encode a single categorical value for `field`.

        Raises KeyError if field or value is unknown.
        """
        if field not in self.mappings:
            raise KeyError(f"Field '{field}' is not a categorical field in encoder")
        mapping = self.mappings[field]
        if value not in mapping:
            raise KeyError(
                f"Value '{value}' not in allowed categories for field '{field}'"
            )
        return mapping[value]

    def encode_row(self, row: dict) -> dict:
        """Return a copy of `row` with categorical fields encoded to integers."""
        out = dict(row)
        for field in self.mappings:
            if field in out:
                out[field] = self.encode(field, out[field])
        return out
