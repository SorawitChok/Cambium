"""
Exception classes for Cambium.
"""


class CambiumError(Exception):
    """Base exception for Cambium."""

    pass


class BlockValidationError(CambiumError):
    """Custom block failed validation."""

    def __init__(self, block_idx: int, reason: str):
        self.block_idx = block_idx
        self.reason = reason
        super().__init__(f"Block {block_idx} validation failed: {reason}")


class ShapeMismatchError(CambiumError):
    """Block output shape doesn't match input shape."""

    def __init__(self, expected, got, block_idx: int):
        self.expected = expected
        self.got = got
        self.block_idx = block_idx
        super().__init__(f"Block {block_idx} output shape {got} doesn't match expected {expected}")


class ConfigMismatchError(CambiumError):
    """Block requires config keys that model doesn't have."""

    def __init__(self, missing_keys: list, available_keys: list):
        self.missing_keys = missing_keys
        self.available_keys = available_keys
        super().__init__(f"Missing config keys: {missing_keys}. Available: {available_keys}")


class ExpansionError(CambiumError):
    """Error during model expansion."""

    pass


class GraftingError(CambiumError):
    """Error while loading or grafting a block from a remote model."""

    pass


class DataError(CambiumError):
    """Error while loading, parsing, or tokenizing training data."""

    pass
