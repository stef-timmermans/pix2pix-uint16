"""
This file contains helper functions for working with supported data types.
"""

def dtype_max(dtype: str) -> float:
    """
    Return maximum representable value for supported integer image dtypes.

    Used to convert raw integer images into normalized float tensor space.

    Args:
        dtype (str): Image dtype string (e.g. "uint8", "uint16", "uint32")

    Returns:
        float: Maximum representable value.
    """
    if dtype == "uint8":
        return 255.0
    if dtype == "uint16":
        return 65535.0
    if dtype == "uint32":
        return 4294967295.0
    raise ValueError(f"Unsupported dtype: {dtype}")
