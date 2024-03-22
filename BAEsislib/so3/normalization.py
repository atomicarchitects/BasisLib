r"""Normalizations for spherical harmonics."""

import math
from typing import Literal


valid_normalizations = ("4pi", "orthonormal", "racah", "schmidt")

Normalization = Literal[valid_normalizations]


def check_normalization_is_valid(normalization: Normalization) -> None:
    """Checks whether normalization has a valid value."""
    if normalization not in valid_normalizations:
        raise ValueError(
            f"normalization must be in {valid_normalizations}, "
            f"received '{normalization}'"
        )


def normalization_constant(normalization: Normalization, degree: int) -> float:
    """Returns the normalization constant for a given normalization."""
    check_normalization_is_valid(normalization)

    if normalization == "4pi":
        return math.sqrt(2 * degree + 1)
    elif normalization == "orthonormal":
        return math.sqrt((2 * degree + 1) / (4 * math.pi))
    else:  # 'racah' or 'schmidt'.
        return 1
