from BasisLib.config import Config
from .common import _check_degree_is_positive_or_zero
from .common import _cartesian_permutation
from .common import _cartesian_permutation_for_degree
from .clebsch_gordan_lut import generate_clebsch_gordan_lookup_table
import numpy as np

def clebsch_gordan(
    max_degree1: int,
    max_degree2: int,
    max_degree3: int,
    cartesian_order: bool = Config.cartesian_order,
) -> np.array:
    r"""Clebsch-Gordan coefficients for coupling all degrees at once.

    See the :ref:`corresponding section in the overview <CouplingIrreps>` for more
    details on coupling irreps.

    Args:
        max_degree1: Maximum degree of the first factor.
        max_degree2: Maximum degree of the second factor.
        max_degree3: Maximum degree of the tensor product.
        cartesian_order: If ``True``, Cartesian order is assumed.

    Returns:
        The values of all Clebsch-Gordan coefficients for coupling degrees up to the
        requested maximum degrees stored in an Array of shape
        ``((max_degree1+1)**2, (max_degree2+1)**2, (max_degree3+1)**2))``.

    Raises:
        ValueError: If ``max_degree1``, ``max_degree2``, or ``max_degree3`` are not
        positive or zero.
    """
    # Perform checks.

    _check_degree_is_positive_or_zero(max_degree1)
    _check_degree_is_positive_or_zero(max_degree2)
    _check_degree_is_positive_or_zero(max_degree3)


    # Load/Generate lookup table with Clebsch-Gordan coefficients.
    max_degree = max(max_degree1, max_degree2, max_degree3)
    lookup_table = generate_clebsch_gordan_lookup_table(max_degree)
    # Extract relevant slices and convert to jax array.
    cg = np.asarray(
        lookup_table['cg'][
            : (max_degree1 + 1) ** 2,
            : (max_degree2 + 1) ** 2,
            : (max_degree3 + 1) ** 2,
        ]
    )
    # Optionally reorder spherical harmonics to Cartesian order.
    if cartesian_order:
        p1 = _cartesian_permutation(max_degree1)
        p2 = _cartesian_permutation(max_degree2)
        p3 = _cartesian_permutation(max_degree3)
        cg = cg[p1, :, :][:, p2, :][:, :, p3]

    return cg


def clebsch_gordan_for_degrees(
    degree1: int,
    degree2: int,
    degree3: int,
    cartesian_order: bool = Config.cartesian_order,
) -> np.array:
    r"""Clebsch-Gordan coefficients for coupling only specific degrees.

    See also :func:`clebsch_gordan <e3x.so3.clebsch_gordan>` fore more details.

    Args:
        degree1: Degree of the first factor.
        degree2: Degree of the second factor.
        degree3: Degree of the tensor product.
        cartesian_order: If ``True``, Cartesian order is assumed.

    Returns:
        The values of the Clebsch-Gordan coefficients for coupling the requested
        degrees stored in an Array of shape
        ``(2*degree1+1, 2*degree2+1, 2*degree3+1)``.

    Raises:
        ValueError: If ``degree1``, ``degree2``, or ``degree3`` are not positive or
        zero.
    """
    # Perform checks.
    _check_degree_is_positive_or_zero(degree1)
    _check_degree_is_positive_or_zero(degree2)
    _check_degree_is_positive_or_zero(degree3)

    # Load/Generate lookup table with Clebsch-Gordan coefficients.
    max_degree = max(degree1, degree2, degree3)
    lookup_table = generate_clebsch_gordan_lookup_table(max_degree)
    # Extract relevant slices and convert to jax array.
    cg = np.asarray(
            lookup_table['cg'][
                degree1**2 : (degree1 + 1) ** 2,
                degree2**2 : (degree2 + 1) ** 2,
                degree3**2 : (degree3 + 1) ** 2,
            ]
        )
    # Optionally reorder spherical harmonics to Cartesian order.
    if cartesian_order:
        p1 = _cartesian_permutation_for_degree(degree1)
        p2 = _cartesian_permutation_for_degree(degree2)
        p3 = _cartesian_permutation_for_degree(degree3)
        cg = cg[p1, :, :][:, p2, :][:, :, p3]

    return cg