import numpy as np
from BasisLib.config import Config
from .common import _integer_powers
from .common import _cartesian_permutation_wigner_d_entries
from .wigner_d_lookup_lut import _generate_wigner_d_lookup_table

def _check_rotation_matrix_shape(rot: np.array) -> None:
  """Helper function to check the shape of a rotation matrix.

  Args:
    rot: Array that should be checked for the correct shape.

  Raises:
    ValueError: If the shape is invalid for a rotation matrix.
  """
  if rot.shape[-2:] != (3, 3):
    raise (
        ValueError(
            'rotation matrices must have shape (..., 3, 3), received '
            f'shape {rot.shape}'
        )
    )

def random_rotation(
    perturbation: float = 1.0,
    num: int = 1,  # When num=1, leading dimension is automatically squeezed.
):
  r"""Samples a random :math:`3\times3` rotation matrix.

  Samples random :math:`3\times3` rotation matrices from :math:`\mathrm{SO(3)}`.
  The ``perturbation`` parameter controls how strongly random points on a sphere
  centered on the origin are perturbed by the rotation. For
  ``perturbation=1.0``, any point on the sphere is rotated to any other point on
  the sphere with equal probability. If ``perturbation<1.0``, returned rotation
  matrices are biased to identity matrices. For example, with
  ``perturbation=0.5``, a point on the sphere is rotated to any other point on
  the same hemisphere with equal probability.

  Args:
    perturbation: A value between 0.0 and 1.0 that determines the perturbation.
    num: Number of returned rotation matrices.

  Returns:
    An Array of shape :math:`(\mathrm{num}, 3, 3)` or :math:`(3, 3)` (if num =
    1) representing random :math:`3\times3` rotation matrices.
  """

  # Check that perturbation is a meaningful value.
  if not 0.0 <= perturbation <= 1.0:
    raise ValueError(
        f'perturbation must be between 0.0 and 1.0, received {perturbation}'
    )
  # Draw random numbers and transform them.
  twopi = 2 * np.pi
  u = np.random.uniform(size=(num, 3))
  sqrt1 = np.sqrt(1 - u[..., 0])
  sqrt2 = np.sqrt(u[..., 0])
  angl1 = twopi * u[..., 1]
  angl2 = twopi * u[..., 2]
  # Construct random quaternion.
  r = sqrt1 * np.sin(angl1)
  i = sqrt1 * np.cos(angl1)
  j = sqrt2 * np.sin(angl2)
  k = sqrt2 * np.cos(angl2)
  # Perturbation (Slerp starting from identity quaternion).
  flip = r < 0  # Flip sign if r < 0 (always take the shorter route).
  r = np.where(flip, -r, r)
  i = np.where(flip, -i, i)
  j = np.where(flip, -j, j)
  k = np.where(flip, -k, k)
  phi = np.arccos(r)
  sinphi = np.sin(phi)
  # Prevent division by zero.
  zeromask = np.abs(sinphi) < 1e-9
  f1 = np.where(
      zeromask, 1 - perturbation, np.sin((1 - perturbation) * phi) / sinphi
  )
  f2 = np.where(zeromask, perturbation, np.sin(perturbation * phi) / sinphi)
  r, i, j, k = f1 + f2 * r, f2 * i, f2 * j, f2 * k
  # Construct rotation matrix.
  i2, j2, k2 = i * i, j * j, k * k
  ij, ik, jk, ir, jr, kr = i * j, i * k, j * k, i * r, j * r, k * r
  row1 = np.stack((1 - 2 * (j2 + k2), 2 * (ij - kr), 2 * (ik + jr)), axis=-1)
  row2 = np.stack((2 * (ij + kr), 1 - 2 * (i2 + k2), 2 * (jk - ir)), axis=-1)
  row3 = np.stack((2 * (ik - jr), 2 * (jk + ir), 1 - 2 * (i2 + j2)), axis=-1)
  rot = np.squeeze(np.stack((row1, row2, row3), axis=-1))
  return rot

def wigner_d(
    rot: np.array,
    max_degree: int,
    cartesian_order: bool = Config.cartesian_order,
) -> np.array:
  r"""Wigner-D matrix corresponding to a given :math:`3\times3` rotation matrix.

  Transform :math:`3\times3` rotation matrices to
  :math:`(\mathrm{max\_degree}+1)^2 \times (\mathrm{max\_degree}+1)^2` Wigner-D
  matrices that can be used to rotate irreducible representations of
  :math:`\mathrm{SO}(3)`.

  Args:
    rot: An Array of shape :math:`(\dots, 3, 3)` representing :math:`3\times3`
      rotation matrices.
    max_degree: Maximum degree of the irreducible representations.
    cartesian_order: If True, Cartesian order is assumed.

  Returns:
    An Array of shape
    :math:`(\dots, (\mathrm{max\_degree}+1)^2,(\mathrm{max\_degree}+1)^2)`
    representing Wigner-D matrices corresponding to the input rotations.

  Raises:
    ValueError: If ``rot`` does not have shape `(..., 3, 3)`.
  """
  _check_rotation_matrix_shape(rot)  # Raise if shape is not (..., 3, 3).

  # Load/Generate lookup table and convert to jax arrays.
  lookup_table = _generate_wigner_d_lookup_table(max_degree)
  cm = lookup_table['cm']
  ls = lookup_table['ls']
  # Optionally reorder to Cartesian order.
  if cartesian_order:
    cm = cm[:, _cartesian_permutation_wigner_d_entries(max_degree)]

  # Calculate all relevant monomials of the rotation matrix entries.
  # Note: This is done via integer powers and indexing on purpose! Using
  # jnp.power or the "**"-operator for this operation leads to NaNs in the
  # gradients for some inputs (jnp.power is not NaN-safe).
  rot_powers = _integer_powers(rot.reshape(*rot.shape[:-2], 1, -1), max_degree)
  monomials = (
      rot_powers[..., 0][..., ls[:, 0]]  #   R_00**l_00.
      * rot_powers[..., 1][..., ls[:, 1]]  # R_01**l_01.
      * rot_powers[..., 2][..., ls[:, 2]]  # R_02**l_02.
      * rot_powers[..., 3][..., ls[:, 3]]  # R_10**l_10.
      * rot_powers[..., 4][..., ls[:, 4]]  # R_11**l_11.
      * rot_powers[..., 5][..., ls[:, 5]]  # R_12**l_12.
      * rot_powers[..., 6][..., ls[:, 6]]  # R_20**l_20.
      * rot_powers[..., 7][..., ls[:, 7]]  # R_21**l_21.
      * rot_powers[..., 8][..., ls[:, 8]]  # R_22**l_22.
  )

  # Entries of the Wigner-D matrix are linear combinations of the monomials.
  dmat_entries = np.matmul(monomials, cm)

  # Assemble Wigner-D matrix.
  dmat = np.zeros_like(  # Initialize Wigner-D matrix to zeros.
      rot, shape=(*rot.shape[:-2], (max_degree + 1) ** 2, (max_degree + 1) ** 2)
  )
  for l in range(max_degree + 1):  # Set entries of non-zero blocks on diagonal.
    i = l**2  # Start index Wigner-D slice.
    j = (l + 1) ** 2  # Stop index Wigner-D slice.
    b = ((l + 1) * (2 * l + 1) * (2 * l + 3)) // 3  # Start index entries.
    a = b - (2 * l + 1) ** 2  # Stop index entries.
    num = 2 * l + 1  # Matrix block has shape (..., 2*l+1, 2*l+1).
    dmat[..., i:j, i:j] = dmat_entries[..., a:b].reshape((*rot.shape[:-2], num, num))
  return dmat
