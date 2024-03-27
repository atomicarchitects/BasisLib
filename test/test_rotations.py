import BasisLib
import numpy as np
import pytest

@pytest.mark.parametrize('max_degree', [0, 1, 2, 3])
@pytest.mark.parametrize('cartesian_order', [True, False])
def test_wigner_d(
    max_degree: int, cartesian_order: bool, num: int = 10
) -> None:
  rot = BasisLib.so3.random_rotation(num=num)
  wigner_d = BasisLib.so3.wigner_d(
      rot, max_degree=max_degree, cartesian_order=cartesian_order
  )
  r = np.random.normal(size=(num, 3))  # Random vectors.
  r_rot = np.einsum('...a,...ab->...b', r, rot)  # Rotate.
  ylm = BasisLib.so3.spherical_harmonics(  # From non-rotated vectors.
      r, max_degree=max_degree, cartesian_order=cartesian_order
  )
  ylm_rot = BasisLib.so3.spherical_harmonics(  # From rotated vectors.
      r_rot, max_degree=max_degree, cartesian_order=cartesian_order
  )
  # Rotate output from non-rotated vectors.
  ylm_wigner_d = np.einsum('...a,...ab->...b', ylm, wigner_d)
  assert np.allclose(ylm_wigner_d, ylm_rot, atol=1e-5)
