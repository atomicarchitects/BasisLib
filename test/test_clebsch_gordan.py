import re
import BasisLib
from testing import subtests
import numpy as np
import pytest

@pytest.fixture(name='expected_cg')
def fixture_expected_cg() -> np.array:
  """Clebsch-Gordant for max_degree1 = max_degree2 = max_degree3 = 2."""
  # pyformat: disable
  return np.asarray([
      [
          [1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 1]
      ],
      [
          [0, 1, 0, 0, 0, 0, 0, 0, 0],
          [np.sqrt(3)/3, 0, 0, 0, 0, 0, -np.sqrt(6)/6, 0, -np.sqrt(2)/2],
          [0, 0, 0, np.sqrt(2)/2, 0, np.sqrt(2)/2, 0, 0, 0],
          [0, 0, -np.sqrt(2)/2, 0, np.sqrt(2)/2, 0, 0, 0, 0],
          [0, 0, 0, np.sqrt(30)/10, 0, -np.sqrt(6)/6, 0, 0, 0],
          [0, 0, np.sqrt(30)/10, 0, np.sqrt(6)/6, 0, 0, 0, 0],
          [0, -np.sqrt(10)/10, 0, 0, 0, 0, 0, np.sqrt(2)/2, 0],
          [0, 0, 0, 0, 0, 0, -np.sqrt(2)/2, 0, np.sqrt(6)/6],
          [0, -np.sqrt(30)/10, 0, 0, 0, 0, 0, -np.sqrt(6)/6, 0]
      ],
      [
          [0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, -np.sqrt(2)/2, 0, np.sqrt(2)/2, 0, 0, 0],
          [np.sqrt(3)/3, 0, 0, 0, 0, 0, np.sqrt(6)/3, 0, 0],
          [0, np.sqrt(2)/2, 0, 0, 0, 0, 0, np.sqrt(2)/2, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, -np.sqrt(6)/3],
          [0, np.sqrt(30)/10, 0, 0, 0, 0, 0, -np.sqrt(6)/6, 0],
          [0, 0, np.sqrt(10)/5, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, np.sqrt(30)/10, 0, np.sqrt(6)/6, 0, 0, 0],
          [0, 0, 0, 0, np.sqrt(6)/3, 0, 0, 0, 0]
      ],
      [
          [0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, np.sqrt(2)/2, 0, np.sqrt(2)/2, 0, 0, 0, 0],
          [0, -np.sqrt(2)/2, 0, 0, 0, 0, 0, np.sqrt(2)/2, 0],
          [np.sqrt(3)/3, 0, 0, 0, 0, 0, -np.sqrt(6)/6, 0, np.sqrt(2)/2],
          [0, np.sqrt(30)/10, 0, 0, 0, 0, 0, np.sqrt(6)/6, 0],
          [0, 0, 0, 0, 0, 0, np.sqrt(2)/2, 0, np.sqrt(6)/6],
          [0, 0, 0, -np.sqrt(10)/10, 0, -np.sqrt(2)/2, 0, 0, 0],
          [0, 0, np.sqrt(30)/10, 0, -np.sqrt(6)/6, 0, 0, 0, 0],
          [0, 0, 0, np.sqrt(30)/10, 0, -np.sqrt(6)/6, 0, 0, 0]
      ],
      [
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, np.sqrt(30)/10, 0, np.sqrt(6)/6, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, np.sqrt(6)/3],
          [0, np.sqrt(30)/10, 0, 0, 0, 0, 0, -np.sqrt(6)/6, 0],
          [np.sqrt(5)/5, 0, 0, 0, 0, 0, -np.sqrt(14)/7, 0, 0],
          [0, -np.sqrt(10)/10, 0, 0, 0, 0, 0, np.sqrt(42)/14, 0],
          [0, 0, 0, 0, -np.sqrt(14)/7, 0, 0, 0, 0],
          [0, 0, 0, np.sqrt(10)/10, 0, np.sqrt(42)/14, 0, 0, 0],
          [0, 0, -np.sqrt(10)/5, 0, 0, 0, 0, 0, 0]
      ],
      [
          [0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, np.sqrt(30)/10, 0, -np.sqrt(6)/6, 0, 0, 0, 0],
          [0, np.sqrt(30)/10, 0, 0, 0, 0, 0, np.sqrt(6)/6, 0],
          [0, 0, 0, 0, 0, 0, -np.sqrt(2)/2, 0, -np.sqrt(6)/6],
          [0, np.sqrt(10)/10, 0, 0, 0, 0, 0, np.sqrt(42)/14, 0],
          [np.sqrt(5)/5, 0, 0, 0, 0, 0, np.sqrt(14)/14, 0, -np.sqrt(42)/14],
          [0, 0, 0, np.sqrt(30)/10, 0, np.sqrt(14)/14, 0, 0, 0],
          [0, 0, -np.sqrt(10)/10, 0, np.sqrt(42)/14, 0, 0, 0, 0],
          [0, 0, 0, np.sqrt(10)/10, 0, -np.sqrt(42)/14, 0, 0, 0]
      ],
      [
          [0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, -np.sqrt(10)/10, 0, 0, 0, 0, 0, -np.sqrt(2)/2, 0],
          [0, 0, np.sqrt(10)/5, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, -np.sqrt(10)/10, 0, np.sqrt(2)/2, 0, 0, 0],
          [0, 0, 0, 0, -np.sqrt(14)/7, 0, 0, 0, 0],
          [0, 0, 0, -np.sqrt(30)/10, 0, np.sqrt(14)/14, 0, 0, 0],
          [np.sqrt(5)/5, 0, 0, 0, 0, 0, np.sqrt(14)/7, 0, 0],
          [0, np.sqrt(30)/10, 0, 0, 0, 0, 0, np.sqrt(14)/14, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, -np.sqrt(14)/7]
      ],
      [
          [0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, np.sqrt(2)/2, 0, -np.sqrt(6)/6],
          [0, 0, 0, np.sqrt(30)/10, 0, -np.sqrt(6)/6, 0, 0, 0],
          [0, 0, np.sqrt(30)/10, 0, np.sqrt(6)/6, 0, 0, 0, 0],
          [0, 0, 0, -np.sqrt(10)/10, 0, np.sqrt(42)/14, 0, 0, 0],
          [0, 0, np.sqrt(10)/10, 0, np.sqrt(42)/14, 0, 0, 0, 0],
          [0, -np.sqrt(30)/10, 0, 0, 0, 0, 0, np.sqrt(14)/14, 0],
          [np.sqrt(5)/5, 0, 0, 0, 0,0, np.sqrt(14)/14, 0, np.sqrt(42)/14],
          [0, np.sqrt(10)/10, 0, 0, 0, 0, 0, np.sqrt(42)/14, 0]
      ],
          [
              [0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, -np.sqrt(30)/10, 0, 0, 0, 0, 0, np.sqrt(6)/6, 0],
              [0, 0, 0, 0, -np.sqrt(6)/3, 0, 0, 0, 0],
              [0, 0, 0, np.sqrt(30)/10, 0, np.sqrt(6)/6, 0, 0, 0],
              [0, 0, np.sqrt(10)/5, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, -np.sqrt(10)/10, 0, -np.sqrt(42)/14, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -np.sqrt(14)/7],
              [0, -np.sqrt(10)/10, 0, 0, 0, 0, 0, np.sqrt(42)/14, 0],
              [np.sqrt(5)/5, 0, 0, 0, 0, 0, -np.sqrt(14)/7, 0, 0]
          ]
      ])
  # pyformat: enable

@pytest.mark.parametrize('cartesian_order', [False, True])
def test_clebsch_gordan(
    cartesian_order: bool,
    expected_cg: np.asarray,
    max_degree: int = 2,
):
  cg = BasisLib.so3.clebsch_gordan(
      max_degree, max_degree, max_degree, cartesian_order=cartesian_order
  )
  if cartesian_order:
    p = BasisLib.so3._cartesian_permutation(max_degree)
    expected_cg = expected_cg[p, :, :][:, p, :][:, :, p]
  assert np.allclose(cg, expected_cg, atol=1e-5)


@pytest.mark.parametrize('l1', [0, 1, 2])
@pytest.mark.parametrize('l2', [0, 1, 2])
@pytest.mark.parametrize('l3', [0, 1, 2])
@pytest.mark.parametrize('cartesian_order', [False, True])
def test_clebsch_gordan_for_degrees(
    l1: int,
    l2: int,
    l3: int,
    cartesian_order: bool,
    expected_cg: np.array,
) -> None:
  cg = BasisLib.so3.clebsch_gordan_for_degrees(
      degree1=l1, degree2=l2, degree3=l3, cartesian_order=cartesian_order
  )
  expected_cg = expected_cg[
      l1**2 : (l1 + 1) ** 2, l2**2 : (l2 + 1) ** 2, l3**2 : (l3 + 1) ** 2
  ]
  if cartesian_order:
    p1 = BasisLib.so3._cartesian_permutation_for_degree(l1)
    p2 = BasisLib.so3._cartesian_permutation_for_degree(l2)
    p3 = BasisLib.so3._cartesian_permutation_for_degree(l3)
    expected_cg = expected_cg[p1, :, :][:, p2, :][:, :, p3]
  assert np.allclose(cg, expected_cg, atol=1e-5)