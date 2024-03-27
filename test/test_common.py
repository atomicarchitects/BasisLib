import BasisLib
import numpy as np

def test_integer_powers() -> None:
  x = np.asarray([[0.0, -1.0, 2.0, -1.5]])
  max_degree = 2
  assert np.allclose(
      BasisLib.so3.common._integer_powers(x, max_degree),
      np.asarray([
          [1.0, 1.0, 1.0, 1.0],
          [0.0, -1.0, 2.0, -1.5],
          [0.0, 1.0, 4.0, 2.25],
      ]),
  )
