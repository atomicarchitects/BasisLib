# Copyright 2024 The e3x Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import BAEsislib
from testing import subtests
import numpy as np
import pytest


@pytest.mark.parametrize("x", [-1.5, 0.0, 1.0])
@pytest.mark.parametrize("y", [-1.5, 0.0, 1.0])
@pytest.mark.parametrize("z", [-1.5, 0.0, 1.0])
def test_spherical_harmonics(x: float, y: float, z: float) -> None:
    r = np.asarray([x, y, z])
    output = BAEsislib.so3.spherical_harmonics(
        r,
        max_degree=2,
        r_is_normalized=True,
        cartesian_order=False,
        normalization="racah",
    )
    expected = np.asarray(
        [
            1,
            y,
            z,
            x,
            np.sqrt(3) * x * y,
            np.sqrt(3) * y * z,
            (z**2 - 0.5 * (x**2 + y**2)),
            np.sqrt(3) * x * z,
            np.sqrt(3) / 2 * (x**2 - y**2),
        ]
    )
    assert np.allclose(output, expected, atol=1e-5)

@pytest.mark.parametrize('r_is_normalized', [True, False])
def test_spherical_harmonics_r_is_normalized(r_is_normalized: bool) -> None:
  r = np.asarray([1.0, 1.0, 1.0])
  expected = np.asarray([1.0, 1.0, 1.0, 1.0])
  if not r_is_normalized:
    expected [1:] /= np.sqrt(3)
  assert np.allclose(
      BAEsislib.so3.spherical_harmonics(
          r,
          max_degree=1,
          r_is_normalized=r_is_normalized,
          cartesian_order=True,
          normalization='racah',
      ),
      expected,
      atol=1e-5,
  )

@pytest.mark.parametrize('cartesian_order', [True, False])
def test_spherical_harmonics_cartesian_order(cartesian_order: bool) -> None:
  r = np.asarray([0.0, 2.0, 3.0])
  output = BAEsislib.so3.spherical_harmonics(
      r,
      max_degree=1,
      r_is_normalized=True,
      cartesian_order=cartesian_order,
      normalization='racah',
  )
  if cartesian_order:
    expected = np.asarray([1.0, 0.0, 2.0, 3.0])
  else:
    expected = np.asarray([1.0, 2.0, 3.0, 0.0])
  assert np.allclose(output, expected, atol=1e-5)

@pytest.mark.parametrize(
    'normalization', ['4pi', 'orthonormal', 'racah', 'schmidt']
)
def test_spherical_harmonics_normalization(normalization: str) -> None:
  r = np.asarray([0.0, 2.0, 3.0])
  output = BAEsislib.so3.spherical_harmonics(
      r,
      max_degree=1,
      r_is_normalized=True,
      cartesian_order=True,
      normalization=normalization,
  )
  expected = np.asarray([1.0, 0.0, 2.0, 3.0])
  if normalization == '4pi':
    expected *= np.sqrt(2 * np.asarray([0, 1, 1, 1]) + 1)
  elif normalization == 'orthonormal':
    expected *= np.sqrt((2 * np.asarray([0, 1, 1, 1]) + 1) / (4 * np.pi))
  assert np.allclose(output, expected, atol=1e-5)
