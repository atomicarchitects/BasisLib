from BasisLib.config import Config
from BasisLib.ops import normalize
from .common import _check_degree_is_positive_or_zero
from .common import _integer_powers
from .common import _cartesian_permutation
from .normalization import normalization_constant
from .normalization import Normalization
from .spherical_harmonics_lut import generate_spherical_harmonics_lookup_table
import numpy as np


def spherical_harmonics(
    r: np.array,
    max_degree: int,
    r_is_normalized: bool = False,
    cartesian_order: bool = Config.cartesian_order,
    normalization: Normalization = Config.normalization,
) -> np.array:
    r"""Real Cartesian spherical harmonics :math:`Y_\ell^m(\vec{r})`.

  Evaluates :math:`Y_\ell^m(\vec{r})` for all :math:`\ell=0,\dots,L` and
  :math:`m=-\ell,\dots,\ell` with :math:`L` = ``max_degree``. The
  `spherical harmonics <https://en.wikipedia.org/wiki/Spherical_harmonics>`_ are
  basis functions for irreducible representations of :math:`\mathrm{SO}(3)`. In
  total, there are :math:`(L+1)^2` spherical harmonics for a given :math:`L`.
  For example, these are all spherical harmonics for :math:`L=3` (blue:
  positive, red: negative, arrows show the :red:`x`-, :green:`y`-, and
  :blue:`z`-axes, click & drag to rotate):

  .. raw:: html

   <iframe src="../_static/spherical_harmonics.html" width="670" height="670"
   frameBorder="0" scrolling="no">spherical harmonics up to degree 3</iframe>

  In general, the real Cartesian spherical harmonics are given by

  .. math::
    Y_{\ell}^{m}(\vec{r}) = \mathcal{N}\begin{cases}
    \sqrt{2}\cdot \Pi_\ell^{\lvert m\rvert}(z) \cdot A_{\lvert m \rvert}(x,y)
    & m < 0 \\
    \Pi_\ell^{0}(z)
    & m = 0 \\
    \sqrt{2}\cdot \Pi_\ell^{m}(z) \cdot B_m(x,y)
    & m > 0 \\
    \end{cases}
  .. math::
    A_{m}(x,y) = \sum_{k=0}^{ m}\binom{m}{k}x^{k} y^{m-k}
      \sin\left(\frac{\pi}{2}(m-k)\right)
  .. math::
    B_{m}(x,y) = \sum_{k=0}^{m}\binom{m}{k}x^{k} y^{m-k}
      \cos\left(\frac{\pi}{2}(m-k)\right)
  .. math::
    \Pi_{\ell}^{m}(z) = \sqrt{\frac{(\ell-m)!}{(\ell+m)!}}
      \sum_{k=0}^{\lfloor(\ell-m)/2\rfloor} \ \frac{(-1)^k}{2^\ell}
      \binom{\ell}{k} \binom{2\ell-2k}{\ell}\frac{(\ell-2k)!}{(\ell-2k-m)!}
      r^{2k-\ell}z^{\ell-2k-m}

  with :math:`\vec{r}=[x\ y\ z]^\intercal \in \mathbb{R}^3` and
  :math:`r = \lVert \vec{r} \rVert`. Here, :math:`\mathcal{N}` is a
  normalization constant that depends on the chosen normalization scheme. When
  ``normalization`` is ``'racah'`` or ``'schmidt'`` Racah's normalization (also
  known as Schmidt's semi-normalization) is used (the integral runs over the
  surface of the unit sphere :math:`\Omega`):

  .. math::
    \mathcal{N} = 1 \qquad
    \int_{\Omega} Y_\ell^m(\vec{r}) Y_{\ell'}^{m'}(\vec{r}) d\Omega =
      \frac{4\pi}{2\ell+1}\delta_{\ell\ell'}\delta_{mm'}\,,

  when ``normalization`` is ``'4pi'``:

  .. math::
    \mathcal{N} = \sqrt{2\ell+1} \qquad
    \int_{\Omega} Y_\ell^m(\vec{r}) Y_{\ell'}^{m'}(\vec{r}) d\Omega =
      4\pi\delta_{\ell\ell'}\delta_{mm'}\,,

  and when ``normalization`` is ``'orthonormal'``:

  .. math::
    \mathcal{N} = \sqrt{\frac{2\ell+1}{4\pi}} \qquad
    \int_{\Omega} Y_\ell^m(\vec{r}) Y_{\ell'}^{m'}(\vec{r}) d\Omega =
      \delta_{\ell\ell'}\delta_{mm'}\,.

  Args:
    r: Array of shape ``(..., 3)`` containing Cartesian vectors :math:`\vec{r}`.
    max_degree: Maximum degree :math:`L` of the spherical harmonics.
    r_is_normalized: If True, :math:`\vec{r}` is assumed to be already
      normalized.
    cartesian_order: If ``True``, spherical harmonics are returned in Cartesian
      order.
    normalization: Which normalization is used for the spherical harmonics.

  Returns:
    The values :math:`Y_\ell^m(\vec{r})` of all spherical harmonics up to
    :math:`\ell` = ``max_degree``. Values are returned in an Array of shape
    ``(..., (max_degree+1)**2)`` ordered
    :math:`[Y_{0}^{0}\ Y_{1}^{1}\ Y_{1}^{-1}\ Y_{1}^{0}\ Y_{2}^{2}\ \cdots]`.
    If ``cartesian_order = False``, values are ordered
    :math:`[Y_{0}^{0}\ Y_{1}^{-1}\ Y_{1}^{0}\ Y_{1}^{1}\ Y_{2}^{-2}\ \cdots]`
    instead.

  Raises:
    ValueError: If ``r`` has an invalid shape (not a 3-vector), ``max_degree``
    is not positive or zero, or ``normalization`` has an invalid value.
  """
    # Perform checks.
    if r.shape[-1] != 3:
        raise ValueError(f"r must have shape (..., 3), received shape {r.shape}")
    _check_degree_is_positive_or_zero(max_degree)

    # Load/Generate lookup table and convert to jax array.
    lookup_table = generate_spherical_harmonics_lookup_table(max_degree)
    cm = lookup_table["cm"]
    ls = lookup_table["ls"]
    # Apply normalization constants.
    for l in range(max_degree + 1):
        cm[:, l**2 : (l + 1) ** 2] *= normalization_constant(normalization, l)

    # Optionally reorder spherical harmonics to Cartesian order.
    if cartesian_order:
      cm = cm[:, _cartesian_permutation(max_degree)]

    # Normalize r (if not already normalized).
    if not r_is_normalized:
        r = normalize(r, axis=-1)

    # Calculate all relevant monomials in the (x, y, z)-coordinates.
    # Note: This is done via integer powers and indexing on purpose! Using
    # jnp.power or the "**"-operator for this operation leads to NaNs in the
    # gradients for some inputs (jnp.power is not NaN-safe).
    r_powers = _integer_powers(np.expand_dims(r, axis=-2), max_degree)
    monomials = (
        r_powers[..., 0][..., ls[:, 0]]  #   x**lx.
        * r_powers[..., 1][..., ls[:, 1]]  # y**ly.
        * r_powers[..., 2][..., ls[:, 2]]  # z**lz.
    )

    # Calculate and return spherical harmonics (linear combination of monomials).
    return np.matmul(monomials, cm)
