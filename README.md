# BasisLib

Copy of Numpy-based basis functions utilities from [e3x](https://github.com/google-research/e3x/).

Useful external APIs:

- `BasisLib.so3.generate_clebsch_gordan_lookup_table`
- `BasisLib.so3.generate_spherical_harmonics_lookup_table`

Refer to `BasisLib.so3.spherical_harmonics` and `BasisLib.so3.clesch_gordan` for their usage.

### Example

```bash
python -m BasisLib.so3.clebsch_gordan_lut --path $HOME/BasisLib/BasisLib/so3_clebsch_gordan_lut.npz --max_degree 50 --num_processes 47
```
