# BasisLib

Copy of Numpy-based basis functions utilities from [e3x](https://github.com/google-research/e3x/).

Useful external APIs:

- `BasisLib.so3.generate_clebsch_gordan_lookup_table`: Current `.npz` cache supports upto `lmax=20`
- `BasisLib.so3.generate_spherical_harmonics_lookup_table`: Current `.npz` cache supports upto `lmax=15`

Refer to `BasisLib.so3.spherical_harmonics` and `BasisLib.so3.clesch_gordan` for their usage.

### Workflow for generating more coefficients

```bash
python -m BasisLib.so3.clebsch_gordan_lut --path $HOME/BasisLib/BasisLib/so3_clebsch_gordan_lut.npz --max_degree 50 --num_processes 47
```
