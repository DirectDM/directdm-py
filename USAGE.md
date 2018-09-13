# Usage of DirectDM-py

Here is a short description of the main classes of `DirectDM-py`

Once installed properly, you can import the package as
```
import directdm as ddm
```

The main content of the package are four classes for the Wilson coefficients in the three-flavor, flour-flavor, and five-flavor effective theory, and above the electroweak scale, respectively.



## class `WC_3f(coeff_dict)`

A class of Wilson coefficients in the three-flavor scheme.

- Mandatory argument: `coeff_dict`: A `python` dictionary for the initial conditions of the Wilson coefficients, defined at the renormalization scale `mu_c = 2 GeV`. A sample dictionary could look like `{'C61u' : 1./100**2, 'C61d' : 1./100**2}` which would set the coefficients of the dimension-six operators `Q_{1,u}^(6)` and `Q_{1,d}^(6)` to unity, with an EFT scale of 100 GeV. All coefficients that are not specified are set to zero by default. The possible keys for the dictionary depend on the DM type and can be displayed via `ddm.WC_3f({}, DM_type).wc_name_list`.
- Optional argument: `DM_type`: Specifies the DM type. `DM_type="D"` for Dirac fermion, `DM_type="M"` for Majorana fermion, `DM_type="C"` for complex scalar, and `DM_type="R"` for real scalar DM. The default value is `DM_type="D"`.

### methods:

- `run()`: perform the one-loop RG evolution from scale `mu_c = 2 GeV` and output a dictionary with the resulting Wilson coefficients. It has only one, optional, argument: `mu_low=<scale>` is the lower scale of the RG evolution (it should be given in GeV; the default value is 2 GeV.
- `cNR(DM_mass, qvec)`: return a dictionary containing the coefficients of the nuclear operators, `c_i^N`, as defined in [Haxton et al](https://arxiv.org/abs/1203.3542) (the possible keys can be displayed via `ddm.WC_3f(coeff_dict).cNR_name_list`). The two mandatory arguments are the DM mass `DM_mass` and the spatial momentum transfer `qvec`, both in units of GeV. Two optional arguments, `RGE` and `NLO`, can be set to `True` or `False`. `RGE=False` will switch off the QCD and QED running (the default is `RGE=True`); `NLO=True` will add the coherently enhanced NLO terms for the tensor operators (see [Bishara et al.](https://arxiv.org/abs/1707.06998) for the details); the default is `NLO=False`.
- `write_mma(DM_mass, qvec)`: write a text file containing a list of the NR coefficients, in the order `{c_1^p, c_2^p, ..., c_{12}^p, c_1^n, c_2^n, ..., c_{12}^n}`, that can be loaded into the [DMFormFactor](https://www.ocf.berkeley.edu/~nanand/software/dmformfactor/) package. The two mandatory arguments are the DM mass `DM_mass` and the spatial momentum transfer `qvec`, both in units of GeV. In addition, the file name can be specified via the optional argument `filename="<filename>"` (default `"cNR.m"`) and the path where the file is saved can be specified via `path="<path>"` (default `"./"`). The other optional arguments `RGE` and `NLO` have the same effect as explained above.



## class `WC_4f(coeff_dict)`

A class of Wilson coefficients in the four-flavor scheme.

- Mandatory argument: `coeff_dict`: A `python` dictionary for the initial conditions of the Wilson coefficients, defined at the renormalization scale `mu_b = mb(mb)` (the MSbar bottom-quark mass). A sample dictionary could look like `{'C61u' : 1./100**2, 'C61d' : 1./100**2}` which would set the coefficients of the dimension-six operators `Q_{1,u}^(6)` and `Q_{1,d}^(6)` to unity, with an EFT scale of 100 GeV. All coefficients that are not specified are set to zero by default. The possible keys for the dictionary depend on the DM type and can be displayed via `ddm.WC_4f({}, DM_type).wc_name_list`.
- Optional argument: `DM_type`: Specifies the DM type. `DM_type="D"` for Dirac fermion, `DM_type="M"` for Majorana fermion, `DM_type="C"` for complex scalar, and `DM_type="R"` for real scalar DM. The default value is `DM_type="D"`.

### methods:

- `run()`: perform the one-loop RG evolution from scale `mu_b = mb(mb)`and output a dictionary with the resulting Wilson coefficients. It has only one, optional, argument: `mu_low=<scale>` is the lower scale of the RG evolution (it should be given in GeV; the default value is 2 GeV.
- `match()`: perform the running as described above and the matching to the theory with three active quark flavors. The matching scale may be specified with the optional argument `mu=<scale>` (the default scale is 2 GeV). The output is a python dictionary. Setting the optional argument `RGE=False` will switch off the QCD and QED running above the matching scale (default is `RGE=True`).
- `cNR(DM_mass, qvec)`: return a dictionary containing the coefficients of the nuclear operators, `c_i^N`, as defined in [Haxton et al](https://arxiv.org/abs/1203.3542) (the possible keys can be displayed via `ddm.WC_3f(coeff_dict).cNR_name_list`). The two mandatory arguments are the DM mass `DM_mass` and the spatial momentum transfer `qvec`, both in units of GeV. Two optional arguments, `RGE` and `NLO`, can be set to `True` or `False`. `RGE=False` will switch off the QCD and QED running (the default is `RGE=True`); `NLO=True` will add the coherently enhanced NLO terms for the tensor operators (see [Bishara et al.](https://arxiv.org/abs/1707.06998) for the details); the default is `NLO=False`.
- `write_mma(DM_mass, qvec)`: write a text file containing a list of the NR coefficients, in the order `{c_1^p, c_2^p, ..., c_{12}^p, c_1^n, c_2^n, ..., c_{12}^n}`, that can be loaded into the [DMFormFactor](https://www.ocf.berkeley.edu/~nanand/software/dmformfactor/) package. The two mandatory arguments are the DM mass `DM_mass` and the spatial momentum transfer `qvec`, both in units of GeV. In addition, the file name can be specified via the optional argument `filename="<filename>"` (default `"cNR.m"`) and the path where the file is saved can be specified via `path="<path>"` (default `"./"`). The other optional arguments `RGE` and `NLO` have the same effect as explained above.



## class `WC_5f(coeff_dict)`

A class of Wilson coefficients in the five-flavor scheme.

- Mandatory argument: `coeff_dict`: A `python` dictionary for the initial conditions of the Wilson coefficients, defined at the renormalization scale `mu_Z = MZ` (the Z-boson mass). A sample dictionary could look like `{'C61u' : 1./100**2, 'C61d' : 1./100**2}` which would set the coefficients of the dimension-six operators `Q_{1,u}^(6)` and `Q_{1,d}^(6)` to unity, with an EFT scale of 100 GeV. All coefficients that are not specified are set to zero by default. The possible keys for the dictionary depend on the DM type and can be displayed via `ddm.WC_5f({}, DM_type).wc_name_list`.
- Optional argument: `DM_type`: Specifies the DM type. `DM_type="D"` for Dirac fermion, `DM_type="M"` for Majorana fermion, `DM_type="C"` for complex scalar, and `DM_type="R"` for real scalar DM. The default value is `DM_type="D"`.

### methods:

- `run()`: perform the one-loop RG evolution from scale `mu_Z = MZ` and output a dictionary with the resulting Wilson coefficients. It has only one, optional, argument: `mu_low=<scale>` is the lower scale of the RG evolution (it should be given in GeV; the default value is `mb(mb)` = 4.18 GeV.
- `match()`: perform the running as described above and the matching to the theory with four active quark flavors. The matching scale may be specified with the optional argument `mu=<scale>` (the default scale is `mb(mb)` = 4.18 GeV). The output is a python dictionary. Setting the optional argument `RGE=False` will switch off the QCD and QED running above the matching scale (default is `RGE=True`).
- `cNR(DM_mass, qvec)`: return a dictionary containing the coefficients of the nuclear operators, `c_i^N`, as defined in [Haxton et al](https://arxiv.org/abs/1203.3542) (the possible keys can be displayed via `ddm.WC_3f(coeff_dict).cNR_name_list`). The two mandatory arguments are the DM mass `DM_mass` and the spatial momentum transfer `qvec`, both in units of GeV. Two optional arguments, `RGE` and `NLO`, can be set to `True` or `False`. `RGE=False` will switch off the QCD and QED running (the default is `RGE=True`); `NLO=True` will add the coherently enhanced NLO terms for the tensor operators (see [Bishara et al.](https://arxiv.org/abs/1707.06998) for the details); the default is `NLO=False`.
- `write_mma(DM_mass, qvec)`: write a text file containing a list of the NR coefficients, in the order `{c_1^p, c_2^p, ..., c_{12}^p, c_1^n, c_2^n, ..., c_{12}^n}`, that can be loaded into the [DMFormFactor](https://www.ocf.berkeley.edu/~nanand/software/dmformfactor/) package. The two mandatory arguments are the DM mass `DM_mass` and the spatial momentum transfer `qvec`, both in units of GeV. In addition, the file name can be specified via the optional argument `filename="<filename>"` (default `"cNR.m"`) and the path where the file is saved can be specified via `path="<path>"` (default `"./"`). The other optional arguments `RGE` and `NLO` have the same effect as explained above.



## class `WC_EW(coeff_dict, Ychi, dchi)`

A class of Wilson coefficients in the five-flavor scheme.

- Mandatory arguments: `coeff_dict`: A `python` dictionary for the initial conditions of the Wilson coefficients, defined at the renormalization scale `mu_Lambda` (see below). A sample dictionary could look like `{'C611' : 1./1000**2, 'C621' : 1./1000**2}` which would set the coefficients of the dimension-six operators `Q_{1,1}^(6)` and `Q_{1,1}^(6)` to unity with an EFT scale of 1 TeV. All coefficients that are not specified are set to zero by default. The possible keys for the dictionary can be displayed via `ddm.WC_EW({}, Ychi, dchi).wc_name_list_dim_5` and `ddm.WC_EW({}, Ychi, dchi).wc_name_list_dim_6`. The arguments `Ychi` and `dchi` give the DM hypercharge and weak isospin, resectively (see [Bishara et al.](https://arxiv.org/abs/1809.03506) for the precise definition). 
- Optional arguments: `DM_type`: Specifies the DM type. Only `DM_type="D"` for Dirac fermion is currently implemented; other values will lead to the abortion of the program. The default value is `DM_type="D"`.

### methods:

- `run(mu_Lambda)`: perform the one-loop RG evolution from scale `mu_Lambda` and output a dictionary with the resulting Wilson coefficients. It has one optional argument: `muz=<scale>` is the lower scale of the RG evolution (it should be given in GeV; the default value is `MZ`.
- `match(DM_mass, mu_Lambda)`: perform the running from scale `mu_Lambda` and the matching to the theory with broken electroweak symmetry and five active quark flavors at scale `MZ`. The output is a python dictionary. Setting the optional argument `RUN_EW=False` will switch off the RG evolution above the weak scale (default is `RUN_EW=True`).
- `cNR(DM_mass, qvec, mu_Lambda)`: return a dictionary containing the coefficients of the nuclear operators, `c_i^N`, as defined in [Haxton et al](https://arxiv.org/abs/1203.3542) (the possible keys can be displayed via `ddm.WC_3f(coeff_dict).cNR_name_list`). The two mandatory arguments are the DM mass `DM_mass` and the spatial momentum transfer `qvec`, both in units of GeV. Three optional arguments, `RGE`, `NLO`, and `RUN_EW`, can be set to `True` or `False`. `RGE=False` will switch off the QCD and QED running (the default is `RGE=True`); `NLO=True` will add the coherently enhanced NLO terms for the tensor operators (see [Bishara et al.](https://arxiv.org/abs/1707.06998) for the details); the default is `NLO=False`; `RUN_EW=False` will switch off the RG evolution above the weak scale (default is `RUN_EW=True`).
- `write_mma(DM_mass, qvec, mu_Lambda)`: write a text file containing a list of the NR coefficients, in the order `{c_1^p, c_2^p, ..., c_{12}^p, c_1^n, c_2^n, ..., c_{12}^n}`, that can be loaded into the [DMFormFactor](https://www.ocf.berkeley.edu/~nanand/software/dmformfactor/) package. The two mandatory arguments are the DM mass `DM_mass` and the spatial momentum transfer `qvec`, both in units of GeV. In addition, the file name can be specified via the optional argument `filename="<filename>"` (default `"cNR.m"`) and the path where the file is saved can be specified via `path="<path>"` (default `"./"`). The other optional arguments `RGE`, `NLO`, and `RUN_EW`, have the same effect as explained above.




The included `example.py` file has basic examples for using the functions provided by the code.


