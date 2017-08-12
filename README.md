**directdm**: a python3 package for dark matter direct detection
=====

`directdm` takes the Wilson coefficients of relativistic operators that couple DM to the SM quarks, leptons, and gauge bosons and matches them onto a non-relativisitc Galilean invariant EFT in order to caclulate the direct detection scattering rates.

You can get the latest version of `directdm` on [`github`](https://directdm.github.io).

## Usage

The included `example.py` file has basic examples for using the functions provided by the code. 

Here is a simple example assuming that the `example.py` file is in same directory as `./directdm/`.

Load the package
```
import directdm as ddm
```

Set the DM type to a Dirac fermion and set some Wilson coefficients in the three-flavor basis
```
wc_dict = {'C61u' : 1./100**2 , 'C61d' : 1./100**2}
wc3f = ddm.WC_3f(wc_dict, DM_type="D")
```

Match the three-flavor Wilson coefficients onto the non-relativstic ones (the DM mass has been set to 100 GeV, and the momentum transfer is 50 MeV):
```
print(wc3f.cNR(100, 50e-3))
```

Write the list of proton and neutron NR Wilson coefficients into a file in the current directory with filename 'filename':
```
wc3f.write_mma(100, 50e-3, filename='wc3.m')
```

## Citation
If you use `DirectDM` please cite us! To get the `BibTeX` entries, click on: [inspirehep query](https://inspirehep.net/search?p=arxiv:1708.02678+or+arxiv:1707.06998+or+arxiv:1611.00368&of=hx) 



## Contributors

   * Fady Bishara (University of Oxford)
   * Joachim Brod (TU Dortmund)
   * Jure Zupan (University of Cincinnati)
   * Benjamin Grinstein (UC San Diego)

## License
`DirectDM` is distributed under the MIT license.


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
