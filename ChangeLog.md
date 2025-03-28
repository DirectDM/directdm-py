# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)

## [Unreleased] - 2025-03-25
### Fixed
- Wrong index ranges in wilson_coefficients.py

## [2.2.2] - 2023-07-10
### Fixed
- Key error in wilson_coefficients.py (`cNR13n` was not defined)
- Removed spurious NLO terms in `cNR` coefficients
- Deleted spurious dimension-seven Wilson coefficients. The basis now corresponds to [arxix [v3] of this paper](https://arxiv.org/abs/1710.10218).

### Changed
- `example.py` includes a few more examples for RG running below the weak scale

## [2.2.1] - 2023-04-21
### Fixed
- Inconsistent implementation of double weak insertions for lepton operators
- Typos in matching of dimension-seven Wilson coefficients (matching to wrong NR coefficient, typos in form factors)
- A bug in NLO option: cNR coefficients were partially overwritten
- decoupling alphas from 4 to 3 flavors at 2 GeV instead of mc(mc)
- Error in QED anomalous dimension for tensor / scalar operators

### Changed
- Updated charm-quark mass to most recent PDG value
- Removed internal options "DOUBLE_WEAK" and "double_QCD"
- class `WC_3f`: Removed method `run` and corresponding option `RGE`
- Globally removed the option to set the lower running scale
- Moved definitions of more dependent parameters to num_input

### Added
- Additional dependent parameters (masses and strong coupling at various scales) to num_input
- NR coefficients `cNR13p`, `cNR13n`, `cNR14p`, `cNR14n`
- "slope of strange vector current" contributions for double weak insertions

## [2.2.0] - 2020-11-30
### Fixed
- Typos in expressions for cNR (no impact on numerics)

### Changed
- Updated USAGE.md to include all input parameters
- Removed default input dictionary from single nucleon form factor classes

### Added
- The strange electric charge radius to the input parameters
- The slope of the vector current form factor at zero momentum transfer (only strange quark)
- Missing NLO terms for the strange vector current

## [2.1.4] - 2020-11-25
### Added
- The chiral NLO terms for the strange-quark vector current operators (these are leading since the LO terms vanish)

## [2.1.3] - 2020-11-22
### Fixed
- Inconsistencies in implementation of dependent input parameters

### Changed
- Updated numerical values of input parameters according to PDG 2020 and most recent lattice results
- Only "primary" input parameters can be changed by the user (see USAGE.md for more details). All dependent parameters are calculated automatically.
  An invalid key in the input dictionary will cause the code to abort. This is maybe somewhat drastic, but
  otherwise the code would run without the updated parameter which is probably not intended either.
- The QCD and electroweak MSbar top-quark mass at mu = MZ is now consistently used in the numerics.
  The approximate relations for converting the renormalization scheme are taken from [this reference](https://arxiv.org/abs/1212.4319).

### Added
- Some four-loop relations for running and decoupling; just for completeness and convenience;
  they are currently not used in the code

## [2.1.2] - 2020-07-22
### Fixed
- Normalization of nonrelativistic coefficients for scalar DM (thanks to Marco Fedele for pointing this out!)

## [2.1.1] - 2020-01-07
### Changed
- `USAGE.md`: Clarification that numerical inputs of Wilson coefficients should explicitly include the scale.

### Fixed
- Corrected typos in `USAGE.md`
- Errors in QCD anomalous dimension

### Added
- Hisano's fermionic twist-two contributions from [this publication](https://arxiv.org/abs/1104.0228)

## [2.1.0] - 2019-05-08
### Changed
- Input parameters are now given in form of a dictionary that can easily be updated by user-defined input values

### Fixed
- Corrected typos in `USAGE.md`
- Corrected typo in FGtilde form factor

### Added
- Running routines for quark masses
- Added module `dm_eft.py` containing "wrapper" classes that provide user-updated input to Wilson coefficient classes.
- Updated `USAGE.md`

## [2.0.1] - 2018-09-13
### Added
- File `USAGE.md` with basic explanation of relevant classes

### Changed
- Initial conditions of Wilson coefficients above the weak scale require scale explicitly
- Method `run` of class `WC_EW` requires scale `mu_Lambda` as starting value for RG evolution
- Updated `example.py`
- Updated `README.md`

### Fixed
- Typos in `rge.py`
- Typos in `example.py`

## [2.0.0] - 2018-09-12
### Added
- The class `WC_EW` for the Wilson coefficients above the electroweak scale.
  It performs the running above the weak scale, and matching to the five-flavor EFT,
  as detailed in [this publication](https://arxiv.org/abs/1809.03506)
- The class `CmuEW` for the electroweak RG evolution
- The files `full_adm_g1.py`, `full_adm_g2.py`,  `full_adm_yc.py`,  `full_adm_ytau.py`,
  `full_adm_yb.py`,  `full_adm_yt.py` with the electroweak anomalous dimensions
- The file `dim4_gauge_contribution.py` with the electroweak matching contributions from [Hisano et al.](https://arxiv.org/abs/1104.0228)

### Changed
- Numerical input from PDG 2018
- Running of strong coupling (consistent decoupling at flavor thresholds)
- example.py includes an example for Wilson coefficients specified in electroweak unbroken theory

### Fixed
- Bug in running of the strong coupling constant (decoupling at wrong loop order)

## [1.2.2] - 2018-05-03
### Fixed
- Typos in README
- Bug in matching methods (wrong order of optional arguments)

## [1.2.1] - 2018-04-18
### Added
- Uploaded package to the [Python Package Index](https://pypi.org/)

## [1.2.0] - 2018-01-31
### Added
- Weak mixing below the weak scale, as detailed in [this publication](https://arxiv.org/abs/1801.04240)
- Note that the results for electrons and muons neglect potentially large contributions; hence, a warning is issued in these cases

### Changed
- Replaced argument name `mchi` by `DM_mass`
- Updated nucleon masses from [PDG](http://pdg.lbl.gov/)

### Fixed
- A bug in `write_mma` (RG evolution was not switched of by optional argument `RGE=False`)
- The charm quark is now consistently integrated out at 2 GeV
- A bug in c1p for complex scalar DM

## [1.1.1] - 2017-10-27
### Added
- The module `single_nucleon_form_factors.py` with the classes for the single-nucleon form factors

### Fixed
- A bug in cNR11n and cNR12n
- A bug in the NLO value for cNR5

### Changed
- All expressions for the cNR are now given directly in terms of form factors
- Updated input values for nuclear magnetic moments

## [1.1.0] - 2017-10-24
### Added
- Remaining dimension-seven operators below the electroweak scale
- This `Changelog.md` file

### Removed
- Removed the optional argument `dict` everywhere; output is now always a python dictionary

### Changed
- The `example.py` now includes examples for the new operators, and no examples relating to the `dict` option 
- Updated the input values for the quark contributions to the nucleon magnetic moments

### Fixed
- Bug in RGE option for cNR
- Standard order of scalar Wilson coefficients
- Error in 3-flavor anomalous dimension

## [1.0.0] - 2017-09-01
### Added
*Initial release*

- `directdm` package folder
- `example.py`
- `setup.py`
- `README.md`
- `LICENCE`
