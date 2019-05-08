# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)

## [Unreleased]
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
