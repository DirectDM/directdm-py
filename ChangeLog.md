# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)

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

### Added
- The class `WC_EW` for the Wilson coefficients above the electroweak scale
- The class `CmuEW` for the electroweak RG evolution
- The files `full_adm_g1.py`, `full_adm_g2.py`,  `full_adm_yt.py` with the electroweak anomalous dimensions

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
