# Processing code for box model evaluation of MAM4
Python code used for scenario generation, data process, and analysis, and visualization for Fierce et al. (2024), Constraining black carbon aging in global models to reflect timescales for internal mixing, JAMES (under review).

This package contains Python code to:
  * Create and run ensembles of simulations with the PartMC-MOSAIC and MAM4 box models
  * Process the box model simluation output
  * Analyze the output and create figures

This package also includes the input settings for the ensembles of PartMC-MOSAIC and MAM4 simulations used in the inter-comparison study.

## Box models
This package was developed and tested with the following models:
  * Benchmark model: partmc-2.6.1 (https://github.com/compdyn/partmc) and mosaic-2012-01-25 (available from Rahul Zaveri)
  * Reduced model: beta version of the Modal Aerosol Model box model (https://github.com/eagles-project/mam_refactor)

## Python Dependencies
  * numpy
  * scipy
  * netCDF4
  * pickle
  * os
  * seaborn
