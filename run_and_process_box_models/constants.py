#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""


boltz = 1.3806505e-23 * 1e7 # J/K to erg/K
avogad = 6.02214179e23 # 1/mol
mwair = 2.89644e-2 * 1e3 # kg/mol to g/mol
rgas = 8.314472 * 1e-2 # J/mole/K to atmos/(mol/liter)/K

rho_so4 = 1800.
rho_h2so4 = rho_so4
rho_bc = 1800.
rho_h2o = 1000.
rho_poa = 1000.
rho_soa = 1400.
rho_dust = 2600.
rho_nacl = 2200.

MW_h2so4 = 98.
MW_air = 29.

mw_na = 23.
mw_cl = 35.

kappa_so4 = 0.65
kappa_h2so4 = kappa_so4
kappa_bc = 0.
kappa_h2o = 0.
kappa_poa = 0.001
kappa_soa = 0.1
kappa_dust = 0.1
kappa_nacl = 0.53


#
##     dens (kg/m^3)   ions in soln (1)    molec wght (kg/mole)   kappa (1)
#SO4            1800                0                   96d-3      0.65
#NO3            1800                0                   62d-3      0.65
#Cl             2200                0                   35.5d-3    0.53
#NH4            1800                0                   18d-3      0.65
#MSA            1800                0                   95d-3      0.53
#ARO1           1400                0                   150d-3     0.1
#ARO2           1400                0                   150d-3     0.1
#ALK1           1400                0                   140d-3     0.1
#OLE1           1400                0                   140d-3     0.1
#API1           1400                0                   184d-3     0.1
#API2           1400                0                   184d-3     0.1
#LIM1           1400                0                   200d-3     0.1
#LIM2           1400                0                   200d-3     0.1
#CO3            2600                0                   60d-3      0.53
#Na             2200                0                   23d-3      0.53
#Ca             2600                0                   40d-3      0.53
#OIN            2600                0                   1d-3       0.1
#OC             1000                0                   1d-3       0.001
#BC             1800                0                   1d-3       0
#H2O            1000                0                   18d-3      0
#~                                                                         
#
#
#
#    if spec_name == 'so4':
#        spec_densities[kk] = c.rho_so4
#        spec_kappas[kk] = c.kap_so4
#    elif spec_name == 'pom':
#        spec_densities[kk] = c.rho_poa
#        spec_kappas[kk] = c.kap_poa
#    elif spec_name == 'soa':
#        spec_densities[kk] = c.rho_soa
#        spec_kappas[kk] = c.kap_soa     
#    elif spec_name == 'bc':
#        spec_densities[kk] = c.rho_bc
#        spec_kappas[kk] = c.kap_bc
#    elif spec_name == 'dst':
#        spec_densities[kk] = c.rho_dust
#        spec_kappas[kk] = c.kap_dust
#    elif spec_densities == 'ncl':
#        spec_dens[kk] = c.rho_nacl
#        spec_kappas[kk] = c.kap_nacl