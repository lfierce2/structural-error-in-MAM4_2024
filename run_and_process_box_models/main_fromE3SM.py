#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main script to drive simulations

author: Laura Fierce
"""

import ensemble_runner
import os
import benchmark_splitToNodes
import pickle

# ensemble_prefix = 'ensemble_40_sortaBig_'
# ensemble_prefix = 'ensemble_42_smallTest_'
# ensemble_prefix = 'ensemble_42_bigger_'
# ensemble_prefix = 'ensemble_43_bigger_'
# ensemble_prefix = 'ensemble_20_smallTest_'
# ensemble_prefix = 'ensemble_21_bigger_'
# ensemble_prefix = 'ensemble_27_big_'
# ensemble_prefix = 'ensemble_28_smallTest_'
# process = 'coag'
# lab = 'c'

process = 'both'
lab = 'a'

# import os
# for ii in range(28191374,28192862+1):
#     os.system('scancel ' + str(ii))

# ensemble_prefix = 'ensemble_29_bigTest_'

# ensemble_prefix = 'ensemble_43_bigger_'
# ensemble_prefix = 'ensemble_44_bigTest_'
ensemble_prefix = 'ensemble_47_big'

ensemble_name = ensemble_prefix + '_' + lab + '_' + process

dry = True # if False, let RH vary; if True, set RH = 0 for all cases

if 'Volcano' in ensemble_prefix:
    scenario_type = 'volcano'
else:
    scenario_type = '4mode_noemit'


if 'smallTest' in ensemble_prefix:
    n_part = 100
    n_repeat = 10
    N_samples = 2
if 'bigTest' in ensemble_prefix:
    n_part = 5000
    n_repeat = 5
    N_samples = 2
elif 'bigger' in ensemble_prefix:
    n_part = 10000
    n_repeat = 5
    N_samples = 1000
    # N_nodes = 10
elif 'test' in ensemble_prefix:
    n_part = 500
    n_repeat = 1
    N_samples = 10
elif 'big' in ensemble_prefix:
    n_part = 5000
    n_repeat = 5
    N_samples = 100
elif 'one' in ensemble_prefix:
    n_part = 500
    n_repeat = 1
    N_samples = 1
elif 'realBig' in ensemble_prefix:
    n_part = 2000
    n_repeat = 5
    N_samples = 10000
elif 'sortaBig' in ensemble_prefix:
    n_part = 2000
    n_repeat = 5
    N_samples = 50

# t_max = 3600.*4.
# del_t = 60.
# t_output = 600.
t_max = 24.*3600. #2.*3600
del_t = 60.
t_output = 600.
run_time_min = 120.

project = 'particula'


# fractionSurface = 0.9
# useSorted = False
# e3sm_filename = '/pscratch/sd/l/lfierce/e3sm_simulations/PartMC_test_map_05.nc'
# e3sm_filename = '/global/cscratch1/sd/yuyao/for_laura/PartMC_test_map_05.nc'
#e3sm_filename = '/Users/fier887/Downloads/PartMC_test_map_05.nc'
just_the_filename = 'E3SM_output_for_PartMC_0607_surf.nc'
if os.path.exists('/people/fier887/'):
    e3sm_filename = '/pic/projects/sooty2/fierce/e3sm_simulations/' + just_the_filename
    ensemble_over_dir = '/pic/projects/sooty2/fierce/box_simulations3/'
    mam_build_location='/qfs/people/fier887/mam_refactor/build/standalone'
    mam_input='/qfs/people/fier887/mam_refactor/standalone/tests/smoke_test.nl'
    mam_output = '/qfs/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc'
    partmc_run_dir = '/people/fier887/partmc-2.6.1/scenarios/7_box_eval/'
    core_dir = '/people/fier887/mam_refactor/core/'
elif os.path.exists('/global/cscratch1/sd/lfierce'):
    e3sm_filename = '/global/cscratch1/sd/yuyao/for_laura/' + just_the_filename
    ensemble_over_dir = '/global/cscratch1/sd/lfierce/box_simulations3/'
    mam_build_location='/global/homes/l/lfierce/mam_refactor-main/build/standalone'
    mam_input='/global/homes/l/lfierce/mam_refactor-main/standalone/tests/smoke_test.nl'
    mam_output = '/global/homes/l/lfierce/mam_refactor-main/build/standalone/tests/mam_output.nc'
    partmc_run_dir = '/global/homes/l/lfierce/partmc_mosaic/partmc/scenarios/7_box_eval/'
    core_dir = '/global/homes/l/lfierce/mam_refactor-main/core/'
elif os.path.exists('/pscratch/sd/l/lfierce/'):
    e3sm_filename = '/pscratch/sd/l/lfierce/e3sm_simulations/' + just_the_filename
    ensemble_over_dir = '/pscratch/sd/l/lfierce/box_simulations3/'
    mam_build_location='/global/homes/l/lfierce/mam_refactor-main/build/standalone'
    mam_input='/global/homes/l/lfierce/mam_refactor-main/standalone/tests/smoke_test.nl'
    mam_output = '/global/homes/l/lfierce/mam_refactor-main/build/standalone/tests/mam_output.nc'
    partmc_run_dir = '/global/homes/l/lfierce/partmc_mosaic/partmc/scenarios/7_box_eval/'
    core_dir = '/global/homes/l/lfierce/mam_refactor-main/core/'
elif os.path.exists('/Users/fier887/'):
    e3sm_filename = '/Users/fier887/Downloads/' + just_the_filename
    ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/'

ensemble_settings,mam_ensemble_settings = ensemble_runner.go_allFromE3SM(
    ensemble_name,
    e3sm_filename,
    dry = dry,
    run_time_min = run_time_min,
    scenario_type=scenario_type,
    N_samples = N_samples,
    n_part = n_part,
    n_repeat = n_repeat,
    t_max = t_max,
    del_t = del_t,
    t_output = t_output,
    span_observed_flux=True,
    correlate_flux_with_num=False,
    flux_dist = 'loguniform',#'lognormal',#'loguniform',
    flux_param1 = 1e-2*1e-9,#0.1*1e-9,#1e-2*1e-9,
    flux_param2 = 1e1*1e-9,#1.5,#
    sigma_log10_flux=0.5,
    split_repeats=True,
    no_soa=True,only_subsaturated_cells=True,
    ht = 500.,
    min_frac_from_prod=1.,
    project=project)

ensemble_dir = ensemble_over_dir + ensemble_name + '/scenarios/'
benchmark_splitToNodes.mam_ensemble_create(
        ensemble_dir,mam_ensemble_settings,#mu1_vals,mu2_vals,mu3_vals,mu4_vals,
        scenario_type='4mode_noemit',uniform_comp=False,
        mam_build_location=mam_build_location,
        mam_input=mam_input,
        mam_output=mam_output,
        core_dir=core_dir)

benchmark_splitToNodes.mam_ensemble_run(
        ensemble_dir,
        mam_build_location=mam_build_location,
        mam_input=mam_input,
        mam_output=mam_output,
        core_dir=core_dir)

processes = ['cond','coag']
labs = ['b','c']
for (process,lab) in zip(processes,labs):
    updated_ensemble_name = ensemble_prefix + '_' + lab + '_' + process
    updated_ensemble_settings,updated_mam_ensemble_settings = ensemble_runner.go_otherProcesses(
            ensemble_name,
            updated_ensemble_name,
            run_time_min=run_time_min,
            split_repeats=True,
            scenario_type='4mode_noemit',uniform_comp=False)
    
    ensemble_dir = ensemble_over_dir + updated_ensemble_name + '/scenarios/'

    pickle.dump(updated_ensemble_settings,open(ensemble_dir + 'ensemble_settings.pkl','wb'),protocol=0)
    pickle.dump(updated_mam_ensemble_settings,open(ensemble_dir + 'mam_ensemble_settings.pkl','wb'),protocol=0)
    
    benchmark_splitToNodes.mam_ensemble_create(
            ensemble_dir,updated_mam_ensemble_settings,#mu1_vals,mu2_vals,mu3_vals,mu4_vals,
            scenario_type='4mode_noemit',uniform_comp=False,
            mam_build_location=mam_build_location,
            mam_input=mam_input,
            mam_output=mam_output,
            core_dir=core_dir)
    
    benchmark_splitToNodes.mam_ensemble_run(
            ensemble_dir,
            mam_build_location=mam_build_location,
            mam_input=mam_input,
            mam_output=mam_output,
            core_dir=core_dir)

# ensemble_settings,mam_ensemble_settings = ensemble_runner.go_fromE3SM(
#     ensemble_name,
#     e3sm_filename,
#     fractionSurface=fractionSurface,
#     useSorted=useSorted,
#     run_time_min = run_time_min,
#     scenario_type=scenario_type,
#     N_samples = N_samples,
#     n_part = n_part,
#     n_repeat = n_repeat,
#     t_max = t_max,
#     del_t = del_t,
#     t_output = t_output,
#     split_repeats=True,
#     flux_dist = 'loguniform',
#     flux_param1 = 1e-2*1e-9/3600.,
#     flux_param2 = 1e1*1e-9/3600., # too high? note: per hour, not per second (make sure this is right)
#     min_frac_from_prod=0.) # note: need to limit initial H2SO4 concentrations to have reasonable value



# import os
# for job_id in range(2171597,2171615):
#     # print('scancel ' + str(job_id))
#     os.system('scancel ' + str(job_id))
# run_time = 30. 
# N_samples=100
# split_repeats=True
# N_modes=4
# N_all_varnames = ['tot_num','frac_accum','nonaccum_frac_coarse','ofaccum_frac_fresh']
# N_all_dists = ['loguniform','uniform','loguniform','loguniform']
# N_all_param1 = [3e7,0.,1e-8,1e-13]
# N_all_param2 = [2e12,1.,1.,1.]
# dgm_all_varnames = ['dgm_accum','dgm_aik','dgm_coarse','dgm_fresh']
# dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform']
# dgm_all_param1 = [0.5e-7, 0.5e-8, 1.000e-6, 1e-8]
# dgm_all_param2 = [1.1e-7, 3e-8, 2.000e-6, 6e-8]            
# mode_sigs = np.log([1.8,1.6,1.6,1.8])
# mode1_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom']
# mode1_dists = ['uniform','uniform','uniform','uniform','uniform']
# mode1_param1 = [0.,0.,0.,0.,0.]
# mode1_param2 = [1.,1.,1.,1.,1.]
# mode2_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl']
# mode2_dists = ['uniform','uniform']
# mode2_param1 = [1.,0.]
# mode2_param2 = [1.,0.]
# mode3_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom']
# mode3_dists = ['uniform','uniform','uniform','uniform','uniform']
# mode3_param1 = [0.,0.,0.,0.,0.]
# mode3_param2 = [1.,1.,1.,1.,1.]
# mode4_varnames = ['frac_bc']
# mode4_dists = ['uniform']
# mode4_param1 = [0.]
# mode4_param2 = [1.]
# flux_dist = 'loguniform'
# flux_param1 = 1e-2*1e-9
# flux_param2 = 1e1*1e-9
# rh_dist='uniform'
# rh_param1 = 0.
# rh_param2 = 0.99
# temp_dist='uniform'
# temp_param1 = 240.
# temp_param2 = 310.
# n_part = 10000
# n_repeat = 10        
# t_max = 3600.*4.
# del_t = 60.
# t_output = 600.
# min_frac_from_prod=0.
# p=101325.
# ht=500.