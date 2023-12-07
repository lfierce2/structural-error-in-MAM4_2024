#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions needed to run ensemble of PartMC-MOSAIC and MAM4 box model scenarios

@author: Laura Fierce
"""
import ensemble_maker
import scenario_runner
import process_readVariables
import numpy as np
import os
from netCDF4 import Dataset
import analyses

# def go_requeue(ensemble_name):
    
def go(ensemble_name,
       scenario_type='4mode_noemit',
       run_time_min = 30., # minutes
       N_samples=100,
       n_part = 10000,
       n_repeat = 10,
       t_max = 3600.*4.,
       del_t = 60.,
       t_output = 600.,
       split_repeats=True, # N_modes=4,
       N_all_varnames = ['tot_num','frac_accum','nonaccum_frac_coarse','ofaccum_frac_fresh'],
       N_all_dists = ['loguniform','uniform','loguniform','loguniform'],
       N_all_param1 = [3e7,0.0005,1e-8,1e-13],
       N_all_param2 = [2e12,1.,0.983,0.98],
       # N_all_param1 = [3e7,0.,1e-8,1e-13],
       # N_all_param2 = [2e12,1.,1.,1.],
       dgm_all_varnames = ['dgm_accum','dgm_aik','dgm_coarse','dgm_fresh'],
       dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
       dgm_all_param1 = [0.535e-7, 8.7e-8, 1.000e-6, 1e-8],
       dgm_all_param2 = [1.1e-7, 5.15e-8, 2.000e-6, 6e-8],            
       mode_sigs = np.log([1.8,1.6,1.6,1.8]),
       mode1_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
       mode1_dists = ['uniform','uniform','uniform','uniform','uniform'],
       mode1_param1 = [0.,0.,0.,0.,0.],
       mode1_param2 = [1.,1.,1.,1.,1.],
       mode2_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl'],
       mode2_dists = ['uniform','uniform'],
       mode2_param1 = [1.,0.],
       mode2_param2 = [1.,0.],
       mode3_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
       mode3_dists = ['uniform','uniform','uniform','uniform','uniform'],
       mode3_param1 = [0.,0.,0.,0.,0.],
       mode3_param2 = [1.,1.,1.,1.,1.],
       mode4_varnames = ['frac_bc'],
       mode4_dists = ['uniform'],
       mode4_param1 = [0.],
       mode4_param2 = [1.],
       flux_dist = 'loguniform',
       flux_param1 = 1e-2*1e-9,
       flux_param2 = 1e1*1e-9,
       rh_dist='uniform',
       rh_param1 = 0.,
       rh_param2 = 0.99,
       temp_dist='uniform',
       temp_param1 = 240.,
       temp_param2 = 310.,
       min_frac_from_prod=0.,
       p=101325.,ht=500.):
    
    ensemble_dir = scenario_runner.get_store_over_dir() + 'box_simulations3/' + ensemble_name + '/'
    if not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)
    
    scenarios_dir = ensemble_dir + 'scenarios/'
    if not os.path.exists(scenarios_dir):
        os.mkdir(scenarios_dir)
    
    runfiles_dir = ensemble_dir + 'runfiles/'
    if not os.path.exists(runfiles_dir):
        os.mkdir(runfiles_dir)
    
    ensemble_settings,mam_ensemble_settings = ensemble_maker.get_lhs_ensemble_settings(
            ensemble_dir,N_samples,
            N_all_varnames = N_all_varnames, 
            N_all_dists = N_all_dists, 
            N_all_param1 = N_all_param1, 
            N_all_param2 = N_all_param2,
            dgm_all_varnames = dgm_all_varnames,
            dgm_all_dists = dgm_all_dists,
            dgm_all_param1 = dgm_all_param1,
            dgm_all_param2 = dgm_all_param2,
            mode_sigs = mode_sigs,
            mode1_varnames = mode1_varnames,
            mode1_dists = mode1_dists,
            mode1_param1 = mode1_param1,
            mode1_param2 = mode1_param2,
            mode2_varnames = mode2_varnames,
            mode2_dists = mode2_dists,
            mode2_param1 = mode2_param1,
            mode2_param2 = mode2_param2,
            mode3_varnames = mode3_varnames,
            mode3_dists = mode3_dists,
            mode3_param1 = mode3_param1,
            mode3_param2 = mode3_param2,
            mode4_varnames = mode4_varnames,
            mode4_dists = mode4_dists,
            mode4_param1 = mode4_param1,
            mode4_param2 = mode4_param2,
            flux_dist = flux_dist,
            flux_param1 = flux_param1,
            flux_param2 = flux_param2,
            rh_dist = rh_dist,
            rh_param1 = rh_param1,
            rh_param2 = rh_param2,
            temp_dist = temp_dist,
            temp_param1 = temp_param1,
            temp_param2 = temp_param2,
            n_part = n_part,
            n_repeat = n_repeat,        
            t_max = t_max,
            del_t = del_t, t_output = t_output,
            min_frac_from_prod = min_frac_from_prod,
            p = p, ht = ht)
    
    run_time_str = scenario_runner.get_run_time_str(run_time_min)
    # if run_time < 60.:
    #     run_time = '0:00:' + str(int(run_time_min))
    # else:
    #     hrs = run_time_min/60.
    #     print('need to code longer run times!')
    
    run_filenames = []
    for ii,settings in enumerate(ensemble_settings):
        print_scenario_here = scenarios_dir +  str(ii).zfill(6) + '/'
        if not os.path.exists(print_scenario_here):
            os.mkdir(print_scenario_here)
        
        # create all of the scenarios & put all the run_filenames in a list
        if split_repeats:
            run_filenames_onescenario = scenario_runner.splitRepats(print_scenario_here,settings,scenario_type=scenario_type,uniform_comp=False)
            for run_filename in run_filenames_onescenario:
                run_filenames.append(run_filename)
        else:
            print('not yet coded without split repeats!')
    
    for ii,run_filename in enumerate(run_filenames):
        scenario_runner.send_run_to_slurm(run_filename, runfiles_dir, run_time_str=run_time_str)
        
    return ensemble_settings,mam_ensemble_settings

def go_fromE3SM(ensemble_name,e3sm_filename,
       fractionSurface=0.9,useSorted=False,
       scenario_type='4mode_noemit',
       run_time_min = 30., # minutes
       N_samples=100,
       n_part = 10000,
       n_repeat = 10,
       t_max = 3600.*4.,
       del_t = 60.,
       t_output = 600.,
       split_repeats=True, # N_modes=4,
       N_all_varnames = ['tot_num','frac_accum','nonaccum_frac_coarse','ofaccum_frac_fresh'],
       N_all_dists = ['loguniform','uniform','loguniform','loguniform'],
       N_all_param1 = [3e7,0.,1e-6,1e-5],
       N_all_param2 = [2e12,1.,0.5,1.],
       # N_all_param1 = [3e7,0.,1e-8,1e-13],
       # N_all_param2 = [2e12,1.,1.,1.],
       dgm_all_varnames = ['dgm_accum','dgm_aik','dgm_coarse','dgm_fresh'],
       dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
       dgm_all_param1 = [0.5e-7, 0.5e-8, 1.000e-6, 1e-8],
       dgm_all_param2 = [1.1e-7, 3e-8, 2.000e-6, 6e-8],        
       mode_sigs = np.log([1.8,1.6,1.6,1.8]),
       mode1_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
       mode1_dists = ['uniform','uniform','uniform','uniform','uniform'],
       mode1_param1 = [0.,0.,0.,0.,0.],
       mode1_param2 = [1.,1.,1.,1.,1.],
       mode2_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl'],
       mode2_dists = ['uniform','uniform'],
       mode2_param1 = [1.,0.],
       mode2_param2 = [1.,0.],
       mode3_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
       mode3_dists = ['uniform','uniform','uniform','uniform','uniform'],
       mode3_param1 = [0.,0.,0.,0.,0.],
       mode3_param2 = [1.,1.,1.,1.,1.],
       mode4_varnames = ['frac_bc'],
       mode4_dists = ['uniform'],
       mode4_param1 = [0.],
       mode4_param2 = [1.],
       flux_dist = 'loguniform',
       flux_param1 = 1e-2*1e-9,
       flux_param2 = 1e1*1e-9,
       rh_dist='uniform',
       rh_param1 = 0.,
       rh_param2 = 0.99,
       temp_dist='uniform',
       temp_param1 = 240.,
       temp_param2 = 310.,
       min_frac_from_prod=0.8,
       p=101325.,ht=500.,
       project='nuvola'):
    
    # f = Dataset(e3sm_filename)
    # # varnames = ['logN0_1','logN0_2','logN0_3','logN0_4']
    # varnames = ['mu0_1','mu0_2','mu0_3','mu0_4','logN0_1','logN0_2','logN0_3','logN0_4']
    # all_vars = ()
    # n_bins = ()
    # n_bins_one = 10 
    
    # for varname in varnames:
    #     all_vars += (analyses.retrieve_e3sm_data(f,varname).ravel(),)
    
        # n_bins += (n_bins_one,)
        # n_bins_one += 1
    # output = np.histogramdd(all_vars,bins=n_bins)
    # normalized_hist = output[0]/np.sum(output[0].ravel())
    
    
    ensemble_dir = scenario_runner.get_store_over_dir() + 'box_simulations3/' + ensemble_name + '/'
    if not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)
    
    scenarios_dir = ensemble_dir + 'scenarios/'
    if not os.path.exists(scenarios_dir):
        os.mkdir(scenarios_dir)
    
    runfiles_dir = ensemble_dir + 'runfiles/'
    if not os.path.exists(runfiles_dir):
        os.mkdir(runfiles_dir)
    
    ensemble_settings,mam_ensemble_settings = ensemble_maker.get_lhs_ensemble_settings_fromE3SM(
            ensemble_dir,e3sm_filename,N_samples,
            fractionSurface = fractionSurface,useSorted=useSorted,
            N_all_varnames = N_all_varnames, 
            N_all_dists = N_all_dists, 
            N_all_param1 = N_all_param1, 
            N_all_param2 = N_all_param2,
            dgm_all_varnames = dgm_all_varnames,
            dgm_all_dists = dgm_all_dists,
            dgm_all_param1 = dgm_all_param1,
            dgm_all_param2 = dgm_all_param2,
            mode_sigs = mode_sigs,
            mode1_varnames = mode1_varnames,
            mode1_dists = mode1_dists,
            mode1_param1 = mode1_param1,
            mode1_param2 = mode1_param2,
            mode2_varnames = mode2_varnames,
            mode2_dists = mode2_dists,
            mode2_param1 = mode2_param1,
            mode2_param2 = mode2_param2,
            mode3_varnames = mode3_varnames,
            mode3_dists = mode3_dists,
            mode3_param1 = mode3_param1,
            mode3_param2 = mode3_param2,
            mode4_varnames = mode4_varnames,
            mode4_dists = mode4_dists,
            mode4_param1 = mode4_param1,
            mode4_param2 = mode4_param2,
            flux_dist = flux_dist,
            flux_param1 = flux_param1,
            flux_param2 = flux_param2,
            rh_dist = rh_dist,
            rh_param1 = rh_param1,
            rh_param2 = rh_param2,
            temp_dist = temp_dist,
            temp_param1 = temp_param1,
            temp_param2 = temp_param2,
            n_part = n_part,
            n_repeat = n_repeat,        
            t_max = t_max,
            del_t = del_t, t_output = t_output,
            min_frac_from_prod = min_frac_from_prod,
            p = p, ht = ht,
            project=project)
    
    run_time_str = scenario_runner.get_run_time_str(run_time_min)
    # if run_time < 60.:
    #     run_time = '0:00:' + str(int(run_time_min))
    # else:
    #     hrs = run_time_min/60.
    #     print('need to code longer run times!')
    
    run_filenames = []
    for ii,settings in enumerate(ensemble_settings):
        print_scenario_here = scenarios_dir +  str(ii).zfill(6) + '/'
        if not os.path.exists(print_scenario_here):
            os.mkdir(print_scenario_here)
        
        # create all of the scenarios & put all the run_filenames in a list
        if split_repeats:
            run_filenames_onescenario = scenario_runner.splitRepats(print_scenario_here,settings,scenario_type=scenario_type,uniform_comp=False)
            for run_filename in run_filenames_onescenario:
                run_filenames.append(run_filename)
        else:
            print('not yet coded without split repeats!')
    
    for ii,run_filename in enumerate(run_filenames):
        scenario_runner.send_run_to_slurm(run_filename, runfiles_dir, run_time_str=run_time_str, project=project)
        
    return ensemble_settings,mam_ensemble_settings

# def go_fromE3SM(ensemble_name,e3sm_filename,
#        fractionSurface=0.9,useSorted=False,
#        scenario_type='4mode_noemit',
#        run_time_min = 30., # minutes
#        N_samples=100,
#        n_part = 10000,
#        n_repeat = 10,
#        t_max = 3600.*4.,
#        del_t = 60.,
#        t_output = 600.,
#        split_repeats=True, # N_modes=4,
#        N_all_varnames = ['tot_num','frac_accum','nonaccum_frac_coarse','ofaccum_frac_fresh'],
#        N_all_dists = ['loguniform','uniform','loguniform','loguniform'],
#        N_all_param1 = [3e7,0.,1e-6,1e-5],
#        N_all_param2 = [2e12,1.,0.5,1.],
#        # N_all_param1 = [3e7,0.,1e-8,1e-13],
#        # N_all_param2 = [2e12,1.,1.,1.],
#        dgm_all_varnames = ['dgm_accum','dgm_aik','dgm_coarse','dgm_fresh'],
#        dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
#        dgm_all_param1 = [0.5e-7, 0.5e-8, 1.000e-6, 1e-8],
#        dgm_all_param2 = [1.1e-7, 3e-8, 2.000e-6, 6e-8],            
#        mode_sigs = np.log([1.8,1.6,1.6,1.8]),
#        mode1_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
#        mode1_dists = ['uniform','uniform','uniform','uniform','uniform'],
#        mode1_param1 = [0.,0.,0.,0.,0.],
#        mode1_param2 = [1.,1.,1.,1.,1.],
#        mode2_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl'],
#        mode2_dists = ['uniform','uniform'],
#        mode2_param1 = [1.,0.],
#        mode2_param2 = [1.,0.],
#        mode3_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
#        mode3_dists = ['uniform','uniform','uniform','uniform','uniform'],
#        mode3_param1 = [0.,0.,0.,0.,0.],
#        mode3_param2 = [1.,1.,1.,1.,1.],
#        mode4_varnames = ['frac_bc'],
#        mode4_dists = ['uniform'],
#        mode4_param1 = [0.],
#        mode4_param2 = [1.],
#        flux_dist = 'loguniform',
#        flux_param1 = 1e-2*1e-9,
#        flux_param2 = 1e1*1e-9,
#        rh_dist='uniform',
#        rh_param1 = 0.,
#        rh_param2 = 0.99,
#        temp_dist='uniform',
#        temp_param1 = 240.,
#        temp_param2 = 310.,
#        min_frac_from_prod=0.8,
#        p=101325.,ht=500.):
    
#     # f = Dataset(e3sm_filename)
#     # # varnames = ['logN0_1','logN0_2','logN0_3','logN0_4']
#     # varnames = ['mu0_1','mu0_2','mu0_3','mu0_4','logN0_1','logN0_2','logN0_3','logN0_4']
#     # all_vars = ()
#     # n_bins = ()
#     # n_bins_one = 10 
    
#     # for varname in varnames:
#     #     all_vars += (analyses.retrieve_e3sm_data(f,varname).ravel(),)
    
#         # n_bins += (n_bins_one,)
#         # n_bins_one += 1
#     # output = np.histogramdd(all_vars,bins=n_bins)
#     # normalized_hist = output[0]/np.sum(output[0].ravel())
    
    
#     ensemble_dir = scenario_runner.get_store_over_dir() + 'box_simulations3/' + ensemble_name + '/'
#     if not os.path.exists(ensemble_dir):
#         os.mkdir(ensemble_dir)
    
#     scenarios_dir = ensemble_dir + 'scenarios/'
#     if not os.path.exists(scenarios_dir):
#         os.mkdir(scenarios_dir)
    
#     runfiles_dir = ensemble_dir + 'runfiles/'
#     if not os.path.exists(runfiles_dir):
#         os.mkdir(runfiles_dir)
    
#     ensemble_settings,mam_ensemble_settings = ensemble_maker.get_lhs_ensemble_settings_fromE3SM(
#             ensemble_dir,e3sm_filename,N_samples,
#             fractionSurface = fractionSurface,useSorted=useSorted,
#             N_all_varnames = N_all_varnames, 
#             N_all_dists = N_all_dists, 
#             N_all_param1 = N_all_param1, 
#             N_all_param2 = N_all_param2,
#             dgm_all_varnames = dgm_all_varnames,
#             dgm_all_dists = dgm_all_dists,
#             dgm_all_param1 = dgm_all_param1,
#             dgm_all_param2 = dgm_all_param2,
#             mode_sigs = mode_sigs,
#             mode1_varnames = mode1_varnames,
#             mode1_dists = mode1_dists,
#             mode1_param1 = mode1_param1,
#             mode1_param2 = mode1_param2,
#             mode2_varnames = mode2_varnames,
#             mode2_dists = mode2_dists,
#             mode2_param1 = mode2_param1,
#             mode2_param2 = mode2_param2,
#             mode3_varnames = mode3_varnames,
#             mode3_dists = mode3_dists,
#             mode3_param1 = mode3_param1,
#             mode3_param2 = mode3_param2,
#             mode4_varnames = mode4_varnames,
#             mode4_dists = mode4_dists,
#             mode4_param1 = mode4_param1,
#             mode4_param2 = mode4_param2,
#             flux_dist = flux_dist,
#             flux_param1 = flux_param1,
#             flux_param2 = flux_param2,
#             rh_dist = rh_dist,
#             rh_param1 = rh_param1,
#             rh_param2 = rh_param2,
#             temp_dist = temp_dist,
#             temp_param1 = temp_param1,
#             temp_param2 = temp_param2,
#             n_part = n_part,
#             n_repeat = n_repeat,        
#             t_max = t_max,
#             del_t = del_t, t_output = t_output,
#             min_frac_from_prod = min_frac_from_prod,
#             p = p, ht = ht)
    
#     run_time_str = scenario_runner.get_run_time_str(run_time_min)
#     # if run_time < 60.:
#     #     run_time = '0:00:' + str(int(run_time_min))
#     # else:
#     #     hrs = run_time_min/60.
#     #     print('need to code longer run times!')
    
#     run_filenames = []
#     for ii,settings in enumerate(ensemble_settings):
#         print_scenario_here = scenarios_dir +  str(ii).zfill(6) + '/'
#         if not os.path.exists(print_scenario_here):
#             os.mkdir(print_scenario_here)
        
#         # create all of the scenarios & put all the run_filenames in a list
#         if split_repeats:
#             run_filenames_onescenario = scenario_runner.splitRepats(print_scenario_here,settings,scenario_type=scenario_type,uniform_comp=False)
#             for run_filename in run_filenames_onescenario:
#                 run_filenames.append(run_filename)
#         else:
#             print('not yet coded without split repeats!')
    
#     for ii,run_filename in enumerate(run_filenames):
#         scenario_runner.send_run_to_slurm(run_filename, runfiles_dir, run_time_str=run_time_str)
        
#     return ensemble_settings,mam_ensemble_settings

    
def go_allFromE3SM(
        ensemble_name,e3sm_filename,
        dry = False,
        scenario_type='4mode_noemit',
        run_time_min = 30., # minutes
        N_samples=100,
        n_part = 10000,
        n_repeat = 10,        
        allow_halving = 'yes',
        allow_doubling = 'yes',
        t_max = 3600.*4.,
        del_t = 60.,
        t_output = 600.,
        weighting_exponent=0,
        split_repeats=True,
        min_frac_from_prod=None,
        mode_sigs = np.log([1.8,1.6,1.8,1.6]),
        ht=500.,
        span_observed_flux=False,
        correlate_flux_with_num=False,
        flux_dist = 'loguniform',#'lognormal',#'loguniform',
        flux_param1 = 1e-2*1e-9,#0.1*1e-9,#1e-2*1e-9,
        flux_param2 = 1e1*1e-9,#1.5,#
        sigma_log10_flux=0.5,
        no_soa=True,only_subsaturated_cells=True,
        project = 'nuvola'):
    
    ensemble_dir = scenario_runner.get_store_over_dir() + 'box_simulations3/' + ensemble_name + '/'
    if not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)
    
    scenarios_dir = ensemble_dir + 'scenarios/'
    if not os.path.exists(scenarios_dir):
        os.mkdir(scenarios_dir)
    
    runfiles_dir = ensemble_dir + 'runfiles/'
    if not os.path.exists(runfiles_dir):
        os.mkdir(runfiles_dir)
    
    ensemble_settings,mam_ensemble_settings = ensemble_maker.get_lhs_ensemble_settings_allFromE3SM(
            ensemble_dir,e3sm_filename,N_samples,
            dry = dry,
            mode_sigs = mode_sigs,
            n_part = n_part,
            n_repeat = n_repeat,     
            t_max = t_max,
            weighting_exponent=weighting_exponent,
            allow_halving = allow_halving,
            allow_doubling = allow_doubling,
            del_t = del_t,t_output = t_output,
            ht = ht,
            no_soa=no_soa,
            min_frac_from_prod=min_frac_from_prod,
            span_observed_flux=span_observed_flux,
            correlate_flux_with_num=correlate_flux_with_num,
            flux_dist = flux_dist, flux_param1 = flux_param1, flux_param2 = flux_param2,
            sigma_log10_flux=sigma_log10_flux,
            only_subsaturated_cells=only_subsaturated_cells)
    
    run_time_str = scenario_runner.get_run_time_str(run_time_min)
    
    run_filenames = []
    for ii,settings in enumerate(ensemble_settings):
        print_scenario_here = scenarios_dir +  str(ii).zfill(6) + '/'
        if not os.path.exists(print_scenario_here):
            os.mkdir(print_scenario_here)
        
        if split_repeats:
            run_filenames_onescenario = scenario_runner.splitRepats(print_scenario_here,settings,scenario_type=scenario_type,uniform_comp=False)
            for run_filename in run_filenames_onescenario:
                run_filenames.append(run_filename)
        else:
            print('not yet coded without split repeats!')
    
    for ii,run_filename in enumerate(run_filenames):
        scenario_runner.send_run_to_slurm(run_filename, runfiles_dir, run_time_str=run_time_str, project=project)
        
    return ensemble_settings,mam_ensemble_settings

def go_otherProcesses(
        orig_ensemble_name,
        updated_ensemble_name,
        run_time_min=30,
        split_repeats=True,
        scenario_type='4mode_noemit',uniform_comp=False):
    
    run_time_str = scenario_runner.get_run_time_str(run_time_min)
    
    orig_ensemble_dir = scenario_runner.get_store_over_dir() + 'box_simulations3/' + orig_ensemble_name + '/'
    updated_ensemble_dir = scenario_runner.get_store_over_dir() + 'box_simulations3/' + updated_ensemble_name + '/'
    if not os.path.exists(updated_ensemble_dir):
        os.mkdir(updated_ensemble_dir)
    
    scenarios_dir = updated_ensemble_dir + 'scenarios/'
    if not os.path.exists(scenarios_dir):
        os.mkdir(scenarios_dir)
    
    runfiles_dir = updated_ensemble_dir + 'runfiles/'
    if not os.path.exists(runfiles_dir):
        os.mkdir(runfiles_dir)
    
    updated_ensemble_settings,updated_mam_ensemble_settings = ensemble_maker.update_process_settings(
            orig_ensemble_dir,updated_ensemble_dir)
    
    run_filenames = []
    for ii,settings in enumerate(updated_ensemble_settings):
        print_scenario_here = scenarios_dir +  str(ii).zfill(6) + '/'
        if not os.path.exists(print_scenario_here):
            os.mkdir(print_scenario_here)
        
        if split_repeats:
            run_filenames_onescenario = scenario_runner.splitRepats(print_scenario_here,settings,scenario_type=scenario_type,uniform_comp=False)
            for run_filename in run_filenames_onescenario:
                run_filenames.append(run_filename)
        else:
            print('not yet coded without split repeats!')
    
    
    for ii,run_filename in enumerate(run_filenames):
        scenario_runner.send_run_to_slurm(run_filename, runfiles_dir, run_time_str=run_time_str)
        
    return updated_ensemble_settings,updated_mam_ensemble_settings

    # print('len(run_filenames)',len(run_filenames))
    
    # sh_filename = 'ensemble_runfiles/run_' + ensemble_name + '.sh'
    # if os.path.exists('/people/fier887/'):
    #     sh_lines = [
    #         '#!/bin/tcsh',
    #         '#SBATCH -A sooty2',
    #         '#SBATCH -p slurm',
    #         '#SBATCH -t ' + run_time_str,
    #         '#SBATCH -N 1',
    #         '#SBATCH -o ensemble_runfiles/run_' + ensemble_name + '.out',
    #         '#SBATCH -e ensemble_runfiles/run_' + ensemble_name + '.err',
    #         '']
    #         # '#!/bin/bash',
    #         # '#SBATCH --job-name=run_' + ensemble_name,
    #         # '#SBATCH --account=sooty',
    #         # '#SBATCH --nodes=1',
    #         # '#SBATCH --time=02:00:00',
    #         # '#SBATCH --error=run_' + ensemble_name + '.err',
    #         # '#SBATCH --output=run_' + ensemble_name + '.out',
    #         # '']
    # elif os.path.exists('/global/homes/l/lfierce'):
    #     sh_lines = [
    #         '#!/bin/bash',
    #         '#SBATCH --job-name=run_' + ensemble_name,
    #         '#SBATCH --qos=flex',
    #         '#SBATCH --constraint=knl',
    #         '#SBATCH --account=m3525',
    #         '#SBATCH --nodes=1',
    #         '#SBATCH --time=02:00:00',
    #         '#SBATCH --time-min=00:30:00',
    #         '#SBATCH --error=%x-%j.err',
    #         '#SBATCH --output=%x-%j.out',
    #         '']
    # for ii,run_filename in enumerate(run_filenames):
    #     python_filename = scenario_runner.create_python_driver(run_filename,runfiles_dir=runfiles_dir)
    #     sh_lines.append('srun python ' + python_filename) # this isn't right --> need to run python file that checks if run is done
    # with open(sh_filename, 'w') as f:
    #     for line in sh_lines:
    #         f.write(line)
    #         f.write('\n')
    # os.system('chmod +x ' + sh_filename)
    # os.system('sbatch ' + sh_filename)    
    
# def requeue_unstarted(ensemble_dir,n_repeat,run_time=30,qos='regular'):
#     all_scenarios = process_readVariables.get_scenarios(ensemble_dir)
#     runfiles_dir = ensemble_dir + '../runfiles/'
#     for scenario in all_scenarios:
#         scenario_dir = ensemble_dir + scenario + '/'
#         for ii in range(n_repeat):
#             repeat_id = str(int(ii+1)).zfill(4)
            
#             # state = np.loadtxt(scenario_dir + 'state_' + repeat_id + '.txt')
#             if state<2:
#                 run_time_str = scenario_runner.get_run_time_str(run_time)
#                 run_filename = scenario_dir + 'run_' + repeat_id + '.sh'
#                 scenario_runner.send_run_to_slurm(run_filename, runfiles_dir, run_time_str=run_time_str,qos=qos)
#                 print('requeued', scenario, repeat_id)
                
                
def requeue(ensemble_dir,n_repeat,run_time=12*60,qos='regular'):
    all_scenarios = process_readVariables.get_scenarios(ensemble_dir)
    runfiles_dir = ensemble_dir + '../runfiles/'
    for scenario in all_scenarios:
        scenario_dir = ensemble_dir + scenario + '/'
        for ii in range(n_repeat):
            repeat_id = str(int(ii+1)).zfill(4)
            state = np.loadtxt(scenario_dir + 'state_' + repeat_id + '.txt')
            if state<2:
                run_time_str = scenario_runner.get_run_time_str(run_time)
                run_filename = scenario_dir + 'run_' + repeat_id + '.sh'
                scenario_runner.send_run_to_slurm(run_filename, runfiles_dir, run_time_str=run_time_str,qos=qos)
                print('requeued', scenario, repeat_id)
            # else:
            #     print('already done!', scenario, repeat_id)