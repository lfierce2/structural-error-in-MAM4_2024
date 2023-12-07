#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions needed to create and run an ensemble of PartMC-MOSAIC scenarios.

Code was tested with the following model versions: partmc-2.6.1 and mosaic-2012-01-25

@author: Laura Fierce
"""

import numpy as np
import constants as c
from scipy.constants import R
from scipy.stats import norm
from pyDOE import lhs
from scipy.special import erfinv
import pickle
import os
from netCDF4 import Dataset
import analyses

def get_lhs_ensemble_settings(
        ensemble_dir,N_samples,
        N_all_varnames = ['tot_num','frac_accum','nonaccum_frac_coarse','ofaccum_frac_fresh'],
        N_all_dists = ['loguniform','uniform','loguniform','loguniform'],
        N_all_param1 = [3e7,0.,1e-8,1e-13],
        N_all_param2 = [2e12,1.,1.,1.],
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
        n_part = 10000,
        n_repeat = 10,        
        t_max = 3600.*4.,
        del_t = 60.,t_output = 600.,
        min_frac_from_prod=0.,
        p=101325.,ht=500.,
        weighting_exponent=0):
    
    ensemble_name = get_ensemble_name(ensemble_dir)
    numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals, mfso42_vals, mfsoa2_vals, mfncl2_vals, mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals, mfbc4_vals, mfpom4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals, tmp_vals = sample_inputs(
        N_samples, 
        N_all_varnames = N_all_varnames, N_all_dists = N_all_dists, N_all_param1 = N_all_param1, N_all_param2 = N_all_param2,
        dgm_all_varnames = dgm_all_varnames, dgm_all_dists = dgm_all_dists, dgm_all_param1 = dgm_all_param1, dgm_all_param2 = dgm_all_param2,
        mode_sigs = mode_sigs,
        flux_dist = flux_dist, flux_param1 = flux_param1, flux_param2 = flux_param2,
        rh_dist = rh_dist, rh_param1 = rh_param1, rh_param2 = rh_param2,
        temp_dist = temp_dist, temp_param1 = temp_param1, temp_param2 = temp_param2,
        t_max = t_max, min_frac_from_prod = min_frac_from_prod,p = p)
    
    process = get_process(ensemble_name)    
    # start_time = time.time()
    if process == 'nada':
        do_mosaic = 'no'
        do_coagulation = 'no'
        do_nucleation = 'no'
    elif process == 'both':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'cond':
        do_mosaic = 'yes'
        do_coagulation = 'no'
        do_nucleation = 'no'
    elif process == 'coag':
        do_mosaic = 'no'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'all':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'yes'
    elif process == 'noNPF':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'noCoag':
        do_mosaic = 'yes'
        do_coagulation = 'no'
        do_nucleation = 'yes'
    
    if do_coagulation == 'yes':
        mdo_coag = 1
    else:
        mdo_coag = 0
    
    if do_mosaic == 'yes':
        mdo_gasaerexch = 1
    else:
        mdo_gasaerexch = 0
        
    if do_nucleation == 'yes':
        mdo_newnuc = 1
    else:
        mdo_newnuc = 0
    
    mdo_rename = 1
    
    mam_dt = int(del_t)
    mam_output_intvl = int(t_output/del_t)
    mam_nstep = int(t_max/mam_dt)
    
    ensemble_settings = []
    mam_ensemble_settings = []
    mam4_spec_names = ['bc','pom','dst','ncl','soa','so4']
    modes = range(1,5)
    for qq,(numc1,numc2,numc3,numc4,mu1,mu2,mu3,mu4,mfso41,mfpom1,mfsoa1,mfbc1,mfdst1,mfncl1,mfso42,mfsoa2,mfncl2,
         mfso43,mfpom3,mfsoa3,mfbc3,mfdst3,mfncl3,mfbc4,mfpom4,h2so4_chem_prod_rate,qh2so4,rh_val,temp_val) in enumerate(zip(
            numc1_vals,numc2_vals,numc3_vals,numc4_vals,mu1_vals,mu2_vals,mu3_vals,mu4_vals,
            mfso41_vals,mfpom1_vals,mfsoa1_vals,mfbc1_vals,mfdst1_vals,mfncl1_vals,
            mfso42_vals,mfsoa2_vals,mfncl2_vals,mfso43_vals,mfpom3_vals,mfsoa3_vals,mfbc3_vals,mfdst3_vals,mfncl3_vals,mfbc4_vals,mfpom4_vals,
            h2so4_chem_prod_rate_vals,qh2so4_vals,rh_vals,tmp_vals)):
        
        if h2so4_chem_prod_rate<0:
            h2so4_chem_prod_rate = 0.
        
        tot1 = mfso41 + mfpom1 + mfsoa1 + mfbc1 + mfdst1 + mfncl1
        tot2 = mfso42 + mfsoa2 + mfncl2
        tot3 = mfso43 + mfpom3 + mfsoa3 + mfbc3 + mfdst3 + mfncl3
        tot4 = mfbc4 + mfpom4
        
        mfso41 = mfso41/tot1
        mfpom1 = mfpom1/tot1
        mfsoa1 = mfsoa1/tot1
        mfbc1 = mfbc1/tot1
        mfdst1 = mfdst1/tot1
        mfncl1 = mfncl1/tot1
        
        mfso42 = mfso42/tot2
        mfso42 = mfsoa2/tot2
        mfso42 = mfncl2/tot2
        
        mfso43 = mfso43/tot3
        mfpom3 = mfpom3/tot3
        mfsoa3 = mfsoa3/tot3
        mfbc3 = mfbc3/tot3
        mfdst3 = mfdst3/tot3
        mfncl3 = mfncl3/tot3
        
        mfbc4 = mfbc4/tot4
        mfpom4 = mfpom4/tot4
        
        # move all of this into its own function? (make the option of mam4_specs vs normal partmc_specs)
        aero_init = {'dist':{}}
        # mode_dat = {}
        
        for ii,mode in enumerate(modes):
            aero_init['dist'][mode] = {'diam_type':'geometric','mode_type':'log_normal'}
            aero_init['dist'][mode]['num_conc'] = eval('numc' + str(mode))
            aero_init['dist'][mode]['geom_mean_diam'] = np.exp(eval('mu' + str(mode)))
            aero_init['dist'][mode]['log10_geom_std_dev'] = mode_sigs[ii]

            aero_init['dist'][mode]['comp'] = {}
            for mam4_spec_name in mam4_spec_names:
                try:
                    mam4_mass_frac = eval('mf' + mam4_spec_name + str(mode))
                except:
                    mam4_mass_frac = 0.
                
                partmc_spec_names,partmc_mass_fracs = mam4_to_partmc_composition(mam4_spec_name,mam4_mass_frac)
                for (partmc_spec_name,partmc_mass_frac) in zip(partmc_spec_names,partmc_mass_fracs):
                    aero_init['dist'][mode]['comp'][partmc_spec_name] = partmc_mass_frac
                    print(partmc_spec_name,partmc_mass_frac)
        h2so4_emission_rate = h2so4_chem_prod_rate*ht*p/(R*temp_val) #ht/0.0224 
        h2so4_ppb = qh2so4*1e9 # MW_h2so4/MW_air*
        
        height = {}
        height['time'] = 0.
        height['height'] = ht
        
        temp = {}
        temp['time'] = 0.
        temp['temp'] = temp_val
        
        pres = {}
        pres['time'] = 0.
        pres['pressure'] = p

        spec = {
                'n_part':int(n_part),
                'n_repeat':int(n_repeat),
                't_max':int(t_max),
                'del_t':del_t,
                't_output':int(t_output),
                'do_coagulation':do_coagulation,
                'do_mosaic':do_mosaic,
                'rel_humidity':rh_val,
                'do_nucleation':do_nucleation,
                'weighting_exponent':weighting_exponent}
        
        partmc_settings = {
                'aero_init':aero_init,
                'gas_init':{'H2SO4':h2so4_ppb},
                'gas_emit':{'time':0., 'rate':1.,'H2SO4':h2so4_emission_rate},
                'height':height,
                'temp':temp,
                'pres':pres,
                'spec':spec}
        
        ensemble_settings.append(partmc_settings)
        
        mam_settings = {
            'scenario':str(qq).zfill(6),
            'chem_input':{
                    'numc1':numc1,'numc2':numc2,'numc3':numc3,'numc4':numc4,
                    'mfso41':mfso41,'mfpom1':mfpom1,'mfsoa1':mfsoa1,'mfbc1':mfbc1,'mfdst1':mfdst1,'mfncl1':mfncl1,
                    'mfso42':mfso42,'mfsoa2':mfsoa2,'mfncl2':mfncl2,
                    'mfso43':mfso43,'mfpom3':mfpom3,'mfsoa3':mfsoa3,'mfbc3':mfbc3,'mfdst3':mfdst3,'mfncl3':mfncl3,
                    'mfbc4':mfbc4,'mfpom4':mfpom4,
                    'h2so4_chem_prod_rate':h2so4_chem_prod_rate,'qh2so4':qh2so4},
            'time_input':{'mam_dt':mam_dt, 'mam_nstep':mam_nstep,'mam_output_intvl':mam_output_intvl},
            'cntl_input':{'mdo_gasaerexch':mdo_gasaerexch,'mdo_coag':mdo_coag,'gas_unit':0,'mdo_rename':mdo_rename,'mdo_newnuc':mdo_newnuc},
            'met_input':{'press':101325.,'RH_CLEA':rh_val},
            'psd_params':{
                'mu1':mu1,'mu2':mu2,'mu3':mu3,'mu4':mu4,
                'sig1':mode_sigs[0],'sig2':mode_sigs[1],'sig3':mode_sigs[2],'sig4':mode_sigs[3]}}
        mam_ensemble_settings.append(mam_settings)
    if not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)
    
    pickle.dump(ensemble_settings,open(ensemble_dir + 'ensemble_settings.pkl','wb'),protocol=0)
    pickle.dump(mam_ensemble_settings,open(ensemble_dir + 'mam_ensemble_settings.pkl','wb'),protocol=0)
    
    return ensemble_settings, mam_ensemble_settings

def get_lhs_ensemble_settings_allFromE3SM(
        ensemble_dir,e3sm_filename,N_samples,
        dry=False,
        # fractionSurface=0.9,useSorted=False,
        # N_all_varnames = ['tot_num','frac_accum','nonaccum_frac_coarse','ofaccum_frac_fresh'],
        # N_all_dists = ['loguniform','uniform','loguniform','loguniform'],
        # # N_all_param1 = [3e7,0.,1e-8,1e-13],
        # N_all_param2 = [2e12,1.,1.,1.],
        # dgm_all_varnames = ['dgm_accum','dgm_aik','dgm_coarse','dgm_fresh'],
        # dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
        # dgm_all_param1 = [0.5e-7, 0.5e-8, 1.000e-6, 1e-8],
        # dgm_all_param2 = [1.1e-7, 3e-8, 2.000e-6, 6e-8],            
        mode_sigs = np.log([1.8,1.6,1.8,1.6]),
        # mode1_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
        # mode1_dists = ['uniform','uniform','uniform','uniform','uniform'],
        # mode1_param1 = [0.,0.,0.,0.,0.],
        # mode1_param2 = [1.,1.,1.,1.,1.],
        # mode2_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl'],
        # mode2_dists = ['uniform','uniform'],
        # mode2_param1 = [1.,0.],
        # mode2_param2 = [1.,0.],
        # mode3_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
        # mode3_dists = ['uniform','uniform','uniform','uniform','uniform'],
        # mode3_param1 = [0.,0.,0.,0.,0.],
        # mode3_param2 = [1.,1.,1.,1.,1.],
        # mode4_varnames = ['frac_bc'],
        # mode4_dists = ['uniform'],
        # mode4_param1 = [0.],
        # mode4_param2 = [1.],
        # flux_dist = 'loguniform',
        # flux_param1 = 1e-2*1e-9,
        # flux_param2 = 1e1*1e-9,
        # rh_dist='uniform',
        # rh_param1 = 0.,
        # rh_param2 = 0.99,
        # temp_dist='uniform',
        # temp_param1 = 240.,
        # temp_param2 = 310.,
        n_part = 10000,
        n_repeat = 10,        
        t_max = 3600.*4.,
        del_t = 60.,t_output = 600.,
        allow_halving = 'yes',
        allow_doubling = 'yes',
        weighting_exponent=0,
        min_frac_from_prod=None,
        span_observed_flux=False,
        correlate_flux_with_num=False,
        flux_dist = 'loguniform',#'lognormal',#'loguniform',
        flux_param1 = 1e-2*1e-9,#0.1*1e-9,#1e-2*1e-9,
        flux_param2 = 1e1*1e-9,#1.5,#
        no_soa=True,only_subsaturated_cells=True,
        sigma_log10_flux=0.5,
        # min_frac_from_prod=0.,
        # p=101325.,
        ht=500.):
    
    ensemble_name = get_ensemble_name(ensemble_dir)
    numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals, mfso42_vals, mfsoa2_vals, mfncl2_vals, mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals, mfbc4_vals, mfpom4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals, tmp_vals, press_vals, indices = sample_inputs_allFromE3SM(
        N_samples, e3sm_filename,#fractionSurface=fractionSurface, useSorted=useSorted,
        # N_all_varnames = N_all_varnames, N_all_dists = N_all_dists, N_all_param1 = N_all_param1, N_all_param2 = N_all_param2,
        # dgm_all_varnames = dgm_all_varnames, dgm_all_dists = dgm_all_dists, dgm_all_param1 = dgm_all_param1, dgm_all_param2 = dgm_all_param2,
        mode_sigs = mode_sigs,
        flux_dist = flux_dist, flux_param1 = flux_param1, flux_param2 = flux_param2,
        # rh_dist = rh_dist, rh_param1 = rh_param1, rh_param2 = rh_param2,
        # temp_dist = temp_dist, temp_param1 = temp_param1, temp_param2 = temp_param2,
        min_frac_from_prod=min_frac_from_prod,
        t_max = t_max,
        span_observed_flux=span_observed_flux,
        correlate_flux_with_num=correlate_flux_with_num,
        sigma_log10_flux=sigma_log10_flux,
        no_soa=no_soa,only_subsaturated_cells=only_subsaturated_cells)#, min_frac_from_prod = min_frac_from_prod,p = p)
    
    process = get_process(ensemble_name)
    # start_time = time.time()
    if process == 'nada':
        do_mosaic = 'no'
        do_coagulation = 'no'
        do_nucleation = 'no'
    elif process == 'both':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'cond':
        do_mosaic = 'yes'
        do_coagulation = 'no'
        do_nucleation = 'no'
    elif process == 'coag':
        do_mosaic = 'no'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'all':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'yes'
    elif process == 'noNPF':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'noCoag':
        do_mosaic = 'yes'
        do_coagulation = 'no'
        do_nucleation = 'yes'
    
    if do_coagulation == 'yes':
        mdo_coag = 1
    else:
        mdo_coag = 0
    
    if do_mosaic == 'yes':
        mdo_gasaerexch = 1
    else:
        mdo_gasaerexch = 0
        
    if do_nucleation == 'yes':
        mdo_newnuc = 1
    else:
        mdo_newnuc = 0
    
    mdo_rename = 1
    
    mam_dt = int(del_t)
    mam_output_intvl = int(t_output/del_t)
    mam_nstep = int(t_max/mam_dt)
    
    ensemble_settings = []
    mam_ensemble_settings = []
    mam4_spec_names = ['bc','pom','dst','ncl','soa','so4']
    modes = range(1,5)
    for qq,(numc1,numc2,numc3,numc4,mu1,mu2,mu3,mu4,mfso41,mfpom1,mfsoa1,mfbc1,mfdst1,mfncl1,mfso42,mfsoa2,mfncl2,
         mfso43,mfpom3,mfsoa3,mfbc3,mfdst3,mfncl3,mfbc4,mfpom4,h2so4_chem_prod_rate,qh2so4,rh_val,temp_val,press_val,idx_val) in enumerate(zip(
            numc1_vals,numc2_vals,numc3_vals,numc4_vals,mu1_vals,mu2_vals,mu3_vals,mu4_vals,
            mfso41_vals,mfpom1_vals,mfsoa1_vals,mfbc1_vals,mfdst1_vals,mfncl1_vals,
            mfso42_vals,mfsoa2_vals,mfncl2_vals,mfso43_vals,mfpom3_vals,mfsoa3_vals,mfbc3_vals,mfdst3_vals,mfncl3_vals,mfbc4_vals,mfpom4_vals,
            h2so4_chem_prod_rate_vals,qh2so4_vals,rh_vals,tmp_vals,press_vals,indices)):
        if dry:
            rh_val = 0.
        # move all of this into its own function? (make the option of mam4_specs vs normal partmc_specs)
        aero_init = {'dist':{}}
        # mode_dat = {}
        
        for ii,mode in enumerate(modes):
            aero_init['dist'][mode] = {'diam_type':'geometric','mode_type':'log_normal'}
            aero_init['dist'][mode]['num_conc'] = eval('numc' + str(mode))
            aero_init['dist'][mode]['geom_mean_diam'] = np.exp(eval('mu' + str(mode)))
            aero_init['dist'][mode]['log10_geom_std_dev'] = np.log10(np.exp(mode_sigs[ii]))

            aero_init['dist'][mode]['comp'] = {}
            for mam4_spec_name in mam4_spec_names:
                try:
                    mam4_mass_frac = eval('mf' + mam4_spec_name + str(mode))
                except:
                    mam4_mass_frac = 0.
                
                partmc_spec_names,partmc_mass_fracs = mam4_to_partmc_composition(mam4_spec_name,mam4_mass_frac)
                for (partmc_spec_name,partmc_mass_frac) in zip(partmc_spec_names,partmc_mass_fracs):
                    aero_init['dist'][mode]['comp'][partmc_spec_name] = partmc_mass_frac
                    print(partmc_spec_name,partmc_mass_frac)
        
        h2so4_emission_rate = h2so4_chem_prod_rate*ht*press_val/(R*temp_val) #ht/0.0224 
        h2so4_ppb = qh2so4*1e9 # MW_h2so4/MW_air*
        
        height = {}
        height['time'] = 0.
        height['height'] = ht
        
        temp = {}
        temp['time'] = 0.
        temp['temp'] = temp_val
        
        pres = {}
        pres['time'] = 0.
        pres['pressure'] = press_val

        spec = {
                'n_part':int(n_part),
                'n_repeat':int(n_repeat),
                't_max':int(t_max),
                'del_t':del_t,
                't_output':int(t_output),
                'allow_doubling':allow_doubling,
                'allow_halving':allow_halving,
                'do_coagulation':do_coagulation,
                'do_mosaic':do_mosaic,
                'rel_humidity':rh_val,
                'do_nucleation':do_nucleation,
                'weighting_exponent':weighting_exponent}
        
        partmc_settings = {
                # 'scenario':str(qq).zfill(6),
                # 'e3sm_index':idx_val,
                'aero_init':aero_init,
                'gas_init':{'H2SO4':h2so4_ppb},
                'gas_emit':{'time':0., 'rate':1.,'H2SO4':h2so4_emission_rate},
                'height':height,
                'temp':temp,
                'pres':pres,
                'spec':spec}
        
        ensemble_settings.append(partmc_settings)
        
        mam_settings = {
            'scenario':str(qq).zfill(6),
            'e3sm_index':idx_val,
            'sa_flux':avg_vol_flux[qq],
            'chem_input':{
                    'numc1':numc1,'numc2':numc2,'numc3':numc3,'numc4':numc4,
                    'mfso41':mfso41,'mfpom1':mfpom1,'mfsoa1':mfsoa1,'mfbc1':mfbc1,'mfdst1':mfdst1,'mfncl1':mfncl1,
                    'mfso42':mfso42,'mfsoa2':mfsoa2,'mfncl2':mfncl2,
                    'mfso43':mfso43,'mfpom3':mfpom3,'mfsoa3':mfsoa3,'mfbc3':mfbc3,'mfdst3':mfdst3,'mfncl3':mfncl3,
                    'mfbc4':mfbc4,'mfpom4':mfpom4,
                    'h2so4_chem_prod_rate':h2so4_chem_prod_rate,'qh2so4':qh2so4},
            'time_input':{'mam_dt':mam_dt, 'mam_nstep':mam_nstep,'mam_output_intvl':mam_output_intvl},
            'cntl_input':{'mdo_gasaerexch':mdo_gasaerexch,'mdo_coag':mdo_coag,'gas_unit':0,'mdo_rename':mdo_rename,'mdo_newnuc':mdo_newnuc},
            'met_input':{'press':101325.,'RH_CLEA':rh_val},
            'psd_params':{
                'mu1':mu1,'mu2':mu2,'mu3':mu3,'mu4':mu4,
                'sig1':mode_sigs[0],'sig2':mode_sigs[1],'sig3':mode_sigs[2],'sig4':mode_sigs[3]}}
        mam_ensemble_settings.append(mam_settings)
    if not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)
    
    pickle.dump(ensemble_settings,open(ensemble_dir + 'ensemble_settings.pkl','wb'),protocol=0)
    pickle.dump(mam_ensemble_settings,open(ensemble_dir + 'mam_ensemble_settings.pkl','wb'),protocol=0)
    
    return ensemble_settings, mam_ensemble_settings

def update_process_settings(
        orig_ensemble_dir,updated_ensemble_dir):
    
    ensemble_settings = pickle.load(open(orig_ensemble_dir + 'ensemble_settings.pkl','rb'))
    mam_ensemble_settings = pickle.load(open(orig_ensemble_dir + 'mam_ensemble_settings.pkl','rb'))    
    
    ensemble_name = get_ensemble_name(updated_ensemble_dir)
    
    process = get_process(ensemble_name)
    if process == 'nada':
        do_mosaic = 'no'
        do_coagulation = 'no'
        do_nucleation = 'no'
    elif process == 'both':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'cond':
        do_mosaic = 'yes'
        do_coagulation = 'no'
        do_nucleation = 'no'
    elif process == 'coag':
        do_mosaic = 'no'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'all':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'yes'
    elif process == 'noNPF':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'noCoag':
        do_mosaic = 'yes'
        do_coagulation = 'no'
        do_nucleation = 'yes'
    
    if do_coagulation == 'yes':
        mdo_coag = 1
    else:
        mdo_coag = 0
    
    if do_mosaic == 'yes':
        mdo_gasaerexch = 1
    else:
        mdo_gasaerexch = 0
        
    if do_nucleation == 'yes':
        mdo_newnuc = 1
    else:
        mdo_newnuc = 0
    
    updated_ensemble_settings = ensemble_settings
    updated_mam_ensemble_settings = mam_ensemble_settings
    for qq in range(len(ensemble_settings)):
        spec = ensemble_settings[qq]['spec']
        spec['do_coagulation'] = do_coagulation
        spec['do_mosaic'] = do_mosaic
        spec['do_nucleation'] = do_nucleation
        updated_ensemble_settings[qq]['spec'] = spec
        
        cntl_input = mam_ensemble_settings[qq]['cntl_input']
        cntl_input['mdo_gasaerexch'] = mdo_gasaerexch
        cntl_input['mdo_coag'] = mdo_coag
        cntl_input['mdo_newnuc'] = mdo_newnuc
        updated_mam_ensemble_settings[qq]['cntl_input'] = cntl_input
        
    pickle.dump(updated_ensemble_settings,open(updated_ensemble_dir + 'ensemble_settings.pkl','wb'),protocol=0)
    pickle.dump(updated_mam_ensemble_settings,open(updated_ensemble_dir + 'mam_ensemble_settings.pkl','wb'),protocol=0)
    
    return updated_ensemble_settings, updated_mam_ensemble_settings
# def get_lhs_ensemble_settings_allFromE3SM(
#         ensemble_dir,e3sm_filename,N_samples,
#         fractionSurface=0.9,useSorted=False,
#         mode_sigs = np.log([1.8,1.6,1.6,1.8]),
#         n_part = 10000,
#         n_repeat = 10,
#         t_max = 3600.*4.,
#         del_t = 60.,t_output = 600.,
#         min_frac_from_prod=0.,
#         ht = 500, p=101325.):
    
#     ensemble_name = get_ensemble_name(ensemble_dir)
#     numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals, mfso42_vals, mfsoa2_vals, mfncl2_vals, mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals, mfbc4_vals, mfpom4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals, tmp_vals = sample_inputs_fromE3SM(
#         N_samples, e3sm_filename, fractionSurface=fractionSurface, useSorted=useSorted, mode_sigs = mode_sigs)
    
#     process = get_process(ensemble_name)    
#     # start_time = time.time()
#     if process == 'nada':
#         do_mosaic = 'no'
#         do_coagulation = 'no'
#         do_nucleation = 'no'
#     elif process == 'both':
#         do_mosaic = 'yes'
#         do_coagulation = 'yes'
#         do_nucleation = 'no'
#     elif process == 'cond':
#         do_mosaic = 'yes'
#         do_coagulation = 'no'
#         do_nucleation = 'no'
#     elif process == 'coag':
#         do_mosaic = 'no'
#         do_coagulation = 'yes'
#         do_nucleation = 'no'
#     elif process == 'all':
#         do_mosaic = 'yes'
#         do_coagulation = 'yes'
#         do_nucleation = 'yes'
#     elif process == 'noNPF':
#         do_mosaic = 'yes'
#         do_coagulation = 'yes'
#         do_nucleation = 'no'
#     elif process == 'noCoag':
#         do_mosaic = 'yes'
#         do_coagulation = 'no'
#         do_nucleation = 'yes'
    
#     if do_coagulation == 'yes':
#         mdo_coag = 1
#     else:
#         mdo_coag = 0
    
#     if do_mosaic == 'yes':
#         mdo_gasaerexch = 1
#     else:
#         mdo_gasaerexch = 0
        
#     if do_nucleation == 'yes':
#         mdo_newnuc = 1
#     else:
#         mdo_newnuc = 0
    
#     mdo_rename = 1
    
#     mam_dt = int(del_t)
#     mam_output_intvl = int(t_output/del_t)
#     mam_nstep = int(t_max/mam_dt)
    
#     ensemble_settings = []
#     mam_ensemble_settings = []
#     mam4_spec_names = ['bc','pom','dst','ncl','soa','so4']
#     modes = range(1,5)
#     modes = range(1,5)
#     for qq,(numc1,numc2,numc3,numc4,mu1,mu2,mu3,mu4,mfso41,mfpom1,mfsoa1,mfbc1,mfdst1,mfncl1,mfso42,mfsoa2,mfncl2,
#          mfso43,mfpom3,mfsoa3,mfbc3,mfdst3,mfncl3,mfbc4,mfpom4,h2so4_chem_prod_rate,qh2so4,rh_val,temp_val) in enumerate(zip(
#             numc1_vals,numc2_vals,numc3_vals,numc4_vals,mu1_vals,mu2_vals,mu3_vals,mu4_vals,
#             mfso41_vals,mfpom1_vals,mfsoa1_vals,mfbc1_vals,mfdst1_vals,mfncl1_vals,
#             mfso42_vals,mfsoa2_vals,mfncl2_vals,mfso43_vals,mfpom3_vals,mfsoa3_vals,mfbc3_vals,mfdst3_vals,mfncl3_vals,mfbc4_vals,mfpom4_vals,
#             h2so4_chem_prod_rate_vals,qh2so4_vals,rh_vals,tmp_vals)):
        
#         # move all of this into its own function? (make the option of mam4_specs vs normal partmc_specs)
#         aero_init = {'dist':{}}
#         # mode_dat = {}
        
#         for ii,mode in enumerate(modes):
#             aero_init['dist'][mode] = {'diam_type':'geometric','mode_type':'log_normal'}
#             aero_init['dist'][mode]['num_conc'] = eval('numc' + str(mode))
#             aero_init['dist'][mode]['geom_mean_diam'] = np.exp(eval('mu' + str(mode)))
#             aero_init['dist'][mode]['log10_geom_std_dev'] = np.log10(np.exp(mode_sigs[ii]))

#             aero_init['dist'][mode]['comp'] = {}
#             for mam4_spec_name in mam4_spec_names:
#                 try:
#                     mam4_mass_frac = eval('mf' + mam4_spec_name + str(mode))
#                 except:
#                     mam4_mass_frac = 0.
                
#                 partmc_spec_names,partmc_mass_fracs = mam4_to_partmc_composition(mam4_spec_name,mam4_mass_frac)
#                 for (partmc_spec_name,partmc_mass_frac) in zip(partmc_spec_names,partmc_mass_fracs):
#                     aero_init['dist'][mode]['comp'][partmc_spec_name] = partmc_mass_frac
#                     print(partmc_spec_name,partmc_mass_frac)
#         h2so4_emission_rate = h2so4_chem_prod_rate*ht*p/(R*temp_val) #ht/0.0224 
#         h2so4_ppb = qh2so4*1e9 # MW_h2so4/MW_air*
        
#         height = {}
#         height['time'] = 0.
#         height['height'] = ht
        
#         temp = {}
#         temp['time'] = 0.
#         temp['temp'] = temp_val
        
#         pres = {}
#         pres['time'] = 0.
#         pres['pressure'] = p

#         spec = {
#                 'n_part':int(n_part),
#                 'n_repeat':int(n_repeat),
#                 't_max':int(t_max),
#                 'del_t':del_t,
#                 't_output':int(t_output),
#                 'do_coagulation':do_coagulation,
#                 'do_mosaic':do_mosaic,
#                 'rel_humidity':rh_val,
#                 'do_nucleation':do_nucleation}
        
#         partmc_settings = {
#                 'aero_init':aero_init,
#                 'gas_init':{'H2SO4':h2so4_ppb},
#                 'gas_emit':{'time':0., 'rate':1.,'H2SO4':h2so4_emission_rate},
#                 'height':height,
#                 'temp':temp,
#                 'pres':pres,
#                 'spec':spec}
        
#         ensemble_settings.append(partmc_settings)
        
#         mam_settings = {
#             'scenario':str(qq).zfill(6),
#             'chem_input':{
#                     'numc1':numc1,'numc2':numc2,'numc3':numc3,'numc4':numc4,
#                     'mfso41':mfso41,'mfpom1':mfpom1,'mfsoa1':mfsoa1,'mfbc1':mfbc1,'mfdst1':mfdst1,'mfncl1':mfncl1,
#                     'mfso42':mfso42,'mfsoa2':mfsoa2,'mfncl2':mfncl2,
#                     'mfso43':mfso43,'mfpom3':mfpom3,'mfsoa3':mfsoa3,'mfbc3':mfbc3,'mfdst3':mfdst3,'mfncl3':mfncl3,
#                     'mfbc4':mfbc4,'mfpom4':mfpom4,
#                     'h2so4_chem_prod_rate':h2so4_chem_prod_rate,'qh2so4':qh2so4},
#             'time_input':{'mam_dt':mam_dt, 'mam_nstep':mam_nstep,'mam_output_intvl':mam_output_intvl},
#             'cntl_input':{'mdo_gasaerexch':mdo_gasaerexch,'mdo_coag':mdo_coag,'gas_unit':0,'mdo_rename':mdo_rename,'mdo_newnuc':mdo_newnuc},
#             'met_input':{'press':101325.,'RH_CLEA':rh_val},
#             'psd_params':{
#                 'mu1':mu1,'mu2':mu2,'mu3':mu3,'mu4':mu4,
#                 'sig1':mode_sigs[0],'sig2':mode_sigs[1],'sig3':mode_sigs[2],'sig4':mode_sigs[3]}}
#         mam_ensemble_settings.append(mam_settings)
#     if not os.path.exists(ensemble_dir):
#         os.mkdir(ensemble_dir)
    
#     pickle.dump(ensemble_settings,open(ensemble_dir + 'ensemble_settings.pkl','wb'),protocol=0)
#     pickle.dump(mam_ensemble_settings,open(ensemble_dir + 'mam_ensemble_settings.pkl','wb'),protocol=0)
    
#     return ensemble_settings, mam_ensemble_settings
    
def get_lhs_ensemble_settings_fromE3SM(
        ensemble_dir,e3sm_filename,N_samples,
        fractionSurface=0.9,useSorted=False,
        N_all_varnames = ['tot_num','frac_accum','nonaccum_frac_coarse','ofaccum_frac_fresh'],
        N_all_dists = ['loguniform','uniform','loguniform','loguniform'],
        N_all_param1 = [3e7,0.,1e-8,1e-13],
        N_all_param2 = [2e12,1.,1.,1.],
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
        n_part = 10000,
        n_repeat = 10,        
        t_max = 3600.*4.,
        del_t = 60.,t_output = 600.,
        min_frac_from_prod=0.,
        p=101325.,ht=500.):
    
    ensemble_name = get_ensemble_name(ensemble_dir)
    numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals, mfso42_vals, mfsoa2_vals, mfncl2_vals, mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals, mfbc4_vals, mfpom4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals, tmp_vals = sample_inputs_fromE3SM(
        N_samples, e3sm_filename, fractionSurface=fractionSurface, useSorted=useSorted,
        N_all_varnames = N_all_varnames, N_all_dists = N_all_dists, N_all_param1 = N_all_param1, N_all_param2 = N_all_param2,
        dgm_all_varnames = dgm_all_varnames, dgm_all_dists = dgm_all_dists, dgm_all_param1 = dgm_all_param1, dgm_all_param2 = dgm_all_param2,
        mode_sigs = mode_sigs,
        flux_dist = flux_dist, flux_param1 = flux_param1, flux_param2 = flux_param2,
        rh_dist = rh_dist, rh_param1 = rh_param1, rh_param2 = rh_param2,
        temp_dist = temp_dist, temp_param1 = temp_param1, temp_param2 = temp_param2,
        t_max = t_max, min_frac_from_prod = min_frac_from_prod,p = p)
    
    process = get_process(ensemble_name)    
    # start_time = time.time()
    if process == 'nada':
        do_mosaic = 'no'
        do_coagulation = 'no'
        do_nucleation = 'no'
    elif process == 'both':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'cond':
        do_mosaic = 'yes'
        do_coagulation = 'no'
        do_nucleation = 'no'
    elif process == 'coag':
        do_mosaic = 'no'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'all':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'yes'
    elif process == 'noNPF':
        do_mosaic = 'yes'
        do_coagulation = 'yes'
        do_nucleation = 'no'
    elif process == 'noCoag':
        do_mosaic = 'yes'
        do_coagulation = 'no'
        do_nucleation = 'yes'
    
    if do_coagulation == 'yes':
        mdo_coag = 1
    else:
        mdo_coag = 0
    
    if do_mosaic == 'yes':
        mdo_gasaerexch = 1
    else:
        mdo_gasaerexch = 0
        
    if do_nucleation == 'yes':
        mdo_newnuc = 1
    else:
        mdo_newnuc = 0
    
    mdo_rename = 1
    
    mam_dt = int(del_t)
    mam_output_intvl = int(t_output/del_t)
    mam_nstep = int(t_max/mam_dt)
    
    ensemble_settings = []
    mam_ensemble_settings = []
    mam4_spec_names = ['bc','pom','dst','ncl','soa','so4']
    modes = range(1,5)
    for qq,(numc1,numc2,numc3,numc4,mu1,mu2,mu3,mu4,mfso41,mfpom1,mfsoa1,mfbc1,mfdst1,mfncl1,mfso42,mfsoa2,mfncl2,
         mfso43,mfpom3,mfsoa3,mfbc3,mfdst3,mfncl3,mfbc4,mfpom4,h2so4_chem_prod_rate,qh2so4,rh_val,temp_val) in enumerate(zip(
            numc1_vals,numc2_vals,numc3_vals,numc4_vals,mu1_vals,mu2_vals,mu3_vals,mu4_vals,
            mfso41_vals,mfpom1_vals,mfsoa1_vals,mfbc1_vals,mfdst1_vals,mfncl1_vals,
            mfso42_vals,mfsoa2_vals,mfncl2_vals,mfso43_vals,mfpom3_vals,mfsoa3_vals,mfbc3_vals,mfdst3_vals,mfncl3_vals,mfbc4_vals,mfpom4_vals,
            h2so4_chem_prod_rate_vals,qh2so4_vals,rh_vals,tmp_vals)):
        
        # move all of this into its own function? (make the option of mam4_specs vs normal partmc_specs)
        aero_init = {'dist':{}}
        # mode_dat = {}
        
        for ii,mode in enumerate(modes):
            aero_init['dist'][mode] = {'diam_type':'geometric','mode_type':'log_normal'}
            aero_init['dist'][mode]['num_conc'] = eval('numc' + str(mode))
            aero_init['dist'][mode]['geom_mean_diam'] = np.exp(eval('mu' + str(mode)))
            aero_init['dist'][mode]['log10_geom_std_dev'] = np.log10(np.exp(mode_sigs[ii]))

            aero_init['dist'][mode]['comp'] = {}
            for mam4_spec_name in mam4_spec_names:
                try:
                    mam4_mass_frac = eval('mf' + mam4_spec_name + str(mode))
                except:
                    mam4_mass_frac = 0.
                
                partmc_spec_names,partmc_mass_fracs = mam4_to_partmc_composition(mam4_spec_name,mam4_mass_frac)
                for (partmc_spec_name,partmc_mass_frac) in zip(partmc_spec_names,partmc_mass_fracs):
                    aero_init['dist'][mode]['comp'][partmc_spec_name] = partmc_mass_frac
                    print(partmc_spec_name,partmc_mass_frac)
        h2so4_emission_rate = h2so4_chem_prod_rate*ht*p/(R*temp_val) #ht/0.0224 
        h2so4_ppb = qh2so4*1e9 # MW_h2so4/MW_air*
    
        height = {}
        height['time'] = 0.
        height['height'] = ht
        
        temp = {}
        temp['time'] = 0.
        temp['temp'] = temp_val
        
        pres = {}
        pres['time'] = 0.
        pres['pressure'] = p

        spec = {
                'n_part':int(n_part),
                'n_repeat':int(n_repeat),
                't_max':int(t_max),
                'del_t':del_t,
                't_output':int(t_output),
                'do_coagulation':do_coagulation,
                'do_mosaic':do_mosaic,
                'rel_humidity':rh_val,
                'do_nucleation':do_nucleation}
        
        partmc_settings = {
                'aero_init':aero_init,
                'gas_init':{'H2SO4':h2so4_ppb},
                'gas_emit':{'time':0., 'rate':1.,'H2SO4':h2so4_emission_rate},
                'height':height,
                'temp':temp,
                'pres':pres,
                'spec':spec}
        
        ensemble_settings.append(partmc_settings)
        
        mam_settings = {
            'scenario':str(qq).zfill(6),
            'chem_input':{
                    'numc1':numc1,'numc2':numc2,'numc3':numc3,'numc4':numc4,
                    'mfso41':mfso41,'mfpom1':mfpom1,'mfsoa1':mfsoa1,'mfbc1':mfbc1,'mfdst1':mfdst1,'mfncl1':mfncl1,
                    'mfso42':mfso42,'mfsoa2':mfsoa2,'mfncl2':mfncl2,
                    'mfso43':mfso43,'mfpom3':mfpom3,'mfsoa3':mfsoa3,'mfbc3':mfbc3,'mfdst3':mfdst3,'mfncl3':mfncl3,
                    'mfbc4':mfbc4,'mfpom4':mfpom4,
                    'h2so4_chem_prod_rate':h2so4_chem_prod_rate,'qh2so4':qh2so4},
            'time_input':{'mam_dt':mam_dt, 'mam_nstep':mam_nstep,'mam_output_intvl':mam_output_intvl},
            'cntl_input':{'mdo_gasaerexch':mdo_gasaerexch,'mdo_coag':mdo_coag,'gas_unit':0,'mdo_rename':mdo_rename,'mdo_newnuc':mdo_newnuc},
            'met_input':{'press':101325.,'RH_CLEA':rh_val},
            'psd_params':{
                'mu1':mu1,'mu2':mu2,'mu3':mu3,'mu4':mu4,
                'sig1':mode_sigs[0],'sig2':mode_sigs[1],'sig3':mode_sigs[2],'sig4':mode_sigs[3]}}
        mam_ensemble_settings.append(mam_settings)
    if not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)
    
    pickle.dump(ensemble_settings,open(ensemble_dir + 'ensemble_settings.pkl','wb'),protocol=0)
    pickle.dump(mam_ensemble_settings,open(ensemble_dir + 'mam_ensemble_settings.pkl','wb'),protocol=0)
    
    return ensemble_settings, mam_ensemble_settings

# def get_mam_ensemble_settings(ensemble_settings):
    
#     for qq,partmc_settings in enumerate(ensemble_settings):
        
#         # move all of this into its own function? (make the option of mam4_specs vs normal partmc_specs)
#         aero_init = {'dist':{}}
#         # mode_dat = {}
        
#         for ii,mode in enumerate(modes):
#             aero_init['dist'][mode] = {'diam_type':'geometric','mode_type':'log_normal'}
#             aero_init['dist'][mode]['num_conc'] = eval('numc' + str(mode))
#             aero_init['dist'][mode]['geom_mean_diam'] = np.exp(eval('mu' + str(mode)))
#             aero_init['dist'][mode]['log10_geom_std_dev'] = np.log10(np.exp(mode_sigs[ii]))

#             aero_init['dist'][mode]['comp'] = {}
#             for mam4_spec_name in mam4_spec_names:
#                 try:
#                     mam4_mass_frac = eval('mf' + mam4_spec_name + str(mode))
#                 except:
#                     mam4_mass_frac = 0.
                
#                 partmc_spec_names,partmc_mass_fracs = mam4_to_partmc_composition(mam4_spec_name,mam4_mass_frac)
#                 for (partmc_spec_name,partmc_mass_frac) in zip(partmc_spec_names,partmc_mass_fracs):
#                     aero_init['dist'][mode]['comp'][partmc_spec_name] = partmc_mass_frac
#                     print(partmc_spec_name,partmc_mass_frac)
#         h2so4_emission_rate = h2so4_chem_prod_rate*ht*p/(R*temp_val) #ht/0.0224 
#         h2so4_ppb = qh2so4*1e9 # MW_h2so4/MW_air*
    
#         height = {}
#         height['time'] = 0.
#         height['height'] = ht
        
#         temp = {}
#         temp['time'] = 0.
#         temp['temp'] = temp_val
        
#         pres = {}
#         pres['time'] = 0.
#         pres['pressure'] = p

#         spec = {
#                 'n_part':int(n_part),
#                 'n_repeat':int(n_repeat),
#                 't_max':int(t_max),
#                 'del_t':del_t,
#                 't_output':int(t_output),
#                 'do_coagulation':do_coagulation,
#                 'do_mosaic':do_mosaic,
#                 'rel_humidity':rh_val,
#                 'do_nucleation':do_nucleation}
        
#         partmc_settings = {
#                 'aero_init':aero_init,
#                 'gas_init':{'H2SO4':h2so4_ppb},
#                 'gas_emit':{'time':0., 'rate':1.,'H2SO4':h2so4_emission_rate},
#                 'height':height,
#                 'temp':temp,
#                 'pres':pres,
#                 'spec':spec}
    
#     mam_settings = {}
#     mam_settings{'scenario'} = str(ii).zfill(6)
#     mam_settings{'chem_input'} = {}
#     for mode in enumerate(modes):
#         mam_settings['chem_input']['numc' + str(mode)] = partmc_settings['aero_init']['dist'][mode]['num_conc']
#         mam_settings['chem_input']['numc' + str(mode)] = partmc_settings['aero_init']['dist'][mode]['num_conc']        

#     mam_settings = {
#         'scenario':str(qq).zfill(6),
#         'chem_input':{
#                 'numc1':numc1,'numc2':numc2,'numc3':numc3,'numc4':numc4,
#                 'mfso41':mfso41,'mfpom1':mfpom1,'mfsoa1':mfsoa1,'mfbc1':mfbc1,'mfdst1':mfdst1,'mfncl1':mfncl1,
#                 'mfso42':mfso42,'mfsoa2':mfsoa2,'mfncl2':mfncl2,
#                 'mfso43':mfso43,'mfpom3':mfpom3,'mfsoa3':mfsoa3,'mfbc3':mfbc3,'mfdst3':mfdst3,'mfncl3':mfncl3,
#                 'mfbc4':mfbc4,'mfpom4':mfpom4,
#                 'h2so4_chem_prod_rate':h2so4_chem_prod_rate,'qh2so4':qh2so4},
#         'time_input':{'mam_dt':mam_dt, 'mam_nstep':mam_nstep,'mam_output_intvl':mam_output_intvl},
#         'cntl_input':{'mdo_gasaerexch':mdo_gasaerexch,'mdo_coag':mdo_coag,'gas_unit':0,'mdo_rename':mdo_rename,'mdo_newnuc':mdo_newnuc},
#         'met_input':{'press':101325.,'RH_CLEA':rh_val},
#         'psd_params':{
#             'mu1':mu1,'mu2':mu2,'mu3':mu3,'mu4':mu4,
#             'sig1':mode_sigs[0],'sig2':mode_sigs[1],'sig3':mode_sigs[2],'sig4':mode_sigs[3]}}
    
def molecules_to_ions(mass_tot,molecule_name):
    # note: should work for absolute mass, mass concentration, or mass frac
    if molecule_name == 'ncl' or molecule_name == 'NaCl' or molecule_name == 'salt'  or molecule_name == 'sea_salt': 
        mass_fracs = np.array([c.mw_na/(c.mw_na + c.mw_cl),c.mw_cl/(c.mw_na + c.mw_cl)])
    elif molecule_name == '(NH4)2SO4' or molecule_name == 'ammonium_sulfate':
        mass_fracs = np.array([2*c.mw_nh4/(2*c.mw_nh4 + c.mw_so4),c.mw_so4/(2*c.mw_nh4 + c.mw_so4)])
    elif molecule_name == 'NH4NO3' or molecule_name == 'ammonium_nitrate':
        mass_fracs = np.array([c.mw_nh4/(c.mw_nh4 + c.mw_no3),c.mw_no3/(c.mw_nh4 + c.mw_no3)])
        
    mass_ions = mass_tot*mass_fracs
    return mass_ions
    
def mam4_to_partmc_composition(mam4_spec_name,mam4_mass_frac):
    if mam4_spec_name.upper() == 'NCL':
        partmc_mass_fracs = molecules_to_ions(mam4_mass_frac,mam4_spec_name)
        partmc_spec_names = ['Na','Cl']
    elif mam4_spec_name.upper() == 'POM':
        partmc_mass_fracs = np.array([mam4_mass_frac]) #np.array([1.])
        partmc_spec_names = ['OC']
    elif mam4_spec_name.upper() == 'DST':
        partmc_mass_fracs = np.array([mam4_mass_frac]) #np.array([1.])
        partmc_spec_names = ['OIN']
    elif mam4_spec_name.upper() == 'SOA':
        partmc_spec_names = ['ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']
        partmc_mass_fracs = np.ones(len(partmc_spec_names))*mam4_mass_frac/len(partmc_spec_names)
    else:
        partmc_mass_fracs = np.array([mam4_mass_frac]) #np.array([1.])
        partmc_spec_names = [mam4_spec_name.upper()]
    return partmc_spec_names,partmc_mass_fracs
    
# def mam_to_partmc_spec(mam_spec):
#     if mam_spec.upper() == 'POM' or mam_spec.upper() == 'SOA' or mam_spec.upper() == 'DST' or mam_spec.upper() == 'NCL':
#         partmc_spec = 'OC'
#     else:
#         partmc_spec = mam_spec.upper()
#     return partmc_spec

def get_ensemble_name(ensemble_dir):
    idx = 0; indices = []; 
    while idx>=0:
        idx = ensemble_dir.find('/',idx+1)
        if idx >=0:
            indices.append(idx)
        else:
            break
    if len(ensemble_dir[indices[-1]+1:])>0:
        ensemble_name = ensemble_dir[indices[-1]+1:]
    else:
        ensemble_name = ensemble_dir[indices[-2]+1:indices[-1]]
    return ensemble_name

def get_process(ensemble_name):
    idx = 0; indices = []; 
    while idx>=0:
        idx = ensemble_name.find('_',idx+1)
        if idx >=0:
            indices.append(idx)
        else:
            break
    process = ensemble_name[indices[-1]+1:]
    return process

# =============================================================================
# sample input parameters across ensemble (note: set up for benchmarking case right now. How to generalize?)
# =============================================================================
def sample_inputs(
        N_samples,
        N_all_varnames = ['tot_num','frac_aitk','frac_coarse','frac_fresh'],#,'h2so4_prod_rate','qh2so4'],
        N_all_dists = ['loguniform','uniform','loguniform','uniform'],#,'loguniform','loguniform'],
        N_all_param1 = [1e8,0.5,0.001,0.],#,5e-16,1e-12],
        N_all_param2 = [1e12,0.9,0.1,1.,5e-13],
        dgm_all_varnames = ['dgm_aik','dgm_accum','dgm_coarse','dgm_fresh'],
        dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
        dgm_all_param1 = [0.0535e-6, 0.0087e-6, 1.000e-6, 0.010e-6],
        dgm_all_param2 = [0.4400e-6, 0.0520e-6, 4.000e-6, 0.100e-6],
        mode_sigs = np.log([1.8,1.6,1.6,1.8]),
        mode1_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
        mode1_dists = ['uniform','uniform','uniform','uniform','uniform'],
        mode1_param1 = [0.,0.,0.,0.,0.],
        mode1_param2 = [1.,1.,1.,1.,1.],
        mode2_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl'],
        mode2_dists = ['uniform','uniform'],
        mode2_param1 = [0.,0.],
        mode2_param2 = [1.,1.],
        mode3_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
        mode3_dists = ['uniform','uniform','uniform','uniform','uniform'],
        mode3_param1 = [0.,0.,0.,0.,0.],
        mode3_param2 = [1.,1.,1.,1.,1.],
        mode4_varnames = ['frac_bc'],
        mode4_dists = ['uniform'],
        mode4_param1 = [0.],
        mode4_param2 = [1.],
        flux_dist = 'loguniform',#'lognormal',#'loguniform',
        flux_param1 = 1e-2*1e-9,#0.1*1e-9,#1e-2*1e-9,
        flux_param2 = 1e1*1e-9,#1.5,#
        rh_dist='uniform',
        rh_param1 = 0.,
        rh_param2 = 0.99,
        temp_dist='uniform',
        temp_param1 = 240.,
        temp_param2 = 310.,
        t_max=60.,min_frac_from_prod=0.,
        p=101325.):
    
    lhd = lhs(
        len(N_all_varnames) + len(dgm_all_varnames) + 
        len(mode1_varnames) + len(mode2_varnames) + 
        len(mode3_varnames) + len(mode4_varnames) + 3, samples=N_samples)
    X = np.zeros(lhd.shape)
    for vv,varname in enumerate(N_all_varnames):
        idx, = np.where([onevarname==varname for onevarname in N_all_varnames])
        param1 = N_all_param1[idx[0]]
        param2 = N_all_param2[idx[0]]
        dist = N_all_dists[idx[0]]
        X[:,vv] = get_sample(lhd[:,vv],dist,param1,param2)
    
    for vv,varname in enumerate(dgm_all_varnames):
        idx, = np.where([onevarname==varname for onevarname in dgm_all_varnames])
        param1 = dgm_all_param1[idx[0]]
        param2 = dgm_all_param2[idx[0]]
        dist = dgm_all_dists[idx[0]]
        X[:,vv+4] = get_sample(lhd[:,vv],dist,param1,param2)
        
    tot_num = X[:,0]
    frac_aitk = X[:,1]
    frac_coarse = X[:,2]
    frac_fresh = X[:,3]
    frac_accum = 1. - X[:,2] - X[:,1]
    
    mode_Ns = np.zeros([N_samples,4])
    mode_Ns[:,0] = tot_num*frac_accum*(1.-frac_fresh) # aged accumulation
    mode_Ns[:,1] = tot_num*frac_aitk # aitken
    mode_Ns[:,2] = tot_num*frac_coarse
    mode_Ns[:,3] = tot_num*frac_accum*frac_fresh
    
    mode_mus = np.log(X[:,range(4,8)])
    
    avg_vol_flux = np.zeros(N_samples)    
    h2so4_chem_prod_rate_vals = np.zeros(N_samples)
    qh2so4_vals = np.zeros(N_samples)
    ii = 0
    
    while ii<N_samples:
        avg_vol_flux[ii] = get_sample(np.random.uniform(),flux_dist,flux_param1,flux_param2)
        dh2so4_dt_tot = get_dh2so4dt_mol(avg_vol_flux[ii],mode_Ns[ii,:],mode_mus,mode_sigs)
        
        frac_prod = min_frac_from_prod + (1.-min_frac_from_prod)*np.random.uniform() # randomly vary the amount of dh2so4_dt from production vs initial
        h2so4_chem_prod_rate_vals[ii] = frac_prod*dh2so4_dt_tot 
        qh2so4_vals[ii] = (1.-frac_prod)*dh2so4_dt_tot*t_max
        ii += 1
    
    numc1_vals = mode_Ns[:,0]
    numc2_vals = mode_Ns[:,1]
    numc3_vals = mode_Ns[:,2]
    numc4_vals = mode_Ns[:,3]
    
    mu1_vals = mode_mus[:,0]
    mu2_vals = mode_mus[:,1]
    mu3_vals = mode_mus[:,2]
    mu4_vals = mode_mus[:,3]
    
    lhd_mode1 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames))]
    lhd_mode2 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames))]
    lhd_mode3 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames))]
    lhd_mode4 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames) + len(mode4_varnames))]
    
    mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals = get_mode_comp('mode1',mode1_varnames,mode1_dists,mode1_param1,mode1_param2,lhd_mode1)
    mfso42_vals, mfsoa2_vals, mfncl2_vals = get_mode_comp('mode2',mode2_varnames,mode2_dists,mode2_param1,mode2_param2,lhd_mode2)
    mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals = get_mode_comp('mode3',mode3_varnames,mode3_dists,mode3_param1,mode3_param2,lhd_mode3)
    mfbc4_vals, mfpom4_vals = get_mode_comp('mode4',mode4_varnames,mode4_dists,mode4_param1,mode4_param2,lhd_mode4)
    
    rh_vals = get_sample(lhd[:,-2],rh_dist,rh_param1,rh_param2)
    tmp_vals = get_sample(lhd[:,-1],temp_dist,temp_param1,temp_param2)
    
    return numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals, mfso42_vals, mfsoa2_vals, mfncl2_vals, mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals, mfbc4_vals, mfpom4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals, tmp_vals

def sample_inputs_allFromE3SM(
        N_samples, e3sm_filename,
        # fractionSurface = 0.9,useSorted=False,
        # e3sm_varnames = [],
        # N_all_varnames = ['tot_num','frac_aitk','frac_coarse','frac_fresh'],#,'h2so4_prod_rate','qh2so4'],
        # # N_all_dists = ['loguniform','uniform','loguniform','uniform'],#,'loguniform','loguniform'],
        # # N_all_param1 = [1e8,0.5,0.001,0.],#,5e-16,1e-12],
        # # N_all_param2 = [1e12,0.9,0.1,1.,5e-13],
        # dgm_all_varnames = ['dgm_aik','dgm_accum','dgm_coarse','dgm_fresh'],
        # # dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
        # # dgm_all_param1 = [0.0535e-6, 0.0087e-6, 1.000e-6, 0.010e-6],
        # # dgm_all_param2 = [0.4400e-6, 0.0520e-6, 4.000e-6, 0.100e-6],
        mode_sigs = np.log([1.8,1.6,1.6,1.8]),
        # mode1_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
        # mode1_dists = ['uniform','uniform','uniform','uniform','uniform'],
        # mode1_param1 = [0.,0.,0.,0.,0.],
        # mode1_param2 = [1.,1.,1.,1.,1.],
        # mode2_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl'],
        # mode2_dists = ['uniform','uniform'],
        # mode2_param1 = [0.,0.],
        # mode2_param2 = [1.,1.],
        # mode3_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
        # mode3_dists = ['uniform','uniform','uniform','uniform','uniform'],
        # mode3_param1 = [0.,0.,0.,0.,0.],
        # mode3_param2 = [1.,1.,1.,1.,1.],
        # mode4_varnames = ['frac_bc'],
        # mode4_dists = ['uniform'],
        # mode4_param1 = [0.],
        # mode4_param2 = [1.],
        flux_dist = 'loguniform',#'lognormal',#'loguniform',
        flux_param1 = 1e-2*1e-9,#0.1*1e-9,#1e-2*1e-9,
        flux_param2 = 1e1*1e-9,#1.5,#
        # rh_dist='uniform',
        # rh_param1 = 0.,
        # rh_param2 = 0.99,
        # temp_dist='uniform',
        # temp_param1 = 240.,
        # temp_param2 = 310.,
        dt_e3sm = 1800.,
        span_observed_flux=False,
        correlate_flux_with_num=False,
        sigma_log10_flux=0.5,
        min_frac_from_prod=None,
        no_soa=True,only_subsaturated_cells=True,
        t_max=60.,max_rh=0.9999):#,min_frac_from_prod=0.):
    
    
    # for vv,varname in enumerate(N_all_varnames):
    #     idx, = np.where([onevarname==varname for onevarname in N_all_varnames])
    #     param1 = N_all_param1[idx[0]]
    #     param2 = N_all_param2[idx[0]]
    #     dist = N_all_dists[idx[0]]
    #     X[:,vv] = get_sample(lhd[:,vv],dist,param1,param2)
    
    # for vv,varname in enumerate(dgm_all_varnames):
    #     idx, = np.where([onevarname==varname for onevarname in dgm_all_varnames])
    #     param1 = dgm_all_param1[idx[0]]
    #     param2 = dgm_all_param2[idx[0]]
    #     dist = dgm_all_dists[idx[0]]
    #     X[:,vv+4] = get_sample(lhd[:,vv],dist,param1,param2)
        
    # tot_num = X[:,0]
    # frac_aitk = X[:,1]
    # frac_coarse = X[:,2]
    # frac_fresh = X[:,3]
    # frac_accum = 1. - X[:,2] - X[:,1]
    
    # mode_Ns = np.zeros([N_samples,4])
    # mode_Ns[:,0] = tot_num*frac_accum*(1.-frac_fresh) # aged accumulation
    # mode_Ns[:,1] = tot_num*frac_aitk # aitken
    # mode_Ns[:,2] = tot_num*frac_coarse
    # mode_Ns[:,3] = tot_num*frac_accum*frac_fresh
    
    # mode_mus = np.log(X[:,range(4,8)])
    
    
    
    f = Dataset(e3sm_filename)
    # varnames = ['logN0_1','logN0_2','logN0_3','logN0_4']
    e3sm_varnames = [
        'N0_1','N0_2','N0_3','N0_4',
        'mu0_1','mu0_2','mu0_3','mu0_4',
        'mode1_frac_hygroscopic','mode1_frac_hygroscopic_thats_ncl','mode1_frac_hydrophobic_thats_insol','mode1_frac_insol_thats_bc','mode1_frac_org_thats_pom',
        'mode2_frac_hygroscopic','mode2_frac_hygroscopic_thats_ncl',
        'mode3_frac_hygroscopic','mode3_frac_hygroscopic_thats_ncl','mode3_frac_hydrophobic_thats_insol','mode3_frac_insol_thats_bc','mode3_frac_org_thats_pom',
        'mode4_frac_bc',
        'sa_flux','precursor_gases','RELHUM','TS','PS']
    
    RH = analyses.retrieve_e3sm_data(f,'RELHUM')/100.
    if only_subsaturated_cells:
        idx, = np.where(RH[:,-1,:,:].ravel()<100.)
    else:
        idx, = np.where(RH[:,-1,:,:].ravel()>=-100.)
        
    all_params_surface = ()
    for varname in e3sm_varnames:
        all_params_gridded = analyses.retrieve_e3sm_data(f,varname,dt_e3sm=dt_e3sm)
        if len(all_params_gridded.shape) == 4:
            all_params_surface += (all_params_gridded[:,-1,:,:].ravel()[idx],)
        elif len(all_params_gridded.shape) == 3:
            all_params_surface += (all_params_gridded[:,:,:].ravel()[idx],)
        # elif len(all_params_gridded.shape) == 3:
            # all_params_surface += (all_params_gridded[:,-1,:,:].ravel(),)
        # all_params_notSurface += (all_params_gridded[:,1:,:,:].ravel(),)
        # all_vars += (analyses.retrieve_e3sm_data(f,varname).ravel(),)
    
    # lhd_dsdParams = lhd[:,:8]
    lhd = lhs(
        2, samples=N_samples)
    X_allParams,indices = sample_surfaceParams_fromE3SM(all_params_surface,N_samples,lhd[:,0])
    
    # e3sm_varnames = [
    #     'N0_1','N0_2','N0_3','N0_4',
    #     'mu0_1','mu0_2','mu0_3','mu0_4',
    #     'mode1_frac_hygroscopic','mode1_frac_hygroscopic_thats_ncl','mode1_frac_hydrophobic_thats_insol','mode1_frac_insol_thats_bc','mode1_frac_org_thats_pom',
    #     'mode2_frac_hygroscopic','mode2_frac_hygroscopic_thats_ncl',
    #     'mode3_frac_hygroscopic','mode3_frac_hygroscopic_thats_ncl','mode3_frac_hydrophobic_thats_insol','mode3_frac_insol_thats_bc','mode3_frac_org_thats_pom',
    #     'mode4_frac_bc',
    #     'sa_flux','precursor_gases','RELHUM','TS','PS']

    numc1_vals = X_allParams[:,0]
    numc2_vals = X_allParams[:,1]
    numc3_vals = X_allParams[:,2]
    numc4_vals = X_allParams[:,3]
    mu1_vals = np.log(10.**X_allParams[:,4])
    mu2_vals = np.log(10.**X_allParams[:,5])
    mu3_vals = np.log(10.**X_allParams[:,6])
    mu4_vals = np.log(10.**X_allParams[:,7])
    
    X_mode1 = X_allParams[:,range(8,13)]
    X_mode2 = X_allParams[:,range(13,15)]
    X_mode3 = X_allParams[:,range(15,20)]
    X_mode4 = X_allParams[:,np.array([20])]
    
    mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals = get_mode_comp_fromX('mode1',X_mode1,no_soa=no_soa)
    mfso42_vals, mfsoa2_vals, mfncl2_vals, mfpoa2_vals = get_mode_comp_fromX('mode2',X_mode2,no_soa=no_soa)
    mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals = get_mode_comp_fromX('mode3',X_mode3,no_soa=no_soa)
    mfbc4_vals, mfpom4_vals = get_mode_comp_fromX('mode4',X_mode4,no_soa=no_soa)
    if span_observed_flux == True:
        if correlate_flux_with_num:
            avg_vol_flux = np.zeros_like(X_allParams[:,21])
            for ii in range(len(avg_vol_flux)):
                Ntot = np.sum(X_allParams[:,:4])
                avg_vol_flux[ii] = observed_num_to_flux(Ntot,lhd[ii,1],sigma_log10_flux=sigma_log10_flux)
        else:
            avg_vol_flux = get_sample(lhd[:,1],flux_dist,flux_param1,flux_param2)
    else:
        avg_vol_flux = X_allParams[:,21]
    if min_frac_from_prod == 1.:
        qh2so4_vals = np.zeros_like(X_allParams[:,22])
    else:
        qh2so4_vals = X_allParams[:,22]
    rh_vals = X_allParams[:,23]/100.
    rh_vals[rh_vals>max_rh] = max_rh
    tmp_vals = X_allParams[:,24]
    press_vals = X_allParams[:,25]
    # ii = 0
    # while ii<N_samples:
        
        # mu2_vals = np.log(10.**X_allParams[:,5])
        # mu3_vals = np.log(10.**X_allParams[:,6])
        # mu4_vals = np.log(10.**X_allParams[:,7])
    
    # avg_vol_flux = np.zeros(N_samples)    
    # h2so4_chem_prod_rate_vals = np.zeros(N_samples)
    # qh2so4_vals = np.zeros(N_samples)
    h2so4_chem_prod_rate_vals = np.zeros(N_samples)
    ii = 0
    while ii<N_samples:
        # avg_vol_flux[ii] = get_sample(np.random.uniform(),flux_dist,flux_param1,flux_param2)
        mode_Ns = np.array([numc1_vals[ii],numc2_vals[ii],numc3_vals[ii],numc4_vals[ii]])
        mode_mus = np.array([mu1_vals[ii],mu2_vals[ii],mu3_vals[ii],mu4_vals[ii]])
        
        dh2so4_dt_tot = get_dh2so4dt_mol(avg_vol_flux[ii],mode_Ns,mode_mus,mode_sigs)
        if min_frac_from_prod == None:
            frac_prod = (dh2so4_dt_tot*t_max - qh2so4_vals[ii])/(dh2so4_dt_tot*t_max)
            h2so4_chem_prod_rate_vals[ii] = frac_prod*dh2so4_dt_tot
        elif min_frac_from_prod == 1.:
            
            h2so4_chem_prod_rate_vals[ii] = dh2so4_dt_tot 
            qh2so4_vals[ii] = 0.
        else:
            frac_prod = min_frac_from_prod + (1.-min_frac_from_prod)*np.random.uniform()
            h2so4_chem_prod_rate_vals[ii] = frac_prod*dh2so4_dt_tot 
            qh2so4_vals[ii] = (1.-frac_prod)*dh2so4_dt_tot*t_max
            print('ii',ii,'frac_prod',frac_prod,'h2so4_chem_prod_rate',h2so4_chem_prod_rate_vals[ii],'qh2so4',qh2so4_vals[ii])
            
        # frac_prod = min_frac_from_prod + (1.-min_frac_from_prod)*np.random.uniform() # randomly vary the amount of dh2so4_dt from production vs initial
        # h2so4_chem_prod_rate_vals[ii] = frac_prod*dh2so4_dt_tot 
        # qh2so4_vals[ii] = (1.-frac_prod)*dh2so4_dt_tot*t_max
        ii += 1
    h2so4_chem_prod_rate_vals[h2so4_chem_prod_rate_vals<0.] = 0.
    
    # lhd_mode1 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames), len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames))]
    # lhd_mode2 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames))]
    # lhd_mode3 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames))]
    # lhd_mode4 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames) + len(mode4_varnames))]
    
    
    # rh_vals = get_sample(lhd[:,-2],rh_dist,rh_param1,rh_param2)
    # tmp_vals = get_sample(lhd[:,-1],temp_dist,temp_param1,temp_param2)
    
    return numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals, mfso42_vals, mfsoa2_vals, mfncl2_vals, mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals, mfbc4_vals, mfpom4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals, tmp_vals, press_vals, indices
   
def observed_num_to_flux(Ntot,cdf_val,sigma_log10_flux=0.5):
    # Ntot is in m^{-3}
    # assume log10flux is dist
    log10N_per_cm3 = np.log10(Ntot/100.**3)
    mean_log10flux_nm_per_h = lambda logNtot: 4/3.*log10N_per_cm3 - 17/3
    log10_flux_vals = np.linspace(-7,5,10000)
    
    all_cdf_vals = norm(loc=mean_log10flux_nm_per_h(log10N_per_cm3),scale=sigma_log10_flux).cdf(log10_flux_vals)
    log10flux_nm_per_h = np.interp(cdf_val,all_cdf_vals,log10_flux_vals)
    
    SAflux_m_per_s = 10.**log10flux_nm_per_h*1e-9/3600.
    return SAflux_m_per_s
    
def sample_inputs_fromE3SM(
        N_samples, e3sm_filename,
        fractionSurface = 0.9,useSorted=False,
        N_all_varnames = ['tot_num','frac_aitk','frac_coarse','frac_fresh'],#,'h2so4_prod_rate','qh2so4'],
        N_all_dists = ['loguniform','uniform','loguniform','uniform'],#,'loguniform','loguniform'],
        N_all_param1 = [1e8,0.5,0.001,0.],#,5e-16,1e-12],
        N_all_param2 = [1e12,0.9,0.1,1.,5e-13],
        dgm_all_varnames = ['dgm_aik','dgm_accum','dgm_coarse','dgm_fresh'],
        dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
        dgm_all_param1 = [0.0535e-6, 0.0087e-6, 1.000e-6, 0.010e-6],
        dgm_all_param2 = [0.4400e-6, 0.0520e-6, 4.000e-6, 0.100e-6],
        mode_sigs = np.log([1.8,1.6,1.6,1.8]),
        mode1_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
        mode1_dists = ['uniform','uniform','uniform','uniform','uniform'],
        mode1_param1 = [0.,0.,0.,0.,0.],
        mode1_param2 = [1.,1.,1.,1.,1.],
        mode2_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl'],
        mode2_dists = ['uniform','uniform'],
        mode2_param1 = [0.,0.],
        mode2_param2 = [1.,1.],
        mode3_varnames = ['frac_hygroscopic','frac_hygroscopic_thats_ncl','frac_hydrophobic_thats_insol','frac_insol_thats_bc','frac_org_thats_pom'],
        mode3_dists = ['uniform','uniform','uniform','uniform','uniform'],
        mode3_param1 = [0.,0.,0.,0.,0.],
        mode3_param2 = [1.,1.,1.,1.,1.],
        mode4_varnames = ['frac_bc'],
        mode4_dists = ['uniform'],
        mode4_param1 = [0.],
        mode4_param2 = [1.],
        flux_dist = 'loguniform',#'lognormal',#'loguniform',
        flux_param1 = 1e-2*1e-9,#0.1*1e-9,#1e-2*1e-9,
        flux_param2 = 1e1*1e-9,#1.5,#
        rh_dist='uniform',
        rh_param1 = 0.,
        rh_param2 = 0.99,
        temp_dist='uniform',
        temp_param1 = 240.,
        temp_param2 = 310.,
        t_max=60.,min_frac_from_prod=0.,
        p=101325.):
    
    lhd = lhs(
        len(N_all_varnames) + len(dgm_all_varnames) + 
        len(mode1_varnames) + len(mode2_varnames) + 
        len(mode3_varnames) + len(mode4_varnames) + 3, samples=N_samples)
    X = np.zeros(lhd.shape)
    for vv,varname in enumerate(N_all_varnames):
        idx, = np.where([onevarname==varname for onevarname in N_all_varnames])
        param1 = N_all_param1[idx[0]]
        param2 = N_all_param2[idx[0]]
        dist = N_all_dists[idx[0]]
        X[:,vv] = get_sample(lhd[:,vv],dist,param1,param2)
    
    for vv,varname in enumerate(dgm_all_varnames):
        idx, = np.where([onevarname==varname for onevarname in dgm_all_varnames])
        param1 = dgm_all_param1[idx[0]]
        param2 = dgm_all_param2[idx[0]]
        dist = dgm_all_dists[idx[0]]
        X[:,vv+4] = get_sample(lhd[:,vv],dist,param1,param2)
        
    tot_num = X[:,0]
    frac_aitk = X[:,1]
    frac_coarse = X[:,2]
    frac_fresh = X[:,3]
    frac_accum = 1. - X[:,2] - X[:,1]
    
    mode_Ns = np.zeros([N_samples,4])
    mode_Ns[:,0] = tot_num*frac_accum*(1.-frac_fresh) # aged accumulation
    mode_Ns[:,1] = tot_num*frac_aitk # aitken
    mode_Ns[:,2] = tot_num*frac_coarse
    mode_Ns[:,3] = tot_num*frac_accum*frac_fresh
    
    mode_mus = np.log(X[:,range(4,8)])
    
    avg_vol_flux = np.zeros(N_samples)    
    h2so4_chem_prod_rate_vals = np.zeros(N_samples)
    qh2so4_vals = np.zeros(N_samples)
    ii = 0
    
    while ii<N_samples:
        avg_vol_flux[ii] = get_sample(np.random.uniform(),flux_dist,flux_param1,flux_param2)
        dh2so4_dt_tot = get_dh2so4dt_mol(avg_vol_flux[ii],mode_Ns[ii,:],mode_mus,mode_sigs)
        
        frac_prod = min_frac_from_prod + (1.-min_frac_from_prod)*np.random.uniform() # randomly vary the amount of dh2so4_dt from production vs initial
        h2so4_chem_prod_rate_vals[ii] = frac_prod*dh2so4_dt_tot 
        qh2so4_vals[ii] = (1.-frac_prod)*dh2so4_dt_tot*t_max
        ii += 1
    
    
    f = Dataset(e3sm_filename)
    # varnames = ['logN0_1','logN0_2','logN0_3','logN0_4']
    varnames = ['N0_1','N0_2','N0_3','N0_4','mu0_1','mu0_2','mu0_3','mu0_4']
    all_dsdParams_surface = ()
    all_dsdParams_notSurface = ()
    for varname in varnames:
        all_dsdParams_gridded = analyses.retrieve_e3sm_data(f,varname)
        all_dsdParams_surface += (all_dsdParams_gridded[:,-1,:,:].ravel(),)
        all_dsdParams_notSurface += (all_dsdParams_gridded[:,1:,:,:].ravel(),)
        # all_vars += (analyses.retrieve_e3sm_data(f,varname).ravel(),)
    
    lhd_dsdParams = lhd[:,:8]
    X_dsdParams = sample_dsdParams_fromE3SM(lhd_dsdParams,all_dsdParams_surface,all_dsdParams_notSurface,fractionSurface=fractionSurface,useSorted=useSorted)
    
    numc1_vals = X_dsdParams[:,0]
    numc2_vals = X_dsdParams[:,1]
    numc3_vals = X_dsdParams[:,2]
    numc4_vals = X_dsdParams[:,3]
    
    mu1_vals = np.log(10.**X_dsdParams[:,4])
    mu2_vals = np.log(10.**X_dsdParams[:,5])
    mu3_vals = np.log(10.**X_dsdParams[:,6])
    mu4_vals = np.log(10.**X_dsdParams[:,7])
    
    # numc1_vals = mode_Ns[:,0]
    # numc2_vals = mode_Ns[:,1]
    # numc3_vals = mode_Ns[:,2]
    # numc4_vals = mode_Ns[:,3]
    
    # mu1_vals = mode_mus[:,0]
    # mu2_vals = mode_mus[:,1]
    # mu3_vals = mode_mus[:,2]
    # mu4_vals = mode_mus[:,3]
    
    lhd_mode1 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames))]
    lhd_mode2 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames))]
    lhd_mode3 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames))]
    lhd_mode4 = lhd[:,range(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames),len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames) + len(mode4_varnames))]
    
    mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals = get_mode_comp('mode1',mode1_varnames,mode1_dists,mode1_param1,mode1_param2,lhd_mode1)
    mfso42_vals, mfsoa2_vals, mfncl2_vals = get_mode_comp('mode2',mode2_varnames,mode2_dists,mode2_param1,mode2_param2,lhd_mode2)
    mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals = get_mode_comp('mode3',mode3_varnames,mode3_dists,mode3_param1,mode3_param2,lhd_mode3)
    mfbc4_vals, mfpom4_vals = get_mode_comp('mode4',mode4_varnames,mode4_dists,mode4_param1,mode4_param2,lhd_mode4)
    
    rh_vals = get_sample(lhd[:,-2],rh_dist,rh_param1,rh_param2)
    tmp_vals = get_sample(lhd[:,-1],temp_dist,temp_param1,temp_param2)
    
    return numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals, mfso42_vals, mfsoa2_vals, mfncl2_vals, mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals, mfbc4_vals, mfpom4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals, tmp_vals


        
    
# =============================================================================
#  sample input parameters
# =============================================================================
def sample_surfaceParams_fromE3SM(
        all_params_surface,N_samples,lhd):
    
    Nvars = len(all_params_surface)
    N_surface = len(all_params_surface[0])
    

    X = np.zeros([N_samples,len(all_params_surface)])
    
    indices = []
    for ii in range(N_samples):
        idx = int(lhd[ii]*N_surface)
        for jj in range(Nvars):
            X[ii,jj] = all_params_surface[jj][idx]
        indices.append(idx)
    # if useSorted:
    #     N_surface = len(all_params_surface[0])
    #     N_notSurface = len(all_params_notSurface[0])
        
    #     all_surface_sorted = np.zeros([N_surface,N_vars])
    #     all_notSurface_sorted = np.zeros([N_notSurface,N_vars])
        
    #     for jj in range(N_vars):
    #         all_surface_sorted[:,jj] = np.sort(all_params_surface[jj])
    #         all_notSurface_sorted[:,jj] = np.sort(all_params_notSurface[jj])
        
    #     these, = np.where(np.sum(all_surface_sorted[:,:4],axis=1)>0.)
    #     all_surface_sorted = all_surface_sorted[these,:]
    #     N_surface = len(these)
        
    #     those, = np.where(np.sum(all_notSurface_sorted[:,:4],axis=1)>0.)
    #     all_notSurface_sorted = all_notSurface_sorted[those,:]
    #     N_notSurface = len(those)
        
    #     for ii in range(N_samples):
    #         for jj in range(N_vars):
    #             if np.random.rand()<=fractionSurface:
    #                 idx = int(lhd[ii,jj]*N_surface)
    #                 X[ii,jj] = all_surface_sorted[idx,jj]
    #             else:
    #                 idx = int(lhd[ii,jj]*N_notSurface)
    #                 X[ii,jj] = all_notSurface_sorted[idx,jj]
    # else:
    #     lhd_one = lhd[:,0]
    #     Ntot_surface = np.zeros_like(all_params_surface[0])
    #     Ntot_notSurface = np.zeros_like(all_params_notSurface[0])        
    #     for jj in range(4):
    #         Ntot_surface += all_params_surface[jj]
    #         Ntot_notSurface += all_params_notSurface[jj]
        
    #     for ii in range(N_samples):
    #         if np.random.rand()<=fractionSurface:
    #             these, = np.where(Ntot_surface>0.)
    #             for jj in range(N_vars):
    #                 idx = these[int(lhd_one[ii]*len(these))]
    #                 X[ii,jj] = all_params_surface[jj][idx]
    #         else:
    #             these, = np.where(Ntot_notSurface>0.)
    #             for jj in range(N_vars):
    #                 idx = these[int(lhd_one[ii]*len(these))]
    #                 X[ii,jj] = all_params_notSurface[jj][idx]
    return X, indices


# def sample_allParams_fromE3SM(
#         lhd,all_params_surface,all_params_notSurface,
#         fractionSurface=0.9,useSorted=False):
#     N_samples = lhd.shape[0]
#     N_vars = lhd.shape[1]
#     X = np.zeros([N_samples,N_vars])
    
#     if useSorted:
#         N_surface = len(all_params_surface[0])
#         N_notSurface = len(all_params_notSurface[0])
        
#         all_surface_sorted = np.zeros([N_surface,N_vars])
#         all_notSurface_sorted = np.zeros([N_notSurface,N_vars])
        
#         for jj in range(N_vars):
#             all_surface_sorted[:,jj] = np.sort(all_params_surface[jj])
#             all_notSurface_sorted[:,jj] = np.sort(all_params_notSurface[jj])
        
#         these, = np.where(np.sum(all_surface_sorted[:,:4],axis=1)>0.)
#         all_surface_sorted = all_surface_sorted[these,:]
#         N_surface = len(these)
        
#         those, = np.where(np.sum(all_notSurface_sorted[:,:4],axis=1)>0.)
#         all_notSurface_sorted = all_notSurface_sorted[those,:]
#         N_notSurface = len(those)
        
#         for ii in range(N_samples):
#             for jj in range(N_vars):
#                 if np.random.rand()<=fractionSurface:
#                     idx = int(lhd[ii,jj]*N_surface)
#                     X[ii,jj] = all_surface_sorted[idx,jj]
#                 else:
#                     idx = int(lhd[ii,jj]*N_notSurface)
#                     X[ii,jj] = all_notSurface_sorted[idx,jj]
#     else:
#         lhd_one = lhd[:,0]
#         Ntot_surface = np.zeros_like(all_params_surface[0])
#         Ntot_notSurface = np.zeros_like(all_params_notSurface[0])        
#         for jj in range(4):
#             Ntot_surface += all_params_surface[jj]
#             Ntot_notSurface += all_params_notSurface[jj]
        
#         for ii in range(N_samples):
#             if np.random.rand()<=fractionSurface:
#                 these, = np.where(Ntot_surface>0.)
#                 for jj in range(N_vars):
#                     idx = these[int(lhd_one[ii]*len(these))]
#                     X[ii,jj] = all_params_surface[jj][idx]
#             else:
#                 these, = np.where(Ntot_notSurface>0.)
#                 for jj in range(N_vars):
#                     idx = these[int(lhd_one[ii]*len(these))]
#                     X[ii,jj] = all_params_notSurface[jj][idx]
#     return X


def sample_dsdParams_fromE3SM(
        lhd_dsdParams,all_dsdParams_surface,all_dsdParams_notSurface,
        fractionSurface=0.9,useSorted=False):
    N_samples = lhd_dsdParams.shape[0]
    N_vars = lhd_dsdParams.shape[1]
    X = np.zeros([N_samples,N_vars])
        
    if useSorted:
        N_surface = len(all_dsdParams_surface[0])
        N_notSurface = len(all_dsdParams_notSurface[0])
        
        all_surface_sorted = np.zeros([N_surface,N_vars])
        all_notSurface_sorted = np.zeros([N_notSurface,N_vars])
        
        for jj in range(N_vars):
            all_surface_sorted[:,jj] = np.sort(all_dsdParams_surface[jj])
            all_notSurface_sorted[:,jj] = np.sort(all_dsdParams_notSurface[jj])
        
        these, = np.where(np.sum(all_surface_sorted[:,:4],axis=1)>0.)
        all_surface_sorted = all_surface_sorted[these,:]
        N_surface = len(these)
        
        those, = np.where(np.sum(all_notSurface_sorted[:,:4],axis=1)>0.)
        all_notSurface_sorted = all_notSurface_sorted[those,:]
        N_notSurface = len(those)
        
        for ii in range(N_samples):
            for jj in range(N_vars):
                if np.random.rand()<=fractionSurface:
                    idx = int(lhd_dsdParams[ii,jj]*N_surface)
                    X[ii,jj] = all_surface_sorted[idx,jj]
                else:
                    idx = int(lhd_dsdParams[ii,jj]*N_notSurface)
                    X[ii,jj] = all_notSurface_sorted[idx,jj]
    else:
        lhd_one = lhd_dsdParams[:,0]
        Ntot_surface = np.zeros_like(all_dsdParams_surface[0])
        Ntot_notSurface = np.zeros_like(all_dsdParams_notSurface[0])        
        for jj in range(4):
            Ntot_surface += all_dsdParams_surface[jj]
            Ntot_notSurface += all_dsdParams_notSurface[jj]
        
        for ii in range(N_samples):
            if np.random.rand()<=fractionSurface:
                these, = np.where(Ntot_surface>0.)
                for jj in range(N_vars):
                    idx = these[int(lhd_one[ii]*len(these))]
                    X[ii,jj] = all_dsdParams_surface[jj][idx]
            else:
                these, = np.where(Ntot_notSurface>0.)
                for jj in range(N_vars):
                    idx = these[int(lhd_one[ii]*len(these))]
                    X[ii,jj] = all_dsdParams_notSurface[jj][idx]
    # all_vars_sorted = np.zeros([N_vars,len(all_vars[0])])
    # for jj in range(N_vars):
    #     all_vars_sorted[jj,:] = np.sort(all_vars[jj])
    
    # X = np.zeros([N_samples,N_vars])
    # for ii in range(N_samples):
    #     for jj in range(N_vars):
    #         idx = int(lhd[ii,jj]*N_samples)
    #         X[ii,jj] = all_vars_sorted[jj,idx]
            # X[ii,jj] = all_vars_sorted[idx,ii]
    return X
        # [int(lhd[ii,jj]*N_vars) for jj in lhd.shape[1]]
    
def get_sample(cdf_val,dist,param1,param2):
    if dist == 'uniform':
        x = cdf_val*(param2 - param1) + param1
    elif dist == 'loguniform':
        x = 10**(cdf_val*(np.log10(param2) - np.log10(param1)) + np.log10(param1))
    elif dist == 'normal':
        x = param1 + param2*np.sqrt(2)*erfinv(2*cdf_val-1)
    elif dist == 'lognormal':
        x = 10**(np.log10(param1) + np.log10(param2)*np.sqrt(2)*erfinv(2*cdf_val-1))
    return x

# =============================================================================
#  assign composition related to MAM4
# =============================================================================

def get_mode_comp_fromX(mode_name,X_comp,no_soa=True):
    if mode_name == 'mode1' or mode_name == 'mode3':
        frac_hygroscopic = X_comp[:,0]
        frac_hygroscopic_thats_ncl = X_comp[:,1]
        frac_hydrophobic_thats_insol = X_comp[:,2]        
        frac_insol_thats_bc = X_comp[:,3]
        if no_soa:
            frac_org_thats_pom = np.ones(X_comp[:,0].shape)
        else:
            frac_org_thats_pom = X_comp[:,4]
        frac_hydrophobic = 1. - frac_hygroscopic
        frac_insol = frac_hydrophobic*frac_hydrophobic_thats_insol
        frac_org = frac_hydrophobic*(1.-frac_hydrophobic_thats_insol)
        
        frac_so4 = frac_hygroscopic*(1. - frac_hygroscopic_thats_ncl)
        frac_pom = frac_org*frac_org_thats_pom
        frac_soa = frac_org*(1.-frac_org_thats_pom)
        frac_bc = frac_insol*frac_insol_thats_bc
        frac_dst = frac_insol*(1.-frac_insol_thats_bc)
        frac_ncl = frac_hygroscopic*frac_hygroscopic_thats_ncl
        
        return frac_so4, frac_pom, frac_soa, frac_bc, frac_dst, frac_ncl
    
    elif mode_name == 'mode2':
        frac_hygroscopic = X_comp[:,0]
        frac_hygroscopic_thats_ncl = X_comp[:,1]
        frac_so4 = frac_hygroscopic*(1.-frac_hygroscopic_thats_ncl)
        if no_soa:
            frac_soa = np.zeros_like(frac_hygroscopic)
            frac_pom = (1.-frac_hygroscopic)
        else:
            frac_soa = (1.-frac_hygroscopic)
            frac_pom = np.zeros_like(frac_hygroscopic)
        
        frac_ncl = frac_hygroscopic*frac_hygroscopic_thats_ncl
        
        return frac_so4, frac_soa, frac_ncl, frac_pom
    
    elif mode_name == 'mode4':
        frac_bc = X_comp[:,0]
        frac_pom = 1. - frac_bc
        
        return frac_bc, frac_pom


def get_mode_comp(mode_name,mode_varnames,mode_dists,mode_param1,mode_param2,lhd,no_soa=True):    
    X_comp = np.zeros(lhd.shape)
    for vv,varname in enumerate(mode_varnames):
        idx, = np.where([onevarname==varname for onevarname in mode_varnames])
        param1 = mode_param1[idx[0]]
        param2 = mode_param2[idx[0]]
        dist = mode_dists[idx[0]]
        X_comp[:,vv] = get_sample(lhd[:,vv],dist,param1,param2)
        
    if mode_name == 'mode1' or mode_name == 'mode3':
        frac_hygroscopic = X_comp[:,0]
        frac_hygroscopic_thats_ncl = X_comp[:,1]
        frac_hydrophobic_thats_insol = X_comp[:,2]        
        frac_insol_thats_bc = X_comp[:,3]
        if no_soa:
            frac_org_thats_pom = np.ones(X_comp[:,0].shape)
        else:
            frac_org_thats_pom = X_comp[:,4]
        
        frac_hydrophobic = 1. - frac_hygroscopic
        frac_insol = frac_hydrophobic*frac_hydrophobic_thats_insol
        frac_org = frac_hydrophobic*(1.-frac_hydrophobic_thats_insol)
        
        frac_so4 = frac_hygroscopic*(1. - frac_hygroscopic_thats_ncl)
        frac_pom = frac_org*frac_org_thats_pom
        frac_soa = frac_org*(1.-frac_org_thats_pom)
        frac_bc = frac_insol*frac_insol_thats_bc
        frac_dst = frac_insol*(1.-frac_insol_thats_bc)
        frac_ncl = frac_hygroscopic*frac_hygroscopic_thats_ncl
        
        return frac_so4, frac_pom, frac_soa, frac_bc, frac_dst, frac_ncl

    elif mode_name == 'mode2':        
        if no_soa:
            frac_hygroscopic = np.ones(X_comp[:,0].shape)
        else:
            frac_hygroscopic = X_comp[:,0]
        frac_hygroscopic_thats_ncl = X_comp[:,1]
        frac_so4 = frac_hygroscopic*(1.-frac_hygroscopic_thats_ncl)
        frac_soa = (1.-frac_hygroscopic)
        frac_ncl = frac_hygroscopic*frac_hygroscopic_thats_ncl
        
        return frac_so4, frac_soa, frac_ncl

    elif mode_name == 'mode4':
        frac_bc = X_comp[:,0]
        frac_pom = 1. - frac_bc
        
        return frac_bc, frac_pom

# =============================================================================
# compute aerosol properties; convert H2SO4
# =============================================================================
def get_Ntot(mode_Ns):
    return sum(mode_Ns)

def get_moment(p,mode_Ns,mode_mus,mode_sigs):
    return sum(mode_Ns*np.exp(p*mode_mus + 0.5*p**2*mode_sigs**2))
    # sum_this = np.array([mode_N*np.exp(p*mode_mu + 0.5*p**2*mode_sig**2) for (mode_N,mode_mu,mode_sig) in zip(mode_Ns,mode_mus,mode_sigs)])
    # return sum(sum_this)#sum(mode_Ns*np.exp(p*mode_mus + 0.5*p**2*mode_sigs**2))

def get_avg_flux(mode_Ns,mode_mus,mode_sigs,h2so40,dh2so4_dt_emit,t_max):
    dh2so4_dt = h2so40/t_max + dh2so4_dt_emit
    Atot = get_Atot(mode_Ns,mode_mus,mode_sigs)
    vol_flux = (dh2so4_dt/c.rho_h2so4)/Atot
    return vol_flux

def get_dh2so4dt_mol(vol_flux,mode_Ns,mode_mus,mode_sigs,p=101325.,temp = 273.):
    
    A_tot = get_Atot(mode_Ns,mode_mus,mode_sigs) # m^2 aerosol per m^3 air
    # A_tot = sum(get_Atot(mode_Ns,mode_mus,mode_sigs)) # m^2 aerosol per m^3 air
    dvolh2so4_dt = vol_flux*A_tot # [m^3/t h2so4]/[m^3 air]
    dmassh2so4_dt = dvolh2so4_dt*c.rho_h2so4 # (kg/t)/[m^3 air]
    dmolh2so4_dt = dmassh2so4_dt/c.MW_h2so4 # mol/t/[m^3 air]
    
    dh2so4dt_molPermol = dmolh2so4_dt*R*temp/p
    
    return dh2so4dt_molPermol # mol/time
    
def get_dh2so4_dt(vol_flux,mode_Ns,mode_mus,mode_sigs):
    A_tot = get_Atot(mode_Ns,mode_mus,mode_sigs)
    return vol_flux*A_tot*c.rho_h2so4

def h2so4_conc_to_molRatio(h2so4_conc,p=101325.,temp = 273.): # [mass h2so4]/[m^3 air]
    return h2so4_conc*R*temp/(c.MW_h2so4*p) # [mol]/[mol-air]

def get_Atot(mode_Ns,mode_mus,mode_sigs):
    
    p = 2
    return np.pi*get_moment(p,mode_Ns,mode_mus,mode_sigs)