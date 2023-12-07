#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions needed to create and run a single PartMC-MOSAIC scenario.

Code was tested with the following model versions: partmc-2.6.1 and mosaic-2012-01-25

author: Laura Fierce
"""

import os
import numpy as np

def create(print_scenario_here,settings,scenario_type='4mode_noemit',uniform_comp=False):
    default_inputs = default(scenario_type=scenario_type)
    inputs = update_inputs(default_inputs,settings)
    print_inputs(print_scenario_here,inputs,uniform_comp=uniform_comp)
    return inputs, default_inputs
    
def create_splitRepeats(print_scenario_here,settings,scenario_type='4mode_noemit',uniform_comp=False):
    spec_filenames = []
    repeat_ids = []
    for ii in range(settings['spec']['n_repeat']):
        default_inputs = default(scenario_type=scenario_type)
        repeat_id = str(ii+1).zfill(4) 
        print_repeat_here = print_scenario_here + 'repeat' + repeat_id + '/'
        if not os.path.exists(print_repeat_here):
            os.mkdir(print_repeat_here)
        if not os.path.exists(print_repeat_here + 'out'):
            os.mkdir(print_repeat_here + 'out/')
        settings_oneRepeat= settings
        settings_oneRepeat['spec']['n_repeat'] = 1
        output_prefix = print_repeat_here +'out/box_' + repeat_id
        settings_oneRepeat['spec']['output_prefix'] = output_prefix  # prefix of output files
        inputs = update_inputs(default_inputs,settings)
        spec_filename = print_inputs_oneRepeat(print_repeat_here,inputs,repeat_id,uniform_comp=uniform_comp)
        spec_filenames.append(spec_filename)
        repeat_ids.append(repeat_id)
        state_filename = print_scenario_here + '/state_' + str(ii + 1).zfill(4) + '.txt'
        initialize_state(state_filename)
        
    return inputs, default_inputs, spec_filenames, repeat_ids

def initialize_state(state_filename):
    with open(state_filename, 'w') as f:
        f.write(str(0))
    f.close()

def print_inputs_oneRepeat(print_repeat_here,inputs,repeat_id,uniform_comp=False):
    for input_group in inputs.keys():
        if input_group.endswith('data'):
            filename = input_group + '.dat'
            with open(print_repeat_here + filename,'w') as f:
                for line in inputs[input_group]:
                    f.write(line)
                    f.write('\n')
            f.close()
            os.system('chmod +r ' + print_repeat_here + filename)
        elif input_group.startswith('aero'):
            dist_filename = input_group + '_dist.dat'
            if input_group in ['aero_emit','aero_back']:
                filename = input_group + '.dat'
                f = open(print_repeat_here + filename,'w')
                for varname in ['time','rate','dist']:
                    if varname == 'dist':
                        f.write(' '.join([varname,dist_filename]))
                    else:
                        f.write(' '.join([varname,"{:.4e}".format(inputs[input_group][varname])]))
                    f.write('\n')
                f.close()
            
            f_dist = open(print_repeat_here + dist_filename,'w')
            for mode in inputs[input_group]['dist'].keys():
                f_dist.write(' '.join(['mode_name',str(mode)]))
                f_dist.write('\n')
                
                comp_filename = input_group + '_' + str(mode) + '_comp.dat'
                f_dist.write(' '.join(['mass_frac',comp_filename]))
                f_dist.write('\n')
                for varname in inputs[input_group]['dist'][mode].keys():
                    if varname == 'comp':    
                        f_comp = open(print_repeat_here + comp_filename,'w')
                        these_specs = [spec for spec in inputs[input_group]['dist'][mode][varname].keys() if inputs[input_group]['dist'][mode][varname][spec]>0]
                        if uniform_comp:
                            for spec in these_specs:
                                spec_partmc = 'SO4'
                                f_comp.write(' '.join([spec_partmc,str(1.)]))
                                f_comp.write('\n')                            
                        else:
                            for spec in these_specs:
                                spec_partmc = mam_to_partmc_spec(spec) 
                                f_comp.write(' '.join([spec_partmc,str(inputs[input_group]['dist'][mode][varname][spec])]))
                                f_comp.write('\n')
                        f_comp.flush()
                        f_comp.close()
                    else:
                        f_dist.write(' '.join([varname,str(inputs[input_group]['dist'][mode][varname])]))
                        f_dist.write('\n')
                f_dist.write('\n')
            f_dist.close()
        else:
            if input_group == 'spec':
                filename = 'urban_plume_' + repeat_id + '.spec'
                spec_filename = print_repeat_here + filename 
            else:
                filename = input_group + '.dat'
            f2 = open(print_repeat_here + filename,'w')
            for varname in inputs[input_group].keys():
                f2.write(' '.join([varname,str(inputs[input_group][varname])]))
                f2.write('\n')
            f2.write('\n')
            f2.close()
    return spec_filename
            
def print_inputs(print_scenario_here,inputs,uniform_comp=False):
    for input_group in inputs.keys():
        print(input_group,inputs[input_group])
        if input_group.startswith('aero'):
            dist_filename = input_group + '_dist.dat'
            if input_group in ['aero_emit','aero_back']:
                filename = input_group + '.dat'
                f = open(print_scenario_here + filename,'w')
                for varname in ['time','rate','dist']:
                    if varname == 'dist':
                        f.write(' '.join([varname,dist_filename]))
                    else:
                        f.write(' '.join([varname,"{:.4e}".format(inputs[input_group][varname])]))
                    f.write('\n')
                f.close()
            
            f_dist = open(print_scenario_here + dist_filename,'w')
            for mode in inputs[input_group]['dist'].keys():
                f_dist.write(' '.join(['mode_name',str(mode)]))
                f_dist.write('\n')
                
                comp_filename = input_group + '_' + str(mode) + '_comp.dat'
                f_dist.write(' '.join(['mass_frac',comp_filename]))
                f_dist.write('\n')
                for varname in inputs[input_group]['dist'][mode].keys():
                    if varname == 'comp':    
                        f_comp = open(print_scenario_here + comp_filename,'w')
                        these_specs = [spec for spec in inputs[input_group]['dist'][mode][varname].keys() if inputs[input_group]['dist'][mode][varname][spec]>0]
                        if uniform_comp:
                            for spec in these_specs:
                                spec_partmc = 'SO4'
                                f_comp.write(' '.join([spec_partmc,str(1.)]))
                                f_comp.write('\n')                            
                        else:
                            for spec in these_specs:
                                spec_partmc = mam_to_partmc_spec(spec) 
                                f_comp.write(' '.join([spec_partmc,str(inputs[input_group]['dist'][mode][varname][spec])]))
                                f_comp.write('\n')
                        f_comp.flush()
                        f_comp.close()
                    else:
                        f_dist.write(' '.join([varname,str(inputs[input_group]['dist'][mode][varname])]))
                        f_dist.write('\n')
                f_dist.write('\n')
            f_dist.close()
        else:
            if input_group == 'spec':
                filename = 'urban_plume.spec'
            else:
                filename = input_group + '.dat'
            f2 = open(print_scenario_here + filename,'w')
            for varname in inputs[input_group].keys():
                f2.write(' '.join([varname,str(inputs[input_group][varname])]))
                f2.write('\n')
            f2.write('\n')
            f2.close()

def mam_to_partmc_spec(mam_spec):
    if mam_spec.upper() == 'POM': # or mam_spec.upper() == 'SOA':
        partmc_spec = 'OC'
    elif mam_spec.upper() == 'DST':
        partmc_spec = 'OIN'
    elif mam_spec.upper() == 'NA':
        partmc_spec = 'Na'
    elif mam_spec.upper() == 'CL':
        partmc_spec = 'Cl'
    # elif mam_spec.upper() == 'NCL':
    else:
        partmc_spec = mam_spec.upper()
    return partmc_spec
    
def update_inputs(default_inputs,settings):
    inputs = default_inputs.copy()
    for input_group in settings.keys():
        for varname in settings[input_group].keys():
            inputs[input_group][varname] = settings[input_group][varname]
    
    if inputs['spec']['do_mosaic'] == 'no':
        del inputs['spec']['do_optical']
    
    if inputs['spec']['do_coagulation'] == 'no' and inputs['spec']['run_type'] == 'particle':
        del inputs['spec']['coag_kernel']
    
    if inputs['spec']['do_nucleation'] == 'no':
        del inputs['spec']['nucleate']
        
    if inputs['spec']['run_type'] == 'sectional':
        del inputs['spec']['n_repeat']
        del inputs['spec']['n_part']
        del inputs['spec']['restart']
        del inputs['spec']['do_camp_chem']
        del inputs['spec']['gas_init']
    elif inputs['spec']['run_type'] == 'particle':
        if 'n_bin' in inputs['spec']:
            del inputs['spec']['n_bin']
        if 'd_min' in inputs['spec']:
            del inputs['spec']['d_min']
        if 'd_max' in inputs['spec']:
            del inputs['spec']['d_max']
    
    return inputs

def default(scenario_type='4mode_noemit'):
    
    if scenario_type == '4mode_noemit':
        spec = {}
        spec['run_type'] = 'particle'
        spec['output_prefix'] = 'out/box'   # prefix of output files
        spec['n_repeat'] = 5                # number of Monte Carlo repeats
        spec['n_part'] = 5000               # total number of particles
        spec['restart'] = 'no'              # whether to restart from saved state (yes/no)
        spec['t_max'] = 86400               # total simulation time (s)
        spec['del_t'] = 60                  # timestep (s)
        spec['t_output'] = 3600             # output interval (0 disables) (s)
        spec['t_progress'] = 600              # progress printing interval (0 disables) (s)
        spec['n_bin'] = 10
        spec['d_min'] =  1e-10              # minimum diameter (m)
        spec['d_max'] = 1e-4                     # maximum diameter (m)
        
        spec['do_camp_chem'] = 'no'         # whether to use CAMP for chemistry
        
        spec['gas_data'] = 'gas_data.dat'    # file containing gas data
        spec['gas_init'] = 'gas_init.dat'    # initial gas concentrations
        
        spec['aerosol_data'] = 'aero_data.dat'      # file containing aerosol data
        spec['do_fractal'] = 'no'                  # whether to do fractal treatment
        spec['aerosol_init'] = 'aero_init_dist.dat' # aerosol initial condition file
        
        spec['temp_profile'] = 'temp.dat'           # temperature profile file
        spec['pressure_profile'] = 'pres.dat'       # pressure profile file
        spec['height_profile'] = 'height.dat'       # height profile file
        spec['gas_emissions'] = 'gas_emit.dat'      # gas emissions file
        spec['gas_background'] = 'gas_back.dat'     # background gas concentrations file
        spec['aero_emissions'] = 'aero_emit.dat'    # aerosol emissions file
        spec['aero_background'] = 'aero_back.dat'   # aerosol background file
        spec['loss_function'] = 'none'              # loss function specification
        
        spec['rel_humidity'] = 0.95               # initial relative humidity (1)
        spec['latitude'] = 0                      # latitude (degrees, -90 to 90)
        spec['longitude'] = 0                     # longitude (degrees, -180 to 180)
        spec['altitude'] = 0                      # altitude (m)
        spec['start_time'] = 21600                # start time (s since 00:00 UTC)
        spec['start_day'] = 200                   # start day of year (UTC)
        
        spec['do_coagulation'] = 'yes'              # whether to do coagulation (yes/no)
        spec['coag_kernel'] = 'brown'               # coagulation kernel
        spec['do_condensation'] = 'no'              # whether to do condensation (yes/no)
        spec['do_mosaic'] = 'yes'                   # whether to do MOSAIC (yes/no)
        spec['do_optical'] = 'no'                   # whether to compute optical props (yes/no)
        spec['do_nucleation'] = 'yes'                # whether to do nucleation (yes/no)
        spec['nucleate'] = 'sulf_acid'               # whether to do nucleation (yes/no)
        
        spec['rand_init'] = 0                     # random initialization (0 to use time)
        spec['allow_doubling'] = 'yes'              # whether to allow doubling (yes/no)
        spec['allow_halving'] = 'yes'               # whether to allow halving (yes/no)
        spec['do_select_weighting'] = 'yes'          # whether to select weighting explicitly (yes/no)
        spec['weight_type']= 'power'
        spec['weighting_exponent'] = 0
        spec['record_removals'] = 'yes'             # whether to record particle removals (yes/no)
        spec['do_parallel'] = 'no'                  # whether to run in parallel (yes/no)
        
    # =============================================================================
    # evironmental properties
    # =============================================================================
        height = {}
        height['time'] = 0.
        height['height'] = 290.
        
        temp = {}
        temp['time'] = 0.
        temp['temp'] = 200.
    
        pres = {}
        pres['time'] = 0.
        pres['pressure'] = 101325.
        
    # =============================================================================
    # initial aerosol and gas
    # =============================================================================
        aero_init_dists = {}
        aero_init_dists[1]={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':3.2e9,
                           'geom_mean_diam':2e-8,
                           'log10_geom_std_dev':0.161,
                           'comp':{
                               'SO4':1.,
                               'OC':0.,
                               'BC':0.,
                               'OIN':0.,
                               'Na':0.,
                               'Cl':0.}}
        
        aero_init_dists[2]={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':3.2e9,
                           'geom_mean_diam':2e-8,
                           'log10_geom_std_dev':0.161,
                           'comp':{
                               'SO4':1.,
                               'Na':0.,
                               'Cl':0.}}
    
        aero_init_dists[3]={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':3.2e9,
                           'geom_mean_diam':2e-8,
                           'log10_geom_std_dev':0.161,
                           'comp':{
                               'SO4':1.,
                               'OC':0.,
                               'BC':0.,
                               'OIN':0.,
                               'Na':0.,
                               'Cl':0.}}
        
        aero_init_dists[4]={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':3.2e9,
                           'geom_mean_diam':2e-8,
                           'log10_geom_std_dev':0.161,
                           'comp':{
                               'OC':1.,
                               'BC':0.}}
        
        aero_init = {}
        aero_init['dist'] = aero_init_dists
        
        # initial gas concentrations
        gas_init = {}
        gas_init['H2SO4'] = 1.0 # ppb
    
    # =============================================================================
    # background aerosol and gas
    # =============================================================================
        # background aerosol
        aero_back_dists = {}
        aero_back_dists['back']={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':3.2e9,
                           'geom_mean_diam':2e-8,
                           'log10_geom_std_dev':0.161,
                           'comp':{'SO4':1},
                           }
        aero_back = {}
        aero_back['time'] = 0.
        aero_back['rate'] = 0.
        aero_back['dist'] = aero_back_dists
        
        # background gas concentrations
        gas_back = get_gas_back(background_type='4mode_noemit')
        
    # =============================================================================
    # aerosol and gas emissions
    # =============================================================================
        # aerosol emissions
        aero_emit = {}
        aero_emit['time'] = 0.
        aero_emit['rate'] = 0.
        aero_emit_dists = {}
        aero_emit_dists['emit']={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':3.2e9,
                           'geom_mean_diam':2e-8,
                           'log10_geom_std_dev':0.161,
                           'comp':{'SO4':1},
                           }
        aero_emit['dist'] = aero_emit_dists
        
        # gas emissions
        gas_emit = {}
        gas_emit['time'] = 0.
        gas_emit['rate'] = 1.
        gas_emit['H2SO4'] = 0.
        
        default_inputs = {}
        default_inputs['spec'] = spec
        default_inputs['height'] = height
        default_inputs['temp'] = temp
        default_inputs['pres'] = pres        
        default_inputs['aero_init'] = aero_init
        default_inputs['gas_init'] = gas_init
        default_inputs['aero_emit'] = aero_emit
        default_inputs['gas_emit'] = gas_emit
        default_inputs['aero_back'] = aero_back
        default_inputs['gas_back'] = gas_back
        default_inputs['gas_data'] = get_gas_data()
        default_inputs['aero_data'] = get_aero_data()        
        
    elif scenario_type == 'volcano':
        spec = {}
        spec['run_type'] = 'particle'
        spec['output_prefix'] = 'out/box'   # prefix of output files
        spec['n_repeat'] = 5                # number of Monte Carlo repeats
        spec['n_part'] = 5000               # total number of particles
        spec['restart'] = 'no'              # whether to restart from saved state (yes/no)
        spec['t_max'] = 86400               # total simulation time (s)
        spec['del_t'] = 60                  # timestep (s)
        spec['t_output'] = 3600             # output interval (0 disables) (s)
        spec['t_progress'] = 600              # progress printing interval (0 disables) (s)
        spec['n_bin'] = 10
        spec['d_min'] =  1e-10              # minimum diameter (m)
        spec['d_max'] = 1e-4                     # maximum diameter (m)
        
        spec['do_camp_chem'] = 'no'         # whether to use CAMP for chemistry
        
        spec['gas_data'] = 'gas_data.dat'    # file containing gas data
        spec['gas_init'] = 'gas_init.dat'    # initial gas concentrations
        
        spec['aerosol_data'] = 'aero_data.dat'      # file containing aerosol data
        spec['do_fractal'] = 'no'                  # whether to do fractal treatment
        spec['aerosol_init'] = 'aero_init_dist.dat' # aerosol initial condition file
        
        spec['temp_profile'] = 'temp.dat'           # temperature profile file
        spec['pressure_profile'] = 'pres.dat'       # pressure profile file
        spec['height_profile'] = 'height.dat'       # height profile file
        spec['gas_emissions'] = 'gas_emit.dat'      # gas emissions file
        spec['gas_background'] = 'gas_back.dat'     # background gas concentrations file
        spec['aero_emissions'] = 'aero_emit.dat'    # aerosol emissions file
        spec['aero_background'] = 'aero_back.dat'   # aerosol background file
        spec['loss_function'] = 'none'              # loss function specification
        
        spec['rel_humidity'] = 0.99               # initial relative humidity (1)
        spec['latitude'] = 0                      # latitude (degrees, -90 to 90)
        spec['longitude'] = 0                     # longitude (degrees, -180 to 180)
        spec['altitude'] = 0                      # altitude (m)
        spec['start_time'] = 21600                # start time (s since 00:00 UTC)
        spec['start_day'] = 200                   # start day of year (UTC)
        
        spec['do_coagulation'] = 'yes'              # whether to do coagulation (yes/no)
        spec['coag_kernel'] = 'brown'               # coagulation kernel
        spec['do_condensation'] = 'no'              # whether to do condensation (yes/no)
        spec['do_mosaic'] = 'yes'                   # whether to do MOSAIC (yes/no)
        spec['do_optical'] = 'no'                   # whether to compute optical props (yes/no)
        spec['do_nucleation'] = 'yes'               # whether to do nucleation (yes/no)
        spec['nucleate'] = 'sulf_acid'               # whether to do nucleation (yes/no)
        
        spec['rand_init'] = 0                     # random initialization (0 to use time)
        spec['allow_doubling'] = 'yes'              # whether to allow doubling (yes/no)
        spec['allow_halving'] = 'yes'               # whether to allow halving (yes/no)
        spec['do_select_weighting'] = 'yes'          # whether to select weighting explicitly (yes/no)
        spec['weight_type']= 'power'
        spec['weighting_exponent'] = -3
        spec['record_removals'] = 'yes'             # whether to record particle removals (yes/no)
        spec['do_parallel'] = 'no'                  # whether to run in parallel (yes/no)
        
    # =============================================================================
    # evironmental properties
    # =============================================================================
        height = {}
        height['time'] = 0.
        height['height'] = 290.
        
        temp = {}
        temp['time'] = 0.
        temp['temp'] = 200.
    
        pres = {}
        pres['time'] = 0.
        pres['pressure'] = 101325.
        
    # =============================================================================
    # initial aerosol and gas
    # =============================================================================
        # initial aerosol
        gmd_accum = 2e-7
        gmd_aitken = 5e-8
        gmd_coarse = 3e-6
        
        gsd_accum = 1.8
        gsd_aitken = 1.6
        gsd_coarse = 1.8
        
        so4_accum = 7.96e-9 # kg/m3
        so4_aitken = 2.6e-10 # kg/m3
        so4_coarse = 8.1e-11 # kg/m3
        
        rho_so4 = 1800. # kg/m^3
        
        get_num = lambda vol_conc, gmd, gsd: vol_conc/(np.pi/6.*np.exp(3*np.log(gmd) + 9./2.*np.log(gsd)**2))
        num_accum = get_num(so4_accum/rho_so4, gmd_accum, gsd_accum)
        num_aitken = get_num(so4_aitken/rho_so4, gmd_aitken, gsd_aitken)
        num_coarse = get_num(so4_coarse/rho_so4, gmd_coarse, gsd_coarse)
        aero_init_dists = {}
        
        aero_init_dists[1] = {
            # 'mode_name':'init',
            # 'mass_frac':'aero_init_comp.dat',   # composition proportions of species
            'diam_type':'geometric',            # type of diameter specified
            'mode_type':'log_normal',           # type of distribution
            'num_conc':num_accum,                   # particle number concentration (#/m^3)
            'geom_mean_diam':gmd_accum,              # geometric mean diameter (m)
            'log10_geom_std_dev':np.log10(gsd_accum),         # log_10 of geometric std dev of diameter'
            'comp':{'SO4':1.} # mass fraction
            }
            
        aero_init_dists[2] = {
            # 'mode_name':'init_large',
            # 'mass_frac':'aero_init_comp.dat',   # composition proportions of species
            'diam_type':'geometric',            # type of diameter specified
            'mode_type':'log_normal',           # type of distribution
            'num_conc':num_aitken,                   # particle number concentration (#/m^3)
            'geom_mean_diam':gmd_aitken,              # geometric mean diameter (m)
            'log10_geom_std_dev':np.log10(gsd_aitken),         # log_10 of geometric std dev of diameter'
            'comp':{'SO4':1.} # mass fraction
            }
        
        aero_init_dists[3] = {
            # 'mode_name':'init_large',
            # 'mass_frac':'aero_init_comp.dat',   # composition proportions of species
            'diam_type':'geometric',            # type of diameter specified
            'mode_type':'log_normal',           # type of distribution
            'num_conc':num_coarse,                   # particle number concentration (#/m^3)
            'geom_mean_diam':gmd_coarse,              # geometric mean diameter (m)
            'log10_geom_std_dev':np.log10(gsd_coarse),         # log_10 of geometric std dev of diameter'
            'comp':{'SO4':1.} # mass fraction
            }
        
        aero_back_dists = {}
        aero_back_dists['back_small']={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':4.5e7,
                           'geom_mean_diam':7e-9,
                           'log10_geom_std_dev':0.6,
                           'comp':{'SO4':1},
                           }
        
        aero_back_dists = {}
        aero_back_dists['back_large']={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':6e7,
                           'geom_mean_diam':25e-7,
                           'log10_geom_std_dev':0.24,
                           'comp':{'SO4':1},
                           }

        
        # aero_init_dists[1] = {
        #     # 'mode_name':'init',
        #     # 'mass_frac':'aero_init_comp.dat',   # composition proportions of species
        #     'diam_type':'geometric',            # type of diameter specified
        #     'mode_type':'log_normal',           # type of distribution
        #     'num_conc':5.0e8,                   # particle number concentration (#/m^3)
        #     'geom_mean_diam':2e-8,              # geometric mean diameter (m)
        #     'log10_geom_std_dev':0.161,         # log_10 of geometric std dev of diameter'
        #     'comp':{'SO4':1.} # mass fraction
        #     }
            
        # aero_init_dists[2] = {
        #     # 'mode_name':'init_large',
        #     # 'mass_frac':'aero_init_comp.dat',   # composition proportions of species
        #     'diam_type':'geometric',            # type of diameter specified
        #     'mode_type':'log_normal',           # type of distribution
        #     'num_conc':5.0e8,                   # particle number concentration (#/m^3)
        #     'geom_mean_diam':1.16e-7,              # geometric mean diameter (m)
        #     'log10_geom_std_dev':0.217,         # log_10 of geometric std dev of diameter'
        #     'comp':{'SO4':1.} # mass fraction
        #     }
        
        # aero_init_dists[3] = {
        #     # 'mode_name':'init_large',
        #     # 'mass_frac':'aero_init_comp.dat',   # composition proportions of species
        #     'diam_type':'geometric',            # type of diameter specified
        #     'mode_type':'log_normal',           # type of distribution
        #     'num_conc':5.0e8,                   # particle number concentration (#/m^3)
        #     'geom_mean_diam':1.16e-7,              # geometric mean diameter (m)
        #     'log10_geom_std_dev':0.217,         # log_10 of geometric std dev of diameter'
        #     'comp':{'SO4':1.} # mass fraction
        #     }
        
        
        aero_init = {}
        aero_init['dist'] = aero_init_dists
        
        # initial gas concentrations
        gas_init = get_gas_back(background_type='urban')
        gas_init['SO2'] = 50.0 # ppb
        del gas_init['time']
        del gas_init['rate']   
    
    # =============================================================================
    # background aerosol and gas
    # =============================================================================
        # background aerosol
        aero_back_dists = {}
        
        # page 430 Seinfeld & Pandis: remote continental
        # aero_back_dists['back_small']={'diam_type':'geometric',
        #                    'mode_type':'log_normal',
        #                    'num_conc':3.2e9,
        #                    'geom_mean_diam':2e-8,
        #                    'log10_geom_std_dev':0.161,
        #                    'comp':{'SO4':1,'NH4':0.375,'OC':1},
        #                    }
        
        # aero_back_dists['back_large']={'diam_type':'geometric',
        #                    'mode_type':'log_normal',
        #                    'num_conc':2.9e9,
        #                    'geom_mean_diam':1.16e-7,
        #                    'log10_geom_std_dev':0.217,
        #                    'comp':{'SO4':1,'NH4':0.375,'OC':1},
        #                    }

        aero_back_dists['freetrop1']={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':129e6,
                           'geom_mean_diam':0.007e-6,
                           'log10_geom_std_dev':0.645,
                           'comp':{'SO4':1,'NH4':0.375,'OC':1},
                           }
        
        aero_back_dists['freetrop2']={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':60e6,
                           'geom_mean_diam':0.253e-6,
                           'log10_geom_std_dev':0.253,
                           'comp':{'SO4':1,'NH4':0.375,'OC':1},
                           }
        aero_back_dists['freetrop3']={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':63.5e6,
                           'geom_mean_diam':0.52e-6,
                           'log10_geom_std_dev':0.425,
                           'comp':{'SO4':1,'NH4':0.375,'OC':1},
                           }        
        
        dilution_rate = 5e-6
        aero_back = {}
        aero_back['time'] = 0.
        aero_back['rate'] = dilution_rate
        aero_back['dist'] = aero_back_dists
        
        # background gas concentrations        
        gas_back = get_gas_back(background_type='urban')
        gas_back['rate'] = dilution_rate
        
    # =============================================================================
    # aerosol and gas emissions
    # =============================================================================
        # aerosol emissions
        aero_emit = {}
        aero_emit['time'] = 0.
        aero_emit['rate'] = 0.
        aero_emit_dists = {}
        aero_emit_dists['emit']={'diam_type':'geometric',
                           'mode_type':'log_normal',
                           'num_conc':3.2e9,
                           'geom_mean_diam':2e-8,
                           'log10_geom_std_dev':0.161,
                           'comp':{'SO4':1},
                           }
        aero_emit['dist'] = aero_emit_dists
        
        # gas emissions
        gas_emit = {}
        gas_emit['time'] = 0.
        gas_emit['rate'] = 0.
        gas_emit['SO2'] = 0.#5.0e-9
        
        default_inputs = {}
        default_inputs['spec'] = spec
        default_inputs['height'] = height
        default_inputs['temp'] = temp
        default_inputs['pres'] = pres
        default_inputs['aero_init'] = aero_init
        default_inputs['gas_init'] = gas_init
        default_inputs['aero_emit'] = aero_emit
        default_inputs['gas_emit'] = gas_emit
        default_inputs['aero_back'] = aero_back
        default_inputs['gas_back'] = gas_back
        default_inputs['gas_data'] = get_gas_data()
        default_inputs['aero_data'] = get_aero_data()        
        
    return default_inputs

def get_gas_back(background_type='urban'):
    gas_back = {}
    # time (s)
    # rate (s^{-1})
    # concentrations (ppb)
    if background_type == 'urban':
        gas_back['time'] = 0.
        gas_back['rate'] = 1.5e-5
        gas_back['NO'] = 0.1E+00
        gas_back['NO2'] = 1.0
        gas_back['NO3'] = 0.
        gas_back['N2O5'] = 0.
        gas_back['HONO'] = 0.
        gas_back['HNO3'] = 1.0E+00
        gas_back['HNO4'] = 0.0E+00
        gas_back['O3'] = 5.0E+01
        gas_back['O1D'] = 0.0E+00
        gas_back['O3P'] = 0.0E+00
        gas_back['OH'] = 0.0E+00
        gas_back['HO2'] = 0.0E+00
        gas_back['H2O2'] = 1.1E+00
        gas_back['CO'] = 2.1E+02
        gas_back['SO2'] = 0.8E+00
        gas_back['H2SO4'] = 0.0E+00
        gas_back['NH3'] = 0.5E+00
        gas_back['HCl'] = 0.7E+00
        gas_back['CH4'] = 2.2E+03
        gas_back['C2H6'] = 1.0E+00
        gas_back['CH3O2'] = 0.0E+00
        gas_back['ETHP'] = 0.0E+00
        gas_back['HCHO'] = 1.2E+00
        gas_back['CH3OH'] = 1.2E-01
        gas_back['CH3OOH'] = 0.5E+00
        gas_back['ETHOOH'] = 0.0E+00
        gas_back['ALD2'] = 1.0E+00
        gas_back['HCOOH'] = 0.0E+00
        gas_back['PAR'] = 2.0E+00
        gas_back['AONE'] = 1.0E+00
        gas_back['MGLY'] = 0.0E+00
        gas_back['ETH'] = 0.2E+00
        gas_back['OLET'] = 2.3E-02
        gas_back['OLEI'] = 3.1E-04
        gas_back['TOL'] = 0.1E+00
        gas_back['XYL'] = 0.1E+00
        gas_back['CRES'] = 0.0E+00
        gas_back['TO2'] = 0.0E+00
        gas_back['CRO'] = 0.0E+00
        gas_back['OPEN'] = 0.0E+00
        gas_back['ONIT'] = 0.1E+00
        gas_back['PAN'] = 0.8E+00
        gas_back['RCOOH'] = 0.2E+00
        gas_back['ROOH'] = 2.5E-02
        gas_back['C2O3'] = 0.0E+00
        gas_back['RO2'] = 0.0E+00
        gas_back['ANO2'] = 0.0E+00
        gas_back['NAP'] = 0.0E+00
        gas_back['ARO1'] = 0.0E+00
        gas_back['ARO2'] = 0.0E+00
        gas_back['ALK1'] = 0.0E+00
        gas_back['OLE1'] = 0.0E+00
        gas_back['XO2'] = 0.0E+00
        gas_back['XPAR'] = 0.0E+00
        gas_back['ISOP'] = 0.5E+00
        gas_back['API'] = 0.0E+00
        gas_back['LIM'] = 0.0E+00
        gas_back['API1'] = 0.0E+00
        gas_back['API2'] = 0.0E+00
        gas_back['LIM1'] = 0.0E+00
        gas_back['LIM2'] = 0.0E+00
        gas_back['ISOPRD'] = 0.0E+00
        gas_back['ISOPP'] = 0.0E+00
        gas_back['ISOPN'] = 0.0E+00
        gas_back['ISOPO2'] = 0.0E+00
        gas_back['DMS'] = 0.0E+00
        gas_back['MSA'] = 0.0E+00
        gas_back['DMSO'] = 0.0E+00
        gas_back['DMSO2'] = 0.0E+00
        gas_back['CH3SO2H'] = 0.0E+00
        gas_back['CH3SCH2OO'] = 0.0E+00
        gas_back['CH3SO2'] = 0.0E+00
        gas_back['CH3SO3'] = 0.0E+00
        gas_back['CH3SO2OO'] = 0.0E+00
    elif background_type == '4mode_noemit':
        gas_back['time'] = 0.
        gas_back['rate'] = 0.
        gas_back['H2SO4'] = 1.0 # ppb
    return gas_back
    
def get_aero_data(fromMAM=True):
    if fromMAM:
        aero_data = [
            '#     dens (kg/m^3)   ions in soln (1)    molec wght (kg/mole)   kappa (1)',
            'SO4            1770                0                   96d-3      0.507',
            'NO3            1770                0                   62d-3      0.507',
            'Cl             1900                0                   35.5d-3    1.160',
            'NH4            1770                0                   18d-3      0.507',
            'MSA            1770                0                   95d-3      0.53',
            'ARO1           1000                0                   150d-3     0.14',
            'ARO2           1000                0                   150d-3     0.14',
            'ALK1           1000                0                   140d-3     0.14',
            'OLE1           1000                0                   140d-3     0.14',
            'API1           1000                0                   184d-3     0.14',
            'API2           1000                0                   184d-3     0.14',
            'LIM1           1000                0                   200d-3     0.14',
            'LIM2           1000                0                   200d-3     0.14',
            'CO3            2600                0                   60d-3      0.53',
            'Na             1900                0                   23d-3      0.53',
            'Ca             2600                0                   40d-3      0.53',
            'OIN            2600                0                   1d-3       0.068',
            'OC             1000                0                   1d-3       0.01',
            'BC             1700                0                   1d-3       1.0d-10',
            'H2O            1000                0                   18d-3      0']
    else:
        aero_data = [
            '#     dens (kg/m^3)   ions in soln (1)    molec wght (kg/mole)   kappa (1)',
            'SO4            1800                0                   96d-3      0.65',
            'NO3            1800                0                   62d-3      0.65',
            'Cl             2200                0                   35.5d-3    0.53',
            'NH4            1800                0                   18d-3      0.65',
            'MSA            1800                0                   95d-3      0.53',
            'ARO1           1400                0                   150d-3     0.1',
            'ARO2           1400                0                   150d-3     0.1',
            'ALK1           1400                0                   140d-3     0.1',
            'OLE1           1400                0                   140d-3     0.1',
            'API1           1400                0                   184d-3     0.1',
            'API2           1400                0                   184d-3     0.1',
            'LIM1           1400                0                   200d-3     0.1',
            'LIM2           1400                0                   200d-3     0.1',
            'CO3            2600                0                   60d-3      0.53',
            'Na             2200                0                   23d-3      1.160',
            'Ca             2600                0                   40d-3      0.53',
            'OIN            2600                0                   1d-3       0.1',
            'OC             1000                0                   1d-3       0.001',
            'BC             1800                0                   1d-3       0',
            'H2O            1000                0                   18d-3      0']
    return aero_data
    
def get_gas_data():
    gas_data = [
        '# list of gas species',
        'H2SO4',
        'HNO3',
        'HCl',
        'NH3',
        'NO',
        'NO2',
        'NO3',
        'N2O5',
        'HONO',
        'HNO4',
        'O3',
        'O1D',
        'O3P',
        'OH',
        'HO2',
        'H2O2',
        'CO',
        'SO2',
        'CH4',
        'C2H6',
        'CH3O2',
        'ETHP',
        'HCHO',
        'CH3OH',
        'ANOL',
        'CH3OOH',
        'ETHOOH',
        'ALD2',
        'HCOOH',
        'RCOOH',
        'C2O3',
        'PAN',
        'ARO1',
        'ARO2',
        'ALK1',
        'OLE1',
        'API1',
        'API2',
        'LIM1',
        'LIM2',
        'PAR',
        'AONE',
        'MGLY',
        'ETH',
        'OLET',
        'OLEI',
        'TOL',
        'XYL',
        'CRES',
        'TO2',
        'CRO',
        'OPEN',
        'ONIT',
        'ROOH',
        'RO2',
        'ANO2',
        'NAP',
        'XO2',
        'XPAR',
        'ISOP',
        'ISOPRD',
        'ISOPP',
        'ISOPN',
        'ISOPO2',
        'API',
        'LIM',
        'DMS',
        'MSA',
        'DMSO',
        'DMSO2',
        'CH3SO2H',
        'CH3SCH2OO',
        'CH3SO2',
        'CH3SO3',
        'CH3SO2OO',
        'CH3SO2CH2OO',
        'SULFHOX']
    return gas_data