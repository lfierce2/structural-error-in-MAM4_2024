#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import numpy as np
import netCDF4, os, shutil
import partmc_scenario, mam_scenario#, quam_scenario
from scipy.constants import R
from scipy.special import erfinv
import constants as c
from pyDOE import lhs
from time import sleep
import random 


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

def mam_ensemble_create_the_rest(
        ensemble_dir,scenarios_to_create,all_ensemble_settings,#all_mu1_vals,all_mu2_vals,all_mu3_vals,all_mu4_vals,
        scenario_type='4mode_noemit',uniform_comp=False,
        mam_build_location='/people/fier887/mam_refactor/build/standalone',
        mam_input='/people/fier887/mam_refactor/standalone/tests/smoke_test.nl',
        mam_output = '/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc',
        core_dir = '/people/fier887/mam_refactor/core/'):
    
    
    ensemble_settings = []
    for qq,scenario in enumerate(scenarios_to_create):
        idx, = np.array([ii for ii in range(len(all_ensemble_settings)) if all_ensemble_settings[ii]['scenario'] == scenario])
        ensemble_settings.append(all_ensemble_settings[idx])
    
    mam_ensemble_create(
        ensemble_dir,ensemble_settings,#,mu2_vals,mu3_vals,mu4_vals,
        scenario_type='4mode_noemit',
        mam_build_location=mam_build_location,
        mam_input=mam_input,
        mam_output=mam_output,
        core_dir=core_dir)

def mam_ensemble_create(
        ensemble_dir,ensemble_settings,#mu1_vals,mu2_vals,mu3_vals,mu4_vals,
        scenario_type='4mode_noemit',uniform_comp=False,
        mam_build_location='/people/fier887/mam_refactor/build/standalone',
        mam_input='/people/fier887/mam_refactor/standalone/tests/smoke_test.nl',
        mam_output = '/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc',
        core_dir = '/people/fier887/mam_refactor/core/'):
    
    start = 0
    end = len(ensemble_settings)
    # ensemble_name = ensemble_dir[ensemble_dir.find('ensemble'):]
    # if ensemble_name.endswith('/'):
    #     ensemble_name = ensemble_name[:-1]
    
    for ii in range(start,end):
        mam_settings = ensemble_settings[ii]
        mode_mus = np.array([
            ensemble_settings[ii]['psd_params']['mu1'],ensemble_settings[ii]['psd_params']['mu2'],
            ensemble_settings[ii]['psd_params']['mu3'],ensemble_settings[ii]['psd_params']['mu4']])
        # mode_sigs = np.array([ensemble_settings[ii]['sig1'],ensemble_settings[ii]['sig2'],ensemble_settings[ii]['sig3'],ensemble_settings[ii]['sig4']])
        
        scenario = str(ii).zfill(6)
        scenario_dir = ensemble_dir + scenario + '/'
        print(scenario_dir)
        if not os.path.exists(scenario_dir):
            print('doesn\'t exist')
            # os.mkdir(scenario_dir)
        
        createOnly_mam(
            scenario_dir,mam_settings,scenario_type,mode_mus,uniform_comp=uniform_comp,
            mam_build_location=mam_build_location,
            mam_input=mam_input,mam_output=mam_output,
            core_dir=core_dir)
        
        # try:
        #     with open('state_create_' + ensemble_name, 'r') as file:
        #         curr_state_text = file.read()
        #     file.close()
        # except:
        #     curr_state_text = ''
        # if len(curr_state_text) == 0:
        #     curr_state = [scenario]
        # elif '\n' not in curr_state_text:
        #     curr_state = [curr_state_text,scenario]
        # else:
        #     curr_state = curr_state_text.split('\n')
        #     curr_state.append(scenario)
            # data_into_list = data
            # file.close()
            # if len(curr_state) == 0:
            #     curr_state = [scenario]
            # elif type(curr_state) == list:
            #     curr_state.append(scenario)
            # elif type(curr_state) == str:
            #     curr_state = [curr_state,scenario]
            # else:
            #     print('poop')
        # with open('state_create_' + ensemble_name, 'w') as file:
        #     file.write('\n'.join(map(str, curr_state)))
        # file.close()

def ensemble_run_some(
        ensemble_dir,scenarios_to_run,
        n_part = 2000, n_repeat = 5,
        t_max = None, t_output = None,
        num_multiplier = 0.773378248237437, uniform_comp=False,
        gsds = [1.6,1.8,1.8,1.6],
        mam_build_location='/people/fier887/mam_refactor/build/standalone',
        mam_input='/people/fier887/mam_refactor/standalone/tests/smoke_test.nl',
        mam_output = '/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc',
        core_dir = '/people/fier887/mam_refactor/core/',
        partmc_run_dir = '/people/fier887/partmc-2.6.1/scenarios/7_box_eval/'):
    
    ensemble_name = ensemble_dir[ensemble_dir.find('ensemble'):-1]
    ensemble_name = ensemble_name[:ensemble_name.find('/')]
    if ensemble_name.endswith('/'):
        ensemble_name = ensemble_name[:-1]
    
    for scenario in scenarios_to_run:
        scenario_dir = ensemble_dir + scenario + '/'
        # runOnly_mam(
        #     scenario_dir,
        #     mam_build_location=mam_build_location,
        #     mam_input=mam_input,mam_output=mam_output,
        #     core_dir=core_dir)
        mode_mus = np.loadtxt(scenario_dir + 'mode_mus.txt')
        gmds = np.exp(mode_mus)
        
        mam_input = scenario_dir + 'mam_input.nl'
        mam_output = scenario_dir + 'mam_output.nc'
        run_partmc(
            scenario_dir,mam_input,mam_output,
            uniform_comp = uniform_comp,
            num_multiplier = num_multiplier,
            gmds = gmds,
            gsds = gsds,
            n_part = n_part, n_repeat = n_repeat,
            partmc_run_dir = partmc_run_dir,
            t_max=t_max, t_output=t_output)
        try:
            with open('state_run_' + ensemble_name, 'r') as file:
                curr_state_text = file.read()
                
            file.close()
            if len(curr_state_text) == 0:
                curr_state = [scenario]
            elif '\n' not in curr_state_text:
                curr_state = [curr_state_text,scenario]
            else:
                curr_state = curr_state_text.split('\n')
                print(curr_state)
                curr_state.append(scenario)
        except:
            curr_state = [scenario]
        
        with open('state_run_' + ensemble_name, 'w') as file:
            file.write('\n'.join(map(str, curr_state)))
        file.close()

def mam_ensemble_run(
        ensemble_dir,
        mam_build_location='/people/fier887/mam_refactor/build/standalone',
        mam_input='/people/fier887/mam_refactor/standalone/tests/smoke_test.nl',
        mam_output = '/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc',
        core_dir = '/people/fier887/mam_refactor/core/'):
    
    # ensemble_name = ensemble_dir[ensemble_dir[:-1].rfind('/')+1:-1]
    # try:
    #     # Try to recover current state
    #     with open ('state_mam_' + ensemble_name, 'r') as file:
    #         start = int(file.read())
    # except:
    #     # Otherwise bootstrap at 1,
    #     # i.e: starting from begining
    start = 0
    
    scenarios = get_scenarios(ensemble_dir)
    end = len(scenarios)
    
    for ii in range(start,end):
        scenario_dir = ensemble_dir + scenarios[ii] + '/'
        runOnly_mam(
            scenario_dir,
            mam_build_location=mam_build_location,
            mam_input=mam_input,
            mam_output=mam_output,
            core_dir=core_dir)
        print(scenario_dir)
        # sleep(1)
        # with open('state_mam_' + ensemble_name, 'w') as file:
        #     file.write(str(ii+1))

def createOnly_mam(
        save_dir,mam_settings,scenario_type,mode_mus,uniform_comp=False,
        mam_build_location='/people/fier887/mam_refactor/build/standalone',
        mam_input='/people/fier887/mam_refactor/standalone/tests/smoke_test.nl',
        mam_output = '/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc',
        core_dir = '/people/fier887/mam_refactor/core/'):
    
    mam_scenario.create(mam_input,mam_settings,scenario_type=scenario_type,uniform_comp=uniform_comp)
    mam_scenario.update_gmds(mode_mus,core_dir=core_dir)
    if os.path.exists(save_dir + 'mam_input.nl'):
        os.remove(save_dir + 'mam_input.nl')
    shutil.copy(mam_input, save_dir + 'mam_input.nl')
    
    if os.path.exists(save_dir + 'mam_output.nc'):
        os.remove(save_dir + 'mam_output.nc')
    shutil.copy(mam_output, save_dir + 'mam_output.nc')
    
    np.savetxt(save_dir + 'mode_mus.txt',mode_mus)
    
def runOnly_mam(
        save_dir,
        mam_build_location='/people/fier887/mam_refactor/build/standalone',
        mam_input='/people/fier887/mam_refactor/standalone/tests/smoke_test.nl',
        mam_output = '/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc',
        core_dir = '/people/fier887/mam_refactor/core/'):
    
    cwd = os.getcwd()
    if os.path.exists(mam_input):
        os.remove(mam_input)
    shutil.copy(save_dir + 'mam_input.nl', mam_input)
    
    os.chdir(mam_build_location)
    os.system('make -j')
    os.system('make test')
    os.chdir(cwd)
    
    if os.path.exists(save_dir + 'mam_output.nc'):
        os.remove(save_dir + 'mam_output.nc')
    shutil.copy(mam_output, save_dir + 'mam_output.nc')

def partmc_ensemble(
        ensemble_dir,
#        mu1_vals,mu2_vals,mu3_vals,mu4_vals,
        n_part = 2000, n_repeat = 5,
        t_max = None, t_output = None,
        num_multiplier = 0.773378248237437, uniform_comp=False,
        gsds = [1.6,1.8,1.8,1.6],
        partmc_run_dir = '/people/fier887/partmc-2.6.1/scenarios/7_box_eval/'):
    
    ensemble_name = ensemble_dir[ensemble_dir[:-1].rfind('/')+1:-1]
    scenarios = get_scenarios(ensemble_dir)
    try:
        # Try to recover current state
        with open ('state_partmc_' + ensemble_name, 'r') as file:
            start = int(file.read())
    except:
        # Otherwise bootstrap at 1,
        # i.e: starting from begining
        start = 0
    
    end = len(scenarios)
    
    for ii in range(start,end):
        scenario = scenarios[ii]
        scenario_dir = ensemble_dir + scenario + '/'
        
        mode_mus = np.loadtxt(scenario_dir + 'mode_mus.txt')
        gmds = np.exp(mode_mus)
        
        mam_input = scenario_dir + 'mam_input.nl'
        mam_output = scenario_dir + 'mam_output.nc'
        
        
        run_partmc(
            scenario_dir,mam_input,mam_output,
            uniform_comp = uniform_comp,
            num_multiplier = num_multiplier,
            gmds = gmds,
            gsds = gsds,
            n_part = n_part, n_repeat = n_repeat,
            partmc_run_dir = partmc_run_dir,
            t_max=t_max, t_output=t_output)
        
        sleep(1)
        
        with open('state_partmc_' + ensemble_name, 'w') as file:
            file.write(str(ii+1))
        
        print('ii:',ii,', scenario:',ensemble_dir + scenario)
        
def sectional_ensemble(
        ensemble_dir,
        n_bins = 10, d_min = 1e-10, d_max = 1e-5,
        t_max = None, t_output = None,
        num_multiplier = 0.773378248237437, uniform_comp=False,
        partmc_run_dir = '/global/homes/l/lfierce/partmc_mosaic/partmc/scenarios/partmc/'):
    scenarios = get_scenarios(ensemble_dir)
    for ii,scenario in enumerate(scenarios):
        save_dir = ensemble_dir + scenario + '/sectional_' +  str(int(n_bins)) + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        shutil.copy(ensemble_dir + scenario + '/mam_input.nl',save_dir)
        shutil.copy(ensemble_dir + scenario + '/mam_output.nc',save_dir)
#        num_multipliers = np.unique(np.loadtxt(save_dir + 'num_multiplier.txt'))
#        num_multiplier = np.unique(num_multipliers)[0]
        mam_input = save_dir + 'mam_input.nl'
        mam_output = save_dir + 'mam_output.nc'
        run_sectional(
            save_dir,mam_input,mam_output,
            uniform_comp = uniform_comp,            
            num_multiplier = num_multiplier,
            n_bins = n_bins, d_min = d_min, d_max = d_max,
            partmc_run_dir = partmc_run_dir,
            t_max=t_max, t_output=t_output)
        print('ii:',ii,', scenario:',ensemble_dir + scenario)
        
# def quam_ensemble(ensemble_dir):
#     scenarios = get_scenarios(ensemble_dir)
#     for ii,scenario in enumerate(scenarios):
#         save_dir = ensemble_dir + scenario + '/'
#         mam_input = save_dir + 'mam_input.nl'
#         run_quam(save_dir,mam_input=mam_input)
    
def run_mam(
        save_dir,mam_settings,scenario_type,mode_mus,uniform_comp=False,
        mam_build_location='/people/fier887/mam_refactor/build/standalone',
        mam_input='/people/fier887/mam_refactor/standalone/tests/smoke_test.nl',
        mam_output = '/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc',
        core_dir = '/people/fier887/mam_refactor/core/'):
    
    cwd = os.getcwd()
    mam_scenario.create(mam_input,mam_settings,scenario_type=scenario_type,uniform_comp=uniform_comp)
    mam_scenario.update_gmds(mode_mus,core_dir=core_dir)
    os.chdir(mam_build_location)
    os.system('make -j')
    os.system('make test')
    os.chdir(cwd)
    if os.path.exists(save_dir + 'mam_input.nl'):
        os.remove(save_dir + 'mam_input.nl')
    shutil.copy(mam_input, save_dir + 'mam_input.nl')
    
    if os.path.exists(save_dir + 'mam_output.nc'):
        os.remove(save_dir + 'mam_output.nc')
    shutil.copy(mam_output, save_dir + 'mam_output.nc')
    
    np.savetxt(save_dir + 'mode_mus.txt',mode_mus)
    
def run_partmc(
        save_dir,mam_input,mam_output,
        uniform_comp = False,
        num_multiplier = 0.773378248237437,
        gmds = [1.1e-7,2.6e-8,2e-6,5e-8],
        gsds = [1.6,1.8,1.8,1.6],
        n_part = 2000, n_repeat = 5,        
        t_max = None, t_output = None,
        partmc_run_dir = '/global/homes/l/lfierce/partmc_mosaic/partmc/scenarios/partmc/'):
    cwd = os.getcwd()
    create_partmc(
        mam_input = mam_input,
        mam_output = mam_output,
        num_multiplier = num_multiplier,
        gmds = gmds,
        gsds = gsds,
        uniform_comp = uniform_comp,
        n_part = n_part, n_repeat = n_repeat,
        partmc_run_dir = partmc_run_dir,
        t_max = t_max, t_output = t_output)
    os.chdir(partmc_run_dir)
    files = [onefile for onefile in os.listdir(partmc_run_dir + 'out') if onefile.endswith('.nc')]
    for file in files:
        os.remove(partmc_run_dir + 'out/' + file)
    
    
    os.system('./1_run.sh')
    os.chdir(cwd)
    files = [onefile for onefile in os.listdir(partmc_run_dir) if onefile.endswith('.dat') or onefile.endswith('.spec')]
    for file in files:
        shutil.copy(partmc_run_dir + file, save_dir)
    
    if os.path.exists(save_dir + 'out'):
        files = [onefile for onefile in os.listdir(save_dir + 'out')]
        for file in files:
            os.remove(save_dir + 'out/' + file)
        os.removedirs(save_dir + 'out')
    shutil.copytree(partmc_run_dir + 'out', save_dir + 'out')

def run_sectional(
        save_dir,mam_input,mam_output,
        num_multiplier = 0.773378248237437,
        n_bins=10,d_min=1e-10,d_max=1e-5,        
        t_max = None, t_output = None,
        uniform_comp = False,
        partmc_run_dir = '/global/homes/l/lfierce/partmc_mosaic/partmc/scenarios/partmc/'):
    cwd = os.getcwd()
    create_sectional(
        mam_input = mam_input,
        mam_output = mam_output,
        num_multiplier = num_multiplier,
        uniform_comp=uniform_comp,
        n_bins = n_bins,
        d_min = d_min, d_max = d_max,
        partmc_run_dir = partmc_run_dir,
        t_max = t_max, t_output = t_output)
    os.chdir(partmc_run_dir)
    files = [onefile for onefile in os.listdir(partmc_run_dir + 'out') if onefile.endswith('.nc')]
    for file in files:
        os.remove(partmc_run_dir + 'out/' + file)
    
    os.system('./1_run.sh')
    os.chdir(cwd)
    files = [onefile for onefile in os.listdir(partmc_run_dir) if onefile.endswith('.dat') or onefile.endswith('.spec')]
    for file in files:
        shutil.copy(partmc_run_dir + file, save_dir)
    
    if os.path.exists(save_dir + 'out'):
        files = [onefile for onefile in os.listdir(save_dir + 'out')]
        for file in files:
            os.remove(save_dir + 'out/' + file)
        os.removedirs(save_dir + 'out')
    shutil.copytree(partmc_run_dir + 'out', save_dir + 'out')


# def run_quam(
#         save_dir,scenario_type='4mode_noemit',
#         mam_input='/people/fier887/mam_refactor/standalone/tests/smoke_test.nl'):
    
#     qh2so4 = get_mam_input(
#         'qh2so4',
#         mam_input=mam_input)
    
#     h2so4_chem_prod_rate = get_mam_input(
#         'h2so4_chem_prod_rate',
#         mam_input=mam_input)
#     ht = get_mam_input('hgt',mam_input = mam_input) # m 
#     press = get_mam_input('press',mam_input = mam_input) # Pa
#     temp = get_mam_input('temp',mam_input = mam_input) # K
#     h2so4_emission_rate = h2so4_chem_prod_rate*ht*press/(R*temp) # mols/(m^2*s)
    
#     coag_on = get_mam_input(
#         'mdo_coag',
#         mam_input=mam_input)        
#     gasaerexch_on = get_mam_input(
#         'mdo_gasaerexch',
#         mam_input=mam_input)
    
#     gsds = [1.8,1.6,1.8,1.6]
#     gmds = [1.1e-7,2.6e-8,2e-6,5e-8]
#     nums = np.zeros(len(gmds))
#     for kk in range(0,4):
#         nums[kk] = get_mam_input(
#             'numc' + str(kk+1),
#             mam_input=mam_input)
#     modes = range(1,len(nums)+1)
#     mode_spec_fracs,spec_densities,spec_kappas,all_specs = get_quam_species(mam_input, modes = modes)
    
#     mam_nstep = get_mam_input(
#             'mam_nstep',
#             mam_input = mam_input)
#     mam_dt = get_mam_input(
#             'mam_dt',
#             mam_input = mam_input)
#     mam_output_intvl = get_mam_input(
#             'mam_output_intvl',
#             mam_input = mam_input)
    
#     t_max_val = mam_nstep*mam_dt
#     t_output_val = mam_output_intvl*mam_dt
#     rh = get_mam_input('RH_CLEA',mam_input = mam_input)
    
#     settings = {
#         'mode_Ns':nums,
#         'mode_gmds':gmds,
#         'mode_gsds':gsds,
#         'mode_spec_fracs':mode_spec_fracs,
#         'spec_names':all_specs,
#         'spec_densities':spec_densities,
#         'spec_kappas':spec_kappas,
#         'h2so4_0':qh2so4, # mixing ratio
#         'dh2so4_dt':h2so4_emission_rate, #mols/(m^2*s)
#         't_end':t_max_val,
#         'dt_output':t_output_val,
#         'cond_on':gasaerexch_on,
#         'coag_on':coag_on,
#         'rh':rh,
#         'temp':temp,
#         'press':press,
#         }
#     inputs = quam_scenario.create(settings,scenario_type=scenario_type)
#     run_dict = quam_scenario.run(inputs)
#     f = open(save_dir + 'quam_output.pkl','wb')
#     pickle.dump(run_dict,f)
    
# def get_quam_species(mam_input, modes = range(1,5)):
#     all_specs = ['so4', 'pom', 'soa', 'bc', 'dst', 'ncl']
#     mode_spec_fracs = []
#     for ii,mode in enumerate(modes):
#         mam_specnames = get_mode_species(mode,mam_input = mam_input)
#         spec_fracs = np.zeros(len(all_specs))    
#         for kk,spec in enumerate(mam_specnames):
#             idx, = np.where([onespec == spec for onespec in all_specs])
#             spec_fracs[idx[0]] = get_mam_input('mf' + spec + str(mode),mam_input = mam_input)
#         mode_spec_fracs.append(spec_fracs)
    
#     mode_spec_fracs = np.array(mode_spec_fracs).transpose()
#     spec_densities = np.zeros(len(all_specs))
#     spec_kappas = np.zeros(len(all_specs))
    
#     for kk,spec_name in enumerate(all_specs):
#         if spec_name == 'so4':
#             spec_densities[kk] = c.rho_so4
#             spec_kappas[kk] = c.kappa_so4
#         elif spec_name == 'pom':
#             spec_densities[kk] = c.rho_poa
#             spec_kappas[kk] = c.kappa_poa
#         elif spec_name == 'soa':
#             spec_densities[kk] = c.rho_soa
#             spec_kappas[kk] = c.kappa_soa     
#         elif spec_name == 'bc':
#             spec_densities[kk] = c.rho_bc
#             spec_kappas[kk] = c.kappa_bc
#         elif spec_name == 'dst':
#             spec_densities[kk] = c.rho_dust
#             spec_kappas[kk] = c.kappa_dust
#         elif spec_name == 'ncl':
#             spec_densities[kk] = c.rho_nacl
#             spec_kappas[kk] = c.kappa_nacl
    
#     return mode_spec_fracs,spec_densities,spec_kappas,all_specs
            

def h2so4_flux_to_production_rate(h2so4_flux,temp,press,height):
    # h2so4_flux in mol/(m^2 s) to h2so4_chem_prod_rate in mol/(m^3 s)
    
    h2so4_chem_prod_rate = h2so4_flux*R*temp/(height*press)
    
    return h2so4_chem_prod_rate

# todo: new version of this function that splits repeats to separate folders (within "scenario" folder)
def create_partmc(
        n_part = 10000,
        n_repeat = 10,
        t_max = None, t_output = None,
        uniform_comp = False,
        num_multiplier = 0.773378248237437,
        gmds = [1.1e-7,2.6e-8,2e-6,5e-8],
        gsds = [1.6,1.8,1.8,1.6],
        mam_input = '/people/fier887/mam_refactor/standalone/tests/smoke_test.nl',
        mam_output = '/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc',
        partmc_run_dir = '/global/homes/l/lfierce/partmc_mosaic/partmc/scenarios/partmc/'):
    # gas h2so4 mixing ratio (mol/mol-air)
    qh2so4 = get_mam_input(
        'qh2so4',
        mam_input=mam_input)
    # h2so4 production rate (ppb/s)
    h2so4_chem_prod_rate = get_mam_input(
        'h2so4_chem_prod_rate',
        mam_input=mam_input)
    
    ht = get_mam_input('hgt',mam_input = mam_input) # m 
    press_val = get_mam_input('press',mam_input = mam_input) # Pa
    temp_val = get_mam_input('temp',mam_input = mam_input) # K
    # h2so4 emission rate (#/m^2)
    # MW_h2so4 = 98.
    # MW_air = 29.
    h2so4_emission_rate = h2so4_chem_prod_rate*ht*press_val/(R*temp_val) #ht/0.0224 
    h2so4_ppb = qh2so4*1e9 # MW_h2so4/MW_air*
    
    coag_on = get_mam_input(
        'mdo_coag',
        mam_input=mam_input)        
    gasaerexch_on = get_mam_input(
        'mdo_gasaerexch',
        mam_input=mam_input)
    
    if gasaerexch_on == 1:
        do_mosaic = 'yes'
    elif gasaerexch_on == 0:
        do_mosaic = 'no'

    if coag_on == 1:
        do_coagulation = 'yes'
    elif coag_on == 0:
        do_coagulation = 'no'
    
    modes = range(1,5)
    #gsds = [1.8,1.6,1.8,1.6]
    #gmds = [1.1e-7,2.6e-8,2e-6,5e-8]
    nums = np.zeros(len(gmds))
    for kk in range(0,4):
        nums[kk] = get_mam_input(
            'numc' + str(kk+1),
            mam_input=mam_input)*num_multiplier
    
    height = {}
    height['time'] = 0.
    height['height'] = ht
    
    temp = {}
    temp['time'] = 0.
    temp['temp'] = temp_val
    
    press = {}
    press['time'] = 0.
    press['press'] = press_val
        
    # FIX THIS!
    aero_init = {'dist':{}}
    # mode_dat = {}
    for ii,mode in enumerate(modes):
        aero_init['dist'][mode] = {'diam_type':'geometric','mode_type':'log_normal'}
        aero_init['dist'][mode]['num_conc'] = nums[ii]
        aero_init['dist'][mode]['geom_mean_diam'] = gmds[ii]
        aero_init['dist'][mode]['log10_geom_std_dev'] = np.log10(gsds[ii])
        mode_specs = get_mode_species(
                mode,
                mam_input = mam_input)
        aero_init['dist'][mode]['comp'] = {}
        for spec in mode_specs:
            aero_init['dist'][mode]['comp'][spec] = get_mam_input(
                    'mf' + spec + str(mode),
                    mam_input = mam_input)
    mam_nstep = get_mam_input(
            'mam_nstep',
            mam_input = mam_input)
    mam_dt = get_mam_input(
            'mam_dt',
            mam_input = mam_input)
    mam_output_intvl = get_mam_input(
            'mam_output_intvl',
            mam_input = mam_input)
    
    if t_max == None:
        t_max_val = mam_nstep*mam_dt
    else:
        t_max_val = t_max
    
    if t_output == None:
        t_output_val = mam_output_intvl*mam_dt
    else:
        t_output_val = t_output
        
    
    spec = {
            'n_part':int(n_part),
            'n_repeat':int(n_repeat),
            't_max':int(t_max_val),
            'del_t':mam_dt,
            't_output':int(t_output_val),
            'do_coagulation':do_coagulation,
            'do_mosaic':do_mosaic,
            'rel_humidity':get_mam_input('RH_CLEA',mam_input = mam_input)}
    
    partmc_settings = {
            'aero_init':aero_init,
            'gas_init':{'H2SO4':h2so4_ppb},
            'gas_emit':{'time':0., 'rate':1.,'H2SO4':h2so4_emission_rate},
            'height':height,
            'temp':temp,
            'press':press,
            'spec':spec}
    
    partmc_scenario.create(partmc_run_dir, partmc_settings,uniform_comp=uniform_comp)
    
    
def create_sectional(
        n_bins = 10,
        d_min=1e-10,d_max=1e-5,
        uniform_comp = False,        
        t_max = None, t_output = None,
        num_multiplier = 0.773378248237437,
        mam_input = '/people/fier887/mam_refactor/standalone/tests/smoke_test.nl',
        mam_output = '/people/fier887/mam_refactor/build/standalone/tests/mam_output.nc',
        partmc_run_dir = '/global/homes/l/lfierce/partmc_mosaic/partmc/scenarios/partmc/'):
    # gas h2so4 mixing ratio (mol/mol-air)
    qh2so4 = get_mam_input(
        'qh2so4',
        mam_input=mam_input)
    # h2so4 production rate (ppb/s)
    h2so4_chem_prod_rate = get_mam_input(
        'h2so4_chem_prod_rate',
        mam_input=mam_input)
    
    ht = get_mam_input('hgt',mam_input = mam_input) # m 
    press_val = get_mam_input('press',mam_input = mam_input) # Pa
    temp_val = get_mam_input('temp',mam_input = mam_input) # K
    # h2so4 emission rate (#/m^2)
    # MW_h2so4 = 98.
    # MW_air = 29.
    h2so4_emission_rate = h2so4_chem_prod_rate*ht*press_val/(R*temp_val) #ht/0.0224 
    h2so4_ppb = qh2so4*1e9 # MW_h2so4/MW_air*
    
    coag_on = get_mam_input(
        'mdo_coag',
        mam_input=mam_input)        
    gasaerexch_on = get_mam_input(
        'mdo_gasaerexch',
        mam_input=mam_input)
    
    if gasaerexch_on == 1:
        do_mosaic = 'yes'
    elif gasaerexch_on == 0:
        do_mosaic = 'no'

    if coag_on == 1:
        do_coagulation = 'yes'
    elif coag_on == 0:
        do_coagulation = 'no'
    
    modes = range(1,5)
    gsds = [1.6,1.8,1.8,1.6] #[1.8,1.6,1.8,1.6]
    gmds = [1.1e-7,2.6e-8,2e-6,5e-8]
    nums = np.zeros(len(gmds))
    for kk in range(0,4):
        nums[kk] = get_mam_input(
            'numc' + str(kk+1),
            mam_input=mam_input)*num_multiplier
    
    height = {}
    height['time'] = 0.
    height['height'] = ht
    
    temp = {}
    temp['time'] = 0.
    temp['temp'] = temp_val
    
    press = {}
    press['time'] = 0.
    press['press'] = press_val
        
    # FIX THIS!
    aero_init = {'dist':{}}
    # mode_dat = {}
    for ii,mode in enumerate(modes):
        aero_init['dist'][mode] = {'diam_type':'geometric','mode_type':'log_normal'}
        aero_init['dist'][mode]['num_conc'] = nums[ii]
        aero_init['dist'][mode]['geom_mean_diam'] = gmds[ii]
        aero_init['dist'][mode]['log10_geom_std_dev'] = np.log10(gsds[ii])
        mode_specs = get_mode_species(
                mode,
                mam_input = mam_input)
        aero_init['dist'][mode]['comp'] = {}
        for spec in mode_specs:
            aero_init['dist'][mode]['comp'][spec] = get_mam_input(
                    'mf' + spec + str(mode),
                    mam_input = mam_input)
    mam_nstep = get_mam_input(
            'mam_nstep',
            mam_input = mam_input)
    mam_dt = get_mam_input(
            'mam_dt',
            mam_input = mam_input)
    mam_output_intvl = get_mam_input(
            'mam_output_intvl',
            mam_input = mam_input)
    
    if t_max == None:
        t_max_val = mam_nstep*mam_dt
    else:
        t_max_val = t_max
    
    if t_output == None:
        t_output_val = mam_output_intvl*mam_dt
    else:
        t_output_val = t_output
        
    
    spec = {
#            'n_part':int(n_part),
#            'n_repeat':int(n_repeat),
            'run_type':'sectional', 
            't_max':int(t_max_val),
            'del_t':mam_dt,
            't_output':int(t_output_val),
            'n_bin':n_bins,
            'd_min':d_min,
            'd_max':d_max,
            'do_coagulation':do_coagulation,
            'do_mosaic':do_mosaic,
            'rel_humidity':get_mam_input('RH_CLEA',mam_input = mam_input)}
    
    partmc_settings = {
            'aero_init':aero_init,
            'gas_init':{'H2SO4':h2so4_ppb},
            'gas_emit':{'time':0., 'rate':1.,'H2SO4':h2so4_emission_rate},
            'height':height,
            'temp':temp,
            'press':press,
            'spec':spec}
    
    partmc_scenario.create(partmc_run_dir, partmc_settings, uniform_comp=uniform_comp)
    
def get_mode_species(
        mode_num,
        mam_input='/people/fier887/mam_refactor/standalone/tests/smoke_test.nl'):
    f_input = open(mam_input,'r')
    input_lines = f_input.readlines()
    mode_specs = []
    for oneline in input_lines:        
        if oneline.startswith('mf'):
            idx_modenum = np.where([one_char == ' ' for one_char in oneline])[0][0] - 1
            if int(oneline[idx_modenum]) == mode_num:
                this_spec = oneline[2:idx_modenum]
                mode_specs.append(this_spec)
    return mode_specs
    
def get_partmc_input(
        varname, scenario_dir):
    if varname == 't_max':
        read_file = scenario_dir + 'urban_plume.spec'
    elif varname == 't_output':
        read_file = scenario_dir + 'urban_plume.spec'
    else:
        print('not yet defined for varname = ' + varname)
    
    f_input = open(read_file,'r')
    input_lines = f_input.readlines()    
    for oneline in input_lines:        
        if oneline.startswith(varname):
            idx1,=np.where([hi == ' ' for hi in oneline])
            if len(idx1)>1:
                vardat = float(''.join([oneline[ii] for ii in range(idx1[0]+1,idx1[1])]))
            else:
                vardat = float(''.join([oneline[ii] for ii in range(idx1[0],len(oneline))]))
    return vardat 

def get_mam_input(
        varname,
        mam_input='../mam_refactor-main/standalone/tests/smoke_test.nl'):
    f_input = open(mam_input,'r')
    input_lines = f_input.readlines()
    yep = 0
    for oneline in input_lines:        
#        print(oneline)
        if oneline.startswith(varname):
            yep += 1 
            idx1,=np.where([hi == '=' for hi in oneline])
            idx2,=np.where([hi == ',' for hi in oneline])
            vardat = float(''.join([oneline[ii] for ii in range(idx1[0]+1,idx2[0])]))
            # print(''.join([oneline[ii] for ii in range(idx1[0]+1,idx2[0])]))
            # print(vardat)
    if yep == 0:
        print(varname,'is not a MAM input parameter')
    elif yep > 1:
        print('more than one line in ', mam_input, 'starts with', varname)
    return vardat

def get_mam_output(
        varname,
        mam_output = '../mam_refactor-main/build/standalone/tests/mam_output.nc',
        initial_only = False):
    f_output = netCDF4.Dataset(mam_output)
    
    if varname not in f_output.variables.keys():
        print(varname,'is not a MAM output parameter')
    else:
        if initial_only:
            return f_output.variables[varname][:,0]
        else:
            return f_output.variables[varname]  
        


def get_Ntot(mode_Ns):
    return sum(mode_Ns)

def get_moment(p,mode_Ns,mode_mus,mode_sigs):
    return sum(mode_Ns*np.exp(p*mode_mus + 0.5*p**2*mode_sigs**2))

def get_avg_flux(mode_Ns,mode_mus,mode_sigs,h2so40,dh2so4_dt_emit,t_max):
    dh2so4_dt = h2so40/t_max + dh2so4_dt_emit
    Atot = get_Atot(mode_Ns,mode_mus,mode_sigs)
    vol_flux = (dh2so4_dt/c.rho_h2so4)/Atot
    return vol_flux

def get_dh2so4dt_mol(vol_flux,mode_Ns,mode_mus,mode_sigs,p=101325.,temp = 273.):
    A_tot = sum(get_Atot(mode_Ns,mode_mus,mode_sigs)) # m^2 aerosol per m^3 air
#    vol_flux = # dvolh2so4_dt/A_tot
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

def sample_inputs(
        N_samples,N_modes=4,
        N_all_varnames = ['tot_num','frac_aitk','frac_coarse','frac_fresh'],#,'h2so4_prod_rate','qh2so4'],
        N_all_dists = ['loguniform','uniform','loguniform','uniform'],#,'loguniform','loguniform'],
        N_all_param1 = [1e8,0.5,0.001,0.],#,5e-16,1e-12],
        N_all_param2 = [1e12,0.9,0.1,1.,5e-13],
        dgm_all_varnames = ['dgm_aik','dgm_accum','dgm_coarse','dgm_fresh'],
        dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
        dgm_all_param1 = [0.0535e-6, 0.0087e-6, 1.000e-6, 0.010e-6],
        dgm_all_param2 = [0.4400e-6, 0.0520e-6, 4.000e-6, 0.100e-6],
        mode_sigs = np.log([1.8,1.6,1.6,1.8]),
        flux_dist = 'loguniform',#'lognormal',#'loguniform',
        flux_param1 = 1e-2*1e-9,#0.1*1e-9,#1e-2*1e-9,
        flux_param2 = 1e1*1e-9,#1.5,#
        t_max=60.,min_frac_from_prod=0.,
        p=101325.,temp = 273.):
    
    lhd = lhs(len(N_all_varnames) + len(dgm_all_varnames), samples=N_samples)
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
        print(avg_vol_flux[ii])
        avg_vol_flux[ii] = get_sample(np.random.uniform(),flux_dist,flux_param1,flux_param2)
        dh2so4_dt_tot = get_dh2so4dt_mol(avg_vol_flux[ii],mode_Ns[ii,:],mode_mus,mode_sigs)
#        dh2so4_dt_mass = get_dh2so4_dt(avg_vol_flux[ii],mode_Ns[ii,:],mode_mus,mode_sigs)
#        dh2so4_dt_tot = h2so4_conc_to_molRatio(dh2so4_dt_mass,p=p,temp=temp)
        
        frac_prod = min_frac_from_prod + (1.-min_frac_from_prod)*np.random.uniform() # randomly vary the amount of dh2so4_dt from production vs initial
        h2so4_chem_prod_rate_vals[ii] = frac_prod*dh2so4_dt_tot 
        qh2so4_vals[ii] = (1.-frac_prod)*dh2so4_dt_tot*t_max
        ii +=1
    numc1_vals = mode_Ns[:,0]
    numc2_vals = mode_Ns[:,1]
    numc3_vals = mode_Ns[:,2]
    numc4_vals = mode_Ns[:,3]
    
    mu1_vals = mode_mus[:,0]
    mu2_vals = mode_mus[:,1]
    mu3_vals = mode_mus[:,2]
    mu4_vals = mode_mus[:,3]    
    
    return numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux
            # numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals, temp_vals

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]    

def sample_inputs_E3SMranges(
        N_samples,N_modes=4,
        N_all_varnames = ['tot_num','frac_accum','nonaccum_frac_coarse','ofaccum_frac_fresh'],
        N_all_dists = ['loguniform','uniform','loguniform','loguniform'],
        N_all_param1 = [3e7,0.1e-8,1e-13],
        N_all_param2 = [2e12,1.,1.,1.],
        dgm_all_varnames = ['dgm_accum','dgm_aik','dgm_coarse','dgm_fresh'],
        dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
        dgm_all_param1 = [0.5e-7, 0.5e-8, 1.000e-6, 1e-8],
        dgm_all_param2 = [1.1e-7, 3e-8, 2.000e-6, 6e-8],
        # N_all_varnames = ['tot_num','frac_aitk','frac_coarse','frac_fresh'],#,'h2so4_prod_rate','qh2so4'],
        # N_all_dists = ['loguniform','uniform','loguniform','uniform'],#,'loguniform','loguniform'],
        # N_all_param1 = [1e8,0.5,0.001,0.],#,5e-16,1e-12],
        # N_all_param2 = [1e12,0.9,0.1,1.,5e-13],
        # dgm_all_varnames = ['dgm_aik','dgm_accum','dgm_coarse','dgm_fresh'],
        # dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
        # dgm_all_param1 = [0.0535e-6, 0.0087e-6, 1.000e-6, 0.010e-6],
        # dgm_all_param2 = [0.4400e-6, 0.0520e-6, 4.000e-6, 0.100e-6],
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
        mode_sigs = np.log([1.8,1.6,1.6,1.8]),
        flux_dist = 'loguniform',#'lognormal',#'loguniform',
        flux_param1 = 1e-2*1e-9,#0.1*1e-9,#1e-2*1e-9,
        flux_param2 = 1e1*1e-9,#1.5,#
        rh_dist='uniform',
        rh_param1=0.,
        rh_param2=1.,
        t_max=60.,min_frac_from_prod=0.,
        p=101325.,temp = 273.):
    
    # if one of the values reads = 'E3SM', then use the E3SM value
    #   E3SM values = ranges for uniform, params for normal
    #   the distribution for each param is always asigned
    
    lhd = lhs(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames) + len(mode4_varnames) + 1, samples=N_samples)
    X = np.zeros([N_samples,len(N_all_varnames) + len(dgm_all_varnames)]) #np.zeros(lhd.shape)
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
        X[:,vv+4] = get_sample(lhd[:,vv+4],dist,param1,param2)
    
    if N_all_varnames[1] == 'frac_aitk':
        tot_num = X[:,0]
        frac_aitk = X[:,1]
        frac_coarse = X[:,2]
        frac_fresh = X[:,3]
        frac_accum = 1. - X[:,2] - X[:,1]
    else:
        N_all_varnames = ['tot_num','frac_accum','nonaccum_frac_coarse','ofaccum_frac_fresh']
        tot_num = X[:,0]
        frac_accum = X[:,1]
        frac_coarse = (1. - frac_accum)*X[:,2]
        frac_aitk = (1. - frac_accum)*(1. - X[:,2])
        frac_fresh = X[:,3]
    print(frac_accum,frac_coarse,frac_aitk,frac_fresh)
    
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
#        dh2so4_dt_mass = get_dh2so4_dt(avg_vol_flux[ii],mode_Ns[ii,:],mode_mus,mode_sigs)
#        dh2so4_dt_tot = h2so4_conc_to_molRatio(dh2so4_dt_mass,p=p,temp=temp)
        frac_prod = min_frac_from_prod + (1.-min_frac_from_prod)*np.random.uniform() # randomly vary the amount of dh2so4_dt from production vs initial
        h2so4_chem_prod_rate_vals[ii] = frac_prod*dh2so4_dt_tot 
        qh2so4_vals[ii] = (1.-frac_prod)*dh2so4_dt_tot*t_max
        ii +=1
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
    
    rh_vals = get_sample(lhd[:,-1],rh_dist,rh_param1,rh_param2)
    
    mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals = get_mode_comp('mode1',mode1_varnames,mode1_dists,mode1_param1,mode1_param2,lhd_mode1)
    mfso42_vals, mfsoa2_vals, mfncl2_vals = get_mode_comp('mode2',mode2_varnames,mode2_dists,mode2_param1,mode2_param2,lhd_mode2)
    mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals = get_mode_comp('mode3',mode3_varnames,mode3_dists,mode3_param1,mode3_param2,lhd_mode3)
    mfbc4_vals, mfpom4_vals = get_mode_comp('mode4',mode4_varnames,mode4_dists,mode4_param1,mode4_param2,lhd_mode4)
    
    return numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals, mfso42_vals, mfsoa2_vals, mfncl2_vals, mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals, mfbc4_vals, mfpom4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals


def sample_inputs_composition(
        N_samples,N_modes=4,
        N_all_varnames = ['tot_num','frac_aitk','frac_coarse','frac_fresh'],#,'h2so4_prod_rate','qh2so4'],
        N_all_dists = ['loguniform','uniform','loguniform','uniform'],#,'loguniform','loguniform'],
        N_all_param1 = [1e8,0.5,0.001,0.],#,5e-16,1e-12],
        N_all_param2 = [1e12,0.9,0.1,1.,5e-13],
        dgm_all_varnames = ['dgm_aik','dgm_accum','dgm_coarse','dgm_fresh'],
        dgm_all_dists = ['loguniform','loguniform','loguniform','loguniform'],
        dgm_all_param1 = [0.0535e-6, 0.0087e-6, 1.000e-6, 0.010e-6],
        dgm_all_param2 = [0.4400e-6, 0.0520e-6, 4.000e-6, 0.100e-6],
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
        mode_sigs = np.log([1.8,1.6,1.6,1.8]),
        flux_dist = 'loguniform',#'lognormal',#'loguniform',
        flux_param1 = 1e-2*1e-9,#0.1*1e-9,#1e-2*1e-9,
        flux_param2 = 1e1*1e-9,#1.5,#
        rh_dist='uniform',
        rh_param1=0.,
        rh_param2=1.,
        t_max=60.,min_frac_from_prod=0.,
        p=101325.,temp = 273.):
    
    lhd = lhs(len(N_all_varnames) + len(dgm_all_varnames) + len(mode1_varnames) + len(mode2_varnames) + len(mode3_varnames) + len(mode4_varnames) + 1, samples=N_samples)
    X = np.zeros([N_samples,len(N_all_varnames) + len(dgm_all_varnames)]) #np.zeros(lhd.shape)
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
        X[:,vv+4] = get_sample(lhd[:,vv+4],dist,param1,param2)
    
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
#        dh2so4_dt_mass = get_dh2so4_dt(avg_vol_flux[ii],mode_Ns[ii,:],mode_mus,mode_sigs)
#        dh2so4_dt_tot = h2so4_conc_to_molRatio(dh2so4_dt_mass,p=p,temp=temp)
        frac_prod = min_frac_from_prod + (1.-min_frac_from_prod)*np.random.uniform() # randomly vary the amount of dh2so4_dt from production vs initial
        h2so4_chem_prod_rate_vals[ii] = frac_prod*dh2so4_dt_tot 
        qh2so4_vals[ii] = (1.-frac_prod)*dh2so4_dt_tot*t_max
        ii +=1
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
    
    rh_vals = get_sample(lhd[:,-1],rh_dist,rh_param1,rh_param2)
    
    mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals = get_mode_comp('mode1',mode1_varnames,mode1_dists,mode1_param1,mode1_param2,lhd_mode1)
    mfso42_vals, mfsoa2_vals, mfncl2_vals = get_mode_comp('mode2',mode2_varnames,mode2_dists,mode2_param1,mode2_param2,lhd_mode2)
    mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals = get_mode_comp('mode3',mode3_varnames,mode3_dists,mode3_param1,mode3_param2,lhd_mode3)
    mfbc4_vals, mfpom4_vals = get_mode_comp('mode4',mode4_varnames,mode4_dists,mode4_param1,mode4_param2,lhd_mode4)
    
    return numc1_vals, numc2_vals, numc3_vals, numc4_vals, mu1_vals, mu2_vals, mu3_vals, mu4_vals, mfso41_vals, mfpom1_vals, mfsoa1_vals, mfbc1_vals, mfdst1_vals, mfncl1_vals, mfso42_vals, mfsoa2_vals, mfncl2_vals, mfso43_vals, mfpom3_vals, mfsoa3_vals, mfbc3_vals, mfdst3_vals, mfncl3_vals, mfbc4_vals, mfpom4_vals, h2so4_chem_prod_rate_vals, qh2so4_vals, avg_vol_flux, rh_vals


def get_mode_comp(mode_name,mode_varnames,mode_dists,mode_param1,mode_param2,lhd):
    
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
        frac_soa = (1.-frac_hygroscopic)
        frac_ncl = frac_hygroscopic*frac_hygroscopic_thats_ncl
        
        return frac_so4, frac_soa, frac_ncl

    elif mode_name == 'mode4':
        frac_bc = X_comp[:,0]
        frac_pom = 1. - frac_bc
        
        return frac_bc, frac_pom
    
     
def get_scenarios(ensemble_dir):
    scenarios = [file for file in os.listdir(ensemble_dir) if not file.startswith('.') and not file.startswith('process') and not file.endswith('.pkl')]
    scenario_nums = np.sort([int(scenario) for scenario in scenarios])
    scenarios = [str(num).zfill(6) for num in scenario_nums]
    
    return scenarios
