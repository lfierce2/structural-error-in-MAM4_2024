#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to process data from ensembles

@author: Laura Fierce
"""

import numpy as np
import helper
import os
import pickle


store_flux = True
store_ccn = True

s_env_vals = np.logspace(-3,2,151)

ensemble_prefix = 'ensemble_47_big'

run_time_str = '02:00:00'
project = 'particula'
labs = ['a','b','c']
processes = ['both','cond','coag']

if os.path.exists('/people/fier887/'):
    ensemble_over_dir = '/pic/projects/sooty2/fierce/box_simulations3/'
elif os.path.exists('/global/cscratch1/sd/lfierce'):
    ensemble_over_dir = '/global/cscratch1/sd/lfierce/box_simulations3/'
elif os.path.exists('/pscratch/sd/l/lfierce/'):
    ensemble_over_dir = '/pscratch/sd/l/lfierce/box_simulations3/'

lnDs = np.log(np.logspace(-10,-5,100))
gsds = [1.8,1.6,1.8,1.6]
for (lab,process) in zip(labs,processes):
    ensemble_name = ensemble_prefix + '_' + lab + '_' + process
    
    ensemble_dir = ensemble_over_dir + ensemble_name + '/scenarios/'
    ensemble_settings = pickle.load(open(ensemble_over_dir + ensemble_name + '/ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    t_max = ensemble_settings[0]['spec']['t_max']
    
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    
    
    Ntimes = int(t_max/dtime + 1)
    tt_save_vals = np.arange(0,Ntimes)
    
    all_scenarios = helper.get_scenarios(ensemble_dir)
    # all_scenarios = process_readVariables.get_scenarios(ensemble_dir)
    
    for tt_save in tt_save_vals:
        helper.store_data(
            ensemble_dir, all_scenarios, tt_save, n_repeat, dtime,
            run_time_str=run_time_str, project=project, gsds=gsds,
            lnDs=lnDs, s_env_vals=s_env_vals,
            store_ccn=store_ccn, store_flux=store_flux)
        
        # process_readVariables.store_data(
        #     ensemble_dir, all_scenarios, tt_save, n_repeat, dtime,
        #     run_time_str=run_time_str, project=project, 
        #     num_multiplier=num_multiplier, gsds=gsds,
        #     lnDs=lnDs, s_env_vals=s_env_vals,
        #     store_ccn=store_ccn, store_flux=store_flux)
