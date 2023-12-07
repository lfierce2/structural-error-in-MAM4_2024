#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used to analyze PartMC-MOSAIC and MAM4 box model output. 

@author: Laura Fierce

"""

import numpy as np
import random
from netCDF4 import Dataset
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import process_readVariables
import pickle
import os

def copy_stuff(
        ensemble_prefix,
        labs = ['a','b','c'],
        processes = ['both','cond','coag'],
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        remote_machine='constance'):
    
    copy_these = ['processed_output','*.pkl']
        
    
    for (lab,process) in zip(labs,processes):
        ensemble_name = ensemble_prefix + '_' + lab + '_' + process
        copy_to_here = ensemble_over_dir + ensemble_name + '/'
        if remote_machine == 'constance':
            copy_from_here = '/pic/projects/sooty2/fierce/box_simulations3/' + ensemble_name + '/'
        else:
            print('add a new remote machine!')
        for copy_this in copy_these:
            os.system(
                'rsync -r ' + copy_from_here + copy_this + ' ' + copy_to_here)
    
# =============================================================================
# identify regimes
# =============================================================================

def get_all_scenario_nums(ensemble_dir):
    try:
        one_KL = np.loadtxt(ensemble_dir + 'processed_output/tt' + str(0).zfill(4) + '/KL_repeats.txt')
    except:
        one_KL = np.loadtxt(ensemble_dir + 'processed_output/tt' + str(1).zfill(4) + '/KL_repeats.txt')
    
    N_scenarios = one_KL.shape[0]
    all_scenarios = np.arange(N_scenarios)
    return all_scenarios
    
def scenarios_in_regimes(ensemble_dir,timesteps,regime_type='process_extremes',avg_scale='lin'):
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    
    all_scenarios = get_all_scenario_nums(ensemble_dir)
    if regime_type == 'process_extremes':
        cond_bounds__nm_per_h = [
            [-np.inf,0.05],
            [-np.inf,0.05],
            [1.,np.inf],
            [1.,np.inf]
            ]
        coag_bounds__cm3 = [
            [-np.inf,1e3],
            [1e4,np.inf],
            [-np.inf,1e3],
            [1e4,np.inf],
            ]
        regime_names = [
            ['slowCond_slowCoag'],
            ['slowCond_fastCoag'],
            ['fastCond_slowCoag'],
            ['fastCond_fastCoag']]
    elif regime_type == 'regions':
        cond_bounds__nm_per_h = [
            [-np.inf,0.05],
            [0.05,1.],
            [1.,np.inf]]
        coag_bounds__cm3 = [
            [-np.inf,1e3],
            [1e3,1e4],
            [1e4,np.inf]]
        regime_names = [
            ['remote'],
            ['rural'],
            ['urban']]
        # cond_bounds__nm_per_h = [
        #     [-np.inf,0.05],
        #     [0.05,0.5],
        #     [0.5,np.inf]]
        # coag_bounds__cm3 = [
        #     [-np.inf,1e3],
        #     [1e3,1e4],
        #     [1e4,np.inf]]
        # regime_names = [
        #     ['remote'],
        #     ['rural'],
        #     ['urban']]
    elif regime_type == '2extremes':
        cond_bounds__nm_per_h = [
            [-np.inf,0.1],
            [1.,np.inf]]
        coag_bounds__cm3 = [
            [-np.inf,1e4],
            [1e4,np.inf]]
        regime_names = [
            ['slow'],
            ['fast']]
    
    elif regime_type == '3extremes':
        cond_bounds__nm_per_h = [
            [-np.inf,0.05],
            [-np.inf,0.05],
            [0.5,np.inf]]
        coag_bounds__cm3 = [
            [-np.inf,1e3],
            [1e4,np.inf],
            [-np.inf,1e3]]
        regime_names = [
            ['slow'],
            ['fast']]
    elif regime_type == 'all_together':
        cond_bounds__nm_per_h = [
            [-np.inf,np.inf]]
        coag_bounds__cm3 = [
            [-np.inf,np.inf]]
        regime_names = [
            ['all']]
    elif regime_type == 'condensation_groups':
        # cond_bounds__nm_per_h = [
        #     [-np.inf,0.04],
        #     [0.004,0.02],
        #     [0.02,0.1],
        #     [0.1,0.5],
        #     [0.5,np.inf]]
        # coag_bounds__cm3 = [
        #     [-np.inf,np.inf],
        #     [-np.inf,np.inf],
        #     [-np.inf,np.inf],
        #     [-np.inf,np.inf],
            # [-np.inf,np.inf]]
        cond_bounds__nm_per_h = [
            [-np.inf,0.005],
            [0.005,0.1],
            [0.1,np.inf]]
        coag_bounds__cm3 = [
            [-np.inf,np.inf],
            [-np.inf,np.inf],
            [-np.inf,np.inf]]
        regime_names = [str(vals) for vals in cond_bounds__nm_per_h]
        
    sa_flux_all = np.zeros([len(all_scenarios),len(timesteps)])
    num_conc_all = np.zeros([len(all_scenarios),len(timesteps)])
    for tt,timestep in enumerate(timesteps):
        tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        time = dtime*timestep
        sa_flux_all[:,tt] = read_boxmodel_data('sa_flux',tt_dir,time)
        # sa_flux_all[:,tt] = read_boxmodel_data('sa_flux_mam',tt_dir,time)
        num_conc_all[:,tt] = read_boxmodel_data('N_tot',tt_dir,time)    
    
    if avg_scale == 'lin':
        sa_flux = np.mean(3600.*1e9*sa_flux_all,axis=1)
        num_conc = np.mean(1e-6*num_conc_all,axis=1)
    elif avg_scale == 'log':
        sa_flux = 10.**np.mean(np.log10(3600.*1e9*sa_flux_all),axis=1)
        num_conc = 10.**np.mean(np.log10(1e-6*num_conc_all),axis=1)
    
    if all(np.isnan(sa_flux)):
        if avg_scale == 'lin':
            sa_flux = np.mean(3600.*1e9*sa_flux_all[:,1:],axis=1)
            num_conc = np.mean(1e-6*num_conc_all[:,1:],axis=1)
        elif avg_scale == 'log':
            sa_flux = 10.**np.mean(np.log10(3600.*1e9*sa_flux_all[:,1:]),axis=1)
            num_conc = 10.**np.mean(np.log10(1e-6*num_conc_all[:,1:]),axis=1)
    
    scenario_groups = ()
    for qq,(cond_bound,coag_bound) in enumerate(zip(cond_bounds__nm_per_h,coag_bounds__cm3)):
        idx, = np.where([
            one_sa_flux>cond_bound[0] and one_sa_flux<cond_bound[1] and one_num_conc>coag_bound[0] and one_num_conc<coag_bound[1] for (one_sa_flux,one_num_conc) in zip(sa_flux,num_conc)])
        scenario_groups += ([all_scenarios[ii] for ii in idx],)
    
    return scenario_groups,regime_names

def get_group_vars(
        varname,ensemble_dir,scenario_groups,timestep,
        recompute_KL = False, thresh=0.1,
        mode_sigs = np.log([1.8,1.6,1.6,1.8])):
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    # n_repeat = ensemble_settings[0]['spec']['n_repeat']
    n_repeat = 5
    time = dtime*timestep
    tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
    
    all_vardat = read_boxmodel_data(varname,tt_dir,time,mode_sigs = mode_sigs,recompute_KL=recompute_KL,n_repeat=n_repeat,thresh=thresh)
    #all_scenarios = process_readVariables.get_scenarios(ensemble_dir + 'scenarios/')
    all_scenarios = get_all_scenario_nums(ensemble_dir)
    
    vardat_groups = ()
    for one_scenario_group in scenario_groups:
        vardat_onegroup = np.zeros(len(one_scenario_group))
        for qq,one_scenario in enumerate(one_scenario_group):
            idx, = np.where([a_scenario == one_scenario for a_scenario in all_scenarios])
            vardat_onegroup[qq] = all_vardat[idx[0]]
        vardat_groups += (vardat_onegroup,)
    
    return vardat_groups

# =============================================================================
# main pieces (dependencies below) todo: generally adopt this structure? better comments later
# =============================================================================

def train_surrogate_model(X_train,y_train,regressor='mlp',solver='adam',max_iter=200,hidden_layer_sizes=(100,100,)):
    if regressor == 'mlp':
        regr = MLPRegressor(solver=solver,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes).fit(X_train, y_train)
    elif regressor == 'gp':
        regr = GaussianProcessRegressor().fit(X_train, y_train)
    elif regressor == 'gbr':
        regr = GradientBoostingRegressor(max_depth=10).fit(X_train,y_train)
        # regr = GradientBoostingRegressor(max_depth=10,learning_rate=0.5,n_estimators=20).fit(X_train,y_train)
    else:
        print('not yet coded for regressor = ' + regressor)
    
    return regr

def train_gbr_model(
        X_train,y_train,
        learning_rate0 = 0.1,
        n_estimators0 = 100,
        max_depth0 = 3,
        min_samples_split0 = 2,
        min_samples_leaf0 = 1,
        max_features0 = None):

    regr = GradientBoostingRegressor(
        learning_rate=learning_rate0,
        n_estimators=n_estimators0,
        max_depth=max_depth0,
        min_samples_split=min_samples_split0,
        min_samples_leaf=min_samples_leaf0,
        max_features=max_features0).fit(X_train,y_train)
    
    return regr


def test_surrogate_model(regr,X_test,y_test,error_metric='R2'):
    if error_metric == 'R2':
        return np.corrcoef(y_test,regr.predict(X_test))**2
    else:
        print('not yet coded for error_metric = ' + error_metric)

# def apply_surrogate_model_to_e3sm():
#     pass


# =============================================================================
# get testing and training data from box models
# =============================================================================
def get_Xy_all(Xvarnames,yvarname,ensemble_dir,timesteps,dtime=600.,thresh=0.1):
    y = np.array([])
    for tt,timestep in enumerate(timesteps):
        time = timestep*dtime
        tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        one_X = []
        for xx,varname in enumerate(Xvarnames):
            vardat = read_boxmodel_data(varname,tt_dir,time,thresh)
            one_X.append(vardat)
        if tt == 0:
            X = np.array(one_X).transpose()
        else:
            X = np.vstack([X,np.array(one_X).transpose()])
        
        one_y = read_boxmodel_data(yvarname,tt_dir,time,thresh)
        y = np.hstack([y,one_y])
    return X,y

def get_Xy_kfold(Xvarnames,yvarname,ensemble_dir,timesteps,n_splits=None):
    X,y = get_Xy_all(Xvarnames,yvarname,ensemble_dir,timesteps)
    folds = kfold_split(X,y,n_splits=n_splits)
    
    Xy_folds = []
    for fold in folds:
        idx_train = fold[0]
        idx_test = fold[1]
        
        X_train = X[idx_train,:]
        y_train = y[idx_train]
        
        X_test = X[idx_test,:]
        y_test = y[idx_test]
        
        Xy_folds.append([X_train,y_train,X_test,y_test])
        
    return Xy_folds

def get_Xy_process(Xvarnames,yvarname,ensemble_dir,timesteps,dtime=600.,frac_testing=0.1,frac_validation=0.,return_validation=False):
    
    X,y = get_Xy_all(Xvarnames,yvarname,ensemble_dir,timesteps,dtime=dtime)
    if return_validation:
        X_train,X_test,X_validation,y_test,y_train,y_validation = test_train_validation_split(
            X,y,frac_testing=frac_testing,frac_validation=frac_validation)
        return X_train,X_test,X_validation,y_test,y_train,y_validation
    else:
        X_train,X_test,y_test,y_train = test_train_split(
            X,y,frac_testing=frac_testing)
        return X_train,X_test,y_test,y_train

def read_boxmodel_data(
        varname,tt_dir,time,
        lnDs = np.log(np.logspace(-10,-5,100)), mode_sigs = np.log([1.8,1.6,1.6,1.8]),recompute_KL = False, n_repeat=1, thresh=0.1):
    try: col_idx = int(varname[-1])-1
    except: pass
    if varname.startswith('err_frac_ccn'):
        s_env_vals = np.loadtxt(tt_dir + 's_env_vals.txt')
        frac_ccn_mam4 = np.loadtxt(tt_dir + 'frac_ccn_mam4.txt')
        frac_ccn_partmc_mean = np.loadtxt(tt_dir + 'frac_ccn_partmc_mean.txt')
        Nscenarios = frac_ccn_mam4.shape[0]
        vardat = np.zeros(Nscenarios)
        for ii in range(Nscenarios):
            vardat[ii] = np.interp(thresh, s_env_vals, frac_ccn_mam4[ii,:]-frac_ccn_partmc_mean[ii,:])
    elif varname.startswith('s_onset_ratio'):
        s_env_vals = np.loadtxt(tt_dir + 's_env_vals.txt')
        frac_ccn_mam4 = np.loadtxt(tt_dir + 'frac_ccn_mam4.txt')
        frac_ccn_partmc_mean = np.loadtxt(tt_dir + 'frac_ccn_partmc_mean.txt')
        Nscenarios = frac_ccn_mam4.shape[0]
        vardat = np.zeros(Nscenarios)
        for ii in range(Nscenarios):
            s_onset_mam4 = np.interp(thresh, frac_ccn_mam4[ii,:], s_env_vals)
            s_onset_partmc = np.interp(thresh, frac_ccn_partmc_mean[ii,:], s_env_vals)
            vardat[ii] = s_onset_mam4/s_onset_partmc
    elif varname.startswith('s_onset_mam4'):
        s_env_vals = np.loadtxt(tt_dir + 's_env_vals.txt')
        frac_ccn_mam4 = np.loadtxt(tt_dir + 'frac_ccn_mam4.txt')
        Nscenarios = frac_ccn_mam4.shape[0]
        vardat = np.zeros(Nscenarios)
        for ii in range(Nscenarios):
            vardat[ii] = np.interp(thresh, frac_ccn_mam4[ii,:], s_env_vals)
    elif varname.startswith('s_onset_partmc'):
        s_env_vals = np.loadtxt(tt_dir + 's_env_vals.txt')
        frac_ccn_partmc_mean = np.loadtxt(tt_dir + 'frac_ccn_partmc_mean.txt')
        Nscenarios = frac_ccn_partmc_mean.shape[0]
        vardat = np.zeros(Nscenarios)
        for ii in range(Nscenarios):
            vardat[ii] = np.interp(thresh, frac_ccn_partmc_mean[ii,:], s_env_vals)
    elif varname.startswith('s_onset_ratio_uniformComp'):
        s_env_vals = np.loadtxt(tt_dir + 's_env_vals.txt')
        frac_ccn_mam4 = np.loadtxt(tt_dir + 'frac_ccn_mam4_uniformComp.txt')
        frac_ccn_partmc_mean = np.loadtxt(tt_dir + 'frac_ccn_partmc_mean_uniformComp.txt')
        Nscenarios = frac_ccn_mam4.shape[0]
        vardat = np.zeros(Nscenarios)
        for ii in range(Nscenarios):
            s_onset_mam4 = np.interp(thresh, frac_ccn_mam4[ii,:], s_env_vals)
            s_onset_partmc = np.interp(thresh, frac_ccn_partmc_mean[ii,:], s_env_vals)
            vardat[ii] = s_onset_mam4/s_onset_partmc
    elif varname.startswith('s_onset_mam4_uniformComp'):
        s_env_vals = np.loadtxt(tt_dir + 's_env_vals.txt')
        frac_ccn_mam4 = np.loadtxt(tt_dir + 'frac_ccn_mam4_uniformComp.txt')
        Nscenarios = frac_ccn_mam4.shape[0]
        vardat = np.zeros(Nscenarios)
        for ii in range(Nscenarios):
            vardat[ii] = np.interp(thresh, frac_ccn_mam4[ii,:], s_env_vals)
    elif varname.startswith('s_onset_partmc_uniformComp'):
        s_env_vals = np.loadtxt(tt_dir + 's_env_vals.txt')
        frac_ccn_partmc_mean = np.loadtxt(tt_dir + 'frac_ccn_partmc_mean_uniformComp.txt')
        Nscenarios = frac_ccn_partmc_mean.shape[0]
        vardat = np.zeros(Nscenarios)
        for ii in range(Nscenarios):
            vardat[ii] = np.interp(thresh, frac_ccn_partmc_mean[ii,:], s_env_vals)
    elif varname.startswith('D50_ratio'):
        dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
        dNdlnD_partmc_allrepeats = np.zeros(dNdlnD_mam4.shape + (n_repeat,))
        for jj,repeat in enumerate(range(1,n_repeat+1)):
            dNdlnD_partmc_allrepeats[:,:,jj] = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(repeat).zfill(4) + '.txt')
        
        Nscenarios = dNdlnD_partmc_allrepeats.shape[0]
        vardat = np.zeros(Nscenarios)
        for ii in range(Nscenarios):
            D50_partmc = np.zeros(n_repeat)
            D50_mam4 = np.zeros(n_repeat)
            for jj,repeat in enumerate(range(1,n_repeat+1)):
                D50_partmc[jj], D50_mam4[jj] = get_D50s(np.exp(lnDs),dNdlnD_partmc_allrepeats[ii,:,jj],dNdlnD_mam4[ii,:])
            vardat[ii] = np.mean(D50_mam4/D50_partmc)
            # print(np.mean(D50_mam4/D50_partmc),np.std(D50_mam4/D50_partmc))
    elif varname.startswith('log_N_tot'):
        Ns = np.loadtxt(tt_dir + 'N0s.txt')
        vardat = np.log10(np.sum(Ns,axis=1))
    elif varname.startswith('log_sa_accumulated'):
        dDs = np.exp(np.loadtxt(tt_dir + 'mus.txt')) - np.exp(np.loadtxt(tt_dir + 'mu0s.txt'))
        Ns = np.loadtxt(tt_dir + 'N0s.txt')
        vardat = np.log10(np.sum(dDs*Ns/2.,axis=1)/np.sum(Ns,axis=1))
    elif varname.startswith('sa_accumulated'):
        mu0s = np.loadtxt(tt_dir + 'mu0s.txt')
        mus = np.loadtxt(tt_dir + 'mus.txt')
        N0s = np.loadtxt(tt_dir + 'N0s.txt')
        Ns = np.loadtxt(tt_dir + 'Ns.txt')
        A0 = np.sum(4*np.pi*N0s*np.exp(2.*mu0s + 2.*mode_sigs**2),axis=1)
        V = np.sum(4./3. * np.pi * Ns * np.exp(3*mus * 4.5*mode_sigs**2),axis=1)
        V0 = np.sum(4./3. * np.pi * Ns * np.exp(3*mu0s * 4.5*mode_sigs**2),axis=1)
        # dDs = np.exp(np.loadtxt(tt_dir + 'mus.txt')) - np.exp(np.loadtxt(tt_dir + 'mu0s.txt'))
        vardat = (V-V0)/A0
        #np.sum(dDs*Ns/2.,axis=1)/np.sum(Ns,axis=1)
    elif varname.startswith('integrated_Ntot'):
        Ns = np.loadtxt(tt_dir + 'N0s.txt')
        vardat = np.sum(Ns,axis=1)*time
    elif varname.startswith('mu_overall'):
        mu0s = np.loadtxt(tt_dir + 'mu0s.txt')
        N0s = np.loadtxt(tt_dir + 'N0s.txt')
        vardat = np.sum(mu0s*N0s,axis=1)/np.sum(N0s,axis=1)
    elif varname.startswith('sa_flux_mam'):
        tt = int(tt_dir[-5:-1])
        
        mus = np.loadtxt(tt_dir + 'mus.txt')
        Ns = np.loadtxt(tt_dir + 'Ns.txt')
        V = np.sum(4./3. * np.pi * Ns * np.exp(3*mus * 4.5*mode_sigs**2),axis=1)
        
        if tt == 0:
            vardat = np.nan*np.ones_like(V)#zeros(V.shape)
        else:
            dtime = time/float(tt)
            tt0_dir = tt_dir[:-5] + str(tt-1).zfill(4) + '/'
            mu0s = np.loadtxt(tt0_dir + 'mus.txt')
            N0s = np.loadtxt(tt0_dir + 'Ns.txt')
            
            A0 = np.sum(4*np.pi*N0s*np.exp(2.*mu0s + 2.*mode_sigs**2),axis=1)
            V0 = np.sum(4./3. * np.pi * Ns * np.exp(3*mu0s * 4.5*mode_sigs**2),axis=1)
            vardat = (V-V0)/(A0*dtime)
    elif varname.startswith('log_sa_flux'):
        dDs = np.exp(np.loadtxt(tt_dir + 'mus.txt')) - np.exp(np.loadtxt(tt_dir + 'mu0s.txt'))
        Ns = np.loadtxt(tt_dir + 'N0s.txt')
        dDsdt = dDs/time
        vardat = np.log10(np.sum(dDsdt*Ns/2.,axis=1)/np.sum(Ns,axis=1))
    elif varname.startswith('integrated_num_conc'):
        try:
            vardat = np.mean(time*np.loadtxt(tt_dir + 'num_conc.txt'),axis=1)
        except:
            Ns = np.loadtxt(tt_dir + 'N0s.txt')
            vardat = np.sum(time*Ns,axis=1)
    elif varname.startswith('integrated_sa_flux'):
        try:
            vardat = np.mean(time*np.loadtxt(tt_dir + 'sa_flux.txt'),axis=1)
        except:
            dDs = np.exp(np.loadtxt(tt_dir + 'mus.txt')) - np.exp(np.loadtxt(tt_dir + 'mu0s.txt'))
            Ns = np.loadtxt(tt_dir + 'N0s.txt')
            dDsdt = dDs/time
            vardat = np.sum(time*dDsdt*Ns/2.,axis=1)/np.sum(Ns,axis=1)
            
    elif varname.startswith('N_tot') or varname.startswith('num_conc'):
        try:
            vardat = np.mean(np.loadtxt(tt_dir + 'num_conc.txt'),axis=1)
        except:
            Ns = np.loadtxt(tt_dir + 'N0s.txt')
            vardat = np.sum(Ns,axis=1)
    elif varname.startswith('sa_flux'):
        try:
            vardat = np.mean(np.loadtxt(tt_dir + 'sa_flux.txt'),axis=1)
        except:
            dDs = np.exp(np.loadtxt(tt_dir + 'mus.txt')) - np.exp(np.loadtxt(tt_dir + 'mu0s.txt'))
            Ns = np.loadtxt(tt_dir + 'N0s.txt')
            dDsdt = dDs/time
            vardat = np.sum(dDsdt*Ns/2.,axis=1)/np.sum(Ns,axis=1)
    elif varname.startswith('dmu'):
        vardat =  np.log10(np.exp(np.loadtxt(tt_dir + 'mus.txt')[:,col_idx])) - np.log10(np.exp(np.loadtxt(tt_dir + 'mu0s.txt')[:,col_idx]))
    elif varname.startswith('dDrel'):
        vardat =  (np.exp(np.loadtxt(tt_dir + 'mus.txt')[:,col_idx]) - np.exp(np.loadtxt(tt_dir + 'mu0s.txt')[:,col_idx]))/np.exp(np.loadtxt(tt_dir + 'mu0s.txt')[:,col_idx])
    elif varname.startswith('dD'):
        vardat =  np.exp(np.loadtxt(tt_dir + 'mus.txt')[:,col_idx]) - np.exp(np.loadtxt(tt_dir + 'mu0s.txt')[:,col_idx])
    elif varname.startswith('N0norm'):
        Ns = np.loadtxt(tt_dir + 'N0s.txt')
        Ntot = np.sum(Ns,axis=1)
        vardat = Ns[:,col_idx]/Ntot
    elif varname.startswith('dNnorm'):
        Ns = np.loadtxt(tt_dir + 'N0s.txt')
        Ntot = np.sum(Ns,axis=1)
        vardat = (np.loadtxt(tt_dir + 'Ns.txt')[:,col_idx] - Ns[:,col_idx])/Ntot
    elif varname.startswith('dlog_Ntot'):
        N0s = np.loadtxt(tt_dir + 'N0s.txt')
        Ns = np.loadtxt(tt_dir + 'Ns.txt')
        vardat = np.log10(np.sum(Ns,axis=1))-np.log10(np.sum(N0s,axis=1))
    elif varname.startswith('dlogN'):
        vardat =  np.log10(np.loadtxt(tt_dir + 'Ns.txt')[:,col_idx]) - np.log10(np.loadtxt(tt_dir + 'N0s.txt')[:,col_idx])
    elif varname.startswith('dN'):
        vardat =  np.loadtxt(tt_dir + 'Ns.txt')[:,col_idx] - np.loadtxt(tt_dir + 'N0s.txt')[:,col_idx]
    elif varname.startswith('N0'):
        vardat =  np.loadtxt(tt_dir + 'N0s.txt')[:,col_idx]
    elif varname.startswith('logN0'):
        vardat =  np.log10(np.loadtxt(tt_dir + 'N0s.txt')[:,col_idx])
    elif varname.startswith('mu0'):
        vardat =  np.log10(np.exp(np.loadtxt(tt_dir + 'mu0s.txt')[:,col_idx]))
    elif varname.startswith('KL_backward'):
        if recompute_KL:
            KL_repeats = recompute_all_KL(tt_dir,n_repeat,backward=True)
            try:
                vardat = np.mean(KL_repeats,axis=1)
            except:
                vardat = KL_repeats
        else:
            try:
                vardat = np.mean(np.loadtxt(tt_dir + 'KL_repeats_backward.txt'),axis=1)
            except:
                vardat = np.loadtxt(tt_dir + 'KL_repeats_backward.txt')
    elif varname.startswith('KL'):
        if recompute_KL:
            KL_repeats = recompute_all_KL(tt_dir,n_repeat,backward=False)
            try:
                vardat = np.mean(KL_repeats,axis=1)
            except:
                vardat = KL_repeats
        else:
            try:
                vardat = np.mean(np.loadtxt(tt_dir + varname + '_repeats.txt'),axis=1)
            except:
                vardat = np.loadtxt(tt_dir + varname + '_repeats.txt')

    else:
        dataset =  np.loadtxt(tt_dir + varname + '.txt')
        vardat = dataset[:,col_idx]
    return vardat

def recompute_all_KL(tt_dir,n_repeat,backward=False):
    dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
    
    dNdlnD_partmc = np.zeros(np.shape(dNdlnD_mam4)+(n_repeat,))
    for ii in range(n_repeat):
        dNdlnD_partmc[:,:,ii] = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii+1).zfill(4) + '.txt')
    
    KL_repeats = np.zeros([dNdlnD_partmc.shape[0],dNdlnD_partmc.shape[2]])
    for ss in range(dNdlnD_partmc.shape[0]):
        for ii in range(dNdlnD_partmc.shape[2]):
            KL_repeats[ss,ii] = get_KL_binned(dNdlnD_partmc[ss,:,ii],dNdlnD_mam4[ss,:],backward=backward)
    return KL_repeats

def read_dsds(tt_dir,n_repeat,lnDs = np.log(np.logspace(-10,-5,100)),backward=False,recompute_KL=False):
    
    dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
    
    dNdlnD_partmc = np.zeros(np.shape(dNdlnD_mam4)+(n_repeat,))
    for ii in range(n_repeat):
        dNdlnD_partmc[:,:,ii] = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii+1).zfill(4) + '.txt')
    
    if recompute_KL:
        KL_repeats = np.zeros([dNdlnD_partmc.shape[0],dNdlnD_partmc.shape[2]])
        for ss in range(dNdlnD_partmc.shape[0]):
            for ii in range(dNdlnD_partmc.shape[2]):
                KL_repeats[ss,ii] = get_KL_binned(dNdlnD_partmc[ss,:,ii],dNdlnD_mam4[ss,:],backward=backward)
    else:
        if backward:
            KL_repeats = np.loadtxt(tt_dir + 'KL_repeats_backward.txt')
        else:
            KL_repeats = np.loadtxt(tt_dir + 'KL_repeats.txt')
    return lnDs,dNdlnD_partmc,dNdlnD_mam4,KL_repeats

def read_frac_ccn(tt_dir,n_repeat):
    s_env_vals = np.loadtxt(tt_dir + 's_env_vals.txt')
    frac_ccn_mam4 = np.loadtxt(tt_dir + 'frac_ccn_mam4.txt')
    frac_ccn_partmc_mean = np.loadtxt(tt_dir + 'frac_ccn_partmc_mean.txt')
    frac_ccn_partmc_std = np.loadtxt(tt_dir + 'frac_ccn_partmc_std.txt')
    
    return s_env_vals, frac_ccn_partmc_mean, frac_ccn_partmc_std, frac_ccn_mam4
 
def kfold_split(X,y,n_splits=None):
    if n_splits == None:
        # N_scenarios = len(y)
        kf = LeaveOneOut()#
        # kf = KFold(n_splits=N_scenarios)
    else:
        kf = KFold(n_splits=n_splits)
    
    folds = []
    for train, test in kf.split(X):
        folds.append([train,test])
    return folds
    
def test_train_split(X,y,frac_testing=0.1):
    scenarios = range(len(y))
    N_test = int(len(scenarios)*frac_testing)
    testing_scenarios = random.choices(scenarios,k=N_test)
    training_scenarios = [scenario for scenario in scenarios if scenario not in testing_scenarios]

    idx_test = np.array([int(scenario) for scenario in testing_scenarios])
    idx_train = np.array([int(scenario) for scenario in training_scenarios])
    
    print(X.shape)
    X_train = X[idx_train,:]
    X_test = X[idx_test,:]
    y_train = y[idx_train]
    y_test = y[idx_test]
    
    return X_train,X_test,y_test,y_train

def test_train_validation_split(X,y,frac_testing=0.1,frac_validation=0.1):
    scenarios = range(len(y))
    N_test = int(len(scenarios)*frac_testing)
    N_validation = int(len(scenarios)*frac_validation)
    testing_scenarios = random.choices(scenarios,k=N_test)
    remaining_scenarios = [scenario for scenario in scenarios if scenario not in testing_scenarios]
    validation_scenarios = random.choices(remaining_scenarios,k=N_validation)
    
    training_scenarios = [scenario for scenario in scenarios if scenario not in testing_scenarios and scenario not in validation_scenarios]
    
    idx_test = np.array([int(scenario) for scenario in testing_scenarios])
    idx_validation = np.array([int(scenario) for scenario in validation_scenarios])
    idx_train = np.array([int(scenario) for scenario in training_scenarios])
    
    print(X.shape)
    X_train = X[idx_train,:]
    X_test = X[idx_test,:]
    X_validation = X[idx_test,:]    
    y_train = y[idx_train]
    y_test = y[idx_test]
    y_validation = y[idx_validation]
    
    return X_train,X_test,X_validation,y_test,y_train,y_validation

# =============================================================================
# get inputs for surrogate model from E3SM file
# =============================================================================
def get_X_e3sm(varnames,f,tt='all',lev='all',return_shape=False):
    X_e3sm_raveled = []
    for varname in varnames:
        one_x_gridded = retrieve_e3sm_data(f,varname)
        if tt == 'all':
            if lev == 'all':        
                X_e3sm_raveled.append(one_x_gridded.ravel())
                the_shape = one_x_gridded.shape
            else:
                X_e3sm_raveled.append(one_x_gridded[:,lev,:,:].ravel())
                the_shape = one_x_gridded[:,lev,:,:].shape
        else:
            if lev == 'all':
                X_e3sm_raveled.append(one_x_gridded[tt,:,:,:].ravel())
                the_shape = one_x_gridded[tt,:,:,:].shape
            else:
                X_e3sm_raveled.append(one_x_gridded[tt,lev,:,:].ravel())
                the_shape = one_x_gridded[tt,lev,:,:].shape
    if return_shape:
        return np.array(X_e3sm_raveled).transpose(),the_shape
    else:
        return np.array(X_e3sm_raveled).transpose()

def get_e3sm_grid(f):
    time = retrieve_e3sm_data(f,'time')
    lev = retrieve_e3sm_data(f,'lev')
    lat = retrieve_e3sm_data(f,'lat')
    lon = retrieve_e3sm_data(f,'lon')
    return time, lev, lat, lon
    
def unravel_y(y_raveled,shape_gridded):
    y_gridded = y_raveled.reshape(shape_gridded)
    return y_gridded

def retrieve_e3sm_data(f,varname,dt_e3sm=1800.):
    try: mode = int(varname[-1])
    except: 
        try: mode = int(varname[4])
        except: pass
    
    if varname.startswith('log_N_tot'):
        Ntot = np.zeros(f['qnum_init_a' + str(1)][:].shape)
        for a_mode in range(1,5):
            Ntot += f['qnum_init_a' + str(a_mode)][:]
        vardat = np.log10(Ntot)
    elif varname.startswith('log_sa_flux'):
        dD_tot = np.zeros(f['qnum_init_a' + str(1)][:].shape)
        Ntot = np.zeros(f['qnum_init_a' + str(1)][:].shape)
        for a_mode in range(1,5):
            Ntot += f['qnum_init_a' + str(a_mode)][:]
            dD_tot += f['qnum_init_a' + str(a_mode)][:]*(f['dgn_a_gat_a' + str(a_mode)][:] - f['dgn_a_init_a' + str(a_mode)][:])
        dDdt = (dD_tot/Ntot)/dt_e3sm
        vardat = np.log10(dDdt/2.)
        # idx_nan = np.where(np.isnan(vardat))
        # vardat[idx_nan] = -100
        
        # idx_inf = np.where(np.isinf(vardat))
        # vardat[idx_inf] = -30
    elif varname.startswith('N_tot'):
        Ntot = np.zeros(f['qnum_init_a' + str(1)][:].shape)
        for a_mode in range(1,5):
            Ntot += f['qnum_init_a' + str(a_mode)][:]
        vardat = Ntot
    elif varname.startswith('sa_flux'):
        dD_tot = np.zeros(f['qnum_init_a' + str(1)][:].shape)
        Ntot = np.zeros(f['qnum_init_a' + str(1)][:].shape)
        for a_mode in range(1,5):
            Ntot += f['qnum_init_a' + str(a_mode)][:]
            dD_tot += f['qnum_init_a' + str(a_mode)][:]*(f['dgn_a_gat_a' + str(a_mode)][:] - f['dgn_a_init_a' + str(a_mode)][:])
        dDdt = (dD_tot/Ntot)/dt_e3sm
        vardat = dDdt/2.
    elif varname.startswith('dlog_Ntot'):
        N0tot = np.zeros(f['qnum_init_a' + str(1)][:].shape)
        Ntot = np.zeros(f['qnum_init_a' + str(1)][:].shape)
        for a_mode in range(1,5):
            N0tot += f['qnum_init_a' + str(a_mode)][:]
            Ntot += f['qnum_pcag_a' + str(a_mode)][:]
        vardat = np.log10(Ntot) - np.log10(N0tot)
    elif varname.startswith('N0norm'):
        Ntot = np.zeros(f['qnum_init_a' + str(mode)][:].shape)
        for a_mode in range(1,5):
            Ntot += f['qnum_init_a' + str(a_mode)][:]
        vardat = f['qnum_init_a' + str(mode)][:]/Ntot
    elif varname.startswith('dNnorm'):
        Ntot = np.zeros(f['qnum_init_a' + str(mode)][:].shape)
        for a_mode in range(1,5):
            Ntot += f['qnum_init_a' + str(a_mode)][:]
        vardat = (f['qnum_pcag_a' + str(mode)][:] - f['qnum_init_a' + str(mode)][:])/Ntot
    elif varname.startswith('N0'):
        vardat = f['qnum_init_a' + str(mode)][:]
    elif varname.startswith('logN0'):
        vardat = np.log10(f['qnum_init_a' + str(mode)][:])
    elif varname.startswith('mu0'):    
        vardat = np.log10(f['dgn_a_init_a' + str(mode)][:])
    elif varname.startswith('mu_gat'):
        vardat = np.log10(f['dgn_a_gat_a' + str(mode)][:])
    elif varname.startswith('mu_rename'):
        vardat = np.log10(f['dgn_a_rename_a' + str(mode)][:])
    elif varname.startswith('dlogN_rename'):
        vardat = f['qnum_rename_a' + str(mode)][:] - f['qnum_init_a' + str(mode)][:]
    elif varname.startswith('dmu_rename'):
        vardat = np.log10(f['dgn_a_rename_a' + str(mode)][:]) - np.log10(f['dgn_a_init_a' + str(mode)][:])
    elif varname.startswith('dlogN_gat'):
        vardat = f['qnum_gat_a' + str(mode)][:] - f['qnum_init_a' + str(mode)][:]
    elif varname.startswith('dmu_gat'):
        vardat = np.log10(f['dgn_a_gat_a' + str(mode)][:]) - np.log10(f['dgn_a_init_a' + str(mode)][:])
    elif varname.startswith('dmu'):
        vardat = np.log10(f['dgn_a_pcag_a' + str(mode)][:]) - np.log10(f['dgn_a_init_a' + str(mode)][:])
    elif varname.startswith('dlogN'):
        vardat = np.log10(f['qnum_pcag_a' + str(mode)][:]) - np.log10(f['qnum_init_a' + str(mode)][:])
    elif varname.startswith('dD'):
        vardat = f['dgn_a_pcag_a' + str(mode)][:] - f['dgn_a_init_a' + str(mode)][:]
    elif varname.startswith('dN'):
        vardat = f['qnum_pcag_a' + str(mode)][:] - f['qnum_init_a' + str(mode)][:]
    elif varname.startswith('precursor_gases'):
        vardat = f['H2SO4'][:] + f['SOAG'][:]
    elif varname.endswith('frac_hygroscopic'):
        tot_mass = get_summed_mass(f,mode,spec_names=['so4','ncl','pom','mom','soa','bc','dst'])
        hygroscopic_mass = get_summed_mass(f,mode,spec_names=['so4','ncl'])
        vardat = hygroscopic_mass/tot_mass
    elif varname.endswith('frac_hygroscopic_thats_ncl'):
        hygroscopic_mass = get_summed_mass(f,mode,spec_names=['so4','ncl'])
        ncl_mass = get_mode_mass(f,mode,'ncl')
        vardat = ncl_mass/hygroscopic_mass
    elif varname.endswith('frac_hydrophobic_thats_insol'):
        hydrophobic_mass = get_summed_mass(f,mode,spec_names=['pom','mom','soa','bc','dst'])
        insol_mass = get_summed_mass(f,mode,spec_names=['bc','dst'])
        vardat = insol_mass/hydrophobic_mass
    elif varname.endswith('frac_insol_thats_bc'):
        insol_mass = get_summed_mass(f,mode,spec_names=['bc','dst'])
        bc_mass = get_mode_mass(f,mode,'bc')
        vardat = bc_mass/insol_mass
    elif varname.endswith('frac_org_thats_pom'):
        org_mass = get_summed_mass(f,mode,spec_names=['pom','mom','soa'])
        pom_mass = get_summed_mass(f,mode,spec_names=['pom','mom'])
        vardat = pom_mass/org_mass
    elif varname.endswith('frac_bc'):
        mode4_mass = get_summed_mass(f,mode,spec_names=['bc','pom','mom'])
        bc_mass = get_mode_mass(f,mode,'bc')
        vardat = bc_mass/mode4_mass
    else:
        vardat = f[varname][:]
    
    return vardat


def get_mode_mass(f,mode_num,spec):
    try:
        vardat = f[spec + '_a' +str(mode_num)][:]
    except:
        vardat = np.zeros_like(f['mom_a' + str(mode_num)][:])
    return vardat

def get_summed_mass(f,mode_num,spec_names=['so4','ncl','pom','mom','soa','bc','dst']):
    vardat = np.zeros_like(f['mom_a' + str(mode_num)][:])
    for spec in spec_names:
        vardat += get_mode_mass(f,mode_num,spec)
    return vardat

from scipy.stats import entropy
def get_KL_binned(dNdlnD_partmc,dNdlnD_mam4,backward=False):
    P_partmc = dNdlnD_partmc/sum(dNdlnD_partmc)
    P_mam4 = dNdlnD_mam4/sum(dNdlnD_mam4)
    
    if backward:
        KL = entropy(P_partmc,P_mam4)
    else:
        KL = entropy(P_mam4,P_partmc)
    # idx, = np.where(P_partmc>0)
    # idx, = np.where(P_mam4>0)
    # if backward:
    #     KL = entropy(P_partmc[idx],P_mam4[idx])
    # else:
    #     KL = entropy(P_mam4[idx],P_partmc[idx])
    # # idx, = np.where([oneP_mam4>0 and oneP_partmc>0 for (oneP_mam4,oneP_partmc) in zip(P_mam4,P_partmc)])
    # idx, = np.where(P_mam4>0)
    # if backward:
    #     KL = np.sum(P_mam4[idx]*np.log(P_mam4[idx]/P_partmc[idx]))
    # else:
    #     KL = np.sum(P_partmc[idx]*np.log(P_partmc[idx]/P_mam4[idx]))
    return KL

def get_D50s(Ds,dNdlnD_partmc,dNdlnD_mam4):
    cdf_partmc = np.cumsum(dNdlnD_partmc)/np.sum(dNdlnD_partmc)
    cdf_mam4 = np.cumsum(dNdlnD_mam4)/np.sum(dNdlnD_mam4)
    
    D50_partmc = np.interp(0.5,cdf_partmc,Ds)
    D50_mam4 = np.interp(0.5,cdf_mam4,Ds)
    
    return D50_partmc, D50_mam4
