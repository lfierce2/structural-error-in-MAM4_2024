#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import os, netCDF4
import numpy as np
# import matplotlib.pyplot as plt
from benchmark_splitToNodes import get_mam_input#, get_partmc_input
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
# from pyrcel.thermo import kohler_crit
# import PyMieScatt
import read_partmc

MW_h2so4 = 98.
MW_air = 29.


def get_mode_dsd_params(scenario_dir,tt,num_multiplier=1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8]):
    if tt == 0:
        mam_input = scenario_dir + 'mam_input.nl'
        Ns = np.zeros([len(gsds)])
        for kk in range(len(Ns)):#,mode in enumerate(modes):
            Ns[kk] = get_mam_input(
                'numc' + str(kk+1),
                mam_input=mam_input)
        mus=np.loadtxt(scenario_dir+'mode_mus.txt')
    else:
        mam_output = scenario_dir + 'mam_output.nc'
        f_output = netCDF4.Dataset(mam_output)
        Ns = f_output['num_aer'][:,tt-1]
        mus = np.log(f_output['dgn_a'][:,tt-1])
    sigs=np.log(gsds)
    return Ns,mus,sigs


def get_mode_dsds(
        lnDs,scenario_dir,tt,num_multiplier = 1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8]):
    dNdlnD_modes = np.zeros([len(gsds),len(lnDs)])
    Ns,mus,sigs = get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
    for kk,(N,mu,sig) in enumerate(zip(Ns,mus,sigs)):
        dNdlnD_modes[kk,:] = N*norm(loc=mu,scale=sig).pdf(lnDs)
    return dNdlnD_modes

#def get_mam4_dsd(lnDs,scenario_dir,tt,num_multiplier = 1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8]):
    
    
def get_mam4_ccn(s_env_vals,scenario_dir,tt,num_multiplier=1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8],return_modes=False):
    mam_input = scenario_dir + 'mam_input.nl'
    temp = get_mam_initial(mam_input,'temp')
    Ns,mus,sigs = get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
    mode_tkappas = get_mode_kappas(scenario_dir,tt)
    
    N_ccn_modes = np.zeros([len(Ns),len(s_env_vals)])
    
    for ii,(N,mu,sig,tkappa) in enumerate(zip(Ns,mus,sigs,mode_tkappas)):
        norm_cdf = lambda lnD: norm(loc=mu,scale=sig).cdf(lnD)
        Ddrys_thresh = []
        for ss,s_env in enumerate(s_env_vals):
            Ddry_thresh = read_partmc.process_compute_Ddry_thresh(s_env, temp, tkappa)
            Ddrys_thresh.append(Ddry_thresh)
            N_ccn_modes[ii,ss] = N*(1.-norm_cdf(np.log(Ddry_thresh)))
            if Ddry_thresh<=0:
                print('ii',ii,'ss',ss,'Ddry_thresh',Ddry_thresh)
    frac_ccn_mam4 = np.sum(N_ccn_modes,axis=0)/np.sum(Ns,axis=0)
    
    if return_modes:
        return frac_ccn_mam4,N_ccn_modes,Ns
    else:
        return frac_ccn_mam4

def get_partmc_ccn(s_env_vals,scenario_dir,tt,unique_repeats=[1],return_repeats=False):
    timestep = tt + 1
    partmc_dir = scenario_dir + 'out/'
    N_repeats = max(unique_repeats)
    N_ccn_repeat = np.zeros([N_repeats,len(s_env_vals)])
    N_tot_repeat = np.zeros(N_repeats)
    frac_ccn_repeat = np.zeros([N_repeats,len(s_env_vals)])
    
    for ii,repeat in enumerate(unique_repeats):
        ncfile = read_partmc.get_ncfile(partmc_dir, timestep, ensemble_number=repeat)
        s_crit = read_partmc.get_partmc_variable(ncfile,'s_critical')
        num_conc = read_partmc.get_partmc_variable(ncfile,'aero_num_conc')
        for ss,s_env in enumerate(s_env_vals):
            N_ccn_repeat[ii,ss] = sum(num_conc*(s_crit<=s_env))
        N_tot_repeat[ii] = sum(num_conc)
        frac_ccn_repeat[ii,:] = N_ccn_repeat[ii,:]/N_tot_repeat[ii]
    frac_ccn_mean = np.sum(N_ccn_repeat,axis=0)/sum(N_tot_repeat)
#    frac_ccn_std = np.zeros(frac_ccn_mean.shape)
#    for ss,s_env in enumerate(s_env_vals):
#        frac_ccn_std[ss] = np.sqrt(sum(N_tot_repeat*(frac_ccn_repeat[:,ss]-frac_ccn_mean[ss])**2)/sum(N_tot_repeat))
    frac_ccn_std = np.std(frac_ccn_repeat,axis=0)
    if return_repeats:
        return frac_ccn_mean, frac_ccn_std, frac_ccn_repeat, N_tot_repeat
    else:
        return frac_ccn_mean, frac_ccn_std
    

    
def get_mode_kappas(scenario_dir,tt):
    mam_output = scenario_dir + 'mam_output.nc'
    f_output = netCDF4.Dataset(mam_output)
    
    mode_tkappas = np.zeros(f_output['num_aer'].shape[0])
    for ii in range(f_output['num_aer'].shape[0]):
        mode_num = ii + 1
        spec_names = get_mode_specnames(mode_num)
        
        vol_tot = 0.
        volKap_tot = 0.
        for kk,spec_name in enumerate(spec_names):
            spec_kappa = get_spec_kappa(spec_name)
            spec_density = get_spec_density(spec_name)
            if tt == 0:
                mam_input = scenario_dir + 'mam_input.nl'
                vol = get_mam_initial(mam_input,'mf' + spec_name  + str(mode_num))/spec_density
            else:
                vol = f_output[spec_name + '_aer'][ii,tt-1]/spec_density
            
            vol_tot += vol
            volKap_tot += vol*spec_kappa
        mode_tkappas[ii] = volKap_tot/vol_tot
    return mode_tkappas

def get_mode_densities(scenario_dir,tt):
    mam_output = scenario_dir + 'mam_output.nc'
    f_output = netCDF4.Dataset(mam_output)
    
    mode_densities = np.zeros(f_output['num_aer'].shape[0])
    for ii in range(f_output['num_aer'].shape[0]):
        mode_num = ii + 1
        spec_names = get_mode_specnames(mode_num)
        
        vol_tot = 0.
        mass_tot = 0.
        for kk,spec in enumerate(spec_names):
            spec_density = get_spec_density(spec)
            if tt == 0:
                mam_input = scenario_dir + 'mam_input.nl'
                mass = get_mam_initial(mam_input,'mf' + spec  + str(mode_num))
            else:
                mass = f_output[spec + '_aer'][ii,tt-1]
                
            vol_tot += mass/spec_density
            mass_tot += mass
        mode_densities[ii] = mass_tot/vol_tot
    return mode_densities
        
def get_spec_kappa(spec_name):
    if spec_name == 'so4':
        spec_kappa = 0.65
    elif spec_name == 'pom':
        spec_kappa = 0.001
    elif spec_name == 'soa':
        spec_kappa = 0.1
    elif spec_name == 'bc':
        spec_kappa = 0.
    elif spec_name == 'dst':
        spec_kappa = 0.1
    elif spec_name == 'ncl':
        spec_kappa = 1.28
    return spec_kappa

def get_spec_density(spec_name):
    if spec_name == 'so4':
        spec_density = 1800.
    elif spec_name == 'pom':
        spec_density = 1000.
    elif spec_name == 'soa':
        spec_density = 1400.
    elif spec_name == 'bc':
        spec_density = 1800.
    elif spec_name == 'dst':
        spec_density = 2600.
    elif spec_name == 'ncl':
        spec_density = 2200.
    return spec_density
        
def get_mode_specnames(mode_num):
    if mode_num == 1 or mode_num == 3:
        mode_specs = ['so4','pom','soa','bc','dst','ncl']
    elif mode_num == 2:
        mode_specs = ['so4','soa','ncl']
    elif mode_num == 4:
        mode_specs = ['pom','bc']
    return mode_specs
        

def get_mam4_dsd(lnDs,scenario_dir,tt,num_multiplier=1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8]):
    dNdlnD_modes = get_mode_dsds(lnDs,scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
    dNdlnD_mam4 = np.sum(dNdlnD_modes,axis=0)
    return dNdlnD_mam4

def get_partmc_dsd(lnDs,scenario_dir,tt,unique_repeats=[1],density_type='hist'):
    timestep = tt + 1
    partmc_dir = scenario_dir + 'out/'
    N_repeats = max(unique_repeats)
    dNdlnD_repeats = np.zeros([N_repeats,len(lnDs)])
    for ii,repeat in enumerate(unique_repeats):
        ncfile = read_partmc.get_ncfile(partmc_dir, timestep, ensemble_number=repeat)
        dNdlnD_repeats[ii,:] = get_partmc_dsd_onefile(lnDs,ncfile,density_type=density_type)
#        dry_dia = read_partmc.get_partmc_variable(ncfile,'dry_diameter')
#        num_conc = read_partmc.get_partmc_variable(ncfile,'aero_num_conc')
#        dndlnD_repeats[ii,:] = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.log(dry_dia),sample_weight=num_conc)
    
    return dNdlnD_repeats

def get_partmc_dsd_onefile(lnDs,ncfile,density_type='hist'):
    dry_dia = read_partmc.get_partmc_variable(ncfile,'dry_diameter')
    num = read_partmc.get_partmc_variable(ncfile,'aero_num_conc')
    
    if density_type == 'hist':
        lnDs_edges = np.linspace(lnDs[0]-(lnDs[1]-lnDs[0])/2.,lnDs[-1]+(lnDs[1]-lnDs[0])/2.,len(lnDs)+1)
        dndlnD,ln_dia_bins = np.histogram(np.log(dry_dia),bins=lnDs_edges,weights=num,density=True)
    elif density_type == 'gaussian':
        kde = KernelDensity(kernel='gaussian', bandwidth=1e-5).fit(np.log(dry_dia).reshape(-1,1),sample_weight=num/sum(num))
        dndlnD = kde.score_samples(lnDs.reshape(-1,1))
    dNdlnD_partmc = sum(num)*dndlnD
    
    return dNdlnD_partmc

def get_mam_initial(mam_input,varname):
    if varname == 'H2SO4_gas':
        vardat_init = get_mam_input(
            'qh2so4',
            mam_input=mam_input)*1e9
    else:
        vardat_init = get_mam_input(varname,mam_input=mam_input)
    return vardat_init


def get_num_multipliers(
        ensemble_dir,modes=[1,2,3,4]):
    scenarios = [file for file in os.listdir(ensemble_dir) if not file.startswith('.')]
    
    for ii,scenario in enumerate(scenarios):
        scenario_dir = ensemble_dir + scenario + '/'
        mam_input = scenario_dir + 'mam_input.nl'
        num_input = np.zeros([len(modes)])
        for kk,mode in enumerate(modes):
            num_input[kk] = get_mam_input(
                'numc' + str(kk+1),
                mam_input=mam_input)
        mam_output = scenario_dir + 'mam_output.nc'
        f_output = netCDF4.Dataset(mam_output)
        num_output = f_output['num_aer'][:,0]
        num_multiplier = num_output/num_input
        np.savetxt(scenario_dir + 'num_multiplier.txt', num_multiplier)

def get_mk_normalized(k,moment_type,mu,sig):
    if moment_type == 'power':
        mk = np.exp(k*mu + k**3/2*sig**2)
    elif moment_type == 'power_log':
        if k == 0.:
            mk = 1
        elif k == 1.:
            mk = mu
        elif k == 2.:
            mk = (mu**2 + sig**2)
        elif k == 3.:
            mk = (mu**3 + 3*mu*sig**2)
        elif k == 4.:
            mk = (mu**4 + 6*mu**2*sig**2 + 3*sig**4)
        elif k == 5.:
            mk = (mu**5 + 10*mu**3*sig**2 + 15*mu*sig**4)
        elif k == 6.:
            mk = (mu**6 + 15*mu**4*sig**2 + 45*mu**2*sig**4 + 15*sig**6)
        elif k == 7.:
            mk = (mu**7 + 21*mu**5*sig**2 + 105*mu**3*sig**4 + 105*mu*sig**6)
        elif k == 8.:
            mk = (mu**8 + 28*mu**6*sig**2 + 210*mu**4*sig**4 + 420*mu**2*sig**6 + 105*sig**8)
    return mk

def get_moments(k,moment_type,Ns,mus,sigs):
    mk = np.zeros(Ns.shape)
    for kk,(mu,sig) in enumerate(zip(mus,sigs)):
        mk[kk] = get_mk_normalized(k,moment_type,mu,sig)
    
    return sum(mk*Ns)

def get_moments_normalized(k,moment_type,Ns,mus,sigs):
    mk = np.zeros(Ns.shape)
    for kk,(mu,sig) in enumerate(zip(mus,sigs)):
        mk[kk] = get_mk_normalized(k,moment_type,mu,sig)
    return sum(mk*Ns)/sum(Ns)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
