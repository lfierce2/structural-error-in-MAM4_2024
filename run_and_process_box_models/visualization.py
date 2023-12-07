#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Laura Fierce

"""

import analyses
import matplotlib.pyplot as plt
import numpy as np
import pickle
from netCDF4 import Dataset
import time
import pandas
import os
import random
import seaborn as sns
from sklearn.neighbors import KernelDensity

def make_input_table(
        ensemble_dir,
        threshold_vals = [0.5,0.05,0.95],
        Xvarnames = [
            'numc2','mu2',
            'mfso42','mfsoa2','mfncl2',
            'numc1','mu1',
            'mfso41','mfsoa1','mfbc1','mfpom1','mfdst1','mfncl1',
            'numc4','mu4',
            'mfbc4','mfpom4',
            'numc3','mu3',
            'mfso43','mfsoa3','mfbc3','mfpom3','mfdst3','mfncl3',
            'h2so4_chem_prod_rate',
            'temp'],
        vartypes = [
            'exp','float','float','float','float',
            'exp','float','float','float','float','float','float','float',
            'exp','float','float','float',
            'exp','float','float','float','float','float','float','float',
            'exp','int'],
        # start here!
        varlabs= [
            'number [cm$^{-3}$]','geom. mean diam [$\mu$m]','mass fraction SO$_4$','mass fraction SOA','mass fraction NaCl',
            'number [cm$^{-3}$]','geom. mean diam [$\mu$m]','mass fraction SO$_4$','mass fraction SOA','mass fraction POA','mass fraction BC','mass fraction dust','mass fraction NaCl',
            'number [cm$^{-3}$]','geom. mean diam [$\mu$m]','mass fraction BC','mass fraction POA',
            'number [cm$^{-3}$]','geom. mean diam [$\mu$m]','mass fraction SO$_4$','mass fraction SOA','mass fraction POA','mass fraction BC','mass fraction dust','mass fraction NaCl',
            'H$_2$SO$_4$ production rate [kg/m$^3$]','temperature [K]'],
        varmultipliers = [
            100**-3,1e6,1.,1.,1.,
            100**-3,1e6,1.,1.,1.,1.,1.,1.,
            100**-3,1e6,1.,1.,
            100**-3,1e6,1.,1.,1.,1.,1.,1.,
            1.,1.],
        collabs = ['input variable','median','5\%','95\%']
        ):
    
    varvals = inputs_from_ensemble_settings(
            ensemble_dir,
            threshold_vals = threshold_vals,
            Xvarnames = Xvarnames)
    for vv in range(varvals.shape[0]):
        varvals[vv,:] = varvals[vv,:]*varmultipliers[vv]
    
    table_lines = make_LaTex_table(varvals,vartypes,varlabs,collabs)
    return table_lines

def make_input_table_CIs(
        ensemble_dir,
        threshold_vals = [0.5,0.05,0.95],
        Xvarnames = [
            'numc2','mu2',
            'mfso42','mfsoa2','mfncl2',
            'numc1','mu1',
            'mfso41','mfsoa1','mfbc1','mfpom1','mfdst1','mfncl1',
            'numc4','mu4',
            'mfbc4','mfpom4',
            'numc3','mu3',
            'mfso43','mfsoa3','mfbc3','mfpom3','mfdst3','mfncl3',
            'h2so4_chem_prod_rate',
            'temp'],
        vartypes = [
            'exp','float','float','float','float',
            'exp','float','float','float','float','float','float','float',
            'exp','float','float','float',
            'exp','float','float','float','float','float','float','float',
            'exp','int'],
        # start here!
        varlabs= [
            'number [cm$^{-3}$]','geom. mean diam [$\mu$m]','mass fraction SO$_4$','mass fraction SOA','mass fraction NaCl',
            'number [cm$^{-3}$]','geom. mean diam [$\mu$m]','mass fraction SO$_4$','mass fraction SOA','mass fraction POA','mass fraction BC','mass fraction dust','mass fraction NaCl',
            'number [cm$^{-3}$]','geom. mean diam [$\mu$m]','mass fraction BC','mass fraction POA',
            'number [cm$^{-3}$]','geom. mean diam [$\mu$m]','mass fraction SO$_4$','mass fraction SOA','mass fraction POA','mass fraction BC','mass fraction dust','mass fraction NaCl',
            'H$_2$SO$_4$ production rate [ppb s$^{-1}$]','temperature [K]'],
        varmultipliers = [
            100**-3,1e6,1.,1.,1.,
            100**-3,1e6,1.,1.,1.,1.,1.,1.,
            100**-3,1e6,1.,1.,
            100**-3,1e6,1.,1.,1.,1.,1.,1.,
            1e9,1.],
        collabs = ['input variable','median','5\%','95\%']
        ):
    
    varvals = inputs_from_ensemble_settings(
            ensemble_dir,
            threshold_vals = threshold_vals,
            Xvarnames = Xvarnames)
    for vv in range(varvals.shape[0]):
        varvals[vv,:] = varvals[vv,:]*varmultipliers[vv]
    
    table_lines = make_LaTex_table_CIs(varvals,vartypes,varlabs,collabs)
    return table_lines
    
def inputs_from_ensemble_settings(
        ensemble_dir,
        threshold_vals = [0.5,0.05,0.95],
        Xvarnames = [
            'numc2','mu2',
            'mfso42','mfsoa2',
            'mfncl2',
            'numc1','mu1',
            'mfso41','mfsoa1','mfbc1','mfpom1','mfdst1','mfncl1',
            'numc4','mu4',
            'mfbc4','mfpom4',
            'numc3','mu3',
            'mfso43','mfsoa3','mfbc3','mfpom3','mfdst3','mfncl3',
            'h2so4_chem_prod_rate',
            'temp']):
    
    mam_ensemble_settings = pickle.load(open(ensemble_dir + 'mam_ensemble_settings.pkl','rb'))
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    all_vars = np.zeros([len(ensemble_settings),len(Xvarnames)])
    for jj in range(len(mam_ensemble_settings)):
        for ii,Xvarname in enumerate(Xvarnames):
            if Xvarname == 'temp':
                all_vars[jj,ii] = ensemble_settings[jj]['temp']['temp']
            elif Xvarnames == 'h2so4_chem_prod_rate':
                all_vars[jj,ii] = ensemble_settings[jj]['gas_emit']['rate']*ensemble_settings[jj]['gas_emit']['H2SO4']
            elif Xvarname.startswith('mu'):
                all_vars[jj,ii] = np.exp(mam_ensemble_settings[jj]['psd_params'][Xvarname])
            elif Xvarname.startswith('mf'):
                mode = Xvarname[-1]
                mode_mf_tot = 0.
                for kk,Xvarname2 in enumerate(Xvarnames):
                    if Xvarname2.startswith('mf') and Xvarname2.endswith(mode):
                        mode_mf_tot += mam_ensemble_settings[jj]['chem_input'][Xvarname2]
                all_vars[jj,ii] = mam_ensemble_settings[jj]['chem_input'][Xvarname]/mode_mf_tot
            else:
                all_vars[jj,ii] = mam_ensemble_settings[jj]['chem_input'][Xvarname]
    
    threshold_predictions = np.zeros([len(Xvarnames),len(threshold_vals)])
    for ii in range(all_vars.shape[1]):
        sorted_vals = np.sort(all_vars[:,ii])
        cdf_vals = np.cumsum(np.ones(len(all_vars[:,ii]))/len(all_vars[:,ii]))
        for tt,threshold in enumerate(threshold_vals):
            threshold_predictions[ii,tt] = np.interp(threshold,cdf_vals,sorted_vals)
    
    return threshold_predictions

def make_LaTex_table(varvals,vartypes,varlabs,collabs):
    table_lines = ['\\begin{center}']
    first_row = '\\begin{tabular}{'
    lab_row = ''
    for cc,collab in enumerate(collabs):
        if cc<(len(collabs)-1):
            first_row += 'l '
            lab_row += '{\\bf ' + collab + '} & '
        else:
            first_row += 'l}'
            lab_row += '{\\bf ' + collab + '} \\\ '
    
    table_lines += [first_row]
    table_lines += [lab_row]
    
    for rr,(varlab,vartype) in enumerate(zip(varlabs,vartypes)):
        onerow = ''
        for cc,collab in enumerate(collabs):
            if cc==0:
                onerow += varlab
            else:
                if varlab.startswith('geom.'):
                    exponential_val = int(np.floor(np.log10(varvals[rr,cc-1])))
                    lead_val = np.round(varvals[rr,cc-1]/10.**exponential_val,3)
                    print(lead_val)
                    if lead_val == np.round(varvals[rr,cc-1]/10.**exponential_val,0):
                        lead_val = int(lead_val)
                    onerow += str(lead_val) + '$\\times$10$^' + str(exponential_val) + '$'
                elif vartype == 'float':
                    onerow += str(np.round(varvals[rr,cc-1],2))                
                elif vartype == 'exp':
                    exponential_val = int(np.floor(np.log10(varvals[rr,cc-1])))
                    lead_val = np.round(varvals[rr,cc-1]/10.**exponential_val,1)
                    if lead_val == np.round(varvals[rr,cc-1]/10.**exponential_val,0):
                        lead_val = int(lead_val)
                    onerow += str(lead_val) + '$\\times$10$^' + str(exponential_val) + '$'
                elif vartype == 'int':
                    onerow += str(int(varvals[rr,cc-1]))
            if cc<(len(collabs)-1):
                onerow += ' & '
            else:
                onerow += ' \\\ '
        table_lines += [onerow]
    table_lines += ['\\end{tabular}']
    table_lines += ['\\end{center}']
    return table_lines

def make_LaTex_table_CIs(varvals,vartypes,varlabs,collabs):
    table_lines = ['\\begin{center}']
    first_row = '\\begin{tabular}{'
    lab_row = ''
    for cc,collab in enumerate(collabs):
        if cc<(len(collabs)-2):
            first_row += 'l '
            lab_row += '{\\bf ' + collab + '} & '
        elif cc<(len(collabs)-1):
            first_row += 'l}'
            lab_row += '{\\bf '+ str(int(collabs[-1][:-2]) - int(collabs[-2][:-2])) +'\\% confidence intervals} \\\ '
    
    table_lines += [first_row]
    table_lines += [lab_row]
    
    for rr,(varlab,vartype) in enumerate(zip(varlabs,vartypes)):
        onerow = ''
        for cc,collab in enumerate(collabs):
            if cc==0:
                onerow += varlab
            else:
                if cc == len(collabs)-2:
                    onerow += '('
                if varlab.startswith('geom.'):
                    onerow += str(np.round(varvals[rr,cc-1],3))
                    # exponential_val = int(np.floor(np.log10(varvals[rr,cc-1])))
                    # lead_val = np.round(varvals[rr,cc-1]/10.**exponential_val,2)
                    # print(lead_val)
                    # if lead_val == np.round(varvals[rr,cc-1]/10.**exponential_val,0):
                    #     lead_val = int(lead_val)
                    # onerow += str(lead_val) + '$\\times$10$^' + str(exponential_val) + '$'
                elif vartype == 'float':
                    onerow += str(np.round(varvals[rr,cc-1],2))
                elif vartype == 'exp':
                    exponential_val = int(np.floor(np.log10(varvals[rr,cc-1])))
                    lead_val = np.round(varvals[rr,cc-1]/10.**exponential_val,1)
                    if lead_val == np.round(varvals[rr,cc-1]/10.**exponential_val,0):
                        lead_val = int(lead_val)
                    onerow += str(lead_val) + '$\\times$10$^' + str(exponential_val) + '$'
                elif vartype == 'int':
                    onerow += str(int(varvals[rr,cc-1]))
                if cc == len(collabs)-2:
                    onerow += ', '
                elif cc == len(collabs)-1:
                    onerow += ')'
            if cc<(len(collabs)-2):
                onerow += ' & '
            elif cc>(len(collabs)-2):
                onerow += ' \\\ '
        table_lines += [onerow]
    table_lines += ['\\end{tabular}']
    table_lines += ['\\end{center}']
    return table_lines
    # for ii,Xvarname in enumerate(Xvarnames):
    #     X_e3sm_raveled,the_shape = analyses.get_X_e3sm([Xvar_prefix + '_' + str(mode)],f,tt=0,lev=-1,return_shape=True)
# def inputs_from_e3sm(
#         e3sm_filename='/Users/fier887/Downloads/E3SM_output_for_PartMC_0607_surf.nc',
#         Xvarnames = [
#             'N0_1','N0_2','N0_3','N0_4',
#             'mu0_1','mu0_2','mu0_3','mu0_4']):
    
#     for ii,Xvarname in enumerate(Xvarnames):
#         X_e3sm_raveled,the_shape = analyses.get_X_e3sm(Xvarname,f,tt=0,lev=-1,return_shape=True)
        
        
def input_pdfs(
        e3sm_filename='/Users/fier887/Downloads/E3SM_output_for_PartMC_0607_surf.nc',
        Xvar_prefixes = ['N0','mu0'],scls=['log','lin'],mode_cols=['C0','C1','C2','C3'],
        modes = range(1,5), showfliers=False,
        fig_width = 6.5, fig_height = 1.5):
    
    f = Dataset(e3sm_filename)
    
    fig,axs = plt.subplots(1,2)
    df_list = []
    for ii,(Xvar_prefix,scl) in enumerate(zip(Xvar_prefixes,scls)):
        # fig,axs = plt.subplots(1,len(modes),sharey=True)
        # Xdat = np.array([])
        # mode_vals = np.array([])
        
        # ylim_max = 0.
        for jj,(mode,col) in enumerate(zip(modes,mode_cols)):
            X_e3sm_raveled,the_shape = analyses.get_X_e3sm([Xvar_prefix + '_' + str(mode)],f,tt=0,lev=-1,return_shape=True)
            if scl == 'log':
                hist_vals,bin_edges = np.histogram(np.log10(X_e3sm_raveled),100)
                bin_mids = 10.**np.linspace(bin_edges[0] + (bin_edges[1]-bin_edges[0])/2.,bin_edges[-1] - (bin_edges[1]-bin_edges[0])/2., len(bin_edges)-1)                
                kde = KernelDensity(bandwidth=(bin_edges[1]-bin_edges[0])*1.5).fit(np.log10(X_e3sm_raveled))
            elif scl == 'lin':
                hist_vals,bin_edges = np.histogram(X_e3sm_raveled,100)
                bin_mids = np.linspace(bin_edges[0] + (bin_edges[1]-bin_edges[0])/2.,bin_edges[-1] - (bin_edges[1]-bin_edges[0])/2., len(bin_edges)-1)
                kde = KernelDensity(bandwidth=(bin_edges[1]-bin_edges[0])*1.5).fit(X_e3sm_raveled)
            
            axs[ii].plot(10.**bin_edges,np.exp(kde.score_samples(bin_edges.reshape(-1,1))));
        
        # if scl == 'log':
        axs[ii].set_xscale('log')
            
            # if mode == 1:
            #     Xdat = X_e3sm_raveled[:,0]
            #     mode_vals = mode*np.ones_like(X_e3sm_raveled[:,0])
            # else:
            #     Xdat = np.hstack([Xdat,X_e3sm_raveled[:,0]])
            #     mode_vals = np.hstack([mode_vals,mode*np.ones_like(X_e3sm_raveled[:,0])])            
            
            # if Xvar_prefixes == 'mu0':
            #     bin_edges,hist_vals = np.histogram(10.**(X_e3sm_raveled),bins=bin_vals)
            # else:
            #     bin_edges,hist_vals = np.histogram(X_e3sm_raveled,bins=bin_vals)
            
            # ylim_max = max([ylim_max,axs[jj].get_ylim()[-1]])
            # axs[jj].plot(bin_vals,hist_vals);
            # axs[jj].set_xscale('log')
                
    fig.set_size_inches(fig_width,fig_height)
    
def input_pdfs_old(
        e3sm_filename='/Users/fier887/Downloads/E3SM_output_for_PartMC_0607_surf.nc',
        Xvar_prefixes = ['N0','mu0'], bins = [np.logspace(5,12,200),np.logspace(-9,-4,300)], modes = range(1,5), showfliers=False,
        # Xvarnames_plot = [['N0_1','N0_2','N0_3','N0_4','mu0_1','mu0_2','mu0_3','mu0_4']],
        fig_width = 6.5, fig_height = 1.5):
    
    f = Dataset(e3sm_filename)
    
    df_list = []
    for ii,(Xvar_prefix,bin_vals) in enumerate(zip(Xvar_prefixes,bins)):
        fig,axs = plt.subplots(1,len(modes),sharey=True)
        Xdat = np.array([])
        mode_vals = np.array([])
        
        ylim_max = 0.
        for jj,mode in enumerate(modes):
            X_e3sm_raveled,the_shape = analyses.get_X_e3sm([Xvar_prefix + '_' + str(mode)],f,tt=0,lev=-1,return_shape=True)
            
            if mode == 1:
                Xdat = X_e3sm_raveled[:,0]
                mode_vals = mode*np.ones_like(X_e3sm_raveled[:,0])
            else:
                Xdat = np.hstack([Xdat,X_e3sm_raveled[:,0]])
                mode_vals = np.hstack([mode_vals,mode*np.ones_like(X_e3sm_raveled[:,0])])            
            
            if Xvar_prefixes == 'mu0':
                bin_edges,hist_vals = np.histogram(10.**(X_e3sm_raveled),bins=bin_vals)
            else:
                bin_edges,hist_vals = np.histogram(X_e3sm_raveled,bins=bin_vals)
            
            ylim_max = max([ylim_max,axs[jj].get_ylim()[-1]])
            axs[jj].plot(bin_vals,hist_vals);
            axs[jj].set_xscale('log')
                
    fig.set_size_inches(fig_width,fig_height)

def box_model_dsds__all_scenarios(
        ensemble_dir,timesteps,
        fig_width=7.5,fig_height=2.,
        recompute_KL=False,
        lnDs = np.log(np.logspace(-10,-5,100)),backward=False,
        save_fig=True,dpi=200,normalize=True):
    dlnD = lnDs[1] - lnDs[0]
    figsize = [fig_width,fig_height]
    all_dNdlnD_partmc = []
    all_dNdlnD_mam4 = []
    all_KL_repeats = []
    all_Ntot = []
    all_sa_flux = []
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    
    for jj,timestep in enumerate(timesteps):
        tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        lnDs,dNdlnD_partmc,dNdlnD_mam4,KL_repeats = analyses.read_dsds(tt_dir,n_repeat,lnDs=lnDs,backward=backward,recompute_KL=recompute_KL)
        all_dNdlnD_partmc.append(dNdlnD_partmc)
        all_dNdlnD_mam4.append(dNdlnD_mam4)
        all_KL_repeats.append(KL_repeats)
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    for jj in range(all_dNdlnD_partmc[0].shape[0]):
        h2so40 = np.round(ensemble_settings[jj]['gas_init']['H2SO4'],1)
        
        fig,axs = plt.subplots(nrows=1,ncols=len(timesteps),figsize=figsize,sharey=True)
        for tt in range(len(timesteps)):
            if not normalize:
                dNdlnD_partmc_mean = np.mean(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
                dNdlnD_partmc_std = np.std(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
                dNdlnD_mam4 = all_dNdlnD_mam4[tt][jj,:]
            else:
                dndlnD_partmc = np.zeros_like(all_dNdlnD_partmc[tt][jj,:,:])
                for ii in range(n_repeat):
                    dndlnD_partmc[:,ii] = all_dNdlnD_partmc[tt][jj,:,ii]/(np.sum(all_dNdlnD_partmc[tt][jj,:,ii])*dlnD)
                dNdlnD_partmc_mean = np.mean(dndlnD_partmc,axis=1)
                dNdlnD_partmc_std = np.std(dndlnD_partmc,axis=1)
                dNdlnD_mam4 = all_dNdlnD_mam4[tt][jj,:]/(np.sum(all_dNdlnD_mam4[tt][jj,:])*dlnD)
            axs[tt].errorbar(np.exp(lnDs),dNdlnD_partmc_mean,yerr=dNdlnD_partmc_std)
            axs[tt].plot(np.exp(lnDs),dNdlnD_mam4)
            KL = np.mean(all_KL_repeats[tt][jj,:],axis=0)
            axs[tt].set_xscale('log')
            axs[tt].set_xlim([min(np.exp(lnDs)),max(np.exp(lnDs))])
            axs[tt].set_title('KL = ' + str(np.round(KL,2)))
        
        if save_fig:
            save_dir = get_save_dir(ensemble_dir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if not os.path.exists(save_dir + 'dsds2/'):
                os.mkdir(save_dir + 'dsds2/')
            
            fig.savefig(save_dir + 'dsds2/' + str(jj).zfill(6) + '.pdf',dpi=dpi)
            plt.show()
            
def box_model_fccn__all_scenarios(
        ensemble_dir,timesteps, dtime, 
        fig_width=7.5,fig_height=2.,
        save_fig=True,dpi=200):
    figsize = [fig_width,fig_height]
    all_frac_ccn_partmc_mean = []
    all_frac_ccn_partmc_std = []
    all_frac_ccn_mam4 = []
    all_KL_repeats = []
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    
    # all_frac_ccn_partmc_mean = np.zeros([len(timesteps),len(ensemble_settings),151])
    # all_frac_ccn_partmc_std = np.zeros([len(timesteps),len(ensemble_settings),151])
    # all_frac_ccn_mam4 = np.zeros([len(timesteps),len(ensemble_settings),151])
    
    for tt,timestep in enumerate(timesteps):
        tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        s_env_vals, frac_ccn_partmc_mean, frac_ccn_partmc_std, frac_ccn_mam4 =  analyses.read_frac_ccn(tt_dir,n_repeat)
        # plt.plot(s_env_vals,frac_ccn_partmc_mean.transpose()); plt.xscale('log')
        # plt.show()
        # plt.plot(s_env_vals,frac_ccn_mam4.transpose()); plt.xscale('log')
        # plt.show()
        
        # all_frac_ccn_partmc_mean[tt,:,:] = frac_ccn_partmc_mean
        # all_frac_ccn_partmc_std[tt,:,:] = frac_ccn_partmc_std
        # all_frac_ccn_mam4[tt,:,:] = frac_ccn_mam4
        
        all_frac_ccn_partmc_mean.append(frac_ccn_partmc_mean)
        all_frac_ccn_partmc_std.append(frac_ccn_partmc_std)
        all_frac_ccn_mam4.append(frac_ccn_mam4)        
        all_KL_repeats.append(np.loadtxt(tt_dir + 'KL_repeats.txt'))
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    for jj in range(all_frac_ccn_partmc_mean[0].shape[0]):        
        fig,axs = plt.subplots(nrows=1,ncols=len(timesteps),figsize=figsize,sharey=True)
        for tt in range(len(timesteps)):
            frac_ccn_partmc_mean = all_frac_ccn_partmc_mean[tt][jj,:]
            frac_ccn_partmc_std = all_frac_ccn_partmc_std[tt][jj,:]
            frac_ccn_mam4 = all_frac_ccn_mam4[tt][jj,:]
            # axs[tt].errorbar(s_env_vals,frac_ccn_partmc_mean,yerr=frac_ccn_partmc_std)
            axs[tt].plot(s_env_vals,all_frac_ccn_partmc_mean[tt][jj,:])
            axs[tt].plot(s_env_vals,all_frac_ccn_mam4[tt][jj,:])
            KL = np.mean(all_KL_repeats[tt][jj,:],axis=0)
            axs[tt].set_xscale('log')
            axs[tt].set_xlim([1e-3,1e2])
            axs[tt].set_title('t = ' + str(timesteps[tt]*dtime) + '\n KL = ' + str(np.round(KL,2)))
        # plt.show()
        if save_fig:
            save_dir = get_save_dir(ensemble_dir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if not os.path.exists(save_dir + 'frac_ccn/'):
                os.mkdir(save_dir + 'frac_ccn/')
            
            fig.savefig(save_dir + 'frac_ccn/' + str(jj).zfill(6) + '.pdf',dpi=dpi)
            plt.show()
    
def box_model_dsds__regimes(
        ensemble_dir,timesteps,
        regime_type='process_extremes',avg_scale='lin',units='min',
        lnDs = np.log(np.logspace(-10,-5,100)),backward=False,sharey=True,
        fig_width=7.5,fig_height = 5., save_fig=True,recompute_KL=True,
        dpi=500,normalize=True,add_text=False):
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    dlnD = lnDs[1] - lnDs[0]
    figsize = [fig_width,fig_height]
    all_dNdlnD_partmc = []
    all_dNdlnD_mam4 = []
    all_KL_repeats = []
    all_Ntot = []
    all_sa_flux = []
    for jj,timestep in enumerate(timesteps):
        tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        lnDs,dNdlnD_partmc,dNdlnD_mam4,KL_repeats = analyses.read_dsds(tt_dir,n_repeat,lnDs=lnDs,backward=backward)
        all_dNdlnD_partmc.append(dNdlnD_partmc)
        all_dNdlnD_mam4.append(dNdlnD_mam4)
        all_KL_repeats.append(KL_repeats)
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    
    scenario_groups,regime_names = analyses.scenarios_in_regimes(ensemble_dir,timesteps,regime_type=regime_type,avg_scale=avg_scale)
    
    scenario_nums = []
    for scenario_group in scenario_groups:
        scenario_nums.append(int(random.choice(scenario_group)))

    fig,axs = plt.subplots(
        nrows=len(scenario_nums),ncols=len(timesteps),
        figsize=figsize,sharey=sharey,sharex=True)
    
    for kk,jj in enumerate(scenario_nums):
        ylim_max = 0.
        for tt,timestep in enumerate(timesteps):
            if not normalize:
                dNdlnD_partmc_mean = np.mean(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
                dNdlnD_partmc_std = np.std(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
                dNdlnD_mam4 = all_dNdlnD_mam4[tt][jj,:]
            else:
                dndlnD_partmc = np.zeros_like(all_dNdlnD_partmc[tt][jj,:,:])
                for ii in range(n_repeat):
                    dndlnD_partmc[:,ii] = all_dNdlnD_partmc[tt][jj,:,ii]/(np.sum(all_dNdlnD_partmc[tt][jj,:,ii])*dlnD)
                dNdlnD_partmc_mean = np.mean(dndlnD_partmc,axis=1)
                dNdlnD_partmc_std = np.std(dndlnD_partmc,axis=1)
                dNdlnD_mam4 = all_dNdlnD_mam4[tt][jj,:]/(np.sum(all_dNdlnD_mam4[tt][jj,:])*dlnD)
            
            axs[kk,tt].errorbar(1e9*np.exp(lnDs),dNdlnD_partmc_mean,yerr=dNdlnD_partmc_std,color='k')
            axs[kk,tt].plot(1e9*np.exp(lnDs),dNdlnD_mam4,color='C0')
            KL = np.mean(all_KL_repeats[tt][jj,:],axis=0)
            axs[kk,tt].set_xscale('log')
            axs[kk,tt].set_xlim([1e9*1.1*min(np.exp(lnDs)),1e9*max(np.exp(lnDs))])
            
            if kk == 0:
                if units == 'h':
                    axs[kk,tt].set_title(str(int(timestep*dtime/3600.)) + ' h')
                elif units == 'min':
                    axs[kk,tt].set_title(str(int(timestep*dtime/60.)) + ' min')
            if kk == len(scenario_nums)-1 and tt == int((len(timesteps)-1)/2):#tt == 1 and kk == len(scenario_nums)-1:
                axs[kk,tt].set_xlabel('dry diameter [nm]')
            if tt == 0 and kk == np.floor(axs.shape[0]/2):
                if normalize:
                    if axs.shape[0] % 2 == 0:
                        axs[kk,tt].set_ylabel('    normalized number density')
                    else:
                        axs[kk,tt].set_ylabel('normalized number density')
                else:
                    if axs.shape[0] % 2 == 0:
                        axs[kk,tt].set_ylabel('     dN/dlnD [m$^{-3}$]')
                    else:
                        axs[kk,tt].set_ylabel('dN/dlnD [m$^{-3}$]')
            ylim_max = max(np.hstack([ylim_max,axs[kk,tt].get_ylim()]))
        if not sharey:
            for tt in range(axs.shape[1]):
                axs[kk,tt].set_ylim([0.,ylim_max])
                if tt>0.:
                    axs[kk,tt].set_yticklabels('')

    if add_text:
        for kk,jj in enumerate(scenario_nums):#range(all_dNdlnD_partmc[0].shape[0]):
            # h2so40 = np.round(ensemble_settings[jj]['gas_init']['H2SO4'],1)
            for tt,timestep in enumerate(timesteps):
                KL = np.mean(all_KL_repeats[tt][jj,:],axis=0)
                ylims = axs[kk,tt].get_ylim()
                xlims = axs[kk,tt].get_xlim()
                axs[kk,tt].text(1.8*xlims[0],ylims[1]*0.8,'KL = ' + str(np.round(KL,1)))
                
            # axs[kk,tt].set_title('KL = ' + str(np.round(KL,2)))
    if save_fig:
        save_dir = get_save_dir(ensemble_dir)
        fig.savefig(save_dir + 'dsd_regimes.pdf',dpi=dpi,bbox_inches='tight')
        
    # fig.savefig('dsds.pdf',dpi=dpi)
    plt.show()
    return scenario_nums
    
def box_model_dsds__regimes(
        ensemble_dir,timesteps,
        regime_type='process_extremes',avg_scale='lin',units='min',
        lnDs = np.log(np.logspace(-10,-5,100)),backward=False,sharey=True,
        fig_width=7.5,fig_height = 5., save_fig=True,recompute_KL=True,
        dpi=500,normalize=True,add_text=False):
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    dlnD = lnDs[1] - lnDs[0]
    figsize = [fig_width,fig_height]
    all_dNdlnD_partmc = []
    all_dNdlnD_mam4 = []
    all_KL_repeats = []
    all_Ntot = []
    all_sa_flux = []
    for jj,timestep in enumerate(timesteps):
        tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        lnDs,dNdlnD_partmc,dNdlnD_mam4,KL_repeats = analyses.read_dsds(tt_dir,n_repeat,lnDs=lnDs,backward=backward)
        all_dNdlnD_partmc.append(dNdlnD_partmc)
        all_dNdlnD_mam4.append(dNdlnD_mam4)
        all_KL_repeats.append(KL_repeats)
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    
    scenario_groups,regime_names = analyses.scenarios_in_regimes(ensemble_dir,timesteps,regime_type=regime_type,avg_scale=avg_scale)
    
    scenario_nums = []
    for scenario_group in scenario_groups:
        scenario_nums.append(int(random.choice(scenario_group)))

    fig,axs = plt.subplots(
        nrows=len(scenario_nums),ncols=len(timesteps),
        figsize=figsize,sharey=sharey,sharex=True)
    
    for kk,jj in enumerate(scenario_nums):
        ylim_max = 0.
        for tt,timestep in enumerate(timesteps):
            if not normalize:
                dNdlnD_partmc_mean = np.mean(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
                dNdlnD_partmc_std = np.std(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
                dNdlnD_mam4 = all_dNdlnD_mam4[tt][jj,:]
            else:
                dndlnD_partmc = np.zeros_like(all_dNdlnD_partmc[tt][jj,:,:])
                for ii in range(n_repeat):
                    dndlnD_partmc[:,ii] = all_dNdlnD_partmc[tt][jj,:,ii]/(np.sum(all_dNdlnD_partmc[tt][jj,:,ii])*dlnD)
                dNdlnD_partmc_mean = np.mean(dndlnD_partmc,axis=1)
                dNdlnD_partmc_std = np.std(dndlnD_partmc,axis=1)
                dNdlnD_mam4 = all_dNdlnD_mam4[tt][jj,:]/(np.sum(all_dNdlnD_mam4[tt][jj,:])*dlnD)
            
            axs[kk,tt].errorbar(1e9*np.exp(lnDs),dNdlnD_partmc_mean,yerr=dNdlnD_partmc_std,color='k')
            axs[kk,tt].plot(1e9*np.exp(lnDs),dNdlnD_mam4,color='C0')
            KL = np.mean(all_KL_repeats[tt][jj,:],axis=0)
            axs[kk,tt].set_xscale('log')
            axs[kk,tt].set_xlim([1e9*1.1*min(np.exp(lnDs)),1e9*max(np.exp(lnDs))])
            
            if kk == 0:
                if units == 'h':
                    axs[kk,tt].set_title(str(int(timestep*dtime/3600.)) + ' h')
                elif units == 'min':
                    axs[kk,tt].set_title(str(int(timestep*dtime/60.)) + ' min')
            if kk == len(scenario_nums)-1 and tt == int((len(timesteps)-1)/2):#tt == 1 and kk == len(scenario_nums)-1:
                axs[kk,tt].set_xlabel('dry diameter [nm]')
            if tt == 0 and kk == np.floor(axs.shape[0]/2)-1:
                if normalize:
                    if axs.shape[0] % 2 == 0:
                        axs[kk,tt].set_ylabel('    normalized number density')
                    else:
                        axs[kk,tt].set_ylabel('normalized number density')
                else:
                    if axs.shape[0] % 2 == 0:
                        axs[kk,tt].set_ylabel('     dN/dlnD [m$^{-3}$]')
                    else:
                        axs[kk,tt].set_ylabel('dN/dlnD [m$^{-3}$]')
            ylim_max = max(np.hstack([ylim_max,axs[kk,tt].get_ylim()]))
        if not sharey:
            for tt in range(axs.shape[1]):
                axs[kk,tt].set_ylim([0.,ylim_max])
                if tt>0:
                    axs[kk,tt].set_yticklabels('')

    if add_text:
        for kk,jj in enumerate(scenario_nums):#range(all_dNdlnD_partmc[0].shape[0]):
            # h2so40 = np.round(ensemble_settings[jj]['gas_init']['H2SO4'],1)
            for tt,timestep in enumerate(timesteps):
                KL = np.mean(all_KL_repeats[tt][jj,:],axis=0)
                ylims = axs[kk,tt].get_ylim()
                xlims = axs[kk,tt].get_xlim()
                axs[kk,tt].text(1.8*xlims[0],ylims[1]*0.8,'KL = ' + str(np.round(KL,1)))
                
            # axs[kk,tt].set_title('KL = ' + str(np.round(KL,2)))
    if save_fig:
        save_dir = get_save_dir(ensemble_dir)
        fig.savefig(save_dir + 'dsd_regimes.pdf',dpi=dpi,bbox_inches='tight')
        
    # fig.savefig('dsds.pdf',dpi=dpi)
    plt.show()
    return scenario_nums

def box_model_dsds__few(
        ensemble_dir,timesteps,scenario_nums,units='min',sharey=True,
        lnDs = np.log(np.logspace(-10,-5,100)),backward=False,save_fig=True,
        fig_width=7.5,fig_height = 5.,
        dpi=500,normalize=True,add_text=False):
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    dlnD = lnDs[1] - lnDs[0]
    figsize = [fig_width,fig_height]
    all_dNdlnD_partmc = []
    all_dNdlnD_mam4 = []
    all_KL_repeats = []
    all_Ntot = []
    all_sa_flux = []
    for jj,timestep in enumerate(timesteps):
        tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        lnDs,dNdlnD_partmc,dNdlnD_mam4,KL_repeats = analyses.read_dsds(tt_dir,n_repeat,lnDs=lnDs,backward=backward)
        all_dNdlnD_partmc.append(dNdlnD_partmc)
        all_dNdlnD_mam4.append(dNdlnD_mam4)
        all_KL_repeats.append(KL_repeats)
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    # fig,axs = plt.subplots(
    #     nrows=len(scenario_nums),ncols=len(timesteps)+1,
    #     figsize=figsize,sharey=sharey,sharex=True)

    fig,axs = plt.subplots(
        nrows=len(scenario_nums)+1,ncols=len(timesteps),
        figsize=figsize,sharey=sharey,sharex=True)    
    # xlim_vals = []
    for kk,jj in enumerate(scenario_nums):#range(all_dNdlnD_partmc[0].shape[0]):
        ylim_max = 0.
        # h2so40 = np.round(ensemble_settings[jj]['gas_init']['H2SO4'],1)
        for tt,timestep in enumerate(timesteps):
            if not normalize:
                dNdlnD_partmc_mean = np.mean(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
                dNdlnD_partmc_std = np.std(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
                dNdlnD_mam4 = all_dNdlnD_mam4[tt][jj,:]
            else:
                dndlnD_partmc = np.zeros_like(all_dNdlnD_partmc[tt][jj,:,:])
                for ii in range(n_repeat):
                    dndlnD_partmc[:,ii] = all_dNdlnD_partmc[tt][jj,:,ii]/(np.sum(all_dNdlnD_partmc[tt][jj,:,ii])*dlnD)
                dNdlnD_partmc_mean = np.mean(dndlnD_partmc,axis=1)
                dNdlnD_partmc_std = np.std(dndlnD_partmc,axis=1)
                dNdlnD_mam4 = all_dNdlnD_mam4[tt][jj,:]/(np.sum(all_dNdlnD_mam4[tt][jj,:])*dlnD)
            
            # axs[kk,tt].errorbar(1e6*np.exp(lnDs),dNdlnD_partmc_mean,yerr=dNdlnD_partmc_std,color='k',linewidth=2.,linestyle='--')
            axs[kk+1,tt].plot(1e6*np.exp(lnDs),dNdlnD_partmc_mean,color='k',linewidth=2.,linestyle='--',label='PartMC-MOSAIC')
            axs[kk+1,tt].plot(1e6*np.exp(lnDs),dNdlnD_mam4,color='C0',linewidth=2.,label='MAM4')
            KL = np.mean(all_KL_repeats[tt][jj,:],axis=0)
            axs[kk+1,tt].set_xscale('log')
            axs[kk+1,tt].set_xlim([1e6*11.*min(np.exp(lnDs)),0.05*1e6*max(np.exp(lnDs))])
            
            if kk == 0:
                if units == 'h':
                    axs[kk+1,tt].set_title(str(int(timestep*dtime/3600.)) + ' h',fontdict={'size':12})
                elif units == 'min':
                    axs[kk+1,tt].set_title(str(int(timestep*dtime/60.)) + ' min',fontdict={'size':12})
                # axs[kk,tt].set_title(str(int(timestep*dtime/3600.)) + ' h')
            if kk == len(scenario_nums)-1 and tt == int((len(timesteps)-1)/2):#tt == 1 and kk == len(scenario_nums)-1:
                if int((len(timesteps)-1)/2) == tt:
                    axs[kk+1,tt].set_xlabel('dry diameter [$\mu$m]',fontdict={'size':12})
                else:
                    axs[kk+1,tt].set_xlabel('dry diameter [$\mu$m]',fontdict={'size':12})
            if tt == 0 and kk == np.floor(axs.shape[0]/2) - 1:
                if normalize:
                    axs[kk+1,tt].set_ylabel('normalized number density\n',fontdict={'size':14})
                else:
                    axs[kk+1,tt].set_ylabel('dN/dlnD [m$^{-3}$]',fontdict={'size':14})
            
            ylim_max = max(np.hstack([ylim_max,axs[kk,tt].get_ylim()]))
        # fig.supxlabel('dry diameter [$\mu$m]',fontdict={'size':12})
            # axs[kk,tt].set_xlim(xlim_vals)
        if not sharey:
            for tt in range(axs.shape[1]):
                axs[kk+1,tt].set_ylim([0.,ylim_max])
                if tt>0.:
                    axs[kk+1,tt].set_yticklabels('')
    if add_text:
        for kk,jj in enumerate(scenario_nums):#range(all_dNdlnD_partmc[0].shape[0]):
            # h2so40 = np.round(ensemble_settings[jj]['gas_init']['H2SO4'],1)
            for tt,timestep in enumerate(timesteps):
                KL = np.mean(all_KL_repeats[tt][jj,:],axis=0)
                ylims = axs[kk,tt].get_ylim()
                axs[kk+1,tt].text(1e9*1.4*min(np.exp(lnDs)),ylims[1]*0.75,'KL = ' + str(np.round(KL,1)))
    kk = 0;
    tt = 0;
    hln1 = axs[kk,tt].plot(1e6*np.exp(lnDs),dNdlnD_partmc_mean,color='k',linewidth=2.,linestyle='--',label='PartMC-MOSAIC')
    hln2 = axs[kk,tt].plot(1e6*np.exp(lnDs),dNdlnD_mam4,color='C0',linewidth=2.,label='MAM4')
    # axs[kk,tt].set_xlim([100.,1000.])
    
    axs[kk,tt].legend(loc='lower left',frameon=False,bbox_to_anchor=(axs[kk,tt].get_xlim()[0],0.4*axs[kk,tt].get_ylim()[1]),ncol=2)
    hln1[0].remove()
    hln2[0].remove()
    # axs[kk,tt].legend(loc='lower left',bbox_to_anchor=(axs[kk,tt].get_xlim()[-1]*2.,axs[kk,tt].get_ylim()[-1]*0.3),frameon=False)
    # for ii in range(axs.shape[0]):
    #     axs[ii,-1].axis('off')
    #     axs[ii,-1].axis('off')
    for ii in range(axs.shape[1]):
        axs[0,ii].axis('off')
        axs[0,ii].axis('off')
    
    for kk in range(axs.shape[0]):
        if kk>0:
            axs[kk,0].set_yticks([0.,0.5])
    
    if save_fig:
        save_dir = get_save_dir(ensemble_dir)
        fig.savefig(save_dir + 'dsd_few.pdf',dpi=dpi,bbox_inches='tight')
        
            # axs[kk,tt].set_title('KL = ' + str(np.round(KL,2)))
    # fig.savefig('dsds.pdf',dpi=dpi)
    plt.show()

def box_model_fccn__few(
        ensemble_dir,timesteps,scenario_nums,units='min',sharey=True,
        backward=True,save_fig=True,
        fig_width=7.5,fig_height = 5.,
        dpi=500,add_text=False):
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    figsize = [fig_width,fig_height]
    
    all_frac_ccn_partmc_mean = []
    all_frac_ccn_partmc_std = []
    all_frac_ccn_mam4 = []
    all_KL_repeats = []
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    
    for jj,timestep in enumerate(timesteps):
        tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        s_env_vals, frac_ccn_partmc_mean, frac_ccn_partmc_std, frac_ccn_mam4 =  analyses.read_frac_ccn(tt_dir,n_repeat)
        all_frac_ccn_partmc_mean.append(frac_ccn_partmc_mean)
        all_frac_ccn_partmc_std.append(frac_ccn_partmc_std)
        all_frac_ccn_mam4.append(frac_ccn_mam4)        
        all_KL_repeats.append(np.loadtxt(tt_dir + 'KL_repeats.txt'))
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    fig,axs = plt.subplots(
        nrows=len(scenario_nums)+1,ncols=len(timesteps),
        figsize=figsize,sharey=sharey,sharex=True)
    
    for kk,jj in enumerate(scenario_nums):#range(all_dNdlnD_partmc[0].shape[0]):
        for tt,timestep in enumerate(timesteps):
            frac_ccn_partmc_mean = all_frac_ccn_partmc_mean[tt][jj,:] #np.mean(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
            frac_ccn_partmc_std = all_frac_ccn_partmc_std[tt][jj,:] #np.std(all_dNdlnD_partmc[tt][jj,:,:],axis=1)
            frac_ccn_mam4 = all_frac_ccn_mam4[tt][jj,:]
            
            axs[kk+1,tt].plot(s_env_vals,frac_ccn_partmc_mean,color='k',linewidth=2.,linestyle='--',label='PartMC-MOSAIC')
            axs[kk+1,tt].plot(s_env_vals,frac_ccn_mam4,color='C0',linewidth=2.,label='MAM4')
            axs[kk+1,tt].set_xscale('log')
            #axs[kk,tt].set_xlim([min(s_env_vals),max(s_env_vals)])
            axs[kk+1,tt].set_xlim([0.2e-1,50.])
            if kk == 0:
                if units == 'h':
                    axs[kk+1,tt].set_title(str(int(timestep*dtime/3600.)) + ' h',fontdict={'size':12})
                elif units == 'min':
                    axs[kk+1,tt].set_title(str(int(timestep*dtime/60.)) + ' min',fontdict={'size':12})
            if kk == len(scenario_nums)-1 and tt == int((len(timesteps)-1)/2):
                if int((len(timesteps)-1)/2) == tt:
                    axs[kk+1,tt].set_xlabel('supersaturation [%]',fontdict={'size':12})
                else:
                    axs[kk+1,tt].set_xlabel('supersaturation [%]',fontdict={'size':12})
            if tt == 0 and kk == np.floor(axs.shape[0]/2)-1:
                axs[kk+1,tt].set_ylabel('fraction activated\n',fontdict={'size':14})
            axs[kk+1,tt].set_ylim([0.,1.])
            
            
            axs[kk+1,tt].set_yticklabels('')
    
    # kk = 0
    # tt = len(timesteps)-1
    
    kk = 0;
    tt = 0;
    hln1 = axs[kk,tt].plot(s_env_vals,frac_ccn_partmc_mean,color='k',linewidth=2.,linestyle='--',label='PartMC-MOSAIC')
    hln2 = axs[kk,tt].plot(s_env_vals,frac_ccn_mam4,color='C0',linewidth=2.,label='MAM4')
    # axs[kk,tt].set_xlim([100.,1000.])
    
    axs[kk,tt].legend(loc='lower left',frameon=False,bbox_to_anchor=(axs[kk,tt].get_xlim()[0],0.4*axs[kk,tt].get_ylim()[1]),ncol=2)
    hln1[0].remove()
    hln2[0].remove()
    
    # axs[kk,tt].legend(loc='lower left',frameon=False,bbox_to_anchor=(axs[kk,tt].get_xlim()[-1],axs[kk,tt].get_ylim()[-1]*1.5))
    # for ii in range(axs.shape[0]):
    #     axs[ii,-1].axis('off')
    #     axs[ii,-1].axis('off')
    
    for ii in range(axs.shape[0]):
        if ii>0:
            axs[ii,0].set_yticks([0.5,1.])
            axs[ii,0].set_yticklabels([0.5,1.])
    
    for ii in range(axs.shape[1]):
        axs[0,ii].axis('off')
        axs[0,ii].axis('off')
    if save_fig:
        save_dir = get_save_dir(ensemble_dir)
        fig.savefig(save_dir + 'frac_ccn_few.pdf',dpi=dpi,bbox_inches='tight')
    plt.show()


def KL_vs_aging(
        ensemble_prefix,timesteps,showfliers=False,recompute_KL=True,
        Nbins=10,save_fig=True,dpi=500.,fig_width=7.5,fig_height=3.,whis=(5,95)):
    ensemble_dir__cond = '/Users/fier887/Downloads/box_simulations3/' + ensemble_prefix + '_b_cond/'
    ensemble_dir__coag = '/Users/fier887/Downloads/box_simulations3/' + ensemble_prefix + '_c_coag/'
    ensemble_dir = '/Users/fier887/Downloads/box_simulations3/' + ensemble_prefix + '_a_both/'
    
    KL_cond = []
    KL_coag = []
    integrated_Ntot = []
    integrated_sa_flux = []
    all_mus_cond = []
    all_mus_coag = []
    all_scenarios = []
    all_timesteps = []
    
    all_N1 = []
    all_N2 = []
    all_N3 = []
    all_N4 = []
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    
    for timestep in timesteps:
        time = timestep*dtime
        tt_dir__cond = ensemble_dir__cond + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        tt_dir__coag = ensemble_dir__coag + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        
        integrated_sa_flux = np.hstack([integrated_sa_flux,1e9*analyses.read_boxmodel_data('sa_accumulated',tt_dir__cond,time)])
        KL_cond = np.hstack([KL_cond,analyses.read_boxmodel_data('KL_backward',tt_dir__cond,time,recompute_KL=recompute_KL)])

        integrated_Ntot = np.hstack([integrated_Ntot,1e-6/3600.*analyses.read_boxmodel_data('integrated_Ntot',tt_dir__coag,time)])
        one_KL = analyses.read_boxmodel_data('KL_backward',tt_dir__coag,time,recompute_KL=recompute_KL)
        KL_coag = np.hstack([KL_coag,one_KL])
        
        all_mus_cond = np.hstack([all_mus_cond,analyses.read_boxmodel_data('mu_overall',tt_dir__cond,time)])
        all_mus_coag = np.hstack([all_mus_coag,analyses.read_boxmodel_data('mu_overall',tt_dir__coag,time)])
        all_N1 = np.hstack([all_N1,analyses.read_boxmodel_data('N0_1',tt_dir__cond,time)])
        all_N2 = np.hstack([all_N2,analyses.read_boxmodel_data('N0_2',tt_dir__cond,time)])
        all_N3 = np.hstack([all_N3,analyses.read_boxmodel_data('N0_3',tt_dir__cond,time)])
        all_N4 = np.hstack([all_N4,analyses.read_boxmodel_data('N0_4',tt_dir__cond,time)])
        
        all_scenarios.append(range(1,len(one_KL)+1))
        all_timesteps.append(timestep*np.ones_like(one_KL))
        
    integratedNtot_bins = np.logspace(0.99*np.log10(min(integrated_Ntot[integrated_Ntot>0.])),1.01*np.log10(max(integrated_Ntot)),Nbins)
    integratedSA_bins = np.logspace(0.99*np.log10(min(integrated_sa_flux[integrated_sa_flux>0.])),1.01*np.log10(max(integrated_sa_flux)),Nbins)
    
    bins_integratedNtot = np.nan*np.ones_like(integrated_Ntot)
    bins_integratedSA = np.nan*np.ones_like(integrated_Ntot)
    
    midpoints_integratedNtot = np.nan*np.ones_like(integrated_Ntot)
    midpoints_integratedSA = np.nan*np.ones_like(integrated_Ntot)
    
    num_in_Ntot_bin = np.zeros_like(integratedNtot_bins[:-1])
    num_in_SA_bin = np.zeros_like(integratedNtot_bins[:-1])
    
    for ii,(low_end,high_end) in enumerate(zip(integratedNtot_bins[:-1],integratedNtot_bins[1:])):
        idx, = np.where((integrated_Ntot>=low_end) & (integrated_Ntot<high_end))
        mid_point = (low_end + high_end)/2.
        midpoints_integratedNtot[idx] = mid_point
        bins_integratedNtot[idx] = ii
        num_in_Ntot_bin[ii] = len(idx)
    
    if num_in_Ntot_bin[-1]/sum(num_in_Ntot_bin)<0.01:
        integratedNtot_bins = np.logspace(np.log10(integratedNtot_bins[0]),np.log10(integratedNtot_bins[-2]),Nbins)
        for ii,(low_end,high_end) in enumerate(zip(integratedNtot_bins[:-1],integratedNtot_bins[1:])):
            idx, = np.where((integrated_Ntot>=low_end) & (integrated_Ntot<high_end))
            mid_point = (low_end + high_end)/2.
            midpoints_integratedNtot[idx] = mid_point
            bins_integratedNtot[idx] = ii
            num_in_Ntot_bin[ii] = len(idx)
    
    for ii,(low_end,high_end) in enumerate(zip(integratedSA_bins[:-1],integratedSA_bins[1:])):
        idx, = np.where((integrated_sa_flux>=low_end) & (integrated_sa_flux<high_end))
        mid_point = (low_end + high_end)/2.
        bins_integratedSA[idx] = ii
        midpoints_integratedSA[idx] = mid_point
        num_in_SA_bin[ii] = len(idx)
        
    # fig = plt.figure()
    fig,axs = plt.subplots(1,2,sharey=True)
    fig.set_size_inches(fig_width,fig_height)
    
    data = np.array([bins_integratedNtot,bins_integratedSA,np.log10(midpoints_integratedNtot),np.log10(midpoints_integratedSA),KL_cond,KL_coag]).transpose()
    df = pandas.DataFrame(data=data,columns=['bins_integratedNtot','bins_integratedSA','midpoints_integratedNtot','midpoints_integratedSA','KL_cond','KL_coag'])
    #fig,ax = plt.subplots(1)
    # ax = fig.add_subplot(1,2,1); 
    # ax.set_xscale('log')
    sns.boxplot(data=df,x='bins_integratedNtot',y='KL_coag',color='C0',ax=axs[0], showfliers=showfliers, whis=whis)
    # sns.boxplot(data=df,x='midpoints_integratedNtot',y='KL_coag',color='C0')
    # Ntot_lims = [3,axs[0].get_xlim()[1]]
    Ntot_lims = axs[0].get_xlim()
    Ntot_ticklabs = np.logspace(3,9,7)
    Ntot_ticks = get_ticks(bins_integratedNtot[~np.isnan(bins_integratedNtot)]-min(bins_integratedNtot[~np.isnan(bins_integratedNtot)]),midpoints_integratedNtot[~np.isnan(bins_integratedNtot)],Ntot_ticklabs,scale='log')
    # Ntot_ticks = get_ticks(midpoints_integratedNtot[~np.isnan(bins_integratedNtot)],midpoints_integratedNtot[~np.isnan(bins_integratedNtot)],Ntot_ticklabs,scale='log')
    
    Ntot_ticks_text = []
    for tick in Ntot_ticklabs:
        Ntot_ticks_text.append('10$^{' + str(int(np.log10(tick))) + '}$')
    
    plt.sca(axs[0])
    axs[0].set_xticks(Ntot_ticks)
    axs[0].set_xticklabels(Ntot_ticks_text,fontdict={'size':10})
    axs[0].set_yticklabels(np.round(axs[0].get_yticks(),2),fontdict={'size':10})
    axs[0].set_xlim(Ntot_lims)
    axs[0].set_xlabel('integrated number \n concentration [cm$^{-3}$h]',fontdict={'size':12})
    axs[0].set_ylabel('KL-divergence',fontdict={'size':12})
    
    sns.boxplot(data=df,x='bins_integratedSA',y='KL_cond',color='C0',ax=axs[1],showfliers=showfliers)
    SA_lims = axs[1].get_xlim()
    SA_ticklabs = np.logspace(-3,2,6)
    SA_ticks = get_ticks(bins_integratedSA[~np.isnan(bins_integratedSA)]-min(bins_integratedSA[~np.isnan(bins_integratedSA)]),midpoints_integratedSA[~np.isnan(bins_integratedSA)],SA_ticklabs,scale='log')
    SA_ticks_text = []
    for tick in SA_ticklabs:
        SA_ticks_text.append('10$^{' + str(int(np.log10(tick))) + '}$')
    
    axs[1].set_xticks(SA_ticks)
    axs[1].set_xticklabels(SA_ticks_text,fontdict={'size':10})
    axs[1].set_yticklabels(np.round(axs[1].get_yticks(),2),fontdict={'size':10})
    axs[1].set_xlim(SA_lims)
    axs[1].set_xlabel('integrated condensation\ngrowth rate [nm]',fontdict={'size':12})
    axs[1].set_ylabel('')
    
    axs[0].set_title('coagulation only')
    axs[1].set_title('gas condensation only')
    
    if save_fig:
        save_dir = get_save_dir(ensemble_dir)
        fig.savefig(save_dir + 'KL_aging.pdf',dpi=dpi,bbox_inches='tight')

def s_thresh_vs_aging(
        ensemble_prefix,timesteps,showfliers=False,recompute_KL=True, thresh = 0.01,
        Nbins=10,save_fig=True,dpi=500.,fig_width=7.5,fig_height=3.,whis=(5,95)):
    ensemble_dir__cond = '/Users/fier887/Downloads/box_simulations3/' + ensemble_prefix + '_b_cond/'
    ensemble_dir__coag = '/Users/fier887/Downloads/box_simulations3/' + ensemble_prefix + '_c_coag/'
    ensemble_dir = '/Users/fier887/Downloads/box_simulations3/' + ensemble_prefix + '_a_both/'
    
    s_onset_ratio_cond = []
    s_onset_ratio_coag = []
    integrated_Ntot = []
    integrated_sa_flux = []
    all_mus_cond = []
    all_mus_coag = []
    all_scenarios = []
    all_timesteps = []
    
    all_N1 = []
    all_N2 = []
    all_N3 = []
    all_N4 = []
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    
    for timestep in timesteps:
        time = timestep*dtime
        tt_dir__cond = ensemble_dir__cond + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        tt_dir__coag = ensemble_dir__coag + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        
        integrated_sa_flux = np.hstack([integrated_sa_flux,1e9*analyses.read_boxmodel_data('sa_accumulated',tt_dir__cond,time)])
        one_s_onset = analyses.read_boxmodel_data('s_onset_ratio',tt_dir__cond,time,recompute_KL=recompute_KL,thresh=thresh)
        s_onset_ratio_cond = np.hstack([s_onset_ratio_cond,one_s_onset])

        integrated_Ntot = np.hstack([integrated_Ntot,1e-6/3600.*analyses.read_boxmodel_data('integrated_Ntot',tt_dir__coag,time)])
        s_onset_ratio_coag = np.hstack([s_onset_ratio_coag,analyses.read_boxmodel_data('s_onset_mam4',tt_dir__coag,time,recompute_KL=recompute_KL,thresh=thresh)])
        
        all_mus_cond = np.hstack([all_mus_cond,analyses.read_boxmodel_data('mu_overall',tt_dir__cond,time)])
        all_mus_coag = np.hstack([all_mus_coag,analyses.read_boxmodel_data('mu_overall',tt_dir__coag,time)])
        all_N1 = np.hstack([all_N1,analyses.read_boxmodel_data('N0_1',tt_dir__cond,time)])
        all_N2 = np.hstack([all_N2,analyses.read_boxmodel_data('N0_2',tt_dir__cond,time)])
        all_N3 = np.hstack([all_N3,analyses.read_boxmodel_data('N0_3',tt_dir__cond,time)])
        all_N4 = np.hstack([all_N4,analyses.read_boxmodel_data('N0_4',tt_dir__cond,time)])
        
        all_scenarios.append(range(1,len(one_s_onset)+1))
        all_timesteps.append(timestep*np.ones_like(one_s_onset))
        
    integratedNtot_bins = np.logspace(0.99*np.log10(min(integrated_Ntot[integrated_Ntot>0.])),1.01*np.log10(max(integrated_Ntot)),Nbins)
    integratedSA_bins = np.logspace(0.99*np.log10(min(integrated_sa_flux[integrated_sa_flux>0.])),1.01*np.log10(max(integrated_sa_flux)),Nbins)
    
    bins_integratedNtot = np.nan*np.ones_like(integrated_Ntot)
    bins_integratedSA = np.nan*np.ones_like(integrated_Ntot)
    
    midpoints_integratedNtot = np.nan*np.ones_like(integrated_Ntot)
    midpoints_integratedSA = np.nan*np.ones_like(integrated_Ntot)
    
    num_in_Ntot_bin = np.zeros_like(integratedNtot_bins[:-1])
    num_in_SA_bin = np.zeros_like(integratedNtot_bins[:-1])
    
    for ii,(low_end,high_end) in enumerate(zip(integratedNtot_bins[:-1],integratedNtot_bins[1:])):
        idx, = np.where((integrated_Ntot>=low_end) & (integrated_Ntot<high_end))
        mid_point = (low_end + high_end)/2.
        midpoints_integratedNtot[idx] = mid_point
        bins_integratedNtot[idx] = ii
        num_in_Ntot_bin[ii] = len(idx)
    
    if num_in_Ntot_bin[-1]/sum(num_in_Ntot_bin)<0.01:
        integratedNtot_bins = np.logspace(np.log10(integratedNtot_bins[0]),np.log10(integratedNtot_bins[-2]),Nbins)
        for ii,(low_end,high_end) in enumerate(zip(integratedNtot_bins[:-1],integratedNtot_bins[1:])):
            idx, = np.where((integrated_Ntot>=low_end) & (integrated_Ntot<high_end))
            mid_point = (low_end + high_end)/2.
            midpoints_integratedNtot[idx] = mid_point
            bins_integratedNtot[idx] = ii
            num_in_Ntot_bin[ii] = len(idx)
    
    for ii,(low_end,high_end) in enumerate(zip(integratedSA_bins[:-1],integratedSA_bins[1:])):
        idx, = np.where((integrated_sa_flux>=low_end) & (integrated_sa_flux<high_end))
        mid_point = (low_end + high_end)/2.
        bins_integratedSA[idx] = ii
        midpoints_integratedSA[idx] = mid_point
        num_in_SA_bin[ii] = len(idx)
        
    # fig = plt.figure()
    fig,axs = plt.subplots(1,2,sharey=True)
    fig.set_size_inches(fig_width,fig_height)
    
    # data = np.array([bins_integratedNtot,bins_integratedSA,np.log10(midpoints_integratedNtot),np.log10(midpoints_integratedSA),s_onset_ratio_cond,s_onset_ratio_coag]).transpose()
    # df = pandas.DataFrame(data=data,columns=['bins_integratedNtot','bins_integratedSA','midpoints_integratedNtot','midpoints_integratedSA','s_onset_ratio_cond','s_onset_ratio_coag'])

    data = np.array([bins_integratedNtot,bins_integratedSA,np.log10(midpoints_integratedNtot),np.log10(midpoints_integratedSA),s_onset_ratio_cond,s_onset_ratio_coag]).transpose()
    df = pandas.DataFrame(data=data,columns=['bins_integratedNtot','bins_integratedSA','midpoints_integratedNtot','midpoints_integratedSA','s_onset_ratio_cond','s_onset_ratio_coag'])    
    sns.boxplot(data=df,x='bins_integratedNtot',y='s_onset_ratio_coag',color='C0',ax=axs[0], showfliers=showfliers, whis=whis)
    # sns.boxplot(data=df,x='midpoints_integratedNtot',y='KL_coag',color='C0')
    Ntot_lims = axs[0].get_xlim()
    Ntot_ticklabs = np.logspace(3,9,7)
    Ntot_ticks = get_ticks(bins_integratedNtot[~np.isnan(bins_integratedNtot)]-min(bins_integratedNtot[~np.isnan(bins_integratedNtot)]),midpoints_integratedNtot[~np.isnan(bins_integratedNtot)],Ntot_ticklabs,scale='log')
    # Ntot_ticks = get_ticks(midpoints_integratedNtot[~np.isnan(bins_integratedNtot)],midpoints_integratedNtot[~np.isnan(bins_integratedNtot)],Ntot_ticklabs,scale='log')
    
    Ntot_ticks_text = []
    for tick in Ntot_ticklabs:
        Ntot_ticks_text.append('10$^{' + str(int(np.log10(tick))) + '}$')
    
    plt.sca(axs[0])
    axs[0].set_xticks(Ntot_ticks)
    axs[0].set_xticklabels(Ntot_ticks_text,fontdict={'size':10})
    axs[0].set_yticklabels(np.round(axs[0].get_yticks(),2),fontdict={'size':10})
    axs[0].set_xlim(Ntot_lims)
    axs[0].set_xlabel('integrated number \n concentration [cm$^{-3}$h]',fontdict={'size':12})
    axs[0].set_ylabel('$s_{\mathrm{' + str(int(thresh*100)) + ',MAM4}}/s_{\mathrm{' + str(int(thresh*100)) + ',PartMC-MOSAIC}}$',fontdict={'size':12})
    
    # if save_fig:
    #     save_dir = get_save_dir(ensemble_dir__cond)
    #     fig.savefig(save_dir + 'KL_aging_coag.pdf',dpi=500,bbox_inches='tight')

    
    # fig2,ax2 = plt.subplots(1)
    # plt.sca(axs[1])#fig.add_subplot(1,2,2)
    sns.boxplot(data=df,x='bins_integratedSA',y='s_onset_ratio_cond',color='C0',ax=axs[1],showfliers=showfliers)
    SA_lims = axs[1].get_xlim()
    # SA_ticklabs = np.logspace(-2,1,4)
    SA_ticklabs = np.logspace(-3,2,6)
    SA_ticks = get_ticks(bins_integratedSA[~np.isnan(bins_integratedSA)]-min(bins_integratedSA[~np.isnan(bins_integratedSA)]),midpoints_integratedSA[~np.isnan(bins_integratedSA)],SA_ticklabs,scale='log')
    SA_ticks_text = []
    for tick in SA_ticklabs:
        SA_ticks_text.append('10$^{' + str(int(np.log10(tick))) + '}$')
    
    axs[1].set_xticks(SA_ticks)
    axs[1].set_xticklabels(SA_ticks_text,fontdict={'size':10})
    axs[1].set_yticklabels(np.round(axs[1].get_yticks(),2),fontdict={'size':10})
    axs[1].set_xlim(SA_lims)
    axs[1].set_xlabel('integrated condensation\ngrowth rate [nm]',fontdict={'size':12})
    axs[1].set_ylabel('')
    
    axs[0].set_title('coagulation only')
    axs[1].set_title('gas condensation only')
    
    if save_fig:
        save_dir = get_save_dir(ensemble_dir)
        fig.savefig(save_dir + 's' + str(thresh*100).zfill(2) + '_aging.pdf',dpi=dpi,bbox_inches='tight')
    
def get_ticks(bin_vals,midpoint_vals,ticklabs,scale='log'):
    min_bin = min(bin_vals)
    max_bin = max(bin_vals)
    
    if scale == 'lin':
        min_midpoint = min(midpoint_vals)
        max_midpoint = max(midpoint_vals)
    elif scale == 'log':
        min_midpoint = np.log10(min(midpoint_vals))
        max_midpoint = np.log10(max(midpoint_vals))
    
    m = (max_midpoint - min_midpoint)/(max_bin - min_bin)
    b = min_midpoint - m*min_bin
    
    if scale == 'lin':
        ticks = (ticklabs - b)/m
    elif scale == 'log':
        ticks = (np.log10(ticklabs) - b)/m
    print(b,m)
    
    return ticks


 
def KL_vs_time__few(
        ensemble_prefix,scenario_nums,timesteps,
        processes = ['both','cond','coag'],labs=['a','b','c'],
        unit = 'min',dpi=500.,fig_size=[3.5,7.5],
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        backward=False,recompute_KL=False,save_fig=True):
    
    ensemble_dir = ensemble_over_dir + ensemble_prefix + '_a_both/'
    one_KL = np.loadtxt(ensemble_dir + 'processed_output/tt' + str(timesteps[0]).zfill(4) + '/KL_repeats.txt')
    
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    
    n_repeat = one_KL.shape[1]
    n_scenarios = one_KL.shape[0]
    KL = np.zeros((len(timesteps),) + one_KL.shape + (len(processes),))
    KL_recomputed = np.zeros_like(KL)
    
    for tt,timestep in enumerate(timesteps):
        for pp,(process,lab) in enumerate(zip(processes,labs)):
            tt_dir = ensemble_over_dir + ensemble_prefix + '_' + lab + '_' + process + '/processed_output/tt' + str(timestep).zfill(4) + '/'
            dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
            for ii in range(n_repeat):
                dNdlnD_partmc = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii + 1).zfill(4) + '.txt')
                for ss in range(n_scenarios):
                    KL_recomputed[tt,ss,ii,pp] = analyses.get_KL_binned(dNdlnD_partmc[ss,:],dNdlnD_mam4[ss,:],backward=backward)
            if recompute_KL:
                dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
                for ii in range(n_repeat):
                    dNdlnD_partmc = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii + 1).zfill(4) + '.txt')
                    for ss in range(n_scenarios):
                        KL[tt,ss,ii,pp] = analyses.get_KL_binned(dNdlnD_partmc[ss,:],dNdlnD_mam4[ss,:],backward=backward)
            else:
                if backward:
                    KL[tt,:,:,pp] = np.loadtxt(tt_dir + 'KL_repeats_backward.txt')
                else:
                    KL[tt,:,:,pp] = np.loadtxt(tt_dir + 'KL_repeats.txt')
    
    time_seconds = np.array([float(timestep*dtime) for timestep in timesteps])
    if len(processes)==1:
        fig,ax = plt.subplots(nrows=1,ncols=1,sharex=True)
    
        if unit == 'min':
            for ss in scenario_nums:#range(KL.shape[1]):
                ax.errorbar(time_seconds/60.,np.mean(KL[:,ss,:,pp],axis=1),yerr=np.std(KL[:,ss,:,pp],axis=1),label=process,linewidth=3.)
            ax.set_xlim([min(time_seconds/60.),max(time_seconds/60.)])
            ax.set_xlabel('time [min]')
        elif unit == 'h':
            for ss in scenario_nums:#range(KL.shape[1]):
                ax.errorbar(time_seconds/3600.,np.mean(KL[:,ss,:,pp],axis=1),yerr=np.std(KL[:,ss,:,pp],axis=1),label=process,linewidth=3.)
            ax.set_xlim([min(time_seconds/3600.),max(time_seconds/3600.)])
            ax.set_xlabel('time [h]')
        else:
            for ss in scenario_nums:#range(KL.shape[1]):
                ax.errorbar(time_seconds,np.mean(KL[:,ss,:,pp],axis=1),yerr=np.std(KL[:,ss,:,pp],axis=1),label=process,linewidth=3.)
            ax.set_xlim([min(time_seconds),max(time_seconds)])
            ax.set_xlabel('time [s]')
        
        ax.set_xticklabels([int(xtick) for xtick in ax.get_xticks()],fontsize=12)
        ax.set_yticklabels(np.round(ax.get_yticks(),decimals=1),fontsize=12)
        ax.set_xlabel('time [min]',fontsize=12)
        ax.set_ylabel('KL-divergence',fontsize=12)
        save_dir = get_save_dir(ensemble_dir)
        if save_fig:
            fig.savefig(save_dir + 'KL_vs_time__few_oneprocess.pdf', dpi=dpi)
    else:
        fig,axs = plt.subplots(nrows=len(scenario_nums),ncols=1,sharex=True,figsize=fig_size)
        
        for pp in range(len(processes)):
            if unit == 'min':
                for s_idx,ss in enumerate(scenario_nums):#range(KL.shape[1]):
                    axs[s_idx].errorbar(time_seconds/60.,np.mean(KL[:,ss,:,pp],axis=1),yerr=np.std(KL[:,ss,:,pp],axis=1),label=process,linewidth=3.)
                axs[s_idx].set_xlim([min(time_seconds/60.),max(time_seconds/60.)])
                axs[s_idx].set_xlabel('time [min]')
            elif unit == 'h':
                for s_idx,ss in enumerate(scenario_nums):#range(KL.shape[1]):
                    axs[s_idx].errorbar(time_seconds/3600.,np.mean(KL[:,ss,:,pp],axis=1),yerr=np.std(KL[:,ss,:,pp],axis=1),label=process,linewidth=3.)
                axs[s_idx].set_xlim([min(time_seconds/3600.),max(time_seconds/3600.)])
                axs[s_idx].set_xlabel('time [h]')
            else:
                for s_idx,ss in enumerate(scenario_nums):#range(KL.shape[1]):
                    axs[s_idx].errorbar(time_seconds,np.mean(KL[:,ss,:,pp],axis=1),yerr=np.std(KL[:,ss,:,pp],axis=1),label=process,linewidth=3.)
                axs[s_idx].set_xlim([min(time_seconds),max(time_seconds)])
                axs[s_idx].set_xlabel('time [s]')

            for s_idx,ss in enumerate(scenario_nums):#range(KL.shape[1]):            
                axs[s_idx].set_xticklabels([int(xtick) for xtick in axs[s_idx].get_xticks()],fontsize=12)
                axs[s_idx].set_yticklabels(np.round(axs[s_idx].get_yticks(),decimals=1),fontsize=12)
            if tt == len(timesteps):
                ax.set_xlabel('time [min]',fontsize=12)
           
        axs[int(np.floor(len(scenario_nums)/2.))].set_ylabel('KL-divergence',fontsize=12)
        save_dir = get_save_dir(ensemble_dir)
        if save_fig:
            fig.savefig(save_dir + 'KL_vs_time__few_allProcesses.pdf', dpi=dpi)
    # plt.legend()
    # fig.show()
    
    
def KL_vs_time__withProcesses_few(
        ensemble_prefix,scenario_nums,processes,labs,timesteps,
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        backward=False,recompute_KL=False):
    one_KL = np.loadtxt(ensemble_over_dir + ensemble_prefix + '_a_both/' + 'processed_output/tt' + str(timesteps[0]).zfill(4) + '/KL_repeats.txt')
    
    n_repeat = one_KL.shape[1]
    n_scenarios = one_KL.shape[0]
    KL = np.zeros((len(timesteps),) + one_KL.shape + (len(processes),))
    KL_recomputed = np.zeros_like(KL)
    
    for tt,timestep in enumerate(timesteps):
        for pp,(process,lab) in enumerate(zip(processes,labs)):
            tt_dir = ensemble_over_dir + ensemble_prefix + '_' + lab + '_' + process + '/processed_output/tt' + str(timestep).zfill(4) + '/'
            dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
            for ii in range(n_repeat):
                dNdlnD_partmc = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii + 1).zfill(4) + '.txt')
                for ss in range(n_scenarios):
                    KL_recomputed[tt,ss,ii,pp] = analyses.get_KL_binned(dNdlnD_partmc[ss,:],dNdlnD_mam4[ss,:],backward=backward)
            if recompute_KL:
                dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
                for ii in range(n_repeat):
                    dNdlnD_partmc = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii + 1).zfill(4) + '.txt')
                    for ss in range(n_scenarios):
                        KL[tt,ss,ii,pp] = analyses.get_KL_binned(dNdlnD_partmc[ss,:],dNdlnD_mam4[ss,:],backward=backward)
            else:
                if backward:
                    KL[tt,:,:,pp] = np.loadtxt(tt_dir + 'KL_repeats_backward.txt')
                else:
                    KL[tt,:,:,pp] = np.loadtxt(tt_dir + 'KL_repeats.txt')

    fig,axs = plt.subplots(nrows=len(scenario_nums),ncols=1,sharex=True)
    for ss in range(KL.shape[1]):
        for pp,process in enumerate(processes):
            axs[ss].errorbar(timesteps,np.mean(KL[:,ss,:,pp],axis=1),yerr=np.std(KL[:,ss,:,pp],axis=1),label=process)
    
    axs[0].add_legend()
    fig.show()
    
def KL_vs_time__withProcesses(
        ensemble_prefix,timesteps,
        processes=['both','cond','coag'],labs=['a','b','c'],
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        backward=False,recompute_KL=False):
    one_KL = np.loadtxt(ensemble_over_dir + ensemble_prefix + '_a_both/' + 'processed_output/tt' + str(timesteps[0]).zfill(4) + '/KL_repeats.txt')
    
    n_repeat = one_KL.shape[1]
    n_scenarios = one_KL.shape[0]
    KL = np.zeros((len(timesteps),) + one_KL.shape + (len(processes),))
    KL_recomputed = np.zeros_like(KL)
    
    for tt,timestep in enumerate(timesteps):
        for pp,(process,lab) in enumerate(zip(processes,labs)):
            tt_dir = ensemble_over_dir + ensemble_prefix + '_' + lab + '_' + process + '/processed_output/tt' + str(timestep).zfill(4) + '/'
            dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
            for ii in range(n_repeat):
                dNdlnD_partmc = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii + 1).zfill(4) + '.txt')
                for ss in range(n_scenarios):
                    KL_recomputed[tt,ss,ii,pp] = analyses.get_KL_binned(dNdlnD_partmc[ss,:],dNdlnD_mam4[ss,:],backward=backward)
            if recompute_KL:
                dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
                for ii in range(n_repeat):
                    dNdlnD_partmc = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii + 1).zfill(4) + '.txt')
                    for ss in range(n_scenarios):
                        KL[tt,ss,ii,pp] = analyses.get_KL_binned(dNdlnD_partmc[ss,:],dNdlnD_mam4[ss,:],backward=backward)
            else:
                if backward:
                    KL[tt,:,:,pp] = np.loadtxt(tt_dir + 'KL_repeats_backward.txt')
                else:
                    KL[tt,:,:,pp] = np.loadtxt(tt_dir + 'KL_repeats.txt')
                
    for ss in range(KL.shape[1]):
        fig,axs = plt.subplots(nrows=1,ncols=1,sharex=True)
        for pp,process in enumerate(processes):
            axs.errorbar(timesteps,np.mean(KL[:,ss,:,pp],axis=1),yerr=np.std(KL[:,ss,:,pp],axis=1),label=process)
            
        axs.legend()
        fig.show()
        
def D50_vs_time__withProcesses(
        ensemble_prefix,timesteps,
        processes=['both','cond','coag'],labs=['a','b','c'],
        lnDs = np.log(np.logspace(-10,-5,100)),
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        backward=False,recompute_KL=False):

    
    ensemble_dir = ensemble_over_dir + ensemble_prefix + '_' + labs[0] + '_' + processes[0] + '/'
    tt_dir_one = ensemble_dir + 'processed_output/tt' + str(0).zfill(4) + '/'
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    
    n_scenarios = len(ensemble_settings)
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    
    s_env_vals, frac_ccn_partmc_mean, frac_ccn_partmc_std, frac_ccn_mam4 =  analyses.read_frac_ccn(tt_dir_one,1)
    
    dNdlnD_mam4 = np.zeros((len(timesteps),len(processes),n_scenarios,len(lnDs)))
    dNdlnD_partmc = np.zeros((len(timesteps),len(processes),n_scenarios,len(lnDs)))
    frac_ccn_mam4 = np.zeros((len(timesteps),len(processes),n_scenarios,len(s_env_vals)))
    frac_ccn_partmc = np.zeros((len(timesteps),len(processes),n_scenarios,len(s_env_vals)))
    D50_mam4 = np.zeros((len(timesteps),len(processes),n_scenarios))
    D50_partmc = np.zeros((len(timesteps),len(processes),n_scenarios))
    s50_mam4 = np.zeros((len(timesteps),len(processes),n_scenarios))
    s50_partmc = np.zeros((len(timesteps),len(processes),n_scenarios))
    
    Ns = np.zeros([len(timesteps),n_scenarios,len(processes),4])
    mus = np.zeros([len(timesteps),n_scenarios,len(processes),4])
    sigs = np.zeros([len(timesteps),n_scenarios,len(processes),4])
    KL = np.zeros([len(timesteps),n_scenarios,len(processes)])
    KL_std = np.zeros([len(timesteps),n_scenarios,len(processes)])
    # for (process,lab) in zip(processes,labs):
    #     
        # dNdlnD_mam4 = np.zeros([len(timesteps),len(lnDs),n_scenarios,len(processes)])
        # dNdlnD_partmc = np.zeros([len(timesteps),len(lnDs),n_scenarios,len(processes)])
    for pp,(process,lab) in enumerate(zip(processes,labs)):
        ensemble_dir = ensemble_over_dir + ensemble_prefix + '_' + lab + '_' + process + '/'
        for tt,timestep in enumerate(timesteps):
            tt_dir = ensemble_over_dir + ensemble_prefix + '_' + lab + '_' + process + '/processed_output/tt' + str(timestep).zfill(4) + '/'
            Ns[tt,:,pp,:] = np.loadtxt(tt_dir + 'Ns.txt')
            mus[tt,:,pp,:] = np.loadtxt(tt_dir + 'mus.txt')
            sigs[tt,:,pp,:] = np.loadtxt(tt_dir + 'sigs.txt')
            print('pp',pp,tt_dir)
            dNdlnD_mam4[tt,pp,:,:] = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
            dNdlnD_partmc_repeats = np.zeros([n_scenarios,len(lnDs),n_repeat])
            # KL_repeats = np.zeros([n_scenarios,n_repeat])
            for ii in range(n_repeat):
                dNdlnD_partmc_repeats[:,:,ii] = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii + 1).zfill(4) + '.txt')
            dNdlnD_partmc[tt,pp,:,:] = np.mean(dNdlnD_partmc_repeats,axis=2)
            s_env_vals, frac_ccn_partmc[tt,pp,:,:], frac_ccn_partmc_std, frac_ccn_mam4[tt,pp,:,:] =  analyses.read_frac_ccn(tt_dir,n_repeat)
            
            KL_repeats = np.loadtxt(tt_dir + 'KL_repeats_backward.txt')
            KL[tt,:,pp] = np.mean(KL_repeats,axis=1)
            # KL_std[tt,:,pp] = np.std(KL_repeats,axis=1)
            for ss in range(n_scenarios):
                # KL[tt,ss,pp] = analyses.get_KL_binned(dNdlnD_partmc[tt,pp,ss,:],dNdlnD_mam4[tt,pp,ss,:],backward=backward)
                D50_partmc[tt,pp,ss] = np.interp(0.5,np.cumsum(dNdlnD_partmc[tt,pp,ss,:])/np.sum(dNdlnD_partmc[tt,pp,ss,:]),np.exp(lnDs))
                D50_mam4[tt,pp,ss] = np.interp(0.5,np.cumsum(dNdlnD_mam4[tt,pp,ss,:])/np.sum(dNdlnD_mam4[tt,pp,ss,:]),np.exp(lnDs))
                s50_partmc[tt,pp,ss] = np.interp(0.5,frac_ccn_partmc[tt,pp,ss,:],s_env_vals)
                s50_mam4[tt,pp,ss] = np.interp(0.5,frac_ccn_mam4[tt,pp,ss,:],s_env_vals)
                # D50_partmc[tt,pp,ss] = np.exp(np.interp(0.5,np.cumsum(dNdlnD_partmc[tt,pp,ss,:])/np.sum(dNdlnD_partmc[tt,pp,ss,:]),lnDs))
                # D50_mam4[tt,pp,ss] = np.exp(np.interp(0.5,np.cumsum(dNdlnD_mam4[tt,pp,ss,:])/np.sum(dNdlnD_mam4[tt,pp,ss,:]),lnDs))
                # s50_partmc[tt,pp,ss] = np.exp(np.interp(0.5,frac_ccn_partmc[tt,pp,ss,:],np.log(s_env_vals)))
                # s50_mam4[tt,pp,ss] = np.exp(np.interp(0.5,frac_ccn_mam4[tt,pp,ss,:],np.log(s_env_vals)))
            #     for ss in range(n_scenarios):
            #         KL_recomputed[tt,ss,ii,pp] = analyses.get_KL_binned(dNdlnD_partmc[ss,:],dNdlnD_mam4[ss,:],backward=backward)
            # if recompute_KL:
            #     dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
            #     for ii in range(n_repeat):
            #         dNdlnD_partmc = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii + 1).zfill(4) + '.txt')
            #         for ss in range(n_scenarios):
            #             KL[tt,ss,ii,pp] = analyses.get_KL_binned(dNdlnD_partmc[ss,:],dNdlnD_mam4[ss,:],backward=backward)
            # else:
            #     if backward:
            #         KL[tt,:,:,pp] = np.loadtxt(tt_dir + 'KL_repeats_backward.txt')
            #     else:
            #         KL[tt,:,:,pp] = np.loadtxt(tt_dir + 'KL_repeats.txt')
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # twin1 = ax.twinx()
    
    dtime = 60./60.
    ts = timesteps*dtime
    
    for ss in range(100):
        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(6.5,2.5)
        print(ss)
        p1a, = axs[0].plot(ts,D50_partmc[:,0,ss],color='k',linewidth=2.,linestyle='--',label='PartMC-MOSAIC')
        p1b, = axs[0].plot(ts,D50_mam4[:,0,ss],color='C0',label='MAM4')
        p2a, = axs[1].plot(ts,s50_partmc[:,0,ss],color='k',linewidth=2.,linestyle='--',label='PartMC-MOSAIC')
        p2b, = axs[1].plot(ts,s50_mam4[:,0,ss],color='C0',label='MAM4')
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")
        axs[0].set(xlim=(min(ts),max(ts)), xlabel="time [s]", ylabel="median diameter")#,yscale='log')
        axs[1].set(xlim=(min(ts),max(ts)), xlabel="time [s]", ylabel="supersaturation at which\nhalf activate [%]")#,yscale='log')
        plt.show()
        
    # plt.plot(timesteps,,color='C0',linestyle='--');  plt.plot(timesteps,,color='C0',linestyle='--');
    
    # dtime = 60.
    ts = timesteps*dtime
    
    for ss in range(100):
        fig, ax1 = plt.subplots(1,1)
        
        ax2 = ax1.twinx()
        
        p1, = ax1.plot(ts,KL[:,ss,0],color='C2',label='KL-divergence')
        p2, = ax2.plot(ts,s50_mam4[:,0,ss]/s50_partmc[:,0,ss],color='C4',label='$s_{50}$')
        
        
        # p2a, = ax2.plot(ts,s50_partmc[:,0,ss],color='C4',linestyle='--', label='PartMC-MOSAIC')
        # p2b, = ax2.plot(ts,s50_mam4[:,0,ss],color='C4', label='MAM4')
        
        ax1.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())
    
        ax1.set(xlim=(min(ts),max(ts)), xlabel="time [min]", ylabel="KL-divergence")#,yscale='log')
        ax2.set(xlim=(min(ts),max(ts)), xlabel="time [min]", ylabel="$s_{50,\mathrm{MAM4}}/s_{50,\mathrm{PartMC-MOSAIC}}$")#),yscale='log')
        
        ax1.tick_params(axis='y', labelcolor=p1.get_color())
        ax2.tick_params(axis='y', labelcolor=p2.get_color())
        plt.show()
    
    
    # fig.set_size_inches(4.5,3.5)
    
    # ax.set_yscale('log')
    # twin1.set_yscale('log')
    # ax.set_xlim([min(ts),max(ts)])
    
    # for ss in range(KL.shape[1]):
    #     fig,axs = plt.subplots(nrows=1,ncols=len(labs),sharex=True)
    #     for pp,process in enumerate(processes):
    #         axs.errorbar(timesteps,np.mean(KL[:,ss,:,pp],axis=1),yerr=np.std(KL[:,ss,:,pp],axis=1),label=process)
            
    #     axs.legend()
    #     fig.show()
def scatter_s50_vs_KL(
        ensemble_dir,timesteps,dtime,backward=False,recompute_KL=True,
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        save_fig=True,dpi=500,mode_sigs=np.log([1.8,1.6,1.8,1.6])):
    df = make_dataframe(
        ensemble_dir,timesteps,recompute_KL=recompute_KL,backward=backward)

    # fig,ax1 = plt.subplots(1,1)
    # col1 = 'C2'
    # col2 = 'C4'
    # ax1.scatter(df['KL_backward'],df['s_onset_ratio_50'],color=col1)
    # ax2 = ax1.twinx()
    # ax2.scatter(df['KL_backward'],df['s_onset_ratio_99'],color=col2)
    # # ax1.xlim([0.,max(df['KL_backward'])])
    # # ax1.ylim([0.,ax2.get_ylim()])
    
    # ax1.yaxis.label.set_color(col1)
    # ax2.yaxis.label.set_color(col2)
    
    # ax1.set(xlim=(0.,max(df['KL_backward'])), xscale='log', xlabel="KL-divergence", ylabel="$s_{50,\mathrm{MAM4}}/s_{50,\mathrm{PartMC-MOSAIC}}$")#,yscale='log')
    # ax2.set(xlim=(0.,max(df['KL_backward'])), xscale='log', xlabel="KL-divergence", ylabel="$s_{99,\mathrm{MAM4}}/s_{99,\mathrm{PartMC-MOSAIC}}$")#),yscale='log')
    
    # ax1.tick_params(axis='y', labelcolor=col1)
    # ax2.tick_params(axis='y', labelcolor=col2)
    
    fig,ax = plt.subplots(1,1)
    ax.scatter(df['KL_backward'],df['s_onset_ratio_01'],label='$s_{1\%,\mathrm{MAM4}}/s_{1\%,\mathrm{PartMC-MOSAIC}}$')
    ax.scatter(df['KL_backward'],df['s_onset_ratio_50'],label='$s_{50\%,\mathrm{MAM4}}/s_{50\%,\mathrm{PartMC-MOSAIC}}$')
    ax.scatter(df['KL_backward'],df['s_onset_ratio_99'],label='$s_{99\%,\mathrm{MAM4}}/s_{99%,\mathrm{PartMC-MOSAIC}}$')
    # plt.xlim([0.,ax.get_xlim()[1]])
    ax.set_xscale('log')
    ax.set_xlim([min(df['KL_backward']),max(df['KL_backward'])])
    ax.set_xlabel('KL-divergence',fontsize=12)
    ax.set_ylabel('onset supersaturation for activation\nfrom MAM4 relative to PartMC-MOSAIC',fontsize=12)
    ax.hlines([1.],ax.get_xlim()[0],ax.get_xlim()[1],'k',linewidth=0.5)
    ax.legend(loc="best")
    save_dir = get_save_dir(ensemble_dir)
    fig.savefig(save_dir + 's_ratio__vs__KL.pdf',dpi=dpi,bbox_inches='tight')
    

def scatter_sThresh_contributions(
        ensemble_dir,ensemble_dir_cond,timesteps,dtime,backward=False,recompute_KL=True,
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        save_fig=True,dpi=500,mode_sigs=np.log([1.8,1.6,1.8,1.6])):
    df = make_dataframe(
        ensemble_dir,timesteps,recompute_KL=recompute_KL,backward=backward)
    df_cond = make_dataframe(
        ensemble_dir_cond,timesteps,recompute_KL=recompute_KL,backward=backward)
    # fig,ax1 = plt.subplots(1,1)
    # col1 = 'C2'
    # col2 = 'C4'
    # ax1.scatter(df['KL_backward'],df['s_onset_ratio_50'],color=col1)
    # ax2 = ax1.twinx()
    # ax2.scatter(df['KL_backward'],df['s_onset_ratio_99'],color=col2)
    # # ax1.xlim([0.,max(df['KL_backward'])])
    # # ax1.ylim([0.,ax2.get_ylim()])
    
    # ax1.yaxis.label.set_color(col1)
    # ax2.yaxis.label.set_color(col2)
    
    # ax1.set(xlim=(0.,max(df['KL_backward'])), xscale='log', xlabel="KL-divergence", ylabel="$s_{50,\mathrm{MAM4}}/s_{50,\mathrm{PartMC-MOSAIC}}$")#,yscale='log')
    # ax2.set(xlim=(0.,max(df['KL_backward'])), xscale='log', xlabel="KL-divergence", ylabel="$s_{99,\mathrm{MAM4}}/s_{99,\mathrm{PartMC-MOSAIC}}$")#),yscale='log')
    
    # ax1.tick_params(axis='y', labelcolor=col1)
    # ax2.tick_params(axis='y', labelcolor=col2)
    
    fig,ax = plt.subplots(1,1)
    ax.scatter(df['KL_backward'],df_cond['s_onset_ratio_01']/df['s_onset_ratio_01'],label='$s_{1\%,\mathrm{MAM4}}/s_{1\%,\mathrm{PartMC-MOSAIC}}$')
    ax.scatter(df['KL_backward'],df_cond['s_onset_ratio_50']/df['s_onset_ratio_50'],label='$s_{50\%,\mathrm{MAM4}}/s_{50\%,\mathrm{PartMC-MOSAIC}}$')
    ax.scatter(df['KL_backward'],df_cond['s_onset_ratio_99']/df['s_onset_ratio_99'],label='$s_{99\%,\mathrm{MAM4}}/s_{99%,\mathrm{PartMC-MOSAIC}}$')
    # plt.xlim([0.,ax.get_xlim()[1]])
    ax.set_xscale('log')
    ax.set_xlim([min(df['KL_backward']),max(df['KL_backward'])])
    ax.set_xlabel('KL-divergence',fontsize=12)
    ax.set_ylabel('onset supersaturation for activation\nfrom MAM4 relative to PartMC-MOSAIC',fontsize=12)
    ax.hlines([1.],ax.get_xlim()[0],ax.get_xlim()[1],'k',linewidth=0.5)
    ax.legend(loc="best")
    save_dir = get_save_dir(ensemble_dir)
    # fig.savefig(save_dir + 's_ratio__vs__KL.pdf',dpi=dpi,bbox_inches='tight')
    
def scatter_aging_conditions(
        ensemble_prefix,timesteps,dtime,backward=False,recompute_KL=True,
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        save_fig=True,dpi=500,mode_sigs=np.log([1.8,1.6,1.8,1.6])):
    ensemble_dir = ensemble_over_dir + ensemble_prefix + '_a_both/'
    one_KL = np.loadtxt(ensemble_dir + 'processed_output/tt' + str(timesteps[0]).zfill(4) + '/KL_repeats.txt')
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    n_repeat = ensemble_settings[0]['spec']['n_repeat']
    KL = np.zeros((len(timesteps),) + one_KL.shape)
    sa_flux = np.zeros((len(timesteps),) + one_KL.shape)
    num_conc = np.zeros((len(timesteps),) + one_KL.shape)
    sa_flux_mam4 = np.zeros((len(timesteps),) + one_KL[:,0].shape)
    num_conc_mam4 = np.zeros((len(timesteps),) + one_KL[:,0].shape)
    
    for tt,timestep in enumerate(timesteps):
        tt_dir = ensemble_dir + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        tt_dir__cond = ensemble_over_dir + ensemble_prefix + '_b_cond/' + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        tt_dir__coag = ensemble_over_dir + ensemble_prefix + '_c_coag/' + 'processed_output/tt' + str(timestep).zfill(4) + '/'
        
        if recompute_KL:
            dNdlnD_mam4 = np.loadtxt(tt_dir + 'dNdlnD_mam4.txt')
            for ii in range(one_KL.shape[1]):
                dNdlnD_partmc = np.loadtxt(tt_dir + 'dNdlnD_partmc_repeat' + str(ii + 1).zfill(4) + '.txt')
                for ss in range(one_KL.shape[0]):
                    KL[tt,ss,ii] = analyses.get_KL_binned(dNdlnD_partmc[ss,:],dNdlnD_mam4[ss,:],backward=backward)
        else:
            if backward:
                KL[tt,:,:] = np.loadtxt(tt_dir + 'KL_repeats_backward.txt')
            else:
                KL[tt,:,:] = np.loadtxt(tt_dir + 'KL_repeats.txt')
        
        sa_flux[tt,:,:] = np.loadtxt(tt_dir__cond + 'sa_flux.txt')
        num_conc[tt,:,:] = np.loadtxt(tt_dir__coag + 'num_conc.txt')
        
        sa_flux_mam4[tt,:] = analyses.read_boxmodel_data('sa_flux_mam',tt_dir,timestep*dtime,mode_sigs=mode_sigs,recompute_KL = recompute_KL, n_repeat=n_repeat) #np.loadtxt(tt_dir+ 'sa_flux_mam4.txt')
        num_conc_mam4[tt,:] = analyses.read_boxmodel_data('N_tot',tt_dir,timestep*dtime,mode_sigs=mode_sigs,recompute_KL = recompute_KL, n_repeat=n_repeat) # np.loadtxt(tt_dir + 'num_conc_mam4.txt')
    
    fig,ax = plt.subplots(1)    
    # df = pandas.DataFrame(data=np.array([1e9*3600.*np.mean(np.mean(sa_flux,axis=2),axis=0),np.mean(np.mean(num_conc,axis=2),axis=0)*1e-6]).transpose(),columns=['sa_flux','num_conc'])
    # df = pandas.DataFrame(data=np.array([1e9*3600.*np.mean(np.mean(sa_flux,axis=2),axis=0),np.mean(np.mean(num_conc,axis=2),axis=0)*1e-6]).transpose(),columns=['sa_flux','num_conc'])
    # g = sns.JointGrid(height=3,ratio=4,data=df,x='sa_flux',y='num_conc')
    # g.set(fontsize=16,xscale='log',yscale='log')
    # plt.scatter(np.mean(1e9*3600.*sa_flux_mam4,axis=0),np.mean(1e-6*num_conc_mam4,axis=0),s=50,c='k')
    plt.scatter(np.mean(1e9*3600.*np.mean(sa_flux,axis=2),axis=0),np.mean(1e-6*np.mean(num_conc,axis=2),axis=0),s=50,c='k')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim([1e-3,10.])
    plt.xlabel('condensational growth rate [nm/h]',fontdict={'size':12})
    plt.ylabel('number concentration [cm$^{-3}$]',fontdict={'size':12})
    if save_fig:
        save_dir = get_save_dir(ensemble_dir)
        # fig.savefig('/Users/fier887/Downloads/aging_conditions.pdf',dpi=dpi,bbox_inches='tight')
        fig.savefig(save_dir + 'aging_conditions.pdf',dpi=dpi,bbox_inches='tight')
        
    # g.set_axis_labels(['condensational growth rate [nm h$^{-1}$]','number concentration [cm$^{-3}$]'])    
    # g.plot(sns.scatterplot, sns.histplot)
    # g.plot(sns.scatterplot, sns.kdeplot)
    
    #sns.JointGrid(x='condesational growth rate [nm/h]',y='number concentration [cm$^{-3}$]')
    
        
def KL_regimes(
        ensemble_dir,timesteps,
        recompute_KL=True,backward=False,units='h',add_lines=True,
        regime_type='process_extremes',avg_scale='lin',save_fig=True,
        showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95),
        lnDs = np.log(np.logspace(-10,-5,100))):
    
    df = make_dataframe(
        ensemble_dir,timesteps,recompute_KL=recompute_KL,backward=backward,
        regime_type=regime_type,avg_scale=avg_scale,Ds=np.exp(lnDs))
    fig,ax = plt.subplots(1)
    fig.set_size_inches(fig_width,fig_height)
    regime_names = []
    for regime_name in np.unique(df['regimes']):
        regime_names.append(regime_name[0])
    sns.boxplot(data=df, x='time', y='KL', hue='regime_ids',showfliers=showfliers,whis=whis)
    
    if units == 'h':
        if all(np.floor(np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/3600) == np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/3600):    
            hour_vals = np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/3600
            time_vals = np.array([int(one_hour) for one_hour in hour_vals])
        else:
            time_vals = np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/3600
    elif units == 'min':
        min_vals = np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/60
        time_vals = np.array([int(one_min) for one_min in min_vals])
    
    if add_lines:
        xticks = ax.get_xticks()
        line_x = 0.5*(xticks[:-1]+xticks[1:])
        ylims = ax.get_ylim()
        ax.vlines(line_x,ylims[0],ylims[1],alpha=0.5,color='k')
        ax.set_ylim(ylims)
        
    ax.set_xticklabels(time_vals,fontdict={'size':10})
    ax.set_yticklabels(np.round(ax.get_yticks(),2),fontdict={'size':10})

    if units == 'h':
        ax.set_xlabel('time [h]',fontdict={'size':12})
    elif units == 'min':
        ax.set_xlabel('time [min]',fontdict={'size':12})
    
    ax.set_ylabel('KL-divergence',fontdict={'size':12})
    
    handles, _ = ax.get_legend_handles_labels()          # Get the artists.
    ax.legend(handles,regime_names, loc="best")
    if save_fig:
        save_dir = get_save_dir(ensemble_dir)
        fig.savefig(save_dir + 'KL_regimes.pdf',dpi=500,bbox_inches='tight')

def s_onset_regimes(
        ensemble_dir,timesteps,thresh=0.01,
        recompute_KL=True,backward=False,units='h',add_lines=True,
        regime_type='process_extremes',avg_scale='lin',save_fig=True,
        showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95)):
    
    df = make_dataframe(
        ensemble_dir,timesteps,recompute_KL=recompute_KL,backward=backward,
        regime_type=regime_type,avg_scale=avg_scale)
    fig,ax = plt.subplots(1)
    fig.set_size_inches(fig_width,fig_height)
    regime_names = []
    for regime_name in np.unique(df['regimes']):
        regime_names.append(regime_name[0])
    sns.boxplot(data=df, x='time', y='s_onset_ratio_' + str(int(thresh*100)).zfill(2), hue='regime_ids',showfliers=showfliers,whis=whis)
    
    if units == 'h':
        if all(np.floor(np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/3600) == np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/3600):    
            hour_vals = np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/3600
            time_vals = np.array([int(one_hour) for one_hour in hour_vals])
        else:
            time_vals = np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/3600
    elif units == 'min':
        min_vals = np.linspace(min(df['time']),max(df['time']),len(ax.get_xticks()))/60
        time_vals = np.array([int(one_min) for one_min in min_vals])
    
    if add_lines:
        xticks = ax.get_xticks()
        line_x = 0.5*(xticks[:-1]+xticks[1:])
        ylims = ax.get_ylim()
        ax.vlines(line_x,ylims[0],ylims[1],alpha=0.5,color='k')
        ax.set_ylim(ylims)
        
    ax.set_xticklabels(time_vals,fontdict={'size':10})
    ax.set_yticklabels(np.round(ax.get_yticks(),2),fontdict={'size':10})

    if units == 'h':
        ax.set_xlabel('time [h]',fontdict={'size':12})
    elif units == 'min':
        ax.set_xlabel('time [min]',fontdict={'size':12})
    
    # ax.set_ylabel('$s_{' + str(int(thresh*100)) + '}$',fontdict={'size':12})
    ax.set_ylabel('$s_{\mathrm{' + str(int(thresh*100)) + ',MAM4}}/s_{\mathrm{' + str(int(thresh*100)) + ',PartMC-MOSAIC}}$',fontdict={'size':12})
    
    
    handles, _ = ax.get_legend_handles_labels()          # Get the artists.
    ax.legend(handles,regime_names, loc="best")
    if save_fig:
        save_dir = get_save_dir(ensemble_dir)
        fig.savefig(save_dir + 's' + str(thresh*100) + '_regimes.pdf',dpi=500,bbox_inches='tight')
    
# make this more general! send in varnames. 
def make_dataframe(
        ensemble_dir,timesteps,Ds = [], recompute_KL=True,backward=False,
        mode_sigs = np.log([1.8,1.6,1.8,1.6]), regime_type='process_extremes',avg_scale='lin'):    
    scenario_groups,regime_names = analyses.scenarios_in_regimes(
        ensemble_dir,timesteps,regime_type=regime_type,avg_scale=avg_scale)
    
    ensemble_dir_cond = ensemble_dir[:-7] + 'b_cond/'
    ensemble_dir_coag = ensemble_dir[:-7] + 'c_coag/'
    # ensemble_dir_cond = ensemble_dir[:-7] + 'c_coag/'
    ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
    dtime = ensemble_settings[0]['spec']['t_output']
    
    all_scenarios = []
    all_regimes = []
    all_regime_ids = []
    all_timesteps = []
    all_times = []
    all_s01 = []
    all_s05 = []
    all_s25 = []
    all_s50 = []
    all_s75 = []
    all_s95 = []
    all_s99 = []
    all_s01_mam = []
    all_s05_mam = []
    all_s25_mam = []
    all_s50_mam = []
    all_s75_mam = []
    all_s95_mam = []
    all_s99_mam = []
    all_s01_partmc = []
    all_s05_partmc = []
    all_s25_partmc = []
    all_s50_partmc = []
    all_s75_partmc = []
    all_s95_partmc = []
    all_s99_partmc = []
    
    all_s01_uniformComp = []
    all_s05_uniformComp = []
    all_s25_uniformComp = []
    all_s50_uniformComp = []
    all_s75_uniformComp = []
    all_s95_uniformComp = []
    all_s99_uniformComp = []
    all_s01_mam_uniformComp = []
    all_s05_mam_uniformComp = []
    all_s25_mam_uniformComp = []
    all_s50_mam_uniformComp = []
    all_s75_mam_uniformComp = []
    all_s95_mam_uniformComp = []
    all_s99_mam_uniformComp = []
    all_s01_partmc_uniformComp = []
    all_s05_partmc_uniformComp = []
    all_s25_partmc_uniformComp = []
    all_s50_partmc_uniformComp = []
    all_s75_partmc_uniformComp = []
    all_s95_partmc_uniformComp = []
    all_s99_partmc_uniformComp = []
    
    all_s01_mam_condOnly = []
    all_s05_mam_condOnly = []
    all_s25_mam_condOnly = []
    all_s50_mam_condOnly = []
    all_s75_mam_condOnly = []
    all_s95_mam_condOnly = []
    all_s99_mam_condOnly = []
    all_s01_partmc_condOnly = []
    all_s05_partmc_condOnly = []
    all_s25_partmc_condOnly = []
    all_s50_partmc_condOnly = []
    all_s75_partmc_condOnly = []
    all_s95_partmc_condOnly = []
    all_s99_partmc_condOnly = []

    all_s01_mam_coagOnly = []
    all_s05_mam_coagOnly = []
    all_s25_mam_coagOnly = []
    all_s50_mam_coagOnly = []
    all_s75_mam_coagOnly = []
    all_s95_mam_coagOnly = []
    all_s99_mam_coagOnly = []
    all_s01_partmc_coagOnly = []
    all_s05_partmc_coagOnly = []
    all_s25_partmc_coagOnly = []
    all_s50_partmc_coagOnly = []
    all_s75_partmc_coagOnly = []
    all_s95_partmc_coagOnly = []
    all_s99_partmc_coagOnly = []
    
    all_D50 = []
    all_errFracCCN_01 = []
    all_errFracCCN_03 = []
    all_errFracCCN_10 = []
    all_KL = []
    all_KL_backward = []
    all_sa_flux = []
    all_num_conc = []
    all_integrated_sa = []
    all_integrated_num = []
    
    for timestep in timesteps:
        one_time = timestep*dtime
        
        KL_in_groups = analyses.get_group_vars(
                'KL',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs)
        KL_backward_in_groups = analyses.get_group_vars(
                'KL_backward',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs)
        D50_ratio_in_groups = analyses.get_group_vars(
                'D50_ratio',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs)
        s01_in_groups = analyses.get_group_vars(
                's_onset_ratio',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_in_groups = analyses.get_group_vars(
                's_onset_ratio',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_in_groups = analyses.get_group_vars(
                's_onset_ratio',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_in_groups = analyses.get_group_vars(
                's_onset_ratio',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.5)
        s75_in_groups = analyses.get_group_vars(
                's_onset_ratio',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_in_groups = analyses.get_group_vars(
                's_onset_ratio',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_in_groups = analyses.get_group_vars(
                's_onset_ratio',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        
        s01_mam_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_mam_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_mam_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_mam_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.50)
        s75_mam_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_mam_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_mam_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        s01_partmc_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_partmc_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_partmc_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_partmc_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.50)
        s75_partmc_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_partmc_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_partmc_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        
        s01_mam_condOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_mam_condOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_mam_condOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_mam_condOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.50)
        s75_mam_condOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_mam_condOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_mam_condOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        s01_partmc_condOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_partmc_condOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_partmc_condOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_partmc_condOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.50)
        s75_partmc_condOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_partmc_condOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_partmc_condOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_cond,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        
        s01_mam_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_mam_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_mam_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_mam_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.50)
        s75_mam_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_mam_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_mam_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_mam4',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        s01_partmc_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_partmc_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_partmc_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_partmc_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.50)
        s75_partmc_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_partmc_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_partmc_coagOnly_in_groups = analyses.get_group_vars(
                's_onset_partmc',ensemble_dir_coag,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        
        s01_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_ratio_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_ratio_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_ratio_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_ratio_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.5)
        s75_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_ratio_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_ratio_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_ratio_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        
        s01_mam_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_mam4_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_mam_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_mam4_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_mam_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_mam4_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_mam_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_mam4_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.50)
        s75_mam_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_mam4_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_mam_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_mam4_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_mam_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_mam4_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        s01_partmc_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_partmc_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.01)
        s05_partmc_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_partmc_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.05)
        s25_partmc_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_partmc_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.25)
        s50_partmc_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_partmc_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.50)
        s75_partmc_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_partmc_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.75)
        s95_partmc_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_partmc_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.95)
        s99_partmc_uniformComp_in_groups = analyses.get_group_vars(
                's_onset_partmc_uniformComp',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.99)
        errFracCCN_01_in_groups = analyses.get_group_vars(
                'err_frac_ccn',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.1)
        errFracCCN_03_in_groups = analyses.get_group_vars(
                'err_frac_ccn',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=0.3)
        errFracCCN_10_in_groups = analyses.get_group_vars(
                'err_frac_ccn',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs,thresh=1.0)
        sa_flux_in_groups = analyses.get_group_vars(
                'sa_flux',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = mode_sigs)
        num_conc_in_groups = analyses.get_group_vars(
                'num_conc',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = np.log([1.8,1.6,1.6,1.8]))
        integrated_sa_in_groups = analyses.get_group_vars(
                'integrated_sa_flux',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = np.log([1.8,1.6,1.6,1.8]))
        integrated_num_in_groups = analyses.get_group_vars(
                'integrated_num_conc',ensemble_dir,scenario_groups,timestep,
                recompute_KL=recompute_KL, mode_sigs = np.log([1.8,1.6,1.6,1.8]))
        for qq,(
                one_scenario_group,one_regime,one_KL_group,one_KL_backward_group,
                s01_group, s05_group, s25_group, s50_group, s75_group, s95_group, s99_group,
                s01_partmc_group, s05_partmc_group, s25_partmc_group, s50_partmc_group, s75_partmc_group, s95_partmc_group, s99_partmc_group, 
                s01_mam_group, s05_mam_group, s25_mam_group, s50_mam_group, s75_mam_group, s95_mam_group, s99_mam_group, 
                s01_uniformComp_group, s05_uniformComp_group, s25_uniformComp_group, s50_uniformComp_group, s75_uniformComp_group, s95_uniformComp_group, s99_uniformComp_group,
                s01_partmc_uniformComp_group, s05_partmc_uniformComp_group, s25_partmc_uniformComp_group, s50_partmc_uniformComp_group, s75_partmc_uniformComp_group, s95_partmc_uniformComp_group, s99_partmc_uniformComp_group, 
                s01_mam_uniformComp_group, s05_mam_uniformComp_group, s25_mam_uniformComp_group, s50_mam_uniformComp_group, s75_mam_uniformComp_group, s95_mam_uniformComp_group, s99_mam_uniformComp_group, 
                s01_partmc_condOnly_group, s05_partmc_condOnly_group, s25_partmc_condOnly_group, s50_partmc_condOnly_group, s75_partmc_condOnly_group, s95_partmc_condOnly_group, s99_partmc_condOnly_group, 
                s01_mam_condOnly_group, s05_mam_condOnly_group, s25_mam_condOnly_group, s50_mam_condOnly_group, s75_mam_condOnly_group, s95_mam_condOnly_group, s99_mam_condOnly_group, 
                s01_partmc_coagOnly_group, s05_partmc_coagOnly_group, s25_partmc_coagOnly_group, s50_partmc_coagOnly_group, s75_partmc_coagOnly_group, s95_partmc_coagOnly_group, s99_partmc_coagOnly_group, 
                s01_mam_coagOnly_group, s05_mam_coagOnly_group, s25_mam_coagOnly_group, s50_mam_coagOnly_group, s75_mam_coagOnly_group, s95_mam_coagOnly_group, s99_mam_coagOnly_group, 
                errFracCCN_01_group, errFracCCN_03_group, errFracCCN_10_group,
                D50_ratio_group,
                one_sa_flux_group,one_num_conc_group,one_integrated_sa_group,one_integrated_num_group,
                ) in enumerate(zip(
                    scenario_groups,regime_names,KL_in_groups,KL_backward_in_groups,                    
                    s01_in_groups, s05_in_groups, s25_in_groups, s50_in_groups, s75_in_groups, s95_in_groups, s99_in_groups,
                    s01_partmc_in_groups, s05_partmc_in_groups, s25_partmc_in_groups, s50_partmc_in_groups, s75_partmc_in_groups, s95_partmc_in_groups, s99_partmc_in_groups, 
                    s01_mam_in_groups, s05_mam_in_groups, s25_mam_in_groups, s50_mam_in_groups, s75_mam_in_groups, s95_mam_in_groups, s99_mam_in_groups, 
                    s01_uniformComp_in_groups, s05_uniformComp_in_groups, s25_uniformComp_in_groups, s50_uniformComp_in_groups, s75_uniformComp_in_groups, s95_uniformComp_in_groups, s99_uniformComp_in_groups,
                    s01_partmc_uniformComp_in_groups, s05_partmc_uniformComp_in_groups, s25_partmc_uniformComp_in_groups, s50_partmc_uniformComp_in_groups, s75_partmc_uniformComp_in_groups, s95_partmc_uniformComp_in_groups, s99_partmc_uniformComp_in_groups, 
                    s01_mam_uniformComp_in_groups, s05_mam_uniformComp_in_groups, s25_mam_uniformComp_in_groups, s50_mam_uniformComp_in_groups, s75_mam_uniformComp_in_groups, s95_mam_uniformComp_in_groups, s99_mam_uniformComp_in_groups, 
                    s01_partmc_condOnly_in_groups, s05_partmc_condOnly_in_groups, s25_partmc_condOnly_in_groups, s50_partmc_condOnly_in_groups, s75_partmc_condOnly_in_groups, s95_partmc_condOnly_in_groups, s99_partmc_condOnly_in_groups, 
                    s01_mam_condOnly_in_groups, s05_mam_condOnly_in_groups, s25_mam_condOnly_in_groups, s50_mam_condOnly_in_groups, s75_mam_condOnly_in_groups, s95_mam_condOnly_in_groups, s99_mam_condOnly_in_groups, 
                    s01_partmc_coagOnly_in_groups, s05_partmc_coagOnly_in_groups, s25_partmc_coagOnly_in_groups, s50_partmc_coagOnly_in_groups, s75_partmc_coagOnly_in_groups, s95_partmc_coagOnly_in_groups, s99_partmc_coagOnly_in_groups, 
                    s01_mam_coagOnly_in_groups, s05_mam_coagOnly_in_groups, s25_mam_coagOnly_in_groups, s50_mam_coagOnly_in_groups, s75_mam_coagOnly_in_groups, s95_mam_coagOnly_in_groups, s99_mam_coagOnly_in_groups,
                    errFracCCN_01_in_groups, errFracCCN_03_in_groups, errFracCCN_10_in_groups,
                    D50_ratio_in_groups,
                    sa_flux_in_groups,num_conc_in_groups,integrated_sa_in_groups,integrated_num_in_groups)):
            for (
                    one_scenario,one_KL,one_KL_backward,
                    s01, s05, s25, s50, s75, s95, s99, 
                    s01_partmc, s05_partmc, s25_partmc, s50_partmc, s75_partmc, s95_partmc, s99_partmc, 
                    s01_mam, s05_mam,  s25_mam, s50_mam, s75_mam, s95_mam, s99_mam, 
                    s01_uniformComp, s05_uniformComp, s25_uniformComp, s50_uniformComp, s75_uniformComp, s95_uniformComp, s99_uniformComp, 
                    s01_partmc_uniformComp, s05_partmc_uniformComp, s25_partmc_uniformComp, s50_partmc_uniformComp, s75_partmc_uniformComp, s95_partmc_uniformComp, s99_partmc_uniformComp, 
                    s01_mam_uniformComp, s05_mam_uniformComp,  s25_mam_uniformComp, s50_mam_uniformComp, s75_mam_uniformComp, s95_mam_uniformComp, s99_mam_uniformComp, 
                    s01_partmc_condOnly, s05_partmc_condOnly, s25_partmc_condOnly, s50_partmc_condOnly, s75_partmc_condOnly, s95_partmc_condOnly, s99_partmc_condOnly, 
                    s01_mam_condOnly, s05_mam_condOnly,  s25_mam_condOnly, s50_mam_condOnly, s75_mam_condOnly, s95_mam_condOnly, s99_mam_condOnly, 
                    s01_partmc_coagOnly, s05_partmc_coagOnly, s25_partmc_coagOnly, s50_partmc_coagOnly, s75_partmc_coagOnly, s95_partmc_coagOnly, s99_partmc_coagOnly, 
                    s01_mam_coagOnly, s05_mam_coagOnly,  s25_mam_coagOnly, s50_mam_coagOnly, s75_mam_coagOnly, s95_mam_coagOnly, s99_mam_coagOnly, 
                    errFracCCN_01, errFracCCN_03, errFracCCN_10,
                    D50_ratio,
                    one_sa_flux,one_num_conc,one_integrated_sa,one_integrated_num) in zip(
                        one_scenario_group,one_KL_group,one_KL_backward_group,
                        s01_group, s05_group, s25_group, s50_group, s75_group, s95_group, s99_group,
                        s01_partmc_group, s05_partmc_group, s25_partmc_group, s50_partmc_group, s75_partmc_group, s95_partmc_group, s99_partmc_group, 
                        s01_mam_group, s05_mam_group, s25_mam_group, s50_mam_group, s75_mam_group, s95_mam_group, s99_mam_group, 
                        s01_uniformComp_group, s05_uniformComp_group, s25_uniformComp_group, s50_uniformComp_group, s75_uniformComp_group, s95_uniformComp_group, s99_uniformComp_group,
                        s01_partmc_uniformComp_group, s05_partmc_uniformComp_group, s25_partmc_uniformComp_group, s50_partmc_uniformComp_group, s75_partmc_uniformComp_group, s95_partmc_uniformComp_group, s99_partmc_uniformComp_group, 
                        s01_mam_uniformComp_group, s05_mam_uniformComp_group, s25_mam_uniformComp_group, s50_mam_uniformComp_group, s75_mam_uniformComp_group, s95_mam_uniformComp_group, s99_mam_uniformComp_group, 
                        s01_partmc_condOnly_group, s05_partmc_condOnly_group, s25_partmc_condOnly_group, s50_partmc_condOnly_group, s75_partmc_condOnly_group, s95_partmc_condOnly_group, s99_partmc_condOnly_group, 
                        s01_mam_condOnly_group, s05_mam_condOnly_group, s25_mam_condOnly_group, s50_mam_condOnly_group, s75_mam_condOnly_group, s95_mam_condOnly_group, s99_mam_condOnly_group, 
                        s01_partmc_coagOnly_group, s05_partmc_coagOnly_group, s25_partmc_coagOnly_group, s50_partmc_coagOnly_group, s75_partmc_coagOnly_group, s95_partmc_coagOnly_group, s99_partmc_coagOnly_group, 
                        s01_mam_coagOnly_group, s05_mam_coagOnly_group, s25_mam_coagOnly_group, s50_mam_coagOnly_group, s75_mam_coagOnly_group, s95_mam_coagOnly_group, s99_mam_coagOnly_group, 
                        errFracCCN_01_group, errFracCCN_03_group, errFracCCN_10_group,
                        D50_ratio_group,
                        one_sa_flux_group,one_num_conc_group,one_integrated_sa_group,one_integrated_num_group):
                all_scenarios.append(one_scenario)
                all_regimes.append(one_regime)
                all_regime_ids.append(qq)
                all_timesteps.append(timestep)
                all_times.append(one_time)
                # if one_time == 0.:
                #     all_s01.append(1.)
                #     all_s05.append(1.)
                #     all_s25.append(1.)
                #     all_s50.append(1.)
                #     all_s75.append(1.)
                #     all_s95.append(1.)
                #     all_s99.append(1.)                
                # else:
                all_s01.append(s01)
                all_s05.append(s05)
                all_s25.append(s25)
                all_s50.append(s50)
                all_s75.append(s75)
                all_s95.append(s95)
                all_s99.append(s99)
                
                all_s01_partmc.append(s01_partmc)
                all_s05_partmc.append(s05_partmc)
                all_s25_partmc.append(s25_partmc)
                all_s50_partmc.append(s50_partmc)
                all_s75_partmc.append(s75_partmc)
                all_s95_partmc.append(s95_partmc)
                all_s99_partmc.append(s99_partmc)
                
                all_s01_mam.append(s01_mam)
                all_s05_mam.append(s05_mam)
                all_s25_mam.append(s25_mam)
                all_s50_mam.append(s50_mam)
                all_s75_mam.append(s75_mam)
                all_s95_mam.append(s95_mam)
                all_s99_mam.append(s99_mam)
                
                all_s01_uniformComp.append(s01_uniformComp)
                all_s05_uniformComp.append(s05_uniformComp)
                all_s25_uniformComp.append(s25_uniformComp)
                all_s50_uniformComp.append(s50_uniformComp)
                all_s75_uniformComp.append(s75_uniformComp)
                all_s95_uniformComp.append(s95_uniformComp)
                all_s99_uniformComp.append(s99_uniformComp)
                
                all_s01_partmc_uniformComp.append(s01_partmc_uniformComp)
                all_s05_partmc_uniformComp.append(s05_partmc_uniformComp)
                all_s25_partmc_uniformComp.append(s25_partmc_uniformComp)
                all_s50_partmc_uniformComp.append(s50_partmc_uniformComp)
                all_s75_partmc_uniformComp.append(s75_partmc_uniformComp)
                all_s95_partmc_uniformComp.append(s95_partmc_uniformComp)
                all_s99_partmc_uniformComp.append(s99_partmc_uniformComp)
                
                all_s01_mam_uniformComp.append(s01_mam_uniformComp)
                all_s05_mam_uniformComp.append(s05_mam_uniformComp)
                all_s25_mam_uniformComp.append(s25_mam_uniformComp)
                all_s50_mam_uniformComp.append(s50_mam_uniformComp)
                all_s75_mam_uniformComp.append(s75_mam_uniformComp)
                all_s95_mam_uniformComp.append(s95_mam_uniformComp)
                all_s99_mam_uniformComp.append(s99_mam_uniformComp)
                
                all_s01_partmc_condOnly.append(s01_partmc_condOnly)
                all_s05_partmc_condOnly.append(s05_partmc_condOnly)
                all_s25_partmc_condOnly.append(s25_partmc_condOnly)
                all_s50_partmc_condOnly.append(s50_partmc_condOnly)
                all_s75_partmc_condOnly.append(s75_partmc_condOnly)
                all_s95_partmc_condOnly.append(s95_partmc_condOnly)
                all_s99_partmc_condOnly.append(s99_partmc_condOnly)
                
                all_s01_mam_condOnly.append(s01_mam_condOnly)
                all_s05_mam_condOnly.append(s05_mam_condOnly)
                all_s25_mam_condOnly.append(s25_mam_condOnly)
                all_s50_mam_condOnly.append(s50_mam_condOnly)
                all_s75_mam_condOnly.append(s75_mam_condOnly)
                all_s95_mam_condOnly.append(s95_mam_condOnly)
                all_s99_mam_condOnly.append(s99_mam_condOnly)

                all_s01_partmc_coagOnly.append(s01_partmc_coagOnly)
                all_s05_partmc_coagOnly.append(s05_partmc_coagOnly)
                all_s25_partmc_coagOnly.append(s25_partmc_coagOnly)
                all_s50_partmc_coagOnly.append(s50_partmc_coagOnly)
                all_s75_partmc_coagOnly.append(s75_partmc_coagOnly)
                all_s95_partmc_coagOnly.append(s95_partmc_coagOnly)
                all_s99_partmc_coagOnly.append(s99_partmc_coagOnly)
                
                all_s01_mam_coagOnly.append(s01_mam_coagOnly)
                all_s05_mam_coagOnly.append(s05_mam_coagOnly)
                all_s25_mam_coagOnly.append(s25_mam_coagOnly)
                all_s50_mam_coagOnly.append(s50_mam_coagOnly)
                all_s75_mam_coagOnly.append(s75_mam_coagOnly)
                all_s95_mam_coagOnly.append(s95_mam_coagOnly)
                all_s99_mam_coagOnly.append(s99_mam_coagOnly)
                
                all_errFracCCN_01.append(errFracCCN_01)
                all_errFracCCN_03.append(errFracCCN_03)
                all_errFracCCN_10.append(errFracCCN_10)
                all_D50.append(D50_ratio)
                all_KL.append(one_KL)
                all_KL_backward.append(one_KL_backward)
                all_sa_flux.append(one_sa_flux)
                all_num_conc.append(one_num_conc)
                all_integrated_sa.append(one_integrated_sa)
                all_integrated_num.append(one_integrated_num)
                
    
    all_times = [timestep*dtime for timestep in all_timesteps]
    big_list = [all_scenarios,all_regimes,all_regime_ids,all_timesteps,all_times,
                all_s01, all_s05, all_s25, all_s50, all_s75, all_s95, all_s99,
                all_s01_partmc, all_s05_partmc, all_s25_partmc, all_s50_partmc, all_s75_partmc, all_s95_partmc, all_s99_partmc,
                all_s01_mam, all_s05_mam, all_s25_mam, all_s50_mam, all_s75_mam, all_s95_mam, all_s99_mam,
                all_s01_uniformComp, all_s05_uniformComp, all_s25_uniformComp, all_s50_uniformComp, all_s75_uniformComp, all_s95_uniformComp, all_s99,
                all_s01_partmc_uniformComp, all_s05_partmc_uniformComp, all_s25_partmc_uniformComp, all_s50_partmc_uniformComp, all_s75_partmc_uniformComp, all_s95_partmc_uniformComp, all_s99_partmc,
                all_s01_mam_uniformComp, all_s05_mam_uniformComp, all_s25_mam_uniformComp, all_s50_mam_uniformComp, all_s75_mam_uniformComp, all_s95_mam_uniformComp, all_s99_mam,
                all_s01_partmc_condOnly, all_s05_partmc_condOnly, all_s25_partmc_condOnly, all_s50_partmc_condOnly, all_s75_partmc_condOnly, all_s95_partmc_condOnly, all_s99_partmc_condOnly,
                all_s01_mam_condOnly, all_s05_mam_condOnly, all_s25_mam_condOnly, all_s50_mam_condOnly, all_s75_mam_condOnly, all_s95_mam_condOnly, all_s99_mam_condOnly,
                all_s01_partmc_coagOnly, all_s05_partmc_coagOnly, all_s25_partmc_coagOnly, all_s50_partmc_coagOnly, all_s75_partmc_coagOnly, all_s95_partmc_coagOnly, all_s99_partmc_coagOnly,
                all_s01_mam_coagOnly, all_s05_mam_coagOnly, all_s25_mam_coagOnly, all_s50_mam_coagOnly, all_s75_mam_coagOnly, all_s95_mam_coagOnly, all_s99_mam_coagOnly,
                all_errFracCCN_01, all_errFracCCN_03, all_errFracCCN_10,
                all_D50, all_KL,all_KL_backward,all_sa_flux,all_num_conc,all_integrated_sa,all_integrated_num]
    # data = []
    # for one_item in big_list:
    #     data.append(one_item)
    # # np.array().transpose()
    
    data = np.array(big_list).transpose()
    columns = [
        'scenarios','regimes','regime_ids','timesteps','time',
        's_onset_ratio_01', 's_onset_ratio_05', 's_onset_ratio_25', 's_onset_ratio_50', 's_onset_ratio_75', 's_onset_ratio_95', 's_onset_ratio_99', 
        's_onset_partmc_01', 's_onset_partmc_05', 's_onset_partmc_25', 's_onset_partmc_50', 's_onset_partmc_75', 's_onset_partmc_95', 's_onset_partmc_99', 
        's_onset_mam4_01', 's_onset_mam4_05', 's_onset_mam4_25', 's_onset_mam4_50', 's_onset_mam4_75', 's_onset_mam4_95', 's_onset_mam4_99', 
        's_onset_ratio_uniformComp_01', 's_onset_ratio_uniformComp_05', 's_onset_ratio_uniformComp_25', 's_onset_ratio_uniformComp_50', 's_onset_ratio_uniformComp_75', 's_onset_ratio_uniformComp_95', 's_onset_ratio_uniformComp_99', 
        's_onset_partmc_uniformComp_01', 's_onset_partmc_uniformComp_05', 's_onset_partmc_uniformComp_25', 's_onset_partmc_uniformComp_50', 's_onset_partmc_uniformComp_75', 's_onset_partmc_uniformComp_95', 's_onset_partmc_uniformComp_99', 
        's_onset_mam4_uniformComp_01', 's_onset_mam4_uniformComp_05', 's_onset_mam4_uniformComp_25', 's_onset_mam4_uniformComp_50', 's_onset_mam4_uniformComp_75', 's_onset_mam4_uniformComp_95', 's_onset_mam4_uniformComp_99', 
        's_onset_partmc_condOnly_01', 's_onset_partmc_condOnly_05', 's_onset_partmc_condOnly_25', 's_onset_partmc_condOnly_50', 's_onset_partmc_condOnly_75', 's_onset_partmc_condOnly_95', 's_onset_partmc_condOnly_99', 
        's_onset_mam4_condOnly_01', 's_onset_mam4_condOnly_05', 's_onset_mam4_condOnly_25', 's_onset_mam4_condOnly_50', 's_onset_mam4_condOnly_75', 's_onset_mam4_condOnly_95', 's_onset_mam4_condOnly_99', 
        's_onset_partmc_coagOnly_01', 's_onset_partmc_coagOnly_05', 's_onset_partmc_coagOnly_25', 's_onset_partmc_coagOnly_50', 's_onset_partmc_coagOnly_75', 's_onset_partmc_coagOnly_95', 's_onset_partmc_coagOnly_99', 
        's_onset_mam4_coagOnly_01', 's_onset_mam4_coagOnly_05', 's_onset_mam4_coagOnly_25', 's_onset_mam4_coagOnly_50', 's_onset_mam4_coagOnly_75', 's_onset_mam4_coagOnly_95', 's_onset_mam4_coagOnly_99', 
        'errFracCCN_01', 'errFracCCN_03', 'errFracCCN_10',
        'D50_ratio','KL','KL_backward','sa_flux','num_conc','integrated_sa_flux','integrated_num_conc']
    
    df = pandas.DataFrame(data=data,columns=columns)
    return df
    
def get_save_dir(ensemble_dir):
    ensemble_name = ensemble_dir[(ensemble_dir.rfind('/',0,-1)+1):-1]
    save_dir = '../figs/' + ensemble_name + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return save_dir


# =============================================================================
# old figures
# =============================================================================

def error_emulator(
        e3sm_filename,ensemble_dir,timesteps,tt=0,lev=-1,
        yvarname='KL',
        Xvarnames = ['logN0_1','logN0_2','logN0_3','logN0_4','dlogN_1','dlogN_2','dlogN_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_4'],
        fig_dir='figs/',yscale='log',save=True,clims=None,regressor='mlp',solver='adam'):
    # Xvarnames = ['dmu_1','dmu_2','dNnorm_1','dNnorm_2','dNnorm_4']    
    # Xvarnames = ['dNnorm_1','dNnorm_2','dNnorm_4','N0norm_1','N0norm_2','N0norm_3','N0norm_4']    
    # Xvarnames = ['dmu_1','dmu_2','dNnorm_1','dNnorm_2','dNnorm_4','N0norm_1','N0norm_2','N0norm_3','N0norm_4']
    # Xvarnames = ['dNnorm_1','dNnorm_2','dNnorm_4','dmu_2','Nnorm_3']
    # Xvarnames = ['dmu_1','dmu_2','dmu_3','dNnorm_1','dNnorm_2','dNnorm_3','dNnorm_4']
    # Xvarnames = ['N0_1','N0_2','N0_3','N0_4','dN_1','dN_2','dN_3','dN_4']
    # Xvarnames = ['N0norm_1','N0norm_2','N0norm_3','N0norm_4','dNnorm_1','dNnorm_2','dNnorm_3','dNnorm_4','dmu_1','dmu_2','dmu_3','dmu_4']
    # Xvarnames = ['N0_1','N0_2','N0_3','N0_4','dlogN0_1','dlogN0_2','dlogN0_3','dlogN0_4']
    # Xvarnames = ['logN0_1','logN0_2','logN0_3','logN0_4','dlogN_1','dlogN_2','dlogN_3','dlogN_4']
    # Xvarnames = ['N0norm_1','N0norm_2','N0norm_3','N0norm_4','dNnorm_1','dNnorm_2','dNnorm_3','dNnorm_4','dmu_1','dmu_2','dmu_3','dmu_4']
    # Xvarnames = ['N0norm_1','N0norm_2','N0norm_3','N0norm_4','dNnorm_1','dNnorm_2','dNnorm_3','dNnorm_4','dD_1','dD_2','dD_3','dD_4']
    
    X_train,X_test,y_test,y_train = analyses.get_Xy_process(Xvarnames,yvarname,ensemble_dir,timesteps)
    # X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # regr = analyses.train_surrogate_model(X_train,y_train,regressor='mlp',solver='lbfgs',max_iter=5000)
    # regr = analyses.train_surrogate_model(X_train,y_train,regressor='mlp',solver='sgd',max_iter=5000)
    fig,ax = plt.subplots(1)
    if yscale == 'log':
        regr = analyses.train_surrogate_model(X_train,np.log10(y_train),regressor=regressor,solver=solver,max_iter=5000)
        ax.scatter(y_test,10.**regr.predict(X_test)); 
    else:
        regr = analyses.train_surrogate_model(X_train,y_train,regressor=regressor,solver=solver,max_iter=5000)
        ax.scatter(y_test,regr.predict(X_test)); 
    ax.set_xlabel('KL-divergence from benchmarking')
    ax.set_ylabel('KL-divergence from emulator')
    ax.plot([0.,max(y_test)*1.05],[0.,max(y_test)*1.05],color='k',linestyle='--')
    ax.set_xlim([0.,max(y_test)*1.05])
    ax.set_ylim([0.,max(y_test)*1.05])
    if save:
        fig.savefig(fig_dir + 'KLemulator_scatter.png')    
    plt.show()
    
    start1 = time.time()
    e3sm_filename = '/Users/fier887/Downloads/PartMC_test_map_05.nc'
    f = Dataset(e3sm_filename)
    
    t, lev, lat, lon = analyses.get_e3sm_grid(f)
    
    X_e3sm_raveled,the_shape = analyses.get_X_e3sm(Xvarnames,f,tt=0,lev=-1,return_shape=True)
    
    for ii in range(len(Xvarnames)):
        X_e3sm_gridded = analyses.unravel_y(X_e3sm_raveled[:,ii],the_shape)
        c = plt.pcolor(lon,lat,X_e3sm_gridded)
        plt.title(Xvarnames[ii])
        plt.colorbar()
        plt.show()
    
    # for ii in range(len(Xvarnames)):
    #     print(ii,Xvarnames[ii])
    #     print('min',np.min(X_e3sm_raveled[:,ii]))
    #     print('max',np.max(X_e3sm_raveled[:,ii]))
    end1 = time.time()
    start2 = time.time()
    y_e3sm_raveled = regr.predict(X_e3sm_raveled)
    y_e3sm_gridded = analyses.unravel_y(y_e3sm_raveled,the_shape)
    end2 = time.time()
    fig,ax = plt.subplots(1)
    
    print(end1-start1,end2-start2)
    if yscale == 'log':
        c = ax.pcolor(lon,lat,10.**y_e3sm_gridded)
    else:
        c = ax.pcolor(lon,lat,y_e3sm_gridded)
    if clims == None:
        clims = [0.,c.get_clim()[1]]
    c.set_clim(clims) 
    cbar = fig.colorbar(c,ax=ax,label='KL-divergence after 30 min')
    # plt.clabel('KL-divergence after 30~min')
    if save:
        plt.savefig(fig_dir + 'errorEmulator_map_' + yscale + '.png')

def error_emulator_2regressions(
        e3sm_filename,ensemble_dir,timesteps,
        Xvarnames = ['logN0_1','logN0_2','logN0_3','logN0_4','dlogN_1','dlogN_2','dlogN_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_4'],
        Xvarnames2 = ['logN0_1'],
        yvarname='KL',fig_dir='figs/',yscale='log',save=True,clims=None,regressor='mlp',solver='adam',max_iter=10000,hidden_layer_sizes=(100,)):
    
    X_train,X_test,y_test,y_train = analyses.get_Xy_process(Xvarnames,yvarname,ensemble_dir,timesteps)
    
    fig,ax = plt.subplots(1)
    if yscale == 'log':
        idx, = np.where(~np.any(np.isinf(X_train) | np.isnan(X_train),axis=1))
        regr = analyses.train_surrogate_model(X_train[idx,:],np.log10(y_train[idx]),regressor=regressor,solver=solver,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)
        idx2, = np.where(~np.any(np.isinf(X_test) | np.isnan(X_test),axis=1))
        ax.scatter(y_test[idx2],10.**regr.predict(X_test[idx2,:]),alpha=0.05);
    else:
        idx, = np.where(~np.any(np.isinf(X_train) | np.isnan(X_train),axis=1))
        regr = analyses.train_surrogate_model(X_train[idx,:],y_train[idx],regressor=regressor,solver=solver,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)
        idx2, = np.where(~np.any(np.isinf(X_test) | np.isnan(X_test),axis=1))
        ax.scatter(y_test[idx2],regr.predict(X_test[idx2,:]),alpha=0.05); 
    
    ax.set_xlabel('KL-divergence from benchmarking')
    ax.set_ylabel('KL-divergence from emulator')
    ax.plot([0.,max(y_test)*1.05],[0.,max(y_test)*1.05],color='k',linestyle='--')
    ax.set_xlim([0.,max(y_test)*1.05])
    ax.set_ylim([0.,max(y_test)*1.05])
    if save:
        fig.savefig(fig_dir + 'KLemulator_scatter.png')    
    plt.show()
    
    X_train,X_test,y_test,y_train = analyses.get_Xy_process(Xvarnames2,yvarname,ensemble_dir,timesteps)
    if yscale == 'log':
        idx, = np.where(~np.any(np.isinf(X_train) | np.isnan(X_train),axis=1))
        regr2 = analyses.train_surrogate_model(X_train[idx,:],np.log10(y_train[idx]),regressor=regressor,solver=solver,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)
        idx2, = np.where(~np.any(np.isinf(X_test) | np.isnan(X_test),axis=1))
        ax.scatter(y_test[idx2],10.**regr.predict(X_test[idx2,:]),alpha=0.05);
    else:
        idx, = np.where(~np.any(np.isinf(X_train) | np.isnan(X_train),axis=1))
        regr2 = analyses.train_surrogate_model(X_train[idx,:],y_train[idx],regressor=regressor,solver=solver,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)
        idx2, = np.where(~np.any(np.isinf(X_test) | np.isnan(X_test),axis=1))
    ax.scatter(y_test[idx2],regr2.predict(X_test[idx2,:]),alpha=0.05); 
    ax.set_xlabel('KL-divergence from benchmarking')
    ax.set_ylabel('KL-divergence from emulator')
    ax.plot([0.,max(y_test)*1.05],[0.,max(y_test)*1.05],color='k',linestyle='--')
    ax.set_xlim([0.,max(y_test)*1.05])
    ax.set_ylim([0.,max(y_test)*1.05])
    if save:
        fig.savefig(fig_dir + 'KLemulator_scatter2.png')    
    plt.show()
    
    start1 = time.time()
    e3sm_filename = '/Users/fier887/Downloads/PartMC_test_map_05.nc'
    f = Dataset(e3sm_filename)
    
    t, lev, lat, lon = analyses.get_e3sm_grid(f)
    
    X_e3sm_raveled,the_shape = analyses.get_X_e3sm(Xvarnames,f,tt=0,lev=-1,return_shape=True)
    X_e3sm_raveled2,the_shape = analyses.get_X_e3sm(Xvarnames,f,tt=0,lev=-1,return_shape=True)
    
    for ii in range(len(Xvarnames)):
        fig,ax = plt.subplots(1)
        X_e3sm_gridded = analyses.unravel_y(X_e3sm_raveled[:,ii],the_shape)
        c = ax.pcolor(lon,lat,X_e3sm_gridded)
        plt.title(Xvarnames[ii])
        plt.colorbar(c)
        plt.show()
    
    y_e3sm_raveled = regr.predict(X_e3sm_raveled)
    y_e3sm_gridded = analyses.unravel_y(y_e3sm_raveled,the_shape)
    
    fig,ax = plt.subplots(1)
    if yscale == 'log':
        c = ax.pcolor(lon,lat,10.**y_e3sm_gridded)
    else:
        c = ax.pcolor(lon,lat,y_e3sm_gridded)
    
    if clims == None:
        clims = [0.,c.get_clim()[1]]
    
    c.set_clim(clims) 
    cbar = fig.colorbar(c,ax=ax,label='KL-divergence after 30 min')
    # plt.clabel('KL-divergence after 30~min')
    if save:
        plt.savefig(fig_dir + 'errorEmulator_map_' + yscale + '.png')
    return X_train,X_test,y_test,y_train,X_e3sm_raveled,y_e3sm_raveled,the_shape


def error_emulator_splitRepeats(
        e3sm_filename,ensemble_dir,timesteps,dtime=600.,
        Xvarnames = ['logN0_1','logN0_2','logN0_3','logN0_4','dlogN_1','dlogN_2','dlogN_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_4'],
        yvarname='s_onset_ratio_50',fig_dir='figs/',yscale='log',save=True,clims=None,regressor='mlp',solver='adam',max_iter=10000,hidden_layer_sizes=(100,)):
    # Xvarnames = ['dmu_1','dmu_2','dNnorm_1','dNnorm_2','dNnorm_4']    
    # Xvarnames = ['dNnorm_1','dNnorm_2','dNnorm_4','N0norm_1','N0norm_2','N0norm_3','N0norm_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_3','dmu_4']
    # Xvarnames = ['dmu_1','dmu_2','dNnorm_1','dNnorm_2','dNnorm_4','N0norm_1','N0norm_2','N0norm_3','N0norm_4']
    # Xvarnames = ['dNnorm_1','dNnorm_2','dNnorm_4','dmu_2','Nnorm_3']
    # Xvarnames = ['dmu_1','dmu_2','dmu_3','dNnorm_1','dNnorm_2','dNnorm_3','dNnorm_4']
    # Xvarnames = ['N0_1','N0_2','N0_3','N0_4','mu0_1','mu0_2','mu0_3','mu0_4','dN_1','dN_2','dN_3','dN_4','dmu_1','dmu_2','dmu_3','dmu_4']
    # Xvarnames = ['N0norm_1','N0norm_2','N0norm_3','N0norm_4','dNnorm_1','dNnorm_2','dNnorm_3','dNnorm_4','dmu_1','dmu_2','dmu_3','dmu_4']
    # Xvarnames = ['N0_1','N0_2','N0_3','N0_4','dlogN0_1','dlogN0_2','dlogN0_3','dlogN0_4']
    # Xvarnames = ['dlogN_1','dlogN_2','dlogN_3','dlogN_4','dmu_1','dmu_2','dmu_3','dmu_4']
    # Xvarnames = ['logN0_1','logN0_2','logN0_3','logN0_4','dlogN_1','dlogN_2','dlogN_3','dlogN_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_3','dmu_4']
    # Xvarnames = ['N0_1','N0_2','N0_3','N0_4','dN_1','dN_2','dN_3','dN_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_3','dmu_4']
    # Xvarnames = ['logN0_1','logN0_2','logN0_3','logN0_4','dlogN_1','dlogN_2','dlogN_3','dlogN_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_3','dmu_4']
    # Xvarnames = ['logN0_1','logN0_2','logN0_3','logN0_4','dlogN_1','dlogN_2','dlogN_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_4']
    
    # X_train,X_test,y_test,y_train = analyses.get_Xy_process(Xvarnames,'KL_backward',ensemble_dir,timesteps)
    X_train,X_test,y_test,y_train = analyses.get_Xy_process(Xvarnames,yvarname,ensemble_dir,timesteps,dtime=dtime)
    
    # X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # regr = analyses.train_surrogate_model(X_train,y_train,regressor='mlp',solver='lbfgs',max_iter=5000)
    # regr = analyses.train_surrogate_model(X_train,y_train,regressor='mlp',solver='sgd',max_iter=5000)
    fig,ax = plt.subplots(1)
    if yscale == 'log':
        idx, = np.where(~np.any(np.isinf(X_train) | np.isnan(X_train),axis=1))
        regr = analyses.train_surrogate_model(X_train[idx,:],np.log10(y_train[idx]),regressor=regressor,solver=solver,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)
        idx2, = np.where(~np.any(np.isinf(X_test) | np.isnan(X_test),axis=1))
        ax.scatter(y_test[idx2],10.**regr.predict(X_test[idx2,:]),alpha=0.05);
    else:
        idx, = np.where(~np.any(np.isinf(X_train) | np.isnan(X_train),axis=1))
        regr = analyses.train_surrogate_model(X_train[idx,:],y_train[idx],regressor=regressor,solver=solver,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)
        idx2, = np.where(~np.any(np.isinf(X_test) | np.isnan(X_test),axis=1))
        ax.scatter(y_test[idx2],regr.predict(X_test[idx2,:]),alpha=0.05); 
    ax.set_xlabel('KL-divergence from benchmarking')
    ax.set_ylabel('KL-divergence from emulator')
    ax.plot([0.,max(y_test)*1.05],[0.,max(y_test)*1.05],color='k',linestyle='--')
    ax.set_xlim([0.,max(y_test)*1.05])
    ax.set_ylim([0.,max(y_test)*1.05])
    if save:
        fig.savefig(fig_dir + 'KLemulator_scatter.png')    
    plt.show()
    
    # fig,ax = plt.subplots(1)
    # start1 = time.time()
    e3sm_filename = '/Users/fier887/Downloads/PartMC_test_map_05.nc'
    f = Dataset(e3sm_filename)
    
    t, lev, lat, lon = analyses.get_e3sm_grid(f)
    
    X_e3sm_raveled,the_shape = analyses.get_X_e3sm(Xvarnames,f,tt=0,lev=-1,return_shape=True)
    
    for ii in range(len(Xvarnames)):
        fig,ax = plt.subplots(1)
        X_e3sm_gridded = analyses.unravel_y(X_e3sm_raveled[:,ii],the_shape)
        c = ax.pcolor(lon,lat,X_e3sm_gridded)
        plt.title(Xvarnames[ii])
        plt.colorbar(c)
        plt.show()
        
    # for ii in range(len(Xvarnames)):
    #     print(ii,Xvarnames[ii])
    #     print('min',np.min(X_e3sm_raveled[:,ii]))
    #     print('max',np.max(X_e3sm_raveled[:,ii]))
    # end1 = time.time()
    # start2 = time.time()
    idx, = np.where(~np.any(np.isinf(X_e3sm_raveled) | np.isnan(X_e3sm_raveled),axis=1))
    y_e3sm_raveled = np.ones_like(X_e3sm_raveled[:,0]) * np.nan
    y_e3sm_raveled[idx] = regr.predict(X_e3sm_raveled[idx,:])
    y_e3sm_gridded = analyses.unravel_y(y_e3sm_raveled,the_shape)
    # end2 = time.time()
    # print(end1-start1,end2-start2)
    fig,ax = plt.subplots(1)
    if yscale == 'log':
        c = ax.pcolor(lon,lat,10.**y_e3sm_gridded)
    else:
        c = ax.pcolor(lon,lat,y_e3sm_gridded)
    
    if clims == None:
        clims = [0.,c.get_clim()[1]]
    
    c.set_clim(clims) 
    cbar = fig.colorbar(c,ax=ax,label='KL-divergence after 30 min')
    # plt.clabel('KL-divergence after 30~min')
    if save:
        plt.savefig(fig_dir + 'errorEmulator_map_' + yscale + '.png')
    return X_train,X_test,y_test,y_train,X_e3sm_raveled,y_e3sm_raveled,the_shape

def optimize_gradiantBoostingRegression(
        ensemble_dir,timesteps,n_splits=None,
        Xvarnames = ['logN0_1','logN0_2','logN0_3','logN0_4','dlogN_1','dlogN_2','dlogN_3','dlogN_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_3','dmu_4'],
        yvarname = 'KL', yscale = 'lin', N_loops = 3, # number of times to loop through optimization process
        frac_testing=0.1,frac_validation=0.1,
        learning_rates = [0.1, 0.01,0.001],
        n_estimators = [50, 100, 200],
        max_depths = [1, 3, 9, 27],
        min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True),
        min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True),
        max_features = list(range(1,16)),
        learning_rate0 = 0.1,
        n_estimators0 = 100,
        max_depth0 = 3,
        min_samples_split0 = 2,
        min_samples_leaf0 = 1,
        max_features0 = None,
        error_metric='NMAE'):
    
    X_train,X_test,X_validation,y_test,y_train,y_validation = analyses.get_Xy_process(
        Xvarnames,yvarname,ensemble_dir,timesteps,
        frac_testing=frac_testing,
        frac_validation=frac_validation,
        return_validation=True)
    
    idx, = np.where(~np.any(np.isinf(X_train) | np.isnan(X_train),axis=1))
    idx2, = np.where(~np.any(np.isinf(X_test) | np.isnan(X_test),axis=1))
    idx3, = np.where(~np.any(np.isinf(X_validation) | np.isnan(X_validation),axis=1))
    
    
    X_train = X_train[idx,:]
    y_train = y_train[idx]
    X_test = X_test[idx2,:]
    y_test = y_test[idx2]
    X_validation = X_validation[idx3,:]
    y_validation = y_validation[idx3]
    
    
    NMAE_learning_rate = np.zeros([len(learning_rates),N_loops])
    NMAE_n_estimators = np.zeros([len(n_estimators),N_loops])
    NMAE_max_depth = np.zeros([len(max_depths),N_loops])
    NMAE_min_samples_split = np.zeros([len(min_samples_splits),N_loops])
    NMAE_min_samples_leaf = np.zeros([len(min_samples_leafs),N_loops])
    NMAE_max_features = np.zeros([len(max_features),N_loops])    
    
    if error_metric == 'NMAE':
        error_fun = lambda true_vals,predicted_vals: np.mean(abs(true_vals-predicted_vals))/np.mean(true_vals)
    elif error_metric == 'NRMSE':
        error_fun = lambda true_vals,predicted_vals: np.mean((true_vals-predicted_vals)**2.)**0.5/np.mean(true_vals)        
    
    for loop in range(N_loops):
        for jj,learning_rate in enumerate(learning_rates):
            regr = analyses.train_gbr_model(
                    X_train,y_train,
                    learning_rate0 = learning_rate,
                    n_estimators0 = n_estimators0,
                    max_depth0 = max_depth0,
                    min_samples_split0 = min_samples_split0,
                    min_samples_leaf0 = min_samples_leaf0,
                    max_features0 = max_features0)
            y_validation_predict = regr.predict(X_validation)
            NMAE_learning_rate[jj,loop] = error_fun(y_validation,y_validation_predict)
        plt.plot(learning_rates,NMAE_learning_rate[:,loop]); plt.title('learning_rate, loop:' + str(loop)); plt.show()
        
        for jj,n_estimator in enumerate(n_estimators):
            regr = analyses.train_gbr_model(
                    X_train,y_train,
                    learning_rate0 = learning_rate0,
                    n_estimators0 = n_estimator,
                    max_depth0 = max_depth0,
                    min_samples_split0 = min_samples_split0,
                    min_samples_leaf0 = min_samples_leaf0,
                    max_features0 = max_features0)
            y_validation_predict = regr.predict(X_validation)
            NMAE_n_estimators[jj,loop] = error_fun(y_validation,y_validation_predict)
        plt.plot(n_estimators,NMAE_n_estimators[:,loop]); plt.title('n_estimators, loop:' + str(loop)); plt.show()
        
        for jj,max_depth in enumerate(max_depths):
            regr = analyses.train_gbr_model(
                    X_train,y_train,
                    learning_rate0 = learning_rate0,
                    n_estimators0 = n_estimators0,
                    max_depth0 = max_depth0,
                    min_samples_split0 = min_samples_split0,
                    min_samples_leaf0 = min_samples_leaf0,
                    max_features0 = max_features0)
            y_validation_predict = regr.predict(X_validation)
            NMAE_max_depth[jj,loop] = error_fun(y_validation,y_validation_predict)
        plt.plot(max_depths,NMAE_max_depth[:,loop]); plt.title('max_depth, loop:' + str(loop)); plt.show()
        
        for jj,min_samples_split in enumerate(min_samples_splits):
            regr = analyses.train_gbr_model(
                    X_train,y_train,
                    learning_rate0 = learning_rate0,
                    n_estimators0 = n_estimators0,
                    max_depth0 = max_depth0,
                    min_samples_split0 = min_samples_split,
                    min_samples_leaf0 = min_samples_leaf0,
                    max_features0 = max_features0)
            y_validation_predict = regr.predict(X_validation)
            NMAE_min_samples_split[jj,loop] = error_fun(y_validation,y_validation_predict)
        plt.plot(min_samples_splits,NMAE_min_samples_split[:,loop]); plt.title('min_samples_split, loop:' + str(loop)); plt.show()
        
        for jj,min_samples_leaf in enumerate(min_samples_leafs):
            regr = analyses.train_gbr_model(
                    X_train,y_train,
                    learning_rate0 = learning_rate0,
                    n_estimators0 = n_estimators0,
                    max_depth0 = max_depth0,
                    min_samples_split0 = min_samples_split0,
                    min_samples_leaf0 = min_samples_leaf,
                    max_features0 = max_features0)
            y_validation_predict = regr.predict(X_validation)
            NMAE_min_samples_leaf[jj,loop] = error_fun(y_validation,y_validation_predict)
        plt.plot(min_samples_leafs,NMAE_min_samples_leaf[:,loop]); plt.title('min_samples_leaf, loop:' + str(loop)); plt.show()
        
        for jj,max_feature in enumerate(max_features):
            regr = analyses.train_gbr_model(
                    X_train,y_train,
                    learning_rate0 = learning_rate0,
                    n_estimators0 = n_estimators0,
                    max_depth0 = max_depth0,
                    min_samples_split0 = min_samples_split0,
                    min_samples_leaf0 = min_samples_leaf0,
                    max_features0 = max_feature)
            y_validation_predict = regr.predict(X_validation)
            NMAE_max_features[jj,loop] = error_fun(y_validation,y_validation_predict)
        plt.plot(max_features,NMAE_max_features[:,loop]); plt.title('max_features, loop:' + str(loop)); plt.show()
        
        idx,=np.where(NMAE_learning_rate[:,loop] == min(NMAE_learning_rate[:,loop]))
        learning_rate0 = learning_rates[idx[0]]
        idx,=np.where(NMAE_n_estimators[:,loop] == min(NMAE_n_estimators[:,loop]))
        n_estimators0 = n_estimators[idx[0]]
        idx,=np.where(NMAE_n_estimators[:,loop] == min(NMAE_n_estimators[:,loop]))
        max_depth0 = max_depths[idx[0]]
        idx,=np.where(NMAE_min_samples_split[:,loop] == min(NMAE_min_samples_split[:,loop]))
        min_samples_split0 = min_samples_splits[idx[0]]
        idx,=np.where(NMAE_min_samples_leaf[:,loop] == min(NMAE_min_samples_leaf[:,loop]))
        min_samples_leaf0 = min_samples_leafs[idx[0]]
        idx,=np.where(NMAE_max_features[:,loop] == min(NMAE_max_features[:,loop]))
        max_features0 = max_features[idx[0]]
    
    regr = analyses.train_gbr_model(
        X_train,y_train,
        learning_rate0 = learning_rate0,
        n_estimators0 = n_estimator,
        max_depth0 = max_depth0,
        min_samples_split0 = min_samples_split0,
        min_samples_leaf0 = min_samples_leaf0,
        max_features0 = max_features0)
    y_test_predict = regr.predict(X_test)
    
    return learning_rate0, n_estimators0, max_depth0, min_samples_split0, min_samples_leaf0, max_features0, y_test, y_test_predict

def error_emulator_kfold(
        ensemble_dir,timesteps,n_splits=None,
        Xvarnames = ['logN0_1','logN0_2','logN0_3','logN0_4','dlogN_1','dlogN_2','dlogN_3','dlogN_4','mu0_1','mu0_2','mu0_3','mu0_4','dmu_1','dmu_2','dmu_3','dmu_4'],
        yvarname='KL',yscale='lin',regressor='mlp',solver='lbfgs',max_iter=5000,hidden_layer_sizes=(100,)):
    # kf = KFold(n_splits=10)
    
    
    Xy_folds = analyses.get_Xy_kfold(Xvarnames,yvarname,ensemble_dir,timesteps,n_splits=n_splits)
    MAE = []
    
    all_ytest = []
    all_ytest_predict = []
    for fold in Xy_folds:
        X_train = fold[0]
        y_train = fold[1]
        X_test = fold[2]
        y_test = fold[3]
        
        if ~np.any(np.isinf(X_test) | np.isnan(X_test)):
            if yscale == 'log':
                idx, = np.where(~np.any(np.isinf(X_train) | np.isnan(X_train),axis=1))
                regr = analyses.train_surrogate_model(X_train[idx,:],np.log10(y_train[idx]),regressor=regressor,solver=solver,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)
                # idx2, = np.where(~np.any(np.isinf(X_test) | np.isnan(X_test),axis=1))
                y_test_predict = 10.**regr.predict(X_test)
            else:
                idx, = np.where(~np.any(np.isinf(X_train) | np.isnan(X_train),axis=1))
                regr = analyses.train_surrogate_model(X_train[idx,:],y_train[idx],regressor=regressor,solver=solver,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)
                # idx2, = np.where(~np.any(np.isinf(X_test) | np.isnan(X_test),axis=1))
                y_test_predict = regr.predict(X_test)
            plt.scatter(y_test,y_test_predict);
            
            all_ytest.append(y_test)
            all_ytest_predict.append(y_test_predict)
            MAE.append(abs(y_test_predict - y_test))
            #R.append(np.corrcoef(y_test[idx2],y_test_predict))
    return MAE,all_ytest,all_ytest_predict
    # X_train,X_test,y_test,y_train = analyses.get_Xy_process(Xvarnames,'KL',ensemble_dir,timesteps)

