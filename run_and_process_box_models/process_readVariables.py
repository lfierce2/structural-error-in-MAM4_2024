#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import numpy as np
import netCDF4
from benchmark_splitToNodes import get_mam_input, get_partmc_input
from scipy.stats import norm
import read_partmc
from sklearn.neighbors import KernelDensity
import os
from scipy.stats import entropy
import PyMieScatt
import pickle
import scenario_runner

# split to separate nodes

def store_output__splitToNodes(
        ensemble_dir,tt_vals,
        dia_mids=np.logspace(-10,-3,1000),
        num_multiplier = 1./1.2930283496840569,
        gsds = [1.8,1.6,1.8,1.6], run_time_str = '00:30:00',
        modes=[1,2,3,4], 
        spec_names=['so4','pom','soa','bc','dst','ncl'],
        return_direct=False,
        mam_input='../mam_refactor-main/standalone/tests/smoke_test.nl',
        wvl=550e-9,
        specdat={}, mixingstate='part-res', density_type='hist',
        n_repeat=10, project='nuvola'):
    
    runfiles_dir = ensemble_dir + '../runfiles/'
    python_dir = scenario_runner.get_python_dir()
    
    all_scenarios = get_scenarios(ensemble_dir)
    for tt in tt_vals:
        # output_dict = {}
        for scenario in all_scenarios:
            scenario_str = str(scenario).zfill(6)
            mam_input = ensemble_dir + scenario_str + '/mam_input.nl'
            
            tt_dir = ensemble_dir + '../processed_output/tt_' + str(tt).zfill(4) + '/'
            output_file = tt_dir + 'scenario_' + scenario_str + '_output.pkl'
            if not os.path.exists(output_file):
                python_filename = runfiles_dir + 'main_process_' + scenario_str + '_' + str(tt).zfill(4) + '.py'
                
                dia_mids_str = '[' + ', '.join([str(one_dia) for one_dia in dia_mids]) + ']'
                
                python_lines = [
                    'import sys',
                    'import numpy as np',
                    'import pickle',
                    'import os',
                    'sys.path.insert(0, ' + '\'' + python_dir + '\')',
                    'import process_readVariables',
                    'output_dict = {}',
                    'tt = ' + str(tt),
                    'ensemble_dir = \'' + ensemble_dir + '\'',
                    'scenario_dir = \'' + ensemble_dir + scenario_str + '/\'',
                    'output_dict[\'scenario_dir\'] = scenario_dir',
                    'partmc_dict = process_readVariables.create_partmc_output_dict(',
                    '        scenario_dir,tt,',
                    '        wvl=550e-9,n_repeat=' + str(n_repeat) + ',',
                    '        specdat = {}, mixingstate=\'part-res\', density_type=\'hist\')',
                    ''
                    'mam4_dict = process_readVariables.create_mam4_output_dict(',
                    '        scenario_dir,tt,',
                    '        dia_mids=' + dia_mids_str + ',',
                    '        num_multiplier = 1./1.2930283496840569,', 
                    '        gsds = [1.8,1.6,1.8,1.6],',
                    '        modes=[1,2,3,4],',
                    '        spec_names=[\'so4\',\'pom\',\'soa\',\'bc\',\'dst\',\'ncl\'],',
                    '        mam_input=\'' + mam_input + '\')',
                    '',
                    'output_dict[\'partmc\'] = partmc_dict',
                    'output_dict[\'mam4\'] = mam4_dict',
                    'if not os.path.exists(ensemble_dir + \'../processed_output\'):',
                    '    os.mkdir(ensemble_dir + \'../processed_output\')',
                    'tt_dir = ' + '\'' + tt_dir + '\'',
                    'if not os.path.exists(tt_dir):',
                    '    os.mkdir(tt_dir)',
                    'output_file = tt_dir + \'scenario_' + scenario_str + '_output.pkl\'',
                    'pickle.dump(output_dict,open(output_file,\'wb\'),protocol=0)',
                    '']
            
                with open(python_filename, 'w') as f:
                    for line in python_lines:
                        f.write(line)
                        f.write('\n')
                f.close()
                
                sh_filename = runfiles_dir + scenario + '_' + str(tt).zfill(4) + '.sh'
                
                if os.path.exists('/people/fier887/'):
                    sh_lines = [
                        '#!/bin/tcsh',
                        '#SBATCH -A ' + project,
                        '#SBATCH -p shared',
                        '#SBATCH -t ' + run_time_str,
                        '#SBATCH -n 1',
                        '#SBATCH -o ' + runfiles_dir + 'process_' + scenario + '_' + str(tt).zfill(4) + '.out',
                        '#SBATCH -e ' + runfiles_dir + 'process_' + scenario + '_' + str(tt).zfill(4) + '.err',
                        '',
                        'python ' + python_filename]
                else:
                    print('not set up to run on this computer')
                
                with open(sh_filename, 'w') as f:
                    for line in sh_lines:
                        f.write(line)
                        f.write('\n')
                f.close()
                os.system('chmod +x ' + sh_filename)
                os.system('sbatch ' + sh_filename)
                
            # output_dict[scenario] = {}
            # scenario_dir = ensemble_dir + scenario + '/'
            # output_dict[scenario]['scenario_dir'] = scenario_dir
            
            # partmc_dict = create_partmc_output_dict(
            #         scenario_dir,tt,
            #         dia_mids=np.logspace(-10,-5,1000),
            #         wvl=550e-9,n_repeat=10,
            #         specdat = {}, mixingstate='part-res', density_type='hist')
            
            # mam4_dict = create_mam4_output_dict(
            #         scenario_dir,tt,
            #         dia_mids=np.logspace(-10,-5,1000),
            #         num_multiplier = 1./1.2930283496840569, 
            #         gsds = [1.8,1.6,1.8,1.6],
            #         modes=[1,2,3,4],
            #         spec_names=['so4','pom','soa','bc','dst','ncl'],
            #         return_direct=False,
            #         mam_input='../mam_refactor-main/standalone/tests/smoke_test.nl',
            #         wvl=550e-9)
            
            # output_dict['partmc'] = partmc_dict
            # output_dict['mam4'] = mam4_dict
            # tt_dir = ensemble_dir + 'processed_output/tt_' + str(tt).zfill(4) + '/'
            # output_file = tt_dir + 'scenario' + scenario + '_output.pkl'
            # pickle.dump(output_dict,open(output_file,'wb'),protocol=0)

def store_output_dicts(
        ensemble_dir,tt_vals,
        dia_mids=np.logspace(-10,-5,1000),
        num_multiplier = 1./1.2930283496840569, 
        gsds = [1.8,1.6,1.8,1.6],
        modes=[1,2,3,4],
        spec_names=['so4','pom','soa','bc','dst','ncl'],
        return_direct=False,
        mam_input='../mam_refactor-main/standalone/tests/smoke_test.nl',
        wvl=550e-9,
        specdat = {}, mixingstate='part-res', density_type='hist'):
    all_scenarios = get_scenarios(ensemble_dir)
    for tt in tt_vals:
        output_dict = {}
        for scenario in all_scenarios:
            output_dict[scenario] = {}
            scenario_dir = ensemble_dir + scenario + '/'
            output_dict[scenario]['scenario_dir'] = scenario_dir
            
            partmc_dict = create_partmc_output_dict(
                    scenario_dir,tt,
                    dia_mids=np.logspace(-10,-5,1000),
                    wvl=550e-9,n_repeat=10,
                    specdat = {}, mixingstate='part-res', density_type='hist')
            
            mam4_dict = create_mam4_output_dict(
                    scenario_dir,tt,
                    dia_mids=np.logspace(-10,-5,1000),
                    num_multiplier = 1./1.2930283496840569, 
                    gsds = [1.8,1.6,1.8,1.6],
                    modes=[1,2,3,4],
                    spec_names=['so4','pom','soa','bc','dst','ncl'],
                    return_direct=False,
                    mam_input='../mam_refactor-main/standalone/tests/smoke_test.nl',
                    wvl=550e-9)
            
            output_dict['partmc'] = partmc_dict
            output_dict['mam4'] = mam4_dict
            tt_dir = ensemble_dir + 'processed_output/tt_' + str(tt).zfill(4) + '/'
            output_file = tt_dir + 'scenario' + scenario + '_output.pkl'
            pickle.dump(output_dict,open(output_file,'wb'),protocol=0)
        
    
def create_mam4_output_dict(
        scenario_dir,tt,
        dia_mids=np.logspace(-10,-5,1000),
        spec_names=['so4','pom','soa','bc','dst','ncl'],
        specnames_core = ['bc'],
        modes=[1,2,3,4],
        num_multiplier = 1./1.2930283496840569,
        gsds = [1.8,1.6,1.8,1.6],
        wvl = 550e-9,
        RH_vals = 1. - np.logspace(-2,0,11),
        mam_input='../mam_refactor-main/standalone/tests/smoke_test.nl'):
    lnDs = np.log(dia_mids)
    mam4_dict = {}
    
    temp = get_mam_initial(mam_input,'temp')
    
    Ns,mus,sigs = get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
    mam4_dict['Ns'] = Ns
    mam4_dict['mus'] = mus
    mam4_dict['sigs'] = sigs
    
    mam4_dict['dia_mids'] = dia_mids
    mam4_dict['RH_vals'] = RH_vals
    
    dNdlnD_modes = get_mode_dsds(
            lnDs,scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
    mam4_dict['dNdlnDdry'] = dNdlnD_modes
    
    volcomps = get_mode_volcomps(
            scenario_dir,tt,
            spec_names=spec_names,modes=modes,
            num_multiplier=num_multiplier,gsds=gsds,
            return_direct=False)
    
    
    mam4_dict['volcomps'] = volcomps

    mode_tkappas = get_effective_kappas(volcomps,spec_names)
    mam4_dict['mode_tkappas'] = mode_tkappas
    
    # mode_RIs = get_effective_RIs(volcomps,spec_names)
    # mam4_dict['mode_RIs'] = mode_RIs
    
    s_crit = np.zeros([len(modes),len(dia_mids)])
    # D_crit = np.zeros([len(modes),len(dia_mids)])
    
    # scat_crossects = np.zeros([len(modes),len(dia_mids)])
    # abs_crossects = np.zeros([len(modes),len(dia_mids)])
    
    # idx_core, = np.where([spec_name in specnames_core for spec_name in spec_names])
    # idx_shell, = np.where([spec_name not in specnames_core for spec_name in spec_names])
    # specnames_shell = [spec_names[ii] for ii in idx_shell]
    
    # abs_crossect_core = np.zeros([len(modes),len(dia_mids)])
    # scat_crossect_core = np.zeros([len(modes),len(dia_mids)])
    # abs_crossect_homog = np.zeros([len(modes),len(dia_mids),len(RH_vals)])
    # scat_crossect_homog = np.zeros([len(modes),len(dia_mids),len(RH_vals)])
    # abs_crossect_cs = np.zeros([len(modes),len(dia_mids),len(RH_vals)])
    # scat_crossect_cs = np.zeros([len(modes),len(dia_mids),len(RH_vals)])
    
    # RIs_homog = get_effective_RIs(volcomps,spec_names)
    # RIs_core = get_effective_RIs(volcomps[:,idx_core],specnames_core)
    # RIs_shell = get_effective_RIs(volcomps[:,idx_shell],specnames_shell)
    
    # Dwets = np.zeros([len(modes),len(dia_mids),len(RH_vals)])
    
    # RIs_core = get_effective_RIs(volcomps[:,idx_core],specnames_core)

    # RIs_homog = np.zeros([len(modes),len(dia_mids),len(RH_vals)])
    # RIs_shell = np.zeros([len(modes),len(dia_mids),len(RH_vals)])
    # volcomps_wet = volcomps.copy() 
    
                    
    # volcomps_wet = np.zeros([1,len(spec_names)+1])
    # for rr,RH in enumerate(RH_vals):
    #     for ii,mode_num in enumerate(modes):
    #         tkappas = np.ones_like(dia_mids)*mode_tkappas[ii]
    #         volfrac_dry = volcomps[ii,:-1]/np.sum(volcomps[ii,:-1])
    #         for jj,(Ddry,tkappa) in enumerate(zip(dia_mids,tkappas)):
    #             Dwets[ii,jj,rr] = read_partmc.get_Dwet(Ddry, tkappa, RH, temp)            
    #             volh2o = np.pi/6.*(Dwets[ii,jj,rr]**3. - Ddry**3.)
    #             volcomps_wet[0,:] = volfrac_dry**np.pi/6.*Ddry**3.
    #             volcomps_wet[0,-1] = volh2o
                
    #             RIs_homog[ii,jj,rr] = get_effective_RIs(volcomps_wet,spec_names)
    #             RIs_shell[ii,jj,rr] = get_effective_RIs(volcomps_wet[:,np.hstack([idx_shell,6])],specnames_shell)
                
    for ii,mode_num in enumerate(modes):
        tkappas = np.ones_like(dia_mids)*mode_tkappas[ii]
        volfrac_dry = volcomps[ii,:-1]/np.sum(volcomps[ii,:-1])
        # m_core = RIs_core[ii]
        
        s_crit[ii,:] = read_partmc.process_compute_Sc(temp, dia_mids, tkappas, return_crit_diam=False)
        # for jj,(Ddry,tkappa) in enumerate(zip(dia_mids,tkappas)):
        #     Dcore = compute_coredia_oneparticle(volfrac_dry,Ddry,idx_core)
        #     if Dcore>0.:
        #         abs_crossect_core[ii,jj], scat_crossect_core[ii,jj] = read_partmc.compute_crossect_oneparticle(
        #             Dcore, Dcore, m_core, m_core, wvl)
        #         for rr,RH in enumerate(RH_vals):
        #             m_homog = RIs_homog[ii,jj,rr]
        #             m_shell = RIs_shell[ii,jj,rr]
                    
        #             Dwet = Dwets[ii,jj,rr]
        #             abs_crossect_cs[ii,jj,rr], scat_crossect_cs[ii,jj,rr] = read_partmc.compute_crossect_oneparticle(
        #                 Dcore, Dwet, m_core, m_shell, wvl)
        #             abs_crossect_homog[ii,jj,rr], scat_crossect_homog[ii,jj,rr] = read_partmc.compute_crossect_oneparticle(
        #                 0., Dwet, 0., m_homog, wvl)
        #     else:
        #         abs_crossect_core[ii,jj] = 0.
        #         scat_crossect_core[ii,jj] = 0.
        #         for rr,RH in enumerate(RH_vals):
        #             m_homog = RIs_homog[ii,jj,rr]
        #             m_shell = RIs_shell[ii,jj,rr]
                    
        #             Dwet = Dwets[ii,jj,rr]
        #             abs_crossect_homog[ii,jj,rr], scat_crossect_homog[ii,jj,rr] = read_partmc.compute_crossect_oneparticle(
        #                 0., Dwet, 0., m_homog, wvl)
        #             abs_crossect_homog[ii,jj,rr] = abs_crossect_cs[ii,jj,rr]
        #             scat_crossect_homog[ii,jj,rr] = scat_crossect_cs[ii,jj,rr]
            # print(ii,jj,Ddry,Dwets[ii,jj,0],Dcore)
            # plt.plot(RH_vals,Dwets[ii,jj,:])
    
    mam4_dict['s_crit'] = s_crit
    # mam4_dict['Dwet'] = Dwets
    
    # mam4_dict['RIs_homog'] = RIs_homog
    # mam4_dict['RIs_shell'] = RIs_shell
    # mam4_dict['RIs_core'] = RIs_core
    
    # mam4_dict['abs_crossect_core'] = abs_crossect_core
    # mam4_dict['scat_crossect_core'] = scat_crossect_core
    # mam4_dict['abs_crossect_homog'] = abs_crossect_homog
    # mam4_dict['scat_crossect_homog'] = scat_crossect_homog
    # mam4_dict['abs_crossect_cs'] = abs_crossect_cs
    # mam4_dict['scat_crossect_cs'] = scat_crossect_cs
    return mam4_dict

def create_partmc_output_dict(
        scenario_dir,tt,
        n_repeat=10, wvl=550e-9,
        specdat = {}, mixingstate='part-res', density_type='hist',
        RH_vals = 1. - np.logspace(-2,0,11),
        specnames_core = ['BC'],
        spec_names =  ['SO4', 'NO3', 'Cl', 'NH4', 'MSA', 'ARO1', 'ARO2', 
                       'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                       'CO3', 'Na', 'Ca', 'OIN', 'OC', 'BC', 'H2O']):
    
    partmc_dict = {}
    
    for repeat in range(1,n_repeat+1):
        partmc_dir = scenario_dir + 'repeat' + str(repeat).zfill(4) +  '/out/'
        # repeats_sorted,timesteps_sorted = read_partmc.unravel_ncfiles(partmc_dir)
        partmc_dict[repeat] = {}
        ncfile = read_partmc.get_ncfile(partmc_dir, tt + 1, ensemble_number=repeat)
        
        partmc_dict[repeat]['time'] = read_partmc.get_partmc_variable(ncfile,'time',specdat=specdat,mixingstate=mixingstate,rh='partmc',temperature='partmc')
        partmc_dict[repeat]['RH_vals'] = RH_vals
        
        Ns = read_partmc.get_partmc_variable(ncfile,'aero_num_conc',specdat=specdat,mixingstate=mixingstate,rh='partmc',temperature='partmc')
        partmc_dict[repeat]['Ns'] = Ns
        
        volcomps = read_partmc.get_partmc_variable(ncfile,'aero_particle_vol',specdat=specdat,mixingstate=mixingstate,rh='partmc',temperature='partmc').transpose()
        partmc_dict[repeat]['vol_comps'] = volcomps
        
        RH = read_partmc.get_partmc_variable(ncfile,'relative_humidity',specdat=specdat,mixingstate=mixingstate,rh='partmc',temperature='partmc')
        temp = read_partmc.get_partmc_variable(ncfile,'temperature',specdat=specdat,mixingstate=mixingstate,rh='partmc',temperature='partmc')
        
        partmc_dict[repeat]['temp'] = temp
        partmc_dict[repeat]['RH'] = RH
        
        Ddrys = read_partmc.get_partmc_variable(ncfile,'dry_diameter',specdat=specdat,mixingstate=mixingstate,rh='partmc',temperature='partmc')
        partmc_dict[repeat]['Ddrys'] = Ddrys
        
        tkappas = read_partmc.get_partmc_variable(ncfile,'tkappa',specdat=specdat,mixingstate=mixingstate,rh='partmc',temperature='partmc')
        partmc_dict[repeat]['tkappas'] = tkappas
        
        # idx_core, = np.where([spec_name in specnames_core for spec_name in spec_names])
        # idx_shell, = np.where([spec_name not in specnames_core for spec_name in spec_names])
        # specnames_shell = [spec_names[ii] for ii in idx_shell]
        
        # abs_crossect_core = np.zeros([len(Ddrys)])
        # scat_crossect_core = np.zeros([len(Ddrys)])
        # abs_crossect_homog = np.zeros([len(Ddrys),len(RH_vals)])
        # scat_crossect_homog = np.zeros([len(Ddrys),len(RH_vals)])
        # abs_crossect_cs = np.zeros([len(Ddrys),len(RH_vals)])
        # scat_crossect_cs = np.zeros([len(Ddrys),len(RH_vals)])
        
        # RIs_core = get_effective_RIs(volcomps[:,idx_core],specnames_core)

        # RIs_homog = np.zeros([len(Ns),len(RH_vals)])
        # RIs_shell = np.zeros([len(Ns),len(RH_vals)])
        # volcomps_wet = volcomps.copy() 
        
        # Dwets = np.zeros([len(Ns),len(RH_vals)])
        # for rr,RH in enumerate(RH_vals):
        #     for ii,(Ddry,tkappa) in enumerate(zip(Ddrys,tkappas)):
        #         Dwets[ii,rr] = read_partmc.get_Dwet(Ddry, tkappa, RH, temp)
                
        #         volh2o = np.pi/6.*(Dwets[:,rr]**3. - Ddrys**3.)
        #         volcomps_wet[:,-1] = volh2o
                
        #         RIs_homog = get_effective_RIs(volcomps,spec_names)
        #         RIs_shell = get_effective_RIs(volcomps[:,idx_shell],specnames_shell)
                
        
        # s_crit,D_crit = read_partmc.process_compute_Sc(
        #     partmc_dict[repeat]['temp'], partmc_dict[repeat]['Ddrys'], partmc_dict[repeat]['tkappas'], return_crit_diam=True)
        
        s_crit = read_partmc.process_compute_Sc(
            temp, Ddrys, tkappas, return_crit_diam=False)
        
        partmc_dict[repeat]['s_crit'] = s_crit
        
        # for jj,(Ddry,tkappa,m_homog,m_core,m_shell) in enumerate(zip(Ddrys,tkappas,RIs_homog,RIs_core,RIs_shell)):
        #     # if np.sum(volcomps[jj,:])>0.:
        #     volfrac_dry = volcomps[jj,:-1]/np.sum(volcomps[jj,:-1])
        #     Dcore = compute_coredia_oneparticle(volfrac_dry,Ddry,idx_core)
        #     if Dcore>0.:
        #         abs_crossect_core[jj], scat_crossect_core[jj] = read_partmc.compute_crossect_oneparticle(
        #             Dcore, Dcore, m_core, m_core, wvl)
        #         for rr,RH in enumerate(RH_vals):
        #             Dwet = Dwets[ii,rr]
        #             abs_crossect_cs[jj,rr], scat_crossect_cs[jj,rr] = read_partmc.compute_crossect_oneparticle(
        #                 Dcore, Dwet, m_core, m_shell, wvl)
        #             abs_crossect_homog[jj,rr], scat_crossect_homog[jj,rr] = read_partmc.compute_crossect_oneparticle(
        #                 0., Dwet, 0., m_homog, wvl)
        #     else:
        #         abs_crossect_core[jj] = 0.
        #         scat_crossect_core[jj] = 0.
        #         for rr,RH in enumerate(RH_vals):
        #             Dwet = Dwets[ii,rr]
        #             abs_crossect_homog[jj,rr], scat_crossect_homog[jj,rr] = read_partmc.compute_crossect_oneparticle(
        #                 0., Dwet, 0., m_homog, wvl)
        #             abs_crossect_cs[jj,rr] = abs_crossect_homog[jj,rr]
        #             scat_crossect_cs[jj,rr] = scat_crossect_homog[jj,rr]
        
        # partmc_dict[repeat]['RIs_homog'] = RIs_homog
        # partmc_dict[repeat]['RIs_shell'] = RIs_shell
        # partmc_dict[repeat]['RIs_core'] = RIs_core
        
        # partmc_dict[repeat]['abs_crossect_core'] = abs_crossect_core
        # partmc_dict[repeat]['scat_crossect_core'] = scat_crossect_core
        # partmc_dict[repeat]['abs_crossect_homog'] = abs_crossect_homog
        # partmc_dict[repeat]['scat_crossect_homog'] = scat_crossect_homog
        # partmc_dict[repeat]['abs_crossect_cs'] = abs_crossect_cs
        # partmc_dict[repeat]['scat_crossect_cs'] = scat_crossect_cs
        
    return partmc_dict

def get_scenarios(ensemble_dir):
    filelist = os.listdir(ensemble_dir)
    scenarios = [file for file in filelist if not file.startswith('.') and not file.startswith('process') and not file.endswith('.pkl')]
    num_digits = len(scenarios[0])
    scenario_nums = np.sort([int(scenario) for scenario in scenarios])
    scenarios = [str(num).zfill(num_digits) for num in scenario_nums]
    
    return scenarios


def get_mode_dsd_params(scenario_dir,tt,num_multiplier=1./1.2930283496840569,gsds=[1.8,1.6,1.8,1.6]):
    
    if tt == 0:
        mam_input = scenario_dir + 'mam_input.nl'
        Ns = np.zeros([len(gsds)])
        for kk in range(len(Ns)):#,mode in enumerate(modes):
            Ns[kk] = get_mam_input(
                'numc' + str(kk+1),
                mam_input=mam_input)*num_multiplier
        mus=np.loadtxt(scenario_dir+'mode_mus.txt')
    else:
        mam_output = scenario_dir + 'mam_output.nc'
        f_output = netCDF4.Dataset(mam_output)
        Ns = f_output['num_aer'][:,tt-1]
        mus = np.log(f_output['dgn_a'][:,tt-1])
    sigs=np.log(gsds)
    return Ns,mus,sigs


def get_mode_dsds(
        lnDs,scenario_dir,tt,
        num_multiplier = 1./1.2930283496840569,gsds=[1.8,1.6,1.8,1.6]):
    dNdlnD_modes = np.zeros([len(gsds),len(lnDs)])
    Ns,mus,sigs = get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
    for kk,(N,mu,sig) in enumerate(zip(Ns,mus,sigs)):
        dNdlnD_modes[kk,:] = N*norm(loc=mu,scale=sig).pdf(lnDs)
    return dNdlnD_modes

def get_Mk(k,moment_type,Ns,mus,sigs):
    Mk = 0.
    for ii,(N,mu,sig) in enumerate(zip(Ns,mus,sigs)):
        mk = get_mk_normalized(k,moment_type,mu,sig)
        Mk += N*mk
    return Mk
    
def get_mk_normalized(k,moment_type,mu,sig):
    if moment_type == 'power':
        mk = np.exp(k*mu + (k**2.*sig**2.)/2.)
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
    
def get_mam4_ccn2(s_env_vals,scenario_dir,tt,num_multiplier=1./1.2930283496840569,gsds=[1.8,1.6,1.8,1.6],return_modes=False,fixed_kappa=False):
    mam_input = scenario_dir + 'mam_input.nl'
    temp = get_mam_initial(mam_input,'temp')
    Ns,mus,sigs = get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
    if type(fixed_kappa) == bool and fixed_kappa == False:
        mode_tkappas = get_mode_kappas(scenario_dir,tt)
    else:
        mode_tkappas = fixed_kappa*np.ones(4)
    
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
    
def get_mam4_ccn(
        s_env_vals,scenario_dir,tt,
        dia_mids=np.logspace(-10,-3,1000),num_multiplier=1./1.2930283496840569,
        gsds=[1.8,1.6,1.8,1.6],modes=range(1,5)):
    mam_input = scenario_dir + 'mam_input.nl'
    temp = get_mam_initial(mam_input,'temp')
    Ns,mus,sigs = get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
    dNdlnD_modes = get_mode_dsds(np.log(dia_mids),scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
    dlnD = dia_mids[1] - dia_mids[0]
    N_ccn_modes = np.zeros([len(Ns),len(s_env_vals)])
    spec_names = get_mode_specnames(1)
    volcomps = get_mode_volcomps(
            scenario_dir,tt,
            spec_names=spec_names,modes=modes,
            num_multiplier=num_multiplier,gsds=gsds,
            return_direct=False)
    mode_kappas = get_effective_kappas(volcomps,spec_names)
    Ntot = 0.
    for ii,(N,mu,sig,mode_num) in enumerate(zip(Ns,mus,sigs,modes)):
        tkappas = mode_kappas[ii]*np.ones_like(dia_mids)
        s_crit_onemode = read_partmc.process_compute_Sc(temp, dia_mids, tkappas, return_crit_diam=False)
        Ntot += np.sum(dNdlnD_modes[ii,:]*dlnD)
        for ss,s_thresh in enumerate(s_env_vals):
            N_ccn_modes[ii,ss] = np.sum((s_crit_onemode<=s_thresh)*dNdlnD_modes[ii,:]*dlnD)
    frac_ccn_mam4 = np.sum(N_ccn_modes,axis=0)/Ntot
    return frac_ccn_mam4
        
def compute_coredia_oneparticle(volfrac,dia,idx_core):
    vol_tot = np.pi/6.*dia**3.
    volcomp = vol_tot*volfrac
    vol_core = np.sum(volcomp[idx_core])
    core_dia = (vol_core*6./np.pi)**(1./3.)
    return core_dia
    
def compute_crossect_oneparticle(core_dia, wet_dia, m_core, m_shell, wvl):
    if (core_dia>0) and (wet_dia>core_dia):
        output = PyMieScatt.MieQCoreShell(m_core, m_shell, wvl, core_dia, wet_dia)#, asCrossSection=True)
        scat_crossect = output[1]*np.pi*wet_dia**2/4.
        abs_crossect = output[2]*np.pi*wet_dia**2/4.
    elif core_dia == 0:
        output = PyMieScatt.MieQ(m_shell, wvl, wet_dia)#, asCrossSection = True)
        scat_crossect = output[1]
        if np.imag(m_shell) == 0:
            abs_crossect = 0.
        else:
            abs_crossect = output[2]*np.pi*wet_dia**2/4.
    else:
        output = PyMieScatt.MieQ(m_core, wvl, core_dia)#, asCrossSection = True)
        scat_crossect = output[1]*np.pi*core_dia**2/4.
        abs_crossect = output[2]*np.pi*core_dia**2/4.
        
    return abs_crossect, scat_crossect

def get_partmc_ccn(s_env_vals,scenario_dir,tt,unique_repeats=[1],return_repeats=False):
    timestep = tt + 1
    N_repeats = max(unique_repeats)
    N_ccn_repeat = np.zeros([N_repeats,len(s_env_vals)])
    N_tot_repeat = np.zeros(N_repeats)
    frac_ccn_repeat = np.zeros([N_repeats,len(s_env_vals)])
    
    for ii,repeat in enumerate(unique_repeats):
        partmc_dir = scenario_dir + 'repeat' + str(repeat).zfill(4) + '/out/'
        ncfile = read_partmc.get_ncfile(partmc_dir, timestep, ensemble_number=repeat)
        # ncfile = read_partmc.get_ncfile(partmc_dir, timestep, ensemble_number=repeat)
        s_crit = read_partmc.get_partmc_variable(ncfile,'s_critical')
        num_conc = read_partmc.get_partmc_variable(ncfile,'aero_num_conc')
        for ss,s_env in enumerate(s_env_vals):
            N_ccn_repeat[ii,ss] = sum(num_conc*(s_crit<=s_env))
        N_tot_repeat[ii] = sum(num_conc)
        frac_ccn_repeat[ii,:] = N_ccn_repeat[ii,:]/N_tot_repeat[ii]
    frac_ccn_mean = np.sum(N_ccn_repeat,axis=0)/sum(N_tot_repeat)
    frac_ccn_std = np.std(frac_ccn_repeat,axis=0)
    if return_repeats:
        return frac_ccn_mean, frac_ccn_std, frac_ccn_repeat, N_tot_repeat
    else:
        return frac_ccn_mean, frac_ccn_std

def get_mixing_timescale__all(partmc_dir,timesteps,ensemble_number=1):
    sa_flux_vals = np.zeros(len(timesteps)-1)
    num_conc_vals = np.zeros(len(timesteps)-1)
    if any((timesteps[1:]-timesteps[:-1])>1):
        print('error! timesteps must be consecutive')
    else:
        tt1 = timesteps[0]
        ncfile1 = read_partmc.get_ncfile(partmc_dir, tt1 + 1, ensemble_number=ensemble_number)
        id_t1 = read_partmc.get_partmc_variable(ncfile1,'aero_id')
        time1 = read_partmc.get_partmc_variable(ncfile1,'time') 
        vol_t1 = read_partmc.get_partmc_variable(ncfile1,'dry_vol')
        
        for tt,(tt1,tt2) in enumerate(zip(timesteps[:-1],timesteps[1:])):
            ncfile2 = read_partmc.get_ncfile(partmc_dir, tt+1, ensemble_number=ensemble_number)
            id_t2 = read_partmc.get_partmc_variable(ncfile2,'aero_id') 
            time2 = read_partmc.get_partmc_variable(ncfile2,'time') 
            vol_t2 = read_partmc.get_partmc_variable(ncfile2,'dry_vol')
            
            awet = read_partmc.get_partmc_variable(ncfile1,'wet_area')

            idx_t1, = np.where([id_val in id_t2 for id_val in id_t1])
            idx_t2, = np.where([any(id_t1==id_val) for id_val in id_t2])
            
            sa_flux = sum(vol_t2[idx_t2] - vol_t1[idx_t1])/(sum(awet[idx_t1])*(time2-time1))
            num_conc = sum(read_partmc.get_partmc_variable(ncfile1,'aero_num_conc'))
            
            sa_flux_vals[tt] = sa_flux
            num_conc_vals[tt] = num_conc
            
            # vals for tt2 become vals tt1
            ncfile1 = ncfile2
            id_t1 = id_t2
            time1 = time2
            vol_t1 = vol_t2
    return sa_flux_vals, num_conc_vals
            
def get_mixing_timescale(partmc_dir,tt,ensemble_number=1):
        # for tt in range(1,n_times):#(ncfile1,ncfile2) in enumerate(zip(ncfiles[:-1],ncfiles[1:])):
    ncfile1 = read_partmc.get_ncfile(partmc_dir, tt, ensemble_number=ensemble_number)
    ncfile2 = read_partmc.get_ncfile(partmc_dir, tt+1, ensemble_number=ensemble_number)
    id_t1 = read_partmc.get_partmc_variable(ncfile1,'aero_id')
    id_t2 = read_partmc.get_partmc_variable(ncfile2,'aero_id') 
    time1 = read_partmc.get_partmc_variable(ncfile1,'time') 
    time2 = read_partmc.get_partmc_variable(ncfile2,'time') 
    vol_t1 = read_partmc.get_partmc_variable(ncfile1,'dry_vol')
    vol_t2 = read_partmc.get_partmc_variable(ncfile2,'dry_vol')
    awet = read_partmc.get_partmc_variable(ncfile1,'wet_area')
    
    idx_t1, = np.where([id_val in id_t2 for id_val in id_t1])
    idx_t2, = np.where([any(id_t1==id_val) for id_val in id_t2])
    
    sa_flux = sum(vol_t2[idx_t2] - vol_t1[idx_t1])/(sum(awet[idx_t1])*(time2-time1))
    num_conc = sum(read_partmc.get_partmc_variable(ncfile1,'aero_num_conc'))
    print('tt:',tt,'repeat:',ensemble_number)
    return sa_flux, num_conc

def get_mixing_timescale__noemit(partmc_dir,tt,ensemble_number=1):
    if tt>0:
        ncfile1 = read_partmc.get_ncfile(partmc_dir, tt, ensemble_number=ensemble_number)
        ncfile2 = read_partmc.get_ncfile(partmc_dir, tt+1, ensemble_number=ensemble_number)
        time1 = read_partmc.get_partmc_variable(ncfile1,'time') 
        time2 = read_partmc.get_partmc_variable(ncfile2,'time') 
        vol_t1 = read_partmc.get_partmc_variable(ncfile1,'dry_vol')
        vol_t2 = read_partmc.get_partmc_variable(ncfile2,'dry_vol')
        num_t1 = read_partmc.get_partmc_variable(ncfile1,'aero_num_conc')
        num_t2 = read_partmc.get_partmc_variable(ncfile2,'aero_num_conc')
        awet = read_partmc.get_partmc_variable(ncfile1,'wet_area')
        
        dV_dt = (sum(vol_t2*num_t2) - sum(vol_t1*num_t1))/(time2-time1)
        A0 = sum(awet*num_t1)
        sa_flux = dV_dt/A0
        num_conc = sum(num_t2)
    else:
        ncfile2 = read_partmc.get_ncfile(partmc_dir, tt+1, ensemble_number=ensemble_number)
        num_t2 = read_partmc.get_partmc_variable(ncfile2,'aero_num_conc')
        num_conc = sum(num_t2)
        sa_flux = 0.
        
    return sa_flux,num_conc

def get_mixing_timescale__all_tts__noemit(partmc_dir,tts,ensemble_number=1):
    sa_flux = np.zeros_like(tts)
    num_conc = np.zeros_like(tts)
    
    ncfile1 = read_partmc.get_ncfile(partmc_dir, tts[0]+1, ensemble_number=ensemble_number)
    time1 = read_partmc.get_partmc_variable(ncfile1,'time') 
    vol_t1 = read_partmc.get_partmc_variable(ncfile1,'dry_vol')
    num_t1 = read_partmc.get_partmc_variable(ncfile1,'aero_num_conc')
    
    for tt in tts:
        awet_t1 = read_partmc.get_partmc_variable(ncfile1,'wet_area')
        if tt>0:
            ncfile2 = read_partmc.get_ncfile(partmc_dir, tt+1, ensemble_number=ensemble_number)
            time2 = read_partmc.get_partmc_variable(ncfile2,'time') 
            vol_t2 = read_partmc.get_partmc_variable(ncfile2,'dry_vol')
            num_t2 = read_partmc.get_partmc_variable(ncfile2,'aero_num_conc')
            dV_dt = (sum(vol_t2*num_t2) - sum(vol_t1*num_t1))/(time2-time1)
            A0 = sum(awet_t1*num_t1)
            sa_flux[tt] = dV_dt/A0
            num_conc[tt] = sum(num_t2)
            
            ncfile1 = ncfile2
            time1 = time2
            vol_t1 = vol_t2
            num_t1 = num_t2
        else:
            sa_flux[tt] = 0.
            num_conc[tt] = sum(num_t1)
    
    return sa_flux,num_conc
def get_mixing_timescale__all_tts__mam4(N0s,Ns,mu0s,mus,dtime,mode_sigs=np.log([1.6,1.8,1.8,1.6])):
    if len(np.shape(mu0s)) == 2:
        sum_axis =1
    elif len(np.shape(mu0s)) == 1:
        sum_axis=0
    A0 = np.sum(4*np.pi*N0s*np.exp(2.*mu0s + 2.*mode_sigs**2),axis=sum_axis)
    V = np.sum(4./3. * np.pi * Ns * np.exp(3*mus * 4.5*mode_sigs**2),axis=sum_axis)
    V0 = np.sum(4./3. * np.pi * Ns * np.exp(3*mu0s * 4.5*mode_sigs**2),axis=sum_axis)
    sa_flux = ((V-V0)/dtime)/A0
    num_conc = np.sum(N0s)
    return sa_flux,num_conc

def get_mixing_timescale__mam4(N0s,Ns,mu0s,mus,dtime,mode_sigs=np.log([1.6,1.8,1.8,1.6])):
    if len(np.shape(mu0s)) == 2:
        sum_axis =1
    elif len(np.shape(mu0s)) == 1:
        sum_axis=0
    A0 = np.sum(4*np.pi*N0s*np.exp(2.*mu0s + 2.*mode_sigs**2),axis=sum_axis)
    V = np.sum(4./3. * np.pi * Ns * np.exp(3*mus * 4.5*mode_sigs**2),axis=sum_axis)
    V0 = np.sum(4./3. * np.pi * Ns * np.exp(3*mu0s * 4.5*mode_sigs**2),axis=sum_axis)
    sa_flux = ((V-V0)/dtime)/A0
    num_conc = np.sum(N0s)
    return sa_flux,num_conc

def get_effective_kappas(vol_comps,spec_names):
    tkappas = np.zeros_like(vol_comps[:,0])
    for ii in range(vol_comps.shape[0]):
        vol_tot = 0.
        volKap_tot = 0.
        for kk,spec_name in enumerate(spec_names):
            spec_kappa = get_spec_kappa(spec_name)
            vol_tot += vol_comps[ii,kk]
            volKap_tot += vol_comps[ii,kk]*spec_kappa
        tkappas[ii] = volKap_tot/vol_tot
    return tkappas

def get_effective_densities(vol_comps,spec_names):
    densities = np.zeros_like(vol_comps[:,0])
    for ii in range(vol_comps.shape[0]):
        vol_tot = 0.
        mass_tot = 0.
        for kk,spec_name in enumerate(spec_names):
            spec_density = get_spec_density(spec_name)
            vol_tot += vol_comps[ii,kk]
            mass_tot += vol_comps[ii,kk]*spec_density
        densities[ii] = mass_tot/vol_tot
    return densities
    
def get_effective_RIs(vol_comps,spec_names):
    mode_RIs_real = np.zeros(vol_comps.shape[0])
    mode_RIs_imag = np.zeros(vol_comps.shape[0])
    for ii in range(vol_comps.shape[0]):
        vol_tot = 0.
        volRI_tot = 0.
        for kk,spec_name in enumerate(spec_names):
            spec_RI = get_spec_RI(spec_name)
            vol_tot += vol_comps[ii,kk]
            volRI_tot += vol_comps[ii,kk]*spec_RI
        mode_RIs_real[ii] = np.real(volRI_tot)/vol_tot
        mode_RIs_imag[ii] = np.imag(volRI_tot)/vol_tot
    mode_RIs = mode_RIs_real + mode_RIs_imag*1j
    
    return mode_RIs

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

# def get_partmc_kappas(volcomps,tt,spec_names = []):
#     tkappas = np.zeros(volcomps.shape[0])
    
#     get_spec_partmc_kappa(spec_name,fromMAM=True)

def get_mode_densities(scenario_dir,tt):
    mam_output = scenario_dir + 'mam_output.nc'
    mam_input = scenario_dir + 'mam_input.nl'
    
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
                mass = get_mam_initial(mam_input,'mf' + spec  + str(mode_num))
            else:
                mass = f_output[spec + '_aer'][ii,tt-1]
                
            vol_tot += mass/spec_density
            mass_tot += mass
        mode_densities[ii] = mass_tot/vol_tot
    return mode_densities

def get_mode_mfs(
        scenario_dir,tt,
        spec_names=['so4','pom','soa','bc','dst','ncl'],
        num_multiplier=1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8],
        modes=[1,2,3,4]):
    mode_mfs = np.zeros([len(modes),len(spec_names)])
    for ii,mode_num in enumerate(modes):
        mode_spec_names = get_mode_specnames(mode_num)
        for kk,spec_name in enumerate(spec_names):
            if spec_name in mode_spec_names:
                if tt == 0:
                    mam_input = scenario_dir + 'mam_input.nl'
                    mode_mf = get_mam_initial(mam_input,'mf' + spec_name  + str(mode_num))
                else:
                    mam_output = scenario_dir + 'mam_output.nc'
                    f_output = netCDF4.Dataset(mam_output)
                    mode_mf = f_output[spec_name + '_aer'][ii,tt-1]
                # print(tt,mode_num,spec_name,mode_mf)
            else:
                mode_mf = 0.
            mode_mfs[ii,kk] = mode_mf
        mode_mfs[ii,:] = mode_mfs[ii,:]/np.sum(mode_mfs[ii,:])
    return mode_mfs

def get_mode_vfs(
        scenario_dir,tt,
        spec_names=['so4','pom','soa','bc','dst','ncl'],
        num_multiplier=1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8],
        modes=[1,2,3,4]):
    mode_mfs = get_mode_mfs(
        scenario_dir,tt,
        spec_names=spec_names,
        num_multiplier=num_multiplier,gsds=gsds,
        modes=modes)
    mode_vfs = np.zeros_like(mode_mfs)
    
    for kk,spec_name in enumerate(spec_names):
        spec_density = get_spec_density(spec_name)
        mode_vfs[:,kk] = mode_mfs[:,kk]/spec_density
    
    for ii,mode_num in enumerate(modes):
        mode_vfs[ii,:] = mode_vfs[ii,:]/np.sum(mode_vfs[ii,:])
    
    return mode_vfs

def get_mode_volcomps(
        scenario_dir,tt,
        spec_names=['so4','pom','soa','bc','dst','ncl'],
        num_multiplier=1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8],
        modes=[1,2,3,4],return_direct=False):
    
    mode_vfs = get_mode_vfs(
        scenario_dir,tt,
        spec_names=spec_names,
        num_multiplier=num_multiplier,gsds=gsds,
        modes=modes)
    
    V_modes = np.zeros(len(modes))
    volcomps = np.zeros([len(modes),len(spec_names)])
    volcomps_direct = np.zeros([len(modes),len(spec_names)])
    for ii,mode_num in enumerate(modes):
        Ns,mus,sigs = get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
        V_modes[ii] = Ns[ii]*np.pi/6.*get_mk_normalized(3,'power',mus[ii],sigs[ii])
        volcomps[ii,:] = mode_vfs[ii,:]*V_modes[ii]
        if return_direct:
            if tt>0:
                for kk,spec_name in enumerate(spec_names):
                    mode_spec_names = get_mode_specnames(mode_num)
                    if spec_name in mode_spec_names:
                        spec_density = get_spec_density(spec_name)
                        mam_output = scenario_dir + 'mam_output.nc'
                        f_output = netCDF4.Dataset(mam_output)
                        volcomps_direct[ii,kk] = f_output[spec_name + '_aer'][ii,tt-1]/spec_density
    if return_direct:
        return volcomps,volcomps_direct,V_modes
    else:
        return volcomps
    


def get_mode_volcomps_old(
        scenario_dir,tt,
        spec_names=['so4','pom','soa','bc','dst','ncl'],
        num_multiplier=1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8],
        modes=[1,2,3,4]):
    mode_volcomps = np.zeros([len(modes),len(spec_names)])
    for ii,mode_num in enumerate(modes):
        mode_spec_names = get_mode_specnames(mode_num)
        
        if tt == 0:
            N0s,mu0s,sig0s = get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
            V0 = N0s[ii]*np.pi/6.*get_mk_normalized(3,'power',mu0s[ii],sig0s[ii])
            
            vol_tot = 0.
            for kk,spec_name in enumerate(spec_names):
                if spec_name in mode_spec_names:
                    
                    spec_density = get_spec_density(spec_name)
                    if tt == 0:
                        mam_input = scenario_dir + 'mam_input.nl'
                        vol_tot += get_mam_initial(mam_input,'mf' + spec_name  + str(mode_num))/spec_density
        
        for kk,spec_name in enumerate(spec_names):
            if spec_name in mode_spec_names:
                spec_density = get_spec_density(spec_name)
                
                if tt == 0:
                    mode_mf = get_mam_initial(mam_input,'mf' + spec_name  + str(mode_num))
                    mode_vf = (mode_mf/spec_density)/vol_tot
                    
                    mode_volcomps[ii,kk] = mode_vf*V0
                else:
                    mam_output = scenario_dir + 'mam_output.nc'
                    f_output = netCDF4.Dataset(mam_output)
                    mode_volcomps[ii,kk] = f_output[spec_name + '_aer'][ii,tt-1]/spec_density
                    
    return mode_volcomps
# def get_mode_volcomps(
#         scenario_dir,tt,
#         spec_names=['so4','pom','soa','bc','dst','ncl'],
#         num_multiplier=1./1.2930283496840569,gsds=[1.6,1.8,1.6,1.8]):
#     mam_output = scenario_dir + 'mam_output.nc'
#     f_output = netCDF4.Dataset(mam_output)
    
#     effective_densities = np.zeros(f_output['num_aer'].shape[0])
#     if tt == 0:
#         N0s,mu0s,sig0s = get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
#         # divider_t0 = 0.
#         # lnDs = np.log(np.logspace(-10,-4,1000))
#         # lnD_wid = lnDs[1] - lnDs[0]
#         # dsd0s = get_mode_dsds(
#         #         lnDs,scenario_dir,tt,
#         #         num_multiplier=num_multiplier, gsds=gsds)
#         # for ii in range(f_output['num_aer'].shape[0]):
#         #     top_part = 0.
#         #     bottom_part = 0.
            
#         #     mode_num = ii + 1
#         #     mode_spec_names = get_mode_specnames(mode_num)
            
#         #     for kk,spec_name in enumerate(spec_names):
#         #         if spec_name in mode_spec_names:
#         #             mam_input = scenario_dir + 'mam_input.nl'
#         #             spec_density = get_spec_density(spec_name)
#         #             mode_mf = get_mam_initial(mam_input,'mf' + spec_name  + str(mode_num))
#         #             top_part += np.sum(mode_mf)
#         #             bottom_part += np.sum(mode_mf/spec_density)
#         #     effective_densities[ii] = top_part/bottom_part
            
#         # V0s = np.pi/6.*get_Mk(3,'power',N0s,mu0s,sig0s)\
#     Vtot = 0.
#     mode_volcomps = np.zeros([f_output['num_aer'].shape[0],len(spec_names)])
#     for ii in range(f_output['num_aer'].shape[0]):
#         mode_num = ii + 1
#         tot_mf = 0.
#         tot_vf = 0.
#         mode_spec_names = get_mode_specnames(mode_num)
#         if tt == 0:
#             top_part = 0.
#             bottom_part = 0.
#             for kk,spec_name in enumerate(spec_names):
#                 if spec_name in mode_spec_names:
#                     mam_input = scenario_dir + 'mam_input.nl'
#                     spec_density = get_spec_density(spec_name)
#                     mode_mf = get_mam_initial(mam_input,'mf' + spec_name  + str(mode_num))
#                     top_part += np.sum(mode_mf)
#                     bottom_part += np.sum(mode_mf/spec_density)
#             effective_densities[ii] = top_part/bottom_part
#             print(effective_densities[ii])
#         for kk,spec_name in enumerate(spec_names):
#             if spec_name in mode_spec_names:
#                 spec_density = get_spec_density(spec_name)
#                 if tt == 0:
#                     # V0 = N0s[ii]*4.*np.pi/3.*(np.exp(mu0s[ii])/2.)**3.*np.exp(9*sig0s[ii]**2/2.)
#                     V0 = np.pi/6.*get_mk_normalized(3,'power',mu0s[ii],sig0s[ii])*N0s[ii]
#                     mam_input = scenario_dir + 'mam_input.nl'
#                     # V0_dsd = np.sum(np.pi/6.*np.exp(lnDs)**3*dsd0s[ii,:]*lnD_wid)
#                     mode_mf = get_mam_initial(mam_input,'mf' + spec_name  + str(mode_num))
#                     mode_vf = (mode_mf/spec_density)/bottom_part
#                     mode_mf = mode_mf/top_part
#                     mass0 = effective_densities[ii]*V0
#                     mass_onemode = mass0*mode_mf
#                     vol = mass_onemode/spec_density
#                     # print(V0,V0_params)
#                     # mode_vf = mode_mf/spec_density/divider_t0
#                     # vol = V0*mode_vf
#                     tot_mf += mode_mf
#                     tot_vf += mode_vf
#                 else:
#                     vol = f_output[spec_name + '_aer'][ii,tt-1]/spec_density
#             else:
#                 vol = 0.
#             mode_volcomps[ii,kk] = vol
#         if tt == 0:
#             print(tot_mf,tot_vf)
#         if tt == 0:
#             Vtot += V0
#         else:
#             Vtot = np.nan
#     return mode_volcomps, Vtot
        
def get_mode_RIs(scenario_dir,tt):
    mam_output = scenario_dir + 'mam_output.nc'
    f_output = netCDF4.Dataset(mam_output)
    
    mode_RIs_real = np.zeros(f_output['num_aer'].shape[0])
    mode_RIs_imag = np.zeros(f_output['num_aer'].shape[0])
    for ii in range(f_output['num_aer'].shape[0]):
        mode_num = ii + 1
        spec_names = get_mode_specnames(mode_num)
        
        vol_tot = 0.
        volRI_tot = 0.
        for kk,spec_name in enumerate(spec_names):
            spec_RI = get_spec_RI(spec_name)
            spec_density = get_spec_density(spec_name)
            if tt == 0:
                mam_input = scenario_dir + 'mam_input.nl'
                vol = get_mam_initial(mam_input,'mf' + spec_name  + str(mode_num))/spec_density
            else:
                vol = f_output[spec_name + '_aer'][ii,tt-1]/spec_density
            vol_tot += vol
            volRI_tot += vol*spec_RI
        mode_RIs_real[ii] = np.real(volRI_tot)/vol_tot
        mode_RIs_imag[ii] = np.imag(volRI_tot)/vol_tot
    mode_RIs = mode_RIs_real + mode_RIs_imag*1j
    return mode_RIs




def get_spec_kappa(spec_name,fromMAM=True):
    if fromMAM:
        # https://github.com/eagles-project/mam_refactor/blob/ae07f7572640466492c236a336a7c7ba0713c5e8/core/rad_constituents.F90#L73
        if spec_name in ['so4', 'SO4','NO3','NH4']:
            spec_kappa = 0.507
        elif spec_name in ['pom', 'OC']:
            spec_kappa = 0.01
        elif spec_name in ['soa','ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']:
            spec_kappa = 0.1
        elif spec_name in ['bc','BC']:
            spec_kappa = 1.0e-10
        elif spec_name in ['dst','OIN']:
            spec_kappa = 0.068
        elif spec_name in ['Ca','CO3']:
            spec_kappa = 0.53
        elif spec_name in ['ncl','Na','Cl']:
            spec_kappa = 1.160
        elif spec_name in ['MSA']:
            spec_kappa = 0.53
        elif spec_name in ['H2O']:
            spec_kappa = 0.
    else:
        if spec_name in ['so4', 'SO4','NO3','NH4']:
            spec_kappa = 0.65
        elif spec_name in ['pom', 'OC']:
            spec_kappa = 0.001
        elif spec_name in ['soa','ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']:
            spec_kappa = 0.1
        elif spec_name in ['bc','BC']:
            spec_kappa = 0.
        elif spec_name in ['dst','OIN']:
            spec_kappa = 0.1
        elif spec_name in ['Ca','CO3']:
            spec_kappa = 0.53
        elif spec_name in ['ncl','Na','Cl']:
            spec_kappa = 0.53
        elif spec_name in ['H2O']:
            spec_kappa = 0.
    return spec_kappa

def get_spec_RI(spec_name):
    # all most applicable for 550 nm
    if spec_name in ['so4', 'SO4','NO3','NH4','MSA']:
        spec_RI = 1.5
    elif spec_name in ['pom', 'OC']:
        spec_RI = 1.45
    elif spec_name in ['soa','ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']: # likely varies with wavelength (e.g. BrC)
        spec_RI = 1.45
    elif spec_name in ['bc','BC']: # often assumed to be invariate with wavelength
        spec_RI = 1.82 + 0.74j
    elif spec_name in ['dst','OIN','Ca','CO3']: # varies with dust type and a bit with wavelength
        spec_RI = 1.55 + 3e-3j # https://www.tandfonline.com/doi/abs/10.1111/j.1600-0889.2008.00383.x
    elif spec_name in ['ncl','Na','Cl']: # varies a bit with wavelength
        spec_RI = 1.55 # https://refractiveindex.info/?shelf=main&book=NaCl&page=Li
    elif spec_name in ['h2o','H2O']: # varies a bit with wavelength
        spec_RI = 1.33 
    return spec_RI

def get_spec_density(spec_name):
    if spec_name in ['so4', 'SO4','NO3','NH4']:
        spec_density = 1770.
    elif spec_name in ['pom', 'OC']:
        spec_density = 1000.
    elif spec_name in ['soa','ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']:
        spec_density = 1000.
    elif spec_name in ['bc','BC']:
        spec_density = 1700.
    elif spec_name in ['dst','OIN','Ca','CO3']:
        spec_density = 2600.
    elif spec_name in ['ncl','Na','Cl']:
        spec_density = 1900.
    elif spec_name in ['MSA']:
        spec_density = 1800.
    elif spec_name in ['h2o','H2O']:
        spec_density = 1000.
    return spec_density

    #  (/ 'sulfate         ', &
    # 'ammonium        ', &
    # 'nitrate         ', &
    # 'p-organic       ', &
    # 's-organic       ', &
    # 'black-c         ', &
    # 'seasalt         ', &
    # 'dust            ', &
    # 'm-organic       ' /)
     
    # 1770.0_wp
    # 1770.0_wp
    # 1770.0_wp
    # 1000.0_wp
    # 1000.0_wp
    # 1700.0_wp
    # 1900.0_wp
    # 2600.0_wp
    # 1601.0_wp
    
    # 0.507_wp
    # 0.507_wp
    # 0.507_wp
    # 0.010_wp
    # 0.140_wp
    # 1.0e-10_wp
    # 1.160_wp
    # 0.068_wp
    # 0.100_wp  
    
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

def get_partmc_kappadist_onefile(lnKaps,ncfile,density_type='hist'):
    tkappa = read_partmc.get_partmc_variable(ncfile,'tkappa')
    num = read_partmc.get_partmc_variable(ncfile,'aero_num_conc')
    
    if density_type == 'hist':
        lnKaps_edges = np.linspace(lnKaps[0]-(lnKaps[1]-lnKaps[0])/2.,lnKaps[-1]+(lnKaps[1]-lnKaps[0])/2.,len(lnKaps)+1)
        dndlnKap,ln_kap_bins = np.histogram(np.log(tkappa),bins=lnKaps_edges,weights=num,density=True)
    elif density_type == 'gaussian':
        kde = KernelDensity(kernel='gaussian', bandwidth=1e-5).fit(np.log(tkappa).reshape(-1,1),sample_weight=num/sum(num))
        dndlnKap = kde.score_samples(lnKaps.reshape(-1,1))
    dNdlnKap_partmc = sum(num)*dndlnKap
    
    return dNdlnKap_partmc

def get_partmc_kapdist(lnKaps,scenario_dir,tt,unique_repeats=[1],density_type='hist'):
    timestep = tt + 1
    N_repeats = max(unique_repeats)
    dNdlnKap_repeats = np.zeros([len(lnKaps),N_repeats])
    for ii,repeat in enumerate(unique_repeats):
        partmc_dir = scenario_dir + 'repeat' + str(repeat).zfill(4) + '/out/'
        ncfile = read_partmc.get_ncfile(partmc_dir, timestep, ensemble_number=repeat)
        dNdlnKap_repeats[:,ii] = get_partmc_kappadist_onefile(lnKaps,ncfile,density_type=density_type)
    return dNdlnKap_repeats

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

def get_partmc_dsd(lnDs,scenario_dir,tt,unique_repeats=[1],density_type='hist'):
    timestep = tt + 1
    N_repeats = max(unique_repeats)
    dNdlnD_repeats = np.zeros([len(lnDs),N_repeats])
    for ii,repeat in enumerate(unique_repeats):
        partmc_dir = scenario_dir + 'repeat' + str(repeat).zfill(4) + '/out/'
        ncfile = read_partmc.get_ncfile(partmc_dir, timestep, ensemble_number=repeat)
        dNdlnD_repeats[:,ii] = get_partmc_dsd_onefile(lnDs,ncfile,density_type=density_type)
    return dNdlnD_repeats
    
def get_KL_stochastic(scenario_dir,tt,unique_repeats=[1],num_multiplier=1./1.2930283496840569,gsds=[1.8,1.6,1.8,1.6]):
    partmc_tt = tt + 1
    N_repeats = max(unique_repeats)
    KL_repeats = np.zeros(N_repeats)
    KL_repeats_backward = np.zeros(N_repeats)    
    for jj,repeat in enumerate(unique_repeats):
        partmc_dir = scenario_dir + 'repeat' + str(repeat).zfill(4) + '/out/'
        ncfile = read_partmc.get_ncfile(partmc_dir, partmc_tt, ensemble_number=repeat)
        Ds = read_partmc.get_partmc_variable(ncfile,'dry_diameter')
        Ns = read_partmc.get_partmc_variable(ncfile,'aero_num_conc')
        lnDs = np.log(Ds)
        
        ps_partmc,lnDs_edges = np.histogram(lnDs,weights=Ns/sum(Ns),bins=200)
        lnDs_mids = 0.5*(lnDs_edges[:-1] + lnDs_edges[1:])
        
        dlnD = lnDs_mids[1] - lnDs_mids[0]
        dNdlnD_mam4 = get_mam4_dsd(lnDs_mids,scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)
        ps_mam4 = (dNdlnD_mam4*dlnD)/sum(dNdlnD_mam4*dlnD)
        
        # print(scenario_dir,', repeat',repeat,', sum(ps_mam4)=',sum(ps_mam4),', sum(ps_partmc)=',sum(ps_partmc))
        idx, = np.where(ps_partmc>0)
        KL_repeats[jj] = sum(ps_partmc[idx]*np.log(ps_partmc/ps_mam4)[idx])
        KL_repeats_backward[jj] = sum(ps_mam4[idx]*np.log(ps_mam4/ps_partmc)[idx])
        
        
    return KL_repeats, KL_repeats_backward
    
def get_KL_binned(dNdlnD_partmc,dNdlnD_mam4,backward=False):
    P_partmc = dNdlnD_partmc/sum(dNdlnD_partmc)
    P_mam4 = dNdlnD_mam4/sum(dNdlnD_mam4)
    idx, = np.where(P_partmc>0)
    if backward:
        KL = entropy(P_partmc[idx],P_mam4[idx])
    else:
        KL = entropy(P_mam4[idx],P_partmc[idx])
    return KL



def get_mam_initial(mam_input,varname):
    if varname == 'H2SO4_gas':
        vardat_init = get_mam_input(
            'qh2so4',
            mam_input=mam_input)*1e9
    else:
        vardat_init = get_mam_input(varname,mam_input=mam_input)
    return vardat_init

def store_data(
        ensemble_dir, all_scenarios, tt_save, n_repeat, dtime,
        run_time_str = '00:06:00', project = 'particula', 
        num_multiplier = 1./1.2930283496840569, gsds = [1.8,1.6,1.8,1.6], 
        lnDs = np.log(np.logspace(-10,-5,100)), s_env_vals = np.logspace(-3,2,151), 
        store_ccn=False, store_flux=False):
    
    runfiles_dir = ensemble_dir + '../runfiles/' 
    python_filename = runfiles_dir + 'main_process_tt' + str(tt_save).zfill(4) + '.py'
    sh_filename = runfiles_dir + 'batch_tt' + str(tt_save).zfill(4) + '.sh'
    
    lnDs_str = 'np.array(' + str(list(lnDs)) + ')'
    s_env_vals_str = 'np.array(' + str(list(s_env_vals)) + ')'
    
    python_lines = [
        'import numpy as np',
        'import process_readVariables',
        'import process_benchmarking',
        'import os',
        'import pickle',
        'ensemble_dir = \'' + ensemble_dir + '\'',
        'all_scenarios = ' + str(all_scenarios),
        'tt_save = ' + str(tt_save),
        'n_repeat = ' + str(n_repeat),
        'dtime = ' + str(dtime),
        'unique_repeats = [ii for ii in range(1,n_repeat + 1)]',
        'num_multiplier = ' + str(num_multiplier),
        'gsds = ' + str(gsds),
        'lnDs = ' + lnDs_str,
        's_env_vals = ' + s_env_vals_str,
        'store_ccn = ' + str(store_ccn),
        'store_flux = ' + str(store_flux),
        'Ns = np.zeros([len(all_scenarios),4])',
        'mus = np.zeros([len(all_scenarios),4])',
        'sigs = np.zeros([len(all_scenarios),4])',
        'N0s = np.zeros([len(all_scenarios),4])',
        'mu0s = np.zeros([len(all_scenarios),4])',
        'sig0s = np.zeros([len(all_scenarios),4])',
        'if store_ccn:',
        '    frac_ccn_mam4 = np.zeros([len(all_scenarios),len(s_env_vals)])',
        '    frac_ccn_partmc_mean = np.zeros([len(all_scenarios),len(s_env_vals)])',
        '    frac_ccn_partmc_std = np.zeros([len(all_scenarios),len(s_env_vals)])',
        'dNdlnD_mam4 = np.zeros([len(all_scenarios),len(lnDs)])',
        'dNdlnD_partmc = np.zeros([len(all_scenarios),len(lnDs),n_repeat])',
        'KL_repeats = np.zeros([len(all_scenarios),n_repeat])',
        'KL_repeats_backward = np.zeros([len(all_scenarios),n_repeat])',
        'KL_repeats = np.zeros([len(all_scenarios),n_repeat])',
        'sa_flux = np.zeros([len(all_scenarios),n_repeat])',
        'num_conc = np.zeros([len(all_scenarios),n_repeat])',
        'sa_flux_mam4 = np.zeros([len(all_scenarios)])',
        'num_conc_mam4 = np.zeros([len(all_scenarios)])',
        'not_empty_ss = []',
        'empty_ss = []',
        'for ss,scenario in enumerate(all_scenarios):',
        '    scenario_dir = ensemble_dir + scenario + \'/\'',
        '    dirnames = os.listdir(scenario_dir)',
        '    repeat_dirs = [dirname for dirname in dirnames if dirname.startswith(\'repeat\')]',
        '    partmc_dir = scenario_dir + repeat_dirs[0] + \'/out/\'',
        '    ncfiles = [filename for filename in os.listdir(partmc_dir) if filename.endswith(\'.nc\')]',
        '    if len(ncfiles)>0:',
        '        tt = 0',
        '        N0s[ss,:],mu0s[ss,:],sig0s[ss,:] = process_readVariables.get_mode_dsd_params(scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)',
        '        tt = tt_save',
        '        if store_ccn:',
        '            frac_ccn_mam4[ss,:] = process_readVariables.get_mam4_ccn(s_env_vals,scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)',
        '            frac_ccn_partmc_mean[ss,:], frac_ccn_partmc_std[ss,:] = process_readVariables.get_partmc_ccn(s_env_vals,scenario_dir,tt,unique_repeats=unique_repeats,return_repeats=False)',
        '        Ns[ss,:],mus[ss,:],sigs[ss,:] = process_benchmarking.get_mode_dsd_params(scenario_dir,tt)',
        '        dNdlnD_mam4[ss,:] = process_readVariables.get_mam4_dsd(lnDs,scenario_dir,tt,num_multiplier=num_multiplier,gsds=gsds)',
        '        dNdlnD_partmc[ss,:,:] = process_readVariables.get_partmc_dsd(lnDs,scenario_dir,tt,unique_repeats=unique_repeats,density_type=\'hist\')',
        '        for ii,repeat in enumerate(unique_repeats):',
        '            KL_repeats[ss,ii] = process_readVariables.get_KL_binned(dNdlnD_partmc[ss,:,ii],dNdlnD_mam4[ss,:],backward=False)',
        '            KL_repeats_backward[ss,ii] = process_readVariables.get_KL_binned(dNdlnD_partmc[ss,:,ii],dNdlnD_mam4[ss,:],backward=True)',
        '        if store_flux:',
        '            for ii,repeat in enumerate(unique_repeats):',
        '                repeat_dir = ensemble_dir + scenario  + \'/repeat\' + str(repeat).zfill(4) + \'/\'',
        '                sa_flux[ss,ii], num_conc[ss,ii] = process_readVariables.get_mixing_timescale__noemit(repeat_dir + \'out/\',tt_save,ensemble_number=repeat)',
        '            sa_flux_mam4[ss], num_conc_mam4[ss] = process_readVariables.get_mixing_timescale__mam4(Ns[ss,:],N0s[ss,:],mus[ss,:],mu0s[ss,:],dtime,mode_sigs=sigs[ss,:])',
        '        not_empty_ss.append(ss)',
        '    else:',
        '        empty_ss.append(ss)',
        'save_dir = ensemble_dir + \'../processed_output/\'',
        'if not os.path.exists(save_dir):',
        '    os.mkdir(save_dir)',
        'tt_dir = save_dir + \'tt\' + str(tt_save).zfill(4) + \'/\'',
        'if not os.path.exists(tt_dir):',
        '    os.mkdir(tt_dir)',
        'np.savetxt(tt_dir + \'N0s.txt\', N0s)',
        'np.savetxt(tt_dir + \'mu0s.txt\', mu0s)',
        'np.savetxt(tt_dir + \'sig0s.txt\', sig0s)',
        'np.savetxt(tt_dir + \'dNdlnD_mam4.txt\', dNdlnD_mam4)',
        'for ii in range(1,n_repeat+1):',
        '    np.savetxt(tt_dir + \'dNdlnD_partmc_repeat\' + str(ii).zfill(4) + \'.txt\', dNdlnD_partmc[:,:,ii-1])',
        'np.savetxt(tt_dir + \'Ns.txt\', Ns)',
        'np.savetxt(tt_dir + \'mus.txt\', mus)',
        'np.savetxt(tt_dir + \'sigs.txt\', sigs)',
        'if store_flux:',
        '    np.savetxt(tt_dir + \'sa_flux.txt\', sa_flux)',
        '    np.savetxt(tt_dir + \'num_conc.txt\', num_conc)',
        '    np.savetxt(tt_dir + \'sa_flux_mam4.txt\', sa_flux_mam4)',
        '    np.savetxt(tt_dir + \'num_conc_mam4.txt\', num_conc_mam4)',
        'if store_ccn:',
        '    np.savetxt(tt_dir + \'frac_ccn_mam4.txt\', frac_ccn_mam4)',
        '    np.savetxt(tt_dir + \'frac_ccn_partmc_mean.txt\', frac_ccn_partmc_mean)',
        '    np.savetxt(tt_dir + \'frac_ccn_partmc_std.txt\', frac_ccn_partmc_std)',
        '    np.savetxt(tt_dir + \'s_env_vals.txt\',s_env_vals)',
        'np.savetxt(tt_dir + \'KL_repeats.txt\', KL_repeats)',
        'np.savetxt(tt_dir + \'KL_repeats_backward.txt\', KL_repeats_backward)']

    with open(python_filename, 'w') as f:
        for line in python_lines:
            f.write(line)
            f.write('\n')
    f.close()
    
    if os.path.exists('/people/fier887/'):
        sh_lines = [
            '#!/bin/tcsh',
            '#SBATCH -A ' + project,
            '#SBATCH -p shared',
            '#SBATCH -t ' + run_time_str,
            '#SBATCH -n 1',
            '#SBATCH -o ' + runfiles_dir + 'process_tt' + str(tt_save).zfill(4) + '.out',
            '#SBATCH -e ' + runfiles_dir + 'process_tt' + str(tt_save).zfill(4) + '.err',
            '',
            'python ' + python_filename]
    else:
        print('not set up to run on this computer')
    
    with open(sh_filename, 'w') as f:
        for line in sh_lines:
            f.write(line)
            f.write('\n')
    f.close()
    os.system('chmod +x ' + sh_filename)
    os.system('sbatch ' + sh_filename)
    
def store_data_kappas(
        ensemble_dir, all_scenarios, tt_save, n_repeat, dtime,
        run_time_str = '00:01:00', project = 'particula', 
        num_multiplier = 1./1.2930283496840569, gsds = [1.8,1.6,1.8,1.6], 
        lnDs = np.log(np.logspace(-10,-5,100)), s_env_vals = np.logspace(-3,2,151), 
        store_ccn=False, store_flux=False):
    
    runfiles_dir = ensemble_dir + '../runfiles/' 
    python_filename = runfiles_dir + 'main_process_tt' + str(tt_save).zfill(4) + '_kappa.py'
    sh_filename = runfiles_dir + 'tt' + str(tt_save).zfill(4) + '_kappa.sh'
    
    python_lines = [
        'import numpy as np',
        'import process_readVariables',
        'import process_benchmarking',
        'import os',
        'import read_partmc',
        'import pickle',
        'ensemble_dir = \'' + ensemble_dir + '\'',
        'all_scenarios = ' + str(all_scenarios),
        'tt_save = ' + str(tt_save),
        'n_repeat = ' + str(n_repeat),
        'dtime = ' + str(dtime),
        'unique_repeats = [ii for ii in range(1,n_repeat + 1)]',
        'num_multiplier = ' + str(num_multiplier),
        'gsds = ' + str(gsds),
        'save_dir = ensemble_dir + \'../processed_output/\'',
        'if not os.path.exists(save_dir):',
        '    os.mkdir(save_dir)',
        'tt_dir = save_dir + \'tt\' + str(tt_save).zfill(4) + \'/\'',
        'if not os.path.exists(tt_dir):',
        '    os.mkdir(tt_dir)',
        'mode_tkappas = np.zeros([len(all_scenarios),len(gsds)])',
        'for ss,scenario in enumerate(all_scenarios):',
        '    scenario_dir = ensemble_dir + scenario + \'/\'',
        '    dirnames = os.listdir(scenario_dir)',
        '    mode_tkappas[ss,:] = process_readVariables.get_mode_kappas(scenario_dir,tt_save)',
        'np.savetxt(tt_dir + \'mode_tkappas_mam.txt\', mode_tkappas)',
        'for ii,repeat in enumerate(unique_repeats):',
        '    kap_filename = tt_dir + \'tkappas_partmc_repeat\' + str(repeat).zfill(4) + \'.txt\'',#partmc_tkappas_onerepeat
        # '    partmc_tkappas_onerepeat = []',
        '    for ss,scenario in enumerate(all_scenarios):',
        '        if ss == 0:',
        '            f_kappa = open(kap_filename,\'w\')',
        '        else:'
        '            f_kappa = open(kap_filename,\'a\')',
        '        scenario_dir = ensemble_dir + scenario + \'/\'',
        '        dirnames = os.listdir(scenario_dir)',
        '        repeat_dirs = [dirname for dirname in dirnames if dirname.startswith(\'repeat\')]',
        '        partmc_dir = scenario_dir + repeat_dirs[ii] + \'/out/\'',
        # '        ncfile = read_partmc.get_ncfile(partmc_dir, tt_save + 1, ensemble_number=repeat)',
        '        ncfiles = [filename for filename in os.listdir(partmc_dir) if filename.endswith(\'.nc\')]',
        '        ncfile = partmc_dir + ncfiles[tt_save]',
        '        partmc_tkappa_onescenario = read_partmc.get_partmc_variable(ncfile,\'tkappa\',specdat={},mixingstate=\'part-res\',rh=\'partmc\',temperature=\'partmc\')',
        # '        print(\',\'.join(map(str, partmc_tkappa_onescenario)))',
        '        f_kappa.write(\',\'.join(map(str, partmc_tkappa_onescenario)))',
        '        if ss<len(all_scenarios):',
        '            f_kappa.write(\'\\n\')',
        '        f_kappa.close()']
        # '        partmc_tkappas_onerepeat.append(partmc_tkappa_onescenario)',
        # '    np.savetxt(tt_dir + \'tkappas_partmc_repeat\' + str(repeat).zfill(4) + \'.txt\', partmc_tkappas_onerepeat)']
    with open(python_filename, 'w') as f:
        for line in python_lines:
            f.write(line)
            f.write('\n')
    f.close()
    
    if os.path.exists('/people/fier887/'):
        sh_lines = [
            '#!/bin/tcsh',
            '#SBATCH -A ' + project,
            '#SBATCH -p shared',
            '#SBATCH -t ' + run_time_str,
            '#SBATCH -n 1',
            '#SBATCH -o ' + runfiles_dir + 'process_kap_tt' + str(tt_save).zfill(4) + '.out',
            '#SBATCH -e ' + runfiles_dir + 'process_kap_tt' + str(tt_save).zfill(4) + '.err',
            '',
            'python ' + python_filename]
    else:
        print('not set up to run on this computer')
    
    with open(sh_filename, 'w') as f:
        for line in sh_lines:
            f.write(line)
            f.write('\n')
    f.close()
    os.system('chmod +x ' + sh_filename)
    os.system('sbatch ' + sh_filename)