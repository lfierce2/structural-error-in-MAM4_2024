#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to process PartMC-MOSAIC data. 

Code was tested with the following model versions: partmc-2.6.1 and mosaic-2012-01-25

@author: Laura Fierce
"""
import os 
import netCDF4
import numpy as np
#import matplotlib.pyplot as plt

import scipy.optimize as opt
#from pyrcel.thermo import kohler_crit

# find the correct ncfile
def get_ncfile(partmc_dir, timestep, ensemble_number=1):
    ncfiles = [filename for filename in os.listdir(partmc_dir) if filename.endswith('.nc')]
    prefix = get_prefix(ncfiles[0])
    ncfile = partmc_dir + prefix + str(int(ensemble_number)).zfill(4) + '_0001_' + str(int(timestep)).zfill(8) + '.nc'
    return ncfile

def unravel_ncfiles(partmc_dir):
    ncfiles = [onefile for onefile in os.listdir(partmc_dir) if onefile.endswith('.nc')]
    repeats = []
    timesteps = []
    for ncfilename in ncfiles:
        repeat,timestep = unravel_one_ncfile(ncfilename)
        repeats.append(repeat)
        timesteps.append(timestep)
    
    repeats_sorted = []
    timesteps_sorted = []
    for repeat in range(min(repeats),max(repeats)+1):
        for timestep in range(min(timesteps),max(timesteps)+1):
            repeats_sorted.append(repeat)
            timesteps_sorted.append(timestep)
    return repeats_sorted,timesteps_sorted
    
def unravel_one_ncfile(ncfilename):
    after_prefix = ncfilename[ncfilename.find('_')+1:]
    repeat = int(after_prefix[:after_prefix.find('_')])
    
    after_repeat = after_prefix[after_prefix.find('_')+1:]
    timestep = int(after_repeat[:after_repeat.find('.')])
    return repeat,timestep
    
def get_prefix(ncfilename):
    prefix = ncfilename[:ncfilename.find('_')] + '_'
    return prefix


# get spec names
def get_spec_names(spec_group):
    if spec_group == 'all':
        spec_names =  ['SO4', 'NO3', 'Cl', 'NH4', 'MSA', 'ARO1', 'ARO2', 
                       'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                       'CO3', 'Na', 'Ca', 'OIN', 'OC', 'BC', 'H2O']
    elif spec_group == 'dry':
        spec_names =  ['SO4', 'NO3', 'Cl', 'NH4', 'MSA', 'ARO1', 'ARO2', 
                       'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                       'CO3', 'Na', 'Ca', 'OIN', 'OC', 'BC']
    elif spec_group == 'inorg':
        spec_names = ['SO4','NO3','Cl','NH4']
    elif spec_group == 'soa':
        spec_names = ['ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']
    elif spec_group == 'poa':
        spec_names = ['OC']
    elif spec_group == 'total_org':
        spec_names = ['ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2','OC']
    elif spec_group == 'bc' or spec_group == 'core' :
        spec_names = ['BC']
    elif spec_group == 'dry_shell':
        spec_names = ['SO4', 'NO3', 'Cl', 'NH4', 'MSA', 'ARO1', 'ARO2', 
                       'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                       'CO3', 'Na', 'Ca', 'OIN', 'OC']
    elif spec_group == 'shell' or spec_group == 'wet_shell':
        spec_names =  ['SO4', 'NO3', 'Cl', 'NH4', 'MSA', 'ARO1', 'ARO2', 
                       'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 
                       'CO3', 'Na', 'Ca', 'OIN', 'OC', 'H2O']
    return spec_names

def which_particles(ncfile,bconly=True,nonbconly=False):
    f = netCDF4.Dataset(ncfile)
    mass_comp = f['aero_particle_mass']    
    bcmass = mass_comp[-2,:]    
    if bconly:
        idx_bc, = np.where(bcmass>0.)
        idx = idx_bc
    elif nonbconly:
        idx_all = np.arange(len(bcmass))
        idx_bc, = np.where(bcmass>0.)
        idx = [ii not in idx_bc for ii in idx_all]
    else:
        idx_all = np.arange(len(bcmass))                
        idx = idx_all
        #one_population[mixingstate]['idx_bc']
    return idx
                
def get_partmc_variable(
        ncfile,varname,specdat={},
        mixingstate='part-res',wvl=550e-9,rh='partmc',temperature='partmc',
        monolayer=None,bconly=False,nonbconly=False):
    spec_names = get_spec_names('all')
    if varname == 'aero_particle_mass':        
        # aero_particle_vol = get_partmc_variable(ncfile,'aero_particle_vol',mixingstate=mixingstate,rh=rh,specdat=specdat)
        # density = get_partmc_variable(ncfile,'aero_density')
        # vardat = np.zeros(aero_particle_vol.shape)
        # for kk in range(vardat.shape[0]):
        #     vardat[kk,:] = aero_particle_vol[kk,:]*density[kk]
        f = netCDF4.Dataset(ncfile)
        vardat = f['aero_particle_mass']
    elif varname == 'aero_particle_vol':
        f = netCDF4.Dataset(ncfile)
        partres_particle_mass = f['aero_particle_mass']
        density = get_partmc_variable(ncfile,'aero_density')
        partres_particle_vol = np.zeros(partres_particle_mass.shape)#get_partmc_variable(ncfile,'aero_particle_vol',specdat=specdat,mixingstate='part-res')
        for kk in range(len(density)):
            partres_particle_vol[kk,:] = partres_particle_mass[kk,:]/density[kk]
        if mixingstate == 'part-res':
            vardat = partres_particle_vol
            if rh != 'partmc':
                density = get_partmc_variable(ncfile,'aero_density')
                temp = get_partmc_variable(ncfile,'temperature')
                RH = rh
                mass_comp = np.zeros(partres_particle_vol.shape)
                for kk in range(partres_particle_vol.shape[0]):
                    mass_comp[kk,:] = partres_particle_vol[kk,:]*density[kk]
                effective_kappa = get_effective_kappa(mass_comp, density, 'dry', specdat=specdat)
                Ddry = (6./np.pi*np.sum(partres_particle_vol[:-1,:],axis=0))**(1./3.)
                vardat[-1,:] = get_mass_h2o(Ddry, effective_kappa, density[-1], RH, temp)/density[-1]
        elif mixingstate == 'uniform':
            # total amount of dry aerosol held constant, but uniform mass fractions
            part_dry_vol = np.sum(partres_particle_vol[:-1,:],axis=0)
            avg_vol_frac = np.sum(partres_particle_vol[:-1,:],axis=1)/np.sum(np.sum(partres_particle_vol[:-1,:],axis=1),axis=0)
            if rh == 'partmc':
                RH = get_partmc_variable(ncfile,'relative_humidity')
            else:
                RH = rh
            vardat = np.zeros(partres_particle_vol.shape)          
            for kk in range(partres_particle_vol.shape[0] - 1):
                vardat[kk,:] = avg_vol_frac[kk]*part_dry_vol
            mass_comp = np.zeros(vardat.shape)
            for kk in range(vardat.shape[0]):
                mass_comp[kk,:] = vardat[kk,:]*density[kk]
            density = get_partmc_variable(ncfile,'aero_density')
            temp = get_partmc_variable(ncfile,'temperature')
            effective_kappa = get_effective_kappa(mass_comp, density, 'dry', specdat=specdat)
            dry_dia = (6./np.pi*np.sum(vardat[:-1,:],axis=0))**(1./3.)
            for ii in range(len(dry_dia)):
                vardat[-1,ii] = get_mass_h2o(dry_dia[ii], effective_kappa[ii], density[-1], RH, temp)/density[-1]
        elif mixingstate == 'uniform_bc':
            # total amount of BC held constant, but uniform mass fractions across BC-containing (BC-free unchanged)            
            idx, = np.where(partres_particle_vol[-2,:]>0)
            bc_vol = partres_particle_vol[-2,idx]
            avg_spec_to_bc = np.sum(partres_particle_vol[:-1,idx],axis=1)/sum(bc_vol)
            if rh == 'partmc':
                RH = get_partmc_variable(ncfile,'relative_humidity')
            else:
                RH = rh
            vardat = partres_particle_vol
            for kk in range(partres_particle_vol.shape[0] - 1):
                idx, = np.where(bc_vol>0)
                vardat[kk,idx] = avg_spec_to_bc[kk]*bc_vol[idx]
            mass_comp = np.zeros(vardat.shape)
            for kk in range(vardat.shape[0]):
                mass_comp[kk,:] = vardat[kk,:]*density[kk]
            density = get_partmc_variable(ncfile,'aero_density')
            temp = get_partmc_variable(ncfile,'temperature')
            effective_kappa = get_effective_kappa(mass_comp, density, 'dry', specdat=specdat)
            dry_dia = (6./np.pi*np.sum(vardat[:-1,:],axis=0))**(1./3.)
            for ii in range(len(dry_dia)):
                vardat[-1,:] = get_mass_h2o(dry_dia[ii], effective_kappa[ii], density[-1], RH, temp)/density[-1]
    elif varname == 'tkappa':
        mass_comp = get_partmc_variable(ncfile,'aero_particle_mass',mixingstate=mixingstate,specdat=specdat)
        density = get_partmc_variable(ncfile,'aero_density',mixingstate=mixingstate,specdat=specdat)
        vardat = get_effective_kappa(mass_comp, density, 'dry', specdat=specdat, ncfile=ncfile)
    elif varname == 'shell_tkappa':
        mass_comp = get_partmc_variable(ncfile,'aero_particle_mass',mixingstate=mixingstate,specdat=specdat)
        density = get_partmc_variable(ncfile,'aero_density',mixingstate=mixingstate,specdat=specdat)
        vardat = get_effective_kappa(mass_comp, density, 'shell', specdat=specdat)
    elif varname == 'dry_mass':
        mass_comp = get_partmc_variable(ncfile,'aero_particle_mass',mixingstate=mixingstate,specdat=specdat)
        vardat = np.sum(mass_comp[:-1,:],axis=0)
    elif varname == 'shell_mass':
        mass_comp = get_partmc_variable(ncfile,'aero_particle_mass',mixingstate=mixingstate,specdat=specdat)
        vardat = np.sum(mass_comp[:-2,:],axis=0)        
    elif varname == 'wet_mass':
        mass_comp = get_partmc_variable(ncfile,'aero_particle_mass',mixingstate=mixingstate,specdat=specdat)
        vardat = np.sum(mass_comp[:-1,:],axis=0)
    elif varname == 'bc_mass':
        mass_comp = get_partmc_variable(ncfile,'aero_particle_mass',mixingstate=mixingstate,specdat=specdat)
        vardat = mass_comp[-2,:]
    elif varname == 'BrC_mass':
        mass_comp = get_partmc_variable(ncfile,'aero_particle_mass',mixingstate=mixingstate,specdat=specdat)
        all_spec_names = get_spec_names('all')
        spec_names = get_spec_names('soa')
        # print(all_spec_names,spec_names,[any([one_spec == spec_name for spec_name in spec_names]) for one_spec in all_spec_names])
        idx_soa,=np.where([any([one_spec == spec_name for spec_name in spec_names]) for one_spec in all_spec_names])
        vardat = np.sum(mass_comp[idx_soa,:],axis=0)
    elif varname == 'dry_vol':
        vol_comp = get_partmc_variable(ncfile,'aero_particle_vol',mixingstate=mixingstate,specdat=specdat)
        vardat = np.sum(vol_comp[:-1,:],axis=0)
    elif varname == 'shell_vol':
        vol_comp = get_partmc_variable(ncfile,'aero_particle_vol',mixingstate=mixingstate,specdat=specdat)
        vardat = np.sum(vol_comp[:-2,:],axis=0)             
    elif varname == 'wet_vol':
        vol_comp = get_partmc_variable(ncfile,'aero_particle_vol',mixingstate=mixingstate,specdat=specdat,rh=rh)
        vardat = np.sum(vol_comp,axis=0)
    elif varname == 'wet_area':
        wet_diameter = get_partmc_variable(ncfile,'wet_diameter',mixingstate=mixingstate,specdat=specdat)
        vardat = np.pi*wet_diameter**2
    elif varname == 'bc_vol':
        aero_particle_vol = get_partmc_variable(ncfile,'aero_particle_vol',specdat=specdat,mixingstate=mixingstate)
        vardat = aero_particle_vol[-2,:]
    elif varname == 'BrC_vol':
        aero_particle_vol = get_partmc_variable(ncfile,'aero_particle_vol',specdat=specdat,mixingstate=mixingstate)
        all_spec_names = get_spec_names('all')
        spec_names = get_spec_names('soa')
        # print(all_spec_names,spec_names,[any([one_spec == spec_name for spec_name in spec_names]) for one_spec in all_spec_names])
        idx_soa,=np.where([any([one_spec == spec_name for spec_name in spec_names]) for one_spec in all_spec_names])
        vardat = np.sum(aero_particle_vol[idx_soa,:],axis=0)
    elif varname == 'core_vol':
        aero_particle_vol = get_partmc_variable(ncfile,'aero_particle_vol',specdat=specdat,mixingstate=mixingstate)
        vardat = np.sum([aero_particle_vol[idx,:] for idx in [17,18]],axis=0)
    elif varname == 'dry_diameter':
        dry_vol = get_partmc_variable(ncfile,'dry_vol',mixingstate=mixingstate,specdat=specdat)
        vardat = (dry_vol*6./np.pi)**(1./3.)
    elif varname == 'wet_diameter':
        wet_vol = get_partmc_variable(ncfile,'wet_vol',mixingstate=mixingstate,specdat=specdat,rh=rh)
        vardat = (wet_vol*6./np.pi)**(1./3.)
    elif varname == 'bc_diameter':
        bc_vol = get_partmc_variable(ncfile,'bc_vol',specdat=specdat,mixingstate=mixingstate)
        vardat = (bc_vol*6./np.pi)**(1./3.)
    elif varname == 'core_diameter':
        core_vol = get_partmc_variable(ncfile,'core_vol',specdat=specdat,mixingstate=mixingstate)
        vardat = (core_vol*6./np.pi)**(1./3.)
    elif varname == 'Rbc_mass':
        bc_mass = get_partmc_variable(ncfile,'bc_mass',specdat=specdat,mixingstate=mixingstate)
        wet_mass = get_partmc_variable(ncfile,'wet_mass',mixingstate=mixingstate,specdat=specdat,rh=rh)
        vardat = (wet_mass - bc_mass)/wet_mass
    elif varname == 'Rbc_vol':
        bc_vol = get_partmc_variable(ncfile,'bc_vol',specdat=specdat,mixingstate=mixingstate)
        wet_vol = get_partmc_variable(ncfile,'wet_vol',mixingstate=mixingstate,specdat=specdat,rh=rh)
        vardat = (wet_vol - bc_vol)/wet_vol
    elif varname == 'abs_crossect':
        if rh=='partmc':
            RH = get_partmc_variable(ncfile,'relative_humidity')
        else:
            RH = rh
        temp = get_partmc_variable(ncfile,'temperature')
        vardat, scat_crossect = get_crossection_Mie(ncfile,  wvl, temp, RH,specdat=specdat, mixingstate=mixingstate)
    elif varname == 'scat_crossect':
        if rh=='partmc':
            RH = get_partmc_variable(ncfile,'relative_humidity')
        else:
            RH = rh
        temp = get_partmc_variable(ncfile,'temperature')
        abs_crossect, vardat = get_crossection_Mie(ncfile,  wvl, temp, RH,specdat=specdat, mixingstate=mixingstate)
        
    elif varname == 's_critical':
        if temperature == 'partmc':
            temp = get_partmc_variable(ncfile, 'temperature')
        else:
            temp = temperature
        tkappa = get_partmc_variable(ncfile,'tkappa', specdat=specdat, mixingstate=mixingstate)
        dry_dia = get_partmc_variable(ncfile,'dry_diameter', specdat=specdat, mixingstate=mixingstate)
        vardat = process_compute_Sc(temp, dry_dia, tkappa)
    elif varname == 's_critical_uniformComp':
        if temperature == 'partmc':
            temp = get_partmc_variable(ncfile, 'temperature')
        else:
            temp = temperature
        tkappa_orig = get_partmc_variable(ncfile,'tkappa', specdat=specdat, mixingstate=mixingstate)
        dry_dia = get_partmc_variable(ncfile,'dry_diameter', specdat=specdat, mixingstate=mixingstate)
        aero_num = get_partmc_variable(ncfile,'aero_num_conc', specdat=specdat, mixingstate=mixingstate)
        tkappa = np.ones(len(dry_dia))*np.sum(aero_num*dry_dia**3*tkappa_orig)/np.sum(aero_num*dry_dia**3)
        vardat = process_compute_Sc(temp, dry_dia, tkappa)
    else:
        f = netCDF4.Dataset(ncfile)
        vardat = f.variables[varname][:]
    return vardat
    
def get_crossection_Mie(ncfile,  wvl, temp, RH, 
                    return_this = 'Mie', # other options: BrC_core, BrC_shell, uncoated
                    specdat = {}, mixingstate='part-res',
                    return_idx_bc = False, return_Dwet=False, wet_dia=False, return_coatRI=False):
    if return_this == 'Mie':
        return_Mie = True
        return_BrC_core = False    
        return_BrC_shell = False
        return_uncoated = False
    elif return_this == 'BrC_core':
        return_BrC_core = True        
        return_Mie = False
        return_BrC_shell = False
        return_uncoated = False        
    elif return_this == 'BrC_shell':
        return_BrC_shell = True
        return_Mie = False
        return_BrC_core = False    
        return_uncoated = False        
    elif return_this == 'uncoated':
        return_uncoated = True
        return_Mie = False
        return_BrC_core = False    
        return_BrC_shell = False
    elif return_this == 'Mie_clear':
        return_Mie = True
        return_BrC_core = False    
        return_BrC_shell = False
        return_uncoated = False
    
    mass_comp = get_partmc_variable(ncfile, 'aero_particle_mass', specdat=specdat, mixingstate=mixingstate, rh='partmc')
    density = get_partmc_variable(ncfile, 'aero_density', specdat=specdat, mixingstate=mixingstate)
    
    # spec_names = get_spec_names('all')
    core_dia = get_partmc_variable(ncfile,'bc_diameter', specdat=specdat, mixingstate=mixingstate)
    # core_spec = get_spec_names('core')
    
    #idx_core, = np.where(np.in1d(spec_names, core_spec))
    # idx_core = np.asarray([-2])
    m_core = get_effective_ri(mass_comp,density, 'core', wvl=wvl, specdat=specdat)
    m_shell = get_effective_ri(mass_comp,density, 'shell', wvl=wvl, specdat=specdat)
    idx_bc, = np.where(core_dia>0)
    optical_crossect = np.zeros([len(core_dia),2])
    if return_uncoated:
        if len(idx_bc)>0:
            optical_crossect[idx_bc,:] = np.vstack([compute_crossect_oneparticle(core_dia[i], core_dia[i], m_core[i], m_core[i], wvl) for i in idx_bc])
    
    if return_Mie or return_BrC_core or return_BrC_shell:
        tkappa = get_effective_kappa(mass_comp[:-1,:], density[:-1], 'dry', specdat=specdat)    
        dry_dia = get_partmc_variable(ncfile,'dry_diameter', specdat=specdat, mixingstate=mixingstate)
        if np.any(wet_dia == False):
            wet_dia = np.vstack([get_Dwet(dry_dia[i], tkappa[i], RH, temp) for i in range(len(dry_dia))])[:,0]
        # mass_comp[-1,:] = np.pi/6*(wet_dia**3 - dry_dia**3)*density[-1]
        
        if return_this == 'Mie':
            optical_crossect = np.vstack([compute_crossect_oneparticle(core_dia[i], wet_dia[i], m_core[i], m_shell[i], wvl) for i in range(len(dry_dia))])
        elif return_this == 'Mie_clear':
            optical_crossect = np.vstack([compute_crossect_oneparticle(core_dia[i], wet_dia[i], m_core[i], m_shell[i], wvl) for i in range(len(dry_dia))])
        elif return_this == 'BrC_shell':
            optical_crossect = np.vstack([compute_crossect_oneparticle(wet_dia[i], wet_dia[i], m_shell[i], m_shell[i], wvl) for i in range(len(dry_dia))])
        elif return_this == 'BrC_core':
            optical_crossect = np.vstack([compute_crossect_oneparticle(core_dia[i], core_dia[i], m_shell[i], m_shell[i], wvl) for i in range(len(dry_dia))])            
        
    abs_crossect = optical_crossect[:,0]
    scat_crossect = optical_crossect[:,1]
    
    if return_idx_bc:
        if return_Dwet:
            if return_coatRI:
                return abs_crossect, scat_crossect, wet_dia, m_shell, idx_bc
            else:
                return abs_crossect, scat_crossect, wet_dia, idx_bc
        else:
            return abs_crossect, scat_crossect, idx_bc
    else:
        if return_Dwet:
            if return_coatRI:            
                return abs_crossect, scat_crossect, wet_dia, m_shell
            else:
                return abs_crossect, scat_crossect, wet_dia
        else:
            return abs_crossect, scat_crossect

def get_crossection_Mie__bconly(ncfile,  wvl, temp, RH, 
                    return_these = ['Mie','BrC_shell','BrC_core','Mie_clear','uncoated'], # other options: BrC_core, BrC_shell, uncoated
                    specdat = {}, mixingstate='part-res',
                    return_idx_bc = False, return_Dwet=False, wet_dia=False, return_coatRI=False):
    
    mass_comp = get_partmc_variable(ncfile, 'aero_particle_mass', specdat=specdat, mixingstate=mixingstate, rh='partmc')
    density = get_partmc_variable(ncfile, 'aero_density', specdat=specdat, mixingstate=mixingstate)
    
    # spec_names = get_spec_names('all')
    core_dia = get_partmc_variable(ncfile,'bc_diameter', specdat=specdat, mixingstate=mixingstate)
    # core_spec = get_spec_names('core')
    
    # idx_core = np.asarray([-2])
    m_core = get_effective_ri(mass_comp,density, 'core', wvl=wvl, specdat=specdat)    
    
    idx_bc, = np.where(core_dia>0)
    # bcmass = get_partmc_variable(ncfile,'bc_mass', specdat=specdat, mixingstate=mixingstate)    
    tkappa = get_effective_kappa(mass_comp[:-1,:], density[:-1], 'dry', specdat=specdat)    
    dry_dia = get_partmc_variable(ncfile,'dry_diameter', specdat=specdat, mixingstate=mixingstate)
    wet_dia = np.zeros(dry_dia.shape)
    if np.any(wet_dia == False):
        wet_dia[idx_bc] = np.vstack([get_Dwet(dry_dia[i], tkappa[i], RH, temp) for i in idx_bc])[:,0]
    mass_comp[-1,idx_bc] = np.pi/6*(wet_dia[idx_bc]**3 - dry_dia[idx_bc]**3)*density[-1]
    
    absorbtion_crossects = np.zeros([len(return_these),len(dry_dia)])
    scattering_crossects = np.zeros([len(return_these),len(dry_dia)])
    
    # spec_names = get_spec_names('all')
    # shell_spec = get_spec_names('shell')
    
    
    for kk,return_this in enumerate(return_these):
        m_shell = get_effective_ri(mass_comp,density, 'shell', wvl=wvl, specdat=specdat)
        if return_this == 'Mie':
            optical_crossects = np.vstack([compute_crossect_oneparticle(core_dia[i], wet_dia[i], m_core[i], m_shell[i], wvl) for i in idx_bc])
            absorbtion_crossects[kk,idx_bc] = optical_crossects[:,0].copy()
            scattering_crossects[kk,idx_bc] = optical_crossects[:,1].copy()
        elif return_this == 'Mie_clear':
            optical_crossects2 = np.vstack([compute_crossect_oneparticle(core_dia[i], wet_dia[i], m_core[i], m_shell[i].real, wvl) for i in idx_bc])
            absorbtion_crossects[kk,idx_bc] = optical_crossects2[:,0].copy()
            scattering_crossects[kk,idx_bc] = optical_crossects2[:,1].copy()
        elif return_this == 'BrC_core':
            optical_crosseccts3 = np.vstack([compute_crossect_oneparticle(core_dia[i], core_dia[i], m_shell[i], m_shell[i], wvl) for i in idx_bc])
            absorbtion_crossects[kk,idx_bc] = optical_crosseccts3[:,0].copy()
            scattering_crossects[kk,idx_bc] = optical_crosseccts3[:,1].copy()
        elif return_this == 'BrC_shell':
            optical_crosseccts4 = np.vstack([compute_crossect_oneparticle(wet_dia[i], wet_dia[i], m_shell[i], m_shell[i], wvl) for i in idx_bc])
            absorbtion_crossects[kk,idx_bc] = optical_crosseccts4[:,0].copy()
            scattering_crossects[kk,idx_bc] = optical_crosseccts4[:,1].copy()
        elif return_this == 'uncoated':
            optical_crosseccts5 = np.vstack([compute_crossect_oneparticle(core_dia[i], core_dia[i], m_core[i], m_core[i], wvl) for i in idx_bc])
            absorbtion_crossects[kk,idx_bc] = optical_crosseccts5[:,0].copy()
            scattering_crossects[kk,idx_bc] = optical_crosseccts5[:,1].copy()
    
    if return_idx_bc:
        if return_Dwet:
            if return_coatRI:
                return absorbtion_crossects, scattering_crossects, wet_dia, m_shell, idx_bc
            else:
                return absorbtion_crossects, scattering_crossects, wet_dia, idx_bc
        else:
            return absorbtion_crossects, scattering_crossects, idx_bc
    else:
        if return_Dwet:
            if return_coatRI:            
                return absorbtion_crossects, scattering_crossects, wet_dia, m_shell
            else:
                return absorbtion_crossects, scattering_crossects, wet_dia
        else:
            return absorbtion_crossects, scattering_crossects
        
def get_crossection_Mie__addbcfree(ncfile,  wvl, temp, RH, 
                    return_these = ['Mie','BrC_shell','BrC_core','Mie_clear','uncoated'], # other options: BrC_core, BrC_shell, uncoated
                    specdat = {}, mixingstate='part-res',
                    return_idx_bcfree = False, return_Dwet=False, wet_dia=False, return_coatRI=False):
    
    mass_comp = get_partmc_variable(ncfile, 'aero_particle_mass', specdat=specdat, mixingstate=mixingstate, rh='partmc')[:,:]
    density = get_partmc_variable(ncfile, 'aero_density', specdat=specdat, mixingstate=mixingstate)
    
    # spec_names = get_spec_names('all')
    core_dia = get_partmc_variable(ncfile,'bc_diameter', specdat=specdat, mixingstate=mixingstate)
    # core_spec = get_spec_names('core')
    
    # idx_core = np.asarray([-2])
    m_core = get_effective_ri(mass_comp,density, 'core', wvl=wvl, specdat=specdat)    
    
    idx_bcfree, = np.where(core_dia==0)
    # bcmass = get_partmc_variable(ncfile,'bc_mass', specdat=specdat, mixingstate=mixingstate)    
    tkappa = get_effective_kappa(mass_comp[:-1,:], density[:-1], 'dry', specdat=specdat)    
    dry_dia = get_partmc_variable(ncfile,'dry_diameter', specdat=specdat, mixingstate=mixingstate)
    wet_dia = np.zeros(dry_dia.shape)
    if np.any(wet_dia == False):
        wet_dia[idx_bcfree] = np.vstack([get_Dwet(dry_dia[i], tkappa[i], RH, temp) for i in idx_bcfree])[:,0]
    mass_h2o = np.zeros(len(wet_dia))
    mass_h2o[idx_bcfree] = np.pi/6*(wet_dia[idx_bcfree]**3 - dry_dia[idx_bcfree]**3)*density[-1]
    mass_comp[-1,:] = mass_h2o #np.pi/6*(wet_dia[idx_bcfree]**3 - dry_dia[idx_bcfree]**3)*density[-1]
    
    absorbtion_crossects = np.zeros([len(return_these),len(dry_dia)])
    scattering_crossects = np.zeros([len(return_these),len(dry_dia)])
    
    # spec_names = get_spec_names('all')
    # shell_spec = get_spec_names('shell')
    
    
    for kk,return_this in enumerate(return_these):
        m_shell = get_effective_ri(mass_comp,density, 'shell', wvl=wvl, specdat=specdat)
        if return_this == 'Mie':
            optical_crossects = np.vstack([compute_crossect_oneparticle(core_dia[i], wet_dia[i], m_core[i], m_shell[i], wvl) for i in idx_bcfree])
            absorbtion_crossects[kk,idx_bcfree] = optical_crossects[:,0].copy()
            scattering_crossects[kk,idx_bcfree] = optical_crossects[:,1].copy()
        else:
            print('only coded for return_this == \'Mie\'')
    
    if return_idx_bcfree:
        if return_Dwet:
            if return_coatRI:
                return absorbtion_crossects, scattering_crossects, wet_dia, m_shell, return_idx_bcfree
            else:
                return absorbtion_crossects, scattering_crossects, wet_dia, return_idx_bcfree
        else:
            return absorbtion_crossects, scattering_crossects, return_idx_bcfree
    else:
        if return_Dwet:
            if return_coatRI:            
                return absorbtion_crossects, scattering_crossects, wet_dia, m_shell
            else:
                return absorbtion_crossects, scattering_crossects, wet_dia
        else:
            return absorbtion_crossects, scattering_crossects    
        
def compute_crossect_oneparticle(core_dia, wet_dia, m_core, m_shell, wvl):
    import PyMieScatt
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

def get_Dwet(Ddry, kappa, RH, temp):
    if RH>0. and kappa>0.:
        sigma_w = 0.072; rho_w = 1000; M_w = 18/1e3; R=8.314;
        A = 4*sigma_w*M_w/(R*temp*rho_w)
        zero_this = lambda gf: RH/np.exp(A/(Ddry*gf))-(gf**3.-1.)/(gf**3.-(1.-kappa))
        return Ddry*opt.brentq(zero_this,1.,10000000.)
        #return Ddry*fsolve(zero_this,10.)[0]
    else:
        return Ddry
    
    
def get_mass_h2o(Ddry, effective_kappa, density_h2o, RH, temp):
    Dwet = get_Dwet(Ddry, effective_kappa, RH, temp)
    mass_h2o = (np.pi/6.*Dwet**3. - np.pi/6.*Ddry**3.)*density_h2o
    return mass_h2o

def get_effective_kappa(mass_comp, density, spec_group, ncfile='', specdat={},fromMAM=True):
    all_spec_names = get_spec_names('all')
    spec_names = get_spec_names(spec_group)
    
    numerator = 0.
    denominator = 0.
    for kk,spec_name in enumerate(spec_names):
        spec_kappa = get_spec_partmc_kappa(spec_name,fromMAM=fromMAM)
        spec_density = get_spec_partmc_density(spec_name)
        
        k = np.where(np.in1d(all_spec_names, spec_name))[0][0]
        numerator += spec_kappa*mass_comp[k,:]/spec_density
        denominator += mass_comp[k,:]/spec_density
            
    effective_kappa = numerator/denominator
    return effective_kappa

def get_effective_ri(mass_comp, density, spec_group, wvl=550e-9, specdat={}):
    all_spec_names = get_spec_names('all')
    spec_names = get_spec_names(spec_group)
    numerator_imag = 0.
    numerator_real = 0.
    denominator = 0.

    for kk,spec_name in enumerate(spec_names):
        spec_ri = get_spec_partmc_RI(spec_name)
        spec_density = get_spec_partmc_density(spec_name)
        
        k = np.where(np.in1d(all_spec_names, spec_name))[0][0]
        numerator_imag += np.imag(spec_ri)*mass_comp[k,:]/spec_density
        numerator_real += np.real(spec_ri)*mass_comp[k,:]/spec_density
        denominator += mass_comp[k,:]/spec_density
        
    effective_ri = 1j*numerator_imag/denominator + numerator_real/denominator
    return effective_ri

def get_effective_density(mass_comp,density,spec_group):
    all_spec_names = get_spec_names('all')
    spec_names = get_spec_names(spec_group)
    
    numerator = 0.
    denominator = 0.
    for spec_name in spec_names:
        spec_density = get_spec_partmc_RI(spec_name)
        
        k = np.where(np.in1d(all_spec_names, spec_name))[0][0]#np.in1d(all_spec_names, spec_name)
        numerator += mass_comp[k,:]
        denominator += mass_comp[k,:]/spec_density[k]
    effective_density = numerator/denominator
    return effective_density


def get_spec_partmc_kappa(spec_name,fromMAM=True):
    if fromMAM:
        if spec_name in ['SO4','NO3','NH4']:
            spec_kappa = 0.507
        elif spec_name.startswith('OC'):
            spec_kappa = 0.01
        elif spec_name in ['ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']:
            spec_kappa = 0.1
        elif spec_name.startswith('BC'):
            spec_kappa = 1e-10
        elif spec_name.startswith('OIN'):
            spec_kappa = 0.068
        elif spec_name in ['Na','Cl']:
            spec_kappa = 1.160
        elif spec_name in ['MSA','CO3','Ca']:
            spec_kappa = 0.53
    else:
        if spec_name in ['SO4','NO3','NH4']:
            spec_kappa = 0.65
        elif spec_name.startswith('OC'):
            spec_kappa = 0.001
        elif spec_name in ['ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']:
            spec_kappa = 0.1
        elif spec_name.startswith('BC'):
            spec_kappa = 0.
        elif spec_name.startswith('OIN'):
            spec_kappa = 0.1
        elif spec_name in ['Na','Cl']:
            spec_kappa = 1.28
        elif spec_name in ['MSA','CO3','Ca']:
            spec_kappa = 0.53
    return spec_kappa

def get_spec_partmc_RI(spec_name):
    if spec_name in ['SO4','NO3','NH4']:
        spec_RI = 1.5
    elif spec_name.startswith('OC'):
        spec_RI = 1.45
    elif spec_name in ['ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']: # likely varies with wavelength (e.g. BrC)
        spec_RI = 1.45
    elif spec_name.startswith('BC'): # often assumed to be invariate with wavelength
        spec_RI = 1.82 + 0.74j
    elif spec_name.startswith('OIN'): # varies with dust type and a bit with wavelength
        spec_RI = 1.55 + 3e-3j # https://www.tandfonline.com/doi/abs/10.1111/j.1600-0889.2008.00383.x
    elif spec_name in ['Na','Cl']: # varies a bit with wavelength
        spec_RI = 1.55 # https://refractiveindex.info/?shelf=main&book=NaCl&page=Li
    elif spec_name in ['MSA','CO3','Ca']:
        spec_RI = 1.55
    # elif spec_name.startswith('CO3'):
    #     spec_RI = 1.55 + 3e-3j
    elif spec_name.startswith('H2O'):
        spec_RI = 1.33 
    return spec_RI

def get_spec_partmc_density(spec_name):
    if spec_name in ['SO4','NO3','NH4']:
        spec_density = 1770.
    elif spec_name.startswith('OC'):
        spec_density = 1000.
    elif spec_name in ['ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']: # likely varies with wavelength (e.g. BrC)
        spec_density = 1000.
    elif spec_name.startswith('BC'):
        spec_density = 1700.
    elif spec_name.startswith('OIN'):
        spec_density = 2600.
    elif spec_name in ['Na','Cl']:
        spec_density = 1900.
    elif spec_name.startswith('MSA'):
        spec_density = 1800.
    elif spec_name in ['CO3','Ca']:
        spec_density = 2600.
    elif spec_name.startswith('H2O'):
        spec_density = 1000.
    return spec_density

def process_compute_Sc(temp, dia, tkappa,return_crit_diam=False): # % numActivated/cm^2
    # returns s_crit, in %
    sigma_w = 71.97/1000.; # mN/m  J - Nm --- mN/m = mJ/m^2 = 1000 J/m^2
    m_w = 18/1000.; #kg/mol
    R = 8.314; # J/mol*K
    rho_w = 1000.; # kg/m^3
    A = 4.*sigma_w*m_w/(R*temp*rho_w);
    
    ssat = np.zeros(len(dia));
    if return_crit_diam:
        crit_diam = np.zeros(len(dia));
        for i in range(len(dia)):
            d = dia[i]
            f = lambda x: compute_Sc_funsixdeg(x,A,tkappa[i],d)
            soln = opt.root(f,d*10);
            x = soln.x
            crit_diam[i]=x 
            ssat[i]=(x**3.0-d**3.0)/(x**3-d**3*(1.0-tkappa[i]))*np.exp(A/x);
            ssat[i]=(ssat[i]-1.0)*100
        return ssat,crit_diam
    else:
        for i in range(len(dia)):
            if tkappa[i]>0.2: # equation 10 from Petters and Kreidenweis, 2007
                ssat[i] = (np.exp((4.*A**3./(27.*dia[i]**3.*tkappa[i]))**(0.5))-1.)*100.
            else:
                d = dia[i]
                f = lambda x: compute_Sc_funsixdeg(x,A,tkappa[i],d)
                soln = opt.root(f,d*10);
                x = soln.x
                ssat[i]=(x**3.0-d**3.0)/(x**3-d**3*(1.0-tkappa[i]))*np.exp(A/x);
                ssat[i]=(ssat[i]-1.0)*100
        return ssat

def process_compute_sc_one(temp, dry_dia, tkappa): # % numActivated/cm^2

    sigma_w = 71.97/1000.; # mN/m  J - Nm --- mN/m = mJ/m^2 = 1000 J/m^2
    m_w = 18/1000.; #kg/mol
    R = 8.314; # J/mol*K
    rho_w = 1000.; # kg/m^3
    A = 4.*sigma_w*m_w/(R*temp*rho_w);
    
    if tkappa>0.2:
        ssat = (np.exp((4.*A**3./(27.*dry_dia**3.*tkappa))**(0.5))-1.)*100.
    else:
        f = lambda x: compute_Sc_funsixdeg(x,A,tkappa,dry_dia)
        soln = opt.root(f,dry_dia*10);
        x = soln.x[0]
        ssat = (((x**3.0-dry_dia**3.0)/(x**3-dry_dia**3*(1.0-tkappa))*np.exp(A/x)) - 1.)*100.
    return ssat

def process_compute_lnSc_one(temp, dry_dia, tkappa):
    sigma_w = 71.97/1000.; # mN/m  J - Nm --- mN/m = mJ/m^2 = 1000 J/m^2
    m_w = 18/1000.; #kg/mol
    R = 8.314; # J/mol*K
    rho_w = 1000.; # kg/m^3
    A = 4.*sigma_w*m_w/(R*temp*rho_w);
    
    if tkappa>0.2:
#        ssat = (np.exp((4.*A**3./(27.*dry_dia**3.*tkappa))**(0.5))-1.)*100.
#        Sc = np.exp((4.*A**3./(27.*dry_dia**3.*tkappa))**(0.5))
        lnSc = (4.*A**3./(27.*dry_dia**3.*tkappa))**(0.5)   
    else:
        f = lambda x: compute_Sc_funsixdeg(x,A,tkappa,dry_dia)
        soln = opt.root(f,dry_dia*10);
        x = soln.x[0]
#        ssat = (((x**3.0-dry_dia**3.0)/(x**3-dry_dia**3*(1.0-tkappa))*np.exp(A/x)) - 1.)*100.
#        Sc = ((x**3.0-dry_dia**3.0)/(x**3-dry_dia**3*(1.0-tkappa))*np.exp(A/x))
        lnSc = np.log((x**3.0-dry_dia**3.0)) - np.log(x**3-dry_dia**3*(1.0-tkappa)) + A/x
    return lnSc

def process_compute_Ddry_thresh(s_env, temp, tkappa):
#    Ddry_thresh = opt.brentq(lambda Ddry: sc_vs_drydia(Ddry)-s_env,1e-11,1e6)
    Ddry_guess = process_compute_Ddry_thresh_approx(s_env,temp,tkappa)
    if tkappa>0.2:
        Ddry_thresh = Ddry_guess
    else:
#        sc_vs_drydia = lambda Ddry: process_compute_sc_one(temp, Ddry, tkappa)
#        Ddry_thresh = opt.fsolve(lambda Ddry: sc_vs_drydia(Ddry)-s_env,Ddry_guess)[0]; 
        lnS_env = np.log(1. + s_env/100.)
        lnSc_vs_drydia = lambda lnDdry: process_compute_lnSc_one(temp, np.exp(lnDdry), tkappa)
        Ddry_thresh = np.exp(opt.fsolve(lambda lnDdry: lnSc_vs_drydia(lnDdry)-lnS_env,np.log(Ddry_guess))[0]); #print(Ddry_thresh
#        Ddry_thresh = opt.brenth(lambda Ddry: lnSc_vs_drydia(Ddry)-lnS_env,Ddry_guess*1e-5,Ddry_guess)[0]
#        Ddry_thresh = opt.broyden2(lambda Ddry: lnSc_vs_drydia(Ddry) - lnS_env,Ddry_guess/10.)
#        Ddry_thresh = opt.brentq(lambda Ddry: lnSc_vs_drydia(Ddry)-lnS_env,Ddry_guess*1e-10,Ddry_guess/5.)        
        
#        sc_vs_drydia = lambda Ddry: process_compute_sc_one(temp, Ddry, tkappa)
#        Ddry_thresh = opt.fsolve(lambda Ddry: sc_vs_drydia(Ddry)-s_env,Ddry_guess)[0]; 
#        Ddry_thresh = opt.fsolve(lambda Ddry: lnSc_vs_drydia(Ddry)-lnS_env,Ddry_guess*10.)
#        if Ddry_thresh<=0:
#        Ddry_thresh = opt.brentq(lambda Ddry: lnSc_vs_drydia(Ddry)-lnS_env,Ddry_guess*1e-10,Ddry_guess*10.)
#        if Ddry_thresh <=0.:
#            sc_vs_drydia = lambda Ddry: process_compute_sc_one(temp, Ddry, tkappa)
#            Ddry_thresh = opt.brentq(lambda Ddry: sc_vs_drydia(Ddry)-s_env,Ddry_guess*1e-6,Ddry_guess*10.)
#            Ddry_thresh = opt.fsolve(lambda Ddry: sc_vs_drydia(Ddry)-s_env,Ddry_guess)[0]; print(Ddry_thresh)
#            Ddry_thresh = opt.fsolve(lambda Ddry: lnSc_vs_drydia(Ddry)-lnS_env,Ddry_guess)[0]; #print(Ddry_thresh)
            
    return Ddry_thresh
    
def process_compute_Ddry_thresh_approx(s_env,temp,tkappa):
#    ssat = (np.exp((4.*A**3./(27.*dia[i]**3.*tkappa[i]))**(0.5))-1.)*100.
    sigma_w = 71.97/1000.; # mN/m  J - Nm --- mN/m = mJ/m^2 = 1000 J/m^2
    m_w = 18/1000.; #kg/mol
    R = 8.314; # J/mol*K
    rho_w = 1000.; # kg/m^3
    A = 4.*sigma_w*m_w/(R*temp*rho_w);
    
    S = s_env/100. + 1. 
    Ddry_thresh = ((4.*A**3.)/(27.*tkappa*np.log(S)**2.))**(1./3)
    return Ddry_thresh

def compute_Sc_funsixdeg(diam,A,tkappa,dry_diam):
    c6=1.0;
    c4=-(3.0*(dry_diam**3)*tkappa/A); 
    c3=-(2.0-tkappa)*(dry_diam**3); 
    c0=(dry_diam**6.0)*(1.0-tkappa);
    
    z = c6*(diam**6.0) + c4*(diam**4.0) + c3*(diam**3.0) + c0;
    return z

