
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Main script needed to make figures used in Fierce et al., 2024

@author: Laura Fierce
"""

import visualization
import pickle
import numpy as np

ensemble_prefix = 'ensemble_47_big'
ensemble_name = ensemble_prefix + '_a_both'
ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/'
ensemble_dir = ensemble_over_dir + ensemble_name + '/'
ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
t_max = ensemble_settings[0]['spec']['t_max']
dtime = ensemble_settings[0]['spec']['t_output']
Ntimes = int(t_max/dtime)+1

scenario_nums = [73,92,67,69,94]
timesteps = [0,1,2,3,4]
visualization.box_model_dsds__few(
        ensemble_dir,timesteps,scenario_nums,units='min',sharey=False,
        lnDs = np.log(np.logspace(-10,-5,100)),backward=False,save_fig=True,
        fig_width=10.,fig_height = 5.,
        dpi=500,normalize=True,add_text=False)

visualization.box_model_fccn__few(
        ensemble_dir,timesteps,scenario_nums,units='min',sharey=True,
        backward=True,save_fig=True,
        fig_width=10.,fig_height = 5.,
        dpi=500,add_text=False)

timesteps = range(0,25)
visualization.scatter_s50_vs_KL(
        ensemble_dir,timesteps,dtime,backward=True,recompute_KL=False,
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        save_fig=True,dpi=500,mode_sigs=np.log([1.8,1.6,1.8,1.6]))

table_lines = visualization.make_input_table(ensemble_dir)
f = open('inputs_table.txt','w')
for table_line in table_lines:
    f.write(table_line + '\n')
f.close()


visualization.s_onset_regimes(
        ensemble_dir,timesteps,thresh=0.01,
        recompute_KL=True,backward=False,units='min',add_lines=True,
        regime_type='regions',avg_scale='lin',save_fig=True,
        showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95))

visualization.s_onset_regimes(
        ensemble_dir,timesteps,thresh=0.99,
        recompute_KL=True,backward=False,units='min',add_lines=True,
        regime_type='regions',avg_scale='lin',save_fig=True,
        showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95))

visualization.s_onset_regimes(
        ensemble_dir,timesteps,thresh=0.25,
        recompute_KL=True,backward=False,units='min',add_lines=True,
        regime_type='regions',avg_scale='lin',save_fig=True,
        showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95))
visualization.s_onset_regimes(
        ensemble_dir,timesteps,thresh=0.5,
        recompute_KL=True,backward=False,units='min',add_lines=True,
        regime_type='regions',avg_scale='lin',save_fig=True,
        showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95))

visualization.s_onset_regimes(
        ensemble_dir,timesteps,thresh=0.75,
        recompute_KL=True,backward=False,units='min',add_lines=True,
        regime_type='regions',avg_scale='lin',save_fig=True,
        showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95))

table_lines = visualization.make_input_table(ensemble_dir)
f = open('inputs_table.txt','w')
for table_line in table_lines:
    f.write(table_line + '\n')
f.close()

timestep = timesteps[-1]
scenario_table_lines = visualization.make_scenario_table(
       ensemble_dir,scenario_nums,timestep,
       varnames = ['KL_backward','s_onset_partmc_01','s_onset_mam4_01','s_onset_partmc_50','s_onset_mam4_50','s_onset_partmc_99','s_onset_mam4_99'],
       vartypes = ['float','float','float','float','float','float','float'],
       collabs = ['scenario label','KL-divergence','$s_{\mathrm{1,MAM4}}','$s_{\mathrm{1,PartMC-MOSAIC}}','$s_{\mathrm{50,MAM4}}','$s_{\mathrm{50,PartMC-MOSAIC}}','$s_{\mathrm{99,MAM4}}','$s_{\mathrm{99,PartMC-MOSAIC}}'])#make_scenario_table(ensemble_dir,scenario_nums,timestep)
f = open('scenario_table.txt','w')
for table_line in table_lines:
    f.write(table_line + '\n')
f.close()

scenario_table_lines = visualization.make_scenario_table(
       ensemble_dir,scenario_nums,timestep,
       varnames = ['KL_backward','s_onset_ratio_01','s_onset_ratio_50','s_onset_ratio_99'],
       vartypes = ['float','float','float','float'],
       collabs = ['scenario label','KL-divergence','$s_{\mathrm{1,MAM4}}/$s_{\mathrm{1,PartMC-MOSAIC}}','$s_{\mathrm{50,MAM4}}/$s_{\mathrm{50,PartMC-MOSAIC}}','$s_{\mathrm{99,MAM4}}/$s_{\mathrm{99,PartMC-MOSAIC}}'])#make_scenario_table(ensemble_dir,scenario_nums,timestep)
f = open('scenario_table_ratio.txt','w')
for table_line in table_lines:
    f.write(table_line + '\n')
f.close()
