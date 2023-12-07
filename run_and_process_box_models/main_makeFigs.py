
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
#timesteps = np.hstack([np.arange(0,8),np.arange(9,25)])
visualization.scatter_s50_vs_KL(
        ensemble_dir,timesteps,dtime,backward=True,recompute_KL=False,
        ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
        save_fig=True,dpi=500,mode_sigs=np.log([1.8,1.6,1.8,1.6]))

table_lines = visualization.make_input_table(ensemble_dir)
f = open('inputs_table.txt','w')
for table_line in table_lines:
    f.write(table_line + '\n')
f.close()
# visualization.KL_vs_time__few(
#         ensemble_prefix,scenario_nums,timesteps,
#         processes = ['both'],labs=['a'],
#         # processes = ['both','cond','coag'],labs=['a','b','c'],
#         unit = 'min',dpi=500.,
#         ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
#         backward=True,recompute_KL=False,save_fig=True)
# timesteps = range(1,4)


# scenario_nums = visualization.box_model_dsds__regimes(
#         ensemble_dir,timesteps,
#         regime_type='regions',avg_scale='lin',units='min',
#         lnDs = np.log(np.logspace(-10,-5,100)),backward=False,sharey=False,
#         fig_width=7.5,fig_height = 5., save_fig=True,recompute_KL=True,
#         dpi=500,normalize=True,add_text=False)

# scenario_nums = [41,58,59,73,76,80,87,92,94,98,99]


# visualization.KL_regimes(
#         ensemble_dir,timesteps,
#         recompute_KL=False,backward=True,units='h',add_lines=True,
#         regime_type='regions',avg_scale='lin',save_fig=True,
#         showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95))


# timesteps = range(145)
# visualization.s_thresh_vs_aging(
#         ensemble_prefix,timesteps,showfliers=False,recompute_KL=True, thresh = 0.5,
#         Nbins=10,save_fig=True,dpi=500.,fig_width=7.5,fig_height=3.,whis=(5,95))

# timesteps = range(0,7)
# visualization.box_model_fccn__all_scenarios(
#         ensemble_dir,timesteps, dtime, 
#         fig_width=7.5,fig_height=1.25,
#         save_fig=True,dpi=200)

# visualization.box_model_dsds__all_scenarios(
#         ensemble_dir,timesteps, 
#         fig_width=7.5,fig_height=1.25,
#         save_fig=True,dpi=200)



# timesteps = range(0,145)
# visualization.KL_vs_aging(
#         ensemble_prefix,timesteps,showfliers=False,recompute_KL=False,
#         Nbins=10,save_fig=True,dpi=500.,fig_width=7.5,fig_height=3.,whis=(5,95))

# timesteps = range(0,7)
# showfliers = False
# visualization.KL_regimes(
#     ensemble_dir,timesteps,
#     recompute_KL=False,backward=True,units='min',
#     regime_type='regions',avg_scale='lin',save_fig=True,
#     add_lines=True,showfliers=showfliers)


# visualization.s_onset_regimes(
#         ensemble_dir,timesteps,thresh=0.5,
#         recompute_KL=True,backward=False,units='min',add_lines=True,
#         regime_type='regions',avg_scale='lin',save_fig=True,
#         showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95))

# visualization.s_onset_regimes(
#         ensemble_dir,timesteps,thresh=0.25,
#         recompute_KL=True,backward=False,units='min',add_lines=True,
#         regime_type='regions',avg_scale='lin',save_fig=True,
#         showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95))

# visualization.s_onset_regimes(
#         ensemble_dir,timesteps,thresh=0.75,
#         recompute_KL=True,backward=False,units='min',add_lines=True,
#         regime_type='regions',avg_scale='lin',save_fig=True,
#         showfliers=False,fig_width=6.5,fig_height=3.5,whis=(5,95))

# timesteps = range(0,25)

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
# ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/'

# ensemble_dir = ensemble_over_dir + ensemble_name + '/'

# ensemble_settings = pickle.load(open(ensemble_dir + 'ensemble_settings.pkl','rb'))
# dtime = ensemble_settings[0]['spec']['t_output']

# h2so40 = np.zeros(len(ensemble_settings))
# h2so4_emit = np.zeros(len(ensemble_settings))
# for ii in range(len(ensemble_settings)):
#     h2so40[ii] = ensemble_settings[ii]['gas_init']['H2SO4']
#     h2so4_emit[ii] = ensemble_settings[ii]['gas_emit']['H2SO4']

# # timesteps = range(0,7)
# # timesteps = range(0,25)
# timesteps = range(0,7)
# visualization.KL_vs_aging(ensemble_prefix,timesteps,Nbins=10,save_fig=True,showfliers=showfliers)

# # timesteps = range(0,12,2)
# timesteps = range(0,7)
# # timesteps = range(0,25,4)
# # timesteps = range(0,4*61,60)

# # timesteps = range(0,5)


# timesteps = range(0,7,2)
# scenario_nums = visualization.box_model_dsds__regimes(
#     ensemble_dir,timesteps,
#     regime_type='regions',avg_scale='lin', units='min',
#     lnDs = np.log(np.logspace(-10,-5,100)),backward=False,
#     fig_width=7.5,fig_height = 5., save_fig=True,sharey=False,
#     dpi=500,normalize=True,add_text=False,recompute_KL=True)

# scenario_nums = [81, 6, 89, 43, 18]
# # scenario_nums = [80, 62, 11]
# # timesteps = range(0,7)
# scenario_nums = [6,43,14]
# timesteps = range(0,7,2)
# visualization.box_model_dsds__few(
#         ensemble_dir,timesteps,scenario_nums,sharey=False,
#         lnDs = np.log(np.logspace(-10,-5,100)),backward=False,save_fig=True,
#         fig_width=7.5,fig_height = 4., units='min',
#         dpi=500,normalize=True,add_text=False)


# processes = ['both','cond','coag']
# labs = ['a','b','c']
# timesteps = range(0,7)
# visualization.KL_vs_time__few(
#         ensemble_prefix,scenario_nums,timesteps,
#         processes = ['both'],labs=['a'],
#         ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
#         backward=False,recompute_KL=True,save_fig=True)

# # visualization.KL_vs_time(
# #         ensemble_prefix,scenario_nums,processes,labs,timesteps,
# #         ensemble_over_dir = '/Users/fier887/Downloads/box_simulations3/',
# #         backward=False,recompute_KL=True)
# # timesteps = range(3)#[0,1,2]
# visualization.box_model_dsds__all_scenarios(
#         ensemble_dir,timesteps,
#         recompute_KL=True,
#         lnDs = np.log(np.logspace(-10,-5,100)),backward=False,
#         save_fig=True,dpi=200,normalize=True)

