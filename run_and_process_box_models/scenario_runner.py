#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions needed to run a PartMC-MOSAIC scenario

@author: Laura Fierce
"""

import scenario_maker
import os
import numpy as np

def go(scenario_name,settings,scenario_type='4mode_noemit',run_time_str='0:00:30',uniform_comp=False, project='nuvola'):
    print_scenario_here = get_store_over_dir() + 'box_simulations3/' + scenario_name + '/'
    if not os.path.exists(print_scenario_here):
        os.mkdir(print_scenario_here)
    runfiles_dir = print_scenario_here + 'runfiles/'
    run_filenames = splitRepats(print_scenario_here, settings, scenario_type=scenario_type, uniform_comp=uniform_comp)
    for run_filename in run_filenames:
        send_run_to_slurm(run_filename, runfiles_dir, run_time_str=run_time_str, project=project)

def get_run_time_str(run_time_min):
    hrs = run_time_min/60.
    hrs_floor = int(np.floor(hrs))
    min_left = int(run_time_min - hrs_floor*60.)
    run_time_str = str(hrs_floor) + ':' + str(min_left) + ':00'
    return run_time_str

def splitRepats(print_scenario_here,settings,scenario_type='4mode_noemit',uniform_comp=False):
    inputs, default_inputs, spec_filenames, repeat_ids = scenario_maker.create_splitRepeats(
        print_scenario_here,settings,scenario_type=scenario_type,uniform_comp=uniform_comp)
    run_filenames = []
    for ii,(repeat_id,spec_filename) in enumerate(zip(repeat_ids,spec_filenames)):
        run_filename = print_scenario_here + 'run_' + repeat_id + '.sh'
        make_run_file(spec_filename,run_filename)
        run_filenames.append(run_filename)
    return run_filenames

def make_run_file(spec_filename,run_filename):
    os.system('chmod +rwx ' + spec_filename)
    partmc_driver_path = get_partmc_driver_path()
    run_file_lines = [
        '#!/bin/sh',
        '',
        '# exit on error', 
        'set -e',
        '# turn on command echoing',
        'set -v',
        ' ',
        partmc_driver_path + ' ' + spec_filename]
    
    with open(run_filename, 'w') as f:
        for line in run_file_lines:
            f.write(line)
            f.write('\n')
    f.close()
    os.system('chmod +x ' + run_filename)
    
def create_python_driver(run_filename,runfiles_dir='runfiles/'):
    if not os.path.exists(runfiles_dir):
        os.mkdir(runfiles_dir)
    scenario_num, repeat_num, run_path = get_scenario_id(run_filename)
    python_dir = get_python_dir()
    if type(scenario_num) == str:
        python_filename = runfiles_dir + 'main_run_' + scenario_num + '_' + str(repeat_num).zfill(4) + '.py'
    else:
        python_filename = runfiles_dir + 'main_run_' + str(scenario_num).zfill(6) + '_' + str(repeat_num).zfill(4) + '.py'
    python_lines = [
        'import sys',
        # 'import os',
        # 'cwd = os.getcwd()',
        # '',
        # 'os.chdir(' + '\'' + python_dir + '\'' +')',
        'sys.path.insert(0, ' + '\'' + python_dir + '\')',
        'import scenario_runner',
        'run_filename = ' + '\'' + run_filename + '\'',
        'scenario_runner.run_scenario(run_filename)']
    with open(python_filename, 'w') as f:
        for line in python_lines:
            f.write(line)
            f.write('\n')
    f.close()
    
    return python_filename
    
def send_run_to_slurm(run_filename, runfiles_dir, run_time_str='0:00:30',qos='flex', project='nuvola'):
    scenario_num, repeat_num, run_path = get_scenario_id(run_filename)
    if type(scenario_num) == str:
        run_id = scenario_num + '_' + str(repeat_num).zfill(4)
    else:
        run_id = str(scenario_num).zfill(6) + '_' + str(repeat_num).zfill(4)
    sh_filename = runfiles_dir + run_id + '.sh'
    if os.path.exists('/people/fier887/'):
        sh_lines = [
            '#!/bin/tcsh',
            '#SBATCH -A ' + project,
            '#SBATCH -p shared',
            '#SBATCH -t ' + run_time_str,
            '#SBATCH -N 1',
            '#SBATCH -o ' + runfiles_dir + run_id + '.out',
            '#SBATCH -e ' + runfiles_dir + run_id + '.err',
            '']
    elif os.path.exists('/global/cscratch1/sd/lfierce'):
        # os.path.exists('/global/homes/l/lfierce'):
        if qos == 'flex':
            sh_lines = [
                '#!/bin/bash',
                '#SBATCH --job-name=' + run_id,
                '#SBATCH --qos=flex',
                '#SBATCH --constraint=knl',
                '#SBATCH --account=m3525',
                '#SBATCH --nodes=1',
                '#SBATCH --time=02:00:00',
                '#SBATCH --time-min=' + run_time_str,
                '#SBATCH --error=' + runfiles_dir + run_id + '.err',
                '#SBATCH --output=' + runfiles_dir + run_id + '.out',
                '']
        else:
            sh_lines = [
                '#!/bin/bash',
                '#SBATCH --job-name=' + run_id,
                '#SBATCH --qos=' + qos,
                '#SBATCH --constraint=knl',
                '#SBATCH --account=m3525',
                '#SBATCH --nodes=1',
                '#SBATCH --time=' + run_time_str,
                '#SBATCH --error=' + runfiles_dir + run_id + '.err',
                '#SBATCH --output=' + runfiles_dir + run_id + '.out',
                '']
    elif os.path.exists('/pscratch/sd/l/lfierce/'):
        # os.path.exists('/global/homes/l/lfierce'):
        sh_lines = [
            '#!/bin/bash',
            '#SBATCH -A m3525',
            '#SBATCH -C cpu',
            '#SBATCH -q regular',
            '#SBATCH -t ' + run_time_str,
            '#SBATCH -e ' + runfiles_dir + run_id + '.err',
            '#SBATCH -o ' + runfiles_dir + run_id + '.out',
            '']
    else:
        sh_lines = []
    
    python_filename = create_python_driver(run_filename,runfiles_dir=runfiles_dir)
    
    if os.path.exists('/people/fier887/'):
        sh_lines.append('module load gcc/5.2.0')
    elif os.path.exists('/global/homes/l/lfierce'):
        sh_lines.append('module load gcc/7.3.0')
    
    sh_lines.append('')
    sh_lines.append('python ' + python_filename)
    
    with open(sh_filename, 'w') as f:
        for line in sh_lines:
            f.write(line)
            f.write('\n')
    
    os.system('chmod +x ' + sh_filename)
    os.system('sbatch ' + sh_filename)
    return sh_filename 
    
def run_scenario(run_filename):
    scenario_num, repeat_num, run_path = get_scenario_id(run_filename)
    if type(scenario_num) == str:
        state_filename = run_path + scenario_num + '/state_' + str(repeat_num).zfill(4) + '.txt'
        repeat_path = run_path + scenario_num + '/repeat' + str(repeat_num).zfill(4) + '/'
    else:
        state_filename = run_path + str(scenario_num).zfill(6) + '/state_' + str(repeat_num).zfill(4) + '.txt'
        repeat_path = run_path + str(scenario_num).zfill(6) + '/repeat' + str(repeat_num).zfill(4) + '/'
    
    state = get_state(state_filename)
    
    if state == 0 or state == 1:
        cwd = os.curdir
        os.chdir(repeat_path)
        os.system(run_filename)
        os.chdir(cwd)
        update_state(state_filename,2)
        
def get_state(state_filename):
    # 0: not yet started
    # 1: started but not finished
    # 2: finished
    with open(state_filename, 'r') as f:
        state_str = f.read()
    f.close()
    try:
        state = int(state_str)
    except:
        state = int(float(state_str[:-1]))
    return state
    
def update_state(state_filename,state):
    with open(state_filename, 'w') as f:
        f.write(str(state))
    f.close()
    
def get_partmc_driver_path():
    if os.path.exists('/people/fier887/'):
        partmc_driver_path = '/people/fier887/partmc-2.6.1/build/partmc'
    elif os.path.exists('/global/homes/l/lfierce'):
        partmc_driver_path = '/global/homes/l/lfierce/partmc_mosaic/partmc/build/partmc'
    elif os.path.exists('/Users/fier887/'):
        partmc_driver_path = '/Users/fier887/Library/CloudStorage/OneDrive-PNNL/Documents/partmc-mosiac/partmc-2.6.0/build/partmc'    
    return partmc_driver_path

def get_store_over_dir():
    if os.path.exists('/people/fier887/'):
        store_over_dir = '/pic/projects/sooty2/fierce/'
    elif os.path.exists('/pscratch/sd/l/lfierce/'):
        store_over_dir = '/pscratch/sd/l/lfierce/'
    elif os.path.exists('/global/homes/l/lfierce'):
        store_over_dir = '/global/cscratch1/sd/lfierce/'
    elif os.path.exists('/Users/fier887/'):
        store_over_dir = '/Users/fier887/Library/CloudStorage/OneDrive-PNNL/Documents/EAGLES/'
    return store_over_dir

def get_python_dir():
    if os.path.exists('/people/fier887/'):
        python_dir = '/people/fier887/run_partmc/'
    elif os.path.exists('/global/homes/l/lfierce'):
        python_dir = '/global/homes/l/lfierce/run_partmc/'
    elif os.path.exists('/Users/fier887/'):
        python_dir = '/Users/fier887/Library/CloudStorage/OneDrive-PNNL/Documents/EAGLES/run_partmc/'
    return python_dir

def get_scenario_id(run_filename):
    idx = run_filename.rfind('/')
    idx_start = idx + 1 + run_filename[(idx+1):].rfind('_') + 1
    idx_end = idx + 1 + run_filename[(idx+1):].rfind('.')
    repeat_num = int(run_filename[idx_start:idx_end])
    idx2 = run_filename[:(idx-1)].rfind('/')
    try:
        scenario_num = int(run_filename[(idx2+1):(idx)])
    except:
        idx3 = run_filename[(idx2+1):].rfind('/')
        scenario_num = run_filename[(idx2+1):(idx2 + idx3 + 1)]
    
    run_path = run_filename[:(idx2+1)]
    return scenario_num, repeat_num, run_path
