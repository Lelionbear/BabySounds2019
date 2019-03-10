#!/usr/bin/python
# Tested with python 2.7

import os
import pandas as pd

# Modify openSMILE paths HERE:
SMILEpath = '/tools/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
SMILEconf = '/tools/opensmile-2.3.0/config/ComParE_2016.conf'

# Task name
task_name = os.getcwd().split('/')[-2]  # 'ComParE2019_XXX'

# Paths
audio_folder    = '../wav/'
labels_file     = '../lab/labels.csv'
features_folder = '../features/'

# Define partition names (according to audio files)
partitions = ['train','devel','test']

# Load file list
instances = pd.read_csv(labels_file)['file_name']

# Iterate through partitions and extract features
for part in partitions:
    instances_part = instances[instances.str.startswith(part)]
    output_file      = features_folder + task_name + '.ComParE.'      + part + '.csv'
    output_file_lld  = features_folder + task_name + '.ComParE-LLD.'  + part + '.csv'
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(output_file_lld):
        os.remove(output_file_lld)
    # Extract openSMILE features for the whole partition (standard ComParE and LLD-only)
    for inst in instances_part:
        os.system(SMILEpath + ' -C ' + SMILEconf + ' -I ' + audio_folder + inst + ' -instname ' + inst + ' -csvoutput '+ output_file + ' -timestampcsv 0 -lldcsvoutput ' + output_file_lld + ' -appendcsvlld 1')
    # Compute BoAW representations from openSMILE LLDs
    num_assignments = 10
    for csize in [125,250,500,1000,2000]:
        output_file_boaw = features_folder + task_name + '.BoAW-' + str(csize) + '.' + part + '.csv'
        xbow_config = '-i ' + output_file_lld + ' -attributes nt1[65]2[65] -o ' + output_file_boaw
        if part=='train':
            xbow_config += ' -standardizeInput -size ' + str(csize) + ' -a ' + str(num_assignments) + ' -log -B codebook_' + str(csize)
        else:
            xbow_config += ' -b codebook_' + str(csize)
        os.system('java -Xmx12000m -jar openXBOW.jar -writeName ' + xbow_config)
