import illustris_python as il
import h5py
import numpy as np
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt:                                                                                    
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
import os
import util_mergers as u
import pandas as pd

snapnum_list = [17,25,33,40,50,59,67,72,84]

for bin in snapnum_list:
    #bin = 17
    
    mergers = pd.read_csv('Tables/Aimee_SF/'+str(bin)+'/all_mergers.txt', sep='\t')
    
    
    # Get all snap centers:
    
    MIN = np.min(mergers['mass'].values)#10**7.5
    MAX = np.max(mergers['mass'].values)#10**12
    
    colors = ['#D72638','#3F88C5','#F49D37','#140F2D','#F22B29','#8CFBDE','#F3FFB6']
    
    count = 0
    plt.clf()
    for snap in mergers['center'].unique():
        
        print('making a hist out of these', mergers[mergers['center']==snap]['mass'].values)
    
        plt.hist(mergers[mergers['center']==snap]['mass'].values, label=str(snap), 
            bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 25),
            alpha = 0.5, color=colors[count])
        count+=1
    #plt.hist(mergers[mergers['center']=='13']['mass'].values, label='14')
    #plt.hist(mergers[mergers['center']=='13']['mass'].values, label='15')
    #plt.hist(mergers[mergers['center']=='13']['mass'].values, label='16')
    #plt.hist(mergers[mergers['center']=='13']['mass'].values, label='16')
    plt.legend()
    plt.gca().set_xscale("log")
    plt.xlabel('Stellar Mass')
    plt.title('Bin center = '+str(bin)+', total # mergers = '+str(len(mergers)))
    
    plt.savefig('Figs/Aimee_SF/hist_mass_'+str(bin)+'.png')

            


