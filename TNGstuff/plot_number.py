import pandas as pd
import illustris_python as il
import numpy as np
import matplotlib
import random
matplotlib.use('agg')# must call this before importing pyplot as plt: 
import matplotlib.pyplot as plt
import seaborn as sns
import util_mergers as u 
import os

basePath = '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L75n1820TNG'#'L35n2160TNG'                                                                                                               
sP = basePath+run+'/output/'
nonmerg_name = 'no_enviro'

table_path = '/n/home03/rnevin/TNGstuff/Tables/Aimee_SF/'

redshift_bins = np.sort([int(float(x)) for x in os.listdir(table_path)])

'''
lbt_dict, redshift_dict = u.lookup_dicts()
print(redshift_dict)
print(redshift_dict[str(17)])
'''

print('redshift bins', redshift_bins)


num_mergers = []
num_mergers_center = []



palette = ['#C9CBA3','#FFE1A8','#E26D5C','#723D46','#472D30',
    '#C9CBA3','#FFE1A8','#E26D5C','#723D46','#472D30',
    '#C9CBA3','#FFE1A8','#E26D5C','#723D46','#472D30']
count_color = 0

plt.clf()

for redshift in redshift_bins:
    
    nonmergers_mass = []
    mergers_mass = []

    nonmergers_sfr = []
    mergers_sfr = []
    
    
    
    print('redshift', redshift)
    counter_mergers = 0
    counter_mergers_center = 0
    counter_nonmergers = 0
    list_in_bin = os.listdir(table_path+'/'+str(redshift))
    mergers_all = []
    for file in list_in_bin:
        '''
        if 'all_mergers_in_bin' in file:
            #print('all mergers')
            selection = pd.read_csv(table_path+'/'+str(redshift)+'/'+file, sep = '\t')
            len_all = len(selection)
        '''   
        # just grab the bin center:
        if file[0:14] == 'all_mergers_at' and 'enviro' in file and str(redshift) in file:
            #print(file)
            selection = pd.read_csv(table_path+'/'+str(redshift)+'/'+file, sep = '\t')
            #print(len(selection))
            counter_mergers_center+=len(selection)
        
        if file[0:14] == 'all_mergers_at' and 'enviro' in file:
            #print(file)
            selection = pd.read_csv(table_path+'/'+str(redshift)+'/'+file, sep = '\t')
            #print(len(selection))
            counter_mergers+=len(selection)
            for j in range(len(selection)):
                mergers_mass.append(np.log10(selection['mass'].values[j]))
                mergers_sfr.append(selection['SFR'].values[j])
            
        if file[0:10] == 'nonmergers' and nonmerg_name in file:
            #print(file)
            selection = pd.read_csv(table_path+'/'+str(redshift)+'/'+file, sep = '\t')
            #print(len(selection))
            counter_nonmergers+=len(selection)
            for j in range(len(selection)):
                nonmergers_mass.append(np.log10(selection['Mass'].values[j]))
                nonmergers_sfr.append(selection['SFR'].values[j])
    num_mergers.append(counter_mergers)
    num_mergers_center.append(counter_mergers_center)
    
    print('length of mergers added up', counter_mergers)
    print('length of all nonmergers', counter_nonmergers)
    
    print('color count', count_color)
    print('color', palette[count_color])
    
    print(mergers_mass)
    
    plt.scatter(mergers_mass, mergers_sfr, 
        label='snap '+str(redshift),
        color = palette[count_color])
        
    plt.scatter(nonmergers_mass, nonmergers_sfr, marker = '+', 
        color = palette[count_color])#, marker = '^')
        
    count_color+=1
    
plt.xlabel('stellar mass')
plt.ylabel('log SFR')
plt.yscale('log')
plt.legend()
plt.savefig('Figs/sfr_mass_all.png')


plt.clf()
plt.scatter(redshift_bins, num_mergers, label = 'All snaps in bin', color='#4ECDC4')
plt.scatter(redshift_bins, num_mergers_center, label = 'Full snaps', color='#FF6B6B')
plt.xlabel('Snapnum')
plt.ylabel('# of mergers and # of non-mergers')
plt.legend()
for xy in zip(redshift_bins, num_mergers):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data', color='#4ECDC4')
for xy in zip(redshift_bins, num_mergers_center):                                       # <--
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data', color='#FF6B6B') # <--

plt.grid()
plt.savefig('Figs/num.png')


