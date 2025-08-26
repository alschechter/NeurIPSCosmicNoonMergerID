# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Code to go through and grab environmental
# information from completed tables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import pandas as pd
import illustris_python as il
import numpy as np
import matplotlib
import random
matplotlib.use('agg')# must call this before importing pyplot as plt: 
import matplotlib.pyplot as plt
import seaborn as sns
import util_mergers as u    

# first, get a list of whoch snapshots exist with matched nonmergers:
snapnum_cen_list = [17,25,33,40,50,59,67,72,84]

    
snaplist = np.arange(0,100)
#snaplist = snapnum_cen_list
run_snap = [13, 14, 15, 16, 17, 18, 19]
#run_snap = [49, 50, 51]

print('checking all of these', snaplist)
#snaplist = os.listdir('Tables/Aimee_SF/')
version = 'no_enviro'

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

if run_snap: 
    print('already have run_snap', run_snap)
else:
    run_snap = []
    for snapnum in snaplist:
        snapnum_center = find_nearest(snapnum_cen_list, snapnum)
        
        try:
            selection = pd.read_csv('Tables/Aimee_SF/'+str(snapnum_center)+'/all_mergers_at_'+str(snapnum)+'_enviro.txt', sep = '\t')
            selection = pd.read_csv('Tables/Aimee_SF/'+str(snapnum_center)+'/nonmergers_matched_at_'+str(snapnum)+'_'+str(version)+'.txt', sep = '\t')
                    
            print('            NONMERGERS ALREADY READY TO GO                 ', snapnum)
            run_snap.append(snapnum)
        except:
            continue
    print('run these', run_snap)

# there are two different ways to do comparisons
# one is just the bin center


# Now for each merger table compare to the pool that the non-mergers are drawn from:
basePath = '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L75n1820TNG'#'L35n2160TNG' 
sP = basePath+run+'/output/'

years_ago = 0.01

h = 0.6774
minmass = 10**3*(5.7*10**4)/h # 1000 times the mass resolution
ll = minmass
maxmass = 10**14

    
for snapnum in run_snap:
    snapnum_center = find_nearest(snapnum_cen_list, snapnum)
    mergers = pd.read_csv('Tables/Aimee_SF/'+str(snapnum_center)+'/all_mergers_at_'+str(snapnum)+'_enviro.txt', sep = '\t')
    merg_i = mergers['Subfind_ID'].values.astype(int)
    
    nonmergers = pd.read_csv('Tables/Aimee_SF/'+str(snapnum_center)+'/nonmergers_matched_at_'+str(snapnum)+'_'+version+'.txt', sep = '\t')
    nonmerg_i = nonmergers['SubhaloID'].values.astype(int)
    
    fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']#was \['SubhaloMassInRad']    'SubhaloFlag',                                                                 
    
    subhalos = il.groupcat.loadSubhalos(basePath+run+'/output/',snapnum, fields=fields)
    mass_msun = subhalos['SubhaloMassType'][:,4] * 1e10 / h
    sfr = subhalos['SubhaloSFRinRad']
    
    # This selects all subhalos at this snapnum that are within our mass range
    select_indices = np.where(np.logical_and(mass_msun > minmass, mass_msun < maxmass, sfr > 0.0))[0]
    select_mass = mass_msun[select_indices]
    select_sfr = sfr[select_indices]
 
    pos_m_x = []
    pos_m_y = []
    pos_m_z = []
    for p in merg_i:
        # eliminate subhalos that have had a merger in the last 2 Gyr
        
            
        
        # first, get the location of this one subhalo
        
        if mass_msun[p] < ll or sfr[p]==0.0:
            
            
            continue
        pos_xyz = subhalos['SubhaloCM'][p]
        pos_m_x.append(pos_xyz[0])
        pos_m_y.append(pos_xyz[1])
        pos_m_z.append(pos_xyz[2])
        
    print('length of mergers', len(merg_i), 'length of position', len(pos_m_x))

    pos_n_x = []
    pos_n_y = []
    pos_n_z = []
    for p in nonmerg_i:
        # eliminate subhalos that have had a merger in the last 2 Gyr

        # first, get the location of this one subhalo
        
        if mass_msun[p] < ll or sfr[p]==0.0:
            
            continue
        pos_xyz = subhalos['SubhaloCM'][p]
        pos_n_x.append(pos_xyz[0])
        pos_n_y.append(pos_xyz[1])
        pos_n_z.append(pos_xyz[2])
        
    print('length of nonmergers', len(nonmerg_i), 'length of position', len(pos_n_x))
    
    print(np.shape(pos_n_x))
    
    plt.clf()
    plt.scatter(pos_n_x,pos_n_y, label='Non-mergers', color='#F2542D')
    plt.scatter(pos_m_x,pos_m_y, label='Mergers', color='#0E9594')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('Figs/Aimee_SF/3D_x_y_'+str(snapnum)+'_'+str(version)+'.png')
    
    plt.clf()
    plt.scatter(pos_n_x,pos_n_z, label='Non-mergers', color='#F2542D')
    plt.scatter(pos_m_x,pos_m_z, label='Mergers', color='#0E9594')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend()
    plt.savefig('Figs/Aimee_SF/3D_x_z_'+str(snapnum)+'_'+str(version)+'.png')
    






    
        
