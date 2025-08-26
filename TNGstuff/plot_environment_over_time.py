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
run_snap = [49, 50, 51]

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

variable_list = ['r_1','N','mass']


for variable in variable_list:

    plt.clf()
    
    for snapnum in run_snap:
        snapnum_center = find_nearest(snapnum_cen_list, snapnum)
        mergers = pd.read_csv('Tables/Aimee_SF/'+str(snapnum_center)+'/all_mergers_at_'+str(snapnum)+'_enviro.txt', sep = '\t')
        nonmergers = pd.read_csv('Tables/Aimee_SF/'+str(snapnum_center)+'/nonmergers_matched_at_'+str(snapnum)+'_'+str(version)+'.txt', sep = '\t')
        
        if variable == 'mass':
            plt.scatter(float(snapnum), np.max(mergers[variable].values), color = '#DC493A', marker='^', alpha=0.5)
            plt.scatter(float(snapnum), np.min(mergers[variable].values), color = '#DC493A', marker='^', alpha=0.5)
            
            plt.scatter(float(snapnum), np.min(nonmergers['Mass'].values), color = '#3B413C', marker='^', alpha=0.5)
            plt.scatter(float(snapnum), np.max(nonmergers['Mass'].values), color = '#3B413C', marker='^', alpha=0.5)
            
            plt.scatter(float(snapnum), np.mean(mergers[variable].values), color = '#DC493A')
            plt.scatter(float(snapnum), np.mean(nonmergers['Mass'].values), color = '#3B413C')
        else:
            plt.scatter(float(snapnum), np.max(mergers[variable].values), color = '#DC493A', marker='^', alpha=0.5)
            plt.scatter(float(snapnum), np.min(mergers[variable].values), color = '#DC493A', marker='^', alpha=0.5)
            
            plt.scatter(float(snapnum), np.min(nonmergers[variable].values), color = '#3B413C', marker='^', alpha=0.5)
            plt.scatter(float(snapnum), np.max(nonmergers[variable].values), color = '#3B413C', marker='^', alpha=0.5)
            
            plt.scatter(float(snapnum), np.mean(mergers[variable].values), color = '#DC493A')
            plt.scatter(float(snapnum), np.mean(nonmergers[variable].values), color = '#3B413C')
    plt.xlabel('Snap')
    plt.ylabel(variable)
    #if variable =='N':
    #    plt.ylim([0,2000])
    #if variable =='r_1':
    #    plt.ylim([0,1250])
    plt.savefig('Figs/Aimee_SF/'+variable+'_progression_time_'+str(version)+'.png')
    
    # Can we also please plot the histograms of the distributions??
    merger_list = []
    nonmerger_list = []
    all_list = []
    for snapnum in run_snap:
        snapnum_center = find_nearest(snapnum_cen_list, snapnum)
        mergers = pd.read_csv('Tables/Aimee_SF/'+str(snapnum_center)+'/all_mergers_at_'+str(snapnum)+'_enviro.txt', sep = '\t')
        nonmergers = pd.read_csv('Tables/Aimee_SF/'+str(snapnum_center)+'/nonmergers_matched_at_'+str(snapnum)+'_'+str(version)+'.txt', sep = '\t')
        merger_list.append(mergers)
        nonmerger_list.append(nonmergers)
        all_list.append(mergers)
        all_list.append(nonmergers)
        
    all_mergers = pd.concat(merger_list, axis=0)
    # Reset index
    all_mergers = all_mergers.reset_index()
    
    all_nonmergers = pd.concat(nonmerger_list, axis=0)
    # Reset index
    all_nonmergers = all_nonmergers.reset_index()
    
    all_gals = pd.concat(all_list, axis=0)
    # Reset index
    all_gals = all_gals.reset_index()
    
    print('making a histogram out of these', variable)
    print('min', np.min(all_gals[variable].values), 'max', np.max(all_gals[variable].values))
    if variable == 'mass':
        vals, bins = np.histogram(all_nonmergers['Mass'].values, bins=50)
    else:
        vals, bins = np.histogram(all_gals[variable].values, bins=50)
    plt.clf()
    
    if variable == 'mass':
        plt.hist(all_mergers[variable].values, bins=np.logspace(np.log10(10**7),np.log10(10**12), 50), 
            color = '#F2542D', label='Mergers', alpha=0.7)
        plt.hist(all_nonmergers['Mass'].values, bins=np.logspace(np.log10(10**7),np.log10(10**12), 50), 
            color='#F5DFBB', label='Non-mergers', alpha=0.7)
        plt.xscale('log')
        # bins=np.logspace(np.log10(0.1),np.log10(1.0), 50)
    else:
        plt.hist(all_mergers[variable].values, bins=bins, color = '#F2542D', label='Mergers', alpha=0.7)
        plt.hist(all_nonmergers[variable].values, bins=bins, color='#F5DFBB', label='Non-mergers', alpha=0.7)
    plt.legend()
    plt.xlabel(variable)
    plt.savefig('Figs/Aimee_SF/'+variable+'_histogram_'+str(version)+'.png')
    
    
        
    
    
STOP

# Now for each merger table compare to the pool that the non-mergers are drawn from:
basePath = '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L75n1820TNG'#'L35n2160TNG' 
sP = basePath+run+'/output/'

years_ago = 0.01

h = 0.6774
minmass = 10**3*(5.7*10**4)/h # 1000 times the mass resolution
ll = minmass
maxmass = 10**14

for variable in variable_list:

    plt.clf()
    
    for snapnum in run_snap:
        snapnum_center = find_nearest(snapnum_cen_list, snapnum)
        mergers = pd.read_csv('Tables/Aimee_SF/'+str(snapnum_center)+'/all_mergers_at_'+str(snapnum)+'_enviro.txt', sep = '\t')
        merg_i = mergers['Subfind_ID'].values.astype(int)
        
        fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']#was \['SubhaloMassInRad']    'SubhaloFlag',                                                                 
        
        subhalos = il.groupcat.loadSubhalos(basePath+run+'/output/',snapnum, fields=fields)
        mass_msun = subhalos['SubhaloMassType'][:,4] * 1e10 / h
        sfr = subhalos['SubhaloSFRinRad']
        
        # This selects all subhalos at this snapnum that are within our mass range
        select_indices = np.where(np.logical_and(mass_msun > minmass, mass_msun < maxmass, sfr > 0.0))[0]
        select_mass = mass_msun[select_indices]
        select_sfr = sfr[select_indices]
     
        fields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                          'FirstProgenitorID', 'SubhaloMassType','DescendantID','SubfindID']
        # Get N, r_1, and r_2 for all of the non-mergers in the mass_select_all
        # starting with subhalos
        N_all = []
        r_1_all = []
        r_2_all = []
    
        had_a_merger_in_last_2Gyr = []
        
         # Okay the first step is to get the environmental information for the mergers
        lbt_dict, redshift_dict = u.lookup_dicts()
        # get the lbt for this snapshot
        lbt_snap = lbt_dict[str(snapnum)]
        
        include_these_snaps = []
        
        for w in range(10):
            y = w + 1
            try:
                if lbt_dict[str(snapnum - y)] - lbt_snap > years_ago:
                    
                    break
                else:
                    print('lbt', lbt_snap, 'snap', snapnum)
                    print('snap', snapnum-y)
                    print('lbt', lbt_dict[str(snapnum-y)])
                    include_these_snaps.append(snapnum - y)
            except:
                # This just means that the it doesnt exist in the dict
                #break
                print('does not exist in the dictionary', 'snap', str(snapnum - y))
                continue
            
            
        # This is the number of snapshot steps to include to go the full 2 Gyr back in time
        print('years ago', years_ago)
        print('include these snaps', include_these_snaps)
        try:
            n_steps = snapnum - min(include_these_snaps)
        except ValueError: #
            n_steps = 1
            #STOP
        
        print('number of steps', n_steps)
    
    
        minMR = 1.0/10.0
    
        for p in select_indices:
            # eliminate subhalos that have had a merger in the last 2 Gyr
            try:
                #print('sP', sP, 'snap', snap, 'p', p)
                tree = il.sublink.loadTree(sP,snapnum, p, fields = fields)#, onlyMPB=False)#       
             
                n_merg_2Gyr = u.num_mergers_n_step(sP, tree, n_steps, minMR, massPartType='stars', index=1)
            except AttributeError:
                print('sP', sP, 'snap', snapnum, 'p', p)
                print('tree', tree)
                r_1_all.append(0)
                r_2_all.append(0)
                N_all.append(0)
                continue
                STOP
            had_a_merger_in_last_2Gyr.append(np.sum(n_merg_2Gyr))
    
            # first, get the location of this one subhalo
            pos = subhalos['SubhaloCM'][p]
            if mass_msun[p] < ll or sfr[p]==0.0:
                r_1_all.append(0)
                r_2_all.append(0)
                N_all.append(0)
                continue
            
            # Construct a distance to absolutely every other subhalo:
            
            rel_distance = []
            for w in range(len(subhalos)):
                #print(p, u, mass_msun[p], mass_msun[u])
                if w==p:
                    continue
                if mass_msun[w] < 0.1*mass_msun[p]:
                    continue
                #print('massive enough to pass through initially', p, u)
                #print(mass_msun[p], mass_msun[u], mass_msun[p]/mass_msun[u])
                pos_sub = subhalos['SubhaloCM'][w]
                dist = np.sqrt((pos[0] - pos_sub[0])**2 + (pos[1] - pos_sub[1])**2 + (pos[2] - pos_sub[2])**2)*h #in kpc
                rel_distance.append(dist)
            # Quantify how many are less than 2000 kpc away
            # print(rel_distance)
            #print(np.where(np.array(rel_distance) < 2000), len(np.where(np.array(rel_distance)<2000)[0]))
            
            sorted_d = np.sort(rel_distance)
            try:
                r_1_all.append(sorted_d[0])
            except:
                r_1_all.append(0)
            try:
                r_2_all.append(sorted_d[1])
            except:
                r_2_all.append(0)
            N_all.append(len(np.where(np.array(rel_distance)<2000)[0]))
        
        
    
    
        # Make a dataframe out of all of these galaxies:
        df_all = pd.DataFrame(list(zip(select_indices, select_mass, select_sfr, had_a_merger_in_last_2Gyr, N_all, r_1_all, r_2_all)),
               columns =['SubhaloID', 'Mass','SFR','num_mergers_in_last_2Gyr','N','r_1','r_2'])
        print(df_all)
        
        # Delete everything that has a SFR of 0.0!
        df_all = df_all[df_all['SFR'] > 0.0]
        print('length after deleting gals that have SFR = 0.0', len(df_all))
        
        # I think the first thing to do is eliminate all of the merger indices from here
        df_no_merg = df_all[~df_all.SubhaloID.isin(merg_i)]
        print('length after deleting mergers', len(df_no_merg))
        
        # Now delete anything that has merged in the last 2 Gyr
        new_df_full = df_no_merg[df_no_merg['num_mergers_in_last_2Gyr'] == 0.0]
        nonmergers = new_df_full
        
        print('new nonmergers table')
        print(nonmergers)
        
        if variable == 'mass':
            plt.scatter(float(snapnum), np.max(mergers[variable].values), color = '#DC493A', marker='^', alpha=0.5)
            plt.scatter(float(snapnum), np.min(mergers[variable].values), color = '#DC493A', marker='^', alpha=0.5)
            
            plt.scatter(float(snapnum), np.min(nonmergers['Mass'].values), color = '#3B413C', marker='^', alpha=0.5)
            plt.scatter(float(snapnum), np.max(nonmergers['Mass'].values), color = '#3B413C', marker='^', alpha=0.5)
            
            plt.scatter(float(snapnum), np.mean(mergers[variable].values), color = '#DC493A')
            plt.scatter(float(snapnum), np.mean(nonmergers['Mass'].values), color = '#3B413C')
        else:
            plt.scatter(float(snapnum), np.max(mergers[variable].values), color = '#DC493A', marker='^', alpha=0.5)
            plt.scatter(float(snapnum), np.min(mergers[variable].values), color = '#DC493A', marker='^', alpha=0.5)
            
            plt.scatter(float(snapnum), np.min(nonmergers[variable].values), color = '#3B413C', marker='^', alpha=0.5)
            plt.scatter(float(snapnum), np.max(nonmergers[variable].values), color = '#3B413C', marker='^', alpha=0.5)
            
            plt.scatter(float(snapnum), np.mean(mergers[variable].values), color = '#DC493A')
            plt.scatter(float(snapnum), np.mean(nonmergers[variable].values), color = '#3B413C')
    plt.xlabel('Snap')
    plt.ylabel(variable)
    if variable =='N':
        plt.ylim([0,2000])
    plt.savefig('Figs/Aimee_SF/'+variable+'_progression_time_nonmerger_pool_notime.png')





    
        
