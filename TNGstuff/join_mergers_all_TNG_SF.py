# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the version for Aimee's TNG project:
# It makes a big table of all mergers that occur
# at all snapshots within 0.25 Gyr of the bin center
# snapshot. 

# Note that this precludes descendants in the snapshot
# that are in the snapshot with the largest look back time
# and precludes pre-mergers in the snapshot with
# the smallest look back time.

# Loop version! Goes through all redshifts :)
# Also recently added: Calculates the merger fraction
# at each redshift bin to compare to Vicente's calculations

# Accompanys the build_merger_catalog.py code
# Takes the output from this ^ and joins them together
# centered on the bin center
# into one table called all_mergers_in_bin.py
# Also builds a matched sample of nonmergers in a 
# separate table (matched for redshift and SFR and mass
# and number of nearest neighbors within 2 Mpc and the
# distance in kpc to the first and second nearest neighbor)
# [as in Bickley+2021 - https://ui.adsabs.harvard.edu/abs/2021MNRAS.504..372B/abstract]

# We should probably look into matching for gas fraction
# as well? This is a TO DO

# Now using Bobby's tolerance code to do matching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import illustris_python as il
import numpy as np
import matplotlib
import random
matplotlib.use('agg')# must call this before importing pyplot as plt: 
import matplotlib.pyplot as plt
import seaborn as sns
from util_mergers import *            

basePath = '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L75n1820TNG'#'L35n2160TNG'                                                                                                               
sP = basePath+run+'/output/'
version = 'ngroup_2Gyr_Bobby'
drop_repeats = False

# 17 --> z = 5.0
# 25 --> z = 3.0
# 33 --> z = 2.0
# 40 --> z = 1.5
# 50 --> z = 1.0
# 59 --> z = 0.7
# 67 --> z = 0.5
# 72 --> z = 0.4
# 78 --> z = 0.3
# 84 --> z = 0.2



# This time loop over all of the snapnums:

#snapnum_list = [17,25,33,40,50,59,67,72,78,84,99]
#snapnum_list = [84]
#snapnum_list = [50]
snapnum_list = [33]
for snapnum in snapnum_list:

    all_prog = False # This is an option to include the tables that have all possible progenitors (i.e., two snapshots before)
    
    subhalosh = il.groupcat.loadHeader(basePath+run+'/output/',snapnum)#, fields=fields_header)
    z = subhalosh['Redshift']
    print('redshift', z)
    a = subhalosh['Time']
    print('scale factor', a)

    
    redshift_label = z
    
    # First get a list of snapshots to include based on look back time:
    
    include_these_snaps = which_snaps_to_include(snapnum, 0.5, 0, 99)
    
    # Also maybe go to this folder and get a list of all files in there
    files = os.listdir('Tables/Aimee_SF/'+str(snapnum)+'/')
    
    # Sort by the snapshot at the center:
    center_list = []
    for file in files:
        print(file)
        
        try:
            center = int(str.split(str.split(file, '_at_')[1],'_L35n2160TNG.txt')[0])
        except:
            continue
        if center not in center_list:
            center_list.append(center)
        
    centers = np.sort(center_list)
    
    
    #if centers == list(include_these_snaps):
    test_list1 = centers
    test_list2 = list(include_these_snaps)
    if len(test_list1)== len(test_list2) and len(test_list1) == sum([1 for i, j in zip(test_list1, test_list2) if i == j]):
        print('lists are identical')
    else:
        print('missing something')
        print('include these', list(include_these_snaps))
        print('folder includes these', centers)
        STOP
    
    # Now, figure out how many files you're joining together here:
    centers_str = [str(x) for x in centers]
    #dict_centers = dict.fromkeys(centers_str,[])
    dict_centers = {k: [] for k in centers_str}
    print('dictionary', dict_centers)
    
    for file in files:
        # Now go through and assign all files
        try:
            center = str.split(str.split(file, '_at_')[1],'_L35n2160TNG.txt')[0]
            pulling_from = int(str.split(str.split(file, '_at_')[0],'mergers_snap_from_')[1])
            dict_centers[center].append(pulling_from)
        except:
            continue
    
    # Now we have a dictionary thats filled with everything you would use for smashing together the tables
    print(dict_centers)
    
    # The next step is to figure out how to combine these without repeats
    # So go through by centers and do this
    mergers_at_snap = []
    
    for snapc in centers_str:
        # First get the list of pullings from: 
        look_up = np.sort(dict_centers[snapc])
        print(look_up)
        print('combining this many files', len(dict_centers[snapc]))
        print('looking to open up this')
        list_of_tables = []
        for p in look_up:
            print(snapc, p)
            merger_list = 'Tables/Aimee_SF/'+str(snapnum)+'/mergers_snap_from_'+str(p)+'_at_'+str(snapc)+'_L35n2160TNG.txt'
            print('merger list', merger_list)
            mergers = pd.read_csv(merger_list, sep='\t')
            list_of_tables.append(mergers)
        
        all_mergers = pd.concat(list_of_tables, axis=0)
        # Reset index
        all_mergers = all_mergers.reset_index()
        print(all_mergers)
        
        # Instead of dropping duplicates in the merger table, save their integers:
        dict_type = {'first_progenitor':'1',
                    'next_progenitor':'10',
                    'Merger':'100',
                    'Descendant':'1000'
        }
        # If a galaxy has multiple IDs
        # Get rid of duplicates in the merger table (using Subfind ID)
        
        # Make a new row for all_mergers:
        all_mergers['int_type'] = [dict_type[x] for x in all_mergers['Type'].values]
        
        
        # Find the non-duplicated entries:
        df_not_dup = all_mergers[~all_mergers.duplicated(subset=['Subfind_ID'], keep=False)]
        print('length of non-duplicated', len(df_not_dup))
        
        
        df_dup_all = all_mergers[all_mergers.duplicated(subset=['Subfind_ID'], keep=False)]
        print('length of everything that is duplicated', len(df_dup_all), 'out of total length', len(all_mergers))

        #print('all mergers', all_mergers)        
        #Make a new row for the integers
        #df_dup_all['int_type'] = [dict_type[x] for x in df_dup_all['Type'].values]
        
        df_dup = all_mergers[all_mergers.duplicated(subset=['Subfind_ID'], keep='first')]#False
        print('length of kept IDs that are duplicates', len(df_dup))
        
        
        for o in range(len(df_dup)):
        
            find_value = df_dup['Subfind_ID'].values[o]
            index_value = df_dup['index'].values[o]
            # So go through all of the IDs that are repeats and get where they show up
            id_type = []
            
            for p in range(len(df_dup_all[df_dup_all['Subfind_ID']==find_value])):
                id_type.append(int(dict_type[str(df_dup_all[df_dup_all['Subfind_ID']==find_value]['Type'].values[p])]))
            df_dup_all.loc[df_dup_all['Subfind_ID']==find_value,'int_type'] = np.sum(id_type)
        
        
        # The final step is to get rid of everything that is a repeat
        df_no_dup = df_dup_all[~df_dup_all.duplicated(subset=['Subfind_ID'], keep='first')]
        
        df_no_dup_concat = pd.concat([df_no_dup, df_not_dup])
        df_no_dup_concat = df_no_dup_concat.reset_index()
        print(df_no_dup_concat)
        
        h = 0.6774
        minmass = 10**3*(5.7*10**4)/h # 1000 times the mass resolution
        ll = minmass
        maxmass = 10**14
        
        dropped_reps = df_no_dup_concat[df_no_dup_concat.mass > ll]# a tenth of the lower limit mass
        drop_cols = dropped_reps.drop(columns=['level_0','index'],axis=1)
        
        print('length after cutting low masses', len(dropped_reps))
        
        # Okay remove all of these weird columns
        drop_cols = drop_cols.reset_index()
        
        
        merg_i = drop_cols['Subfind_ID'].values.astype(int)
        
        # Okay now add a row that 
        
        drop_cols['center'] = snapc
        
        print(drop_cols)
        
        # Now save each as a new file and then as an item in the list to concat into a massive table
        drop_cols.to_csv('Tables/Aimee_SF/'+str(snapnum)+'/all_mergers_at_'+str(snapc)+'.txt', sep='\t')
        mergers_at_snap.append(drop_cols)
    
    all_mergers_at_bin = pd.concat(mergers_at_snap, axis=0)
    # Reset index
    all_mergers_at_bin = all_mergers_at_bin.reset_index()
    
    print(all_mergers_at_bin)
    
    all_mergers_at_bin.to_csv('Tables/Aimee_SF/'+str(snapnum)+'/all_mergers_in_bin.txt', sep='\t')
    
    continue
    
    STOP
    
    #
    STOP
    
    
    # Step 1: import all of the relevant merger tables:
    if snapnum > 25:
        center = pd.read_csv('Tables/mergers_snap_'+str(snapnum)+'_'+str(run)+'.txt', sep='\t')
        pre = pd.read_csv('Tables/mergers_snap_'+str(snapnum+1)+'_'+str(run)+'.txt', sep='\t')
        post = pd.read_csv('Tables/mergers_snap_'+str(snapnum-1)+'_'+str(run)+'.txt', sep='\t')
        print('center', len(center), 'pre', len(pre), 'post', len(post))
    
        all_mergers = pd.concat([center, pre, post], axis=0)
    if snapnum == 25:
        center = pd.read_csv('Tables/mergers_snap_'+str(snapnum)+'_'+str(run)+'.txt', sep='\t')
        pre = pd.read_csv('Tables/mergers_snap_'+str(snapnum+1)+'_'+str(run)+'.txt', sep='\t')
        if all_prog:
            pre2 = pd.read_csv('Tables/mergers_snap_all_prog_'+str(snapnum+2)+'_'+str(run)+'.txt', sep='\t')
        else:
            pre2 = pd.read_csv('Tables/mergers_snap_'+str(snapnum+2)+'_'+str(run)+'.txt', sep='\t')
        
        post = pd.read_csv('Tables/mergers_snap_'+str(snapnum-1)+'_'+str(run)+'.txt', sep='\t')
        print('center', len(center), 'pre', len(pre), 'post', len(post))
    
        all_mergers = pd.concat([center, pre,pre2, post], axis=0)
    if snapnum == 17:
        
        center = pd.read_csv('Tables/mergers_snap_'+str(snapnum)+'_'+str(run)+'.txt', sep='\t')
        pre = pd.read_csv('Tables/mergers_snap_'+str(snapnum+1)+'_'+str(run)+'.txt', sep='\t')
        if all_prog:
            pre2 = pd.read_csv('Tables/mergers_snap_all_prog_'+str(snapnum+2)+'_'+str(run)+'.txt', sep='\t')
        else:
            pre2 = pd.read_csv('Tables/mergers_snap_'+str(snapnum+2)+'_'+str(run)+'.txt', sep='\t')
        
        post = pd.read_csv('Tables/mergers_snap_'+str(snapnum-1)+'_'+str(run)+'.txt', sep='\t')
        post2 = pd.read_csv('Tables/mergers_snap_'+str(snapnum-2)+'_'+str(run)+'.txt', sep='\t')
        print('lengths', 'center', len(center), 'pre', len(pre), 'pre2', len(pre2), 'post', len(post),'post2', len(post2))
        
        
    
        all_mergers = pd.concat([center, pre,pre2, post,post2], axis=0)
        
    
    
    all_mergers = all_mergers.reset_index()
    print('length of mergers before you delete any repeats', len(all_mergers))
    
    # Now, get info about all subhalos at bin center 
    
    if run=='L75n1820TNG':
        fields = ['SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']
    else:
        fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']#was \['SubhaloMassInRad']                                                                     
    h = 0.6774
    minmass = 10**3*(5.7*10**4)/h # 1000 times the mass resolution
    ll = minmass
    maxmass = 10**14
    
    print('minimum mass selection', ll, 'max mass', maxmass)
    
    # Get all subhalos at this redshift
    subhalos = il.groupcat.loadSubhalos(basePath+run+'/output/',snapnum, fields=fields)
    mass_msun = subhalos['SubhaloMassType'][:,4] * 1e10 / h
    sfr = subhalos['SubhaloSFRinRad']
    
    # This selects all subhalos at this snapnum that are within our mass range
    select_indices = np.where(np.logical_and(mass_msun > minmass, mass_msun < maxmass, sfr > 0.0))[0]
    select_mass = mass_msun[select_indices]
    select_sfr = sfr[select_indices]
    
    
    '''
    
    print('make sure there arent any that have flags')
    for f in range(len(subhalos['SubhaloFlag'][select_indices])):
        if subhalos['SubhaloFlag'][select_indices][f]!=True:
            print(subhalos['SubhaloFlag'][select_indices][f])
            print('mass',select_mass[f])
            print('sfr',select_sfr[f])
    STOP
    '''
    
    
    
    
    if drop_repeats:
        
        # Get rid of duplicates in the merger table (using Subfind ID)
        df_dup = all_mergers[all_mergers.duplicated(subset=['Subfind_ID'], keep=False)]
        df_no_dup = all_mergers[~all_mergers.duplicated(subset=['Subfind_ID'], keep='first')]
        df_no_dup = df_no_dup.reset_index()
        print('len after cutting the extra duplicates', len(df_no_dup))
        
        print(df_no_dup)
    
        # Now, using the info of the subhalos at bin center,
        # try to get rid of all progenitors that are within a certain distance of each other
        # To see if there are repeats, assign a tolerance distance, maybe 30 kpc?
        repeat = []
        
        
        
        for j in range(len(df_no_dup)):
            subfind_id_here = df_no_dup['Subfind_ID'].values[j]
            #print('subfindid', subfind_id_here)
            pos_1 = subhalos['SubhaloCM'][subfind_id_here]
            counter = 0
            # so for this second for loop you're going to find the next merger
            # to compare to, so this loop starts at the index of the first galaxy (j)
            # and works downward on the merger table by index
            for k in range(len(df_no_dup)-j-1):
                
                i = k+j+1 # always start one after to not be comparing a galaxy to itself
    
                # Check if x y z are within a 10kpc distance
                subfind_id_sec = df_no_dup['Subfind_ID'].values[i]
                pos_2 = subhalos['SubhaloCM'][subfind_id_sec]
                sep = a * np.sqrt((pos_1[0]-pos_2[0])**2+(pos_1[1]-pos_2[1])**2+(pos_1[2]-pos_2[2])**2) / h
                if sep < 10:#*(1+z)**(-1):
                    
                    repeat.append(i)
                    
                else:
                    counter+=1
            
        
            
        #print('length galaxies on top of each other', len(repeat))
        
        
        # Drop the subhalos that are really close to each other
        try:
            drop_repeats = df_no_dup.drop(df_no_dup.index[np.sort(repeat)])
        except IndexError:# which means repeat is empty
            drop_repeats = df_no_dup
        print('length dropping galaxies overlapping', len(drop_repeats))
        
        
        # Also drop all mergers that have a mass less than the lower limit
        # Previously, we were keeping merging companions down to 1/10th of the ll, but decided
        # we should cut these as well
        dropped_reps = drop_repeats[drop_repeats.mass > ll]# a tenth of the lower limit mass
        drop_cols = dropped_reps.drop(columns=['level_0','index'],axis=1)
        
        # Okay remove all of these weird columns
        drop_cols = drop_cols.reset_index()
        
        print('length after dropping mergers that have a mass less than the lower limit', len(drop_cols))
        
        merg_i = drop_cols['Subfind_ID'].values.astype(int)
    else:
        # Instead of dropping duplicates in the merger table, save their integers:
        dict_type = {'first_progenitor':'1',
                    'next_progenitor':'10',
                    'Merger':'100',
                    'Descendant':'1000'
        }
        # If a galaxy has multiple IDs
        # Get rid of duplicates in the merger table (using Subfind ID)
        
        # Make a new row for all_mergers:
        all_mergers['int_type'] = [dict_type[x] for x in all_mergers['Type'].values]
        
        # Find the non-duplicated entries:
        df_not_dup = all_mergers[~all_mergers.duplicated(subset=['Subfind_ID'], keep=False)]
        print('length of non-duplicated', len(df_not_dup))
        
        
        df_dup_all = all_mergers[all_mergers.duplicated(subset=['Subfind_ID'], keep=False)]
        print('length of everything that is duplicated', len(df_dup_all), 'out of total length', len(all_mergers))

        print('all mergers', all_mergers)        
        #Make a new row for the integers
        #df_dup_all['int_type'] = [dict_type[x] for x in df_dup_all['Type'].values]
        
        df_dup = all_mergers[all_mergers.duplicated(subset=['Subfind_ID'], keep='first')]#False
        print('length of one half of duplicates', len(df_dup))
        
        
        for o in range(len(df_dup)):
        
            find_value = df_dup['Subfind_ID'].values[o]
            index_value = df_dup['index'].values[o]
            # So go through all of the IDs that are repeats and get where they show up
            id_type = []
            
            for p in range(len(df_dup_all[df_dup_all['Subfind_ID']==find_value])):
                id_type.append(int(dict_type[str(df_dup_all[df_dup_all['Subfind_ID']==find_value]['Type'].values[p])]))
            df_dup_all.loc[df_dup_all['Subfind_ID']==find_value,'int_type'] = np.sum(id_type)
        print(df_dup_all)
        
        # The final step is to get rid of everything that is a repeat
        df_no_dup = df_dup_all[~df_dup_all.duplicated(subset=['Subfind_ID'], keep='first')]
        
        df_no_dup_concat = pd.concat([df_no_dup, df_not_dup])
        df_no_dup_concat = df_no_dup_concat.reset_index()
        print('len after cutting the extra duplicates', len(df_no_dup))
        
        dropped_reps = df_no_dup_concat[df_no_dup_concat.mass > ll]# a tenth of the lower limit mass
        drop_cols = dropped_reps.drop(columns=['level_0','index'],axis=1)
        
        print('length after cutting low masses', len(dropped_reps))
        
        # Okay remove all of these weird columns
        drop_cols = drop_cols.reset_index()
        
        
        merg_i = drop_cols['Subfind_ID'].values.astype(int)
        
        
    STOP
    # Now get the masses of the mergers and their sfr from the subhalo catalog:
    mass_mergers = mass_msun[merg_i]
    sfr_mergers = sfr[merg_i]
    # I've verified this is the same is in the merger table (with the below),
    # which means that the indexing is correct
    #print(mass_mergers)
    #print(drop_cols['mass'].values)

        
    drop_sfr = drop_cols.drop(drop_cols.index[np.where(sfr_mergers==0.0)[0]])
    dropped = drop_sfr
    merg_i = dropped['Subfind_ID'].values.astype(int)

    mass_mergers = mass_msun[merg_i]
    sfr_mergers = sfr[merg_i]

    # Make a dictionary for lookback time (lbt, in Gyr) and for redshift for the different snapnums:                                                                         \
    lbt_dict = {'10':13.06788231,'11':13.03606974,'12':12.95636765,'13':12.86836771,'14':12.83476842,'15':12.76390407,'16':12.68785986,'17':12.62304621,'18':12.51872105,'19':12.43449791,'20':12.33446648,
                '22':12.11214593,'23':11.98881427,'24':11.85662123,'25':11.65551451,'26':11.56321698,'27':11.41746261,'28':11.26248662,# once we get back to z= 0.3, progenitors can come from two time steps previously
                '31':10.8208574,'32':10.67239916,'33':10.51690585,'34':10.35409873,'35':10.20851499,
                '38':9.76415887,'39':9.59552818,'40':9.50880841,'41':9.30007834,'42':9.14539478,
                '48':8.22539306,'49':8.076,'50':7.924,'51':7.728,'52':7.6090702,
                '57':6.80477729,'58':6.67044646,'59':6.4881983,'60':6.34915242,'61':6.16060764,
                '65':5.52298259,'66':5.37045644,'67':5.21592302,'68':5.05938984,'69':4.9008657,
                '70':4.74036065,'71':4.57788602,'72':4.41345434,'73':4.24707941,'74':4.07877622,
                '76':3.79403028,'77':3.62066719,'78':3.50405532,'79':3.26837091,'80':3.14931086,
                '81':2.168501225, '82':2.78733837,'83':2.6651,'84':2.4803,'85':2.2937,'86':2.16850124,
                '97':0.34007389,'98':0.13648822,'99':3.2051002e-15
        }
    redshift_dict = {'10':7.23627606616736,'11':7.005417045544533,'12':6.491597745667503,'13':6.0107573988449,'14':5.846613747881867,'15':5.5297658079491026,'18':4.664517702470927,'19':4.428033736605549,'20':4.176834914726472,
                    '22':3.7087742646422353,'24':3.2830330579565246,'27':2.7331426173187188,'28':2.5772902716018935,
                    '31':2.207925472383703,'32':2.1032696525957713,'33':2.0020281392528516,'34':1.9040895435327672,'35':1.822689252620354,
                    '39':1.5312390291576135,'41':1.4140982203725216,
                    '48':1.074457894547674,'49':1.035510445664141,'51':0.9505313515850327,
                    '58':0.7326361820223115,'59':0.7001063537185233,'60':0.6761104112134777,'61':0.644641840684537,
                    '66':0.524565820433923,'67':0.5030475232448832,'68':0.4818329434209512,
                    '70':0.4402978492477432,'71':0.41996894199726653,'72':0.3999269646135635,'73':0.38016786726023866,'74':0.36068765726181673,
                    '76':0.32882972420595435,'77':0.31007412012783386,'78':0.2977176845174465,'79':0.2733533465784399,'80':0.2613432561610123,
                    '81':0.24354018155467028,'82':0.22598838626019768,'83':0.21442503551449454,'85':0.1803852617057493, '86':0.1692520332436107,
                    '97':0.023974428382762536,'98':0.009521666967944764,'99':2.220446049250313e-16
        }

    try:
        lbt_snap = lbt_dict[str(snapnum)]
    except KeyError:
        print('not in dict yet')

    # First step is to figure out how far back to trace the merger tree  

    include_these_snaps = []
    '''
    y = 0
    while (lbt_dict[str(snapnum - y)] - lbt_snap) < 2:
        y+=1
        include_these_snaps.append(snapnum-y)
    '''
    for u in range(10):
        y = u + 1
        try:
            #print(snapnum - y, lbt_dict[str(snapnum - y)] - lbt_snap)
            if lbt_dict[str(snapnum - y)] - lbt_snap > 2:
                break
            else:
                include_these_snaps.append(snapnum - y)
        except:
            # This just means that the it doesnt exist in the dict
            #break
            #print('tried to get this one', snapnum - y)
            continue
    
    #print(include_these_snaps, snapnum - min(include_these_snaps))
    
    # This is the number of snapshot steps to include to go the full 2 Gyr back in time
    n_steps = snapnum - min(include_these_snaps)


    #print('merg_i',merg_i)
    
    # Now, for the mergers, measure the nearest neighbor quantities:
    N_within_2MPC = [] # number of galaxies within 2 Mpc (that are above the mass limit below)
    r_1 = []# distance in kpc to the nearest neighbor with m > 0.1 mass
    r_2 = []# same but to the second hearest neighbor
    
    counter = 0
    for id in merg_i:
        #print(id, dropped['Subfind_ID'].values[counter], dropped['Type'].values[counter], dropped['q'].values[counter])
        # So for each ID now go through and compute the relative distance from every other subhalo to this one at this z
        #print('pos', dropped['SubhaloCM'][id])
        pos = subhalos['SubhaloCM'][id]
        
        
        #print('supposed mass of subhalo that is merging', mass_msun[id], 'that youre putting in table', dropped['mass'].values[0])
        '''
        rel_distance = []
        for o in range(len(subhalos)): # go through all subhalos
            if o==id: # don't need to compare to itself
                continue
            if mass_msun[o] < 0.1*mass_msun[id]: # if the mass of the o-th subhalo is less than 10% of the mass of this subhalo
                continue
            pos_sub = subhalos['SubhaloCM'][o]
            dist = a * np.sqrt((pos[0] - pos_sub[0])**2 + (pos[1] - pos_sub[1])**2 + (pos[2] - pos_sub[2])**2) / h #in comoving kpc
            rel_distance.append(dist)
        '''
        
        indices_large_enough = list(np.where(mass_msun >0.1*mass_msun[id])[0])
        rel_distance = []
        for o in indices_large_enough: # go through all subhalos                                                                                                             
            if o==id: # don't need to compare to itself                                                                                                                       
                continue
            if mass_msun[o] < 0.1*mass_msun[id]: # if the mass of the o-th subhalo is less than 10% of the mass of this subhalo                                              
                STOP
                continue
            pos_sub = subhalos['SubhaloCM'][o]
            dist = a * np.sqrt((pos[0] - pos_sub[0])**2 + (pos[1] - pos_sub[1])**2 + (pos[2] - pos_sub[2])**2) / h #in comoving kpc                                           
            rel_distance.append(dist)

        
        #print('mass of merging gal', mass_msun[id])
        #print('length of rel distance / total # subhalos', len(rel_distance), len(indices_large_enough),len(rel_distance)/len(indices_large_enough))

        # Quantify how many are less than 2000 kpc away
        #print(rel_distance)
        #print(np.where(np.array(rel_distance) < 2000), len(np.where(np.array(rel_distance)<2000)[0]))
        sorted_d = np.sort(rel_distance)
        r_1.append(sorted_d[0])
        r_2.append(sorted_d[1])
        N_within_2MPC.append(len(np.where(np.array(rel_distance)<2000)[0]))


        #print('r_1', sorted_d[0], 'r_2', sorted_d[1])
        # Also, for each merger test the code to see how many mergers it had in the last 2 Gyr :)
        fields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                  'FirstProgenitorID', 'SubhaloMassType','DescendantID','SubfindID']
        
        # Need to set onlyMPB = False in order to do merger analysis with the three:
        tree = il.sublink.loadTree(sP,snapnum, id, fields = fields, onlyMPB=False)#, onlyMPB=False)#                                                                                                  
        #
        minMR = 1.0/10.0
        #print('# of mergers if you go back only one step',num_mergers_one_step(tree, minMR, massPartType='stars', index=1))
        
        n_merg_2Gyr = num_mergers_n_step(tree,n_steps,  minMR, massPartType='stars', index=1)
        #print('# of mergers', np.sum(n_merg_2Gyr))
        '''
        numMergers = numMergers(tree,minMassRatio=1.0/10.0)
        print('number of mergers in previous step', numMergers)
        numMergers = numMergers(tree,minMassRatio=1.0/10.0, index = 1)
        print('number of mergers in previous step', numMergers)
        STOP
        '''
        counter+=1
        
    '''                                                                                                                                                                       
                                                                                                                                                                              
    ratio = 1.0/5.0                                                                                                                                                           
    # the following fields are required for the walk and the mass ratio analysis                                                                                              
    fields = ['SubhaloID','NextProgenitorID','MainLeafProgenitorID','FirstProgenitorID','SubhaloMassType']                                                                    
    # Walk back along the tree                                                                                                                                                
    tree = il.sublink.loadTree(sP,snapnum, merg_i[0],fields = fields)#                                                                                                        
    numMergers = il.sublink.numMergers(tree,minMassRatio=ratio)                                                                                                               
                                                                                                                                                                              
    print('# mergers', numMergers, merg_i[0])                                                                                                                                 
                                                                                                                                                                              
                                                                                                                                                                              
    STOP                                                                                                                                                                      
    '''

    '''
    print('for mergers')
    print('length', len(merg_i), len(mass_mergers))
    print('r_1', r_1)
    print('r_2', r_2)
    print('N_within_2MPC', N_within_2MPC)
    '''
 
    plt.clf()
    plt.scatter(r_1, N_within_2MPC, color = '#FFD166')
    plt.xlabel('r_1 in kpc')
    plt.ylabel('Log N within 2 Mpc')
    plt.yscale('log')
    plt.savefig('Figs/TS_'+str(snapnum)+'.png')
    
    plt.clf()
    plt.scatter(mass_mergers, N_within_2MPC, color = '#FFD166')
    plt.xlabel('Log mass of merger')
    plt.ylabel('Log N within 2 Mpc')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('Figs/TS_mass_'+str(snapnum)+'.png')
    
    plt.clf()
    plt.hist(r_1, color = '#FFD166', bins=50)
    plt.savefig('Figs/TS_hist_r_1_'+str(snapnum)+'.png')

    plt.clf()
    plt.hist(N_within_2MPC, color = '#FFD166', bins=50)
    plt.savefig('Figs/TS_hist_N_'+str(snapnum)+'.png')


    if drop_repeats:
        # Save a merger table with all of the environmental information
        # Because I'm deleting all repeated subhalos, aren't many descendants,
        # this is especially true at higher z
        
        if all_prog:
            file_out = open('Tables/mergers_all_nodup_all_prog_'+str(snapnum)+'_'+run+'_'+str(version)+'.txt','w')
        else:
            file_out = open('Tables/mergers_all_nodup_'+str(snapnum)+'_'+run+'_'+str(version)+'.txt','w')
        file_out.write('Subfind_ID'+'\t'+'t-t_merg'+'\t'+'q'+'\t'+'Type'+'\t'+'mass'+'\t'+'N_2Mpc'+'\t'+'r_1'+'\t'+'r_2'+'\n')
        counter_merg = 0
        for j in range(len(merg_i)):
            
            file_out.write(str(dropped['Subfind_ID'].values[j])+'\t'+str(dropped['t-t_merg'].values[j])+'\t'+
            str(dropped['q'].values[j])+'\t'+str(dropped['Type'].values[j])+'\t'+str(dropped['mass'].values[j])+'\t'+str(N_within_2MPC[counter_merg])+'\t'+str(r_1[counter_merg])+'\t'+str(r_2[counter_merg])+'\n')
            counter_merg += 1
        file_out.close()
        
        print('MADE merger table')
    else:
        if all_prog:
            file_out = open('Tables/mergers_all_prog_includes_repeats_'+str(snapnum)+'_'+run+'_'+str(version)+'.txt','w')
        else:
            file_out = open('Tables/mergers_includes_repeats_'+str(snapnum)+'_'+run+'_'+str(version)+'.txt','w')
        file_out.write('Subfind_ID'+'\t'+'t-t_merg'+'\t'+'q'+'\t'+'Type'+'\t'+'int_type'+'\t'+'mass'+'\t'+'N_2Mpc'+'\t'+'r_1'+'\t'+'r_2'+'\n')
        counter_merg = 0
        for j in range(len(merg_i)):
            
            file_out.write(str(dropped['Subfind_ID'].values[j])+'\t'+str(dropped['t-t_merg'].values[j])+'\t'+
            str(dropped['q'].values[j])+'\t'+str(dropped['Type'].values[j])+'\t'+str(dropped['int_type'].values[j])+'\t'+
            str(dropped['mass'].values[j])+'\t'+str(N_within_2MPC[counter_merg])+'\t'+
            str(r_1[counter_merg])+'\t'+str(r_2[counter_merg])+'\n')
            counter_merg += 1
        file_out.close()
        
        print('Made merger table with new column')
    
    STOP
    
    
    
    # But you first need to correct for galaxies that have masses less than the ll and 10**14
    # and sfr ==0.0
    # This really reduces the number of subhalos
    
    #select_indices = np.where(np.logical_and(mass_msun > minmass, mass_msun < maxmass, sfr > 0.0))[0]
    #select_mass = mass_msun[select_indices]
    #select_sfr = sfr[select_indices]
    
    
    # Get N, r_1, and r_2 for all of the non-mergers in the mass_select_all
    # starting with subhalos
    N_all = []
    r_1_all = []
    r_2_all = []

    had_a_merger_in_last_2Gyr = []


    minMR = 1.0/10.0

    for p in select_indices:
        # eliminate subhalos that have had a merger in the last 2 Gyr

        tree = il.sublink.loadTree(sP,snapnum, p, fields = fields)#, onlyMPB=False)#                                                                                      
         
        n_merg_2Gyr = num_mergers_n_step(tree, n_steps, minMR, massPartType='stars', index=1)
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
        for u in range(len(subhalos)):
            #print(p, u, mass_msun[p], mass_msun[u])
            if u==p:
                continue
            if mass_msun[u] < 0.1*mass_msun[p]:
                continue
            #print('massive enough to pass through initially', p, u)
            #print(mass_msun[p], mass_msun[u], mass_msun[p]/mass_msun[u])
            pos_sub = subhalos['SubhaloCM'][u]
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
    print('N_all', N_all)
    print('r_1_all', r_1_all)
    print('r_2_all', r_2_all)
    #N_select_all = N_all[(mass_msun > minmass) & (mass_msun < maxmass) & (sfr > 0.0)]
    #r_1_select_all = r_1_all[(mass_msun > minmass) & (mass_msun < maxmass) & (sfr > 0.0)]
    #r_2_select_all = r_2_all[(mass_msun > minmass) & (mass_msun < maxmass) & (sfr > 0.0)]
    
    plt.clf()
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.hist(N_all, label='All', alpha=0.5, bins=50, range=(0, 50))
    plt.legend()
    
    ax1 = fig.add_subplot(212)
    ax1.hist(N_within_2MPC, label='Mergers', alpha=0.5, bins=50, range=(0, 50))
    #ax1.set_xlim([0,100])
    plt.legend()
    plt.savefig('Figs/hist_N_2Mpc_all_subhalos_'+str(snapnum)+'_'+str(version)+'.png')
    STOP
    

    # I don't think we need all this now:
    '''
    merger_mass_cut_ll = dropped[dropped.mass > ll]
    
    
    #corr_len = len(mass_select_all) - len(merg_i)
    file_out_2 = open('Tables/fraction_mergers_'+str(snapnum)+'_'+run+'_'+str(version)+'.txt','w')
    file_out_2.write('# mergers = '+str(len(merg_i))+'\n')
    file_out_2.write('# mergers below mass cut = '+str(len(merger_mass_cut_ll))+'\n')
    file_out_2.write('# nonmergers = '+str(len(mass_select_all))+'\n')
    #file_out_2.write('# nonmergers - # mergers = '+str(corr_len)+'\n')
    #file_out_2.write('fraction = '+str(round(len(merg_i)/len(mass_select_all),3))+'\n')
    #file_out_2.write('fraction for mergers >ll = '+str(round(len(merger_mass_cut_ll)/len(mass_select_all),3)))
    file_out_2.close()
    
    
    # Also plot out the masses to see what you're selecting from:
    plt.clf()
    logbins = np.logspace(np.log10(10**6),np.log10(10**13),100)
    plt.hist(mass_select_all, bins=logbins, label='All subhalos', alpha=0.5)
    plt.hist(mass_mergers, bins=logbins, label='Mergers mass')
    plt.hist(merger_mass_cut_ll['mass'].values, bins=logbins, label='Mergers mass cut')
    plt.xscale('log')
    plt.legend()
    plt.savefig('Figs/hist_mass_all_subhalos_'+str(snapnum)+'_'+str(version)+'.png')
    
    '''
    
    
    
    
    # Make a selection of things that are the mergers:
    mass_mergers = mass_msun[merg_i]
    sfr_mergers = sfr[merg_i]
    
    # Okay additionally select galaxies that are pre and post-mergers:
    merg_i_pre = []
    merg_i_post = []
    merg_i_merg = []
    for n in range(len(dropped)):
        if dropped['Type'].values[n]=='next_progenitor' or dropped['Type'].values[n]=='first_progenitor':
            merg_i_pre.append(int(dropped['Subfind_ID'].values[n]))
        if dropped['Type'].values[n]=='Merger':
            merg_i_merg.append(int(dropped['Subfind_ID'].values[n]))
        if dropped['Type'].values[n]=='Descendant':
            merg_i_post.append(int(dropped['Subfind_ID'].values[n]))


    merg_i = dropped['Subfind_ID'].values.astype(int)
    
    # Now also go in and select a matched sample of subhalos for your nonmergers :)
    fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType']#was ['SubhaloMassInRad']
    
    # It might actually be useful to sort this by mass, so you can snag all subhalos between a certain mass range:                                      
    
    
    # Now sort into different bins:
    '''
    if snapnum==17:
        mass_bins = [[ll/10,ll],[ll,ll*10],[ll*10,ll*100],[ll*100,ll*1000]]
        sfr_bins = [[10**-6,10**(-5)],[10**(-5),10**(-4)],[10**-4,10**-3],[10**-3,10**-2],[10**-2,10**-1],[10**-1,10**0],[10**0,10**1],[10**1,10**2],[10**2,10**3]]
    if snapnum==25:
        mass_bins = [[ll/10,ll],[ll,ll*10],[ll*10,ll*100],[ll*100,ll*1000],[ll*1000,ll*10000]]
        sfr_bins = [[10**-6,10**(-5)],[10**(-5),10**(-4)],[10**-4,10**-3],[10**-3,10**-2],[10**-2,10**-1],[10**-1,10**0],[10**0,10**1],[10**1,10**2],[10**2,10**3]]
    
    if snapnum==33:
        mass_bins = [[ll/10,ll],[ll,ll*10],[ll*10,ll*100],[ll*100,ll*1000],[ll*1000,ll*10000]]#,[ll*1e4,ll*1e5]]
        sfr_bins = [[10**-6,10**(-5)],[10**(-5),10**(-4)],[10**-4,10**-3],[10**-3,10**-2],[10**-2,10**-1],[10**-1,10**0],[10**0,10**1],[10**1,10**2],[10**2,10**3]]
    if snapnum==40:
        mass_bins = [[ll/10,ll],[ll,ll*10],[ll*10,ll*100],[ll*100,ll*1000],[ll*1000,ll*10000]]#,[ll*1e4,ll*1e5]]
        sfr_bins = [[10**-6,10**(-5)],[10**(-5),10**(-4)],[10**-4,10**-3],[10**-3,10**-2],[10**-2,10**-1],[10**-1,10**0],[10**0,10**1],[10**1,10**2],[10**2,10**3]]
    '''
    mass_bins = [[ll/1000,ll/100],[ll/100,ll/10],[ll/10,ll],[ll,ll*10],[ll*10,ll*100],[ll*100,ll*1000],[ll*1000,ll*10000],[ll*10000,ll*100000]]#,[ll*1e4,ll*1e5]]
    sfr_bins = [[10**-6,10**(-5)],[10**(-5),10**(-4)],[10**-4,10**-3],[10**-3,10**-2],[10**-2,10**-1],[10**-1,10**0],[10**0,10**1],[10**1,10**2],[10**2,10**3]]
    
    
    '''
    if (z < 2) and (z > 0.1):
        mass_bins = [[ll/10,ll],[ll,ll*10],[ll*10,ll*100],[ll*100,ll*1000],[ll*1000,ll*10000]]#,[ll*1e4,ll*1e5]]
        sfr_bins = [[10**-6,10**(-5)],[10**(-5),10**(-4)],[10**-4,10**-3],[10**-3,10**-2],[10**-2,10**-1],[10**-1,10**0],[10**0,10**1],[10**1,10**2],[10**2,10**3]]
    
    if z < 0.1:
        mass_bins = [[ll/10,ll],[ll,ll*10],[ll*10,ll*100],[ll*100,ll*1000]]
        sfr_bins = [[10**-6,10**(-5)],[10**(-5),10**(-4)],[10**-4,10**-3],[10**-3,10**-2],[10**-2,10**-1],[10**-1,10**0],[10**0,10**1],[10**1,10**2]]
    
    '''
    
    print('min max mass mergers', min(mass_mergers), max(mass_mergers))
    print('min sfr', min(sfr_mergers), max(sfr_mergers))
    print(sfr_bins[0][0],sfr_bins[-1][1],mass_bins[0][0],mass_bins[-1][1])
    if (min(sfr_mergers) < sfr_bins[0][0]) or (max(sfr_mergers) > sfr_bins[-1][1]) or (min(mass_mergers) < mass_bins[0][0]) or (max(mass_mergers) > mass_bins[-1][1]):
        print('outside range of bins')
        STOP
    
    
    nonmerg_subfindid = []
    nonmerg_mass = []
    nonmerg_sfr = []
    
    delete_these_mergers = []
    
    # def mergers_one_step(tree, minMR, massPartType='stars', index=1):
    # numMergers, Subfind_ID_array, ratio_out = mergers_one_step(tree,ratio_min)
    for j in range(len(mass_bins)):
        minmass = mass_bins[j][0]
        maxmass = mass_bins[j][1]
        
        #select_mass = mass_msun[select_indices]
        #select_sfr = sfr[select_indices]
        
        # Okay, make this 2D:
        for i in range(len(sfr_bins)):
            minsfr = sfr_bins[i][0]
            maxsfr = sfr_bins[i][1]
            
            # But don't select more mergers!
            select_indices = list(np.where(np.logical_and(np.logical_and(mass_msun > minmass, mass_msun < maxmass), 
                np.logical_and(sfr > minsfr, sfr < maxsfr)))[0])
            print('original length of all galaxies in this bin', len(select_indices))
            select_indices_norep = []
            for element in select_indices:
                if element in merg_i:
                    remove = 'yes'
                    #print('removing this element', element)
                    #was .remove(element)
                else:
                    select_indices_norep.append(element)
            # Now, count how many mergers are in this mass range:
            where_merger_select = np.where(np.logical_and(np.logical_and(mass_mergers > minmass, mass_mergers < maxmass), np.logical_and(sfr_mergers > minsfr, sfr_mergers < maxsfr)))[0]
            nmergers_bin = len(where_merger_select)
            print('binmin mass ', np.log10(minmass), 'binmax ', np.log10(maxmass), 'sfr min', minsfr, 'max', maxsfr)
            print('# of mergers in this bin', nmergers_bin, '# of nonmergers in this bin w/o rep', len(select_indices_norep), '# of subhalos in this bin', len(select_indices))
            
            
                
            try:
                indices = random.sample(select_indices_norep, nmergers_bin)
            except ValueError:
                print('there are more mergers than all nonmerging galaxies at this bin')
                print('nonmerg', len(select_indices_norep), 'merg', nmergers_bin)
                indices_suspect = where_merger_select
                # Make this general because there might be more than one merger in here
                for suspicious_idx in indices_suspect:
                    delete_these_mergers.append(suspicious_idx)
                continue
                
            print('actually got this many', len(indices))
            # Make sure indices is not already selected as a merger, that would be bad
            
            
            
            
            '''for k in range(len(indices)):
                if indices[k] in np.array(merg_i):
                    #print('gotta replace', indices[k])
                    indices[k] = random.sample(select_indices, 1)[0]
                    #print('replacing with this', indices[k])
                    if indices[k] in np.array(merg_i):
                        indices[k] = random.sample(select_indices, 1)[0]
                        if indices[k] in np.array(merg_i):
                            indices[k] = random.sample(select_indices, 1)[0]
                            if indices[k] in np.array(merg_i):
                                indices[k] = random.sample(select_indices, 1)[0]
                                STOP
            '''
            for z in range(nmergers_bin):
                nonmerg_subfindid.append(indices[z])
                nonmerg_mass.append(mass_msun[indices[z]])
                nonmerg_sfr.append(sfr[indices[z]])
            
    if delete_these_mergers: # then delete them
        print('dropping these many mergers that have no match', len(delete_these_mergers))
        print('suspicious index list', delete_these_mergers)
        #print('len mergers', len(mass_mergers), 'after delete', len(mass_mergers.delete(delete_these_mergers_mass)))
        
        # Now delete the rows you don't need 
        
        new_dropped = dropped.drop(dropped.index[delete_these_mergers])
        
        #new_dropped = dropped[~dropped.index(delete_these_mergers)]
        
        
        # Now resave after deleting the mergers that 
        if all_prog:
            file_out = open('Tables/mergers_matched_all_nodup_all_prog_'+str(snapnum)+'_'+run+'_'+str(version)+'.txt','w')
        else:
            file_out = open('Tables/mergers_matched_all_nodup_'+str(snapnum)+'_'+run+'_'+str(version)+'.txt','w')
        file_out.write('Subfind_ID'+'\t'+'t-t_merg'+'\t'+'q'+'\t'+'Type'+'\t'+'mass'+'\n')
        for j in range(len(new_dropped)):
            file_out.write(str(new_dropped['Subfind_ID'].values[j])+'\t'+str(new_dropped['t-t_merg'].values[j])+'\t'+
            str(new_dropped['q'].values[j])+'\t'+str(new_dropped['Type'].values[j])+'\t'+str(new_dropped['mass'].values[j])+'\n')
        file_out.close()
    
    
    # need to re-select sfr and mass to match what we've selected:
    
    plt.clf()
    logbins = np.logspace(np.log10(10**6),np.log10(10**13),19)
    plt.hist(mass_mergers, bins=logbins, label='Mergers mass')
    plt.hist(nonmerg_mass, bins=logbins, label='Nonmergers mass', alpha=0.5)
    plt.xscale('log')
    plt.legend()
    
    #plt.hist(mass_mergers, label='Mergers mass')
    #plt.hist(nonmerg_mass, label='Nonmergers mass')
    #plt.xscale('log')
    if all_prog:
        plt.savefig('Figs/hist_matched_mass_snapnum_all_prog_'+str(snapnum)+'_'+str(version)+'.png')
    
    else:
        plt.savefig('Figs/hist_matched_mass_snapnum_'+str(snapnum)+'_'+str(version)+'.png')
    
    
    plt.clf()
    logbins = np.logspace(np.log10(10**-6),np.log10(10**3),19)
    plt.hist(sfr_mergers, bins=logbins, label='Mergers sfr')
    plt.hist(nonmerg_sfr, bins=logbins, label='Nonmergers sfr', alpha=0.5)
    plt.xscale('log')
    plt.legend()
    
    #plt.hist(mass_mergers, label='Mergers mass')
    #plt.hist(nonmerg_mass, label='Nonmergers mass')
    #plt.xscale('log')
    if all_prog:
        plt.savefig('Figs/hist_matched_sfr_snapnum_all_prog_'+str(snapnum)+'_'+str(version)+'.png')
    else:
        plt.savefig('Figs/hist_matched_sfr_snapnum_'+str(snapnum)+'_'+str(version)+'.png')
    
    print('merger subid', merg_i)
    print(nonmerg_subfindid)
    print(nonmerg_mass)
    
    
    if all_prog:
        file_out = open('Tables/nonmergers_snap_all_prog_'+str(snapnum)+'_'+run+'_'+str(version)+'.txt','w')
    else:
        file_out = open('Tables/nonmergers_snap_'+str(snapnum)+'_'+run+'_'+str(version)+'.txt','w')
    file_out.write('Subfind_ID'+'\t'+'mass'+'\n')
    for j in range(len(nonmerg_subfindid)):
        file_out.write(str(nonmerg_subfindid[j])+'\t'+str(nonmerg_mass[j])+'\n')
    file_out.close()
    
    mass_nonmergers = mass_msun[nonmerg_subfindid]
    sfr_nonmergers = sfr[nonmerg_subfindid]
    # Finally make a plot with nonmergers and mergers on it
    plt.clf()
    
    sns.set_style('whitegrid')
    plt.plot(mass_msun, sfr, '.', color='black', label = 'All Subhalos')
    plt.plot(select_mass, select_sfr,'.', color='orange', label='Mass Selection')
    plt.plot(mass_nonmergers, sfr_nonmergers, '.', color='blue', label='Nonmergers')
    plt.plot(mass_mergers, sfr_mergers, '.', color='red', label='Mergers')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([10**4,10**(12.5)])
    plt.ylim([10**(-5.5),10**3])
    plt.xlabel('Stellar Mass [$M_\odot$]')
    plt.ylabel('Star Formation Rate [$M_\odot / yr$]')
    plt.axvline(x=ll, color='black', ls='--')
    plt.title('$z = $'+str(round(redshift_label,1))+'\n# Mergers = '+str(len(mass_mergers))+', # Nonmergers = '+str(len(mass_nonmergers)))
    if all_prog:
        plt.savefig('Figs/selection_subhalos_merg_gif_all_prog_'+str(run)+'_snap_'+str(snapnum)+'_'+str(version)+'.png')
    else:
        plt.savefig('Figs/selection_subhalos_merg_gif_'+str(run)+'_snap_'+str(snapnum)+'_'+str(version)+'.png')
    

    # And then separate out the pre, post, and mergers                                                                   
    plt.clf()

    sns.set_style('whitegrid')
    plt.plot(mass_msun, sfr, '.', color='#0A210F', label = 'All Subhalos')
    #plt.plot(select_mass, select_sfr,'.', color='orange', label='Mass Selection')
    plt.plot(mass_nonmergers, sfr_nonmergers, '.', color='#ACD2ED', label='Nonmergers')
    plt.plot(mass_msun[merg_i_pre], sfr[merg_i_pre], '.', color='#14591D', label='Pre-mergers')
    plt.plot(mass_msun[merg_i_merg], sfr[merg_i_merg], '.', color='#99AA38', label='Mergers')
    plt.plot(mass_msun[merg_i_post], sfr[merg_i_post], '.', color='#E1E289', label='Post-mergers')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([10**4,10**(12.5)])
    plt.ylim([10**(-5.5),10**3])
    plt.xlabel('Stellar Mass [$M_\odot$]')
    plt.ylabel('Star Formation Rate [$M_\odot / yr$]')
    plt.axvline(x=ll, color='black', ls='--')
    plt.title('$z = $'+str(round(redshift_label,1))+'\n# Mergers = '+str(len(mass_mergers))+', # Nonmergers = '+str(len(mass_nonmergers)))
    if all_prog:
        plt.savefig('Figs/selection_subhalos_merg_gif_all_sep_prog_'+str(run)+'_snap_'+str(snapnum)+'.png')
    else:
        plt.savefig('Figs/selection_subhalos_merg_gif_sep_'+str(run)+'_snap_'+str(snapnum)+'_'+str(version)+'.png')
