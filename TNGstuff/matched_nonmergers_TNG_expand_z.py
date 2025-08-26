# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the version for Aimee's TNG project:
# Loops through all snapshots, grabs huge merger table
# and makes a non-merger sample that is matched in
# a bunch of properties, including density and mass
# and redshift at each bin
# This one allows galaxies to be from different snapshots.
# But still doesn't allow repeats
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import illustris_python as il
import numpy as np
import matplotlib
import random
matplotlib.use('agg')# must call this before importing pyplot as plt: 
import matplotlib.pyplot as plt
import seaborn as sns
import util_mergers as u      

basePath = '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L75n1820TNG'#'L35n2160TNG'        

        
sP = basePath+run+'/output/'
version = 'N_tight_only'
#'enviro'
#'no_enviro'
#'no_enviro_tight'
#'N_enviro'
#'N_enviro_tight'

years_ago = 2

h = 0.6774
minmass = 10**3*(5.7*10**4)/h # 1000 times the mass resolution
ll = minmass
maxmass = 10**14
        
print('minimum mass selection', ll, 'max mass', maxmass)

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

def find_match(version, mass, r1, r2, N2, df):
    ngrow = 0
    #default tolerances
    Mstar_tol = 0.1#0.1 # this is a dex so relative to the np.log [natural log] of the mass
    #ss_tol = 1
    r1_tol = 0.1*r1
    r2_tol = 0.1*r2
    N2_tol = 0.1*N2
    found = False
    #print('trying to match this', mass, r1, r2, N2)
    
    #print(df[(df['num_mergers_in_last_2Gyr']==0.0) & 
    #    (np.log(df['Mass'])-Mstar_tol<=np.log(mass)) &
    #    (np.log(mass)<=np.log(df['Mass'])+Mstar_tol)
    #    ])
    #STOP

    while found == False:
        
        if version == 'enviro':
            indices = df.index[(df['num_mergers_in_last_2Gyr']==0.0) &
                              (np.log(df['Mass'])-Mstar_tol<=np.log(mass)) &
                              (np.log(mass)<=np.log(df['Mass'])+Mstar_tol) & 
                              (df['r_1']-r1_tol<=r1) &
                              (r1<=df['r_1']+r1_tol) &
                              (df['r_2']-r2_tol<=r2) &
                              (r2<=df['r_2']+r2_tol) &
                              (df['N']-N2_tol<=N2) &
                              (N2<=df['N']+N2_tol)].tolist()
        if version == 'N_r_1_enviro':
            indices = df.index[(df['num_mergers_in_last_2Gyr']==0.0) &
                              (np.log(df['Mass'])-Mstar_tol<=np.log(mass)) &
                              (np.log(mass)<=np.log(df['Mass'])+Mstar_tol) & 
                              (df['r_1']-r1_tol<=r1) &
                              (r1<=df['r_1']+r1_tol) &
                              (df['N']-N2_tol<=N2) &
                              (N2<=df['N']+N2_tol)].tolist()
        if version == 'N_enviro':
            indices = df.index[(df['num_mergers_in_last_2Gyr']==0.0) &
                              (np.log(df['Mass'])-Mstar_tol<=np.log(mass)) &
                              (np.log(mass)<=np.log(df['Mass'])+Mstar_tol) &
                              (df['N']-N2_tol<=N2) &
                              (N2<=df['N']+N2_tol)].tolist()
        if version == 'N_tight_enviro':
            N2_tol = 0.01*N2
            indices = df.index[(df['num_mergers_in_last_2Gyr']==0.0) &
                              (np.log(df['Mass'])-Mstar_tol<=np.log(mass)) &
                              (np.log(mass)<=np.log(df['Mass'])+Mstar_tol) &
                              (df['N']-N2_tol<=N2) &
                              (N2<=df['N']+N2_tol)].tolist()
        if version == 'N_tight_only':
            N2_tol = 0.01*N2
            indices = df.index[(df['num_mergers_in_last_2Gyr']==0.0) &
                              (df['N']-N2_tol<=N2) &
                              (N2<=df['N']+N2_tol)].tolist()
        if version == 'no_enviro':
            
            '''
            print('mass here', mass)
            print('np.log of this', np.log(mass))
            print('tolerance', Mstar_tol)
            print(np.log(mass) - Mstar_tol, np.log(mass) + Mstar_tol)
            print('going back to a number')
            print(np.exp(np.log(mass) - Mstar_tol), np.exp(np.log(mass) + Mstar_tol))
            '''
            indices = df.index[(df['num_mergers_in_last_2Gyr']==0.0) &
                              (np.log(df['Mass'])-Mstar_tol<=np.log(mass)) &
                              (np.log(mass)<=np.log(df['Mass'])+Mstar_tol)].tolist()
                              
            
                        
        
        
        if len(indices) > 0:
            match_index = random.choice(indices)
            # Instead of index please get the subhaloID
            match_subid = df.loc[[match_index]]['SubhaloID'].values[0]
            #df['Selected'][match_index] = True
            #pm_div_ctrl_mr.append(np.abs(df['Mstar'][match_index]-mass))
            found = True
            print('length were choosing between', len(indices))
        else:
            Mstar_tol *= 1.5
            #ss_tol += 1
            r1_tol *= 1.5
            r2_tol *= 1.5
            N2_tol *= 1.5
            ngrow += 1
    return match_index, match_subid, ngrow




# This time loop over all of the snapnums:

snapnum_list = [17,25,33,40,50,59,67,72,84]
snapnum_list = [50]
for snapnum in snapnum_list:
    
    if run=='L75n1820TNG':
        fields = ['SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']
    else:
        fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']#was \['SubhaloMassInRad']    'SubhaloFlag',                                                                 


    
    subhalosh = il.groupcat.loadHeader(basePath+run+'/output/',snapnum)#, fields=fields_header)
    z = subhalosh['Redshift']
    print('redshift', z)
    a = subhalosh['Time']
    print('scale factor', a)

    
    redshift_label = z
    
    
    # Load in massive merger tables:
    mergers = pd.read_csv('Tables/Aimee_SF/'+str(snapnum)+'/all_mergers.txt', sep='\t')
    
    print('will need to run through all of these individual snaps', mergers['center'].unique())
    
    # Now go through and get a matched sample at each snapnum center:
    for snap in mergers['center'].unique():
        
        # Get all subhalos at this redshift
        
        if run=='L75n1820TNG':
            fields = ['SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']
        else:
            fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']#was \['SubhaloMassInRad']    'SubhaloFlag',                                                                 

        subhalos = il.groupcat.loadSubhalos(basePath+run+'/output/',snap, fields=fields)
        mass_msun = subhalos['SubhaloMassType'][:,4] * 1e10 / h
        sfr = subhalos['SubhaloSFRinRad']
        
        # This selects all subhalos at this snapnum that are within our mass range
        select_indices = np.where(np.logical_and(mass_msun > minmass, mass_msun < maxmass, sfr > 0.0))[0]
        select_mass = mass_msun[select_indices]
        select_sfr = sfr[select_indices]
        
        # Okay the first step is to get the environmental information for the mergers
        lbt_dict, redshift_dict = u.lookup_dicts()
        # get the lbt for this snapshot
        lbt_snap = lbt_dict[str(snap)]
        
        include_these_snaps = []
        
        for w in range(10):
            y = w + 1
            try:
                if lbt_dict[str(snap - y)] - lbt_snap > years_ago:
                    break
                else:
                    include_these_snaps.append(snap - y)
            except:
                # This just means that the it doesnt exist in the dict
                #break
                continue
            
            
        # This is the number of snapshot steps to include to go the full 2 Gyr back in time
        n_steps = snap - min(include_these_snaps)
        
        print('number of steps', n_steps)
        
        
        # First, check to see if the merger table with environmental information exists for this snap center:
        try:
            selection = pd.read_csv('Tables/Aimee_SF/'+str(snapnum)+'/all_mergers_at_'+str(snap)+'_enviro.txt', sep = '\t')
            merg_i = selection['Subfind_ID'].values.astype(int)
            mass_mergers = mass_msun[merg_i]
            sfr_mergers = sfr[merg_i]
            print('             MERGERS ALREADY READY TO GO                 ', snap)
            #continue
        except:
            # You are going to need to make the table yourself:
            print('             MAKING THE MERGER TABLE                ', snap)
        
            selection = mergers[mergers['center']==snap]
            
            merg_i = selection['Subfind_ID'].values.astype(int)
            
            # First get a list of snapshots to include based on look back time:
            
            
            # Now, get info about all subhalos at bin center 
            
            
            
            
            
            # Now get the mass and sfr of the mergers 
            mass_mergers = mass_msun[merg_i]
            sfr_mergers = sfr[merg_i]
            
            if np.any(sfr_mergers == 0.0):
                # You also have to eliminate mergers with a SFR = 0 because these are weird halos
                print('SFR of mergers has 0.0s in it', sfr_mergers)
                
                plt.hist(sfr_mergers, bins=50)
                plt.savefig('Figs/sfr_zeros.png')
                
                print('length of selection', len(selection))
                
                selection['sfr'] = sfr_mergers
                
                selection = selection[selection.sfr > 0.0]# a tenth of the lower limit mass
                
                print('length after dropping 0.0 in SFR', len(selection))
                merg_i = selection['Subfind_ID'].values.astype(int)
                mass_mergers = mass_msun[merg_i]
                sfr_mergers = sfr[merg_i]
                
            
            
        
        
            #print('merg_i',merg_i)
            
            # Now, for the mergers, measure the nearest neighbor quantities:
            N_within_2MPC = [] # number of galaxies within 2 Mpc (that are above the mass limit below)
            r_1 = []# distance in kpc to the nearest neighbor with m > 0.1 mass
            r_2 = []# same but to the second hearest neighbor
            had_a_merger_in_last_2Gyr = []
            
            counter = 0
            for id in merg_i:
                #print(id, dropped['Subfind_ID'].values[counter], dropped['Type'].values[counter], dropped['q'].values[counter])
                # So for each ID now go through and compute the relative distance from every other subhalo to this one at this z
                #print('pos', dropped['SubhaloCM'][id])
                
                
                
                pos = subhalos['SubhaloCM'][id]
                
                
                
                
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
                
                print('sorted_d', sorted_d)
                STOP
                r_1.append(sorted_d[0])
                r_2.append(sorted_d[1])
                N_within_2MPC.append(len(np.where(np.array(rel_distance)<2000)[0]))
        
        
                #print('r_1', sorted_d[0], 'r_2', sorted_d[1])
                # Also, for each merger test the code to see how many mergers it had in the last 2 Gyr :)
                fields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                          'FirstProgenitorID', 'SubhaloMassType','DescendantID','SubfindID']
                
                # Need to set onlyMPB = False in order to do merger analysis with the three:
                tree = il.sublink.loadTree(sP,snap, id, fields = fields, onlyMPB=False)#, onlyMPB=False)#                                                                                                  
                #
                minMR = 1.0/10.0
                #print('# of mergers if you go back only one step',num_mergers_one_step(tree, minMR, massPartType='stars', index=1))
                
                
                n_merg_2Gyr = u.num_mergers_n_step(sP, tree, n_steps,  minMR, massPartType='stars', index=1)
                had_a_merger_in_last_2Gyr.append(np.sum(n_merg_2Gyr))
                #print('# of mergers', np.sum(n_merg_2Gyr))
                '''
                numMergers = numMergers(tree,minMassRatio=1.0/10.0)
                print('number of mergers in previous step', numMergers)
                numMergers = numMergers(tree,minMassRatio=1.0/10.0, index = 1)
                print('number of mergers in previous step', numMergers)
                STOP
                '''
                counter+=1
            
            print('how is this looking?')
            print(selection)
            
            selection['SFR'] = sfr_mergers
            selection['num_mergers_in_last_2Gyr'] = had_a_merger_in_last_2Gyr
            selection['N'] = N_within_2MPC
            selection['r_1'] = r_1
            selection['r_2'] = r_2
            
            selection_enviro = selection.drop(columns=['Unnamed: 0', 'level_0','index'])
            print(selection_enviro)
            
            selection_enviro.to_csv('Tables/Aimee_SF/'+str(snapnum)+'/all_mergers_at_'+str(snap)+'_enviro.txt', sep = '\t')
            
        # Now, try to see if the non-merger table already exists:
        try:
            selection = pd.read_csv('Tables/Aimee_SF/'+str(snapnum)+'/nonmergers_matched_at_'+str(snap)+'_'+str(version)+'.txt', sep = '\t')
            
            print('            NONMERGERS ALREADY READY TO GO                 ', snap)
            continue
        except:
            print('            MAKING NONMERGERS                 ', snap)
            fields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                              'FirstProgenitorID', 'SubhaloMassType','DescendantID','SubfindID']
            # Get N, r_1, and r_2 for all of the non-mergers in the mass_select_all
            # starting with subhalos
            N_all = []
            r_1_all = []
            r_2_all = []
        
            had_a_merger_in_last_2Gyr = []
        
        
            minMR = 1.0/10.0
        
            for p in select_indices:
                # eliminate subhalos that have had a merger in the last 2 Gyr
                try:
                    #print('sP', sP, 'snap', snap, 'p', p)
                    tree = il.sublink.loadTree(sP,snap, p, fields = fields)#, onlyMPB=False)#       
                 
                    n_merg_2Gyr = u.num_mergers_n_step(sP, tree, n_steps, minMR, massPartType='stars', index=1)
                except AttributeError:
                    print('sP', sP, 'snap', snap, 'p', p)
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
            new_df = new_df_full
            print('length after deleting anything that has merged in the last 2 Gyr', len(new_df))
            
            # Now make these diagnostic tables of the distribution of parameters
            plt.clf()
            fig = plt.figure()
            ax0 = fig.add_subplot(211)
            ax0.hist(new_df['N'].values, label='Non-mergers', color = '#98B06F', alpha=0.5, bins=50)#, range=(0, 50))
            plt.legend()
            
            ax1 = fig.add_subplot(212)
            ax1.hist(selection['N'].values, label='Mergers', color = '#442B48',alpha=0.5, bins=50)#, range=(0, 50))
            #ax1.set_xlim([0,100])
            plt.legend()
            plt.savefig('Figs/Aimee_SF/hist_N_2Mpc_all_subhalos_'+str(snapnum)+'_center_'+str(snap)+'_'+str(version)+'.png')
            
            plt.clf()
            fig = plt.figure()
            ax0 = fig.add_subplot(211)
            ax0.hist(new_df['r_1'].values, label='Non-mergers', color = '#98B06F', alpha=0.5, bins=50)
            plt.legend()
            
            ax1 = fig.add_subplot(212)
            ax1.hist(selection['r_1'].values, label='Mergers', color = '#442B48', alpha=0.5, bins=50)
            #ax1.set_xlim([0,100])
            plt.legend()
            plt.savefig('Figs/Aimee_SF/hist_r_1_all_subhalos_'+str(snapnum)+'_center_'+str(snap)+'_'+str(version)+'.png')
            
            print('are any of these 0.0?',new_df['N'].values)
            new_df_elim_zeros = new_df[new_df['N'] != 0.0]
            print('len after eliminating 0.0 nonmerg = ', len(new_df_elim_zeros))
            print('len of merg before elim zeros', len(selection))
            print('len of merg after eliminating zeros', len(selection[selection['N'] != 0.0]))
            
            
            STOP
            
            # Then go through each merg_i and match with a non-merg i
            non_merger_table_index = []
            grows = []
            subids = []
            match_this = []
            for i in range(len(merg_i)):
                #print(selection[i])
                index, subid, ngrow = find_match(version,
                    selection['mass'].values[i], 
                    selection['r_1'].values[i], 
                    selection['r_2'].values[i], 
                    selection['N'].values[i], new_df)
                
                print('trying to match this', selection['mass'].values[i], 
                    selection['r_1'].values[i], 
                    selection['r_2'].values[i], 
                    selection['N'].values[i])
                print('index match', index)
                print('subid match', subid)
                print('ngrows', ngrow)
                print(new_df[new_df['SubhaloID']==subid])
                non_merger_table_index.append(index)
                grows.append(ngrow)
                subids.append(subid)
                match_this.append(merg_i[i])
                
                # drop the row so you don't double match it
                new_df = new_df[new_df['SubhaloID'] != subid]
                print('length of new_df after dropping', len(new_df))
                
                
            
            print('subids of matches', subids, len(subids))
            print('match this', match_this)
            print('grows', grows)
            print('dataframe of matches')
            
            matches = new_df_full[new_df_full.SubhaloID.isin(subids)]
            print('length of df_non_merg', matches)
            print(matches)
            
            # UGHHHHHHH now sort by the subids list
            new_df_full.SubhaloID = new_df_full.SubhaloID.astype("category")
            new_df_full.SubhaloID.cat.set_categories(subids,inplace=True)
            new_df_full = new_df_full.sort_values(["SubhaloID"])
            # now delete all NaN
            new_df_full.dropna(subset = ["SubhaloID"], inplace=True)
            #new_df_full = new_df_full[new_df_full['SubhaloID']]
            
            
            #df['price'].isin(['100', '221'])
            
            
            print('matches with deets')
            new_df_full['matched_to_merger_subid'] = match_this
            new_df_full['ngrows'] = grows
            
            print(new_df_full)
            new_df_full.to_csv('Tables/Aimee_SF/'+str(snapnum)+'/nonmergers_matched_at_'+str(snap)+'_'+str(version)+'.txt', sep = '\t')
                
            
            # make a new table that has just the IDs that were selected
            
       
    
        
        
    
