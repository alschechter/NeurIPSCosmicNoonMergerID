# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Accompanys the build_merger_catalog.py code
# Takes the output from this ^ and joins them together
# centered on the bin center
# into one table called mergers_nodup_.....
# Also builds a matched sample of nonmergers in a 
# separate table (matched for redshift and SFR and mass
# We should probably look into matching for gas fraction
# as well? This is a TO DO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import illustris_python as il
import numpy as np
import matplotlib
import random
matplotlib.use('agg')# must call this before importing pyplot as plt: 
import matplotlib.pyplot as plt
import seaborn as sns            

basePath = '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L35n2160TNG'                                                                                                               
sP = basePath+run+'/output/'

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

snapnum = 72
all_prog = False # This is an option to include the tables that have all possible progenitors (i.e., two snapshots before)

subhalosh = il.groupcat.loadHeader(basePath+run+'/output/',snapnum)#, fields=fields_header)
z = subhalosh['Redshift']
redshift_label = z


# Step 1: import all of your data for whatever is the bin center:
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
print('len before cutting duplicates', len(all_mergers))

df_dup = all_mergers[all_mergers.duplicated(subset=['Subfind_ID'], keep=False)]
df_no_dup = all_mergers[~all_mergers.duplicated(subset=['Subfind_ID'], keep='first')]
df_no_dup = df_no_dup.reset_index()
print('len after cutting the extra duplicates', len(df_no_dup))

# Now, get info about all subhalos to compare the merger population
if run=='L75n1820TNG':
    fields = ['SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']
else:
    fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType','SubhaloCM']#was \['SubhaloMassInRad']                                                                     
h = 0.6774
minmass = 10**3*(5.7*10**4)/h # 1000 times the mass resolution
ll = minmass
maxmass = 10**13

# Get all subhalos at the bin center
subhalos = il.groupcat.loadSubhalos(basePath+run+'/output/',snapnum, fields=fields)
mass_msun = subhalos['SubhaloMassType'][:,4] * 1e10 / h
sfr = subhalos['SubhaloSFRinRad']
select_indices = np.where(np.logical_and(mass_msun > minmass, mass_msun < maxmass))[0]
select_mass = mass_msun[select_indices]
select_sfr = sfr[select_indices]



# Now, try to get rid of all progenitors that are within a certain distance of each other
# To see if there are repeats, assign a tolerance distance, maybe 30 kpc?
repeat = []

for j in range(len(df_no_dup)):
    subfind_id_here = df_no_dup['Subfind_ID'].values[j]
    #print('subfindid', subfind_id_here)
    pos_1 = subhalos['SubhaloCM'][subfind_id_here]
    counter = 0
    # so for this second for loop you're going to work down only
    for k in range(len(df_no_dup)-j-1):
        
        i = k+j+1 # always start one after
        #print('j', j, 'k', k, 'i', i)
        # Check if x y z are within a 10kpc distance
        subfind_id_sec = df_no_dup['Subfind_ID'].values[i]
        pos_2 = subhalos['SubhaloCM'][subfind_id_sec]
        sep = np.sqrt((pos_1[0]-pos_2[0])**2+(pos_1[1]-pos_2[1])**2+(pos_1[2]-pos_2[2])**2)
        if sep < 10:#*(1+z)**(-1):
            # perhaps there's an option to drop as you go?
            # drop the one that's the repeat 
            
            
            repeat.append(i)
            
        else:
            counter+=1
    

    
#print('length galaxies on top of each other', len(repeat))
#print(np.sort(repeat))
try:
    drop_repeats = df_no_dup.drop(df_no_dup.index[np.sort(repeat)])
except IndexError:# which means repeat is empty
    drop_repeats = df_no_dup
#print('length dropping galaxies overlapping', len(drop_repeats))


# Also drop all mergers that have mass ==0.0:
dropped_reps = drop_repeats[drop_repeats.mass > ll/10]# a tenth of the lower limit mass
drop_cols = dropped_reps.drop(columns=['level_0','index'],axis=1)

# Okay remove all of these weird columns
drop_cols = drop_cols.reset_index()


merg_i = drop_cols['Subfind_ID']
# Also drop all mergers that have sfr ==0.0

mass_mergers = mass_msun[merg_i]
sfr_mergers = sfr[merg_i]

drop_sfr = drop_cols.drop(drop_cols.index[np.where(sfr_mergers==0.0)[0]])
dropped = drop_sfr
merg_i = dropped['Subfind_ID']



if all_prog:
    file_out = open('Tables/mergers_all_nodup_all_prog_'+str(snapnum)+'_'+run+'.txt','w')
else:
    file_out = open('Tables/mergers_all_nodup_'+str(snapnum)+'_'+run+'.txt','w')
file_out.write('Subfind_ID'+'\t'+'t-t_merg'+'\t'+'q'+'\t'+'Type'+'\t'+'mass'+'\n')
for j in range(len(merg_i)):
    file_out.write(str(dropped['Subfind_ID'].values[j])+'\t'+str(dropped['t-t_merg'].values[j])+'\t'+
    str(dropped['q'].values[j])+'\t'+str(dropped['Type'].values[j])+'\t'+str(dropped['mass'].values[j])+'\n')
file_out.close()

print('MADE merger table')


# Make a selection of things that are the mergers:
mass_mergers = mass_msun[merg_i]
sfr_mergers = sfr[merg_i]


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
mass_bins = [[ll/10,ll],[ll,ll*10],[ll*10,ll*100],[ll*100,ll*1000],[ll*1000,ll*10000],[ll*10000,ll*100000]]#,[ll*1e4,ll*1e5]]
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
                print('removing this element', element)
                #was .remove(element)
            else:
                select_indices_norep.append(element)
        # Now, count how many mergers are in this mass range:
        nmergers_bin = len(np.where(np.logical_and(np.logical_and(mass_mergers > minmass, mass_mergers < maxmass), 
            np.logical_and(sfr_mergers > minsfr, sfr_mergers < maxsfr)))[0])
        print('binmin mass ', np.log10(minmass), 'binmax ', np.log10(maxmass), 'sfr min', minsfr, 'max', maxsfr)
        
        if nmergers_bin > select_indices_norep: # so this is not the problem
            print('there are more mergers than all nonmerging galaxies at this bin')
            print('nonmerg', len(select_indices_norep), 'merg', nmergers_bin)
            STOP
            
        indices = random.sample(select_indices_norep, nmergers_bin)
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
        


plt.clf()
logbins = np.logspace(np.log10(10**6),np.log10(10**12),19)
plt.hist(mass_mergers, bins=logbins, label='Mergers mass')
plt.hist(nonmerg_mass, bins=logbins, label='Nonmergers mass', alpha=0.5)
plt.xscale('log')
plt.legend()

#plt.hist(mass_mergers, label='Mergers mass')
#plt.hist(nonmerg_mass, label='Nonmergers mass')
#plt.xscale('log')
if all_prog:
    plt.savefig('Figs/hist_matched_mass_snapnum_all_prog_'+str(snapnum)+'.png')

else:
    plt.savefig('Figs/hist_matched_mass_snapnum_'+str(snapnum)+'.png')


plt.clf()
logbins = np.logspace(np.log10(10**-6),np.log10(10**2),19)
plt.hist(sfr_mergers, bins=logbins, label='Mergers sfr')
plt.hist(nonmerg_sfr, bins=logbins, label='Nonmergers sfr', alpha=0.5)
plt.xscale('log')
plt.legend()

#plt.hist(mass_mergers, label='Mergers mass')
#plt.hist(nonmerg_mass, label='Nonmergers mass')
#plt.xscale('log')
if all_prog:
    plt.savefig('Figs/hist_matched_sfr_snapnum_all_prog_'+str(snapnum)+'.png')
else:
    plt.savefig('Figs/hist_matched_sfr_snapnum_'+str(snapnum)+'.png')

print('merger subid', merg_i)
print(nonmerg_subfindid)
print(nonmerg_mass)


if all_prog:
    file_out = open('Tables/nonmergers_snap_all_prog_'+str(snapnum)+'_'+run+'.txt','w')
else:
    file_out = open('Tables/nonmergers_snap_'+str(snapnum)+'_'+run+'.txt','w')
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
    plt.savefig('Figs/selection_subhalos_merg_gif_all_prog_'+str(run)+'_snap_'+str(snapnum)+'.png')
else:
    plt.savefig('Figs/selection_subhalos_merg_gif_'+str(run)+'_snap_'+str(snapnum)+'.png')

