import pandas as pd
import os
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt: 
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

path_to_tables = '/n/holystore01/LABS/hernquist_lab/Users/rnevin/AimeeTNG50project/Tables/'
path_to_tables = '/n/home03/rnevin/TNGstuff/Tables/Aimee_SF/'
version = 'r_1_enviro'
# Lists of different versions of matching:
#'enviro'
#'no_enviro'
#'no_enviro_tight'
#'N_enviro'
#'N_enviro_tight'

#print(os.listdir(path_to_tables))

redshift_center = [17]
snap_list = [13,14,15,16]

redshift_center = [84]
snap_list = [83,84,85]

redshift_center = [50]
snap_list = [49,50,51]

merger_list = []
for snap in snap_list:
    # Sorry, this might be confusing, given the naming scheme for the nonmerger matched file, 
    # but all of the mergers have this _enviro at the end of the name, which just means they have environmental information
    
    merger_list.append(pd.read_csv(path_to_tables+str(redshift_center[0])+'/all_mergers_at_'+str(snap)+'_enviro.txt', sep = '\t'))

all_merger = pd.concat(merger_list)

# Optionally restrict by mass
#all_merger = all_merger[all_merger['mass'] > 1e10]

#print('mergers', all_merger)       

nonmerger_list = []
for snap in snap_list:
    if version =='':
        nonmerger_list.append(pd.read_csv(path_to_tables+str(redshift_center[0])+'/nonmergers_matched_at_'+str(snap)+'.txt', sep = '\t'))
    else:
        nonmerger_list.append(pd.read_csv(path_to_tables+str(redshift_center[0])+'/nonmergers_matched_at_'+str(snap)+'_'+str(version)+'.txt', sep = '\t'))

all_nonmerger = pd.concat(nonmerger_list)

# Optionally restrict by mass
#all_nonmerger = all_nonmerger[all_nonmerger['Mass'] > 1e10]

#print('nonmergers', all_nonmerger)     

#STOP

# Is there a way to make a histogram of the number of grows?
# What about a plot of that versus the difference in dex from the matched merging example?

mass_diff = []
sfr_diff = []
ssfr_diff = []
rel_diff_mass = []

for i in range(len(all_nonmerger)):# go through and find the mass difference
    mass_non = all_nonmerger['Mass'].values[i]
    sfr_non = all_nonmerger['SFR'].values[i]
    match = all_nonmerger['matched_to_merger_subid'].values[i]
    mass_merg = all_merger[all_merger['Subfind_ID']==match]
    
    mass_diff.append(mass_non - mass_merg['mass'].values[0])
    rel_diff_mass.append((mass_non - mass_merg['mass'].values[0])/mass_merg['mass'].values[0])
    sfr_diff.append(sfr_non - mass_merg['SFR'].values[0])
    ssfr_diff.append(1e10*(sfr_non/mass_non - mass_merg['SFR'].values[0]/mass_merg['mass'].values[0]))
    

# Make a tri-panel plot that compares mass, N, and r_1 for the mergers and non-mergers:
color_n = '#FED766'
color_m = '#ED254E'

n_bins = 15
plt.clf()
fig = plt.figure(figsize = (10,7))
ax0 = fig.add_subplot(311)


bins=np.histogram(np.hstack((all_nonmerger['Mass'].values,all_merger['mass'].values)), bins=30)[1] #get the bin edges
MIN, MAX = bins[0], bins[-1]


#plt.gca().set_xscale("log")
ax0.set_xscale("log")
ax0.hist(all_nonmerger['Mass'].values, label='Nonmergers', color=color_n, alpha=0.7, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), n_bins))
ax0.hist(all_merger['mass'].values, label='Mergers', color=color_m, alpha=0.7, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), n_bins))
plt.legend()
#plt.xscale('log')
#plt.yscale('log')
#ax0.set_xlabel(r'log (M$_*$)')
ax0.annotate(r'M$_*$', xy=(0.4,0.9), xycoords='axes fraction')

#plt.title(str(redshift_center[0])+'_'+str(version))
plt.title('Snap = '+str(redshift_center[0])+', version = '+str(version)+', # grows = '+str(round(np.mean(all_nonmerger['ngrows'].values),3))+', # gal = # control = '+str(len(all_nonmerger)))
ax1 = fig.add_subplot(313)

N_nonmerg = [(1+x) for x in all_nonmerger['N'].values]
N_merg = [(1+x) for x in all_merger['N'].values]
bins=np.histogram(np.hstack((N_nonmerg,N_merg)), bins=30)[1] #get the bin edges
MIN, MAX = bins[0], bins[-1]


#ax1.gca().set_xscale("log")
ax1.hist(N_nonmerg, label='Nonmergers', color=color_n, alpha=0.7, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), n_bins))
ax1.hist(N_merg, label='Mergers', color=color_m, alpha=0.7, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), n_bins))
ax1.annotate(r'N+1', xy=(0.4,0.9), xycoords='axes fraction')
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
#ax1.set_xlabel('log (N+1)')
ax1.set_xscale("log")


ax2 = fig.add_subplot(312)
# Deal with the r_1 values that are 0.0:


bins=np.histogram(np.hstack((np.ma.masked_where(all_nonmerger['r_1'].values==0,all_nonmerger['r_1'].values),np.ma.masked_where(all_merger['r_1'].values==0,all_merger['r_1'].values))), bins=30)[1] #get the bin edges
MIN, MAX = bins[0]+0.01, bins[-1]+0.01

print('bins rs', bins)


#ax1.gca().set_xscale("log")
ax2.hist(all_nonmerger['r_1'].values, label='Nonmergers', color=color_n, alpha=0.7, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), n_bins))
ax2.hist(all_merger['r_1'].values, label='Mergers', color=color_m, alpha=0.7, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), n_bins))
ax2.set_xscale("log")
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
#ax2.set_xlabel('log ($r_1$)')
ax2.annotate(r'$r_1$', xy=(0.4,0.9), xycoords='axes fraction')



plt.savefig('Figs/Aimee_SF/tripanel_'+str(redshift_center[0])+'_'+str(version)+'.png')

STOP



plt.clf()
plt.scatter(sfr_diff, mass_diff,  c = all_nonmerger['ngrows'].values)
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('sfr matched nonmerg - merger')
plt.ylabel('mass matched nonmerger - merger')
plt.colorbar(label = '# of 0.1 dex grows on stellar mass')
plt.title('Snap = '+str(redshift_center[0])+', version = '+str(version)+', # grows = '+str(np.mean(all_nonmerger['ngrows'].values)))
plt.savefig('Figs/Aimee_SF/diff_mass_SFR_'+str(redshift_center[0])+'_'+str(version)+'.png')

mass_diff_log = []


for x in mass_diff:
    if x > 0:
        mass_diff_log.append(math.log10(x))
    else:
        mass_diff_log.append(-math.log10(abs(x)))
#print('total number = ', len(mass_diff_log))

gt_0_elements = [element for element in mass_diff if element > 0]
n_gt_0 = len(gt_0_elements)

lt_0_elements = [element for element in mass_diff if element < 0]
n_lt_0 = len(lt_0_elements)

#print('# gt zero', n_gt_0)
#print('# lt zero', n_lt_0)




plt.clf()
plt.scatter(ssfr_diff, mass_diff_log,  c = all_nonmerger['ngrows'].values, s=1)
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('ssfr matched nonmerg - merger [/10^10]')
plt.ylabel('log mass matched nonmerger - merger')
plt.annotate('# > 0 = '+str(n_gt_0), xy=(0.02,0.95), xycoords='axes fraction')
plt.annotate('# < 0 = '+str(n_lt_0), xy=(0.02,0.05), xycoords='axes fraction')
plt.colorbar(label = '# of 0.1 dex grows on stellar mass')
plt.title(str(redshift_center[0])+'_'+str(version))
plt.savefig('Figs/Aimee_SF/diff_mass_sSFR_'+str(redshift_center[0])+'_'+str(version)+'.png')

plt.clf()
plt.scatter(ssfr_diff, rel_diff_mass,  c = all_nonmerger['ngrows'].values, s=2)
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('ssfr matched nonmerg - merger [/10^10]')
plt.ylabel('relative mass matched nonmerger - merger / merger')
plt.annotate('# > 0 = '+str(n_gt_0), xy=(0.02,0.95), xycoords='axes fraction')
plt.annotate('# < 0 = '+str(n_lt_0), xy=(0.02,0.05), xycoords='axes fraction')
plt.colorbar(label = '# of 0.1 dex grows on stellar mass')
plt.title(str(redshift_center[0])+'_'+str(version))
plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')
#plt.ylim([-10,30])
plt.savefig('Figs/Aimee_SF/diff_rel_mass_sSFR_'+str(redshift_center[0])+'_'+str(version)+'.png')



# Okay lets start looking at interesting properties
# starting with SFR versus mass
plt.clf()

mass_non = [math.log10(x) for x in all_nonmerger['Mass'].values]
sfr_non = [math.log10(x) for x in all_nonmerger['SFR'].values]

mass_merg = [math.log10(x) for x in all_merger['mass'].values]
sfr_merg = [math.log10(x) for x in all_merger['SFR'].values]

plt.scatter(mass_non, sfr_non, label='Nonmergers', color='#0F7173', alpha=0.7)
plt.scatter(mass_merg, sfr_merg, label='Mergers', color='#EF8354', alpha=0.7)
plt.legend()
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('log stellar mass')
plt.ylabel('log SFR')
plt.savefig('Figs/Aimee_SF/mass_SFR_'+str(redshift_center[0])+'_'+str(version)+'.png')

plt.clf()

bins=np.histogram(np.hstack((all_nonmerger['Mass'].values,all_merger['mass'].values)), bins=30)[1] #get the bin edges

print('bins', bins, bins[0], bins[-1])
MIN, MAX = bins[0], bins[-1]


plt.gca().set_xscale("log")
plt.hist(all_nonmerger['Mass'].values, label='Nonmergers', color='#0F7173', alpha=0.7, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50))
plt.hist(all_merger['mass'].values, label='Mergers', color='#EF8354', alpha=0.7, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50))
plt.legend()
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('log stellar mass')
plt.savefig('Figs/Aimee_SF/mass_'+str(redshift_center[0])+'_'+str(version)+'.png')

#sns.jointplot(data=all, x="bill_length_mm", y="bill_depth_mm")
plt.clf()
plt.scatter(all_nonmerger['r_1'].values, all_nonmerger['r_2'].values, label='Nonmergers', color='#0F7173', alpha=0.7)
plt.scatter(all_merger['r_1'].values, all_merger['r_2'].values, label='Mergers', color='#EF8354', alpha=0.7)
plt.legend()
plt.xlabel('r_1')
plt.ylabel('r_2')
plt.savefig('Figs/Aimee_SF/rs_'+str(redshift_center[0])+'_'+str(version)+'.png')


