#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Creates catalogs of mergers centered at a given redshift
# Does this by first selecting from subhalos as
# a function of stellar mass and then loads
# the merger tree for each subhalo, determines if the 
# subhalo mergees and saves its subfindID, stellar mass,
# and mass ratio.

# The trick here is that I've also expanded this catalog
# to either forward or backward trace merger trees to
# the nearest full snapshot.
# For instance, at z=1, snapshot 50 is a full snapshot
# but snapshot 49 is not full, so centering at 49, this 
# code determins which subhalos are merging and then 
# identifies the descendants of these galaxies at snapshot
# 50, which is a full snapshot for which we can run SKIRT.

# Also contains options for printing the lookback time
# and redshift of a given snapshot, which is handy.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import illustris_python as il
import h5py
import numpy as np
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt:                                                                                    
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15

# This was the old basepath
basePath = '/n/hernquistfs3/IllustrisTNG/Runs/'

# Now the new place where illustris lives:
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

snapnum = 77
snapnum_full = 78

all_prog = False

lbt_snap = 0
lbt_bin_center = 0


# Make a dictionary for these:
lbt_dict = {'13':12.86836771,'14':12.83476842,'15':12.76390407,'16':12.68785986,'17':12.62304621,'18':12.51872105,'19':12.43449791,'20':12.33446648,
            '23':11.98881427,'24':11.85662123,'25':11.65551451,'26':11.56321698,'27':11.41746261,# once we get back to z= 0.3, progenitors can come from two time steps previously
            '31':10.8208574,'32':10.67239916,'33':10.51690585,'34':10.35409873,'35':10.20851499,
            '38':9.76415887,'39':9.59552818,'40':9.50880841,'41':9.30007834,'42':9.14539478,
            '48':8.22539306,'49':8.076,'50':7.924,'51':7.728,
            '58':6.67044646,'59':6.4881983,'60':6.34915242,'61':6.16060764,
            '66':5.37045644,'67':5.21592302,'68':5.05938984,
            '70':4.74036065,'71':4.57788602,'72':4.41345434,'73':4.24707941,'74':4.07877622,
            '76':3.79403028,'77':3.62066719,'78':3.50405532,'79':3.26837091,'80':3.14931086,
            '82':2.78733837,'83':2.6651,'84':2.4803,'85':2.2937
            }
            
redshift_dict = {'13':6.0107573988449,'14':5.846613747881867,'15':5.5297658079491026,'18':4.664517702470927,'19':4.428033736605549,'20':4.176834914726472,
                '24':3.2830330579565246,'27':2.7331426173187188,
                '31':2.207925472383703,'32':2.1032696525957713,'33':2.0020281392528516,'34':1.9040895435327672,'35':1.822689252620354,
                '39':1.5312390291576135,'41':1.4140982203725216,
                '48':1.074457894547674,'49':1.035510445664141,'51':0.9505313515850327,
                '58':0.7326361820223115,'59':0.7001063537185233,'60':0.6761104112134777,'61':0.644641840684537,
                '66':0.524565820433923,'67':0.5030475232448832,'68':0.4818329434209512,
                '70':0.4402978492477432,'71':0.41996894199726653,'72':0.3999269646135635,'73':0.38016786726023866,'74':0.36068765726181673,
                '76':0.32882972420595435,'77':0.31007412012783386,'78':0.2977176845174465,'79':0.2733533465784399,'80':0.2613432561610123,
                '82':0.22598838626019768,'83':0.21442503551449454,'85':0.1803852617057493
}
try:
    lbt_snap = lbt_dict[str(snapnum)]
    lbt_bin_center = lbt_dict[str(snapnum_full)]
except KeyError:
    print('not in dict yet')
    
if abs(lbt_snap - lbt_bin_center) > 0.25:# 250 Myr
    print('not within 250 Myr of bin center', lbt_snap, lbt_bin_center)
    STOP
print('difference in lookback time', lbt_bin_center - lbt_snap)
# Just for my own notes about which are included in the 250Myr window 
# (500 Myr centered on z=1)
treeName_default="Sublink"

print('look back time', lbt_snap)
# This function uses the groupcat.loadSubhalos function to grab all the IDs and details
# of galaxies within a certain range of stellar masses at a snapshot

# As a note, I have min_npart as an optional input here but Jacob was saying that its
# not necessary to select by # of particles; this has a nearly 1:1 relationship with
# selecting by stellar mass.
def get_subhalos(run, snapnum, basePath, minmass, maxmass, min_npart, h, plot):
    
    fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType']#was ['SubhaloMassInRad']

    # It might actually be useful to sort this by mass, so you can snag all subhalos between a certain mass range:                                      

    if run=='L35n2160TNG':#run = 'L75n1820TNG'#'L35n2160TNG'  
        subhalos = il.groupcat.loadSubhalos(basePath+run+'/output/',snapnum, fields=fields)# 
        subhalosh = il.groupcat.loadHeader(basePath+run+'/output/',snapnum)#, fields=fields_header)
        print('z', subhalosh['Redshift'], 'a', subhalosh['Time'])
        
        # Set up the proper cosmology to calculate lookback time
        # Get astropy from here: 
        # module --ignore-cache load "astropy-0.4.2-fasrc01"
        
        # Planck Collab 2015
        print(Planck15.Om0, Planck15.H0)
        Om0 = subhalosh['Omega0']
        Ob0 = subhalosh['OmegaLambda']
        print(Om0, Ob0)
        
        #cosmo = Planck15(H0=67.74, Om0=Om0, Ob0=0.05)
        print('lookback time', Planck15.lookback_time(subhalosh['Redshift']))
        #STOP
        if lbt_snap==0:
            STOP
        

        # Put the stellar mass into physical units (getting rid of the h)
        mass_msun = subhalos['SubhaloMassType'][:,4] * 1e10 / h
        #subhalos['SubhaloMass'] #* 1e10 / 0.704                                                                                                  
        #sorted_mass = np.array(sorted(mass_msun[subhalos['SubhaloFlag']]))
        #sorted_indices = np.argsort(mass_msun[subhalos['SubhaloFlag']])
        #sorted_sfr = subhalos['SubhaloSFRinRad'][sorted_indices]
        #sorted_stellar_part = subhalos['SubhaloLenType'][:,4][sorted_indices]
        sfr = subhalos['SubhaloSFRinRad']
        stellar_part = subhalos['SubhaloLenType'][:,4]
    else:
        fields = ['SubhaloMassType','SubhaloSFRinRad', 'SubhaloLenType']
        # so the question is if the group catalogs are broken overall or if they just have slightly different fields
        subhalos = il.groupcat.loadSubhalos(basePath+run+'/output/',snapnum, fields=fields)
    
        mass_msun = subhalos['SubhaloMassType'][:,4] * 1e10 / h                                                                                  
        #sorted_mass = np.array(sorted(mass_msun))
        #sorted_indices = np.argsort(mass_msun)
        #sorted_sfr = subhalos['SubhaloSFRinRad'][sorted_indices]
        #sorted_stellar_part = subhalos['SubhaloLenType'][:,4][sorted_indices]
        sfr = subhalos['SubhaloSFRinRad']
        stellar_part = subhalos['SubhaloLenType'][:,4]
    # It might actually be useful to sort this by mass, so you can snag all subhalos between a certain mass range:
    
    # Now select all subhalos between your chosen range of total masses
    # and optionally make a plot to see what this looks like
    #dists[(np.where((dists >= r) & (dists <= r + dr)))]
    
    select_indices = np.where(np.logical_and(mass_msun > minmass, mass_msun < maxmass))[0]
    select_mass = mass_msun[select_indices]
    select_sfr = sfr[select_indices]
    select_stellar_part = stellar_part[select_indices]


    '''select_1_mass = sorted_mass[(sorted_mass < maxmass) & (sorted_mass > minmass)]
    select_1_index = np.where((sorted_mass < maxmass) & (sorted_mass > minmass))
    select_1_ids = sorted_indices[select_1_index]
    select_1_sfr = sorted_sfr[select_1_index]
    select_1_stellar_part = sorted_stellar_part[select_1_index]

    # Additionally, select only subhalos that have more particles than the minimum number of stellar particles:
    select_stellar_part = select_1_stellar_part[select_1_stellar_part > min_npart]
    select_index = np.where(select_1_stellar_part > min_npart)
    select_ids = select_1_ids[select_index]
    select_sfr = select_1_sfr[select_index]
    select_mass = select_1_mass[select_index]
    '''
    if len(select_stellar_part) != len(select_mass):
        print('Lengths are somehow messed up ', len(select_stellar_part), len(select_mass))
        STOP

    

    if plot=='yes':
        print('length of selection', len(select_indices))
        plt.clf()
        plt.plot(mass_msun, sfr, '.', color='black', label = 'All Subhalos')
        plt.plot(select_mass, select_sfr,'.', color='orange', label='Mass Selection')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Stellar Mass [$M_\odot$]')
        plt.ylabel('Star Formation Rate [$M_\odot / yr$]')
        plt.savefig('Figs/selection_subhalos_'+str(run)+'_snap_'+str(snapnum)+'.png')

        plt.clf()
        plt.plot(mass_msun, stellar_part, '.', color='black', label = 'All Subhalos')
        plt.plot(select_mass, select_stellar_part,'.', color='orange', label='Mass Selection')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Stellar Mass [$M_\odot$]')
        plt.ylabel('# Stellar Particles')
        plt.savefig('Figs/selection_subhalos_npart_'+str(run)+'_snap_'+str(snapnum)+'.png')

    
  
    return select_indices, select_mass, select_stellar_part, mass_msun



# This function loads up the merger tree for each subhalo 
# [THIS CANNOT BE THE MOST EFFECTIVE WAY TO DO THIS.]
# and determines if there is a merger currently happening by
# finding if the first progenitor has a next progenitor
# (remember that the pointer for next prog actually comes
# from the first progenitor.)
def mergers_one_step(tree, minMR, massPartType='stars', index=1):
    
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','DescendantID']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    invMassRatio = 1.0 / minMR

    # this is where it gets confusing. There's a difference between 'subhaloID'
    # and 'subfindID', subhaloID is the identifier within the tree whereas
    # subfind ID can be used, i.e., looking up a galaxy directly in the group
    # catalog. 
    
    # Right now, we are at index zero, in other words looking at a specific subhalo
    # that we fed in.
    
    rootID = tree['SubhaloID'][index]
    sfID = tree['SubfindID'][index]
    fpID   = tree['FirstProgenitorID'][index]
    descID   = tree['DescendantID'][index]
    descsfID = tree['SubfindID'][0]
    
    '''
    print('subhalo ID', rootID, 'fpID', fpID, 'descedant', descID)
    # I'm still not sure I'm getting the correct descendant info,
    # need to make sure this is extending forward in time too
    print('diff between descendant and current', descID - rootID)
    
    print('does this work, descsfid', descsfID, 'main gal sfid', sfID, 'first prog sfid', tree['SubfindID'][2])
    
    print('subfindIDs', tree['SubfindID'])



    print('subhaloids', tree['SubhaloID'])
    print('first prog', tree['FirstProgenitorID'])
    print('next prog', tree['NextProgenitorID'])
    print('desc', tree['DescendantID'])
    '''

    # first, walk back one step to the first progenitor
    if fpID != -1: #it will be -1 if there is no first progenitor
        # Now walk back along the tree to the fpIndex, this should be 
        # 0 + 1 = 1 since we're only taking one step backwards in time
        fpIndex = index + (fpID - rootID)
        
        # Get the mass of the first progenitor
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)
        
        # Now get the index of the descendant of the merger
        descIndex = index - 1
        
        descsfID = tree['SubfindID'][descIndex]
        fpsfID = tree['SubfindID'][fpIndex]
        
        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        if npID != -1:
            npIndex = index + (npID - rootID)
            npsfID = tree['SubfindID'][npIndex]
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)

            #print('there is a NP!', npMass, fpMass)
            #STOP
            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass
                #print('mass ratio', ratio)
                if ratio >= minMR and ratio <= invMassRatio:
                    
                    return 1, [sfID, descsfID, fpsfID, npsfID], ratio
                else:
                    return 0, [sfID, descsfID, fpsfID, npsfID], ratio
            else:
                return 0, 0, 0

        else:
            return 0, 0, 0

def mergers_two_steps(tree, minMR, subid_in, fields, type, massPartType='stars', index=1):
    # It's important to know here how the trees work:
    # Here's the link to the Illustris page on this: https://www.tng-project.org/data/docs/specifications/#sec4a
    # Here's the link to the Lemson & Springel 2006 paper: http://articles.adsabs.harvard.edu/pdf/2006ASPC..351..212L
    
    #
    
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','DescendantID']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    invMassRatio = 1.0 / minMR

    # this is where it gets confusing. There's a difference between 'subhaloID'
    # and 'subfindID', subhaloID is the identifier within the tree whereas
    # subfind ID can be used, i.e., looking up a galaxy directly in the group
    # catalog. 
    
    # Right now, we are at index zero, in other words looking at a specific subhalo
    # that we fed in.
    
    rootID = tree['SubhaloID'][index]
    sfID = tree['SubfindID'][index]
    fpID   = tree['FirstProgenitorID'][index]
    descID   = tree['DescendantID'][index]
    descsfID = tree['SubfindID'][0]
    
    
    
    
    
    '''
    print('subhalo ID', rootID, 'fpID', fpID, 'descedant', descID)
    # I'm still not sure I'm getting the correct descendant info,
    # need to make sure this is extending forward in time too
    print('diff between descendant and current', descID - rootID)
    
    print('does this work, descsfid', descsfID, 'main gal sfid', sfID, 'first prog sfid', tree['SubfindID'][2])
    
    print('subfindIDs', tree['SubfindID'])



    print('subhaloids', tree['SubhaloID'])
    print('first prog', tree['FirstProgenitorID'])
    print('next prog', tree['NextProgenitorID'])
    print('desc', tree['DescendantID'])
    '''

    # first, walk back one step to the first progenitor
    if fpID != -1: #it will be -1 if there is no first progenitor
        # Now walk back along the tree to the fpIndex, this should be 
        # 0 + 1 = 1 since we're only taking one step backwards in time
        fpIndex = index + (fpID - rootID)
        
        # Get the mass of the first progenitor
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)
        
        # Now get the index of the descendant of the merger
        descIndex = index - 1
        
        descsfID = tree['SubfindID'][descIndex]
        
        if type=='Descendant':
            # Okay but you need to get the descendant of the descendant, which requires loading a new tree:
            tree_desc = il.sublink.loadTree(sP,tree['SnapNum'][0], descsfID,fields = fields)#
            descsfID = tree_desc['SubfindID'][0]
        else:
            descsfID = tree['SubfindID'][descIndex]
        fpsfID = tree['SubfindID'][fpIndex]
        
        # explore breadth
        # so nextprogenitors point laterally, so fpIndex is usually 2, meaning one above the starting
        # so points laterally from the first progenitor to find if there are two progenitors
        npID = tree['NextProgenitorID'][fpIndex]

        if npID != -1:# This identifies mergers
            '''
            print('npID', npID, 'fpIndex', fpIndex)
        
            print('subid in', subid_in)
            print('index', index)
            print('tree is loaded')
            print('rootID', rootID)
            print('subfindID', sfID)
            print('fpID', fpID)
            print('descID', descID)
            print('descsfID', descsfID)
            
            print('length of tree', len(tree))
            
            print('printing the entire tree of subhalo IDs', len(tree['SubhaloID']),  tree['SubhaloID'][0:10])
            print('printing then entire tree of sufind IDs', len(tree['SubfindID']),  tree['SubfindID'][0:10])
            print('descendantID, subhaloID', len(tree['DescendantID']),  tree['DescendantID'][0:10])
            print('firstprogID, subhaloID', len(tree['FirstProgenitorID']),tree['FirstProgenitorID'][0:10])
            print('nextprogID', len(tree['NextProgenitorID']),  tree['NextProgenitorID'][0:10])
            print('SnapNum',  tree['SnapNum'][0:10])
            
            '''
            npIndex = index + (npID - rootID)
            #print('npIndex', npIndex)
            if type=='Progenitor':
                npsfID = tree['SubfindID'][npIndex+1]# add one to take one more step back in time to get first prog of the next prog
                #print('SFID np', npsfID)
                npMass  = il.sublink.maxPastMass(tree, npIndex+1, massPartType)
                #print('new fpsfID', tree['SubfindID'][fpIndex+1])
                fpsfID = tree['SubfindID'][fpIndex+1]
            else:
                npsfID = tree['SubfindID'][npIndex]# add one to take one more step back in time to get first prog of the next prog
                #print('SFID np', npsfID)
                npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)
                #print('new fpsfID', tree['SubfindID'][fpIndex+1])
                fpsfID = tree['SubfindID'][fpIndex]
            
            # See if you can grab all other progenitors :)
            #print('where',tree['SubfindID'][np.where(tree['SnapNum']==17)], len(tree['SubfindID'][np.where(tree['SnapNum']==17)]))
            #STOP
            #print('there is a NP!', npMass, fpMass)
            #STOP
            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass
                #print('mass ratio', ratio)
                if ratio >= minMR and ratio <= invMassRatio:
                    print(sfID, descsfID, fpsfID, npsfID)
                    return 1, [sfID, descsfID, fpsfID, npsfID], ratio
                else:
                    return 0, [sfID, descsfID, fpsfID, npsfID], ratio
            else:
                return 0, 0, 0

        else:
            return 0, 0, 0
            
def mergers_two_steps_all_prog(tree, minMR, subid_in, fields, type, massPartType='stars', index=1):
    # It's important to know here how the trees work:
    # Here's the link to the Illustris page on this: https://www.tng-project.org/data/docs/specifications/#sec4a
    # Here's the link to the Lemson & Springel 2006 paper: http://articles.adsabs.harvard.edu/pdf/2006ASPC..351..212L
    
    #
    
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','DescendantID']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    invMassRatio = 1.0 / minMR

    # this is where it gets confusing. There's a difference between 'subhaloID'
    # and 'subfindID', subhaloID is the identifier within the tree whereas
    # subfind ID can be used, i.e., looking up a galaxy directly in the group
    # catalog. 
    
    # Right now, we are at index zero, in other words looking at a specific subhalo
    # that we fed in.
    
    rootID = tree['SubhaloID'][index]
    sfID = tree['SubfindID'][index]
    fpID   = tree['FirstProgenitorID'][index]
    descID   = tree['DescendantID'][index]
    descsfID = tree['SubfindID'][0]
    
    
    
    
    
    

    # first, walk back one step to the first progenitor
    if fpID != -1: #it will be -1 if there is no first progenitor
        # Now walk back along the tree to the fpIndex, this should be 
        # 0 + 1 = 1 since we're only taking one step backwards in time
        fpIndex = index + (fpID - rootID)
        
        # Get the mass of the first progenitor
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)
        
        # Now get the index of the descendant of the merger
        descIndex = index - 1
        
        descsfID = tree['SubfindID'][descIndex]
        
        if type=='Descendant':
            # Okay but you need to get the descendant of the descendant, which requires loading a new tree:
            tree_desc = il.sublink.loadTree(sP,tree['SnapNum'][0], descsfID,fields = fields)#
            descsfID = tree_desc['SubfindID'][0]
        else:
            descsfID = tree['SubfindID'][descIndex]
        fpsfID = tree['SubfindID'][fpIndex]
        
        # explore breadth
        # so nextprogenitors point laterally, so fpIndex is usually 2, meaning one above the starting
        # so points laterally from the first progenitor to find if there are two progenitors
        npID = tree['NextProgenitorID'][fpIndex]

        if npID != -1:# This identifies mergers
            
            npIndex = index + (npID - rootID)
            #print('npIndex', npIndex)
            if type=='Progenitor':
                npsfID = tree['SubfindID'][npIndex+1]# add one to take one more step back in time to get first prog of the next prog
                #print('SFID np', npsfID)
                npMass  = il.sublink.maxPastMass(tree, npIndex+1, massPartType)
                #print('new fpsfID', tree['SubfindID'][fpIndex+1])
                fpsfID = tree['SubfindID'][fpIndex+1]
            else:
                npsfID = tree['SubfindID'][npIndex]# add one to take one more step back in time to get first prog of the next prog
                #print('SFID np', npsfID)
                npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)
                #print('new fpsfID', tree['SubfindID'][fpIndex+1])
                fpsfID = tree['SubfindID'][fpIndex]
            
            # See if you can grab all other progenitors :)
            all_related_indices = np.where(tree['SnapNum']==snapnum_full)[0]
            print('indices of related gals', all_related_indices)
            print('length of related', len(all_related_indices))
            
            # Now get the masses of these subhalos
            masses = []
            for indices in all_related_indices:
                masses.append(il.sublink.maxPastMass(tree, indices, massPartType))
                
            h = 0.6774
            masses_scale = np.array([x * (1e10 / h) for x in masses])
            cutoff = 10**3*(5.7*10**4)/h
            
           
            mass_cutoff = np.where(masses_scale > cutoff/10)[0]
            # Now select only these indices:
            print('get these subfinds', all_related_indices[mass_cutoff])
            list_prog = tree['SubfindID'][all_related_indices[mass_cutoff]]
            
            
            
            #print('where',tree['SubfindID'][np.where(tree['SnapNum']==17)], len(tree['SubfindID'][np.where(tree['SnapNum']==17)]))
            #STOP
            #print('there is a NP!', npMass, fpMass)
            #STOP
            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass
                #print('mass ratio', ratio)
                if ratio >= minMR and ratio <= invMassRatio:
                    return 1, [sfID, descsfID, list_prog, npsfID], ratio
                else:
                    return 0, [sfID, descsfID, fpsfID, npsfID], ratio
            else:
                return 0, 0, 0

        else:
            return 0, 0, 0

 

def main():
    
    h = 0.6774
    ll = 10**3*(5.7*10**4)/h # we've decided to do 1000x the baryonic mass resolution

    #ll = (9.4*10**5)/h
    #10**9
    #ll = 10**8
    ul = 10**14# was 10**12
    
    print('Lower baryonic mass limit', ll)
    # First, grab the identities of subhalos that are between a range of stellar mass at a given snapshot
    
    print('getting subhalos from this snapnum', snapnum)
    subids, select_mass, select_npart, all_mass = get_subhalos(run, snapnum, basePath, ll, ul, 1, h, 'no')
    
    if lbt_snap==0 or lbt_bin_center==0:
        print('improper look back time', lbt_snap, lbt_bin_center)
        STOP# 
    
    all_is = np.linspace(0,len(all_mass)-1, len(all_mass))
    subids_all = [int(x) for x in all_is]

    print('length of all subhalos at this snapshot', len(subids))
    
    print('getting subhalos from the full snapshot', snapnum_full)
    
    subids_f, select_mass_f, select_npart_f, all_mass_f = get_subhalos(run, snapnum_full, basePath, 0, 10**14, 1, h, 'yes')
    all_is_f = np.linspace(0,len(all_mass_f)-1, len(all_mass_f))
    subids_all_f = [int(x) for x in all_is_f]

    print('length of all subhalos at center snapshot', len(subids_f))
    
    
    
    # Define the fields you would like to load when you load the merger trees
    fields = ['SubhaloID','NextProgenitorID','LastProgenitorID','MainLeafProgenitorID','FirstProgenitorID','DescendantID','SubhaloMassType',
          'SubhaloMass','SubfindID','SnapNum']
    '''
    Definitions:
    Everything is assigned in a depth-first fashion (see Lemson & Springel 2006), meaning that the way things
    are numbered is based on their mass and location in the tree structure. Think of the moment in time that you
    are located as the trunk of the tree and all other past mergers as the tree roots. The root is #0 and #1 is the 
    largest root one step back in time. #2 is the largest root of that largest root and so on until you walk
    all the way back along the main progenitor branch to the tip of that one biggest root. At this point you are
    at #n. #n+1 is then a lateral jump to the next available lowst down branch and work that all the way to its tip.
    In this way, you are working all the way down (depth-first) along progressively smaller and smaller branches.
    
    SubhaloID = unique identifier of the subhalo, assigned in a depth-first fashion. The value starts at 0 and 
        goes to last progenitor within a single tree, from here on, ID = subhaloID
        
    NextProgenitorID = The subhalo ID of the next most massive history which shares the same descendent
        as this subhalo. So it will be lateral from this one, or it will be -1 which means a merger is not
        happening between this time step and the next one.

    FirstProgenitorID = This progenitor one step back in time of this current subhalo. It is also the
        most massive progenitor if there are more than one progentors.
        
    LastProgenitorID = points to the subhalo that is the furthest back in time on the current branch you're on. 
        i.e., This relates to the way that merger trees are organized in a depth-first fashion. So for one merger
        tree there can be more than one last progenitor, because you can't jump laterally to a new root, you
        just work back along the one you're on.
        
    MainLeafProgenitorID = ID of the main progenitor branch last progenitor
    
    SubfindID = Index of the subhalo in the subfind group catalog, aka its ID!! THIS IS WHAT WE WILL USE TO RELATE
        BACK TO THE GALAXY IN THE GETSUBHALOS() FUNCTION AND TO FIND IT ITS PARTICLE DATA FOR SKIRT-IFICATION.
    
    '''
    
    # Set a minimum mass ratio
    ratio_min = 0.1
    mergers_id = []
    ratio_list = []
    particle_number = []
    stellar_mass_list = []
    merger_list = []
    desc_list = []
    first_prog_list = []
    next_prog_list = []
    count = 0
    nmerg = 0
    nlen1 = 0
    
    # For each of the selected subhalos, load up the merger tree and search through the tree
    # to see how many mergers there are above the mass ratio cut
    for i in range(len(subids)):
        #if nmerg > 10:
        #    break
        #print(i, 'seaching this subid', subids[i])
        tree = il.sublink.loadTree(sP,snapnum, subids[i],fields = fields)#
        
        #print('loading up the tree for this subhalo id', subids[i]) 
        #print('firstprogid', tree['FirstProgenitorID'][0:100])
        #print('nextprogid', tree['NextProgenitorID'][0:100])
        #print('last prog', tree['LastProgenitorID'][0:100])
        #print('SubhaloID', tree['SubhaloID'][0:100])
        #STOP
        
        try:
            check_keys = tree.keys()
        except AttributeError: # this is if the tree is somehow empty
            continue
        #try:
        
        #print(tree['NextProgenitorID'], len(tree['NextProgenitorID']))
        if len(tree['NextProgenitorID']) <3: # NextprogenitorID is 
            nlen1+=1
            continue
        #print(len(tree['SubhaloID']))
        if abs(snapnum - snapnum_full) ==2:
            if snapnum > snapnum_full:
                type=='Progenitor'
            else:
                type=='Descendant'
            if all_prog:
                numMergers, Subfind_ID_array, ratio_out = mergers_two_steps_all_prog(tree,ratio_min, subids[i], fields, type)#mergers_two_steps_all_prog
            else:
                numMergers, Subfind_ID_array, ratio_out = mergers_two_steps(tree,ratio_min, subids[i], fields, type)
            
        else:
            numMergers, Subfind_ID_array, ratio_out = mergers_one_step(tree,ratio_min)
        count+=1
        if numMergers != 0:
            #print('i = ', i)
            nmerg+=1
            #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            #print('this is your merger', subids[i], round(ratio_out, 1), select_npart[i], select_mass[i])
            #print('mass ratio', ratio_out)
            #print('npart', select_npart[i])
            #print('fraction', nmerg/count)
            #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            mergers_id.append(subids[i])
            ratio_list.append(ratio_out)
            particle_number.append(select_npart[i])
            stellar_mass_list.append(select_mass[i])
            merger_list.append(Subfind_ID_array[0])
            desc_list.append(Subfind_ID_array[1])
            first_prog_list.append(Subfind_ID_array[2])
            next_prog_list.append(Subfind_ID_array[3])
            
        #if nmerg > 10:
        #    break
        #else: # then its a non-merger?l
    print('Run', run)
    print('Snapnum', snapnum)
    print('number of length 1', nlen1)
    print('total number run', count, 'total number of mergers above a mass ratio of ', ratio_min, '# = ', nmerg)
    print('fractional', nmerg/count)
    print('merger id list', mergers_id)
    print('mass ratios', ratio_list)
    print('number of stellar particles', particle_number)
    print('stellar mass', stellar_mass_list)
    print('descendent ids', desc_list)
    print('first prog ids', first_prog_list)
    print('next prog ids', next_prog_list)
    
    # We need to make a table with useful stuff about mergers in it
    if all_prog:
        file_out = open('Tables/mergers_snap_all_prog_'+str(snapnum)+'_'+run+'.txt','w')
    else:
        file_out = open('Tables/mergers_snap_'+str(snapnum)+'_'+run+'.txt','w')
    file_out.write('Subfind_ID'+'\t'+'t-t_merg'+'\t'+'q'+'\t'+'Type'+'\t'+'mass'+'\n')
    for j in range(nmerg):
        if snapnum==snapnum_full:# in other words if you're currently on the full snap:
            file_out.write(str(mergers_id[j])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'Merger'+'\t'+str(stellar_mass_list[j])+'\n')
        # snapnum_full - snapnum == 1
        if snapnum_full - snapnum > 0: #this would happen for snapnum 49 i.e., you want to get descendants one time step forward
            print(nmerg)
            print(desc_list)
            print(ratio_list)
            
            print(len(desc_list),len(ratio_list),len(all_mass_f))
            print(desc_list[j], ratio_list[j])
            print(all_mass_f[desc_list[j]])
            
            file_out.write(str(desc_list[j])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'Descendant'+'\t'+str(all_mass[desc_list[j]])+'\n')
        # snapnum - snapnum_full == 1:
        # So this would be 17 is the bin center and 13 is earlier in time, so descendants of these would be at the bin center ^
        if snapnum_full - snapnum  < 0: #this would happen for snapnum 51 i.e., you want to get both progenitors :)
            # You also have to re-solve for their stellar masses
            if abs(snapnum - snapnum_full) ==2: # then we're getting lots and lots of progenitors
                for k in range(len(first_prog_list[j])):
                    file_out.write(str(first_prog_list[j][k])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'progenitor'+'\t'+str(all_mass_f[first_prog_list[j][k]])+'\n')
                  
            else:              
                file_out.write(str(first_prog_list[j])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'first_progenitor'+'\t'+str(all_mass_f[first_prog_list[j]])+'\n')
                file_out.write(str(next_prog_list[j])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'next_progenitor'+'\t'+str(all_mass_f[next_prog_list[j]])+'\n')


        #file_out.write(str(mergers_id[j])+'\t'+str('Descendent')+'\t'+str(desc_list[j])+'\n')
        #file_out.write(str(mergers_id[j])+'\t'+str('First_prog')+'\t'+str(first_prog_list[j])+'\n')
        #file_out.write(str(mergers_id[j])+'\t'+str('Next_prog')+'\t'+str(next_prog_list[j])+'\n')
    file_out.close()
    STOP

    
    couple_trees = loadMPBs(sP+'../postprocessing/', subids[0:2], snapnum, fields='None')
    print('a couple of trees', couple_trees)
    #def loadMPBs(sP, ids, fields=None, treeName=treeName_default, fieldNamesOnly=False):

if __name__ == "__main__":
    main()
