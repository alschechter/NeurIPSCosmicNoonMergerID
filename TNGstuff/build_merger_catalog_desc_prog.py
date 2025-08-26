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

# This was the old basepath
basePath = '/n/hernquistfs3/IllustrisTNG/Runs/'

basePath = '/n/holylfs/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L35n2160TNG'
sP = basePath+run+'/output/'
snapnum = 84#13 is z = 6, 21 is z = 4  # 33 is z = 2

# Just for my own notes about which are included in the 250Myr window 
# (500 Myr centered on z=1)
min_snapnum = 49
max_snapnum = 51
treeName_default="Sublink"


# This function uses the groupcat.loadSubhalos function to grab all the IDs and details
# of galaxies within a certain range of stellar masses at a snapshot

# As a note, I have min_npart as an optional input here but Jacob was saying that its
# not necessary to select by # of particles; this has a nearly 1:1 relationship with
# selecting by stellar mass.
def get_subhalos(run, snapnum, basePath, minmass, maxmass, min_npart, h, plot):
    
    fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType']#was ['SubhaloMassInRad']

    # It might actually be useful to sort this by mass, so you can snag all subhalos between a certain mass range:                                      

    if run=='L35n2160TNG':#run = 'L75n1820TNG'#'L35n2160TNG'  
        subhalos = il.groupcat.loadSubhalos(basePath+run+'/output/',snapnum, fields=fields)
        subhalosh = il.groupcat.loadHeader(basePath+run+'/output/',snapnum)#, fields=fields_header)
        print('z', subhalosh['Redshift'], 'a', subhalosh['Time'])
        
        
        # Set up the proper cosmology to calculate lookback time
        # Get astropy from here: 
        # module --ignore-cache load "astropy-0.4.2-fasrc01"
        from astropy.cosmology import Planck15
        # Planck Collab 2015
        print(Planck15.Om0, Planck15.H0)
        Om0 = subhalosh['Omega0']
        Ob0 = subhalosh['OmegaLambda']
        print(Om0, Ob0)
        
        #cosmo = Planck15(H0=67.74, Om0=Om0, Ob0=0.05)
        print('lookback time', Planck15.lookback_time(subhalosh['Redshift']))
        
        
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

    
    '''
    print('sorted_mass', sorted_mass[-100:-1])
    print('sorted_ids', sorted_indices[-100:-1])
    
    print('making sure the masking works')
    print(subhalos.keys())
    stop=20
    print(subhalos['SubhaloFlag'][0:stop])
    print(subhalos['SubhaloMassInRad'][0:stop])
    print('masked', subhalos['SubhaloMassInRad'][0:stop][subhalos['SubhaloFlag'][0:stop]])
    print(len(subhalos['SubhaloMassInRad'][0:stop]), len(subhalos['SubhaloMassInRad'][0:stop][subhalos['SubhaloFlag'][0:stop]]))
    STOP
    
    

    # Try geting primary subhalo IDs to test if the above is listed by index?
    GroupFirstSub = il.groupcat.loadHalos(basePath+run+'/output/',snapnum,fields=['GroupFirstSub'])
    print(len(GroupFirstSub), GroupFirstSub[0:100])
    
    STOP
    '''
    print('select_ids', select_indices[0:10])
    print('select mass', select_mass[0:10])
    print('select stellar part', select_stellar_part[0:10])
    return select_indices, select_mass, select_stellar_part



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

 

def main():
    h = 0.6774
    ll = (5.7*10**4)/h
    #ll = (9.4*10**5)/h
    #10**9
    print('Lower baryonic mass limit', (5.7*10**4)/h)
    
    # First, grab the identities of subhalos that are between a range of stellar mass at a given snapshot
    
    subids, select_mass, select_npart = get_subhalos(run, snapnum, basePath, ll, 10**11,1, h, 'yes')
    
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
    file_out = open('Tables/mergers_snap_desc_prog_'+str(snapnum)+'_'+run+'.txt','w')
    file_out.write('Merger_ID'+'\t'+'Type'+'\t'+'Other_ID'+'\t'+'q'+'\t'+'mass'+'\n')
    for j in range(nmerg):
        file_out.write(str(mergers_id[j])+'\t'+str('Merger')+'\t'+str(merger_list[j])+'\t'+str(ratio_list[j])+'\t'+str(stellar_mass_list[j])+'\n')
        file_out.write(str(mergers_id[j])+'\t'+str('Descendent')+'\t'+str(desc_list[j])+'\n')
        file_out.write(str(mergers_id[j])+'\t'+str('First_prog')+'\t'+str(first_prog_list[j])+'\n')
        file_out.write(str(mergers_id[j])+'\t'+str('Next_prog')+'\t'+str(next_prog_list[j])+'\n')
    file_out.close()
    STOP

    
    couple_trees = loadMPBs(sP+'../postprocessing/', subids[0:2], snapnum, fields='None')
    print('a couple of trees', couple_trees)
    #def loadMPBs(sP, ids, fields=None, treeName=treeName_default, fieldNamesOnly=False):

if __name__ == "__main__":
    main()
