#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Uses Dylan Nelson's code to load multiple merger
# trees from a list of subhalos at a given redshift
# then constructs a catalog of subfindid, mass ratio
# maybe stellar mass of the merger?
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import illustris_python as il
import h5py
import numpy as np
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt:                                                                                    
import matplotlib.pyplot as plt


basePath = '/n/hernquistfs3/IllustrisTNG/Runs/'

basePath = '/n/holylfs/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L35n2160TNG'
sP = basePath+run+'/output/'
snapnum = 50#13 is z = 6, 21 is z = 4  # 33 is z = 2
min_snapnum = 49
max_snapnum = 51
treeName_default="Sublink"

def loadMPBs(sP, ids, snap, fields=None, treeName="Sublink", fieldNamesOnly=False):
    """ Load multiple MPBs at once (e.g. all of them), optimized for speed, with a full tree load (high mem).
    Basically a rewrite of illustris_python/sublink.py under specific conditions (hopefully temporary).
      Return: a dictionary whose keys are subhalo IDs, and the contents of each dict value is another
      dictionary of identical stucture to the return of loadMPB().
    """
    from glob import glob
    #assert treeName in ['SubLink','SubLink_gal'] # otherwise need to generalize tree loading

    # make sure fields is not a single element
    if isinstance(fields, str):
        fields = [fields]

    fieldsLoad = fields + ['MainLeafProgenitorID']

    # find full tree data sizes and attributes
    numTreeFiles = len(glob(il.sublink.treePath(sP,treeName,'*')))#was .simPath

    lengths = {}
    dtypes = {}
    seconddims = {}

    for field in fieldsLoad:
        lengths[field] = 0
        seconddims[field] = 0

    for i in range(numTreeFiles):
        with h5py.File(il.sublink.treePath(sP,treeName,i),'r') as f:#
            for field in fieldsLoad:
                dtypes[field] = f[field].dtype
                lengths[field] += f[field].shape[0]
                if len(f[field].shape) > 1:
                    seconddims[field] = f[field].shape[1]

    # allocate for a full load
    fulltree = {}

    for field in fieldsLoad:
        if seconddims[field] == 0:
            fulltree[field] = np.zeros( lengths[field], dtype=dtypes[field] )
        else:
            fulltree[field] = np.zeros( (lengths[field],seconddims[field]), dtype=dtypes[field] )

    # load full tree
    offset = 0

    for i in range(numTreeFiles):
        with h5py.File(il.sublink.treePath(sP,treeName,i),'r') as f:
            for field in fieldsLoad:
                if seconddims[field] == 0:
                    fulltree[field][offset : offset + f[field].shape[0]] = f[field][()]
                else:
                    fulltree[field][offset : offset + f[field].shape[0],:] = f[field][()]
            offset += f[field].shape[0]

    result = {}

    # (Step 1) treeOffsets()
    offsetFile = il.groupcat.offsetPath(sP,snap)
    prefix = 'Subhalo/' + treeName + '/'

    with h5py.File(offsetFile,'r') as f:
        # load all merger tree offsets
        if prefix+'RowNum' not in f:
            return result # early snapshots, no tree offset

        RowNums     = f[prefix+'RowNum'][()]
        SubhaloIDs  = f[prefix+'SubhaloID'][()]

    # now subhalos one at a time (memory operations only)
    for i, id in enumerate(ids):
        if id == -1:
            continue # skip requests for e.g. fof halos which had no central subhalo

        # (Step 2) loadTree()
        RowNum = RowNums[id]
        SubhaloID  = SubhaloIDs[id]
        MainLeafProgenitorID = fulltree['MainLeafProgenitorID'][RowNum]

        if RowNum == -1:
            continue

        # load only main progenitor branch
        rowStart = RowNum
        rowEnd   = RowNum + (MainLeafProgenitorID - SubhaloID)
        nRows    = rowEnd - rowStart + 1

        # init dict
        result[id] = {'count':nRows}

        # loop over each requested field and copy, no error checking
        for field in fields:
            result[id][field] = fulltree[field][RowNum:RowNum+nRows]

    return result

def returnSimpleTree(basePath,subhalo_id,snapshot_number):
    # load sublink chunk offsets from header of first file
    groupFile=basePath+'groups_'+str(snapshot_number)+'/'+'groups_'+str(snapshot_number)+'.0.hdf5'
    with h5py.File(groupFile,'r') as f:
        subhaloFileOffsets = f['Header'].attrs['FileOffsets_Subhalo']
        treeFileOffsets = f['Header'].attrs['FileOffsets_SubLink']

    # calculate target group catalog file chunk which contains this id
    subhaloFileOffsets = int(subhalo_id) - subhaloFileOffsets
    fileNum = np.max( np.where(subhaloFileOffsets >= 0) )
    subhaloFile=basePath+'groups_'+str(snapshot_number)+'/'+'groups_'+str(snapshot_number)+'.'+str(fileNum)+'.hdf5'
    subhaloOffset=subhaloFileOffsets[fileNum]

    #finding the right file for this tree and where exactly to look
    with h5py.File(subhaloFile,'r') as groupFile:
        rowNum=groupFile["Offsets"]['Subhalo_SublinkRowNum'][subhaloOffset]
        lastProgId=groupFile["Offsets"]['Subhalo_SublinkLastProgenitorID'][subhaloOffset]
        subhaloId=groupFile["Offsets"]['Subhalo_SublinkSubhaloID'][subhaloOffset]

    treeFileOffsets=int(rowNum)-treeFileOffsets
    treeFileNum=np.max(np.where(treeFileOffsets >= 0))
    treeFile=basePath+'trees/SubLink/tree_extended.'+str(treeFileNum)+'.hdf5'
    rowStart = treeFileOffsets[treeFileNum]

    with h5py.File(treeFile,'r') as rawTree:
        #finding which entries in tree we're interested in
        firstId = rawTree['RootDescendantID'][rowStart]
        rowStart=rowStart+(firstId-subhaloId)
        lastId = rawTree['LastProgenitorID'][rowStart]
        rowEnd=rowStart+lastId-firstId+1

        nFind=rawTree['SubfindID'][rowStart:rowEnd]
        nSnap=rawTree['SnapNum'][rowStart:rowEnd]
        nSub=rawTree['SubhaloID'][rowStart:rowEnd]
        nFirst=rawTree['FirstProgenitorID'][rowStart:rowEnd]
        nNext=rawTree['NextProgenitorID'][rowStart:rowEnd]
        nDesc=rawTree['DescendantID'][rowStart:rowEnd]

    #initialises the tree
    thisTree=-1*np.ones((136,2),dtype=int)
    thisTree[:,0]=np.arange(135,-1,-1)

    #traces the tree back to the latest subhalo in the mpb
    zIndex=np.argwhere((nFind==subhalo_id) & (nSnap==snapshot_number))[0][0]
    thisIndex=zIndex
    if thisIndex!=0:
        descSub=nDesc[zIndex]
        thisSub=nSub[zIndex]
        descIndex=zIndex+descSub-nSub[zIndex]
        while ((nFirst[descIndex]==thisSub) & (nDesc[descIndex]!=-1)): #while the first progentior of each descendant is this subhalo
            descSub=nDesc[descIndex]
            thisSub=nSub[descIndex]
            descIndex=descIndex+descSub-nSub[descIndex]
            thisIndex=descIndex

    thisSnap=nSnap[thisIndex]
    thisFind=nFind[thisIndex]
    thisTree[135-thisSnap,1]=thisFind # records subfind id of first step

    #initialises the list of merging galaxies
    if thisSnap!=135: #if it doesn't reach z=0 records the subhalo it merges with
        descIndex=thisIndex+nDesc[thisIndex]-nSub[thisIndex]
        descSnap=nSnap[descIndex]
        descFind=nFind[descIndex]
        mergerTree=[[descSnap,descIndex]]
    else:
        mergerTree=[]


    while nFirst[thisIndex]!=-1: # goes through main progenitors
        thisIndex=thisIndex+nFirst[thisIndex]-nSub[thisIndex] #index of next main step in main progenitor branch

        thisSnap=nSnap[thisIndex] #snapshot of this main progenitor
        thisFind=nFind[thisIndex] #subfind id of this main progenitor
        thisTree[135-thisSnap,1]=thisFind # records subfind id of this main progenitor

        nextIndex=thisIndex+0 #stupid python objects
        while nNext[nextIndex]!=-1: # goes through merging halos (next progenitors)

            nextIndex=nextIndex+nNext[nextIndex]-nSub[nextIndex] #index of next progenitor

            #records details of these mergers
            mergerSnap=nSnap[nextIndex]
            mergerSub=nFind[nextIndex]
            mergerTree.append([mergerSnap,mergerSub])

    # got to the end of this branch, so must save it
    filled=np.argwhere(thisTree[:,1]!=-1) # finds the snapshots during which the halo is in the tree
    if filled.size==1:
        thisTree=thisTree[filled[0],:]
    elif thisTree[0,1]==-1: # if the tree doesn't make it to z=0
        thisTree=thisTree[filled[0][0]:filled[-1][0]+1,:]
    else:
        thisTree=thisTree[0:filled[-1][0]+1,:] # if it does

    mergerTree=np.array(mergerTree)
    data={"Main":thisTree}
    data['Mergers']=mergerTree
    return data

def returnSimpleTreeTNG(sP,subhalo_id,snapshot_number):
    # In TNG the offsets are stored separately in postprocessing
    groupFile=sP+'../postprocessing/offsets/offsets_0'+str(snapshot_number)+'.hdf5'
    with h5py.File(groupFile,'r') as f:
        print(f.keys())
        print(f['FileOffsets'].keys())
        print(f['FileOffsets'].attrs['Subhalo'])# I cannot figure out how to access this :/
        subhaloFileOffsets = f['FileOffsets/Subhalo']#f['Header'].attrs['FileOffsets_Subhalo']
        treeFileOffsets = f['FileOffsets/SubLink']#['Header'].attrs['FileOffsets_SubLink']
    
    STOP
    print('subhalo_id', subhalo_id)
    print('subhalo file offsets', subhaloFileOffsets)
    print('sublink file offsets', treeFileOffsets)
    # calculate target group catalog file chunk which contains this id
    subhaloFileOffsets = int(subhalo_id) - subhaloFileOffsets
    fileNum = np.max( np.where(subhaloFileOffsets >= 0) )
    subhaloFile=sP+'groups_0'+str(snapshot_number)+'/fof_subhalo_tab_0'+str(snapshot_number)+'.'+str(fileNum)+'.hdf5'
    subhaloOffset=subhaloFileOffsets[fileNum]

    #finding the right file for this tree and where exactly to look
    with h5py.File(subhaloFile,'r') as groupFile:
        rowNum=groupFile["Offsets"]['Subhalo_SublinkRowNum'][subhaloOffset]
        lastProgId=groupFile["Offsets"]['Subhalo_SublinkLastProgenitorID'][subhaloOffset]
        subhaloId=groupFile["Offsets"]['Subhalo_SublinkSubhaloID'][subhaloOffset]

    treeFileOffsets=int(rowNum)-treeFileOffsets
    treeFileNum=np.max(np.where(treeFileOffsets >= 0))
    treeFile=sP+'trees/SubLink/tree_extended.'+str(treeFileNum)+'.hdf5'
    rowStart = treeFileOffsets[treeFileNum]

    with h5py.File(treeFile,'r') as rawTree:
        #finding which entries in tree we're interested in
        firstId = rawTree['RootDescendantID'][rowStart]
        rowStart=rowStart+(firstId-subhaloId)
        lastId = rawTree['LastProgenitorID'][rowStart]
        rowEnd=rowStart+lastId-firstId+1

        nFind=rawTree['SubfindID'][rowStart:rowEnd]
        nSnap=rawTree['SnapNum'][rowStart:rowEnd]
        nSub=rawTree['SubhaloID'][rowStart:rowEnd]
        nFirst=rawTree['FirstProgenitorID'][rowStart:rowEnd]
        nNext=rawTree['NextProgenitorID'][rowStart:rowEnd]
        nDesc=rawTree['DescendantID'][rowStart:rowEnd]

    #initialises the tree
    thisTree=-1*np.ones((100,2),dtype=int)
    thisTree[:,0]=np.arange(99,-1,-1)

    #traces the tree back to the latest subhalo in the mpb
    zIndex=np.argwhere((nFind==subhalo_id) & (nSnap==snapshot_number))[0][0]
    thisIndex=zIndex
    if thisIndex!=0:
        descSub=nDesc[zIndex]
        thisSub=nSub[zIndex]
        descIndex=zIndex+descSub-nSub[zIndex]
        while ((nFirst[descIndex]==thisSub) & (nDesc[descIndex]!=-1)): #while the first progentior of each descendant is this subhalo
            descSub=nDesc[descIndex]
            thisSub=nSub[descIndex]
            descIndex=descIndex+descSub-nSub[descIndex]
            thisIndex=descIndex

    thisSnap=nSnap[thisIndex]
    thisFind=nFind[thisIndex]
    thisTree[135-thisSnap,1]=thisFind # records subfind id of first step

    #initialises the list of merging galaxies
    if thisSnap!=135: #if it doesn't reach z=0 records the subhalo it merges with
        descIndex=thisIndex+nDesc[thisIndex]-nSub[thisIndex]
        descSnap=nSnap[descIndex]
        descFind=nFind[descIndex]
        mergerTree=[[descSnap,descIndex]]
    else:
        mergerTree=[]


    while nFirst[thisIndex]!=-1: # goes through main progenitors
        thisIndex=thisIndex+nFirst[thisIndex]-nSub[thisIndex] #index of next main step in main progenitor branch

        thisSnap=nSnap[thisIndex] #snapshot of this main progenitor
        thisFind=nFind[thisIndex] #subfind id of this main progenitor
        thisTree[135-thisSnap,1]=thisFind # records subfind id of this main progenitor

        nextIndex=thisIndex+0 #stupid python objects
        while nNext[nextIndex]!=-1: # goes through merging halos (next progenitors)

            nextIndex=nextIndex+nNext[nextIndex]-nSub[nextIndex] #index of next progenitor

            #records details of these mergers
            mergerSnap=nSnap[nextIndex]
            mergerSub=nFind[nextIndex]
            mergerTree.append([mergerSnap,mergerSub])

    # got to the end of this branch, so must save it
    filled=np.argwhere(thisTree[:,1]!=-1) # finds the snapshots during which the halo is in the tree
    if filled.size==1:
        thisTree=thisTree[filled[0],:]
    elif thisTree[0,1]==-1: # if the tree doesn't make it to z=0
        thisTree=thisTree[filled[0][0]:filled[-1][0]+1,:]
    else:
        thisTree=thisTree[0:filled[-1][0]+1,:] # if it does

    mergerTree=np.array(mergerTree)
    data={"Main":thisTree}
    data['Mergers']=mergerTree
    return data

def get_subhalos(run, snapnum, basePath, minmass, maxmass, min_npart, plot):
    
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
        
        # Okay now get the min and the max snapnum that are within 250 of the current snap:
        
        
        
        mass_msun = subhalos['SubhaloMassType'][:,4] * 1e10 / 0.704
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
    
        mass_msun = subhalos['SubhaloMassType'][:,4] * 1e10 / 0.704                                                                                     
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




def numMergers_one_step(tree, minMR, massPartType='stars', index=0):
    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    #print('what the hell happened to it?', minMR)
    invMassRatio = 1.0 / minMR

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]

    if fpID != -1: ####THIS IS WRONG, DON'T STOP WHEN YOUVE GOTTEN TO THE END OF THE MPB
        fpIndex = index + (fpID - rootID)
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)

        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        if npID != -1:
            npIndex = index + (npID - rootID)
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)

            #print('there is a NP!', npMass, fpMass)
            #STOP
            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass
                #print('mass ratio', ratio)
                if ratio >= minMR and ratio <= invMassRatio:
                    
                    return 1, [rootID, fpID, npID], ratio
                else:
                    return 0, [rootID, fpID, npID], ratio
            else:
                return 0, 0, 0

        else:
            return 0, 0, 0

 

def main():
    # First, grab the identities of subhalos that are between a range of stellar mass at a given snapshot
    subids, select_mass, select_npart = get_subhalos(run, snapnum, basePath, 10**8, 10**11,1, 'yes')
    
    # Define the fields you would like to load when you load the merger trees
    fields = ['SubhaloID','NextProgenitorID','LastProgenitorID','MainLeafProgenitorID','FirstProgenitorID','SubhaloMassType',
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
    
    SubhaloMassType =
    SubhaloMass = 
    SubfindID = Index of the subhalo in the subfind group catalog, which relates tree IDs to the IDs of
        everything in one snapshot
    SnapNum = 
    
    '''
    
    # Set a minimum mass ratio
    ratio_min = 0.1
    mergers_id = []
    ratio_list = []
    particle_number = []
    stellar_mass_list = []
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
        
        print('loading up the tree for this subhalo id', subids[i])
        
        print('firstprogid', tree['FirstProgenitorID'][0:100])
        print('nextprogid', tree['NextProgenitorID'][0:100])
        print('last prog', tree['LastProgenitorID'][0:100])
        print('SubhaloID', tree['SubhaloID'][0:100])
        #STOP
        
        try:
            check_keys = tree.keys()
        except AttributeError: # this is if the tree is somehow empty
            continue
        #try:
        
        #print(tree['NextProgenitorID'], len(tree['NextProgenitorID']))
        if len(tree['NextProgenitorID']) ==1: # NextprogenitorID is 
            nlen1+=1
            continue
        numMergers, Subhalo_ID_array, ratio_out = numMergers_one_step(tree,ratio_min)
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
        #else: # then its a non-merger?
    print('Run', run)
    print('Snapnum', snapnum)
    print('number of length 1', nlen1)
    print('total number run', count, 'total number of mergers above a mass ratio of ', ratio_min, '# = ', nmerg)
    print('fractional', nmerg/count)
    print('merger id list', mergers_id)
    print('mass ratios', ratio_list)
    print('number of stellar particles', particle_number)
    print('stellar mass', stellar_mass_list)
    
    # We need to make a table with useful stuff about mergers in it
    file_out = open('Tables/mergers_snap_'+str(snapnum)+'.txt','w')
    file_out.write('Merger_ID'+'\t'+'Subhalo_ID'+'\t'+'q'+'\t'+'mass'+'\n')
    for j in range(nmerg):
        file_out.write(str(mergers_id[j])+'\t'+str(mergers_id[j])+'\t'+str(ratio_list[j])+'\t'+str(stellar_mass_list[j])+'\n')
    file_out.close()
    STOP

    
    couple_trees = loadMPBs(sP+'../postprocessing/', subids[0:2], snapnum, fields='None')
    print('a couple of trees', couple_trees)
    #def loadMPBs(sP, ids, fields=None, treeName=treeName_default, fieldNamesOnly=False):

if __name__ == "__main__":
    main()
