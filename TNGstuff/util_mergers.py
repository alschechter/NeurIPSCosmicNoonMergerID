
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utilities that are going to be useful for IDing mergers :)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import illustris_python as il
#from il.util import partTypeNum
#import h5py
import numpy as np
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt:                                                           
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
import os

def partTypeNum(partType):
    """ Mapping between common names and numeric particle types. """
    if str(partType).isdigit():
        return int(partType)
        
    if str(partType).lower() in ['gas','cells']:
        return 0
    if str(partType).lower() in ['dm','darkmatter']:
        return 1
    if str(partType).lower() in ['dmlowres']:
        return 2 # only zoom simulations, not present in full periodic boxes
    if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
        return 3
    if str(partType).lower() in ['star','stars','stellar']:
        return 4 # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ['wind']:
        return 4 # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ['bh','bhs','blackhole','blackholes']:
        return 5
    
    raise Exception("Unknown particle type name.")

def maxPastMass(tree, index, partType='stars'):
    """ Get maximum past mass (of the given partType) along the main branch of a subhalo
        specified by index within this tree. """
    ptNum = partTypeNum(partType)
    

    try:
        branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index] + 1
    except:
        print('index', index)
        print(len(tree))
        print(tree['MainLeafProgenitorID'][index])
        print(tree['SubhaloID'][index])
    masses = tree['SubhaloMassType'][index: index + branchSize, ptNum]
    return np.max(masses)

def numMergers(tree, minMassRatio=1e-10, massPartType='stars', index=0):
    """ Calculate the number of mergers in this sub-tree (optionally above some mass ratio threshold). """
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    numMergers   = 0
    invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]
    fpID   = tree['FirstProgenitorID'][index]

    print('index', index)
    print('fpID before starting', fpID)

    while fpID != -1:
        fpIndex = index + (fpID - rootID)
        print('fpIndex', fpIndex)
        #fpMass  = maxPastMass(tree, fpIndex, massPartType)
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)


        # explore breadth
        npID = tree['NextProgenitorID'][fpIndex]

        while npID != -1:
            npIndex = index + (npID - rootID)
            npMass  = maxPastMass(tree, npIndex, massPartType)

            # count if both masses are non-zero, and ratio exceeds threshold
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass

                if ratio >= minMassRatio and ratio <= invMassRatio:
                    numMergers += 1

            npID = tree['NextProgenitorID'][npIndex]

        fpID = tree['FirstProgenitorID'][fpIndex]

    return numMergers

def get_subhalos(run, snapnum, basePath, minmass, maxmass, min_npart, h, plot):
    
    fields = ['SubhaloFlag','SubhaloMassType', 'SubhaloSFRinRad', 'SubhaloLenType']#was ['SubhaloMassInRad']

    # It might actually be useful to sort this by mass, so you can snag all subhalos between a certain mass range:                                      

    if run=='L35n2160TNG':#run = 'L75n1820TNG'#'L35n2160TNG'  
        subhalos = il.groupcat.loadSubhalos(basePath+run+'/output/',snapnum, fields=fields)# 
        subhalosh = il.groupcat.loadHeader(basePath+run+'/output/',snapnum)#,n fields=fields_header)
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

def num_mergers_one_step(tree, minMR, massPartType='stars', index=1):

    # verify the input sub-tree has the required fields                                                                                                                       
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','DescendantID']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    invMassRatio = 1.0 / minMR

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
        print('getting first prog mass', fpIndex)                                
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
            print('next prog mass', npIndex)
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)

            #print('there is a NP!', npMass, fpMass)                                                                                                                          
            #STOP                                                                                                                                                             
            # count if both masses are non-zero, and ratio exceeds threshold                                                                                                  
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass
                #print('mass ratio', ratio)                                                                                                                                   
                if ratio >= minMR and ratio <= invMassRatio:

                    return 1, ratio
                else:
                    return 0, 0
            else:
                return 0, 0

        else:
            return 0, 0
def num_mergers_n_step(sP, tree,n_steps, minMR, massPartType='stars', index=1):

    # verify the input sub-tree has the required fields                                                                                                                  
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','DescendantID']

    if not set(reqFields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqFields))

    invMassRatio = 1.0 / minMR

    rootID = tree['SubhaloID'][index] # the rootID is the subhalo ID where this is rooted
    sfID = tree['SubfindID'][index]
    fpID   = tree['FirstProgenitorID'][index]
    descID   = tree['DescendantID'][index]
    descsfID = tree['SubfindID'][0]

    #print('rootID', rootID)

    n_mergers = []
    # first, walk back one step to the first progenitor                                                                                                             
    while (fpID != -1) & (index < n_steps + 1) & (len(tree['SubhaloID'])-1 > (index + (fpID - rootID))): #it will be -1 if there is no first progenitor                                                                                                       
        # Now walk back along the tree to the fpIndex, this should be                                                                                                     
        # 0 + 1 = 1 since we're only taking one step backwards in time                                                                                                  
        
        fpIndex = index + (fpID - rootID)
        

        # Get the mass of the first progenitor                                                                                                                              
        #print('this is the index that is not working', fpIndex, len(tree['MainLeafProgenitorID']), len(tree['SubhaloID']))
        fpMass  = il.sublink.maxPastMass(tree, fpIndex, massPartType)

        # Now get the index of the descendant of the merger                                                                                                                
        descIndex = index - 1
        descsfID = tree['SubfindID'][descIndex]
        fpsfID = tree['SubfindID'][fpIndex]

        # explore breadth                                                                                                                                                    
        npID = tree['NextProgenitorID'][fpIndex]
        if npID != -1:
            npIndex = index + (npID - rootID)
            #print('npIndex', npIndex)
            try:
                npsfID = tree['SubfindID'][npIndex]
            except IndexError:
                print('no npsfID')
            #print('this is the NP index that is not working', npIndex, 'index', index, 'npID', npID, 'rootID', rootID,len(tree['MainLeafProgenitorID']), len(tree['SubhaloID']))
            try:
                npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)
            except:
                #print('FAIL')
                npMass = 0
            #print('there is a NP!', npMass, fpMass)                                                                                          
            #STOP                                                                                                                                                    
            # count if both masses are non-zero, and ratio exceeds threshold                                                                                        
            if fpMass > 0.0 and npMass > 0.0:
                ratio = npMass / fpMass
                #print('mass ratio', ratio)                                                                                                                           
                if ratio >= minMR and ratio <= invMassRatio:

                    n_mergers.append(1)
                else:
                    n_mergers.append(0)
            else:
                n_mergers.append(0)

        else:
            n_mergers.append(0)
        index+=1 # add 1 to walk along MPB
    return n_mergers

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
    
    rootID = tree['SubhaloID'][index]# subhalo
    sfID = tree['SubfindID'][index]
    fpID   = tree['FirstProgenitorID'][index] # subhalo
    descID   = tree['DescendantID'][index] # subhalo
    
    
    
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
    # This is the code for tracing mergers to steps into the future or two steps into the past
    # It's important to know here how the trees work:
    # Here's the link to the Illustris page on this: https://www.tng-project.org/data/docs/specifications/#sec4a
    # Here's the link to the Lemson & Springel 2006 paper: http://articles.adsabs.harvard.edu/pdf/2006ASPC..351..212L
    
    #
    
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','DescendantID']
    
    # Set these equal to weird numbers so you know if the code is breaking
    fpsfID = 999
    npsfID = 999
    descsfID = 999
    
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
            tree_desc = il.sublink.loadTree(sP,tree['SnapNum'][0], descsfID,fields = fields, onlyMPB = True)#
            descsfID = tree_desc['SubfindID'][0]
        
        # explore breadth
        # so nextprogenitors point laterally, so fpIndex is usually 2, meaning one above the starting
        # so points laterally from the first progenitor to find if there are two progenitors
        npID = tree['NextProgenitorID'][fpIndex]

        if npID != -1:# This identifies mergers
            
            
            npIndex = index + (npID - rootID)
            #print('npIndex', npIndex)
            if type=='Progenitor':
                try:
                    npsfID = tree['SubfindID'][npIndex+1]# add one to take one more step back in time to get first prog of the next prog
                    fpsfID = tree['SubfindID'][fpIndex+1]
                    
                    '''
                    npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)
                    print('npMass', npMass)
                    print('fpMass', fpMass)
                    '''
                    
                except:
                    print('cannot trace back in time')
                    '''
                    npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)
                    print('npMass', npMass)
                    print('fpMass', fpMass)
                    STOP
                    '''
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType)
            
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

def mergers_three_steps(tree, minMR, subid_in, fields, type, massPartType='stars', index=1):
    # This is the code for tracing mergers to steps into the future or two steps into the past
    # It's important to know here how the trees work:
    # Here's the link to the Illustris page on this: https://www.tng-project.org/data/docs/specifications/#sec4a
    # Here's the link to the Lemson & Springel 2006 paper: http://articles.adsabs.harvard.edu/pdf/2006ASPC..351..212L
    
    #
    
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','DescendantID']
    
    # Set these equal to weird numbers so you know if the code is breaking
    fpsfID = 999
    npsfID = 999
    descsfID = 999
    

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
            # And you need to do this twice because we're going three steps
            tree_desc = il.sublink.loadTree(sP,tree['SnapNum'][0], descsfID,fields = fields, onlyMPB = True)#
            descsfID_one_step = tree_desc['SubfindID'][0]
            tree_desc_two = il.sublink.loadTree(sP,tree_desc['SnapNum'][0], descsfID_one_step,fields = fields, onlyMPB = True)#
            descsfID = tree_desc_two['SubfindID'][0]
        
        
        # explore breadth
        # so nextprogenitors point laterally, so fpIndex is usually 2, meaning one above the starting
        # so points laterally from the first progenitor to find if there are two progenitors
        npID = tree['NextProgenitorID'][fpIndex]

        if npID != -1:# This identifies mergers
            
            npIndex = index + (npID - rootID)
            #print('npIndex', npIndex)
            if type=='Progenitor':
                try:
                    npsfID = tree['SubfindID'][npIndex+2] # so tracing back in time to the NP two steps ago
                    fpsfID = tree['SubfindID'][fpIndex+2] # so tracing back in time to the FP two steps ago
                except:
                    print('cannot trace this back in time')
            # the mass ratio of the merger is still defined from the stop you're at
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType) 
            
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
            
def mergers_n_steps(sP, tree, minMR, subid_in, fields, type, n_steps, massPartType='stars', index=1):
    # This is the code for tracing mergers to steps into the future or two steps into the past
    # It's important to know here how the trees work:
    # Here's the link to the Illustris page on this: https://www.tng-project.org/data/docs/specifications/#sec4a
    # Here's the link to the Lemson & Springel 2006 paper: http://articles.adsabs.harvard.edu/pdf/2006ASPC..351..212L
    
    #
    
    # verify the input sub-tree has the required fields
    reqFields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType','DescendantID']
    
    # Set these equal to weird numbers so you know if the code is breaking
    fpsfID = 999
    npsfID = 999
    descsfID = 999
    

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
            # 
            if n_steps == 2:
                tree_desc = il.sublink.loadTree(sP,tree['SnapNum'][0], descsfID,fields = fields, onlyMPB = True)#
                descsfID_one_step = tree_desc['SubfindID'][0]
                
            if n_steps == 3:
                tree_desc = il.sublink.loadTree(sP,tree['SnapNum'][0], descsfID,fields = fields, onlyMPB = True)#
                descsfID_one_step = tree_desc['SubfindID'][0]
                tree_desc_two = il.sublink.loadTree(sP,tree_desc['SnapNum'][0], descsfID_one_step,fields = fields, onlyMPB = True)#
                descsfID = tree_desc_two['SubfindID'][0]
                
            if n_steps == 4:
                tree_desc = il.sublink.loadTree(sP,tree['SnapNum'][0], descsfID,fields = fields, onlyMPB = True)#
                descsfID_one_step = tree_desc['SubfindID'][0]
                tree_desc_two = il.sublink.loadTree(sP,tree_desc['SnapNum'][0], descsfID_one_step,fields = fields, onlyMPB = True)#
                descsfID_two_step = tree_desc_two['SubfindID'][0]
                tree_desc_three = il.sublink.loadTree(sP,tree_desc_two['SnapNum'][0], descsfID_two_step,fields = fields, onlyMPB = True)#
                descsfID = tree_desc_three['SubfindID'][0]
            
            if n_steps == 5:
                tree_desc = il.sublink.loadTree(sP,tree['SnapNum'][0], descsfID,fields = fields, onlyMPB = True)#
                descsfID_one_step = tree_desc['SubfindID'][0]
                tree_desc_two = il.sublink.loadTree(sP,tree_desc['SnapNum'][0], descsfID_one_step,fields = fields, onlyMPB = True)#
                descsfID_two_step = tree_desc_two['SubfindID'][0]
                tree_desc_three = il.sublink.loadTree(sP,tree_desc_two['SnapNum'][0], descsfID_two_step,fields = fields, onlyMPB = True)#
                descsfID_three_step = tree_desc_three['SubfindID'][0]
                tree_desc_four = il.sublink.loadTree(sP,tree_desc_three['SnapNum'][0], descsfID_three_step,fields = fields, onlyMPB = True)#
                descsfID = tree_desc_four['SubfindID'][0]
            if n_steps > 5 or n_steps < 2:
                #STOP commented out when expanding windows
                pass 
        
        # explore breadth
        # so nextprogenitors point laterally, so fpIndex is usually 2, meaning one above the starting
        # so points laterally from the first progenitor to find if there are two progenitors
        npID = tree['NextProgenitorID'][fpIndex]

        if npID != -1:# This identifies mergers
            
            npIndex = index + (npID - rootID)
            #print('npIndex', npIndex)
            if type=='Progenitor':
                if n_steps > 5:
                    #STOP commented out when I expanded merger window
                    pass 
                try:
                    npsfID = tree['SubfindID'][npIndex+ n_steps - 1] # so tracing back in time to the NP two steps ago
                    fpsfID = tree['SubfindID'][fpIndex+ n_steps - 1] # so tracing back in time to the FP two steps ago
                except:
                    print('cannot trace this back in time')
            # the mass ratio of the merger is still defined from the stop you're at
            npMass  = il.sublink.maxPastMass(tree, npIndex, massPartType) 
            
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

def lookup_dicts():
    #lookback time
    lbt_dict = {'9':13.11313545493271,'10':13.06788231,'11':13.03606974,'12':12.95636765,'13':12.86836771,'14':12.83476842,'15':12.76390407,'16':12.68785986,'17':12.62304621,'18':12.51872105,'19':12.43449791,'20':12.33446648,'21':12.260159329297,
                '22':12.11214593,'23':11.98881427,'24':11.85662123,'25':11.65551451,'26':11.56321698,'27':11.41746261,'28':11.26248662,# once we get back to z= 0.3, progenitors can come from two time steps previously
                '30':10.962554785869012,'31':10.8208574,'32':10.67239916,'33':10.51690585,'34':10.35409873,'35':10.20851499,'36':10.057171813031864,
                '37':9.89989092108759,'38':9.76415887,'39':9.59552818,'40':9.50880841,'41':9.30007834,'42':9.14539478,
                '47':8.371011553000809,'48':8.22539306,'49':8.076,'50':7.924,'51':7.728,'52':7.6090702,'53':7.446361290235207,
                '56':6.98075358781341,'57':6.80477729,'58':6.67044646,'59':6.4881983,'60':6.34915242,'61':6.16060764,'62':6.016835382183124,
                '65':5.52298259,'66':5.37045644,'67':5.21592302,'68':5.05938984,'69':4.9008657, '70':4.740360583177967,
                '70':4.74036065,'71':4.57788602,'72':4.41345434,'73':4.24707941,'74':4.07877622, '75':3.9655107146766664,
                '76':3.79403028,'77':3.62066719,'78':3.50405532,'79':3.26837091,'80':3.14931086,
                '81':2.168501225,'82':2.78733837,'83':2.6651,'84':2.4803,'85':2.2937,'86':2.16850124,
                '96':0.47501992566677664,'97':0.34007389,'98':0.13648822,'99':3.2051002e-15
                }
                
    redshift_dict = {'9':7.5951071498715965,'10':7.23627606616736,'11':7.005417045544533,'12':6.491597745667503,'13':6.0107573988449,'14':5.846613747881867,'15':5.5297658079491026,'16': 5.227580973127337,'17':4.995933468164624,'18':4.664517702470927,'19':4.428033736605549,'20':4.176834914726472,'21':4.0079451114652676,
                    '22':3.7087742646422353,'23':3.4908613692606485,'24':3.2830330579565246,'25':3.008131071630377,'26':2.8957850057274284,'27':2.7331426173187188,'28':2.5772902716018935,
                    '30':2.3161107439568918,'31':2.207925472383703,'32':2.1032696525957713,'33':2.0020281392528516,'34':1.9040895435327672,'35':1.822689252620354,'36':1.7435705743308647,
                    '37':1.6666695561144653,'38':1.6042345220731056,'39':1.5312390291576135,'40':1.4955121664955557,'41':1.4140982203725216, '42':1.3575766674029972,
                    '47':1.1141505637653806,'48':1.074457894547674,'49':1.035510445664141,'50':0.9972942257819404,'51':0.9505313515850327,'52':0.9230008161779089,'53':0.8868969375752482,
                    '56':0.7910682489463392,'57':0.7574413726158526,'58':0.7326361820223115,'59':0.7001063537185233,'60':0.6761104112134777,'61':0.644641840684537,'62':0.6214287452425136,
                    '65':0.5463921831410221,'66':0.524565820433923,'67':0.5030475232448832,'68':0.4818329434209512,'69':0.4609177941806475,'70':0.4402978492477432,
                    '70':0.4402978492477432,'71':0.41996894199726653,'72':0.3999269646135635,'73':0.38016786726023866,'74':0.36068765726181673,'75':0.3478538418581776,
                    '76':0.32882972420595435,'77':0.31007412012783386,'78':0.2977176845174465,'79':0.2733533465784399,'80':0.2613432561610123,
                    '81':0.24354018155467028,'82':0.22598838626019768,'83':0.21442503551449454,'84':0.19728418237600986,'85':0.1803852617057493,'86':0.1692520332436107,
                    '96':0.0337243718735154,'97':0.023974428382762536,'98':0.009521666967944764,'99':2.220446049250313e-16
                    }
    return lbt_dict, redshift_dict
 
# Using the redshift bin center and the width in Gyr, look up a list of snaps to include:
def which_snaps_to_include(snapnum, width_gyr, min, max):
    print('min and max', min, max)
    
    lbt_dict, redshift_dict = lookup_dicts()
    
                    
    center_lbt = lbt_dict[str(snapnum)]
    
    # First go back in time: 
    list_snaps = []
    list_snaps.append(snapnum)
    
    i = 1
    #while (abs(center_lbt - lbt_dict[str(snapnum-i)]) < 0.5) & ((snapnum - i) > min):
    while ((snapnum - i) > min):
        try:
            if (abs(center_lbt - lbt_dict[str(snapnum-i)]) < width_gyr): #changing 0.25 to width_gyr
                list_snaps.append(snapnum-i)
        except:
            print('snap doesnt exist in dictionary')
        i+=1
    
    # Now go in the other direction
    i = 1
    #while (abs(center_lbt - lbt_dict[str(snapnum+i)]) < 0.5) & ((snapnum + i) < max):
    while ((snapnum + i) < max):
        try:
            if (abs(center_lbt - lbt_dict[str(snapnum+i)]) < width_gyr):#changing 0.25 to width_gyr
                list_snaps.append(snapnum+i)
        except:
            print('snap doesnt exist in dictionary')
        i+=1
    sort_list = np.sort(list_snaps)
    return sort_list
    


