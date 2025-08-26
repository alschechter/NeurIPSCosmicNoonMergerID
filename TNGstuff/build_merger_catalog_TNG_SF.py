#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is code for Aimee's TNG50 SF in mergers project.
# The goal is to build merger tables centered on each snapshot
# that include the IDs of all galaxies that will or have merged
# within the 0.5 Gyr window.

# This is similar to the previous code except now I'm considering
# a bunch of different bin centers

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
import time
start_time = time.time()
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt:                                                                                    
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
import os
import illustris_python as il
import util_mergers as u

#for slurm
sys.path.append(os.getcwd())
# This was the old basepath
basePath = '/n/hernquistfs3/IllustrisTNG/Runs/'

# Now the new place where illustris lives:
basePath = '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L205n2500TNG'#'L75n1820TNG'#'L75n1820TNG'#'L35n2160TNG'
sP = basePath+run+'/output/'




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
    subids, select_mass, select_npart, all_mass = u.get_subhalos(run, snapnum, basePath, ll, ul, 1, h, 'no')
    
    all_is = np.linspace(0,len(all_mass)-1, len(all_mass))
    subids_all = [int(x) for x in all_is]

    print('length of all subhalos at this snapshot', len(subids))
    
    print('getting subhalos from the center snapshot', snapnum_center)
    
    subids_f, select_mass_f, select_npart_f, all_mass_f = u.get_subhalos(run, snapnum_center, basePath, 0, 10**14, 1, h, 'yes')
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
        try:
            tree = il.sublink.loadTree(sP,snapnum, subids[i],fields = fields)#
            #print(type(tree))
        except ValueError:
            continue
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
        
        
        '''
        if abs(snapnum - snapnum_center) == 3:
            if snapnum > snapnum_center:
                type_merg='Progenitor' # meaning that we're looking for progenitors at the center
            else:
                type_merg='Descendant' # meaning that we're looking for descendants at the center
            numMergers, Subfind_ID_array, ratio_out = u.mergers_three_steps(tree,ratio_min, subids[i], fields, type_merg)
            
        
        if abs(snapnum - snapnum_center) == 2:
            if snapnum > snapnum_center:
                type_merg='Progenitor' # meaning that we're looking for progenitors at the center
            else:
                type_merg='Descendant' # meaning that we're looking for descendants at the center
            if all_prog:
                numMergers, Subfind_ID_array, ratio_out = u.mergers_two_steps_all_prog(tree,ratio_min, subids[i], fields, type_merg)#mergers_two_steps_all_prog
            else:
                
                numMergers, Subfind_ID_array, ratio_out = u.mergers_two_steps(tree,ratio_min, subids[i], fields, type_merg)
        '''  
        if (abs(snapnum - snapnum_center) == 1) or (abs(snapnum - snapnum_center) == 0):
            try:
                numMergers, Subfind_ID_array, ratio_out = u.mergers_one_step(tree,ratio_min)
            except TypeError: #tree has class nonetype for some reason
                continue
        else:
            n_steps = abs(snapnum - snapnum_center)
            if snapnum > snapnum_center:
                type_merg='Progenitor'
            else:
                type_merg='Descendant'
            try:
                numMergers, Subfind_ID_array, ratio_out = u.mergers_n_steps(sP, tree,ratio_min, subids[i], fields, type_merg, n_steps)
            except TypeError: #tree has class nonetype for some reason
                continue 
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
    print('fractional', float(nmerg)/float(count))
    print('merger id list', mergers_id)
    print('mass ratios', ratio_list)
    print('number of stellar particles', particle_number)
    print('stellar mass', stellar_mass_list)
    print('descendent ids', desc_list)
    print('first prog ids', first_prog_list)
    print('next prog ids', next_prog_list)
    
    
    
    # We need to make a table with useful stuff about mergers in it
    if all_prog:
        #file_check = 'Tables/Aimee_SF/'+str(snapnum_full)+'/mergers_snap_all_prog_'+str(snapnum)+'_rel_to_'+str(snapnum_center)+'_'+run+'.txt'
    
        file_out = open('Tables/Aimee_SF/'+str(snapnum_full)+'/mergers_snap_all_prog_from_'+str(snapnum)+'_at_'+str(snapnum_center)+'_'+run+'.txt','w')
    else:
        file_out = open('Tables/Aimee_SF/'+str(snapnum_full)+'/mergers_snap_from_'+str(snapnum)+'_at_'+str(snapnum_center)+'_'+run+'.txt','w')
    file_out.write('Subfind_ID'+'\t'+'t-t_merg'+'\t'+'q'+'\t'+'Type'+'\t'+'mass'+'\n')
    for j in range(nmerg):
        if snapnum==snapnum_center:# in other words if you're currently on the full snap:
            file_out.write(str(mergers_id[j])+'\t'+str(lbt_snap - lbt_center)+'\t'+str(ratio_list[j])+'\t'+'Merger'+'\t'+str(stellar_mass_list[j])+'\n')
        # snapnum_full - snapnum == 1
        if snapnum_center - snapnum > 0: #this would happen for snapnum 49 i.e., you want to get descendants one time step forward
            
            try:
                file_out.write(str(desc_list[j])+'\t'+str(lbt_snap - lbt_center)+'\t'+str(ratio_list[j])+'\t'+'Descendant'+'\t'+str(all_mass_f[desc_list[j]])+'\n')
            except:
                continue
        # snapnum - snapnum_full == 1:
        # So this would be 17 is the bin center and 13 is earlier in time, so descendants of these would be at the bin center ^
        if snapnum_center - snapnum  < 0: #this would happen for snapnum 51 i.e., you want to get both progenitors :)
            # You also have to re-solve for their stellar masses
            
            if all_prog:
                for k in range(len(first_prog_list[j])):
                    file_out.write(str(first_prog_list[j][k])+'\t'+str(lbt_snap - lbt_center)+'\t'+str(ratio_list[j])+'\t'+'progenitor'+'\t'+str(all_mass_f[first_prog_list[j][k]])+'\n')
            else:
                file_out.write(str(first_prog_list[j])+'\t'+str(lbt_snap - lbt_center)+'\t'+str(ratio_list[j])+'\t'+'first_progenitor'+'\t'+str(all_mass_f[first_prog_list[j]])+'\n')
                file_out.write(str(next_prog_list[j])+'\t'+str(lbt_snap - lbt_center)+'\t'+str(ratio_list[j])+'\t'+'next_progenitor'+'\t'+str(all_mass_f[next_prog_list[j]])+'\n')

            '''
            if abs(snapnum - snapnum_full) ==2: # then we're getting lots and lots of progenitors
                if all_prog:
                    for k in range(len(first_prog_list[j])):
                        file_out.write(str(first_prog_list[j][k])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'progenitor'+'\t'+str(all_mass_f[first_prog_list[j][k]])+'\n')
                else:
                    file_out.write(str(first_prog_list[j])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'first_progenitor'+'\t'+str(all_mass_f[first_prog_list[j]])+'\n')
                    file_out.write(str(next_prog_list[j])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'next_progenitor'+'\t'+str(all_mass_f[next_prog_list[j]])+'\n')

                    
            else:              
                file_out.write(str(first_prog_list[j])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'first_progenitor'+'\t'+str(all_mass_f[first_prog_list[j]])+'\n')
                file_out.write(str(next_prog_list[j])+'\t'+str(lbt_snap - lbt_bin_center)+'\t'+str(ratio_list[j])+'\t'+'next_progenitor'+'\t'+str(all_mass_f[next_prog_list[j]])+'\n')
            '''

        #file_out.write(str(mergers_id[j])+'\t'+str('Descendent')+'\t'+str(desc_list[j])+'\n')
        #file_out.write(str(mergers_id[j])+'\t'+str('First_prog')+'\t'+str(first_prog_list[j])+'\n')
        #file_out.write(str(mergers_id[j])+'\t'+str('Next_prog')+'\t'+str(next_prog_list[j])+'\n')
    print('close file')
    file_out.close()
'''   
if __name__ == "__main__":
    main()
'''


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

#### Aimee commented out on 6/6/24 to test this code
# snapnum_list = [15,16,17,18,19,
#                 24,25,26,27,
#                 32,33,34,
#                 39,40,41,
#                 49,50,51,
#                 58,59,60,
#                 66,67,68,
#                 71,72,73,
#                 77,78,79,
#                 83,84,85]
# snapnum_full_list = [17,17,17,17,17,
#                     25,25,25,25,
#                     33,33,33,
#                     40,40,40,
#                     50,50,50,
#                     59,59,59,
#                     67,67,67,
#                     72,72,72,
#                     78,78,78,
#                     84,84,84]
                    
                    
# snapnum_list = [49,50,51,
#                 83,84,85]
# snapnum_full_list = [50,50,50,
#                     84,84,84]
                    

                    
# snapnum_list = [9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21,
#     9,10,11,12,13,14,15,16,17,18,19,20,21]
# snapnum_full_list = [17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     17,17,17,17,17,17,17,17,17,17,17,17,17]
# snapnum_center_list = [9,9,9,9,9,9,9,9,9,9,9,9,9,
#     10,10,10,10,10,10,10,10,10,10,10,10,10,
#     11,11,11,11,11,11,11,11,11,11,11,11,11,
#     12,12,12,12,12,12,12,12,12,12,12,12,12,
#     13,13,13,13,13,13,13,13,13,13,13,13,13,
#     14,14,14,14,14,14,14,14,14,14,14,14,14,
#     15,15,15,15,15,15,15,15,15,15,15,15,15,
#     16,16,16,16,16,16,16,16,16,16,16,16,16,
#     17,17,17,17,17,17,17,17,17,17,17,17,17,
#     18,18,18,18,18,18,18,18,18,18,18,18,18,
#     19,19,19,19,19,19,19,19,19,19,19,19,19,
#     20,20,20,20,20,20,20,20,20,20,20,20,20,
#     21,21,21,21,21,21,21,21,21,21,21,21,21]


# snapnum_list = [22,23,24,25,26,27,28,
#     22,23,24,25,26,27,28,
#     22,23,24,25,26,27,28,
#     22,23,24,25,26,27,28,
#     22,23,24,25,26,27,28,
#     22,23,24,25,26,27,28,
#     22,23,24,25,26,27,28]
# snapnum_full_list = [
#     25,25,25,25,25,25,25,
#     25,25,25,25,25,25,25,
#     25,25,25,25,25,25,25,
#     25,25,25,25,25,25,25,
#     25,25,25,25,25,25,25,
#     25,25,25,25,25,25,25,
#     25,25,25,25,25,25,25]
# snapnum_center_list = [22,22,22,22,22,22,22,
#     23,23,23,23,23,23,23,
#     24,24,24,24,24,24,24,
#     25,25,25,25,25,25,25,
#     26,26,26,26,26,26,26,
#     27,27,27,27,27,27,27,
#     28,28,28,28,28,28,28]


# snapnum_list = [30,31,32,33,34,35,36,
#     30,31,32,33,34,35,36,
#     30,31,32,33,34,35,36,
#     30,31,32,33,34,35,36,
#     30,31,32,33,34,35,36,
#     30,31,32,33,34,35,36,
#     30,31,32,33,34,35,36]
# snapnum_full_list = [33,33,33,33,33,33,33,
#     33,33,33,33,33,33,33,
#     33,33,33,33,33,33,33,  
#     33,33,33,33,33,33,33, 
#     33,33,33,33,33,33,33, 
#     33,33,33,33,33,33,33, 
#     33,33,33,33,33,33,33]
# snapnum_center_list = [30,30,30,30,30,30,30,
#     31,31,31,31,31,31,31,
#     32,32,32,32,32,32,32,  
#     33,33,33,33,33,33,33, 
#     34,34,34,34,34,34,34, 
#     35,35,35,35,35,35,35, 
#     36,36,36,36,36,36,36]


# snapnum_list = [37,38,39,40,41,42,
#     37,38,39,40,41,42,
#     37,38,39,40,41,42,
#     37,38,39,40,41,42,
#     37,38,39,40,41,42,
#     37,38,39,40,41,42]
# snapnum_full_list = [40,40,40,40,40,40,
#     40,40,40,40,40,40,  
#     40,40,40,40,40,40,
#     40,40,40,40,40,40,
#     40,40,40,40,40,40,
#     40,40,40,40,40,40]
# snapnum_center_list = [37,37,37,37,37,37,
#     38,38,38,38,38,38,  
#     39,39,39,39,39,39,
#     40,40,40,40,40,40,
#     41,41,41,41,41,41,
#     42,42,42,42,42,42]


# snapnum_list = [47,48,49,50,51,52,53,
#     47,48,49,50,51,52,53,
#     47,48,49,50,51,52,53,
#     47,48,49,50,51,52,53,
#     47,48,49,50,51,52,53,
#     47,48,49,50,51,52,53,
#     47,48,49,50,51,52,53]
# snapnum_full_list = [50,50,50,50,50,50,50,
#     50,50,50,50,50,50,50,  
#     50,50,50,50,50,50,50,
#     50,50,50,50,50,50,50,
#     50,50,50,50,50,50,50,
#     50,50,50,50,50,50,50,
#     50,50,50,50,50,50,50]
# snapnum_center_list = [47,47,47,47,47,47,47,
#     48,48,48,48,48,48,48,  
#     49,49,49,49,49,49,49,
#     50,50,50,50,50,50,50,
#     51,51,51,51,51,51,51,
#     52,52,52,52,52,52,52,
#     53,53,53,53,53,53,53]


# snapnum_list = [56,57,58,59,60,61,62,
#     56,57,58,59,60,61,62,
#     56,57,58,59,60,61,62,
#     56,57,58,59,60,61,62,
#     56,57,58,59,60,61,62,
#     56,57,58,59,60,61,62,
#     56,57,58,59,60,61,62,
#     56,57,58,59,60,61,62]
# snapnum_full_list = [59,59,59,59,59,59,59,
#     59,59,59,59,59,59,59,
#     59,59,59,59,59,59,59,
#     59,59,59,59,59,59,59,
#     59,59,59,59,59,59,59,
#     59,59,59,59,59,59,59,
#     59,59,59,59,59,59,59]
# snapnum_center_list = [56,56,56,56,56,56,56,
#     57,57,57,57,57,57,57,
#     58,58,58,58,58,58,58,
#     59,59,59,59,59,59,59,
#     60,60,60,60,60,60,60,
#     61,61,61,61,61,61,61,
#     62,62,62,62,62,62,62]


# snapnum_list = [65,66,67,68,69,70,
#     65,66,67,68,69,70,
#     65,66,67,68,69,70,
#     65,66,67,68,69,70,
#     65,66,67,68,69,70,
#     65,66,67,68,69,70]
# snapnum_full_list = [67,67,67,67,67,67,
#     67,67,67,67,67,67,
#     67,67,67,67,67,67,
#     67,67,67,67,67,67,
#     67,67,67,67,67,67,
#     67,67,67,67,67,67]
# snapnum_center_list = [65,65,65,65,65,65,
#     66,66,66,66,66,66,
#     67,67,67,67,67,67,
#     68,68,68,68,68,68,
#     69,69,69,69,69,69,
#     70,70,70,70,70,70]
   
#69, 70, 71, 72, 73, 74, 75
# snapnum_list = [69,70,71,72,73,74,75,
#     69,70,71,72,73,74,75,
#     69,70,71,72,73,74,75,
#     69,70,71,72,73,74,75,
#     69,70,71,72,73,74,75,
#     69,70,71,72,73,74,75,
#     69,70,71,72,73,74,75]
# snapnum_full_list = [72,72,72,72,72,72,72,
#     72,72,72,72,72,72,72,
#     72,72,72,72,72,72,72,
#     72,72,72,72,72,72,72,
#     72,72,72,72,72,72,72,
#     72,72,72,72,72,72,72,
#     72,72,72,72,72,72,72]
# snapnum_center_list = [69,69,69,69,69,69,69,
#     70,70,70,70,70,70,70,
#     71,71,71,71,71,71,71,
#     72,72,72,72,72,72,72,
#     73,73,73,73,73,73,73,
#     74,74,74,74,74,74,74,
#     75,75,75,75,75,75,75]
    
#75, 76, 77, 78, 79, 80]
# snapnum_list = [75,76,77,78,79,80,
#     75,76,77,78,79,80,
#     75,76,77,78,79,80,
#     75,76,77,78,79,80,
#     75,76,77,78,79,80,
#     75,76,77,78,79,80]
# snapnum_full_list = [78,78,78,78,78,78,
#     78,78,78,78,78,78,
#     78,78,78,78,78,78,
#     78,78,78,78,78,78,
#     78,78,78,78,78,78,
#     78,78,78,78,78,78]
# snapnum_center_list = [75,75,75,75,75,75,
#     76,76,76,76,76,76,
#     77,77,77,77,77,77,
#     78,78,78,78,78,78,
#     79,79,79,79,79,79,
#     80,80,80,80,80,80]
   
# snapnum_list = [83,84,85,
#     83,84,85,
#     83,84,85]
# snapnum_full_list = [84,84,84,
#     84,84,84,  
#     84,84,84]
# snapnum_center_list = [83,83,83, 
#     84,84,84, 
#     85,85,85]

#aimee modified:
# snapnum_list = [81,82,83,84,85,86,
#     81,82,83,84,85,86,
#     81,82,83,84,85,86,
#     81,82,83,84,85,86,
#     81,82,83,84,85,86,
#     81,82,83,84,85,86]
# snapnum_full_list = [84,84,84,84,84,84,
#     84,84,84,84,84,84,
#     84,84,84,84,84,84,
#     84,84,84,84,84,84,
#     84,84,84,84,84,84,
#     84,84,84,84,84,84]
# snapnum_center_list = [81,81,81,81,81,81,
#     82,82,82,82,82,82,                    
#     83,83,83,83,83,83, 
#     84,84,84,84,84,84,
#     85,85,85,85,85,85,
#     86,86,86,86,86,86]
   
snapnum_list = [96,97,98,99,
    96,97,98,99,
    96,97,98,99,
    96,97,98,99]
snapnum_full_list = [99,99,99,99,
    99,99,99,99,
    99,99,99,99,
    99,99,99,99]
snapnum_center_list = [96,96,96,96,
    97,97,97,97,
    98,98,98,98,
    99,99,99,99]

# snapnum_list = [24,25,26,27, 4x4
#     24,25,26,27,
#     24,25,26,27,
#     24,25,26,27]
# snapnum_full_list = [
#     25,25,25,25,
#     25,25,25,25,
#     25,25,25,25,
#     25,25,25,25]
# snapnum_center_list = [
#     24,24,24,24,
#     25,25,25,25,
#     26,26,26,26,
#     27,27,27,27]
    



counter_loop = 0
for snapnum in snapnum_list:
    
    
    
    try:
        snapnum_full = snapnum_full_list[counter_loop]
    except IndexError:
        print('counter_loop', counter_loop)
        print('length of snapnum_full_list', len(snapnum_full_list))
        print('snapnum last', snapnum_full_list[-1])
    
    snapnum_center = snapnum_center_list[counter_loop]
    
    '''
    if snapnum_center != 18:
        counter_loop+=1
        continue
    '''
    
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('finding mergers at this snapnum', snapnum)
    print('tracing them to this snapshot', snapnum_center)
    print('this is the bin center', snapnum_full)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    all_prog = False
    
    # First check to see if the file already exists. If so, skip this snap:
    if all_prog:
        file_check = 'Tables/Aimee_SF/'+str(snapnum_full)+'/mergers_snap_all_prog_from_'+str(snapnum)+'_at_'+str(snapnum_center)+'_'+run+'.txt'
    else:
        file_check = 'Tables/Aimee_SF/'+str(snapnum_full)+'/mergers_snap_from_'+str(snapnum)+'_at_'+str(snapnum_center)+'_'+run+'.txt'
        
    if os.path.exists(file_check):
        print('already exists ')
        
        counter_loop+=1
        continue
    else:
        print('does not exist yet ', file_check)
        
    
    # Get the dictionaries
    lbt_dict, redshift_dict = u.lookup_dicts()
    print('line before stuck', lbt_dict.keys())
    
    lbt_snap = lbt_dict[str(snapnum)]
    lbt_center = lbt_dict[str(snapnum_center)]
    lbt_full = lbt_dict[str(snapnum_full)]
    
    
    
    if (abs(lbt_full - lbt_snap) > 0.5) or (abs(lbt_full - lbt_center) > 0.5): #I think these should be 0.5?
        print('outside of redshift bin')
        
        counter_loop+=1
        continue
    
    # First check that the one you're trying to find mergers in is within 0.5 Gyr of the snap youre centering on
    if abs(lbt_snap - lbt_center) > 0.5:
        print('NOT running', 'difference to the one Im centering on', abs(lbt_dict[str(snapnum)] - lbt_dict[str(snapnum_center)]))
        print('center:', lbt_center)
        print('snap:', lbt_snap)
        counter_loop+=1
        continue
    else:
        print('Running, difference in lbt', abs(lbt_snap - lbt_center))
        
    
    
    
    # Now get the list of snaps to include relative to the snapshot you're centering on
    list_snaps_redshift = u.which_snaps_to_include(snapnum_full, 0.5, 0, 98)
    
    # Get the list of snaps to include relative to the redshift bin center
    list_snaps_bin_center = u.which_snaps_to_include(snapnum_center, 0.5, list_snaps_redshift[0]-1, list_snaps_redshift[-1]+1)
    print('checking problem snapnum center: ', snapnum_center)
    print('checking problem min snap: ', list_snaps_redshift[0]-1)
    print('checking problem max snap: ', list_snaps_redshift[-1]+1)
    print('list snaps bin center', list_snaps_bin_center)
    print('list snaps redshift bin', list_snaps_redshift)
    
    # Now restrict to snaps included only in both
    
    list1_as_set = set(list_snaps_bin_center)
    list_snaps = list(list1_as_set.intersection(list_snaps_redshift))
    
    print('can find mergers in these snaps', list_snaps) 
    
    descendants = True
    progenitors = True
    if snapnum == list_snaps[0]:# This means its the earliest in the bin, so we are only taking
        # We are not taking descendants
        # I think this only matters for the join step
        descendants = False
    if snapnum == list_snaps[-1]:
        progenitors = False
    
    if snapnum in list_snaps:
        print('Snap in snap list')
        print('Include descendants?', descendants)
        print('Include progenitors?', progenitors)
    else:
        print('Snap not in list, continuing')
        print(list_snaps)
        print(snapnum)
        
        counter_loop+=1
        continue

    # Just for my own notes about which are included in the 250Myr window 
    # (500 Myr centered on z=1)
    treeName_default="Sublink"
    

    # This function uses the groupcat.loadSubhalos function to grab all the IDs and details
    # of galaxies within a certain range of stellar masses at a snapshot
    
    # As a note, I have min_npart as an optional input here but Jacob was saying that its
    # not necessary to select by # of particles; this has a nearly 1:1 relationship with
    # selecting by stellar mass.
    
    
    main()
    counter_loop+=1

np.savetxt('TableCodeTime.txt', np.array([time.time() - start_time]))