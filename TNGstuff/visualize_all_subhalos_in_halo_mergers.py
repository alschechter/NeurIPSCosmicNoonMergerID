#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                     
# Goal: From the tables of mergers, load the subhalo ID of the merger,
# then, use the merger tree to grab all subhalos that are part of the same halo
# and using this code, visualize them using particle data. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                      
import illustris_python as il
import h5py
import numpy as np
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt, in order to save figures on the supercomputer, idk why
import matplotlib.pyplot as plt
import pandas as pd

basePath = '/n/hernquistfs3/IllustrisTNG/Runs/'
basePath = '/n/holylfs/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L35n2160TNG'                                                                                            
sP = basePath+run+'/output/'
snapnum = 51 #13 is z = 6, 21 is z = 4  # 33 is z = 2      
scaling = 0.3333#0.1429 is for 13                                                                                                           
treeName_default="Sublink"

h = 0.6774

#def loadSubhalo(basePath, snapNum, id, partType, fields=None):

# Go and grab the subhalo IDs from the merging catalogs you are making
def load_table(snap, run):
    table = pd.read_table('Tables/mergers_snap_'+str(snap)+'_'+run+'.txt')
    print(table['Subfind_ID'].values)
    return table['Subfind_ID'].values

def main():

    id_list = load_table(snapnum, run)
    for i in range(len(id_list)):
        
        subhalo_info = il.groupcat.loadSubhalos(sP, snapnum, fields=['SubhaloGrNr','SubhaloParent'])
        # Okay so you have to load the entire groupcat apparently
        print('subhaloid',id_list[i])
        print(subhalo_info)
        #print(subhalo_info[id_list[i]])
        parentID = subhalo_info['SubhaloGrNr'][id_list[i]]
        print('parentID',parentID)
        print('subhaloparent index local to the group', subhalo_info['SubhaloParent'][id_list[i]])
        
        # Okay now that I've found out what the id of the halo is,
        # go to the group catalog for the halos and get as much information as you can!
        
        halo_info = il.groupcat.loadHalos(sP,snapnum,fields=['GroupFirstSub','GroupNsubs'])
        print('largest subhalo in this group', halo_info['GroupFirstSub'][parentID], 'number of subhalos in group', halo_info['GroupNsubs'][parentID])
        
        
        
        subhalo = il.snapshot.loadHalo(sP, snapnum, parentID, 'stars')
        # There are a bunch of different fields, we want to get coordinates and mass
        x = (scaling/h)*subhalo['Coordinates'][:,0]#-cen_x
        y = (scaling/h)*subhalo['Coordinates'][:,1]#-cen_y
        density = np.log10(subhalo['Masses'][:])
        counts,ybins,xbins,image = plt.hist2d(x,y,weights = density, bins=[50,50])


        # Wouldn't it be neat to add some gas contours on top?
        subhalo_gas = il.snapshot.loadHalo(sP, snapnum, parentID, 'gas')
        xs_gas = (scaling/h)*subhalo_gas['Coordinates'][:,0]
        ys_gas = (scaling/h)*subhalo_gas['Coordinates'][:,1]
        density_gas = np.log10(subhalo_gas['Masses'][:])
        countsg,ybinsg,xbinsg,imageg = plt.hist2d(xs_gas,ys_gas,weights = density_gas, bins=[100,100])#,norm=LogNorm())
        
        plt.clf()
        contfill = plt.contour(counts, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap='magma_r', alpha=0.5)
        cont = plt.contour(countsg, extent=[xbinsg.min(), xbinsg.max(), ybinsg.min(), ybinsg.max()], linewidths=1, cmap='Blues', zorder=1, alpha=0.5)#, colors='Blues')
        plt.xlabel('$\Delta x$ [kpc]')
        plt.ylabel('$\Delta x$ [kpc]')

        plt.savefig('Figs/merger_vis_halo_'+str(run)+'_'+str(snapnum)+'_'+str(id_list[i])+'.png')


        # Also make a 2d histogram visualization because it looks better
        plt.clf()
        plt.hist2d(xs_gas, ys_gas, weights=density_gas, bins=[150,100])
        plt.xlabel('$\Delta x$ [kpc]')
        plt.ylabel('$\Delta x$ [kpc]')
        plt.savefig('Figs/merger_vis_2d_hist_halo_'+str(run)+'_'+str(snapnum)+'_'+str(id_list[i])+'.png')



if __name__ == "__main__":
    main()


