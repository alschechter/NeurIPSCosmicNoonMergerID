#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                     
# Goal: Use snapshot.py (from il) to load up individual subhalos and visualize
# them using the particle data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                      
import illustris_python as il
import h5py
import numpy as np
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt, in order to save figures on the supercomputer                                                                                 
import matplotlib.pyplot as plt


basePath = '/n/hernquistfs3/IllustrisTNG/Runs/'
basePath = '/n/holylfs/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L35n2160TNG'                                                                                            
sP = basePath+run+'/output/'
snapnum = 33 #13 is z = 6, 21 is z = 4  # 33 is z = 2      
scaling = 0.3333#0.1429 is for 13                                                                                                           
treeName_default="Sublink"

h = 0.704

#def loadSubhalo(basePath, snapNum, id, partType, fields=None):

def main():
    id_list = [325267, 451247, 402857, 285197, 273197, 480438, 388490, 317296, 295666, 387772, 387852, 502652, 513602, 365478, 393262, 247950, 339759, 499634, 435157, 381607, 366086, 266421, 579891, 536442, 386777, 251892, 444769, 472105, 427696, 521388, 332974, 215430, 434433, 480909, 509202, 264770, 375440, 544108, 436886, 374926, 478371, 463242, 483197, 430779, 347290, 352406, 436155, 438407, 329914, 214782, 364626, 548775, 508704, 524777, 266010, 175652, 270540, 487762, 203628, 210627, 366930, 161635, 459763, 145922, 178689, 308269, 541535, 462560, 117098, 334862, 390354, 152211, 347713, 479504, 119507, 497176, 126349, 126347, 461405, 303312, 107194, 376646, 125111, 449464, 114132, 100781, 499610, 327217, 536958, 111581, 88654, 88649, 83579, 377170, 60750, 446181, 48462, 42423, 46540, 507233, 291976, 21554, 482543, 485593, 407684]
    for i in range(len(id_list)):
        subhalo = il.snapshot.loadSubhalo(sP, snapnum, id_list[i], 'stars')
        # There are a bunch of different fields, we want to get coordinates and mass
        x = (scaling/h)*subhalo['Coordinates'][:,0]#-cen_x
        y = (scaling/h)*subhalo['Coordinates'][:,1]#-cen_y
        density = np.log10(subhalo['Masses'][:])
        counts,ybins,xbins,image = plt.hist2d(x,y,weights = density, bins=[50,50])


        # Wouldn't it be neat to add some gas contours on top?
        subhalo_gas = il.snapshot.loadSubhalo(sP, snapnum, id_list[i], 'gas')
        xs_gas = (scaling/h)*subhalo_gas['Coordinates'][:,0]
        ys_gas = (scaling/h)*subhalo_gas['Coordinates'][:,1]
        density_gas = np.log10(subhalo_gas['Masses'][:])
        countsg,ybinsg,xbinsg,imageg = plt.hist2d(xs_gas,ys_gas,weights = density_gas, bins=[100,100])#,norm=LogNorm())
        
        plt.clf()
        contfill = plt.contour(counts, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap='magma_r', alpha=0.5)
        cont = plt.contour(countsg, extent=[xbinsg.min(), xbinsg.max(), ybinsg.min(), ybinsg.max()], linewidths=1, cmap='Blues', zorder=1, alpha=0.5)#, colors='Blues')
        plt.xlabel('$\Delta x$ [kpc]')
        plt.ylabel('$\Delta x$ [kpc]')

        plt.savefig('Figs/merger_vis_'+str(run)+'_'+str(snapnum)+'_'+str(id_list[i])+'.png')


        # Also make a 2d histogram visualization because it looks better
        plt.clf()
        plt.hist2d(xs_gas, ys_gas, weights=density_gas, bins=[150,100])
        plt.xlabel('$\Delta x$ [kpc]')
        plt.ylabel('$\Delta x$ [kpc]')
        plt.savefig('Figs/merger_vis_2d_hist_'+str(run)+'_'+str(snapnum)+'_'+str(id_list[i])+'.png')



if __name__ == "__main__":
    main()


