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
from astropy.cosmology import FlatLambdaCDM
import scipy.stats
import yt

basePath = '/n/hernquistfs3/IllustrisTNG/Runs/'
basePath = '/n/holylfs/LABS/hernquist_lab/IllustrisTNG/Runs/'
run = 'L35n2160TNG'#'L75n1820TNG'#'L35n2160TNG'                                                                                            
sP = basePath+run+'/output/'
path_store = '/n/holystore01/LABS/hernquist_lab/Users/rnevin/hdf5s/'
snapnum = 50 #13 is z = 6, 21 is z = 4  # 33 is z = 2    

treeName_default="Sublink"

# These are all from Planck 2016 (https://arxiv.org/pdf/1812.05609.pdf)
h = 0.6774
H0 = h*100
omega_M = 0.3089
omega_B = 0.0486

size_box = 10# in ckpc, this is the radius, so the diameter is 2x this

cosmo = FlatLambdaCDM(H0=H0, Om0=omega_M, Ob0=omega_B)
print(cosmo)

merg_y_n = 'mergers'




#def loadSubhalo(basePath, snapNum, id, partType, fields=None):

# Go and grab the subhalo IDs from the merging catalogs you are making
def load_table(snap, run, merg):
    table = pd.read_table('Tables/'+merg+'_snap_'+str(snap)+'_'+run+'.txt')
    print(table['Subfind_ID'].values)
    return table['Subfind_ID'].values

def main(merg):

    id_list = load_table(snapnum, run, merg)
    
    for i in range(len(id_list)):
        subhalosh = il.groupcat.loadHeader(sP,snapnum)#, fields=fields_header)                                                                      
        scaling = subhalosh['Time']
        z = subhalosh['Redshift']

        subhalo = il.snapshot.loadSubhalo(sP, snapnum, id_list[i], 'gas')
        # Try writing a hdf5 file to holystore:
        hf = h5py.File(path_store+str(snapnum)+'_'+str(id_list[i])+'.hdf5', 'w')
        hf.create_dataset(data=subhalo)

        # Now read it the heck in:
        ds = yt.load(path_store+str(snapnum)+'_'+str(id_list[i])+'.hdf5')
        STOP


        subhalo_hdf5 = il.snapshot.snapPath(sP, snapnum)
        # See if you can load this into yt
        ds = yt.load(subhalo_hdf5)#, 'dark_matter')

        print(ds)

        subhalobh = il.snapshot.loadSubhalo(sP, snapnum, id_list[i], 4)
        #xcen = (scaling/h)*subhalobh['Coordinates'][0,0]
        #ycen = (scaling/h)*subhalobh['Coordinates'][0,1]
        cen = subhalobh['Coordinates']

        # Make a projection plot :)
        #data = yt.ProjectionPlot(ds, "z", ["gas","density"], width = (20.0, 'kpc')).data #.save()
        #print(data)

        import yt.visualization.eps_writer as eps
        from yt.units import Mpc, kpc
        center = ds.domain_center  # or the location of some halo
        box = ds.box(center - 50*kpc, center + 50*kpc)
        
        slc = yt.SlicePlot(ds, 'z', ['density', 'temperature', 'pressure',
                                     'velocity_magnitude'], 'gas', data_source = box)
        slc.set_width(25, 'kpc')
        eps_fig = eps.multiplot_yt(2, 2, slc, bare_axes=True)
        eps_fig.scale_line(0.2, '5 kpc')
        eps_fig.save_fig('multi', format='eps')




        STOP

        xstars = (scaling/h)*subhalo['Coordinates'][:,0]#-cen_x
        ystars = (scaling/h)*subhalo['Coordinates'][:,1]#-cen_y
        densitystars = np.log10(subhalo['Masses'][:])#SubfindDensity

        stellar_photometry = subhalo['GFM_StellarPhotometrics'][:,3]#sdss r-band
        
        # There are a bunch of different fields, we want to get coordinates and mass
        subhalo_gas = il.snapshot.loadSubhalo(sP, snapnum, id_list[i], 'gas')
        try:
            xgas = (scaling/h)*subhalo_gas['Coordinates'][:,0]#-cen_x
        except:
            continue
        ygas = (scaling/h)*subhalo_gas['Coordinates'][:,1]#-cen_y
        densitygas = np.log10(subhalo_gas['Masses'][:])#Density
        metallicity_ratio = subhalo_gas['GFM_Metallicity'][:]
        
        print(np.shape(xstars), np.shape(xgas))
        #print(np.shape(xstars+xgas))
        x = np.concatenate((xstars,xgas))
        y = np.concatenate((ystars,ygas))
        density = np.concatenate((densitystars, densitygas))
        
        
        # Find the center of mass using the position of the black hole
        subhalobh = il.snapshot.loadSubhalo(sP, snapnum, id_list[i], 4)
        xcen = (scaling/h)*subhalobh['Coordinates'][0,0]
        ycen = (scaling/h)*subhalobh['Coordinates'][0,1]
        
        #counts,ybins,xbins,image = plt.hist2d(x,y,weights = density, bins=[50,50], range=[[xcen-10,xcen+10],[ycen-10,ycen+10]])


        
        '''
        # Also make a 2d histogram visualization because it looks better
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist2d(x, y, weights=density, bins=[1000,1000], range=[[xcen-20,xcen+20],[ycen-20,ycen+20]], cmap='magma')#, norm=matplotlib.colors.LogNorm())
        # The spatial position is measured in ckpc/h, or comoving kpc/h, where h is a historical artifact expression of uncertainty?
        #plt.xlabel('$\Delta x$ [kpc]')
        #plt.ylabel('$\Delta x$ [kpc]')
        ax.set_aspect('equal')
        #plt.axis('off')
        ax.set_axis_off()
        plt.savefig('Figs/quick_pngs/mergers/'+str(run)+'_'+str(snapnum)+'_'+str(id_list[i])+'_10ckpch.png', bbox_inches = 'tight',pad_inches = 0,transparent=True)
        '''

        # I think a more useful plot could be a scipy 2d binned statistic image of the sum of values in each bin (at least for mass):

        plt.clf()
        fig = plt.figure()
        ax=fig.add_subplot(131)
        

        # Currently, we're dividing 4x4 (ckpc) into 100 bins per side (@z=1)
        # How do we actually compare this to the sampling of NIRCam (0.031"/pixel in short wavelength, 0.063"/pixel in long wavelength)?
        
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).value/60
        #print('kpc/arcsec', kpc_per_arcsec)
        #print('4kpc / kpc/arcsec = arcsec', 4/kpc_per_arcsec)
        
        size_box_arc = 2*size_box/kpc_per_arcsec
        
        #print('size box arc', size_box_arc)

        pixelscale = 0.031#"/pixel
        num_pix = int(round(size_box_arc/pixelscale,2))
        print('supposed # of pixels', int(round(size_box_arc/pixelscale,2)))
        STOP
        
        
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(ygas, xgas, metallicity_ratio,statistic='mean', bins=num_pix, range=[[ycen-size_box,ycen+size_box],[xcen-size_box,xcen+size_box]])
        
 
        dim1 = np.flipud(heatmap)
 
        ax.imshow(np.flipud(heatmap), cmap='magma',extent=[yedges[0], yedges[-1], xedges[0],xedges[-1]])

        ax.set_ylim(xedges[0], xedges[-1])
        ax.set_xlim(yedges[0],yedges[-1])
        ax.set_axis_off()
        ax.set_title('Metal Ratio')
        
        ax1=fig.add_subplot(132)

        
        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(ygas, xgas, densitygas,statistic='sum', bins=num_pix, range=[[ycen-size_box,ycen+size_box\
                                                                                                                                       ],[xcen-size_box,xcen+size_box]])


        dim2 = np.flipud(heatmap)

        ax1.imshow(np.flipud(heatmap), cmap='magma',extent=[yedges[0], yedges[-1], xedges[0],xedges[-1]])

        ax1.set_ylim(xedges[0], xedges[-1])
        ax1.set_xlim(yedges[0],yedges[-1])
        ax1.set_axis_off()
        ax1.set_title('Gas mass')

        ax2=fig.add_subplot(133)


        heatmap, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(ystars, xstars, densitystars,statistic='sum', bins=num_pix, range=[[ycen-size_box,ycen+size_box\
                                                                                                                                             ],[xcen-size_box,xcen+size_box]])



        dim3 = np.flipud(heatmap)
        ax2.imshow(np.flipud(heatmap), cmap='magma',extent=[yedges[0], yedges[-1], xedges[0],xedges[-1]])

        ax2.set_ylim(xedges[0], xedges[-1])
        ax2.set_xlim(yedges[0],yedges[-1])
        ax2.set_axis_off()
        ax2.set_title('Stellar mass')        


        plt.savefig('Figs/quick_pngs/'+merg+'/'+str(run)+'_'+str(snapnum)+'_'+str(id_list[i])+'_'+str(2*size_box)+'kpc_NIRCam_short_dust_gas_stars_mass_binnedstats.png', bbox_inches='tight', pad_inches=0,transparent=True)

        # Now, make a data product out of this
        
        cube = np.dstack((dim1,dim2,dim3))
        
        np.save('Numpy_cubes/z_1/'+merg+'/'+str(run)+'_'+str(snapnum)+'_'+str(id_list[i])+'_'+str(2*size_box)+'kpc_NIRCAM_short', cube)
        
        continue

        # Okay but now try to make a 3D (RGB?) image out of the combo of stars, gas, and what?
        plt.clf()                                                                                                                   
        fig = plt.figure()                                                                                                          
        ax = fig.add_subplot(131)                                                                                                   
        ax.hist2d(xstars, ystars, weights=stellar_photometry, bins=[10,10], range=[[xcen-2,xcen+2],[ycen-2,ycen+2]], cmap='magma')
        #ax.hist2d(xgas, ygas, weights=metallicity_ratio, bins=[100,100], range=[[xcen-2,xcen+2],[ycen-2,ycen+2]], cmap='magma')#, norm=matplotlib.colors.LogNorm())                                                                                                              
        # The spatial position is measured in ckpc/h, or comoving kpc/h, where h is a historical artifact expression of uncertainty?                                                                                                                                   
        #plt.xlabel('$\Delta x$ [kpc]')                                                                                             
        #plt.ylabel('$\Delta x$ [kpc]')                                                                                             
        ax.set_aspect('equal')                                                                                                      
        #plt.axis('off')                                                                                                            
        #ax.set_title('Metal Ratio')
        ax.set_title('K-band photometry')
        ax.set_axis_off()       
        
        
        ax1 = fig.add_subplot(132)
        ax1.set_title('Gas mass')
        #ax1.set_title('Gas density')
        ax1.hist2d(xgas, ygas, weights=densitygas, bins=[10,10], range=[[xcen-2,xcen+2],[ycen-2,ycen+2]], cmap='magma')
        ax1.set_aspect('equal')
        ax1.set_axis_off()

        ax2 = fig.add_subplot(133)
        ax2.set_title('Stellar mass')
        #ax2.set_title('Stellar density')
        ax2.hist2d(xstars, ystars, weights=densitystars, bins=[10,10], range=[[xcen-2,xcen+2],[ycen-2,ycen+2]], cmap='magma')
        ax2.set_aspect('equal')
        ax2.set_axis_off()

        plt.savefig('Figs/quick_pngs/mergers/'+str(run)+'_'+str(snapnum)+'_'+str(id_list[i])+'_1ckpch_lowres_gas_stars_mass_photometry.png', bbox_inches = 'tight',pad_inches = 0,transparent=True)  

if __name__ == "__main__":
    main(merg_y_n)

