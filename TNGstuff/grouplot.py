import illustris_python as il
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt:
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np

run = 'L35n2160TNG'
#run = 'L75n1820TNG'
snapnum = 16# z = 2, #21 # z = 4
basePath = '/n/hernquistfs3/IllustrisTNG/Runs/'
basePath = '/n/holylfs/LABS/hernquist_lab/IllustrisTNG/Runs/'

def plot_individuals(ids, mass, sfr):
    for j in range(len(ids)):
        #plt.annotate(str(ids[j]), xy=(mass[ids[j]], sfr[ids[j]]), xycoords='data', color='red')

        plt.scatter(mass[ids[j]], sfr[ids[j]], color='red', zorder=3)


def plot_sfr_mass(run, snapnum, subhalos, id_list):
    #print('shape of the type', subhalos['SubhaloMassType'].info())

    ptNumStars = il.snapshot.partTypeNum('stars') # 4
    mass_msun = subhalos['SubhaloMassType'][:,ptNumStars] * 1e10 / 0.704
    mask_mass_msun_1 = ma.masked_where(subhalos['SubhaloFlag'] ==0, mass_msun)
    mask_mass_msun = ma.masked_where(mask_mass_msun_1 ==0, mask_mass_msun_1)
    mask_sfr_1 = ma.masked_where(subhalos['SubhaloFlag'] ==0, subhalos['SubhaloSFRinRad'])
    mask_sfr = ma.masked_where(mask_mass_msun_1 ==0, mask_sfr_1)
    plt.plot(mask_mass_msun,mask_sfr,'.', color='black', zorder=1)

    # Now try to only plot certain mass ranges
    lower = 1e8
    upper = 1e11
   
    print('mask mass', mask_mass_msun)
    print('lower and upper', lower, upper)
    
    select_indices = np.where(np.logical_and(mask_mass_msun > lower, mask_mass_msun < upper)) 
    
    mass_msun_select = mask_mass_msun[select_indices]
    sfr_select = mask_sfr[select_indices]
    plt.plot(mass_msun_select, sfr_select,'.', color='orange', zorder=2)
    
    plot_individuals(id_list, mask_mass_msun, mask_sfr)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Stellar Mass Total [$M_\odot$]')
    plt.ylabel('Star Formation Rate [$M_\odot / yr$]')
    
    plt.xlim([1e7,10**(10.5)])
    plt.ylim([10**(-2.5),1e2])

    plt.annotate('# Subhalos = '+str(mask_sfr.count()), xy=(0.01,0.92), xycoords='axes fraction')
    plt.annotate('# Select = '+str(mass_msun_select.count()), xy=(0.01,0.85), xycoords='axes fraction', color='orange')
    plt.savefig('Figs/TNG_total_stellar_mass_'+str(run)+'_'+str(snapnum)+'.png', dpi=500)


def get_subhalos(run, snapnum, basePath):
    fields = ['SubhaloMassType','SubhaloMass', 'SubhaloSFRinRad', 'SubhaloFlag']
    return il.groupcat.loadSubhalos(basePath+run+'/output/', snapnum, fields=fields)


def main():
    #id_list = [407684,291976,366930,387852] #33
    #id_list = [19095,188147,43415] # 21
    id_list = [3717]#, 59872,10849]
    subhalos = get_subhalos(run, snapnum, basePath)
    plot_sfr_mass(run, snapnum, subhalos, id_list)

if __name__ == "__main__":
    main()
    
