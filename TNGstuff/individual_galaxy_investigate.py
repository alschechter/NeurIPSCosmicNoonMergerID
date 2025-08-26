import illustris_python as il
import matplotlib
matplotlib.use('agg')# must call this before importing pyplot as plt:
import matplotlib.pyplot as plt

run = 'L35n2160TNG'
#run = 'L75n1820TNG'
snapnum = 33# z = 2, #21 # z = 4
basePath = '/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/'

def list_properties(run, snapnum, subhalos, id_in):
    #print('shape of the type', subhalos['SubhaloMassType'].info())
    print('stellar mass total', 'SFR in Rad')
    print(subhalos['SubhaloMassType'][id_in,4] * 1e10 / 0.704, subhalos['SubhaloSFRinRad'][id_in])


    print('total mass')
    print(subhalos['SubhaloMass'][id_in] * 1e10 / 0.704)
    

    ptNumStars = il.snapshot.partTypeNum('stars') # 4
    mass_msun = subhalos['SubhaloMassInHalfRadType'][:,ptNumStars] * 1e10 / 0.704
    plt.plot(mass_msun,subhalos['SubhaloSFRinRad'],'.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Stellar Mass Total [$M_\odot$]')
    plt.ylabel('Star Formation Rate [$M_\odot / yr$]')
    plt.annotate('# Subhalos = '+str(len(mass_msun)), xy=(0.01,0.92), xycoords='axes fraction')
    plt.savefig('Figs/TNG_'+str(run)+'_'+str(snapnum)+'.png')


def get_subhalos(run, snapnum, basePath):
    fields = ['SubhaloMassType','SubhaloMass', 'SubhaloSFRinRad']
    return il.groupcat.loadSubhalos(basePath+run+'/output/', snapnum, fields=fields)


def main():
    id_input = 29443
    subhalos = get_subhalos(run, snapnum, basePath)
    list_properties(run, snapnum, subhalos, id_input)

if __name__ == "__main__":
    main()
    
