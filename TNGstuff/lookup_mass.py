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
snapnum = 13 #13 is z = 6, 21 is z = 4  # 33 is z = 2      
treeName_default="Sublink"

h = 0.6774

id_list = [4308]


def main():
    # Look up all subhalos
    subhalos = il.groupcat.loadSubhalos(sP, snapnum, fields=['SubhaloMassType'])
    masses = []
    for i in range(len(id_list)):
        masses.append(subhalos[id_list[i]][4] * 1e10 / h)

    print('IDs ~~~~~~ ' , id_list)
    print('Masses ~~~~~~ ', masses)
if __name__ == "__main__":
    main()


