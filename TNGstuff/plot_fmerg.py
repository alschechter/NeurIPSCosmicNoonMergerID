import pandas as pd
import illustris_python as il
import numpy as np
import matplotlib
import random
matplotlib.use('agg')# must call this before importing pyplot as plt: 
import matplotlib.pyplot as plt
import seaborn as sns            

# pull from fmerg tables
snapnum_list = [17,25,33,40,50,59,67,72,84]
redshift_list = [4.995933468164624, 3.008131071630377, 2.0020281392528516, 1.4955121664955557, 0.9972942257819404, 0.7001063537185233, 0.5030475232448832, 0.3999269646135635, 0.19728418237600986]

nmerg = [250.,747.,629.,474.,259.,179.,132.,141.,111.]
nmerg_mass_cut = [144.,664.,566.,422.,239.,171.,129.,138.,106.]
nnonmerg = [1733.,5514.,7318.,7717.,7778.,7690.,7717.,7651.,7390.]

fmerg = []
fmerg_no_cut = []
for i in range(len(nmerg)):
    fmerg.append(nmerg_mass_cut[i]/nnonmerg[i])
    fmerg_no_cut.append(nmerg[i]/nnonmerg[i])

print(len(snapnum_list), len(nmerg), len(nmerg_mass_cut), len(nnonmerg))

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
ax2=ax.twinx()
ax.scatter(redshift_list, fmerg, color='red', label='mass cut')
ax.scatter(redshift_list, fmerg_no_cut, color='orange', label='no mass cut')
fig.legend()
ax.set_ylabel('fmerg', color='red')
ax2.scatter(redshift_list, nnonmerg, color='blue')
ax.set_xlabel('Redshift')
ax2.set_ylabel('# subhalos', color='blue')
plt.savefig('Figs/fmerg_z.png', dpi=1000)

print(fmerg)

