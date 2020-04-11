#!/usr/bin/python

""" Get the AIMD energies for CO adsorption energies """

from useful_functions import get_vibrational_energy, AutoVivification
from ase.thermochemistry import IdealGasThermo
from ase.thermochemistry import HarmonicThermo
from ase.io import read
import numpy as np
import pickle
import os
from pprint import pprint
import matplotlib.pyplot as plt

### For the 211 surface facet
files_slab = {
              'Au_211':['density_slab.pickle'],
              'Au_100':['density_slab.pickle']
              }

int_bounds = {'Au_211':[19.25, 23.5],
              'Au_100':[11, 14]
              }
arrows = {'Au_100':[13, 0.08],
          'Au_211':[21.5, 0.08],
          }

text = {'Au_100':[9.5, 0.08],
          'Au_211':[18, 0.08],
          }

facets = ['Au_211', 'Au_100']

surface_dist = {'Au_100':9.3, 'Au_211':18.7}


fig, ax = plt.subplots(len(facets), figsize=(9,8))
output = 'output/'
os.system('mkdir -p ' + output)

for index, facet in enumerate(facets):
    for f in files_slab[facet]:
        data = pickle.load(open(facet + '/' + f, 'rb'))
        bins = data['binc']
        Hw = data['hist_dicts']['Hw']
        Ow = data['hist_dicts']['Ow']
        if index == 1:
            ax[index].fill_between(bins, Hw, color='tab:blue', alpha=0.5, label='H')
            ax[index].fill_between(bins, Ow, color='tab:red', alpha=0.5, label='O')
            ax[index].legend(loc='best')
        else:
            ax[index].fill_between(bins, Hw, color='tab:blue', alpha=0.5)
            ax[index].fill_between(bins, Ow, color='tab:red', alpha=0.5)
        ax[index].axvline(x=int_bounds[facet][0], color='k', lw=3, ls='--', alpha=0.5)
        ax[index].axvline(x=int_bounds[facet][1], color='k', lw=3, ls='--', alpha=0.5)
        args = [i for i in range(len(bins)) if bins[i]>int_bounds[facet][0] and bins[i]< int_bounds[facet][1]]
        integrated_O = np.trapz(Ow[args], bins[args])
        integrated_H = np.trapz(Hw[args], bins[args])
        ax[index].set_ylabel(r'Density \ $mol cm^{-3}$')
        ax[index].set_xlabel(r'z \ $\AA$')
        ax[index].annotate(round(integrated_O,2), xytext=text[facet], \
                            weight='bold', xy=arrows[facet], color='tab:red',\
                            arrowprops=dict(facecolor='tab:red', shrink=0.05),\
                            horizontalalignment='right', verticalalignment='center',)
        ax[index].annotate('Au('+facet.replace('Au_','')+')', weight='bold', xy=(2.5,0.1))
        ax[index].axvline(x=surface_dist[facet], color='yellow', lw=3, ls='--' )
#xycoords='data',xytext=(200,55), \
            # arrowprops=dict(facecolor='black', shrink=0.05, color=colors_facet['100']),
fig.tight_layout()
plt.savefig(output + '/aimd_density.pdf')
plt.show
