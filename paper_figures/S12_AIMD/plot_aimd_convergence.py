#!/usr/bin/python

""" Get the AIMD energies for CO adsorption energies """

from useful_functions import get_vibrational_energy, AutoVivification
from ase.thermochemistry import IdealGasThermo
from ase.thermochemistry import HarmonicThermo
from ase.io import read
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

### For the 211 surface facet
files_slab = {
              'Au_211':['potential_energies_slab.pickle', 'potential_energies_slab_reference_2.pickle', ],
              'Au_100':['potential_energies_slab.pickle', 'potential_energies_slab_repeat.pickle']
              }
files_ads = {
             'Au_211':['potential_energies_COa.pickle', 'potential_energies_COa_repeat.pickle'],
             'Au_100':['potential_energies_COa.pickle', 'potential_energies_COa_repeat.pickle']
             }

facets = ['Au_211', 'Au_100']

AIMD = {'Au_100': {'COa': {'E': -275.098, 'G': 0.4635955921831484},
                   'slab': {'E': -262.855}},
        'Au_211': {'COa': {'E': -209.1707, 'G': 0.731126904576656},
                   'slab': {'E': -197.203}}}
fig, ax = plt.subplots(len(facets),2)
output = 'output/'
os.system('mkdir -p ' + output)
for index, facet in enumerate(facets):
    for f in files_slab[facet]:
        potential_energy = pickle.load(open(facet + '/' + f, 'rb'))
        time = np.arange(len(potential_energy))
        energies = potential_energy - AIMD[facet]['slab']['E']
        ax[index,0].plot(time/1000, energies, lw=4)
        ax[index,0].set_ylabel(r'E - <E> \ eV')
        ax[index,0].set_xlabel(r'time \ ps')
        ax[index,0].set_title('('+facet.replace('Au_', '')+') slab')

    for f in files_ads[facet]:
        potential_energy = pickle.load(open(facet + '/' + f, 'rb'))
        energies = potential_energy - AIMD[facet]['COa']['E']
        time = np.arange(len(potential_energy))
        ax[index,1].plot(time/1000, energies, lw=4)
        ax[index,1].set_ylabel(r'E - <E> \ eV')
        ax[index,1].set_xlabel(r'time \ ps')
        ax[index,1].set_title('('+facet.replace('Au_', '')+r') CO$^{*}$')

fig.tight_layout()
plt.savefig(output + '/aimd_converge.pdf')
# plt.show
