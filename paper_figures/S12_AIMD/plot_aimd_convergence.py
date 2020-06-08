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
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
### For the 211 surface facet
files_slab = {
             'BEEF':{
              'Au_211':['potential_energies_slab.pickle', 'potential_energies_slab_2.pickle', 'potential_energies_slab_3.pickle' ],
              'Au_100':['potential_energies_slab.pickle', 'potential_energies_slab_2.pickle']
                    },
              'RPBE':{
              'Au_211':['potential_energies_slab.pickle',],
              }
              }
files_ads = {
             'BEEF':{
             'Au_211':['potential_energies_COa.pickle', 'potential_energies_COa_2.pickle', 'potential_energies_COa_3.pickle'],
             'Au_100':['potential_energies_COa.pickle', 'potential_energies_COa_2.pickle']
                    },
              'RPBE':{
              'Au_211':['potential_energies_COa.pickle',],
              }
             }

facets = {'BEEF':['Au_211', 'Au_100'], 'RPBE':['Au_211']}

functionals = ['BEEF', 'RPBE']

AIMD = {'Au_100': {'COa': {'E': -275.0892, 'G': 0.4635955921831484},
                   'slab': {'E': -262.935}},
        'Au_211': {'COa': {'E': -209.117, 'G': 0.731126904576656},
                   'slab': {'E': -197.311}}}

for functional in functionals:
    fig, ax = plt.subplots(len(facets),2)
    output = 'output/'
    os.system('mkdir -p ' + output)
    for index, facet in enumerate(facets[functional]):
        for f in files_slab[functional][facet]:
            potential_energy = pickle.load(open(facet + '_'+ functional+ '/' + f, 'rb'))
            time = np.arange(len(potential_energy))
            energies = potential_energy - AIMD[facet]['slab']['E']
            ax[index,0].plot(time/1000, energies, lw=4)
            ax[index,0].set_ylabel(r'$E - <E>$ / eV')
            ax[index,0].set_xlabel(r'time / ps')
            ax[index,0].set_title('('+facet.replace('Au_', '')+') slab')

        for f in files_ads[functional][facet]:
            potential_energy = pickle.load(open(facet + '_'+ functional+ '/' + f, 'rb'))
            energies = potential_energy - AIMD[facet]['COa']['E']
            time = np.arange(len(potential_energy))
            ax[index,1].plot(time/1000, energies, lw=4)
            ax[index,1].set_ylabel(r'$E - <E>$ / eV')
            ax[index,1].set_xlabel(r'time / ps')
            ax[index,1].set_title('('+facet.replace('Au_', '')+r') CO$^{*}$')

    fig.tight_layout()
    plt.savefig(output + '/'+functional+'_aimd_converge.pdf')
# plt.show
