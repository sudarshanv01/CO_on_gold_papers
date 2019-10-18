#!/usr/bin/python

from ase.db import connect
from ase.io import read, write
import numpy as np
from ase.vibrations import Vibrations
import sys
sys.path.append('/Users/vijays/Documents/tools/scripts')
from useful_functions import get_vibrational_energy
import matplotlib
from ase.thermochemistry import HarmonicThermo
import matplotlib.pyplot as plt
import pickle
#plt.rcParams["figure.figsize"] = (14,8)

"""
THEORY PLOT
Script to plot the variation in binding energy with coverage for Au
Based on three functionals RPBE BEEF-vdw and RPBE-d3 
"""
thermodb = connect('Au_CO.db')
refer_rpbe = connect('references_RPBE_VASP_500.db')
refer_beef = connect('references_BEEF_VASP_500.db')
refer_rpbed3 = connect('references_RPBE_VASP_500.db')

referggadb = {'RPBE':refer_rpbe, 'BEEF-vdw':refer_beef, 'RPBE-d3':refer_rpbed3}

colors_functional = {'RPBE':'red', 
          'BEEF-vdw':'blue', 
          'RPBE-d3':'green'
          }
functionals = ['RPBE', 'RPBE-d3', 'BEEF-vdw']
shorthand_functional = {'RPBE':'RP', 'BEEF-vdw':'BF', 'RPBE-d3':'RP+D3'}
cell_sizes = ['1x3', '2x3', '3x3', '4x3']
coverages_cell = ['1', '0.5', '0.33', '0.25']
energies_gas = {}
# Getting the gas phase energies
for functional in functionals:
    refergga = referggadb[functional]
    COgstat = refergga.get(formula='CO', \
                            #functional = shorthand_functional[functional], \
                            pw_cutoff=500.0)

    COg_energies =  get_vibrational_energy(COgstat, method='novib', geometry='linear', \
            symmetrynumber=1)
    energies_gas[functional] = {
                                'COg':COg_energies['E'], 
                               }

def get_states(stat, functional, cell):
    # for a given database pick out all states
    allstates = []
    for row in stat.select(functional=functional, cell_size=cell):
        allstates.append(row.states)
    unique_states = np.unique(allstates)
    return unique_states

energies_thermo = {}
for functional in functionals:
    energies_thermo[functional] = {}
    for cell in cell_sizes:
        energies_thermo[functional][cell] = {}
        # get all states for a given functional
        states = get_states(thermodb, shorthand_functional[functional], cell)
        energies_thermo[functional]['states'] = states
        for state in states:
            print(functional, cell, state)
            stat = thermodb.get(states=state,\
                    functional=shorthand_functional[functional], \
                    cell_size=cell, \
                    )

            energies_thermo[functional][cell][state] = stat.energy

#plt.figure()
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel(r'$\theta_{DFT}$ \ ML', color='tab:blue')
ax1.set_ylabel(r'$\Delta E_{CO*}$', color=color)

for functional in functionals:
    states = energies_thermo[functional]['states']
    cell_plot = []#energies_thermo[functional]['states']
    for state in states:
        energies_plot = []
        if state != 'slab':
            for cell in cell_sizes:
                #cell_plot.append(cell)
                rel_energy = energies_thermo[functional][cell][state] - \
                             energies_thermo[functional][cell]['slab'] - \
                             energies_gas[functional]['COg']
                energies_plot.append(rel_energy)
            ax1.plot(coverages_cell, energies_plot, '--',
                    color=colors_functional[functional], 
                    alpha=0.1)

# now plot the minimum energy site for a given functional
for functional in functionals:
    states = energies_thermo[functional]['states']
    min_energy_cell = []
    for cell in cell_sizes:
        energy_cell = []
        for state in states:
            if state != 'slab':
                rel_energy = energies_thermo[functional][cell][state] - \
                             energies_thermo[functional][cell]['slab'] - \
                             energies_gas[functional]['COg']
                energy_cell.append(rel_energy)
        energy_cell = np.array(energy_cell)
        min_energy_cell.append(min(energy_cell))
    ax1.plot(coverages_cell, min_energy_cell, 'o-', 
             color=colors_functional[functional],
             mew=2, lw=3, fillstyle='none', 
             label=functional)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twiny()
color = 'tab:blue'
ax2.set_xlabel(r'$\theta_{experiment}$ \ ML', color=color) 
# get Experimental data
data_exp = pickle.load(open('desorption_energy.pickle', 'rb'))
Ed = data_exp['Ed']
theta = data_exp['theta']
ax2.plot(theta, -1 * Ed, color='k', lw=4, label='Experiment')
ax2.set_xlim([0.175, 0])
ax2.tick_params(axis='y', labelcolor=color)
fig.legend(loc='best')

# plot convergence points
ax1.plot('0.33', -0.435, 'd', fillstyle='none', mew=2, color=colors_functional['RPBE'])
ax1.annotate('5 layers', xy=(1.8, -0.4), color='k', \
        weight='bold').draggable()
#ax1.plot('0.33', -0.36, 'd', fillstyle='none', mew=1, color=colors_functional['RPBE'])
#ax1.annotate('3 layers', xy=(1.8, -0.2), color=colors_functional['RPBE'], \
#        weight='bold').draggable()

ax1.plot('0.33', -0.415, 'd', fillstyle='none', mew=2, color=colors_functional['BEEF-vdw'])

#plt.legend(bbox_to_anchor=(1,0), loc="lower right", 
#        bbox_transform=fig.transFigure, ncol=3)
fig.tight_layout() 


#plt.xlabel('Coverage')
#plt.ylabel(r'$\Delta E$ \ eV')
plt.savefig('functional_comparison.png')
plt.show()



