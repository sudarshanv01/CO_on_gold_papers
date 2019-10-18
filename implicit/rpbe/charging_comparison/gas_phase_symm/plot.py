#!/usr/bin/python

from ase.db import connect
from ase.io import read, write
import numpy as np
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/vijays/Documents/tools/scripts')
from useful_functions import get_vibrational_energy
plt.rcParams["figure.figsize"] = (12,16.1)
# Plots the CHE free energy diagram 

energydb = connect("Au_gas_phase.db")
referdb = connect("references_RPBE_600.db")

# Getting the gas phase energies

COgstat = referdb.get(formula='CO', functional='RPBE', \
                        pw_cutoff=600.0)

COg =  get_vibrational_energy(COgstat, method='ideal_gas', geometry='linear', \
        symmetrynumber=1)

# Getting energies of species
states = ['slab', 'CO_top', 'CO_bridge']
facets = [111, 211]
symmetric = [True, False]

colors = {111:'r', 211:'b'}

for facet in facets:
    rel_energy = {}
    for symm in symmetric:
        states_energy = {}
        for i in range(len(states)):
            stat = energydb.get(states=states[i], facet=facet, symmetric=symm)
            energy = stat.energy
            states_energy[states[i]] = float(energy)
        if symm == True:
            CO_top = ( states_energy['CO_top'] - states_energy['slab'] - 2 * COg['E'] ) / 2
            CO_bridge = ( states_energy['CO_bridge'] - states_energy['slab'] - 2 * COg['E'] ) / 2
        if symm == False:
            CO_top = ( states_energy['CO_top'] - states_energy['slab'] -  COg['E'] ) 
            CO_bridge = ( states_energy['CO_bridge'] - states_energy['slab'] -  COg['E'] ) 
        rel_energy[symm] = [CO_top, CO_bridge]
    plt.plot(rel_energy[True][0], rel_energy[False][0], colors[facet] + 'o')
    plt.annotate('$CO_{top}^{' + str(facet) + '}$', \
            xy = ( rel_energy[True][0], rel_energy[False][0]), color=colors[facet]).draggable()
    plt.plot(rel_energy[True][1], rel_energy[False][1], colors[facet]+'o')
    plt.annotate('$CO_{bridge}^{' + str(facet) + '}$', \
            xy = ( rel_energy[True][1], rel_energy[False][1]), color=colors[facet] ).draggable()

plt.ylabel(r'$\Delta E \ asymmetric \ eV$')
plt.xlabel(r'$\Delta E \ symmetric \ eV$')
plt.grid(False)
x = np.linspace(-0.5, 0.1)
plt.plot(x, x, 'k--')
plt.xlim([-0.5, 0.1])
plt.ylim([-0.5, 0.1])
plt.savefig('Au_gas_phase.png')
