#!/usr/bin/python

import pickle
import matplotlib.pyplot as plt
from ase.thermochemistry import HarmonicThermo
import matplotlib
from useful_functions import get_vibrational_energy
from ase.db import connect
from ase.io import read, write
import numpy as np
from ase.vibrations import Vibrations
import sys
sys.path.append('/Users/vijays/Documents/tools/scripts')
# plt.rcParams["figure.figsize"] = (14,8)

"""
THEORY PLOT
Plot the differential energies associated with Pb deposition on different
facets of gold
"""


def get_states(stat, functional, cell):
    # for a given database pick out all states
    allstates = []
    for row in stat.select(functional=functional, cell_size=cell):
        allstates.append(row.states)
    unique_states = np.unique(allstates)
    return unique_states


# Get the reference energies of Pb
refergga = connect('reference_lead.db')
# Choice of functional
functional = 'RPBE'
# Get the Pb for a given reference
Pbstat = refergga.get(formula='Pb4',
                      functional='RP',
                      pw_cutoff=500.0)
# Four atoms in Pb unit cell
Pb_energies = Pbstat.energy / 4
energies_ref = Pb_energies
colors_facet = {'211': 'tab:blue', '100': 'tab:red', '111': 'tab:green'}
cell_sizes = {'100': ['1x1', '2x2', '3x3'],
              '211': ['1x3', '2x3', '3x3'],
              '111': ['1x1', '2x2', '3x3']}

cell_mult = {'100': {'1x1':1, '2x2':2, '3x3':3},
              '211': {'1x3':1/3, '2x3':2/3, '3x3':3/3},
              '111': {'1x1':1, '2x2':2*4, '3x3':3*4}}
coverage_labels = {'100': {'1x1':'1', '2x2':'1/2', '3x3':'1/3'},
              '211': {'1x3':'1', '2x3':'1/2', '3x3':'1/3'},
              '111': {'1x1':'1', '2x2':'2/3', '3x3':'1/3'}}

coverages_cell = {'100': [1, 0.25, 0.11],
                  '211': [1, 0.66, 0.33],
                  '111': [1, 0.25, 0.11]}

facets = ['100', '211', '111']
energies_thermo = {}
for facet in facets:
    thermodb = connect('Au' + facet + '_lead.db')
    energies_thermo[facet] = {}
    for cell in cell_sizes[facet]:
        energies_thermo[facet][cell] = {}
        # get all states for a given functional
        states = get_states(thermodb, 'RP', cell)
        energies_thermo[facet]['states'] = states
        Pb_states = [state for state in states if 'Pb' in state or 'slab' in state]
        for state in Pb_states:
            print(facet, cell, state)
            try:
                stat = thermodb.get(states=state,
                                    functional='RP',
                                    cell_size=cell,
                                    )
            except AssertionError:
                stat = thermodb.get(states=state,
                                    functional='RP',
                                    cell_size=cell,
                                    opt=True,
                                    )
            energies_thermo[facet][cell][state] = stat.energy

def plot_differential_energies(integral):
    plot_diff_energies = []
    plot_diff_energies.append(integral[0])
    # Going from least coverage to most coverage
    addition_energies = integral[1:]
    for i in range(len(addition_energies)):
        nPbdiff = np.array(addition_energies[i]) \
                - np.array(addition_energies[i-1])
        plot_diff_energies.append(nPbdiff)
    return plot_diff_energies

# plt.figure()
plt.figure()
for facet in facets:
    states = energies_thermo[facet]['states']
    cell_plot = []  # energies_thermo[functional]['states']
    for state in states:
        energies_plot = []
        if state != 'slab':
            for cell in cell_sizes[facet]:
                # cell_plot.append(cell)
                rel_energy = energies_thermo[facet][cell][state] - \
                    energies_thermo[facet][cell]['slab'] - \
                    energies_ref
                energies_plot.append(rel_energy)
            diff_energies = plot_differential_energies(energies_plot)
            plt.plot(coverages_cell[facet], diff_energies, '--',
                     color=colors_facet[facet],
                     )
# plt.show()
plt.close()
plt.figure()
# now plot the minimum energy site for a given functional
facet_markers = {'100':'v', '111':'o', '211':'^'}
for facet in facets:
    states = energies_thermo[facet]['states']
    min_energy_cell = []
    for cell in cell_sizes[facet]:
        energy_cell = []
        for state in states:
            if state != 'slab':
                rel_energy = energies_thermo[facet][cell][state] - \
                    energies_thermo[facet][cell]['slab'] - \
                    energies_ref
                energy_cell.append(rel_energy)
        energy_cell = np.array(energy_cell)
        min_energy_cell.append(min(energy_cell))
    plt.plot(coverages_cell[facet], min_energy_cell, '-',
             color=colors_facet[facet],
             mew=2, lw=3, fillstyle='none',
             label='Au(' + facet+ ')', marker=facet_markers[facet])
#plt.tick_params(axis='y', labelcolor=color)
plt.legend(loc='best')

plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
plt.xlabel(r'$\theta$ \ ML', fontsize=24)
plt.ylabel(r'$\Delta E_{Pb}$ \ eV', fontsize=24)
plt.savefig('lead_UPD.svg')
#plt.show()
plt.close()

plt.figure()
# now plot the minimum energy site for a given functional
facet_markers = {'100':'v', '111':'o', '211':'^'}
potential_range = np.linspace(1.2, -1.2)



for facet in facets:
    states = energies_thermo[facet]['states']
    min_energy_cell = []
    plt.figure()
    for cell in cell_sizes[facet]:
        energy_cell = []
        for state in states:
            if state != 'slab':
                rel_energy = energies_thermo[facet][cell][state] - \
                    energies_thermo[facet][cell]['slab'] - \
                                energies_ref
                energy_cell.append(rel_energy)
        energy_cell = np.array(energy_cell)
        min_energy_cell.append(min(energy_cell))
        nPb = cell_mult[facet][cell]
        p = np.poly1d([2/nPb, min(energy_cell)+0.13])
        plt.plot(potential_range, p(potential_range),
                color=colors_facet[facet])
        plt.annotate(r'$\theta = $ ' + coverage_labels[facet][cell] + ' ML',
                    xy=(0, p(0) + 0.25),
                    ).draggable()
        plt.text(-0.3, 0.75, r'Au(' +facet + ')', weight='bold',
            color=colors_facet[facet],
            fontsize=20
            )

    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.axhline(y=0, color='k', ls='--', lw=3)
    plt.xlabel(r'Potential \ V vs SHE', fontsize=24)
    plt.ylabel(r'$\Delta E_{Pb}$ \ eV', fontsize=24)
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])
    # plt.legend(loc='best')
    plt.savefig('lead_UPD_' + facet +'.pdf')
    plt.show()


#for facet in facets:
#    plt.plot([], [], color=colors_facet[facet], label='Au(' + facet+ ')')
