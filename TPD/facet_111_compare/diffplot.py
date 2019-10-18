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
thermodb = connect('Au111_CO.db')
refer_rpbe = connect('references_RPBE_VASP_500.db')
refer_beef = connect('references_BEEF_VASP_500.db')
refer_rpbed3 = connect('references_RPBE_VASP_500.db')

referggadb = {'RPBE':refer_rpbe, 'BEEF-vdw':refer_beef, 'RPBE-d3':refer_rpbed3}

colors_functional = {'RPBE':'red', 
          'BEEF-vdw':'blue', 
          'RPBE-d3':'green'
          }
functionals = ['RPBE', 'BEEF-vdw', 'RPBE-d3']
shorthand_functional = {'RPBE':'RP', 'BEEF-vdw':'BF', 'RPBE-d3':'RP+D3'}
cell_sizes = ['1x1', '2x2', '3x3']
min_cov_cell = '3x3'
#cell_sizes = ['1CO_4x3', '2CO_4x3', '3CO_4x3', '4CO_4x3']
#min_cov_cell = '1CO_4x3'
coverages_cell = {'1x1': '1',  '2x2':0.25,  '3x3':0.11}
mult_factor = {'1x1':6, '2x2':3, '3x3':2}

def get_number_CO(atoms):
    nC = [ atom.index for atom in atoms if atom.symbol == 'C']
    return len(nC)

def diff_energies(lst_stat, mult_factor, COg, lowE_CO_abs):
    # take a list of database rows and convert it to differential energies
    # and coverages
    cell_lst = [ stat.cell_size for stat in lst_stat ]
    # get mulp factor based on the cell size
    mult_lst = []
    for cell in cell_lst:
        mult_lst.append(mult_factor[cell])
    try:
        sorted_mult, sorted_lst = (list(t) for t in zip(*sorted(zip(mult_lst, lst_stat))))
    except TypeError:
        sorted_lst = lst_stat
        sorted_mult = [1, 1, 1, 1]
    natoms_lst = []
    abs_ener_lst = []
    cell_sizes_sorted = []
    for i in range(len(sorted_lst)):
        atoms = sorted_lst[i].toatoms()
        cell_sizes_sorted.append(sorted_lst[i].cell_size)
        energies = sorted_lst[i].energy
        natoms = atoms.repeat([sorted_mult[i], sorted_mult[i], 1])
        natoms_lst.append(natoms)
        abs_ener_lst.append(energies)

    # get the number of CO in 
    abs_ener_lst = np.array(abs_ener_lst)
    nCO_bigcell = []
    for atom in natoms_lst:
        nCO = get_number_CO(atom)
        nCO_bigcell.append(nCO)
        
    plot_nCO = nCO_bigcell
    plot_diff_energies = []
    plot_diff_energies.append(lowE_CO_abs)
    nCO_bigcell = np.array(nCO_bigcell)
    print(nCO_bigcell)
    print(abs_ener_lst)
    # Going from least coverage to most coverage
    for i in range(len(nCO_bigcell)-1):
        nCOdiff = np.array(nCO_bigcell[i+1]) - np.array(nCO_bigcell[i])
        diff_energies = ( sorted_mult[i+1]**2 * abs_ener_lst[i+1] - \
                sorted_mult[i]**2 * abs_ener_lst[i] - \
                nCOdiff * COg ) / nCOdiff 
        print(sorted_mult[i+1] * abs_ener_lst[i+1] - \
                                  sorted_mult[i] * abs_ener_lst[i])
        print(diff_energies)
        plot_diff_energies.append(diff_energies)
    print(cell_sizes_sorted)
    return [cell_sizes_sorted, plot_diff_energies]

def get_lowest_absolute_energies(lst_stat_CO, lst_stat_slab, largest_cell, COg):
    for stat in lst_stat_CO:
        cell_size = stat.cell_size
        if cell_size == largest_cell:
            energy_CO = stat.energy
    for stat in lst_stat_slab:
        cell_size = stat.cell_size
        if cell_size == largest_cell:
            energy_slab = stat.energy
    return energy_CO - energy_slab - COg


energies_gas = {}
# Getting the gas phase energies
for functional in functionals:
    refergga = referggadb[functional]
    COgstat = refergga.get(formula='CO', \
                            #functional = shorthand_functional[functional], \
                            pw_cutoff=500.0)

    COg_energies =  get_vibrational_energy(COgstat, [], method='novib', geometry='linear', \
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

def monotonic(x):
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)

energies_thermo = {}
dict_stat = {}
for functional in functionals:
    dict_stat[functional] = {}
    states = get_states(thermodb, shorthand_functional[functional], '1x1')
    for state in states:
        dict_stat[functional][state] = {}
        for cell in cell_sizes:
            print(functional, state, cell)
            dict_stat[functional][state][cell] = {}
            # get all states for a given functional
            try:
                stat = thermodb.get(states=state,\
                        opt=True,
                        functional=shorthand_functional[functional], \
                        cell_size=cell, \
                        )
            except:
                stat = thermodb.get(states=cell,\
                        functional=shorthand_functional[functional], \
                        cell_size=cell, \
                        )


            dict_stat[functional][state][cell] = stat

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel(r'$\theta_{DFT}$ \ ML', color='tab:blue')
ax1.set_ylabel(r'$\Delta E_{CO*}$', color=color)

for functional in functionals:
    print(functional)
    states = get_states(thermodb, shorthand_functional[functional], '1x1')
    # get the differential energy 
    for state in states:
        lst_stat_slab = []
        for cell in cell_sizes:
            if state == 'slab':
                lst_stat_slab.append(dict_stat[functional][state][cell])
    all_diff_energies = []
    for state in states:
        if state != 'slab':
            lst_stat_CO = []
            for cell in cell_sizes:
                lst_stat_CO.append(dict_stat[functional][state][cell])
        lowest_CO = get_lowest_absolute_energies(lst_stat_CO, lst_stat_slab, min_cov_cell, \
                energies_gas[functional]['COg'])
        cell_sizes, diff_energy  = diff_energies(lst_stat_CO, mult_factor, energies_gas[functional]['COg'], \
                lowest_CO)
        all_diff_energies.append(np.array(diff_energy))
        coverages = []
        for cell in cell_sizes:
            coverages.append(coverages_cell[cell])
        #plt.plot(coverages, diff_energy, color=colors_functional[functional], alpha=0.1)
        print(diff_energy)
        if monotonic(diff_energy):
            ax1.plot(coverages, diff_energy, '-',
                    color=colors_functional[functional], 
                    alpha=0.1)
    # plotting minimum 
    all_diff_energies = np.array(all_diff_energies)
    min_diff = [min(column) for column in zip(*all_diff_energies)]
    #ax1.plot(coverages,  min_diff, '-',  color=colors_functional[functional],\
    #        label=functional)
    ax1.plot([], [], '-',  color=colors_functional[functional],\
           label=functional)



ax1.tick_params(axis='y', labelcolor=color)
#ax2 = ax1.twiny()
#color = 'tab:blue'
#ax2.set_xlabel(r'$\theta_{experiment}$ \ ML', color=color) 
## get Experimental data
#data_exp = pickle.load(open('desorption_energy.pickle', 'rb'))
#Ed = data_exp['Ed']
#theta = data_exp['theta']
#ax2.plot(np.flip(theta), -1 * np.flip(Ed), color='k', lw=4, label='Experiment')
#ax2.set_xlim([ 0, 0.175])
#ax2.tick_params(axis='y', labelcolor=color)
fig.legend(loc='best')

plt.ylim([-1, 1])
#plt.legend(bbox_to_anchor=(1,0), loc="lower right", 
#        bbox_transform=fig.transFigure, ncol=3)
fig.tight_layout() 


#plt.xlabel('Coverage')
#plt.ylabel(r'$\Delta E$ \ eV')
plt.savefig('diff_functional_comparison.png')
plt.show()

