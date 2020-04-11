#!/usr/bin/python

from ase.io import read
from ase.db import connect 
import numpy as np
from ase import units
import sys, os
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,16.1)
sys.path.append('/Users/vijays/Documents/tools/scripts')
from useful_functions import energy_avg_wf, energy_surface_charge, get_order, \
        fit_to_curve,  get_vibrational_energy, get_surface_area, get_vasp_nelect0

# Referernces etc
electrodb = connect('metals_111_charging.db')
referdb = connect("references_RPBE_VASP_500.db")
symmetric = [True, False]
symmetric_label = {True:'symmetric', False:'unsymmetric'}
COgstat = referdb.get(formula='CO')
COg =  get_vibrational_energy(COgstat, method='novib', geometry='linear', \
        symmetrynumber=1)
all_objects = {}
states = ['slab', 'CO_ontop', 'CO_bridge', 'CO_hollow']
adsorbates = ['CO_ontop', 'CO_bridge', 'CO_hollow']
refer_dict = {
              'CO_ontop':COg['E'], \
              'CO_bridge':COg['E'], \
              'CO_hollow':COg['E'], \
              }

metals = ['Pd', 'Pt', 'Ir' , 'Ni']
colors = {'Au':'gold', 'Ag':'gray', 'Cu':'crimson', 'Ir': 'midnightblue', \
        'Pt':'skyblue', 'Pd':'darkcyan', 'Ni':'lime'}

ltype = ['--', '-.', ':']
markers = {True:'o', False:'v'}
plt.figure()

for metal in metals:
    for symm in symmetric:
        # Get all elements into a list
        for i in range(len(states)):
            temp_list_storage = []
            for row in electrodb.select(states = states[i], facet=metal, symmetric=symm):
                temp_list_storage.append(row)
            all_objects[states[i]] = temp_list_storage

        for i in range(len(adsorbates)):

            is_db_data = all_objects['slab']
            fs_db_data = all_objects[adsorbates[i]]
            is_data, fs_data = get_order(is_db_data, fs_db_data)
            sa = get_surface_area(is_data, fs_data)
            vasp_sigma, rel_energy = energy_surface_charge(is_data, fs_data)
            metal_nelect0 = get_vasp_nelect0(is_db_data[0].toatoms())
            sigma =  metal_nelect0 / sa * (1e6 * 1.6e-19 / 1e-16 ) - vasp_sigma
            if symm:
                binding_energy = ( rel_energy - 2 * refer_dict[adsorbates[i]] ) / 2
                plt.plot(sigma / 2 , binding_energy, color=colors[metal], \
                        linestyle= ltype[i], marker=markers[symm], markersize=14)
            else:
                binding_energy = ( rel_energy -  refer_dict[adsorbates[i]] ) 
                plt.plot(sigma , binding_energy, color=colors[metal], \
                        linestyle= ltype[i], marker=markers[symm], markersize=14)

            # Plotting the Energy vs sigma curve
            annotate = adsorbates[i].replace('CO_', ' ')
            #plt.annotate(annotate, xy=(sigma[0], binding_energy[0]), \
            #        color=colors[metal]).draggable()
            #range_sigma = np.arange(min(sigma), max(sigma))
            #plt.plot(range_sigma, p_sigma_deltaE(range_sigma), '--k')
    plt.plot([],[], color=colors[metal], label=metal)
    


for symm in symmetric:
    plt.plot([], [], marker=markers[symm], label=symmetric_label[symm],color='dimgray')

for i in range(len(adsorbates)):
    plt.plot([], [], ls=ltype[i], color='dimgray', label=adsorbates[i].replace('_', ' '))

plt.grid(False)
plt.ylabel(r'$\Delta E \ / eV$')
plt.xlabel(r'$\sigma \ / \mu C / cm^{2} $')
plt.legend(loc='center left')
plt.savefig('sigma_deltaE.png')
plt.show()
plt.close()
