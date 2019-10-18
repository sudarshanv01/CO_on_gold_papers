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
        fit_to_curve,  get_vibrational_energy, get_surface_area

# Referernces etc
referdb = connect("references_RPBE_600.db")
COgstat = referdb.get(formula='CO', functional='RPBE', \
                        pw_cutoff=600.0)
COg =  get_vibrational_energy(COgstat, method='ideal_gas', geometry='linear', \
        symmetrynumber=1)
all_objects = {}
states = ['slab', 'CO_top', 'CO_bridge']

refer_dict = {
              'CO_top':COg['E'], \
              'CO_bridge':COg['E'], \
              }

charging_method = ['pcc', 'linmodpb', 'pcc_vacuum']
facets = ['111','211']
cell_symm = ['symmetric', 'nonsymm']
colors = {'linmodpb':'crimson', 'pcc':'midnightblue', 'pcc_vacuum':'peru'}
states = ['slab', 'CO_top', 'CO_bridge']
adsorbates = ['CO_bridge']
ltype = ['--', ':']
lw = {'111':1, '211':3}
markers={'symmetric':'o', 'nonsymm':'*'}
plt.figure()

for charging in charging_method:
    for symm in cell_symm:
        # Get all elements into a list
        for facet in facets:
            electrodb = connect('Au' + facet + '_' + charging + '_' + symm +  '.db')
            for i in range(len(states)):
                temp_list_storage = []
                for row in electrodb.select(states = states[i]):
                    temp_list_storage.append(row)
                all_objects[states[i]] = temp_list_storage
            for i in range(len(adsorbates)):
                try:
                    is_db_data = all_objects['slab']
                    fs_db_data = all_objects[adsorbates[i]]
                    is_data, fs_data = get_order(is_db_data, fs_db_data)
                    sa = get_surface_area(is_data, fs_data)
                    sigma, rel_energy = energy_surface_charge(is_data, fs_data)
                    if symm == 'nonsymm':
                        binding_energy = ( rel_energy - refer_dict[adsorbates[i]] ) 
                    elif symm == 'symmetric':
                         binding_energy = ( rel_energy - 2 * refer_dict[adsorbates[i]] ) / 2
                    print(binding_energy)
                    # Plotting the Energy vs sigma curve
                    if charging == 'pcc' and cell_symm == 'nonsymm': 
                        plt.plot(sigma , binding_energy, color=colors[charging], linestyle= ltype[i], marker=markers[symm], lw=lw[facet])
                    else:
                        plt.plot(sigma / 2, binding_energy, color=colors[charging], linestyle= ltype[i], marker=markers[symm], lw=lw[facet])
                    annotate = adsorbates[i].replace('_', ' ')
                    #plt.annotate(annotate, xy=(sigma[1], binding_energy[1] ), \
                    #        color=colors[charging]).draggable()
                    plt.title(annotate)
                    #range_sigma = np.arange(min(sigma), max(sigma))
                    #plt.plot(range_sigma, p_sigma_deltaE(range_sigma), '--k')
                except IndexError:
                    #print('Calculation not available for ' + metal + ' ' + adsorbates[i])
                    print('Calculation not available for ' + facet + charging + symm)
    plt.plot([],[], color=colors[charging], label=charging)
for symm in cell_symm:
    plt.plot([],[], color='k', marker=markers[symm], label=symm)


plt.grid(False)
plt.ylabel(r'$\Delta E \ / eV$')
plt.xlabel(r'$\sigma \ / \mu C / cm^{2} $')
plt.legend(loc='center left')
plt.savefig('sigma_deltaE.png')
plt.show()
plt.close()
