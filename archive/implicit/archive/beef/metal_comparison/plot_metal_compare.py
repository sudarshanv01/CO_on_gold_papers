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
referdb = connect("references_BEEF_vdw_vasp.db")
COgstat = referdb.get(formula='CO', functional='BF', \
                        pw_cutoff=400)
COg =  get_vibrational_energy(COgstat, method='ideal_gas', geometry='linear', \
        symmetrynumber=1)
all_objects = {}
states = ['slab', 'CO_top', 'CO_bridge']
nelect0 = {'slab':396, 'CO_top':416, 'CO_bridge':416}
adsorbates = ['CO_top', 'CO_bridge']
refer_dict = {
              'CO_top':2 * COg['E'], \
              'CO_bridge':2 * COg['E'], \
              }

metals = ['Au', 'Ag', 'Cu' , 'Pt', 'Pd']
colors = {'Au':'gold', 'Ag':'gray', 'Cu':'crimson', 'Pt':'skyblue', 'Pd':'darkcyan'}
metal_nelect0 = {'Au': 396, 'Ag': 396, 'Cu':396, 'Pt':360, 'Pd':360}
states = ['slab', 'CO_top', 'CO_bridge']
adsorbates = ['CO_top', 'CO_bridge']
ltype = ['--', '-.']
plt.figure()

for metal in metals:
    electrodb = connect(metal + '211_implicit_BEEF.db')
    # Get all elements into a list
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
            vasp_sigma, rel_energy = energy_surface_charge(is_data, fs_data)
            sigma =  metal_nelect0[metal] / sa * (1e6 * 1.6e-19 / 1e-16 ) - vasp_sigma

            binding_energy = ( rel_energy - refer_dict[adsorbates[i]] ) / 2
            print(metal)
            print(np.polyfit(sigma, binding_energy,1))
            # Plotting the Energy vs sigma curve
            plt.plot(sigma, binding_energy, color=colors[metal], linestyle= ltype[i])
            annotate = adsorbates[i].replace('_', ' ')
            plt.annotate(annotate, xy=(sigma[0], binding_energy[0]), \
                    color=colors[metal]).draggable()
            #range_sigma = np.arange(min(sigma), max(sigma))
            #plt.plot(range_sigma, p_sigma_deltaE(range_sigma), '--k')
        except IndexError:
            print('Calculation not available for ' + metal + ' ' + adsorbates[i])
    plt.plot([],[], color=colors[metal], label=metal)


plt.grid(False)
plt.ylabel(r'$\Delta E \ / eV$')
plt.xlabel(r'$\sigma \ / \mu C / cm^{2} $')
plt.legend(loc='center left')
plt.savefig('sigma_deltaE.png')
plt.show()
plt.close()
