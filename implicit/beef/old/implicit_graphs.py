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
electrodb = connect('Au211_implicit_BEEF.db')
all_objects = {}
states = ['slab', 'CO_top', 'CO_bridge']
nelect0 = {'slab':396, 'CO_top':416, 'CO_bridge':416}
adsorbates = ['CO_top', 'CO_bridge']
refer_dict = {
              'CO_top':2 * COg['E'], \
              'CO_bridge':2 * COg['E'], \
              }

# Get everything into a dict
for i in range(len(states)):
    temp_list_storage = []
    for row in electrodb.select(states = states[i]):
        temp_list_storage.append(row)
    all_objects[states[i]] = temp_list_storage

##########################
# Plotting data

plt.figure()
for i in range(len(adsorbates)):

    is_db_data = all_objects['slab']
    fs_db_data = all_objects[adsorbates[i]]
    is_data, fs_data = get_order(is_db_data, fs_db_data)

    # Plotting against avg_wf 
    sa = get_surface_area(is_data, fs_data)
    vasp_sigma, rel_energy = energy_surface_charge(is_data, fs_data)
    sigma =  nelect0['slab'] / sa * (1e6 * 1.6e-19 / 1e-16 ) - vasp_sigma
    avgwf, rel_energy = energy_avg_wf(is_data, fs_data)
    binding_energy = ( rel_energy - refer_dict[adsorbates[i]] ) / 2
    # Getting fits
    p_sigma_deltaE, fit_sigma_deltaE = fit_to_curve(sigma, binding_energy, 2)
    p_avgwf_deltaE, fit_avgwf_deltaE = fit_to_curve(avgwf, binding_energy, 2)

    # Plotting the Energy vs sigma curve
    plt.plot(sigma, binding_energy, 'o', markersize=14, label=adsorbates[i])
    range_sigma = np.arange(min(sigma), max(sigma))
    plt.plot(range_sigma, p_sigma_deltaE(range_sigma), '--k')

plt.grid(False)
plt.ylabel(r'$\Delta E \ / eV$')
plt.xlabel(r'$\sigma \ / \mu C / cm^{2} $')
plt.legend(loc='best')
plt.savefig('sigma_deltaE.png')
plt.close()

    

