#!/usr/bin/python

from ase.io import read
from ase.db import connect
import numpy as np
import sys, os
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,16)
sys.path.append('/Users/vijays/Documents/tools/scripts')
from useful_functions import energy_surface_area, get_vibrational_energy, \
        get_order

referdb = connect("references_BEEF_vdw_vasp.db")

COgstat = referdb.get(formula='CO', functional='BF', \
                        pw_cutoff=400)

COg =  get_vibrational_energy(COgstat, method='ideal_gas', geometry='linear', \
        symmetrynumber=1)


electrodb = connect('Au211_cells.db')
all_objects = {}
states = ['slab', 'CO_top', 'CO_bridge']
adsorbates = ['CO_top', 'CO_bridge']
for i in range(len(states)):
    temp_list_storage = []
    for row in electrodb.select(states = states[i]):
        temp_list_storage.append(row)
    all_objects[states[i]] = temp_list_storage

plt.figure()
for i in range(len(adsorbates)):

    is_db_data = all_objects['slab']
    fs_db_data = all_objects[adsorbates[i]]

    is_data, fs_data = get_order(is_db_data, fs_db_data)

    sa, rel_energy = energy_surface_area(is_data, fs_data)
    plt.plot(sa, rel_energy - COg['E'], 'o', markersize=14,  label=adsorbates[i])
    plt.plot(sa, rel_energy - COg['E'], 'k--')


plt.ylabel(r'$\Delta E / eV$')
plt.xlabel(r'$ Surface\ area\ / A^{2}$')
plt.legend(loc='best')
plt.grid(False)
plt.savefig('explicit_cell.png')

