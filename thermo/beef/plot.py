#!/usr/bin/python

from ase.db import connect
from ase.io import read, write
import numpy as np
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
import matplotlib.pyplot as plt

# Plots the CHE free energy diagram 

energydb = connect("Au211_thermo.db")
referdb = connect("references_BEEF_vdw_vasp.db")

def get_vibrational_energy(stat, method, geometry=None, symmetrynumber=None,\
        temperature=298.15, pressure=101325., spin=0):
    
    # Based on the method, calculate the vibrational energy
    # Need to give the database entry and the method
    # used to calculate the vibrational contribution

    from ase.thermochemistry import IdealGasThermo
    from ase.thermochemistry import HarmonicThermo
    
    # collect the atoms object 
    atoms = stat.toatoms()
    # Get things for thermo module
    potential_energy = atoms.get_potential_energy()
    vib_energies = np.array(stat.data.vib_energies).real
     

    if method == 'ideal_gas':
        thermo = IdealGasThermo(vib_energies = vib_energies,\
                                potentialenergy = potential_energy, \
                                atoms = atoms, \
                                geometry=geometry, \
                                symmetrynumber = symmetrynumber, \
                                spin = spin \
                                )
        gibbs = thermo.get_gibbs_energy(temperature, pressure)
        entropy = thermo.get_entropy(temperature, pressure)
        enthalpy = thermo.get_enthalpy(temperature, pressure)

    elif method == 'harmonic':
        print("vibs")
        print(vib_energies)
        print( potential_energy)
        thermo = HarmonicThermo(vib_energies = vib_energies, \
                                potentialenergy = potential_energy, \
                                )

        gibbs = thermo.get_helmholtz_energy(temperature, pressure)
        entropy = thermo.get_entropy(temperature, pressure)
        enthalpy = thermo.get_internal_energy(temperature, pressure)

    return {'G':gibbs, 'H':enthalpy, 'TS':entropy, 'E':potential_energy}

# Getting the gas phase energies

COgstat = referdb.get(formula='CO', functional='BF', \
                        pw_cutoff=400)

COg =  get_vibrational_energy(COgstat, method='ideal_gas', geometry='linear', \
        symmetrynumber=1)

# Getting energies of species
states = ['slab', 'CO_top', 'CO_bridge']
states_energy = {}
states_free_energy = {}

for i in range(len(states)):
    stat = energydb.get(states=states[i])
    energy = stat.energy
    states_energy[states[i]] = float(energy)

## Getting Gibbs free energy corrections
#for i in range(len(states)):
#    stat = energydb.get(states=states[i])
#    if states[i] == 'slab':
#        states_free_energy[states[i]] = float(stat.energy)
#    else:
#        free_energy = get_vibrational_energy(stat, method='harmonic')
#        states_free_energy[states[i]] = float(free_energy['G'])
#        print(states_free_energy[states[i]])

# Plotting a free energy diagram 
# Reference to CO2 gas phase 

print(COg['G'])
print(COg['E'])
CO_top_E = states_energy['CO_top']  - states_energy['slab'] - COg['E'] 
CO_bridge_E = states_energy['CO_bridge']  - states_energy['slab'] - COg['E']
CO_top_G = states_energy['CO_top'] + 0.002 - states_energy['slab'] - COg['G'] 
CO_bridge_G = states_energy['CO_bridge']+ 0.09 - states_energy['slab'] - COg['G']

print('CO top Electronic Energy:%1.3f'%CO_top_E)
print('CO bridge Electronic Energy:%1.3f'%CO_bridge_E)
print('CO top Free Energy:%1.3f'%CO_top_G)
print('CO bridge Free Energy:%1.3f'%CO_bridge_G)

#####################################
# Plotting a FED
#energies = [0, COOHa, COa, COfree]
#plot_energies = [ener for ener in energies for _ in (0, 1)]
#x = np.arange(len(plot_energies))
#
#plt.plot(x, plot_energies, 'r-')
#plt.ylabel('Energies [eV]')
#plt.savefig('FED.png')



                                
