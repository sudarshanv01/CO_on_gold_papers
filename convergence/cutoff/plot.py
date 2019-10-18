#!/usr/bin/python

from ase.db import connect
from ase.io import read, write
import numpy as np
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11,16.1)

# Plots the CHE free energy diagram 

energydb = connect("Au211_BEEF_cutoff_convergence.db")
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
states = ['slab', 'CO_top']
cutoffs = np.array([300, 400, 500, 600.0])
states_energy = {}
states_free_energy = {}

for i in range(len(states)):
    states_energy[states[i]] = []
    for j in range(len(cutoffs)):
        stat = energydb.get(states=states[i], pw_cutoff = cutoffs[j])
        energy = stat.energy
        states_energy[states[i]].append(float(energy))

CO_top_E = np.array(states_energy['CO_top'])  - np.array(states_energy['slab']) - COg['E'] 

#####
plt.plot(cutoffs, CO_top_E, 'ko')
plt.plot(cutoffs, CO_top_E, 'b-', linewidth=3)
plt.ylabel(r'$\Delta E_{CO}$ \ eV', fontsize=24)
plt.xlabel('Cutoff \ eV', fontsize=24)
plt.grid(False)
plt.savefig('convergence_cutoff.png')

