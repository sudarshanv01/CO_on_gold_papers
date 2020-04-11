#!/usr/bin/python

""" Get the AIMD energies for CO adsorption energies """

from useful_functions import get_vibrational_energy, AutoVivification
from ase.thermochemistry import IdealGasThermo
from ase.thermochemistry import HarmonicThermo
from ase.io import read
from pprint import pprint
import numpy as np

def get_harmonic(vib_list, temperature, pressure):
    cmtoeV = 0.00012
    vib_energies = vib_list * cmtoeV
    thermo = HarmonicThermo(vib_energies = vib_energies, \
                            #potentialenergy = potential_energy, \
                            )
    gibbs = thermo.get_helmholtz_energy(temperature, verbose=False)
    #entropy = thermo.get_entropy(temperature, pressure)
    #enthalpy = thermo.get_internal_energy(temperature, pressure)
    return gibbs

# Gas phase free energies
COg = {}
vib_list = [1993.435752, 232.820271, 232.021371]
cmtoeV = 0.00012
COg['E'] = -12.0900
COg_atoms = read('COg/CO.traj')
vib_ener = cmtoeV * np.array(vib_list)
thermo = IdealGasThermo( vib_ener,\
                        potentialenergy = COg['E'], \
                        atoms = COg_atoms, \
                        geometry='linear', \
                        symmetrynumber = 2, \
                        spin = 0 \
                        )
COg['G'] = thermo.get_gibbs_energy(temperature=298, pressure=101325, verbose=False)
COg['S'] = thermo.get_entropy(temperature=298, pressure=101325, verbose=False)


## AIMD free energies
AIMD = AutoVivification()
AIMD['211']['COa']['E'] = -209.1707
AIMD['211']['slab']['E'] = -197.203
vib_211 = np.array([1814.001, 367.591, 340.559, 230.796, 167.919, 69.744])
## Free energy contribution based on static calculation
AIMD['211']['COa']['G'] = AIMD['211']['COa']['E'] - AIMD['211']['slab']['E'] \
                            - COg['G'] \
                            + get_harmonic(vib_211, 298.15, 101325)
print('AIMD: Free energy of CO on Au(211): %1.3f'%AIMD['211']['COa']['G'])
AIMD['100']['COa']['E'] = -275.098
AIMD['100']['slab']['E'] = -262.855
vib_100 = np.array([1815.706, 371.374, 358.245, 248.225, 172.559, 78.188])
## Free energy contribution based on static calculation
AIMD['100']['COa']['G'] = AIMD['100']['COa']['E'] - AIMD['100']['slab']['E'] \
                            - COg['G'] \
                            + get_harmonic(vib_100, 298.15, 101325)

print('AIMD: Free energy of CO on Au(100): %1.3f'%AIMD['100']['COa']['G'])


## Gas phase energies
Vacuum = AutoVivification()
Vacuum['211']['COa']['E'] = -12.92
Vacuum['211']['slab']['E'] = -0.36
Vacuum['211']['COa']['G'] = Vacuum['211']['COa']['E'] - Vacuum['211']['slab']['E'] \
                            - COg['G'] \
                            + get_harmonic(vib_211, 298.15, 101325)

print('VACUUM: Free energy of CO on Au(211): %1.3f'%Vacuum['211']['COa']['G'])

Vacuum['100']['COa']['E'] = -12.71
Vacuum['100']['slab']['E'] = -0.27
Vacuum['100']['COa']['G'] = Vacuum['100']['COa']['E'] - Vacuum['100']['slab']['E'] \
                            - COg['G'] \
                            + get_harmonic(vib_100, 298.15, 101325)

print('VACUUM: Free energy of CO on Au(100): %1.3f'%Vacuum['100']['COa']['G'])

pprint(AIMD)
