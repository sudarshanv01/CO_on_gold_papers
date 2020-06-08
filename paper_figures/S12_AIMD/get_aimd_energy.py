#!/usr/bin/python

""" Get the AIMD energies for CO adsorption energies """

from useful_functions import get_vibrational_energy, AutoVivification
from ase.thermochemistry import IdealGasThermo
from ase.thermochemistry import HarmonicThermo
from ase.io import read
from pprint import pprint
import numpy as np
from glob import glob

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

def get_config_entropy(temperature, theta):
    temperature = np.array(temperature)
    theta = np.array(theta)
    config = 8.617e-5 * temperature * np.log((theta)/(1-theta))
    return config



## AIMD free energies

if __name__ == '__main__':
    # Gas phase free energies
    COg = {}
    COg['RPBE'] = {}
    COg['BEEF'] = {}
    vib_list = [2121.52, 39.91, 39.45]
    cmtoeV = 0.00012
    COg['BEEF']['E'] = -12.0900
    COg['RPBE']['E'] = -14.4726
    # COg['RPBE']['E']
    folders = {'211'}
    COg_atoms = read('COg/CO.traj')
    vib_ener = cmtoeV * np.array(vib_list)
    thermo = IdealGasThermo( vib_ener,\
                            potentialenergy = COg['BEEF']['E'], \
                            atoms = COg_atoms, \
                            geometry='linear', \
                            symmetrynumber = 2, \
                            spin = 0 \
                            )

    COg['BEEF']['G'] = thermo.get_gibbs_energy(temperature=298, pressure=101325, verbose=False)
    COg['BEEF']['S'] = thermo.get_entropy(temperature=298, pressure=101325, verbose=False)

    thermo = IdealGasThermo( vib_ener,\
                            potentialenergy = COg['RPBE']['E'], \
                            atoms = COg_atoms, \
                            geometry='linear', \
                            symmetrynumber = 2, \
                            spin = 0 \
                            )

    COg['RPBE']['G'] = thermo.get_gibbs_energy(temperature=298, pressure=101325, verbose=False)
    COg['RPBE']['S'] = thermo.get_entropy(temperature=298, pressure=101325, verbose=False)
    vibrations = {'211':np.array([2044.1, 282.2, 201.5, 188.5, 38.3, 11.5]),
                  '100':np.array([1886.5, 315.2, 273.4, 222.2, 152.7, 49.8])}

    facets = {'BEEF':['211', '100'], 'RPBE':['211']}
    functionals = ['BEEF', 'RPBE']

    AIMD = AutoVivification()
    results = AutoVivification()

    for functional in functionals:
        for facet in facets[functional]:
            files = glob('Au_'+facet+'_'+functional+'/*.txt')
            energies = []
            AIMD[functional][facet]['slab'] = []
            AIMD[functional][facet]['COa'] = []
            for f in files:
                species = f.split('/')[-1].split('.')[0].split('_')[2]
                with open(f, 'r') as fil:
                    energy = float(fil.readline())
                    AIMD[functional][facet][species].append(energy)
    pprint(AIMD)
    for functional in AIMD:
        for facet in AIMD[functional]:
            rel_energy = min(AIMD[functional][facet]['COa']) \
                        - min(AIMD[functional][facet]['slab']) \
                        - COg[functional]['E']
            results[functional][facet]['E'] = rel_energy
            if facet == '211':
                rel_free_energy = min(AIMD[functional][facet]['COa']) \
                            - min(AIMD[functional][facet]['slab']) \
                            - COg[functional]['G']\
                            + get_harmonic(vibrations[facet], 298.15, 101325)\
                            + get_config_entropy(298, 1/3)
            elif facet == '100':
                rel_free_energy = min(AIMD[functional][facet]['COa']) \
                            - min(AIMD[functional][facet]['slab']) \
                            - COg[functional]['G']\
                            + get_harmonic(vibrations[facet], 298.15, 101325)\
                            + get_config_entropy(298, 1/9)
            results[functional][facet]['G'] = rel_free_energy
    pprint(results)


#
#
# ## Free energy contribution based on static calculation
# AIMD['100']['COa']['G'] = AIMD['100']['COa']['E'] - AIMD['100']['slab']['E'] \
#                             - COg['G'] \
#                             + get_harmonic(vib_100, 298.15, 101325)\
#                             + get_config_entropy(298, 1/9)
#
# AIMD['100']['COa']['dE'] = AIMD['100']['COa']['E'] - AIMD['100']['slab']['E'] \
#                             - COg['E']
#
# # print('AIMD: Free energy of CO on Au(100): %1.3f'%AIMD['100']['COa']['G'])
#
#
# ## Gas phase energies
# Vacuum = AutoVivification()
# Vacuum['211']['COa']['E'] = -12.92
# Vacuum['211']['slab']['E'] = -0.36
# Vacuum['211']['COa']['dG'] = Vacuum['211']['COa']['E'] - Vacuum['211']['slab']['E'] \
#                             - COg['G'] \
#                             + get_harmonic(vib_211, 298.15, 101325)\
#                             + get_config_entropy(298, 1/3)
# Vacuum['211']['COa']['dE'] = Vacuum['211']['COa']['E'] - Vacuum['211']['slab']['E'] \
#                             - COg['E'] \
#
# # print('VACUUM: Free energy of CO on Au(211): %1.3f'%Vacuum['211']['COa']['G'])
#
# Vacuum['100']['COa']['E'] = -12.71
# Vacuum['100']['slab']['E'] = -0.27
# Vacuum['100']['COa']['dG'] = Vacuum['100']['COa']['E'] - Vacuum['100']['slab']['E'] \
#                             - COg['G'] \
#                             + get_harmonic(vib_100, 298.15, 101325)\
#                             + get_config_entropy(298, 1/9)
# Vacuum['100']['COa']['dE'] = Vacuum['100']['COa']['E'] - Vacuum['100']['slab']['E'] \
#                             - COg['E'] \
# # print('VACUUM: Free energy of CO on Au(100): %1.3f'%Vacuum['100']['COa']['G'])
#
# pprint(AIMD)
# pprint(Vacuum)
