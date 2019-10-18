#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read

# Plot the equilibrium coverage on a log scale for a partial pressure of CO
# equal to one

deltaE = np.linspace(0.1, 1.0, 200)
# adsorbate thermochemistry
#vibration_energies_ads = 0.00012 * np.array([50.0, 50.0, 129.65, 155.55, 189.8, 227.5, 2073.25])
vibration_energies_ads = 0.00012 * np.array([129.65, 155.55, 189.8, 227.5, 2073.25])
thermo_ads = HarmonicThermo(vibration_energies_ads)
# gas CO thermochemistry
atoms = read('co.traj')
vibration_energies_gas = 0.00012 * np.array([89.8, 127.2, 2145.5])
thermo_gas = IdealGasThermo(vibration_energies_gas, atoms = atoms, \
        geometry='linear', symmetrynumber=1, spin=0)
T_range = np.linspace(160, 300, num=200) # K
print(T_range)
T_plot = np.array([160, 200, 250, 300])
kB = 8.617e-05 # eV/K
pco = 101325. #pressure of CO

for i in range(len(T_plot)):
    T = T_plot[i]
    entropy_ads = thermo_ads.get_entropy(temperature=T) 
    entropy_gas = thermo_gas.get_entropy(temperature=T, \
                                                      pressure=pco)
    print('Temperature: %d'%T)
    print('Entropy of gas: %1.7f'%entropy_gas)
    print('Entropy of adsorbate: %1.7f'%entropy_ads)
    entropy_difference =  (entropy_gas - entropy_ads)
    print('Entropic difference: %1.5f'%entropy_difference)
    free_correction = -1 * entropy_difference * T #free_correction_ads - free_correction_gas
    print('-T Delta S %1.2f'%(free_correction))
    deltaG = deltaE + free_correction
    print('Delta G')
    print(deltaG)
    K = np.exp( -1 * deltaG / kB / T )
    partial_co = pco / 101325
    theta_co = 1 / ( 1 + K / partial_co )

    plt.plot(-deltaE, theta_co, label='temperature: ' + str(T) + 'K', lw=3)

#plt.yscale('log')
plt.ylabel(r'$\theta_{CO}$ \ ML')
plt.axvline(-0.33, color='k', ls='--', lw=2)
plt.axvspan(-0.45, -0.7, color='indianred', alpha=0.25) #label='Estimated from TPD')
plt.annotate('BEEF-vdW 3x3', xy=(-0.33, 0.2), color='k').draggable()
plt.annotate('Estimated from TPD', xy=(-0.5, 0.4), color='indianred').draggable()
plt.ylim([None, 1])
plt.xlabel(r'$-\Delta E_{TPD} $ \ eV')
plt.legend(loc='best')
plt.savefig('deltaE_coverage_T.pdf')
plt.show()
