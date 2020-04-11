#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

CO = {'0.0':-153.937581200, '-0.25':-154.27165027616925, '-0.5':-154.7735650598929, '-0.75':-155.06954057050217}
slab = {'0.0':-141.8281459, '-0.25':-142.252497969, '-0.5':-142.63194041860788, '-0.75':-143.04348751346606}

COg = -12.09009295
energies = []
charges = []
for key in CO:
    charges.append(float(key))
    energy = CO[key] - slab[key] - COg
    energies.append(energy)

plt.plot(charges, energies, 'bo', markersize=16)
plt.ylabel(r'$\Delta E \ eV$')
plt.xlabel('Homogeneous background charge  e')
plt.grid(False)
plt.savefig('charge_energy.png')


