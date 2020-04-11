#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

CO = {'0.0':-209.033003308042, '-0.5':-212.24799109955765, '-1.0':-212.81674265999436}
slab = {'0.0':-197.20179605109155, '-0.5':-199.994005694662, '-1.0':-200.60210690550662}

COg = -12.09009295
energies = []
charges = []
for key in CO:
    charges.append(float(key))
    energy = CO[key] - slab[key] - COg
    energies.append(energy)

plt.plot(charges, energies, 'bo', markersize=16)
plt.ylabel(r'$\Delta E \ eV$')
plt.xlabel('Charge on slab  eV')
plt.grid(False)
plt.savefig('charge_energy.png')


