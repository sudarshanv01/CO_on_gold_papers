#!/usr/bin/python

from ase.neb import NEBTools
from ase.io import read


neb_atoms = read('neb.traj', ':')
neb_atoms.reverse()

neb = NEBTools(neb_atoms)
# get the barrier
barrier = neb.get_barrier()
print('Forward Barrier height is: %1.2f'%barrier[0])
print('Reverse Barrier height is: %1.2f'%barrier[1])
# plot the band
fig = neb.plot_band()
fig.savefig('neb_band.svg')
