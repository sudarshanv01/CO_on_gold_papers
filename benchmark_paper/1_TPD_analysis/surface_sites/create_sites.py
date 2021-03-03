#!/usr/bin/python


from ase.io import read
from ase.build import surface

au_211 = surface('Au', (2, 1, 1), 6, periodic=True, vacuum=10)
au_310 = surface('Au', (3, 1, 0), 6, periodic=True, vacuum=10)

n310 = au_310.repeat([2, 3, 1])
n211 = au_211.repeat([2, 3, 1])

n211.write('211.traj')
n310.write('310.traj')
