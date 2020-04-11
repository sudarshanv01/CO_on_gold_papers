#!/usr/bin/python

import os
from ase import Atoms
import sys
test_structures = 'structures_testing/'
os.system('mkdir -p ' + test_structures)

# Node where the structures are stored
node = load_node(sys.argv[1])
type = sys.argv[2]


if type == 'slab':

    atoms_dict = node.attributes
    atoms = Atoms(**atoms_dict)
    atoms.write(test_structures + 'slab.traj')

if type == 'adsorbate':

    results = node.attributes
    for structure_number, structure_dict in results.items():
        atoms = Atoms(**structure_dict)
        atoms.write(test_structures + 'structure' + structure_number + '.traj')
