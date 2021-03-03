from ase.io import read, write 
import sys
import numpy as np 
from copy import deepcopy

if __name__ == '__main__':

    atoms1 = read(sys.argv[1])
    atoms2 = read(sys.argv[2])
    atoms3 = read(sys.argv[3])

    cell = atoms1.get_cell()
    length = np.linalg.norm(cell[0,:])
    print(length)

    for atom in atoms2:
        atom.x += length + 4
    for atom in atoms3:
        atom.x += 2 * length + 8

    atoms = deepcopy(atoms1)
    atoms.extend(atoms2)
    atoms.extend(atoms3)

    atoms.write('Au_Pb.traj')
