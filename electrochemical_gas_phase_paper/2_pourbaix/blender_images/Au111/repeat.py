
from ase.io import read, write
import sys

if __name__ == '__main__':
    rep = int(sys.argv[3])
    atoms = read(sys.argv[1])
    natoms = atoms.repeat([rep, rep, 1])
    natoms.write(sys.argv[2])

