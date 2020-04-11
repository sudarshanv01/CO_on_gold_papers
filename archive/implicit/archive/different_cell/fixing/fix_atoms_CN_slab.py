#!/usr/bin/env python

import sys
import numpy as np

from ase.build import fcc100, fcc111, fcc110, fcc211
from ase.visualize import view
from ase.io import read, write
from ase.constraints import FixAtoms


def get_CN(atoms, rcut, type_a='*', type_b='*'):
    rpos = atoms.get_scaled_positions()
    cell = atoms.get_cell()
    inds = []
    for ty in [type_a, type_b]:
        if ty == '*':
            ty = list(range(len(atoms)))
        else:
            ty = np.array([np.where(atoms.get_atomic_numbers() == t)[0]
                           for t in ty]).flatten()
        inds.append(ty)
    cns = []
    for i in range(len(inds[0])):
        cns.append(__get_immediate_CN(rpos[inds[1], :], rpos[i, :], cell, rcut).size - 1)
    return(inds[0], cns)


def __get_immediate_CN(pos_array, pos, cell, rcut):
    ''' function to calculate distance array (pos_array - pos) and determine
        entries within distance rcut
        input:  pos_array = positions which to calculate distances from
                pos       = origin position
                cell      = transformation for distance vectors
                rcut      = cutoff for which to obtain points within distance
        output: cord      = entries of points in pos_array within distance rcut
    '''
    dvec = _correct_vec(pos_array-pos)
    dvec = np.dot(dvec, cell)
    dist = np.linalg.norm(dvec, axis=1)
    cord = np.where(dist <= rcut)[0]
    return(cord)


def _correct_vec(vec):
    ''' correct vectors in fractional coordinates
        (assuming vectors minimal connection between 2 points)
    '''
    vec[np.where(vec >= 0.5)] -= 1.0
    vec[np.where(vec < -0.5)] += 1.0
    return(vec)


def set_constraints_for_slab(slab, half=False, rcut=2.6):  # std = Cu/beef/spss
    ''' NOTE: this picks only the max Coordinated slab atoms as bulk
                puts a warning if you have differently coordinated atoms
                (i.e. no clear distinction between bulk and surface)
        half - also fix lower surface (pick of min-z coordinate hardcoded)
        TODO: make automatic convergence of rcut
    '''
    slab.set_pbc(True)
    inds, CNs = get_CN(slab, rcut=rcut)
    if np.unique(CNs).size != 2:
        print("Warning: no clear-cut separation of bulk and surface: CNs=%s" % str(CNs))
    ind_bulk = np.where(CNs == np.unique(CNs).max())[0]  # here max coords are picked
    if half:  # add lower half to bulk
        pos = slab.get_positions()
        ind_add = np.where(pos[:, 2] < pos[ind_bulk, 2].max())[0]
        ind_bulk = np.unique(np.hstack((ind_bulk, ind_add)).flatten())
    c = FixAtoms(indices=ind_bulk)
    slab.set_constraint(c)


if __name__ == "__main__":
    cmdlineargs = sys.argv
    if len(sys.argv) == 1:
        from copy import deepcopy
        print('no input given - demo via Cu(100): see ase gui')
        slab = fcc100('Cu', a=3.67412, size=(3, 3, 5), vacuum=10.0, orthogonal=True)
        toview = [deepcopy(slab)]
        set_constraints_for_slab(slab, half=False)
        toview.append(slab)
        view(toview)

    else:
        a = read(sys.argv[1])
        set_constraints_for_slab(a, half=False)
        write('.'.join(sys.argv[1].split('.')[:-1]+['traj']), a)
