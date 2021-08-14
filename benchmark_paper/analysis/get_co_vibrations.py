

"""
Plot the adsorption energies of CO with different functionals
"""

from collections import defaultdict
VerifyVasp = WorkflowFactory('vasp.verify')
RelaxVasp = WorkflowFactory('vasp.relax')
from pprint import pprint
import numpy as np
from pathlib import Path
import os.path as op
import json
from ase.io import read

def parse_vibrations_from_OUTCAR(node):
    with node.open('OUTCAR', 'r') as handle:
        outcar = handle.readlines()
    # read the vibrations from the OUTCAR file 
    all_freq = []
    for line in outcar:
        if 'THz' in line:
            frequencies_data = line.split()
            # check if it is imaginary 
            hnu_in_cminv = frequencies_data[-4]
            if 'f/i' in line:
                hnu_in_cminv = -1 * float(hnu_in_cminv)
            else:
                hnu_in_cminv = float(hnu_in_cminv)
            all_freq.append(hnu_in_cminv)
    # remove redunant frequencies 
    index_cut = len(all_freq) / 2
    index_cut = int(index_cut)
    all_freq = np.array(all_freq)
    all_freq = all_freq[:index_cut]

    return all_freq.tolist()

if __name__ == '__main__':

    groupnames = []
    qb = QueryBuilder()
    qb.append(Group, tag='Group', filters={'label':{'like': 'dynmat%'}})

    for group in qb.all(flat=True):
        groupnames.append(group.label)

    print(f'Looking at groups {groupnames}')
    vibrations = defaultdict(lambda: defaultdict(dict)) 
    gas_vib = {}

    for group_name in groupnames:
        print(f'Group:{group_name}')

        qb = QueryBuilder()
        qb.append(Group, filters={'label':group_name}, tag='Group')
        qb.append(VerifyVasp, with_group='Group', tag='Screening')

        if len(group_name.split('/')) == 4:
            _, facet, cell, functional = group_name.split('/')
            _is_gas = False
        elif len(group_name.split('/')) == 2:
            _, functional = group_name.split('/')
            _is_gas = True

        step_atoms = float(cell.split('x')[-1]) 
        coverage = 1 / step_atoms

        energies = []
        for i, node in enumerate(qb.all(flat=True)):
            if node.is_finished_ok:
                energies = parse_vibrations_from_OUTCAR(node.outputs.retrieved)
                if not _is_gas:
                    vibrations[functional][facet][coverage] =  energies      
                else:
                    gas_vib[functional] = energies

    with open(op.join('output/gas_phase_vib.json'), 'w') as handle:
        json.dump(gas_vib, handle, indent=4)
    with open(op.join('output/vibrations.json'), 'w') as handle:
        json.dump(vibrations, handle, indent=4)
