

"""
Plot the adsorption energies of CO with different functionals
"""

from collections import defaultdict
RelaxVasp = WorkflowFactory('vasp.relax')
from pprint import pprint
import numpy as np
from pathlib import Path
import os.path as op
import json

def get_single_energy(groupname):

    qbs = QueryBuilder()
    qbs.append(Group, filters={'label':groupname}, tag='Group')
    qbs.append(RelaxVasp, with_group='Group', tag='Screening')

    assert len(qbs.all(flat=True)) == 1; 'Only one single calculation'
    for node in qbs.all(flat=True):
        results = node.outputs.misc.get_dict()
        energy = results['total_energies']['energy_extrapolated']
    
    return energy

if __name__ == '__main__':

    groupnames = []
    qb = QueryBuilder()
    qb.append(Group, tag='Group', filters={'label':{'like': 'co%'}})

    for group in qb.all(flat=True):
        groupnames.append(group.label)

    print(f'Looking at groups {groupnames}')
    internal_energies = defaultdict(lambda: defaultdict(list)) 

    for group_name in groupnames:
        print(f'Group:{group_name}')

        qb = QueryBuilder()
        qb.append(Group, filters={'label':group_name}, tag='Group')
        qb.append(RelaxVasp, with_group='Group', tag='Screening')

        _, facet, cell, functional = group_name.split('/')
        slab_group_name = group_name.replace('co_adsorption', 'slab')
        energy_slab = get_single_energy(slab_group_name)        
        reference_groupname = f'references/{functional}'
        energy_ref = get_single_energy(reference_groupname)

        step_atoms = float(cell.split('x')[-1]) 
        coverage = 1 / step_atoms

        # Trajectories
        outdir = op.join(*['output'] + group_name.split('/')) 
        Path(outdir).mkdir(parents=True, exist_ok=True)

        energies = []
        for i, node in enumerate(qb.all(flat=True)):
            if node.is_finished_ok:

                # get the energies
                results = node.outputs.misc.get_dict()            
                energy_CO = results['total_energies']['energy_extrapolated']
                energy_ads = energy_CO - energy_slab - energy_ref
                energies.append(energy_ads)

                # get the structures as well
                structure = node.outputs.relax.structure
                ase_structure = structure.get_ase()
                ase_structure.write(op.join(outdir, f'co_{i}.traj'))

        internal_energies[functional][facet].append([coverage, np.min(energies)])        

    with open(op.join('output/energies.json'), 'w') as handle:
        json.dump(internal_energies, handle, indent=4)
