
"""
Run vibration calculations with VASP 
"""
from pprint import pprint
import numpy as np
from pathlib import Path
import os.path as op
import json
from ase.io import read
import time
import sys
VerifyVasp = WorkflowFactory('vasp.verify')
RelaxVasp = WorkflowFactory('vasp.relax')
from aiida.tools.groups import GroupPath
from aiida import orm, engine

def get_single_energy(groupname):

    qbs = QueryBuilder()
    qbs.append(Group, filters={'label':groupname}, tag='Group')
    qbs.append(RelaxVasp, with_group='Group', tag='Screening')

    assert len(qbs.all(flat=True)) == 1; 'Only one single calculation'
    for node in qbs.all(flat=True):
        results = node.outputs.misc.get_dict()
        energy_outcar = parse_energy_from_OUTCAR(node.outputs.retrieved)
        energy = results['total_energies']['energy_extrapolated']
        try:
            assert energy_outcar == energy
        except AssertionError:
            print(f'Energy of single {energy_outcar} != {energy}; Replacing with energy from OUTCAR')
            energy = energy_outcar
    
    return energy

def parse_energy_from_OUTCAR(node):
    with node.open('OUTCAR', 'r') as handle:
        atoms_outcar = read(handle, format='vasp-out')
    energy = atoms_outcar.calc.results['energy']
    return energy

if __name__ == '__main__':

    groupnames = []
    qb = QueryBuilder()
    qb.append(Group, tag='Group', filters={'label':{'like': 'co%'}})

    for group in qb.all(flat=True):
        groupnames.append(group.label)

    print(f'Looking at groups {groupnames}')

    for group_name in groupnames:
        print(f'Group:{group_name}')

        qb = QueryBuilder()
        qb.append(Group, filters={'label':group_name}, tag='Group')
        qb.append(RelaxVasp, with_group='Group', tag='Screening')


        energies = []
        nodes_relax = []
        for i, node in enumerate(qb.all(flat=True)):
            if node.is_finished_ok:
                # get the energies
                results = node.outputs.misc.get_dict()            
                energy_CO = results['total_energies']['energy_extrapolated']
                energy_CO_outcar = parse_energy_from_OUTCAR(node.outputs.retrieved)

                try:
                    assert energy_CO == energy_CO_outcar 
                except AssertionError:
                    energy_CO = energy_CO_outcar
                energies.append(energy_CO)
                nodes_relax.append(node)
        lowest_energy_node = nodes_relax[np.argmin(energies)]

        # create new group to house the vibration calculations
        new_group_name = group_name.replace('co_adsorption', 'dynmat').replace('references', 'dynmat')
        group = Group(label=new_group_name)
        try:
            group.store()
        except Exception:
            pass

        # Create the builder
        builder = VerifyVasp.get_builder()

        # specify code 
        builder.code = load_code('vasp-5.4.4-vdw@juwels_scr')

        # specify the structure
        builder.structure = lowest_energy_node.outputs.relax.structure

        # specify dynamics
        # freeze everything that is not CO
        ase_structure = lowest_energy_node.outputs.relax.structure.get_ase()
        dof_positions = [] 

        for atom in ase_structure:
            if atom.symbol in ['C', 'O']:
                dof_positions.append([True, True, True])
            else:
                dof_positions.append([False, False, False])

        dynamics = {'positions_dof': orm.List(list=dof_positions)}
        builder.dynamics = dynamics

        # specify kpoints
        builder.kpoints = lowest_energy_node.inputs.kpoints

        # specify options
        if len(ase_structure) < 3:
            num_machines = 1
        elif len(ase_structure) < 20:
            num_machines = 2
        else:
            num_machines = 4
        options = {}
        options['resources'] = {'num_machines': num_machines}
        options['max_wallclock_seconds'] = 24 * 60 * 60
        builder.options = orm.Dict(dict=options)

        # specify parser settings
        settings = {'parser_settings': {'add_dynmat':True, 'add_hessian':True}} 
        builder.settings = orm.Dict(dict=settings) 

        # specify parameters
        parameters = lowest_energy_node.inputs.parameters.get_dict()
        parameters['incar'].pop('npar', {})
        parameters['incar']['ibrion'] = 5
        parameters['incar']['nfree'] = 2
        parameters['incar']['nwrite'] = 3
        parameters['incar']['isym'] = 0
        builder.parameters = orm.Dict(dict=parameters)

        # specify potential family
        builder.potential_family = lowest_energy_node.inputs.potential_family
        builder.potential_mapping = lowest_energy_node.inputs.potential_mapping
        builder.clean_workdir = orm.Bool(False)

        calculation = engine.submit(builder)
        path = GroupPath()
        path[new_group_name].get_group().add_nodes(calculation)
        time.sleep(3)

	

