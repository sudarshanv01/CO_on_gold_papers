
from aiida import orm
from aiida.engine import run, submit
from pprint import pprint
import numpy as np
import json
from aiida.tools.groups import GroupPath
from ase import Atoms
from ase import build
from pathlib import Path


"""
Sample all possible adsorption sites for a given structure
"""

def runner(structure, parameters, kpoint_mesh, dynamics=[]):

    # use the base VASP workchain
    RelaxVasp = WorkflowFactory('vasp.relax')

    # generate the inputs
    builder = RelaxVasp.get_builder()

    builder.metadata.label = 'Lattice Relaxation Calculation'
    builder.metadata.description = 'Relaxing for generating vacancies'

    builder.verbose = orm.Bool(True)

    # set the code
    code = load_code('vasp-5.4.4@dtu_xeon8')
    builder.code = code

    # set the structure
    StructureData = DataFactory('structure')
    structure = StructureData(ase=structure)
    builder.structure = structure

    # k-points
    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh(kpoint_mesh)
    builder.kpoints = kpoints

    # set the parameters
    builder.parameters = orm.Dict(dict=parameters)

    # set dynamics
    builder.dynamics = dynamics

    # set the PAW potentials
    builder.potential_family = orm.Str('PBE.54')
    builder.potential_mapping = orm.Dict(dict={'Au':'Au'})

    # setup options
    options = {}
    options['resources'] = {'num_machines': 1}
    options['max_wallclock_seconds'] = 60 * 60
    builder.options = orm.Dict(dict=options)

    # settings dics
    settings = {'parser_settings': {}} 
    builder.settings = orm.Dict(dict=settings) 

    builder.relax.perform = orm.Bool(True)
    builder.relax.algo = orm.Str('cg')
    builder.relax.force_cutoff = orm.Float(1e-2)
    builder.relax.positions = orm.Bool(True)
    builder.relax.shape = orm.Bool(False)
    builder.relax.volume = orm.Bool(False)
    builder.relax.steps = orm.Int(500)

    calculation = submit( RelaxVasp, **builder)
    #path = GroupPath()
    #path["lattices/RPBE"].get_group().add_nodes(calculation)


if __name__ == '__main__':

    node = load_node(1056)
    testdir = 'testdir'
    Path(testdir).mkdir(parents=True, exist_ok=True)
    # get the parameters from the lattice calculations
    parameters = node.inputs.parameters.get_dict()
    parameters['incar']['ldipol'] = True
    parameters['incar']['idipol'] = 3
    parameters['incar']['dipol'] = [0.5, 0.5, 0.5]
    
    structure = node.outputs.relax.structure
    ase_structure = structure.get_ase()
    a = np.linalg.norm(ase_structure.cell[0])
    unit_cell = build.bulk( 'Au', 'fcc', a=a)

    miller_index = (3, 1, 0)
    facet = ''.join([str(elem) for elem in miller_index])
    repeat_ndim = 2
    layers = 11
    
    factors = [1, ]#2, 3, 4 ]
    for factor in factors:
        # generate the new structure
        atoms = build.surface(unit_cell, miller_index, layers=layers, vacuum=8.0, periodic=True)
        if repeat_ndim == 1:
            atoms = atoms.repeat((factor, 1, 1))
            kpoint_mesh = [np.ciel(12/factor), 12, 1]
        elif repeat_ndim == 2:
            atoms = atoms.repeat((factor, factor, 1))
            kpoint_mesh = [np.ceil(12/factor), np.ceil(12/factor), 1]

        # constrain the bottom layers of the metal
        all_z = [atom.position[2] for atom in atoms]
        unique_z = np.sort(np.unique(all_z))
        split_z = np.array_split(unique_z, 2)
        dynamics_pos = []
        for atom in atoms:
            if atom.position[2] in split_z[1]:
                dynamics_pos.append([True, True, True])
            elif atom.position[2] in split_z[0]:
                dynamics_pos.append([False, False, False])
        dynamics = {'positions_dof': orm.List(list=dynamics_pos)}
        runner(atoms, parameters, kpoint_mesh, dynamics)
        
        #atoms.center()
        #atoms.write(f'{testdir}/{facet}_{factor}.cif')


