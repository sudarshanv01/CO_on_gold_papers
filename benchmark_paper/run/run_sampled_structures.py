
from aiida import orm
from aiida.engine import run, submit
from pprint import pprint
import numpy as np
import json
from aiida.tools.groups import GroupPath
from ase import Atoms
from ase import build
from pathlib import Path
from aiida.orm import QueryBuilder, Group
import sys
import ase
import ase.data as ase_data
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
import time


"""
Sample all possible adsorption sites for a given structure
"""

def runner_slab(structure, parameters, kpoint_mesh, dynamics=[]):

    # use the base VASP workchain
    RelaxVasp = WorkflowFactory('vasp.relax')

    # generate the inputs
    builder = RelaxVasp.get_builder()

    builder.metadata.label = 'Slab Relaxation Calculation'
    builder.metadata.description = 'Slab calculation for finding the adsorption energy'

    builder.verbose = orm.Bool(True)

    # set the code
    code = load_code('vasp-5.4.4@'+computer)
    builder.code = code

    # set the structure
    structure = StructureData(ase=structure)
    builder.structure = structure

    # k-points
    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh(kpoint_mesh)
    builder.kpoints = kpoints

    total_k = kpoint_mesh[0] * kpoint_mesh[1]
    if total_k % 4 == 0:
        npar = 4
    elif total_k % 3 == 0:
        npar = 3
    elif total_k % 2 == 0:
        npar = 2

    # Alter the parameters
    parameters['incar']['npar'] = npar

    # set the parameters
    builder.parameters = orm.Dict(dict=parameters)

    # set dynamics
    builder.dynamics = dynamics

    # set the PAW potentials
    builder.potential_family = orm.Str('PBE.54')
    builder.potential_mapping = orm.Dict(dict={'Au':'Au'})

    # setup options
    options = {}
    options['resources'] = {'num_machines': num_proc}
    options['max_wallclock_seconds'] = 24 * 60 * 60
    builder.options = orm.Dict(dict=options)

    # settings dics
    settings = {'parser_settings': {}} 
    builder.settings = orm.Dict(dict=settings) 

    builder.relax.perform = orm.Bool(True)
    builder.relax.algo = orm.Str('cg')
    builder.relax.force_cutoff = orm.Float(2.5e-2)
    builder.relax.positions = orm.Bool(True)
    builder.relax.shape = orm.Bool(False)
    builder.relax.volume = orm.Bool(False)
    builder.relax.steps = orm.Int(500)

    calculation = submit( RelaxVasp, **builder)

    # store the calculation in the right group
    new_group_name = f'slab/{facet}/{repeat}/{new_group_base}'
    group = Group(label=new_group_name)
    try:
        group.store()
    except Exception:
        pass

    path = GroupPath()
    path[new_group_name].get_group().add_nodes(calculation)

def runner_CO(structure, parameters, kpoint_mesh, dynamics=[]):

    # use the base VASP workchain
    RelaxVasp = WorkflowFactory('vasp.relax')

    # generate the inputs
    builder = RelaxVasp.get_builder()

    builder.metadata.label = 'CO Relaxation Calculation'
    builder.metadata.description = 'Calculation for finding the adsorption energy'

    builder.verbose = orm.Bool(True)

    # set the code
    code = load_code('vasp-5.4.4@'+computer)
    builder.code = code

    # set the structure
    structure = StructureData(ase=structure)
    builder.structure = structure

    # k-points
    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh(kpoint_mesh)
    builder.kpoints = kpoints

    total_k = kpoint_mesh[0] * kpoint_mesh[1]
    if total_k % 4 == 0:
        npar = 4
    elif total_k % 3 == 0:
        npar = 3
    elif total_k % 2 == 0:
        npar = 2

    # Alter the parameters
    parameters['incar']['npar'] = npar

    # set the parameters
    builder.parameters = orm.Dict(dict=parameters)

    # set dynamics
    builder.dynamics = dynamics

    # set the PAW potentials
    builder.potential_family = orm.Str('PBE.54')
    builder.potential_mapping = orm.Dict(dict={'Au':'Au', 'C':'C', 'O':'O'})

    # setup options
    options = {}
    options['resources'] = {'num_machines': num_proc}
    options['max_wallclock_seconds'] = 24 * 60 * 60
    builder.options = orm.Dict(dict=options)

    # settings dics
    settings = {'parser_settings': {}} 
    builder.settings = orm.Dict(dict=settings) 

    builder.relax.perform = orm.Bool(True)
    builder.relax.algo = orm.Str('cg')
    builder.relax.force_cutoff = orm.Float(2.5e-2)
    builder.relax.positions = orm.Bool(True)
    builder.relax.shape = orm.Bool(False)
    builder.relax.volume = orm.Bool(False)
    builder.relax.steps = orm.Int(500)

    calculation = submit( RelaxVasp, **builder)

    # store the calculation in the right group
    new_group_name = f'co_adsorption/{facet}/{repeat}/{new_group_base}'
    group = Group(label=new_group_name)
    try:
        group.store()
    except Exception:
        pass

    path = GroupPath()
    path[new_group_name].get_group().add_nodes(calculation)


if __name__ == '__main__':

    groupname = sys.argv[1]
    RelaxVasp = WorkflowFactory('vasp.relax')
    StructureData = DataFactory('structure')
    # Get the lattice node from the group that is requested
    
    qb = QueryBuilder()
    qb.append(Group, filters={'label':groupname}, tag='Group')
    # all lattice calculations are with RelaxVasp
    qb.append(RelaxVasp, with_group='Group', tag='calctype')

    new_group_base = groupname.split('/')[-1]

    # get the lattice node:
    for data in qb.all(flat=True):
        latticenode = data
        break

    testdir = 'testdir'
    Path(testdir).mkdir(parents=True, exist_ok=True)
    # get the parameters from the lattice calculations
    parameters = latticenode.inputs.parameters.get_dict()
    parameters['incar']['ldipol'] = True
    parameters['incar']['idipol'] = 3
    parameters['incar']['dipol'] = [0.5, 0.5, 0.5]
    
    structure = latticenode.outputs.relax.structure
    ase_structure = structure.get_ase()
    a = np.linalg.norm(ase_structure.cell[0])

    unit_cell = build.bulk( 'Au', 'fcc', a=a)
    kpoints_lattice = latticenode.inputs.kpoints.get_kpoints_mesh()[0]

    # special for 310
    # kpoints_lattice[0] = kpoints_lattice[0] / 3

    miller_index = (2, 1, 1)
    layers = 8
    facet = ''.join([str(elem) for elem in miller_index])

    # Factors for 310 calculation 
    # factors = [ 
    #             [1, 1],
    #             [1, 2],
    #             [1, 3],
    #             [1, 4],
    #             [1, 5],
    #         ]
    # Factors for 211 calculation
    factors = [
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
                [3, 5],
        ] 
        

    for factor in factors:
        # generate the new structure
        # For 310
        #atoms = build.surface(unit_cell, miller_index, layers=layers, vacuum=15.0, periodic=True)
        #atoms = atoms.repeat((factor[0], factor[1], 1))

        # For 211
        atoms = build.fcc211('Au',a=a, size=factor+[4], vacuum=15.0, orthogonal=True) 
        atoms.set_pbc([True, True, True])

        # decide which computer to use
        if len(atoms) < 20:
            num_proc = 2
            computer = 'juwels_scr'
        elif len(atoms) < 40:
            num_proc = 4
            computer = 'juwels_scr'
        elif len(atoms) < 60:
            num_proc = 4
            computer = 'juwels_scr'
        else:
            num_proc = 4
            computer = 'juwels_scr'

        repeat = 'x'.join([str(fac) for fac in factor])

        kpoint_mesh = [np.ceil(kpoints_lattice[0]/factor[0]), 
                       np.ceil(kpoints_lattice[1]/factor[1]), 1]

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
        runner_slab(atoms, parameters, kpoint_mesh, dynamics)

        # add CO to symmetrically inequivalent sites
        slab_structure = StructureData(ase=atoms)
        pymat_slab = slab_structure.get_pymatgen()
        co_molecule = ase.Atoms('OC', positions=[(0,0,0.493), (0,0,-0.657)])
        co_molecule.set_cell([1, 1, 1])
        co_molecule.center()
        mol_structure = StructureData(ase=co_molecule)
        pymat_molecule = mol_structure.get_pymatgen()
        symbols = atoms.get_chemical_symbols()
        cov_radii = [ase_data.covalent_radii[ase_data.atomic_numbers[symbol]] for symbol in symbols]
        height = 0.5 + np.max(cov_radii)

        asf = AdsorbateSiteFinder(pymat_slab)
        slab_with_ads = asf.generate_adsorption_structures(pymat_molecule, \
                                                        translate=False, \
                                                        reorient=False, \
                                                        find_args={'distance':height,},
                                                        repeat=(1,1,1),
                                                           )
        aaa = AseAtomsAdaptor()

        all_structures = []
        for index, ads_slab in enumerate(slab_with_ads):
            co_atoms = aaa.get_atoms(ads_slab)
            co_atoms.wrap()
            dynamics_pos.append([True, True, True])
            dynamics_pos.append([True, True, True])
            dynamics = {'positions_dof': orm.List(list=dynamics_pos)}
            co_atoms.write(f'testdir/co_{index}.cif')
            time.sleep(3)
            runner_CO(co_atoms, parameters, kpoint_mesh, dynamics)