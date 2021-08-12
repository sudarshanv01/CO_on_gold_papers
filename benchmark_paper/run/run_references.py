
from aiida import orm
from aiida.engine import run, submit
from pprint import pprint
import numpy as np
import json
from aiida.tools.groups import GroupPath
from ase import Atoms
from ase import build
from aiida.common.extendeddicts import AttributeDict


"""
Relax the cell for different functionals 
"""

def calculator():
    param_dict = {
        'incar':{
            'encut': 500,
            'ismear': 0,
            'sigma': 1e-4,
            'ibrion': 2,
            'ispin': 1,
            'lorbit': 11,
            'nelm': 250,
            'prec': 'Accurate',
            'ivdw': 11,
            'ediff': 1e-7,
            'gga': 'PE',
            'ldipol': True,
            'idipol': 4,
            'dipol':[0.5, 0.5, 0.5],   
        }
    }

    return param_dict

def runner(structure):

    # use the base VASP workchain
    BaseVasp = WorkflowFactory('vasp.vasp')
    RelaxVasp = WorkflowFactory('vasp.relax')
    VerifyVasp = WorkflowFactory('vasp.verify')

    # generate the inputs
    builder = RelaxVasp.get_builder()

    builder.metadata.label = 'Lattice Relaxation Calculation'
    builder.metadata.description = 'Relaxing for generating vacancies'

    builder.verbose = orm.Bool(True)

    # set the code
    code = load_code('vasp-5.4.4@dtu_xeon16')
    builder.code = code

    # set the structure
    StructureData = DataFactory('structure')
    structure = StructureData(ase=structure)
    builder.structure = structure

    # k-points
    KpointsData = DataFactory('array.kpoints')
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([1, 1, 1])
    builder.kpoints = kpoints

    # set the parameters
    parameters = calculator() 
    builder.parameters = orm.Dict(dict=parameters)

    # set the PAW potentials
    builder.potential_family = orm.Str('PBE.54')
    builder.potential_mapping = orm.Dict(dict={'C':'C', 'O':'O'})

    # setup options
    options = {}
    options['resources'] = {'num_machines': 1}
    options['max_wallclock_seconds'] = 30 * 60
    builder.options = orm.Dict(dict=options)

    # settings dics
    # parser_settings = {'output_params': ['total_energies', 'maximum_force']}
    settings = {'parser_settings': {}} 
    builder.settings = orm.Dict(dict=settings) 

    builder.relax.perform = orm.Bool(True)
    builder.relax.algo = orm.Str('cg')
    builder.relax.force_cutoff = orm.Float(1e-2)
    builder.relax.positions = orm.Bool(True)
    builder.relax.shape = orm.Bool(False)
    builder.relax.volume = orm.Bool(False)
    builder.relax.steps = orm.Int(100)

    calculation = submit( RelaxVasp, **builder)
    path = GroupPath()
    path["references/PBE-D3"].get_group().add_nodes(calculation)


if __name__ == '__main__':

    structure = build.molecule('CO')
    structure.set_cell([10,10,10])
    structure.center()
    runner(structure)
