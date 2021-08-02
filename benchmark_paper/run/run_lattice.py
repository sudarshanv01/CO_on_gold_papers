
from aiida import orm
from aiida.engine import run, submit
from pprint import pprint
import numpy as np
import json
from aiida.tools.groups import GroupPath
from ase import Atoms
from ase import build

"""
Relax the cell for different functionals 
"""

def calculator():
    param_dict = {
        'incar':{
            'encut': 500,
            'ismear': 0,
            'sigma': 0.2,
            'ibrion': 2,
            'ispin': 1,
            'lorbit': 11,
            'nelm': 100,
            'prec': 'Accurate',
            'ediffg': -0.02,
            'ediff': 1e-7,
            'isif': 3,
            'nsw': 100,
            'gga': 'RP',
        }

    }

    return param_dict

def runner(structure):

    # use the base VASP workchain
    BaseVasp = WorkflowFactory('vasp.vasp')
    RelaxVasp = WorkflowFactory('vasp.relax')
    VerifyVasp = WorkflowFactory('vasp.verify')

    # generate the inputs
    builder = VerifyVasp.get_builder()

    builder.metadata.label = 'Lattice Relaxation Calculation'
    builder.metadata.description = 'Relaxing for generating vacancies'

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
    kpoints.set_kpoints_mesh([12, 12, 12])
    builder.kpoints = kpoints

    # set the parameters
    parameters = calculator() 
    builder.parameters = Dict(dict=parameters)

    # set the PAW potentials
    builder.potential_family = orm.Str('PBE.54')
    builder.potential_mapping = orm.Dict(dict={'Au':'Au'})

    # setup options
    options = {}
    options['resources'] = {'num_machines': 1}
    options['max_wallclock_seconds'] = 30 * 60
    builder.options = orm.Dict(dict=options)

    # settings dics
    # parser_settings = {'output_params': ['total_energies', 'maximum_force']}
    parser_settings = {'add_misc':True, 'add_structure':True} 
    builder.settings = orm.Dict(dict=parser_settings)


    calculation = submit( VerifyVasp, **builder)
    path = GroupPath()
    path["lattices/RPBE"].get_group().add_nodes(calculation)


if __name__ == '__main__':

    structure = build.bulk('Au', 'fcc', a=4.05, cubic=True)
    runner(structure)
