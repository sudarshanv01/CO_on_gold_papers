
from aiida import orm
from aiida.engine import run, submit
from pprint import pprint
import numpy as np
import json
from aiida.tools.groups import GroupPath
from ase import Atoms
from ase import build

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

    lnode = load_node()

    lparameters = lnode.inputs.parameters
    # relax only cell positions
    lparameters['isif'] = 2
    # create the dipole moment
    lparameters['ldipole'] = True
    lparameters['dipol'] = [0.5, 0.5, 0.5]
    lparameters['idipol'] = 3

    factors = [2, 3, 4, 5, 6]
