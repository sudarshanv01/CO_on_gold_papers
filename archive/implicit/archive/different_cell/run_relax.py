import numpy as np
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, Bool, Str, Float
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import submit, run
from aiida import load_profile
import sys
load_profile()


def main(code_string, potential_family, resources, structure):
    Dict = DataFactory('dict')

    # set the WorkChain you would like to call
    workflow = WorkflowFactory('vasp.vasp')

    # organize options (needs a bit of special care)
    options = AttributeDict()
    options.account = ''
    options.qos = ''
    options.resources = resources
    options.queue_name = ''
    options.max_wallclock_seconds = 24 * 60 * 60

    # organize settings
    settings = AttributeDict()
    # the workchains should configure the required parser settings on the fly
    parser_settings = {'output_params': ['total_energies', 'maximum_force']}
    settings.parser_settings = parser_settings

    # set inputs for the following WorkChain execution

    inputs = AttributeDict()
    # set code
    inputs.code = Code.get_from_string(code_string)
    # set structure
    inputs.structure = structure  # get_structure_Si()
    # set parameters
    inputs.parameters = Dict(dict={
        # DFT parameters
        'encut': 500,
        'ismear': 0,
        'sigma': 0.1,
        'system': 'CO on Gold',
        # Implicit solvation parameters
        # 'tau': 0,
        # 'lambda_d_k': 3,
        # 'lsol': True,
    })
    # set k-point grid density
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([4, 4, 1])
    inputs.kpoints = kpoints
    # set potentials and their mapping
    inputs.potential_family = Str(potential_family)
    inputs.potential_mapping = Dict(dict={'Au': 'Au', 'C': 'C', 'O': 'O'})
    # set options
    inputs.options = Dict(dict=options)
    # set settings
    inputs.settings = Dict(dict=settings)
    # set workchain related inputs
    # turn relaxation on
    #inputs.relax = Bool(False)
    # inputs.force_cutoff = Float(0.025)
    # inputs.convergence_on = Bool(False)
    # inputs.convergence_positions = Float(0.1)
    # inputs.relax_parameters = Dict(dict={  # 'ediffg': -0.0001,
    #    'ibrion': 2,
    # 'nsw': 500,
    # })
    inputs.verbose = Bool(False)
    # submit the requested workchain with the supplied inputs
    submit(workflow, **inputs)


if __name__ == '__main__':
    # code_string is chosen among the list given by 'verdi code list'
    code_string = 'vasp_std@xeon8'
    prev_node = load_node(sys.argv[1])
    structure = prev_node.inputs.structure
    # potential_family is chosen among the list given by
    # 'verdi data vasp-potcar listfamilies'
    potential_family = 'PBE.54'

    # metadata.options.resources
    # See https://aiida.readthedocs.io/projects/aiida-core/en/latest/scheduler/index.html
    resources = {'num_machines': 1}
    # resources = {'parallel_env': 'mpi*', 'tot_num_mpiprocs': 12}

    main(code_string, potential_family, resources, structure)
