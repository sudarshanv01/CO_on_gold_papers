#!/usr/bin/python

from aiida.orm import Code
from aiida.plugins import CalculationFactory
from aiida.engine import run, run_get_pk
import sys

folder = sys.argv[1]
facet = folder.split('/')[0]
state = folder.split('/')[1]
group = Group.get(label='gold_CO_vacuum')
basedir = "/home/cat/vijays/project/2_gold_electroxidation/dft_Au/cell_size/aiida_pickup/"
# first, get the code which the calculation will use
code = Code.get_from_string('vasp_std@xeon8')  # use the name of your code and computer
label = facet + ' ' + state
# then comes the information which can not be read from input files
resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 8}  # whatever is appropriate for you

parser_settings = {'output_params': ['total_energies', 'maximum_force']}
settings_dict = {'parser_settings': parser_settings}

process, inputs = CalculationFactory('vasp.vasp').immigrant(
    code, basedir + folder, resources=resources, settings=settings_dict,
    label=label)
node, pk = run_get_pk(process, **inputs)
calc = load_node(pk)
calc.label = label
# group.add_nodes(calc)
