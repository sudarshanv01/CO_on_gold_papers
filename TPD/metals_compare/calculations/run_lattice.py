#!/usr/bin.python

from ase.lattice.hexagonal import Graphite
from aiida.engine import run, submit
from aiida.plugins import CalculationFactory
from aiida.orm import Code, Dict, Float, Str, StructureData, Int, Group
from aiida.engine import WorkChain, ToContext, calcfunction
from find_lattice import Find_lattice
from ase.io import read
from useful_functions import write_to_log
from ase.build import bulk
eV2Ry = 0.0734986176


if __name__ == '__main__':
    print('Starting workflow to find Lattice constant')
    bulk_structure = bulk('Ag', cubic=True)
    group = Group.get(label='TM_CO2R')

    node = submit(Find_lattice,
        code=load_code('qe-6.1-pw@xeon8_special'),
        pseudo_family=Str('SSSP'),
        functional=Str('RPBE'),
        pw_cutoff=Float(500 * eV2Ry),
        dual=Int(8),
        input_structure=StructureData(ase=bulk_structure),
        )
    label_calc = 'Cu Lattice Constant'
    node.label = label_calc
    group.add_nodes(node)
    write_to_log('Running lattice optimization with pk:' +  str(node.pk))
