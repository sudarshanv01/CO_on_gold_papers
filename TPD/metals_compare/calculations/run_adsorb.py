#!/usr/bin/python

""" Running the adsorb workchain """

from adsorb import Adsorb
from ase.lattice.hexagonal import Graphite
from aiida.engine import run, submit
from aiida.plugins import CalculationFactory, DataFactory
from aiida.orm import Code, Dict, Float, Str, StructureData, Bool, Int, Group, List
from aiida.engine import WorkChain, ToContext, calcfunction
from ase.io import read
import json
import numpy as np
import sys
from useful_functions import write_to_log


def get_calculation_builder(size, metal, adsorbate, pw_cutoff, dual):
    # To determine the needed cutoffs get the chemical symbols
    eV2Ry = 0.0734986176
    PwCalculation = CalculationFactory('quantumespresso.pw')
    builder = PwCalculation.get_builder()
    parameters = {
        'CONTROL': {
            'calculation': 'relax',
            'tstress': False,  # gets the stress
            'tprnfor': True,  # gets the forces
            'dipfield': False,
            #'forc_conv_thr':5e-2,
            #'etot_conv_thr':1e-1,
            'nstep': 500,
        },
        'SYSTEM': {
            'ecutwfc': pw_cutoff * eV2Ry,
            'ecutrho': pw_cutoff * dual * eV2Ry,
            'occupations': 'smearing',
            'smearing': 'fd',
            'degauss': 0.00734986475817,
            #'nspin': 1,
            'input_dft': 'RPBE',
            'nosym':True, # Dont use symmetry
            #'starting_magnetization(1)': 0.1,
            #'edir': 3,  # z direction
            #'emaxpos': 0.0,
            #'eopreg': 0.025,
        },
        'ELECTRONS': {
            'diagonalization': 'david',
            'conv_thr': 7.34986475817e-07,
            'mixing_beta': 0.3,
            'electron_maxstep': 200,
        }
        }
    builder.parameters = Dict(dict=parameters)
    builder.metadata.label = 'Optimizing structure for ' + metal + ' ' + adsorbate
    builder.metadata.description = "Relaxations"
    builder.metadata.options.resources = {'num_machines': 1}
    builder.metadata.options.max_wallclock_seconds = 55 * 60 * 60 # Autorestarts happen if time runs out

    return builder

def get_kpoints(size):
    kpoints = []
    for atoms_no in size[0:2]:
        kpt = 12 / atoms_no
        kpoints.append(kpt)
    kpoints.append(1)
    return kpoints

"""Inputs"""
# Metal atom to calculate
metal = 'Pd' # Only FCC structures
lattice_constants = {'Pt':3.99, 'Cu':3.62, 'Au':4.05, 'Pd':3.836} # TODO: Change this
adsorbates = ['CO', 'COOH'] # Currently CO and COOH supported
facet_list = ['211'] # Currently only 111 211 100 supported
sizes = [[3, 3, 4]] # x y z for all structures
pw_cutoff = 500 #eV
dual = 8
pseudo_family = 'SSSP'
codename = 'qe-6.1-pw@xeon8_scratch'

if __name__ == '__main__':
    GROUP_KEY = 'TM_CO2R'
    lattice_constant = lattice_constants[metal]
    group = Group.get(label=GROUP_KEY)
    for adsorbate in adsorbates:
        for facet in facet_list:
            for size in sizes:
                # Get the builder based on the size
                builder = get_calculation_builder(size=size,
                                                  metal=metal,
                                                  adsorbate=adsorbate,
                                                  pw_cutoff=pw_cutoff,
                                                  dual=dual)
                kpoint_mesh = get_kpoints(size)
                KpointsData = DataFactory('array.kpoints')
                kpoints = KpointsData()
                kpoints.set_kpoints_mesh(kpoint_mesh)
                node = submit(Adsorb,
                       **{ 'bulk_atoms':Str(metal),
                           'lattice_constant':Float(lattice_constant),
                           'adsorbate':Str(adsorbate),
                           'facet':Str(facet),
                           'size':List(list=size),
                           'relaxchain':{
                                    'clean_workdir':Bool(True),
                                    'final_scf':Bool(False),
                                    'relaxation_scheme':Str('relax'),
                                    'meta_convergence':Bool(False),
                                    'base':{
                                            'kpoints':kpoints,
                                            'pseudo_family':Str(pseudo_family),

                                        'pw':{
                                                'code':load_code(codename),
                                                'parameters':builder.parameters,
                                                'metadata':builder.metadata,
                                              },
                                        }
                                    }
                        }
                       )
                group.add_nodes(node)
                label_calc = 'Relaxation ' + metal + ' ' + facet +  ' ' + adsorbate
                node.label = label_calc
                write_to_log('Submitted ' + label_calc + ' with pk: ' + str(node.pk) )
