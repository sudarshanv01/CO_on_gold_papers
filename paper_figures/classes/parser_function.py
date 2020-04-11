#!/usr/bin/python


""" Functions for parsing and plotting data related to the Gold project """

from useful_classes import experimentalTPD
import numpy as np
from glob import glob
from useful_functions import AutoVivification, get_vibrational_energy
from pprint import pprint
import matplotlib.pyplot as plt
import os, sys
import mpmath as mp
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.io import read
from ase.db import connect
from matplotlib.ticker import FormatStrFormatter

def get_stable_site_vibrations():
    vibration_energies = {}
    vibration_energies['211'] = 0.00012 * np.array([2044.1, 282.2, 201.5, 188.5, 38.3, 11.5])
    vibration_energies['111'] = 0.00012 * np.array([2084.8, 201.7, 110.3, 110.2, 70.7, 70.8])
    vibration_energies['100'] = 0.00012 * np.array([1886.5, 315.2, 273.4, 222.2, 152.7, 49.8])
    vibration_energies['110'] = 0.00012 * np.array([2054.5, 262.9, 183.4, 147.3, 30.9, 30.])

    return vibration_energies

def get_gas_vibrations():
    atoms = read('../databases/co.traj')
    vibration_energies_gas = 0.00012 * np.array([2121.52, 39.91, 39.45])
    thermo_gas = IdealGasThermo(vibration_energies_gas, atoms = atoms, \
            geometry='linear', symmetrynumber=1, spin=0)

    return thermo_gas


def get_coverage_details():
    cell_sizes = {
                  '100': ['1x1', '2x2', '3x3'],
                  '211': ['1x3', '2x3', '3x3', '1CO_4x3'],
                  '111': ['1x1', '2x2', '3x3'],
                  '110': ['1x1', '2x2', '3x3'],
                  'recon_110': ['1x1', '2x1', '3x1'],
                  }
    coverages_cell = {
                  '100': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  '211': {'1x3':1, '2x3':1/2, '3x3':1/3, '1CO_4x3':1/4, \
                          '2CO_4x3':1/2, '3CO_4x3':3/4, '4CO_4x3':1},
                  '111': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  '110': {'1x1':1, '2x2':1/4, '3x3':1/9},
                  'recon_110':{'1x1':1, '2x1':1/2, '3x1':1/3},
                }
    coverage_labels = {
                  '100': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '211': {'1x3':'1', '2x3':r'$\frac{2}{3}$', '3x3':r'$\frac{1}{3}$', '1CO_4x3':r'$\frac{1}{4}$'},
                  '111': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  '110': {'1x1':'1', '2x2':r'$\frac{1}{4}$', '3x3':r'$\frac{1}{9}$'},
                  'recon_110':{'1x1':'1', '2x1':r'$\frac{1}{2}$', '3x1':r'$\frac{1}{3}$'},
                  }
    return cell_sizes, coverages_cell, coverage_labels

def get_lowest_absolute_energies(lst_stat_CO, lst_stat_slab, largest_cell, COg):
    for stat in lst_stat_CO:
        cell_size = stat.cell_size
        if cell_size == largest_cell:
            energy_CO = stat.energy
    for stat in lst_stat_slab:
        cell_size = stat.cell_size
        if cell_size == largest_cell:
            energy_slab = stat.energy

    return energy_CO - energy_slab - COg

def get_number_CO(atoms):
    nC = [ atom.index for atom in atoms if atom.symbol == 'C']
    return len(nC)

def get_differential_energy(absolute_energy, atoms_dict, facet, COg):
    # Takes a list of absolute energy and return the differential energy
    # get the different surface facets
    facet_factors = []
    factors = {}
    print('Getting differential energy for %s ...'%facet)
    # check if both x y need to be multipled
    regular = True if facet in ['100', '111', '110'] else False
    # Homework to determine what cell size we are dealing with
    for cell_size in absolute_energy:
        if '_' in cell_size:
            factor = cell_size.split('_')[-1].split('x')[0]
        else:
            factor = cell_size.split('x')[0]
        facet_factors.append(int(factor))
        factors[cell_size] = factor
        # Initialize sorted_data
        sorted_data = {}
        for state in absolute_energy[cell_size]:
            sorted_data[state] = [[], []]


    # Find the lowest common multiple of the cell size factors
    lcm = np.lcm.reduce(facet_factors)

    # sorted_data = {}
    for cell_size in absolute_energy:
        factor = factors[cell_size]
        multiply_cell_with = float(lcm) / float(factor)
        assert multiply_cell_with.is_integer(), 'multiplication factor has to be int'
        multiply_cell_with = int(multiply_cell_with)
        print('Factor for multiplication: %1.0f for cell %s'%(multiply_cell_with,cell_size))
        for state in absolute_energy[cell_size]:
            atoms = atoms_dict[cell_size][state]
            if regular:
                norm_atoms = atoms.repeat([multiply_cell_with, multiply_cell_with, 1])
                norm_energy = absolute_energy[cell_size][state] * multiply_cell_with**2
            else:
                norm_atoms = atoms.repeat([multiply_cell_with, 1, 1])
                norm_energy = absolute_energy[cell_size][state] * multiply_cell_with

            num_CO = get_number_CO(norm_atoms)
            sorted_data[state][0].append(num_CO)
            sorted_data[state][1].append(norm_energy)

    avg_energy = {}
    slab_energy = {}
    for state in sorted_data:
        if 'slab' in state:
            numCO, energies = sorted_data[state]
            energies = np.array(energies)
            slab_energy[state] = energies[np.argsort(-1 * energies)]
    for state in sorted_data:
        if 'slab' not in state:
            numCO, energies = sorted_data[state]
            numCO = np.array(numCO)
            energies = np.array(energies)
            sorted_nCO = np.sort(numCO)
            sorted_energy = energies[np.argsort(numCO)]
            avg_energy[state] = np.empty(len(sorted_energy))
            for i in range(len(sorted_energy)):
                if i == 0: # First index keep the same
                    avg_energy[state][i] =( sorted_energy[i] \
                                         -  slab_energy['state_slab'][i] \
                                         -  sorted_nCO[i] * COg )\
                                         /  sorted_nCO[i]
                else:
                    nCO_diff = sorted_nCO[i] - sorted_nCO[i-1]
                    avg_energy[state][i] = (  sorted_energy[i] - sorted_energy[i-1]\
                                           - nCO_diff * COg )/ nCO_diff

    return avg_energy




def diff_energies(lst_stat, mult_factor, COg, lowE_CO_abs, facet):
    # take a list of database rows and convert it to differential energies
    # and coverages
    cell_lst = [ stat.cell_size for stat in lst_stat ]
    # get mulp factor based on the cell size
    mult_lst = []
    for cell in cell_lst:
        mult_lst.append(mult_factor[cell])
    if any('_' in x for x in cell_lst):
        mult_lst = [int(mult.split('CO')[0]) for mult in cell_lst]
    sorted_lst = np.array(lst_stat)[np.argsort(mult_lst)]
    sorted_mult = np.sort(mult_lst)

    natoms_lst = []
    abs_ener_lst = []
    cell_sizes_sorted = []
    for i in range(len(sorted_lst)):
        atoms = sorted_lst[i].toatoms()
        cell_sizes_sorted.append(sorted_lst[i].cell_size)
        energies = sorted_lst[i].energy
        if facet in ['211', 'recon_110']:
            natoms = atoms.repeat([sorted_mult[i], 1, 1])
        else:
            natoms = atoms.repeat([sorted_mult[i], sorted_mult[i], 1])
        natoms_lst.append(natoms)
        abs_ener_lst.append(energies)
    # get the number of CO in a large cell of the same size
    abs_ener_lst = np.array(abs_ener_lst)
    nCO_bigcell = []
    for atom in natoms_lst:
        nCO = get_number_CO(atom)
        nCO_bigcell.append(nCO)
    plot_nCO = nCO_bigcell
    plot_diff_energies = []
    # Start with the energy corresponding to the lowest coverage
    plot_diff_energies.append(lowE_CO_abs)
    nCO_bigcell = np.array(nCO_bigcell)
    # Going from least coverage to most coverage
    print(nCO_bigcell, abs_ener_lst)
    for i in range(len(nCO_bigcell)-1):
        nCOdiff = np.array(nCO_bigcell[i+1]) - np.array(nCO_bigcell[i])
        diff_energies = ( nCO_bigcell[i+1] * abs_ener_lst[i+1] - \
                nCO_bigcell[i] * abs_ener_lst[i] - \
                nCOdiff * COg ) / nCO_bigcell[i+1]
        plot_diff_energies.append(diff_energies)
    print([cell_sizes_sorted, plot_diff_energies])
    return [cell_sizes_sorted, plot_diff_energies]
