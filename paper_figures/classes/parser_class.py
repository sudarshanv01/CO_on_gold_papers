  #!/usr/bin/python

import numpy as np
from useful_functions import AutoVivification
from pprint import pprint
from ase.db import connect
import matplotlib.pyplot as plt
import os
import csv

class ParseInfo:
    def __init__(self, thermodb, referdb, list_cells, facet, functional, \
                ref_type):
        # Takes in the databases and the list of cells
        # for that facet
        self.thermodb = thermodb
        self.referdb = referdb
        self.list_cells = list_cells
        self.results = AutoVivification()
        self.rel_energy = AutoVivification()
        self.absolute_energy = AutoVivification()
        self.functional = functional
        self.states = AutoVivification()
        self.min_energy_cell = {}
        self.facet = facet
        self.ref_type = ref_type
        self.atoms = AutoVivification()


    def _get_states(self, cell):
        # for a given database pick out all states
        allstates = []

        for row in self.thermodb.select(functional=self.functional, cell_size=cell,\
                                        facets='facet_'+self.facet):
            allstates.append(row.states)
        unique_states = np.unique(allstates)
        return unique_states

    def _parse_energies(self):
        # Gets all energies from the database
        # This is a func specific to this
        for cell in self.list_cells:
            self.states[cell] = self._get_states(cell)
            for state in self.states[cell]:
                stat = self.thermodb.get(states=state,
                                    functional=self.functional,
                                    cell_size=cell,
                                    facets='facet_'+self.facet,
                                    )
                self.results[cell][state] = stat.energy
                self.atoms[cell][state] = stat.toatoms()

    def _get_reference(self):
        if self.ref_type == 'Pb':
            Pbstat = self.referdb.get(formula='Pb4',
                              functional=self.functional,
                              pw_cutoff=500.0)
        # Four atoms in Pb unit cell
            Pb_energies = Pbstat.energy / 4
            self.energy_reference = Pb_energies
        elif self.ref_type == 'CO':
            COstat = self.referdb.get(formula='CO',
                                      functional=self.functional,
                                      pw_cutoff=500.0)
            self.energy_reference = COstat.energy


    def get_pourbaix(self):
        # Get the DFT energies
        self._parse_energies()
        # Get the reference energies
        self._get_reference()
        for index, cell in enumerate(self.list_cells):
            energy_cell = []
            for state in self.states[cell]:
                self.absolute_energy[cell][state] = self.results[cell][state]
                if self.ref_type == 'CO':
                    if state != 'state_slab':
                        self.rel_energy[cell][state] = self.results[cell][state] - \
                                            self.results[cell]['state_slab'] - \
                                            self.energy_reference
                        energy_cell.append(self.rel_energy[cell][state])
                elif self.ref_type == 'Pb':
                    if state != 'slab':
                        self.rel_energy[cell][state] = self.results[cell][state] - \
                                            self.results[cell]['slab'] - \
                                            self.energy_reference
                        energy_cell.append(self.rel_energy[cell][state])
            energy_cell = np.array(energy_cell)
            self.min_energy_cell[cell] = min(energy_cell)
