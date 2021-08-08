
# ------ regular imports
from ase.symbols import Symbols
from ase import Atoms
from aiida.tools.groups import GroupPath
import numpy as np
from aiida import orm
from aiida.engine import submit
from aiida.common import exceptions
# ------ specific imports
import sys
import time


BaseVasp = WorkflowFactory('vasp.vasp')
RelaxVasp = WorkflowFactory('vasp.relax')
VerifyVasp = WorkflowFactory('vasp.verify')

if __name__ == '__main__':
    """
    Check the status of all the calculations in a group
    """
    group_name = sys.argv[1] 

    qb = QueryBuilder()
    qb.append(Group, filters={'label':group_name}, tag='Group')
    qb.append(RelaxVasp, with_group='Group', tag='Screening')

    for node in qb.all(flat=True):
        if node.is_failed or node.is_killed:
            print('Node pk: %d failed'%node.pk)
            builder = node.get_builder_restart()

            try:
                print('Starting from old structure...')
                new_structure = node.called_descendants[0].outputs.relax.structure
            except Exception:
                print('Starting from input structure...')
                new_structure = node.inputs.structure
            builder.structure = new_structure

            options = {}
            options['resources'] = {'num_machines': 2}
            options['max_wallclock_seconds'] = 24 * 60 * 60
            builder.options = orm.Dict(dict=options)

            builder.code = load_code('vasp-5.4.4@dtu_xeon40')
            
            calculation = submit(builder)

            # ## add new calculation to group
            path = GroupPath()
            path[group_name].get_group().add_nodes(calculation)

            time.sleep(2)
            ## remove older calculation
            path[group_name].get_group().remove_nodes(node) 

            print('Removed %d from group %s and added %d'%(node.pk, group_name, calculation.pk))

            time.sleep(2)

        elif node.is_finished_ok:
            print('Node pk: %d completed'%node.pk)
        else:
            print('Node pk: %d still running'%node.pk)
        
     
        
