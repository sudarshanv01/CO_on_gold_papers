
# ------ regular imports
from ase.symbols import Symbols
from ase import Atoms
from aiida.tools.groups import GroupPath
import numpy as np
from aiida import orm
from aiida.engine import submit
# ------ specific imports
import sys
import time

# ------ Common workflows to pass to Query builder

BaseVasp = WorkflowFactory('vasp.vasp')
RelaxVasp = WorkflowFactory('vasp.relax')
VerifyVasp = WorkflowFactory('vasp.verify')

if __name__ == '__main__':
    """
    Check the status of all the calculations in a group
    """
    #group_name = sys.argv[1] 
    groupnames = []
    qb = QueryBuilder()
    qb.append(Group, tag='Group')

    for group in qb.all(flat=True):
        groupnames.append(group.label)

    print(groupnames)
    failed_groups = []
    for group_name in groupnames:
        print(f'Group:{group_name}')

        qb = QueryBuilder()
        qb.append(Group, filters={'label':group_name}, tag='Group')
        qb.append(RelaxVasp, with_group='Group', tag='Screening')

        for node in qb.all(flat=True):
            if node.is_failed or node.is_killed or node.is_excepted:
                print('Node pk: %d failed'%node.pk)
                failed_groups.append(group_name)
            elif node.is_finished_ok:
                print('Node pk: %d completed'%node.pk)
            else:
                print('Node pk: %d still running'%node.pk)

    failed_groups = np.unique(failed_groups)
    print(failed_groups)     
        
