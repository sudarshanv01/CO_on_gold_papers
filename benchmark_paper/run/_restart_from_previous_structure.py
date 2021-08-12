
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
        if node.is_failed or node.is_killed or node.is_excepted:
            print('Node pk: %d failed'%node.pk)
            builder = node.get_builder_restart()

            descendents = node.called_descendants
            new_structure = None
            for i in range(len(descendents)):
                try:
                    new_structure = descendents[i].outputs.structure
                    print('Starting from old output structure')
                    break
                except Exception:
                    pass
            if not new_structure:
                new_structure = node.inputs.structure 
                print('Starting from input structure')

            builder.structure = new_structure
            
            # make parallelisation settings
            kpts = builder.kpoints.attributes['mesh']
            total_k = kpts[0] * kpts[1]
            if total_k % 4 == 0:
                npar = 4
            elif total_k % 3 == 0:
                npar = 3
            elif total_k % 2 == 0:
                npar = 2

            # Alter the parameters
            parameters = node.inputs.parameters.get_dict()
            parameters['incar']['npar'] = npar
            builder.parameters = orm.Dict(dict=parameters)

            ase_structure = new_structure.get_ase()
            if len(ase_structure) >= 50:
                num_machines = 4
            elif 50 > len(ase_structure) :
                num_machines = 2

            options = {}
            options['resources'] = {'num_machines': num_machines}
            options['max_wallclock_seconds'] = 24 * 60 * 60
            builder.options = orm.Dict(dict=options)

            builder.code = load_code('vasp-5.4.4@juwels')
            
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
        
     
        
