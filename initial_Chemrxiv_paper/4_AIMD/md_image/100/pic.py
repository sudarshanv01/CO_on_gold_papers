from ase.io import *
from ase_povray import povray_parameter
import numpy as np
from ase.visualize import view


name=['atoms','atoms_nowater']

for i in name:
    B=read(i+'.traj')
    B.positions[:,0]+=0.5
    B.positions[:,0]+=B.cell[0,0]/3.
    
    
    B.set_scaled_positions(B.get_scaled_positions())
    if i == 'atoms':
        B[86].symbol='N'
    elif i == 'atoms_nowater':
        B[36].symbol = 'N'
    #B[57].symbol='C'

    #B[79].position[0]+=-B.cell[0,0]
    #B[93].position[0]+=-B.cell[0,0]
    #B[64].position[0]+=B.cell[0,0]
    A=B.repeat((3,1,1))

    PovRay=povray_parameter(A,atoms_radii={'Au':1.5,'O':0.15,'H':0.15,'Li':0.8,'He':0.5,'B':0.5,'N':0.8,'C':0.9,'K':0.7})

    PovRay.set_colors({'K':(1,0.,1),
                               'N':(0.9,0.05,0.05),
                               'C':(0.1,0.,0),
                               'O':(1,0.05,0.05),
                               'H':(0.000,1,1),
                               'B':(0.000,1,1),
                               'Li':(0.000,0.0,1.0),
                               'He':(1,1,0),
                               'Au':(1,0.84,0)})
    PovRay.set_textures({'K':'glass',
                               'O':'glass',
                               'N':'glass',
                               'He':'glass',
                               'B':'glass',
                               'H':'glass',
                               'C':'glass',
                               'Li':'glass',
                               'Au':'glass'})


    PovRay.kwargs['rotation']='-90x,0y,-90z'
    PovRay.kwargs['canvas_width']=500
    PovRay.kwargs['transparent']=False

    PovRay.kwargs['bbox']=(5, -20, 25, -10)

    at=[]
    at_r=[]
    bond=[]
    O=np.where(A.get_atomic_numbers()==8)[0]
    H=np.where(A.get_atomic_numbers()==1)[0]

    for o in O:
        for h in H:
            if A.get_distance(o,h)<1.3:
                bond.append([o,h])

    for b in bond:
        if len(b)>1:
            PovRay.kwargs['bondatoms'].append([b[0],b[1]])

    PovRay.kwargs['bondlinewidth']=0.15

    PovRay.kwargs['area_light'] = [(5., 5., 40.),'White',10, 10, 8, 8]
    PovRay.kwargs['camera_type']='orthographic'
    write(i+'.pov',A,**PovRay.kwargs)

