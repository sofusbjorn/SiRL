from articulated_system import ArticulatedSystem, Revolute
from spatial_algebra import Transform, Inertia
import numpy as np


def inertia_from_geom(shape, size, density=1000):
    if shape == 'box':
        mass = size[0] * size[1] * size[2] * density
        it = np.array(
            [
                [size[1]**2+size[2]**2, 0, 0],
                [0, size[0]**2+size[2]**2, 0],
                [0, 0, size[0]**2+size[1]**2]
            ]
        ) * mass / 12.
    elif shape == 'cylinder':
        return NotImplementedError()
    elif shape == 'sphere':
        return NotImplementedError()
    else:
        return NotImplementedError()
    return mass, it

class ThreeLinks(ArticulatedSystem):
    def __init__(self, link_len=0.3, link_den=2700) -> None:
        super().__init__(num_links=3)
        link_length = link_len
        link_width = 0.05
        m, it = inertia_from_geom('box', [link_width, link_length, link_width], link_den)
        #specify link parameters
        #link 0
        self.links[0].joint_frame = Transform(trans=np.array([0, -link_length/2., 0]))
        self.links[0].joint_dofs = Revolute()   #1 DOF revolute joint along z axis
        self.links[0].inertia = Inertia.from_it_and_mass(it=it, mass=m)

        self.links[0].viz_shape['size'] = [link_width, link_length, link_width]

        #link 1
        self.links[1].prev_transform = Transform(trans=np.array([0, link_length/2., 0]))
        self.links[1].joint_frame = Transform(trans=np.array([0, -link_length/2., 0]))
        self.links[1].joint_dofs = Revolute()   #1 DOF revolute joint along z axis
        self.links[1].inertia = Inertia.from_it_and_mass(it=it, mass=m)

        self.links[1].viz_shape['size'] = [link_width, link_length, link_width]

        #link 2
        self.links[2].prev_transform = Transform(trans=np.array([0, link_length/2., 0]))
        self.links[2].joint_frame = Transform(trans=np.array([0, -link_length/2., 0]))
        self.links[2].joint_dofs = Revolute()   #1 DOF revolute joint along z axis
        self.links[2].inertia = Inertia.from_it_and_mass(it=it, mass=m)

        self.links[2].viz_shape['size'] = [link_width, link_length, link_width]
