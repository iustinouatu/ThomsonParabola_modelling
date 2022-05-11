""" This file contains classes definitions for a few design aspects of the detector geometry.

The Fields Class which has as attributes (apart from the name) 
the strength of the field, 
the length over which it's active, 
and the drifting length (in free space) after the field finished to be non-zero.

The Electrode class is helpful to check if clipping occurs during the particles flight in the detector.
Has as an attribute the y-coordinate of the bottom electrode (in a geometry where the E-field is along positive y-direction)

The Detector_Screen class is there to simulate a passive detector on which particles are collected.
It has as an attribute the z-coordinate (measured from the source (z=0)) of the detector screen object.

The create_Geometry_Objects() function helps translating user-input to actual geometrical objects and fields in the simulation.
"""

class Field:
    def __init__(self, name, strength, l, D):
        self._name = name # protected attribute (proceeded by _).
        self._strength = strength # a private attribute would be preceeded by __ , and can only be accessed from within the Field class. or by using _object.__name = new_name_for_Field, but no-one does this
        self._l = l
        self._D = D 

    def __str__(self):
        return "{} is a {}-field with strength={} , l_{}={} , D_{}={} , all in SI units".format(self, self.name, self.strength, self.name, self.l, self.name, self.D)
    def __repr__(self):
        return f'Field(name={self._name}, strength={self._strength}, l={self._l}, D={self._D})'
  
class Electrode:
    def __init__(self, name, y_electrode):
        self._name = name
        self._y_electrode = y_electrode
    def __str__(self):
        pass
    def __repr__(self):
        pass

class Detector_Screen:
    def __init__(self, name, z_det):
        self._name = name
        self._z_det = z_det # where is the detector placed (SI units)

    def __str__(self):
        return f'This Detector Screen object is a passive {self._name} detector placed at z = {self._z_det} meters.'
    def __repr__(self):
        return f'Detector_Screen(z_det={self._z_det})'

def create_Geometry_Objects(E, l_E, D_E,   B, l_B, D_B,  z_det,   y_electrode_bottom):
    Efieldobj = Field("E-field", E, l_E, D_E)
    Bfieldobj = Field("B-field", B, l_B, D_B)
    detector_obj = Detector_Screen("detector", z_det)
    electrode_bottom_obj = Electrode("bottom_electrode", y_electrode_bottom)
    return Efieldobj, Bfieldobj, detector_obj, electrode_bottom_obj