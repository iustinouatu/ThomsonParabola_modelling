import numpy as np

class Species:
    """ Class used to represent 1 "particle". This is usually an ion, but can be anything as long as its name is in the list. 

    Attributes
    ----------
    _name : str
    _mass : float (in SI, units of Kg)
    _charge : float (in SI, units of Columb C)
    _qonm : float (in SI, units of Coulomb/Kg)
    x : float (where the particle is along the x-axis)
    y : float (where the particle is along the y-axis)
    z : float (where the particle is along the z-axis)
    ux : float (velocity of particle along x-axis)
    uy : flaot (velocity of particle along y-axis)
    uz : float (velocity of particle along z-axis)

    Methods
    -------
    Species_get_xyz():
        Returns a np array shape (3,) with current spatial coordinates of the particle. Position indexed 0 of that array is x-coord, indexed 1 is y-coord.
    
    Species_get_uxuyuz():
        Returns a np array shape (3,) with current velocities along x,y,z of the particle. Position indexed 0 of that array is x-velocity, ...
    
    Species_push_from_origin_to_endoffields():
        Performs integration in time of EOM's via adaptive step-size RK45 Felhberg method, given initial conditions for x,y,z, ux,uy,uz, and an initial step-size.
        Calls RK45integrator() from the Species class.
        Returns a np array shape (6,): [x,y,z, ux,uy,uz] , with conditions at the end of E/B fields region.

    @staticmethod
    Species_push_from_endoffields_to_detector(conds, z_det):
        Given initial conditions (conds), propagates (balistically) the particle from the end of E/B fields region to the detector screen (placed at z_det)
        Returns a np array size (2,) with the x,y coordinates of the particle when it reaches the detector screen. 
        arguments:
            conds: np.array of shape (6,) which represent the x,y,z,ux,uy,uz at end of RK45 integration
            z_det: float, represents where the detector (screen) is placed along z-axis, in SI (meters)
    """

    def __init__(self, name, mass, charge, r, velo):
        # r is a np array of shape (3,) , velo is a np array of shape (3,)
        self._name = name # for identification purposes
        self._mass = mass # underscore means the attribute is protected
        self._charge = charge
        self._qonm = charge / mass
        x, y, z = r # unpack the input argument to __init__()
        self._x = x
        self._y = y
        self._z = z
        #mean, sigma = distr # unpack distr (a list of 2 floats)
        #self._mean = mean
        #self._sigma = sigma
        # self._no_of_parts = uzs_at_t0.shape[0]
        #self._uzs_at_t0 = self.draw_from_Gaussian()
        ux, uy, uz = velo # velo is an input argument to the CTOR
        #self.ux = np.zeros( (self._no_of_parts, ) ) # at t0
        self._ux = ux
        self._uy = uy
        self._uz = uz
        #self.uy = np.zeros( (self._no_of_parts, ) ) # at t0
        #self.uz = uzs_at_t0 # at t0

    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, other_x):
        self._x = other_x

    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, other_y):
        self._y = other_y

    @property
    def z(self):
        return self._z
    @z.setter
    def z(self, other_z):
        self._z = other_z

    @property
    def ux(self):
        return self._ux
    @ux.setter
    def ux(self, other_ux):
        self._ux = other_ux

    @property
    def uy(self):
        return self._uy
    @uy.setter
    def uy(self, other_uy):
        self._uy = other_uy

    @property
    def uz(self):
        return self._uz
    @uz.setter
    def uz(self, other_uz):
        self._uz = other_uz



    def __str__(self):
        return "A {} species with mass={} , charge={}, no_of_parts={},".format(self._name, self._mass, self._charge, self._no_of_parts)
    def __repr__(self):
        return f'Species(name={self._name}, mass={self._mass}, charge={self._charge}, r=[{self._x, self._y, self._z}], distr=[{self._mean, self._sigma}], no_of_parts={self._no_of_parts})'
    
    def Species_get_xyz(self):
        return np.array([self._x, self._y, self._z]) # shape (3,)

    def Species_get_uxuyuz(self):
        return np.array([self._ux, self._uy, self._uz]) # shape (3,)


    @staticmethod # doesn't have self as an argument. only logically connected to this class, otherwise is unrelated to it in any way. 
    def Species_push_from_endoffields_to_detector(conds, z_det): # returns the x,y coords on the detection screen placed at z = z_det
        # conds is a np.array of shape (6,) which represent the x,y,z,ux,uy,uz at end of RK45 integration
        r_at_exit = conds[0:3] # get x,y,z coords at end of RK45 integration 
        us_at_exit = conds[3:6] # get ux, uy, uz at end of RK45 integration

        # find the drifting time drift_time in no E and B fields, E = B = 0. 
        # after how much time does the particle (now moving in free space) hits the detector?
        z_difference = z_det - r_at_exit[2] # z_det is where the detector is placed along z
        drift_time = z_difference / us_at_exit[2] # a float.  us_at_exit[2] is uz at exit, i.e. the velocity along z at exit

        # find the r-coords at the end of the drift time (so when z = z_end = where the detector is placed)
        r_at_end = r_at_exit + (us_at_exit * drift_time)
        return r_at_end[0:2] # x and y coords returned only


class Source:
    """A class which is an interface to a method which gives the initial particles' velocity distribution when they enter the aperture.

    Attributes
    ----------

    _name: str, just an identifier
    _mean: float, the mean (in SI units, m/s) of the Gaussian distribution from which velocities will be drawn. 
    _sigma: float, the sigma (in SI units, m/s) from the Gaussian distribution from which velocities will be drawn.
    _no_of_parts: float, how many draws from the above mentioned distribution will be drawn.

    Methods
    -------
    draw_from_Gaussian():
        Returns a np array shape (_no_of_parts, ) containing floats representing the initial velocities along z of the particles entering the aperture.
    """

    def __init__(self, name, distr, no_of_parts):
        # distr is a np.array of shape (2,): first entry is mean, second entry is sigma
        self._name = name
        mean, sigma = distr 
        self._mean = mean
        self._sigma = sigma
        self._no_of_parts = no_of_parts

    def __str__(self):
        pass
    def __repr__(self):
        pass

    def draw_from_Gaussian(self):
        uzs_at_t0 = np.random.normal(self._mean, self._sigma, self._no_of_parts) # returns shape (self._no_of_parts,)
        return uzs_at_t0 # a numpy array shape (self._no_of_parts, )

    @staticmethod
    def from_MeV_to_mpersec(energy_MeV):
        pass