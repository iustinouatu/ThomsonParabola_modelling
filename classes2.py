import numpy as np
from scipy.constants import elementary_charge as e_charge, m_p, c
from matplotlib import pyplot as plt
import cProfile, pstats, io
from pathlib import Path

def from_KEineV_to_uzinit(KE): # utility function
    KE_in_J = from_KEineV_to_KEinJ(KE)
    v_total = ( c / ( (KE_in_J/(m_p*(c**2))) + 1) )  *  np.sqrt( ((KE_in_J/(m_p*(c**2))) + 1)**2 - 1 )
    return v_total # a float

def from_KEineV_to_KEinJ(KE): # utility function
    KE_in_J = (1.60217662 * 10**(-19)) * KE # KE in eV
    return KE_in_J

# class Integrator(Species):
#     def __init__(self, E, B, l_B, y_bottom_electrode, z_end):
#         self._E = E
#         self._B = B
#         self._l_B = l_B
#         self._y_bottom_electrode = y_bottom_electrode
#         self._z_end = z_end
    # can we access qonm in this class, below??? does it know about _qonm from Species class? probably yes
def derivatives(t, vec, qonm): # vec is a numpy array: [x,y,z, u_x, u_y, u_z]. does it actually have to be a numpy array or can it be a simple list?
    gamma = np.sqrt(  (1 + (np.sum(vec[3:]**2.0) / (c**2.0)) )  ) # gamma = sqrt ( 1 + (u^2/c^2) ). u^2 = u_x^2 + u_y^2 + u_z^2. u_x is vec[3], u_y is vec[4], u_z is vec[5]
    # could do vec = np.array(vec) , if vec from arguments is a plain list
    # then do gamma = np.sqrt(  (1 + np.sum(vec[3:]**2.0) / (c**2.0) )  ).  # but I don't see the advantage of this, we are still creating a numpy array!
    # incident velocity along +z, E and B along +y. Cartesian coordinate system
    # E and B are global variables, can be used below
    dxondt = vec[3] / gamma # vec[3] is u_x
    dyondt = vec[4] / gamma # vec[4] is u_y
    dzondt = vec[5] / gamma # vec[5] is u_z
    duxondt = (-qonm * B / gamma) * vec[-1]  # vec[-1] is last elemenet of vec, i.e. vec[5], and is u_z
    duyondt = qonm * E
    duzondt = (qonm * B / gamma) * vec[3] # vec[3] is u_x
    return np.array([dxondt, dyondt, dzondt, duxondt, duyondt, duzondt]) # do you actually need to return a numpy array? yes, otherwise error at K2=derivatives(t+dt/4, vec + (dt/4)*K1)

RKF4_container = [0.0, 0.0, 0.0,    0.0, 0.0, 0.0]
def get_RKF4_approx(t, vec, dt, qonm): # vec has 6 elements: [x,y,z, ux, uy, uz]
    K1 = derivatives(t, vec, qonm) # K1 will have 6 elements 
    K2 = derivatives(t+dt/4.,          vec + (dt/4.)*K1     , qonm                ) # K2 will have 6 elements
    K3 = derivatives(t + dt*(3./8.) ,  vec + dt*( (3./32.)*K1 + (9./32.)*K2), qonm ) # K3 will have 6 elements
    K4 = derivatives(t + dt*(12./13.), vec + dt*( (1932./2197.)*K1 - (7200./2197.)*K2 + (7296./2197.)*K3 ) , qonm  ) # K4 will have 6 elements
    K5 = derivatives(t + dt      ,     vec + dt*( (439./216.)*K1 - 8.*K2 + (3680./513.)*K3 - (845./4104)*K4 ) , qonm   ) # K5 will have 6 elements
    RKF4 = vec + dt * ( (25./216)*K1 + (1408/2565.)*K3 + (2197./4104.)*K4 - (1./5.)*K5  ) # RKF4 will have 6 elements
    RKF4_container[0] = RKF4 # does this work? RKF4_container is a list of floats, RKF4 is a numpy array. tried in interactive mode, yes.
    RKF4_container[1] = K1
    RKF4_container[2] = K2
    RKF4_container[3] = K3
    RKF4_container[4] = K4
    RKF4_container[5] = K5
    return RKF4_container # a list containing 6 numpy arrays, each array containing 6 floats.

def get_RKF5_approx_efficiently(t, vec, dt, Ks, qonm):
    # Ks is a numpy array
    # Ks[0] is K1, Ks[1] is K2, ... , Ks[4] is K5
    # each Kj has 6 components, so can be indexed as Kj[0...5] or as Ks[j][0...5] where j=0,1,2,3,4 (5 values)
    # vec has 6 elements

    K6 = derivatives(t + dt/2. , vec +  dt*( -(8./27.)*Ks[0] + 2.*Ks[1] - (3544./2565.)*Ks[2] + (1859./4104.)*Ks[3] - (11./40.)*Ks[4] ), qonm ) # K6 will have 6 elements
    # Ks[0...5] all have 6 elements in them
    # so RKF6 below will have 6 elements in it.
    RKF5 = vec + dt * (  (16./135.)*Ks[0] + (6656./12825.)*Ks[2] + (28561./56430.)*Ks[3] - (9./50.)*Ks[4]  +(2./55.)*K6  )
    return RKF5 # RKF5 will contain 6 floats ''in it''

class Species:
    def __init__(self, name, mass, charge, r, velo):
        # r is a np array of shape (3,) , velo is a np array of shape (3,)
        self._name = name # for identification purposes
        self._mass = mass # underscore means the attribute is protected
        self._charge = charge
        self._qonm = charge / mass
        x, y, z = r # unpack the input argument to __init__()
        self.x = x
        self.y = y
        self.z = z
        #mean, sigma = distr # unpack distr (a list of 2 floats)
        #self._mean = mean
        #self._sigma = sigma
        # self._no_of_parts = uzs_at_t0.shape[0]
        #self._uzs_at_t0 = self.draw_from_Gaussian()
        ux, uy, uz = velo # velo is an input argument to the CTOR
        #self.ux = np.zeros( (self._no_of_parts, ) ) # at t0
        self.ux = ux
        self.uy = uy
        self.uz = uz
        #self.uy = np.zeros( (self._no_of_parts, ) ) # at t0
        #self.uz = uzs_at_t0 # at t0

    def __str__(self):
        return "A {} species with mass={} , charge={}, no_of_parts={},".format(self._name, self._mass, self._charge, self._no_of_parts)
    def __repr__(self):
        return f'Species(name={self._name}, mass={self._mass}, charge={self._charge}, r=[{self.x, self.y, self.z}], distr=[{self._mean, self._sigma}], no_of_parts={self._no_of_parts})'
    
    def Species_get_xyz(self):
        return np.array([self.x, self.y, self.z]) # shape (3,)

    def Species_get_uxuyuz(self):
        return np.array([self.ux, self.uy, self.uz]) # shape (3,)

    def RK45integrator(self, yscal, l_B, y_bottom_elec): # member function of the Species() class
        counter = 0
        nmax = 10**4
        steps_accepted = 0
        epsilon_0 = 10**(-20)
        t = 0
        dt = 10**(-50) # initial try for the timestep dt
        ts = []
        beta = 0.9
        vec = np.array([self.x,self.y,self.z,  self.ux,self.uy,self.uz])
        #vec = conds # conds is a np.array [x, y, z, ux, uy, uz], shape (6,)
        z_to_compare = 0.0 # initial z-value for the comparison used to see if we need to stop RK45 routine or not
        y_to_compare = 0.0  # initial y-value for the comparison used to see if we need to stop RK45 routine or not
        results = []
        ERRCON = (beta/5.)**(5) # cryptic value taken and used as from Press and Teukolsky, 1992, Adaptive Stepsize RK Integration paper
        global no_of_particles_which_haveexitB
        global no_of_particles_which_havehitelectrode
        while(counter <= nmax):
            counter += 1 # we performed 1 iteration of the while-loop!
            if (z_to_compare >= l_B): # free particle - flight
                no_of_particles_which_exitB += 1
                break
            if (y_to_compare >= y_bottom_elec):
                no_of_particles_which_hitelectrode += 1
                break
            container_from_RKF4method = get_RKF4_approx(t, vec, dt, self._qonm)
            RKF4 = container_from_RKF4method[0] # RKF4 method's approximation for the 6 odes' solutions at t_{n+1}. returns a np array of 6 floats because we have x,y,z,ux,uy,uz
            Ks =  container_from_RKF4method[1:] # a list of 5 np arrays (K1--->K5), each np array containing 6 floats
            RKF5 = get_RKF5_approx_efficiently(t, vec, dt, Ks, self._qonm) # a np.array with 6 floats in it
            y_to_compare = RKF4[1]
            z_to_compare = RKF4[2]
            scaled_errors_at_this_step = [ abs( (RKF5[i] - RKF4[i]) / yscal[i] ) for i in range(3) ] # i runs from 0-->2 (including 2), only care about error on x,y,z and not on ux,uy,uz 
            max_from_scalederrors_at_this_step = np.max(scaled_errors_at_this_step)
            if (max_from_scalederrors_at_this_step < epsilon_0 and max_from_scalederrors_at_this_step != 0.0): # good!
                # yes, step accepted! need optimal timestep (can increase it!)
                steps_accepted += 1
                ts.append(t)
                dt_new = beta * dt * (epsilon_0/max_from_scalederrors_at_this_step)**(0.25) 
                if (max_from_scalederrors_at_this_step <= (ERRCON * epsilon_0)): # fractional error is not that small, can increase timestep according to the found optimal value
                    dt_new = 5 * dt
                dt = dt_new
                results.append(RKF4)
                vec = RKF4 # for next iteration of the while-loop
            else:
                if (max_from_scalederrors_at_this_step == 0.0): # it's perfect!
                    steps_accepted += 1
                    ts.append(t)
                    t += dt
                    dt_new = 10 * dt # artificially increase the timestep, but not as dt_new dictates (that increase would dictate dt_new = inf for when errors = 0)
                    dt = dt_new
                    results.append(RKF4)
                    vec = RKF4
                else: # means that max_from_scalederrors_at_this_step > epsilon_0 and max_from_scalederrors_at_this_step != 0
                    # no, step not accepted. reiterate step using a lower timestep
                    dt_new = beta * dt * (epsilon_0/max_from_scalederrors_at_this_step)**(0.2)
                    if (dt_new < 0.01 * dt): # dt_new is really really small
                        dt_new = 0.01 * dt
                    dt = dt_new
        print("we exited the while-loop!")
        ts = np.array(ts)
        results = np.array(results) # all the integration timesteps laid down vertically. x,y,z, ux,uy,uz laid down horizontally across each line (across each integration timestep)
        return results[-1, :] # these results are from when: 1) particle has just hit bottom detector OR 2) particle has just exited the fields region at z = l_B
        # results[-1, :] is shape (6,)

    # @property
    # def name(self):
    #     return self._name
    # @name.setter
    # def name(self, name):
    #     self._name = name

    def Species_push_from_origin_to_endoffields(self):
        # use the RK45 integrator designed for l_E = l_B
        # initial_conds = np.array([self.x, self.y, self.z, self.ux, self.uy, self.uz])
        results_at_endoffields = self.RK45integrator(yscal_maxvalues, l_B, y_bottom_elec) # kinda useless? why not use the RKF45integrator() method directly?
        return results_at_endoffields # a numpy array shape (6,): x,y,z,  ux,uy,uz, at end of E and B fields

    @staticmethod # doesn't have self as an argument. only logically connected to this class, otherwise is unrelated to it in any way. 
    def Species_push_from_endoffields_to_detector(conds, z_det): # returns the x,y coords on the detection screen placed at z = z_det
        r_at_exit = conds[0:3] # get x,y,z coords at end of RK45 integration 
        us_at_exit = conds[3:6] # get ux, uy, uz at end of RK45 integration

        # find the drifting time drift_time in no E and B fields, E = B = 0. 
        # after how much time does the particle (now moving in free space) hits the detector?
        # z_exit = r_at_exit[2] # exit means exit from the E and B fields
        z_difference = z_det - r_at_exit[2] # z_end is where the detector is placed along z
        drift_time = z_difference / us_at_exit[2] # a float.  us_at_exit[2] is uz at exit, i.e. the velocity along z at exit

        # find the r-coords at the end of the drift time (so when z = z_end = where the detector is placed)
        r_at_end = r_at_exit + (us_at_exit * drift_time)
        return r_at_end[0:2] # x and y coords returned only

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

class Source:
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
        return uzs_at_t0 # shape (self._no_of_parts, )

######################################################
def profile(fnc):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner
#######################################################
detector_obj = Detector_Screen("detector", 1.2)
electrode_bottom_obj = Electrode("bottom_electrode", 0.01)
yscal_maxvalues = np.array([10.0, 0.01, 0.2]) # max values for x, y, z of a particle during it's flight through the E and B fields
l_B = 0.2 # meters
y_bottom_elec = electrode_bottom_obj._y_electrode

Efieldobj = Field("E-field", 2.0 * (10**6),   0.2,  1.0)
Bfieldobj = Field("B-field",     0.91,        0.2,  1.0)
E = Efieldobj._strength
B = Bfieldobj._strength

z_det = detector_obj._z_det
##################################################

KEineV = 100.0 * 10**(6)
KEinMeV = KEineV / (10**6) # for utility, for printing, etc.
mean_uzinit = from_KEineV_to_uzinit(KEineV)
sigma_uzinit = mean_uzinit / 10.0 # very narrow distribution if sigma = mean / 10
source = Source("source_{}MeV".format(KEinMeV), np.array([mean_uzinit, sigma_uzinit]), 10**3 )
uzs_at_t0 = source.draw_from_Gaussian()
particles = dict()
no_of_simulated_particles = 10**2
no_of_particles_which_havehitelectrode = 0
no_of_particles_which_haveexitB = 0
conds_at_detector_all = []
# @profile
# def f():

for i in range(no_of_simulated_particles):
    particles['proton_%d' % (i+1)] = Species("proton_{}".format(i+1), m_p, e_charge, np.array([0., 0., 0.]), np.array([0., 0., uzs_at_t0[i]]))
    conds_at_endoffields = particles['proton_%d' % (i+1)].Species_push_from_origin_to_endoffields()
    conds_at_detector = Species.Species_push_from_endoffields_to_detector(conds_at_endoffields, z_det)
    conds_at_detector_all.append(conds_at_detector)
    if (i % 5 == 0):
        print(f'Particle {i} has been processed!')

print(f'{no_of_particles_which_haveexitB} particles have exited B')
print(f'{no_of_particles_which_havehitelectrode} particles have hit the bottom electrode')
conds_at_detector_all = np.array(conds_at_detector_all)
with open("xyscatter_ondetector_{}protons_{}MeV.txt".format(no_of_simulated_particles, KEinMeV) ,"w") as f:
    np.savetxt(f, conds_at_detector_all)

plt.figure()
for i in range(no_of_simulated_particles):
    plt.scatter(conds_at_detector_all[i, 0], conds_at_detector_all[i, 1], color = 'red', s=0.2)
plt.xlabel("x [meters]")
plt.ylabel("y [meters]")
plt.title("Draws for protons' initial velocities are from a Gaussian distr" + "\n with mean (velocity) corresponding to {} MeV".format(KEinMeV) + "\n and sigma = mean / 10")
plt.savefig("xyscatter_on_detector_{}protons_{}MeV_graph.pdf".format(no_of_simulated_particles, KEinMeV), bbox_inches='tight')
