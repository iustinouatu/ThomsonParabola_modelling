import os
import cProfile, pstats, io
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c, e as elementary_charge, m_p
# this code performs adaptive optimal-step-size RKF45 integration -->
# on a set of N-coupled 1st-order ODEs

# for example, a particle subject to a force F in 3D can be taught of as: dx/dt , dy/dt, dz/dt and d^2 x / dt^2, d^2 y / dt^2, d^2 z / dt^2
# so 6-coupled first order ODE's if we set dx / dt = vx and dy / dt = vy and dz / dt = vz 
# thus: d(vx) / dt = ... , d(vy) / dt = ... , d(vz) / dt = ...
# dx / dt = ... , dy / dt = ... , dz / dt = ... 
# we need 6 initial conditions to solve this IVP: x at t=0, y at t=0, z at t=0; vx at t=0, vy at t=0, vz at t=0

charge  =  1.0 * elementary_charge # a proton
mass = 1.0 * m_p # m_p is in SI units (kg)
qonm = charge / mass # what happens for multiple species? use Classes
E = 2. * (10**6) # in V/m
B = 0.91 # in Tesla: T
l_B = 0.2 # in SI
z_end = 1.0 + l_B # in SI. "Dev of a high res and high disp TP": https://doi.org/10.1063/1.3523428 --->
# says that z_end is in range {0.3 + l_B; 1.0 + l_B}. I chose one value

def writeRKF4_to_file(RKF4):
    with open('results_RKF4values_from6odesadaptiveRKF45.txt','ab') as f:
        np.savetxt(f, RKF4, newline=" ") # saves arrays looking like np.array([1,2,3,4,5,6]) as rows (6 values delimited by space, saved horizontally), down across the file. 1 array per line (so N arrays down the document) 
        f.write(b"\n") # add a newline character inputted in binary mode (because we opened the file in binary mode due to np.savetxt). need to let it know that i want the next call of this function to np.savetxt() on the next line, not on the same line

def derivatives(t, vec): # vec is a numpy array: [x,y,z, u_x, u_y, u_z]. does it actually have to be a numpy array or can it be a simple list?
    gamma = np.sqrt(  (1 + np.sum(vec[3:]**2.0) / (c**2.0) )  ) # gamma = sqrt ( 1 + (u^2/c^2) ). u^2 = u_x^2 + u_y^2 + u_z^2. u_x is vec[3], u_y is vec[4], u_z is vec[5]
    # could do vec = np.array(vec) , if vec from arguments is a plain list
    # then do gamma = np.sqrt(  (1 + np.sum(vec[3:]**2.0) / (c**2.0) )  ).  # but I don't see the advantage of this, we are still creating a numpy array!
    dxondt = vec[3] / gamma # vec[3] is u_x
    dyondt = vec[4] / gamma # vec[4] is u_y
    dzondt = vec[5] / gamma # vec[5] is u_z
    duxondt = (-qonm * B / gamma) * vec[-1]  # vec[-1] is last elemenet of vec, i.e. vec[5], and is u_z
    duyondt = qonm * E
    duzondt = (qonm * B / gamma) * vec[3] # vec[3] is u_x
    return np.array([dxondt, dyondt, dzondt,  duxondt, duyondt, duzondt]) # do you actually need to return a numpy array? yes, otherwise error. at K2=derivatives(t+dt/4, vec + (dt/4)*K1)


RKF4_container = [0., 0., 0., 0., 0., 0.]
def get_RKF4_approx(t, vec, dt):
    # vec has 6 elements
    K1 = derivatives(t, vec) # K1 will have 6 elements 
    K2 = derivatives(t+dt/4.,          vec + (dt/4.)*K1                      ) # K2 will have 6 elements
    K3 = derivatives(t + dt*(3./8.) ,  vec + dt*( (3./32.)*K1 + (9./32.)*K2) ) # K3 will have 6 elements
    K4 = derivatives(t + dt*(12./13.), vec + dt*( (1932./2197.)*K1 - (7200./2197.)*K2 + (7296./2197.)*K3 )      ) # K4 will have 6 elements
    K5 = derivatives(t + dt      ,     vec + dt*( (439./216.)*K1 - 8.*K2 + (3680./513.)*K3 - (845./4104)*K4 )   ) # K5 will have 6 elements
    RKF4 = vec + dt * ( (25./216)*K1 + (1408/2565.)*K3 + (2197./4104.)*K4 - (1./5.)*K5  ) # RKF4 will have 6 elements
    RKF4_container[0] = RKF4 # does this work? RKF4_container is a list of floats, RKF4 is a numpy array. tried in interactive mode, yes.
    RKF4_container[1] = K1
    RKF4_container[2] = K2
    RKF4_container[3] = K3
    RKF4_container[4] = K4
    RKF4_container[5] = K5
    return RKF4_container # a list containing 6 numpy arrays, each array containing 6 floats.

def get_RKF5_approx_efficiently(t, vec, dt, Ks):
    # Ks is a numpy array
    # Ks[0] is K1, Ks[1] is K2, ... , Ks[4] is K5
    # each Kj has 6 components, so can be indexed as Kj[0...5] or as Ks[j][0...5] where j=0,1,2,3,4 (5 values)
    # vec has 6 elements

    K6 = derivatives(t + dt/2. , vec +  dt*( -(8./27.)*Ks[0] + 2.*Ks[1] - (3544./2565.)*Ks[2] + (1859./4104.)*Ks[3] - (11./40.)*Ks[4] )   ) # K6 will have 6 elements
    # Ks[0...5] all have 6 elements in them
    # so RKF6 below will have 6 elements in it.
    RKF5 = vec + dt * (  (16./135.)*Ks[0] + (6656./12825.)*Ks[2] + (28561./56430.)*Ks[3] - (9./50.)*Ks[4]  +(2./55.)*K6  )
    return RKF5 # RKF5 will contain 6 floats ''in it''

def get_no_of_lines_of_file(file):
    my_file = Path(file)
    if my_file.is_file():
        filee = open(file, "r")
        line_count = 0
        for line in filee:
            if line != "\n":
                line_count += 1
        filee.close()
        return line_count
    else:
        return 0

x = 0.0 # INITIAL CONDITIONS: aperture is considered to be at the origin of the Cartesian coordinate system
y = 0.0 # INITIAL CONDITIONS: aperture is considered to be at the origin of the Cartesian coordinate system
z = 0.0 # INITIAL CONDITIONS: aperture is considered to be at the origin of the Cartesian coordinate system
ux = 0.0 # INITIAL CONDITIONS:
uy = 0.0 # INITIAL CONDITIONS:
uz = 1000.0 # in SI. particle has only velocity along z axis at entrance through the aperture (so at t=0)

vec = np.array([x,y,z,  ux,uy,uz]) # INITIAL CONDITIONS packed into an array

t = 0.0 # INITIAL CONDITIONS: initial time, t0
tfinal = 0.0000005 # up to which time we want to integrate the EOM's
dt = 0.00000001 # INITIAL CONDITIONS: initial timestep
dt_max = tfinal / 100. # not to step out of the integration bounds if the error is really small and gives a h_optimal too large and thus t += dt_new >> t_max, thus exiting the while-loop
beta = 0.9 # ''safety factor'' for timestep updates, see https://www.uni-muenster.de/imperia/md/content/physik_tp/lectures/ss2017/numerische_Methoden_fuer_komplexe_Systeme_II/rkm-1.pdf

#ts,   xs, ys, zs,   uxs, uys, uzs = [],   [],[],[],    [],[],[] # might not be necessary to define all the x's and u's
ts = []
# results = []
epsilon_0 = 10**(-15) # tolerance for error, where error = [abs(RKF5[i] - RKF4[i]) for i in range(3)], so error is 3 floats

nmax = 10**4 # maximum number of iterations of the while-loop below. if it reaches this no of iterations, it stops, desn't care if it didn't reach the tfinal (upper integration bound)
counter = 0 # counts how many times the while-loop below has iterated
steps_accepted = 0 # counts how many iterations have been accepted from the total #=counter iterations of the while-loop below
# while (t < tfinal and counter < nmax):


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

# def write_results_to_file(file, results): # results has shape (steps_accepted, 6)
     

# @profile
def integration():
    counter = 0
    nmax = 10**4
    steps_accepted = 0
    epsilon_0 = 10**(-20)
    t = 0
    dt = 10**(-50) # initial try for the timestep dt
    ts = []
    dt_max = tfinal / 100. # not to step out of the integration bounds if the error is really small and gives a h_optimal too large and thus t += dt_new >> t_max, thus exiting the while-loop
    beta = 0.9 # ''safety factor'' for timestep updates, see https://www.uni-muenster.de/imperia/md/content/physik_tp/lectures/ss2017/numerische_Methoden_fuer_komplexe_Systeme_II/rkm-1.pdf
    x = 0.0 # INITIAL CONDITIONS: aperture is considered to be at the origin of the Cartesian coordinate system
    y = 0.0 # INITIAL CONDITIONS: aperture is considered to be at the origin of the Cartesian coordinate system
    z = 0.0 # INITIAL CONDITIONS: aperture is considered to be at the origin of the Cartesian coordinate system
    ux = 0.0 # INITIAL CONDITIONS:
    uy = 0.0 # INITIAL CONDITIONS:
    uz = 100.0 # in SI. particle has only velocity along z axis at entrance through the aperture (so at t=0)
    vec = np.array([x,y,z,  ux,uy,uz]) # INITIAL CONDITIONS packed into an array
    z_to_compare = 0.0 # initial z-value for the comparison used to see if we need to stop RK45 routine or not 
    l_B = 0.2
    y_to_compare = 0.0  # initial y-value for the comparison used to see if we need to stop RK45 routine or not
    y_of_bottom_electrode = 0.01
    results = []
    ERRCON = (beta/5.)**(5) # cryptic value taken and used as from Press and Teukolsky, 1992, Adaptive Stepsize RK Integration paper
    contor_true = 0
    y_scal_maxvalues = np.array([10.0, 0.01, 0.2]) # used for scaling as in: scaled_errors = [errs[i] / yscal[i] for i in range(3)]

    while(counter <= nmax):
        counter += 1 # we performed 1 iteration of the while-loop!
        if (z_to_compare >= l_B): # free particle - flight
            print("l_B has been reached!")
            break # trajectory can be calculated via simple, constant-velocity, formulas. exit the RK integration routine
        if (y_to_compare >= y_of_bottom_electrode): # if true, clipping occurs at this timestep
            print("The electrode has been hit!")
            break
        container_from_RKF4method = get_RKF4_approx(t, vec, dt) # returns a list of 6 np arrays, each np array containing 6 floats in it
        RKF4 = container_from_RKF4method[0] # RKF4 method's approximation for the 6 odes' solutions at t_{n+1}. returns a np array of 6 floats because we have x,y,z,ux,uy,uz
        Ks =  container_from_RKF4method[1:] # a list of 5 np arrays (K1--->K5), each np array containing 6 floats
        RKF5 = get_RKF5_approx_efficiently(t, vec, dt, Ks) # a np.array with 6 floats in it
        y_to_compare = RKF4[1]
        z_to_compare = RKF4[2]
        listt = [RKF4[i]==0.0 for i in range(3)]
        for i in range(2):
            if (listt[i]==True):
                contor_true += 1
             
        scaled_errors_at_this_step = [ abs( (RKF5[i] - RKF4[i]) / y_scal_maxvalues[i] ) for i in range(3) ] # i runs from 0-->2 (including 2), only care about error on x,y,z and not on ux,uy,uz 
        max_from_scalederrors_at_this_step = np.max(scaled_errors_at_this_step)
        if (max_from_scalederrors_at_this_step < epsilon_0 and max_from_scalederrors_at_this_step != 0.0): # good!
            # yes, step accepted! need optimal timestep (can increase it!)
            steps_accepted += 1
            ts.append(t)
            dt_new = beta * dt * (epsilon_0/max_from_scalederrors_at_this_step)**(0.25) #
            if (max_from_scalederrors_at_this_step > (ERRCON * epsilon_0)): # fractional error is not that small, can increase timestep according to the found optimal value
                print("fractional error is just fine for dt to be increased according to dt_new")
            else: # fractional error is really really small
                print("fractional error is really small and is not fine to increase dt according to dt_new")
                dt_new = 5 * dt
            print("relative change in dt is: ")
            print( abs ((dt - dt_new)/dt) )
            dt = dt_new
            results.append(RKF4)
            vec = RKF4 # for next iteration of the while-loop
        else:
            if (max_from_scalederrors_at_this_step == 0.0): # it's perfect!
                print("zero error")
                steps_accepted += 1
                ts.append(t)
                t += dt
                dt_new = 10 * dt # artificially increase the timestep, but not as dt_new dictates (that increase would dictate dt_new = inf for when errors = 0)
                dt = dt_new
                results.append(RKF4)
                vec = RKF4
            else: # means that max_from_scalederrors_at_this_step > epsilon_0 and max_from_scalederrors_at_this_step != 0
                # no, step not accepted. reiterate step using a lower timestep
                print("we cannot accept this step because it gives too much of an (relative) error!")
                dt_new = beta * dt * (epsilon_0/max_from_scalederrors_at_this_step)**(0.2)
                #print("the relative change in dt would be (for when we cannot accept): ")
                #print( abs ((dt - dt_new)/dt) )
                if (dt_new < 0.01 * dt): # dt_new is really really small
                    print("dt_new is really small (when we cannot accept the step because of too much of an (relative) error)")
                    dt_new = 0.01 * dt
                dt = dt_new
                # no changes made to time t and vec
                # now repeat this step (reiterate step) with a new timestep, given by dt_new (necessarily smaller than old dt)
    print("we exited the while-loop!")
    print("no of iterations of the while loop was: {}".format(counter))
    print("no of steps accepted was: {}".format(steps_accepted))
    ts = np.array(ts)
    #print("last time t was: {}".format(ts[-1])) # to check if we integrated up until, or close to, tfinal
    results = np.array(results)
    #print(results.shape) # (steps_accepted, 6)
    #np.savetxt("results_RKF4values_from6odesadaptiveRKF45_fromPROF2.txt", results) # saves results shape (steps_accepted, 6) as it should
    print(f'contor_true is {contor_true}') # shall yield 0 always
    
integration()