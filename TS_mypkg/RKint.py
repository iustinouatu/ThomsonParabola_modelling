import numpy as np
from scipy.constants import c


def derivatives(t, vec, qonm, E, B): # vec is a numpy array: [x,y,z, u_x, u_y, u_z]. does it actually have to be a numpy array or can it be a simple list?
    """ This function returns a np array shape (6, ) containing numeric values at this timestep, for the 6 Right-Hand Sides of the 6 coupled ODEs (EOM's) this integrator solves.
    
    It initially calculates the gamma factor of the particle and then uses it to apply relativistic corrections to the coupled EOM's for the particle motion in E/B static fields 
    
    Parameters
    ----------
    t : float (current time value, useless at the moment)
    vec : np.array of shape (6, ) containing the x,y,z, ux,uy,uz of the particle at the current timestep.
    qonm : float (charge/mass ratio of the particle, in SI (no tricks, just total charge / total mass))
    E : float (value in SI (V/m) of the static electrical field through which particles move)
    B : float (value in SI (T) of the static magnetic field through which the particles move)

    Returns
    -------
    np.array shape (6, ) containing the RHSides of the 6 coupled ODE's at the current timestep
    """
    
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
def get_RKF4_approx(t, vec, dt, qonm, E, B): # vec has 6 elements: [x,y,z, ux, uy, uz]
    """ Computes the RK approximation at the current timestep of order 4 from the RK45 Fehlberg adaptive-stepsize integration scheme.

    Parameters
    ----------
    t : float (current time value in the integration routine)
    vec : np.array of shape (6, ) containing the x,y,z, ux,uy,uz of the particle at the current timestep.
    dt : flaot (current timestep value)
    qonm : float (charge/mass ratio of the particle, in SI (no tricks, just total charge / total mass))
    E : float (value in SI (V/m) of the static electrical field through which particles move)
    B : float (value in SI (T) of the static magnetic field through which the particles move)

    Returns
    -------
    list of len 6, first element contains the 4-th order approximations of the 6 ODEs at the next timestep 
    the next 5 elements of the list each being a np.array of 6 floats denoting the 6 values of all the 6 RHSides of the 6 coupled ODE's.
    the difference between the 5 elements is that each of them is for a different time value and each of them is dependent on the previous ones.
    these 5 elements of the list are returned for getting the 5-th order approximation from the RK45 Fehlberg integration as efficiently as possible.
    (Not to throw away derivatives evaluations and re-compute them again.)
    """

    K1 = derivatives(t, vec, qonm, E, B) # K1 will have 6 elements 
    K2 = derivatives(t+dt/4.,          vec + (dt/4.)*K1     , qonm               , E, B ) # K2 will have 6 elements
    K3 = derivatives(t + dt*(3./8.) ,  vec + dt*( (3./32.)*K1 + (9./32.)*K2), qonm, E, B ) # K3 will have 6 elements
    K4 = derivatives(t + dt*(12./13.), vec + dt*( (1932./2197.)*K1 - (7200./2197.)*K2 + (7296./2197.)*K3 ) , qonm , E, B ) # K4 will have 6 elements
    K5 = derivatives(t + dt      ,     vec + dt*( (439./216.)*K1 - 8.*K2 + (3680./513.)*K3 - (845./4104)*K4 ) , qonm , E, B  ) # K5 will have 6 elements
    RKF4 = vec + dt * ( (25./216)*K1 + (1408/2565.)*K3 + (2197./4104.)*K4 - (1./5.)*K5  ) # RKF4 will have 6 elements
    RKF4_container[0] = RKF4 # does this work? RKF4_container is a list of floats, RKF4 is a numpy array. tried in interactive mode, yes.
    RKF4_container[1] = K1
    RKF4_container[2] = K2
    RKF4_container[3] = K3
    RKF4_container[4] = K4
    RKF4_container[5] = K5
    return RKF4_container # a list containing 6 numpy arrays, each array containing 6 floats.


def get_RKF5_approx_efficiently(t, vec, dt, Ks, qonm, E, B):
    """ Computes the RK approximation at the current timestep of order 5 from the RK45 Fehlberg adaptive-stepsize integration scheme. 

    Re-uses the derivatives values previously calculated by the RK method of order 4.

    Parameters
    ----------
    t : float (current time value)
    vec : np.array shape (6, ) : values at current time for x,y,z, ux, uy, uz
    dt : float (current timestep value)
    Ks : list len 5, each element being a np.array, each np.array is shaped (6, )
    qonm : float (charge/mass ratio of the particle, in SI (no tricks, just total charge / total mass))
    E : float (value in SI (V/m) of the static electrical field through which particles move)
    B : float (value in SI (T) of the static magnetic field through which the particles move)

    Returns
    -------
    np.array shape (6, ) with the 5-th order approximations of the 6 ODEs at the next timestep.
    """

    # Ks is a list of np arrays
    # Ks[0] is K1, Ks[1] is K2, ... , Ks[4] is K5
    # each element from Ks has 6 components, so can be indexed as Ks[0...5]
    # vec has 6 elements
    K6 = derivatives(t + dt/2. , vec +  dt*( -(8./27.)*Ks[0] + 2.*Ks[1] - (3544./2565.)*Ks[2] + (1859./4104.)*Ks[3] - (11./40.)*Ks[4] ), qonm , E, B) # K6 will have 6 elements
    # Ks[0...5] all have 6 elements in them
    # so RKF6 below will have 6 elements in it.
    RKF5 = vec + dt * (  (16./135.)*Ks[0] + (6656./12825.)*Ks[2] + (28561./56430.)*Ks[3] - (9./50.)*Ks[4]  +(2./55.)*K6  )
    return RKF5 # RKF5 will contain 6 floats ''in it''

#no_of_particles_which_haveexitB = 0
#no_of_particles_which_havehitelectrode = 0
def RK45integrator(x,y,z,ux,uy,uz,       yscal, tol,  
                  length_of_thisregion, y_bottom_elec, qonm, 
                   E,  B):
    """ Integrates the relativistic EOMs for a given particle. 

    From given initial coordinates with given initial velocities (6 boundary conditions, sufficient for solving 6 coupled first-order ODE's),
    up until the particle reaches the end of E/B fields or it touches the bottom electrode ('clipping'),
    or it does more than nmax integration steps (usually a v. large value which is not attained for reasonably chosen physical parameters)

    Parameters
    ----------
    x : float (initial x-coordinate of the particle)
    y : float (initial y-coordinate of the particle)
    z : float (initial z-coordinate of the particle)
    ux : float (initial velocity of the particle, along x-axis)
    uy : float (initial velocity of the particle, along y-axis)
    uz : float (initial velocity of the particle, along z-axis)

    yscal : list of 3 floats. contains the maximum values (in modulus) the x,y,z coordinates of the particle can attain in this Physical process at hand. 
    tol : float (the tolerance: the maximum relative error of the current timestep (relative to the maximum value of the variable inputted in yscal))
    
    length_of_thisregion : float (Geometry: the length along which E/B field region extend along z-axis, in SI (meters))
    y_bottom_elec : float (the y-coordinate of the bottom electrode, in SI (meters))
    qonm : float (charge/mass ratio of the particle, in SI (no tricks, just total charge / total mass))
    E : float (value in SI (V/m) of the static electrical field through which particles move)
    B : float (value in SI (T) of the static magnetic field through which the particles move)

    Returns
    -------
    no_of_particles_which_haveexitB : int (number of particles which have exit the fields region (so successfully capturated on the detector screen))
    no_of_particles_which_hitelectrode : int (number of particles which have hit the bottom electrode (clipping))
    results[-1, :] : np.array shape (6, ) : the x,y,z, ux, uy, uz all in SI, at the end of the integration done by this function.
    """

    no_of_particles_which_hitelectrode = 0
    no_of_particles_which_haveexitB = 0
    counter = 0
    nmax = 2 * 10**5
    steps_accepted = 0
    epsilon_0 = tol 
    smallest_dt = 10**(-12) # 1000 fs = 1 ps
    t = 0
    dt = 10**(-100) # initial try for the timestep dt
    ts = []
    beta = 0.9
    vec = np.array([x,y,z,   ux,uy,uz])
    z_to_compare = z # initial z-value for the comparison used to see if we need to stop RK45 routine or not
    y_to_compare = y  # initial y-value for the comparison used to see if we need to stop RK45 routine or not
    results = []
    ERRCON = (beta/5.)**(5) # cryptic value taken and used as from Press and Teukolsky, 1992, Adaptive Stepsize RK Integration paper
    while(counter <= nmax):
        counter += 1 # we performed 1 iteration of the while-loop!
        if (z_to_compare >= length_of_thisregion): 
            no_of_particles_which_haveexitB += 1
            #print("We'll break on branch 1 inside RKint")
            break # goes out of the while loop
        if (E != 0.0):
            if (y_to_compare >= y_bottom_elec):
                no_of_particles_which_hitelectrode += 1
                #print("We'll break on branch 2 inside RKint")
                break
        container_from_RKF4method = get_RKF4_approx(t, vec, dt, qonm, E, B)
        RKF4 = container_from_RKF4method[0] # RKF4 method's approximation for the 6 odes' solutions at t_{n+1}. returns a np array of 6 floats because we have x,y,z,ux,uy,uz
        Ks =  container_from_RKF4method[1:] # a list of 5 np arrays (K1--->K5), each np array containing 6 floats
        RKF5 = get_RKF5_approx_efficiently(t, vec, dt, Ks, qonm, E , B) # a np.array with 6 floats in it
        y_to_compare = RKF4[1]
        z_to_compare = RKF4[2]
        scaled_errors_at_this_step = [ abs( (RKF5[i]-RKF4[i])/yscal[i] ) for i in range(3) ] # i runs from 0-->2 (including 2), only care about error on x,y,z and not on ux,uy,uz 
        max_from_scalederrors_at_this_step = np.max(scaled_errors_at_this_step)
        if (max_from_scalederrors_at_this_step < epsilon_0 and max_from_scalederrors_at_this_step != 0.0): # good!
            # yes, step accepted! need optimal timestep (can increase it!)
            steps_accepted += 1
            ts.append(t)
            dt_new = beta * dt * (epsilon_0/max_from_scalederrors_at_this_step)**(0.25) 
            if (max_from_scalederrors_at_this_step <= (ERRCON * epsilon_0)): # fractional error is not that small, can increase timestep according to the found optimal value
                dt_new = 1.5 * dt # here was 5
            if (dt_new <= smallest_dt):
                dt = dt_new
            else:
                dt = smallest_dt
            results.append(RKF4)
            vec = RKF4 # for next iteration of the while-loop
        else:
            if (max_from_scalederrors_at_this_step == 0.0): # it's perfect!
                steps_accepted += 1
                ts.append(t)
                t += dt # below was 10
                dt_new = 3 * dt # artificially increase the timestep, but not as dt_new dictates (that increase would dictate dt_new = inf for when errors = 0)
                if (dt_new <= smallest_dt):
                    dt = dt_new
                results.append(RKF4)
                vec = RKF4
            else: # means that max_from_scalederrors_at_this_step > epsilon_0 and max_from_scalederrors_at_this_step != 0
                # no, step not accepted. reiterate step using a lower timestep
                dt_new = beta * dt * (epsilon_0/max_from_scalederrors_at_this_step)**(0.2)
                if (dt_new < 0.001*dt): # dt_new is really small
                    dt_new = 0.001*dt
                if (dt_new <= smallest_dt):
                    dt = dt_new
                else: 
                    dt = smallest_dt
                # print("We decreased the step size")
    # print("we exited the while-loop!")
    ts = np.array(ts)
    results = np.array(results) # all the integration timesteps laid down vertically. x,y,z, ux,uy,uz laid down horizontally across each line (across each integration timestep)
    return no_of_particles_which_haveexitB, no_of_particles_which_hitelectrode, results[-1, :] # these results are from when: 1) particle has just hit bottom detector OR 2) particle has just exited this region
    # returned results[-1, :] is shape (6, )
