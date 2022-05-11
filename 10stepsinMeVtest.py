import os
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
E = 2.0 * (10**6) # in V/m
B = 0.91 # in Tesla: T
l_B = 0.2 # 20 cm, in SI
z_end = 1.0 + l_B # in SI. "Dev of a high res and high disp TP": https://doi.org/10.1063/1.3523428 --->
# says that z_end is in range {0.3 + l_B; 1.0 + l_B}. I chose one value

def writeRKF4_to_file(RKF4):
    with open('results_RKF4values_from6odesadaptiveRKF45.txt','ab') as f:
        np.savetxt(f, RKF4, newline=" ") # saves arrays looking like np.array([1,2,3,4,5,6]) as rows (6 values delimited by space, saved horizontally), down across the file. 1 array per line (so N arrays down the document) 
        f.write(b"\n") # add a newline character inputted in binary mode (because we opened the file in binary mode due to np.savetxt). need to let it know that i want the next call of this function to np.savetxt() on the next line, not on the same line

def derivatives(t, vec): # vec is a numpy array: [x,y,z, u_x, u_y, u_z]. does it actually have to be a numpy array or can it be a simple list?
    gamma = np.sqrt(  (1 + (np.sum(vec[3:]**2.0) / (c**2.0)) )  ) # gamma = sqrt ( 1 + (u^2/c^2) ). u^2 = u_x^2 + u_y^2 + u_z^2. u_x is vec[3], u_y is vec[4], u_z is vec[5]
    # could do vec = np.array(vec) , if vec from arguments is a plain list
    # then do gamma = np.sqrt(  (1 + np.sum(vec[3:]**2.0) / (c**2.0) )  ).  # but I don't see the advantage of this, we are still creating a numpy array!
    dxondt = vec[3] / gamma # vec[3] is u_x
    dyondt = vec[4] / gamma # vec[4] is u_y
    dzondt = vec[5] / gamma # vec[5] is u_z
    duxondt = (-qonm * B / gamma) * vec[-1]  # vec[-1] is last elemenet of vec, i.e. vec[5], and is u_z
    duyondt = qonm * E
    duzondt = (qonm * B / gamma) * vec[3] # vec[3] is u_x
    return np.array([dxondt, dyondt, dzondt, duxondt, duyondt, duzondt]) # do you actually need to return a numpy array? yes, otherwise error at K2=derivatives(t+dt/4, vec + (dt/4)*K1)

RKF4_container = [0., 0., 0., 0., 0., 0.]
def get_RKF4_approx(t, vec, dt):
    # vec has 6 elements: [x,y,z, ux, uy, uz]
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

def MeV_to_J(value):
    # value is in MeV
    return value * 1.602 * 10**(-13) # this is Joules

for kk in range(10):
    MeV = kk + 1.0
    Joules = MeV_to_J(MeV)
    x = 0.0 # INITIAL CONDITIONS: aperture is considered to be at the origin of the Cartesian coordinate system
    y = 0.0 # INITIAL CONDITIONS: aperture is considered to be at the origin of the Cartesian coordinate system
    z = 0.0 # INITIAL CONDITIONS: aperture is considered to be at the origin of the Cartesian coordinate system
    ux = 0.0 # INITIAL CONDITIONS:
    uy = 0.0 # INITIAL CONDITIONS:
    # uz = 100000000.0 # in SI. particle has only velocity along z axis at entrance through the aperture (so at t=0)

    uz = ( c / (Joules/(m_p*(c**2)) +1 ) ) * np.sqrt( ( Joules/(m_p*(c**2)) + 1 )**2 - 1.0 ) # to get uz s.t. KE becomes equal to the variable named xMeV_inJ
    uinit = np.array([ux, uy, uz])
    gamma_init = 1 / (np.sqrt(1 - ((np.sum(uinit**2))/c**2) ))
    KE_init_joules = (gamma_init - 1.0) * (m_p * c**2) # in Joules
    KE_init_MeV = KE_init_joules * 6241506479963.2


    vec = np.array([x,y,z,  ux,uy,uz]) # INITIAL CONDITIONS packed into an array

    t = 0.0 # INITIAL CONDITIONS: initial time, t0
    tfinal = 0.05 # Useless apart from being used in dt_max below. It was meant to represent up to which time we want to integrate the EOM's.
    dt = 0.0000000000000001 # INITIAL CONDITIONS: initial timestep
    dt_max = tfinal / 100. # not to step out of the integration bounds if the error is really small and gives a h_optimal too large and thus t += dt_new >> t_max, thus exiting the while-loop
    beta = 0.9 # ''safety factor'' for timestep updates, see https://www.uni-muenster.de/imperia/md/content/physik_tp/lectures/ss2017/numerische_Methoden_fuer_komplexe_Systeme_II/rkm-1.pdf

    #ts,   xs, ys, zs,   uxs, uys, uzs = [],   [],[],[],    [],[],[] # might not be necessary to define all the x's and u's
    ts = []
    results = []
    epsilon_0 = 10**(-15) # tolerance for error, where error = [abs(RKF5[i] - RKF4[i]) for i in range(3)], so error is 3 floats

    nmax = 1 * 10**5 # maximum number of iterations of the while-loop below. if it reaches this no of iterations, it stops, desn't care if it didn't reach the tfinal (upper integration bound)
    counter = 0 # counts how many times the while-loop below has iterated
    steps_accepted = 0 # counts how many iterations have been accepted from the total #=counter iterations of the while-loop below
    z_to_compare = 0.0 # initialization
    y_to_compare = 0.0 # initialization
    y_of_bottom_electrode = 0.01 # 1cm, as in: https://doi.org/10.1063/1.3523428
    # condition = True
    # while (t < tfinal and counter < nmax):
    while(counter <= nmax):
        counter += 1 # we performed 1 iteration of the while-loop!
        if (counter % 1000 == 0):
            print (f'counter value = {counter}')
        if (y_to_compare >= y_of_bottom_electrode): # if true, clipping occurs at this timestep
            print("The bottom electrode has been hit!")
            break
        if (z_to_compare >= l_B): # free particle - flight
            print("z = l_B has been reached. Free particle flight from now on!")
            break # trajectory can be calculated via simple, constant-velocity, formulas. exit the RK integration routine

        container_from_RKF4method = get_RKF4_approx(t, vec, dt) # returns a list of 6 np arrays, each np array containing 6 floats in it
        RKF4 = container_from_RKF4method[0] # RKF4 method's approximation for the 6 odes' solutions at t_{n+1}. returns a np array of 6 floats, it contains: x,y,z, ux,uy,uz
        z_to_compare = RKF4[2]
        y_to_compare = RKF4[1]
        #print("y_to_compare is: ")
        #print(y_to_compare)
        Ks =  container_from_RKF4method[1:] # a list of 5 np arrays (K1--->K5), each np array containing 6 floats
        RKF5 = get_RKF5_approx_efficiently(t, vec, dt, Ks) # a np.array with 6 floats in it
        error_at_this_step = [abs(RKF5[i] - RKF4[i]) for i in range(3)] # i runs from 0-->2 (including 2), only care about error on x,y,z and not on ux,uy,uz
        if (np.max(error_at_this_step) < epsilon_0 and np.max(error_at_this_step) != 0.0): # good!
            # yes, step accepted! need optimal timestep now. calculate it below
            steps_accepted += 1
            dt_new = beta * dt * (epsilon_0/np.max(error_at_this_step))**(0.25)
            ts.append(t)
            if (dt_new <= dt_max):
                t += dt_new
                dt = dt_new
            else:
                t += dt # add the old dt because the newly calculated dt_new is huge!
                # dt = dt # so we don't use the newly calculated timestep dt_new, but carry on with the old one!
            #neww = RKF4 # RKF4 is a np array with 6 floats in it
            # print(RKF4.shape) # shape (6,)
            results.append(RKF4)
            vec = RKF4
        else:
            if (np.max(error_at_this_step) == 0.0): # it's perfect! keep carrying on with this timestep which gives 0 error.
                steps_accepted += 1
                ts.append(t)
                t += dt
                # neww = RKF4 # RKF4 is a np array with 6 floats in it
                # np.savetxt("results_6odes_adaptiveRKF45.txt", RKF4.reshape(1, RKF4.shape[0]) )
                #print(RKF4.shape) # shape (6,)
                results.append(RKF4)
                vec = RKF4
            else: # means that max(error_at_this_step) > epsilon_0 and that error_at_this_step != 0
                # no, step not accepted. reiterate step using a lower timestep
                dt_new = beta * dt * (epsilon_0/np.max(error_at_this_step))**(0.2)
                dt = dt_new

                # no changes made to time t and vec
                # now repeat this step (reiterate step)
    print("we exited the while-loop!")
    print("no of iterations of the while loop was: {}".format(counter))
    print("no of steps accepted was: {}".format(steps_accepted))
    ts = np.array(ts)
    # print("last time t was: {}".format(ts[-1])) # to check if we integrated up until, or close to, tfinal
    # results is a list of many many np arrays, each of which contains 6 floats: (x, y, z, ux, uy, uz) 
    # there are that many np arrays as the timesteps used in the integration.
    # so results[0] corresponds to x_0, y_0, z_0, ux_0, uy_0, uz_0
    # results[1] corresponds to x_1, y_1, z_1, ux_1, uy_1, uz_1 , where 1 denotes the 2nd timestep (t_1)
    # print(np.array(results).shape)
    np.savetxt("results_RKF4values_from6odesadaptiveRKF45_{}MeV.txt".format(MeV), np.array(results))
    # Want to write a header at the top of the file containing the results of the integration
    # Need to write the header after the integration has finished! ---> because
    # I want to put in the header the steps_accepted and counter variables, which are not known at compile time, but only after the integration has run! 

    src = open("results_RKF4values_from6odesadaptiveRKF45_{}MeV.txt".format(MeV), "r") 
    fline = "x, y, z, ux, uy, uz, " + "tfinal={}, ".format(tfinal) + "counter={}, ".format(counter) + "steps_accepted={}, ".format(steps_accepted) + "E={} V/m, ".format(E) + "B={} T, ".format(B) + "l_B={} m, ".format(l_B) + "\n" 
    oline = src.readlines() 
    oline.insert(0,fline) 
    src.close() 
    src = open("results_RKF4values_from6odesadaptiveRKF45_{}MeV.txt".format(MeV),"w") 
    src.writelines(oline) 
    src.close() 

    # ############################################################################################################
    # ------------------------------------------------------------------------------------------------------------
    # work on free-particle flight from now on
    line_count = get_no_of_lines_of_file("results_RKF4values_from6odesadaptiveRKF45_{}MeV.txt".format(MeV))

    r_at_exit = np.loadtxt("results_RKF4values_from6odesadaptiveRKF45_{}MeV.txt".format(MeV), usecols=np.arange(0,3), skiprows=line_count-1) # get x,y,z coords at end of RK45 integration 
    us_at_exit = np.loadtxt("results_RKF4values_from6odesadaptiveRKF45_{}MeV.txt".format(MeV), usecols=np.arange(3,6), skiprows=line_count-1) # get ux, uy, uz at end of RK45 integration

    # find the drifting time drift_time in no E and B fields, E = B = 0. 
    # after how much time does the particle (now moving in free space) hits the detector?
    z_exit = r_at_exit[2] # exit means exit from the E and B fields
    z_difference = z_end - z_exit # z_end is where the detector is placed along z
    drift_time = z_difference / us_at_exit[2] # a float.  us_at_exit[2] is uz at exit, i.e. the velocity along z at exit

    # find the r-coords at the end of the drift time (so when z = z_end = where the detector is placed)
    r_at_end = r_at_exit + (us_at_exit * drift_time)
    # print("shape of r_at_end is:")
    # print(r_at_end.shape)
    print(r_at_end[2] == z_end) # check that the particle reaches z_end indeed. it returns True. misleading


    # PLOTTING at end of RK iterations:
    with open("results_RKF4values_from6odesadaptiveRKF45_{}MeV.txt".format(MeV) , "r") as f:
        zs = np.loadtxt( f, usecols=(2,), skiprows=1 ) # shall show that last z-value is equal to l_B
    with open("results_RKF4values_from6odesadaptiveRKF45_{}MeV.txt".format(MeV) , "r") as f:
        ys = np.loadtxt( f, usecols=(1,), skiprows=1 )
    with open("results_RKF4values_from6odesadaptiveRKF45_{}MeV.txt".format(MeV) , "r") as f:
        xs = np.loadtxt( f, usecols=(0,), skiprows=1 )

    # print(zs.shape)
    # print(ys.shape)
    # print(xs.shape)

    # z-y scatter
    plt.figure() # the plotting below is wrong
    plt.scatter(zs, ys, color='red', s = 1.2, linewidths=0.1)
    # ax = plt.gca()
    # ax.set_ylim(ax.get_ylim()[::-1])
    plt.xlabel("z [meters]")
    plt.ylabel("y [meters]")
    plt.title("E and B along positive y. incident velocity along positive z")
    plt.savefig("2Dplot_zyplane_EBalongy_uinitalongz_{}MeV.pdf".format(MeV), bbox_inches='tight')

    # z-x scatter
    plt.figure() # the plotting below is wrong
    plt.scatter(zs, xs, color='red', s = 1.2, linewidths=0.1)
    plt.xlabel("z [meters]")
    plt.ylabel("x [meters]")
    plt.title("E and B along positive y. incident velocity along positive z")
    plt.savefig("2Dplot_zxplane_EBalongy_uinitalongz_{}MeV.pdf".format(MeV), bbox_inches='tight')

# 3D trajectory as from the RK routine
# fig2 = plt.figure(figsize=(10.0, 10.0))
# ax = fig2.add_subplot(111, projection='3d')
# ax.plot(xs_for_plot, ys_for_plot, zs_for_plot)
# ax.set_title("")
# ax.set_xlabel("x [meters]")
# ax.set_ylabel("y [meters]")
# ax.set_zlabel("z [meters]")
# plt.savefig("3Dplot_adaptiveRK45_firsttry.pdf")

    with open("r_at_end_{}MeV.txt".format(MeV), "w") as f:
        np.savetxt(f, r_at_end, newline=" ")
# want to plot particle on the detector screen
# so on a (x-y) plane at a z_end

plt.figure()
for kkk in range(10):
    MeV = kkk + 1.0
    with open("r_at_end_{}MeV.txt".format(MeV), "r") as f:
        rs_at_end = np.loadtxt(f)
    plt.scatter(rs_at_end[0], rs_at_end[1], color='red') # r_at_end[0] is x at end, r_at_end[1] is y at end
plt.xlabel("x [meters]")
plt.ylabel("y [meters]")
plt.title("incident position on detector screen")
plt.show()
plt.savefig("xyscatter_at_detector_screen_allMeV.pdf", bbox_inches='tight')


# we need to find the z-value from zs_recorded which is closest to the input z
# indexx = min(range(len(zs_recorded)), key=lambda i: abs(zs_recorded[i]-z_detector))
# print("closest z-value to input z-detector={} m is z-value={} m ".format(z_detector, zs_recorded[indexx]))
######
# OR #
######
# The sbelow finds the index of a z-value from results closest in modulus to an input value z_detector
# in the most inneficient way.
# difference = np.inf
# while(j < len(results)):
#     difference_new = abs(z_detector - results[i][2])
#     if (difference_new < difference):
#         difference = difference_new
#     if (differnece_new > difference):
#         indexx = i - 1 # we are interested in the previous i 
#         break # we got the index