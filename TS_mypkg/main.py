import Species, Geometry, utility_fns, databases, RKint # why not from TS_mypkg import ... ? <--- gives ERROR
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm # facilitates an easy workflow to have a different color for each curve on the plot at the end

all_possible_names = databases.all_possible_names
masses = databases.masses
charges = databases.charges

"""
# Geometry explanation: initial velocity of particles along z axis.
# E and B fields parallel one to each other and oriented along positive y direction.
# Both E and B fields stop (instantly go to 0 value) at same z location, denoted by l_B below.
# Both E and B fields are constant and not influenced in any way by the passing, moving, particles.
# The code asks for geometry input from the user.

# Particles' input to this code are accepted per ''chunk'' of particles.
# A chunk of particles is a group of some particles of same species, which will be input to the aperture in a way dependent on an option chosen by the user.
# Each chunk will have this option selected for it. if you input 3 chunks, 3 (possibly different) options will have to be chosen.

# Option 1: aperture gets #_particles of the same species. particles have only u_z != 0 and their u_z's are drawn from a Gaussian with mean inputted by you and sigma = mean/10. particles have initial x,y = 0 (and z = 0 ofc)
# Option 2: aperture gets #_particles of the same species. particles have only u_z != 0 and all their u_z's are equal between them and inputted by you. particles have initial coordinate x not necessarily equal to 0 (but z = 0 ofc).
# Option 2 is suited to see the dispersion on screen due to the non-pointlike aperture (with a finite radius, Radius R != 0.0).
# Option 2 allows to interacetively select an aperture which is non-pointlike only along X or only along Y, or along both directions (thus 3 different possibilities if option 2 is chosen)
"""

def dictated_by_1(no_of_parts, input_MeVs, opt1_velosopt_value):
    """  Method to return initial conditions for a chunk of particles inputted using option 1.

    Returns 2 things:
    1) Returns a list of len 1 containing a np.array of size (3, ). Array contains the x,y,z initial coordinates of the particles from the chunk.
    2) Returns a np.array of shape (no_of_parts, ) containing the initial velocities along z axis of the particles from the chunk.
    The velocities are obtained by drawing from a Gaussian distribution with mean equal to the input argument input_MeVs of this function and sigma = mean/10.

    Parameters
    ----------
    no_of_parts : int (how many particles of a given species to deal with)
    input_MeVs : float (mean KEnergy, in MeVs)
    opt1_velosopt_value : int (0, 1, 2, or 3)


    Returns
    -------
    list of len 1, np.array shape (no_of_parts, )
    """

    uz_init = utility_fns.from_KEineV_to_uzinit(input_MeVs * (10**6))

    if (opt1_velosopt_value == 1): # return same velocity for all particles
        initial_uzs = np.empty( (no_of_parts,) )
        initial_uzs.fill(uz_init)
    elif (opt1_velosopt_value == 2): # draw from a gaussian
        initial_uzs = Species.Source('Source_for_1', np.array([uz_init, uz_init/10.0]), no_of_parts).draw_from_Gaussian() # a np array shape (no_of_parts, )
    elif(opt1_velosopt_value == 3): # get velocities from an input file
        print("You need to provide the input file! For the moment, I will just return garbage values.")
        initial_uzs = np.empty( (no_of_parts,) )
    else:
        print("I cannot get the initial velocities of the particles from this chunk because the sub-option introduced differs from 0, 1, or 2. I will just return random values (garbage in Python)")
        initial_uzs = np.empty( (no_of_parts,) )

    initial_coords = np.array([0.0, 0.0, 0.0]) # x, y, z

    return [initial_coords], initial_uzs # initial coords is a np.array shape (3,), initial_uzs is shape (no_of_parts, )


def dictated_by_2(no_of_parts, input_MeVs, Xtrue, Ytrue, Rx, Ry, opt2_velosopt_value): # dispersion due to aperture for fixed incident MeV energy
    """  Method to return initial conditions for a chunk of particles inputted using option 2.

    Returns 2 things:
    1) Returns a list of len 2 containing 2 np.arrays of size (no_of_parts, ). Array contains the x (and y respectively in the 2nd array), initial coordinates of the particles from the chunk.
    2) Returns a np.array of shape (no_of_parts, ) containing the initial velocities along z axis of the particles from the chunk.
    Initial velocities returned array is filled with the exact same float.

    Parameters
    ----------
    no_of_parts : int (how many particles of a given species to deal with)
    input_MeVs : float (initial KEnergy, in MeV)
    Rx : float (major (minor) axis of the aperture, in SI units (meters), X-axis)
    Ry : float (minor (major) axis of the aperture, in SI units (meters), Y-axis)
    opt2_velosopt_value : int (0, 1, 2, or 3)

    Returns
    -------
    list of len 2, np.array shape (no_of_parts, )
    """

    want_aperture_notpointlike_along_x = Xtrue
    want_aperture_notpointlike_along_y = Ytrue

    uz_init = utility_fns.from_KEineV_to_uzinit(input_MeVs * (10**6))
    
    if (opt2_velosopt_value == 1):
        initial_uzs = np.empty( (no_of_parts,) ) # shape (no_of_parts, ), same float in all the no_of_parts locations of the array
        initial_uzs.fill(uz_init)    
    elif (opt2_velosopt_value == 2):
        initial_uzs = Species.Source('Source_for_2_2', np.array([uz_init, uz_init/10.0]), no_of_parts).draw_from_Gaussian() # returns a np array shape (self._no_of_parts, )
    elif (opt2_velosopt_value == 3):
        print("need to provide the input file")
        initial_uzs = np.empty( (no_of_parts,) ) # garbage values

    if (want_aperture_notpointlike_along_x == True and want_aperture_notpointlike_along_y == False):
        initial_xs = np.random.uniform(0, 1, no_of_parts) * Rx # aperture has radius 0.005 m = 0.5 cm. top x = 0. , bottom x = 0.01 m, center x = 0.005m
        initial_ys = np.zeros(no_of_parts)
    elif (want_aperture_notpointlike_along_x == False and want_aperture_notpointlike_along_y == True):
        initial_ys = np.random.uniform(0, 1, no_of_parts) * Ry # aperture has radius 0.005 m = 0.5 cm. top y = 0. , bottom y = 0.01 m, center y = 0.005m
        initial_xs = np.zeros(no_of_parts)
    elif (want_aperture_notpointlike_along_x == True and want_aperture_notpointlike_along_y == True):
        initial_xs = np.random.uniform(0, 1, no_of_parts) * Rx # aperture has radius 0.005 m = 0.5 cm. top coord = 0. , bottom coord = 0.01 m, center coord = 0.005m
        initial_ys = np.random.uniform(0, 1, no_of_parts) * Ry
    
    return [initial_xs, initial_ys] , initial_uzs


def get_particles_init_conds(no_of_parts, input_MeV, what_you_want_to_do, Xtrue, Ytrue, Rx, Ry, opt1_velosopt_value, opt2_velosopt_value):
    """ This function is used to return initial x,y,z coordinates of the particles and initial velocities along z-axis of the particles FROM A GIVEN CHUNK.
    
    Given how many particles of a given species you simulate, their initial KEnergy (mean or fixed, depending on option choice), the option choice (and if option is 2, aperture type)
    it returns the particles x,y,z initial coordinates and their initial velocity along z-axis.

    # If option is 1, it considers a pointlike aperture so x = y = z = 0.0 (exactly) and velocities are all equal and derived from input_MeV (option 1_1)
    or are drawn from a Gaussian distribution with mean equal to the input parameter input_MeV and sigma = mean / 10 (option 1_2)
    or come from an input file (option 1_3).

    # If option is 2, it considers a non-pointlike aperture along either x and OR y 
    (of radii Rx, Ry m when both x-y behaviour is taken into account, or length = Rx or Ry when only x or y behaviour is taken into account)
    and velocities along z-axis are all the same and equal to conversion(input_MeV) (option 2_1) 
    or are drawn from a Gaussian Distr with mean and sigma=mean/10 (option 2_2)
    or come from an input file (option 2_3).

    Parameters
    ----------
    no_of_parts : int ()
    input_MeV : float ()
    what_you_want_to_do : int (either 1 or 2 at the moment)
    Xtrue : bool ()
    Ytrue : bool ()
    Rx : float (major (minor) axis of the aperture, in SI units (meters)). only used in this function if option was chosen to be 2 for this chunk.
    Ry : float (minor (major) axis of the aperture, in SI units (meters)). only used in this function if option was chosen to be 2 for this chunk.
    opt1_velosopt_value : int (0, 1, 2 or 3)
    opt2_velosopt_value : int (0, 1, 2 or 3)

    Returns
    -------
    list of len 1 or 2, np.array shape (no_of_parts, )
    """

    if (what_you_want_to_do == 1): # aperture is pointlike (xinit = yinit = zinit = 0.0), no aperture effects considered.
        initial_coords, initial_uzs = dictated_by_1(no_of_parts, input_MeV, opt1_velosopt_value) # returned initial_coords is a list of 3 floats
        return initial_coords, initial_uzs # initial coords is a list of len 1. initial_uzs is np.array of shape (no_of_parts, )
    else:
        if (what_you_want_to_do == 2): # aperture effects are considered. can get velocities from conversion(input_MeV) or to draw from Gaussian or to get them from input file.
            initial_coords, initial_uzs = dictated_by_2(no_of_parts, input_MeV, Xtrue, Ytrue, Rx, Ry, opt2_velosopt_value)
            return initial_coords, initial_uzs # initial coords is a list of len 2. initial_uzs is np.array of shape (no_of_parts, )
        else:
            print("say again what you want to do?")


def create_Species_Objects(name, mass, charge, r, velo, no_of_particles, dict_to_put_in): # creates 100 (say) Species objects, all of same species
    """ This function creates no_of_particles objects of Species-type, based on the species characteristics and initial conditions.

    Based on the name of the species (proton, Carbon0+, Carbon1+..., Carbon6+, Xe0+, ... Xe54+) and its mass and charge,
    together with initial x,y,z coordinates and initial ux,uy,uz velocities, creates no_of_particles objects of Species type and
    stores them in a dictionary dict_to_put_in.
    Object number 1 is stored as the value for the key = 'particle_1', object number 2 is stored at the value for the key = 'particle_2' ...
    up until the counter reaches no_of_particles and the last object is stored for the key = 'particle_%d' % (no_of_particles).
    Returns the dictionary created with these objects as its values for its keys.

    Parameters
    ----------
    name : str (name of the species you want to create no_of_particles particles of its type)
    mass : float (mass in SI of the species named name)
    charge : float (charge in SI of the species named name)
    r : list (len 1 or 2, depending on the code-option chosen (len 1 for option 1)) containing the initial coordinates of the particles you want to initiate.
    if len 1, r[0] is a np.array shape (3, ) containing the x,y,z initial coordinates of the particles
    if len 2, r[0][i] is the initial x-coordinate in SI of the particle i (i runs from 0 to no_of_particles-1), r[1][i] is the initial y-coordinate in SI of the smae particle
    velo : list len no_of_particles (contains the initial z-velocities in SI of the particles you want to be initiated)
    no_of_particles : int (how many particles you want to be initiated)
    dict_to_put_in : dictionary (returned 'container' which holds the initiated particles, i.e. the initiated Species objects)

    Returns
    -------
    dict_to_put_in : dictionary containing as its keys' values the Species objects created by this function using the input parameters.
    """

    for i in range(no_of_particles): # for each particle out of this chunk of 100 (say) particles
        if (len(r) == 1): # it's option 1 then
            coords = r[0] # np array shape (3,)
        else:
            coords = np.empty( (3,) )
            if (len(r) == 2): # it's option 2 then
                coords[0] = r[0][i] # the x for this particle
                coords[1] = r[1][i] # the y for this particle
                coords[2] = 0.0 # the z for this particle
            else:
                print("Error at creating Species objects!")
                # raise ValueError('A very specific bad thing happened.')

        dict_to_put_in['particle_%d' %(i+1)] = Species.Species("{}_{}".format(name, (i+1)), mass, charge, coords, [0.0, 0.0, velo[i]])
    return dict_to_put_in

def main():
    """ Function being called at the execution of the code via $ python3 main.py. From here the program starts running.

    Inputs from user from keyboard:
    -------------------------------
    no_of_regions: int
    E : np.array of no_of_regions floats (E-field values in SI units (V/m))
    B : np.array of no_of_regions floats (B-field values in SI units (T))
    lengths : np.array of no_of_regions floats (lengths of the regions with fields)

    # l_E : float (length in SI units (meters) along z-axis across which E-field value is non-zero), made equal to l_B in the code
    # D_E : float (length in SI units (meters) along z-axis from the end of the E-field to the detector screen), made equal to D_B in the code
    z_det : float (z coordinate (measured from the aperture, i.e. from the origin) in SI units (meters) at which the detector screen in placed)
    y_electrode_bottom : float (y coordinate of the bottom electrode. helpful to see if clipping occurs or not)
    various info about the chunks of particles : various types, see below
    tols : list of floats. for each chunk of particle, the relative error tolerance "toler" is saved in the list "tols". "toler" can be different for different chunks of particles.


    Results which can be used at end of script execution.
    -------------------------
    final_coords_at_detectorscreen : list of dictionaries. each dictionary represents one chunk of particles. 
                                     each dictionary contains the x,y coordinates on the detector screen for the particles from that chunk.
    """
    no_of_regions = int(input("Please enter the number of regions with E/B field(s)"))
    E, B, lengthss = np.zeros((no_of_regions, )), np.zeros((no_of_regions, )), np.zeros((no_of_regions, )) 

    for i in range(no_of_regions):
        E[i] = float(input("Please enter the fields values for this region. \n" + "E = ? [V/m] \n"))
        B[i] = float(input("B = ? [T] \n"))
        lengthss[i] = float(input("Length of this region = ? [m] \n"))

    lengths = np.zeros((no_of_regions, ))
    for i in range(no_of_regions):
        lengths[i] = np.sum(lengthss[:i]) + lengthss[i]

    z_det = float(input("Distance at which the screen is placed from the source (distance measured across z): ? [m] \n"))
    y_electrode_bottom = float(input("Distance at which the bottom electrode is placed from the origin (distance measured along +y): ? [m] \n"))
    yscal_maxvalues = np.array([10.0, 0.05, 1.5]) # max values for x, y, z of a particle during it's flight through the E and B fields

    counter_chunks_of_input = 0
    names, no_of_particles, input_MeV, whats, apsX, apsY =  [], [], [], [], [], []
    opt1_velosopts_container, opt2_velosopts_container, general_velosopts_container, tols = [], [], [], []
    contor_what_equal_2 = 0 # helpful not to ask for input from user multiple times if he already asked for option2 for at least 1 chunk.
    while (True):
        response = input("Do you want to create another chunk of particles? [Y/N] \n")
        if (response == "Y" or response == "y" or response == "Yes"):
            counter_chunks_of_input += 1
            name = input("Species Name? can only choose from (careful not to introduce typos!): [proton; C0+...6+; Xe0+...54+; Ar0+...54+] \n")
            condnames = True
            while (condnames):
                if name in all_possible_names:
                    print("You introduced a correct name of Species.")
                    names.append(name)
                    condnames = False
                else:
                    print("Error: You introduced a wrong name of Species. ABORT")
                    print("Try again. Please introduce a valid species name!")
                    name = input("Species Name? can only choose from (careful not to introduce typos!): [proton; C0+...6+; Xe0+...54+] \n")
            
            number_of_particles = int(input("How many {}s ? \n".format(name)))
            no_of_particles.append(number_of_particles)
            input_energy = float(input("Initial KEnergy in MeV ? \n"))
            input_MeV.append(input_energy)
            toler = float(input("Tolerance (for integration purposes) for this chunk of particles? \n"))
            tols.append(toler) # for each chunk of particles
            what = int(input("What do you want to do with this chunk of particles? [1/2]" + "\n" + "1 means NO to APERTURE effects, 2 means YES \n"))
            whats.append(what)

            if (what == 2): # if you want to consider aperture effects for this chunk
                contor_what_equal_2 += 1
                aperture_nonpoint_alongX = input("Do you want the aperture to be NON-pointlike along X? [Y/N] \n")
                condX = True
                while (condX):
                    if (aperture_nonpoint_alongX =='Y' or aperture_nonpoint_alongX =='y' or aperture_nonpoint_alongX =='Yes' or aperture_nonpoint_alongX =='YES'):
                        aperture_nonpoint_alongX = True
                        apsX.append(aperture_nonpoint_alongX)
                        if (contor_what_equal_2 == 1): # only happens for the first chunk of particles which requests aperture effects
                            Rx = float(input("Aperture radius R in meters for X-axis? [non-zero, positive value needed] \n"))
                        condX = False
                    elif (aperture_nonpoint_alongX =='N' or aperture_nonpoint_alongX =='n' or aperture_nonpoint_alongX =='No' or aperture_nonpoint_alongX =='NO'):
                        aperture_nonpoint_alongX = False
                        apsX.append(aperture_nonpoint_alongX)
                        Rx = 0.0
                        condX = False
                    else:
                        print("wrong answer for X-direction aperture type")
                        aperture_nonpoint_alongX = input("Do you want the aperture to be NON-pointlike along X? [Y/N] \n")

                aperture_nonpoint_alongY = input("Do you want the aperture to be NON-pointlike along Y? [Y/N] \n")
                condY = True
                while(condY):
                    if (aperture_nonpoint_alongY == 'Y' or aperture_nonpoint_alongY == 'y' or aperture_nonpoint_alongY == 'Yes' ): # only happens for the first chunk of particles which requests aperture effects
                        aperture_nonpoint_alongY = True
                        apsY.append(aperture_nonpoint_alongY)
                        if (contor_what_equal_2 == 1): # only happens for 1st chunk of particles which request aperture effects, else Ry already set.
                            Ry = float(input("Aperture radius R in meters for Y-axis? [non-zero, positive value needed] \n"))
                        condY = False
                    elif (aperture_nonpoint_alongY == 'N' or aperture_nonpoint_alongY == 'n'):
                        aperture_nonpoint_alongY = False
                        apsY.append(aperture_nonpoint_alongY)
                        if (contor_what_equal_2 == 1):
                            Ry = 0.0
                        condY = False
                    else:
                        print("wrong answer for Y-direction aperture type")
                        aperture_nonpoint_alongY = input("Do you want the aperture to be NON-pointlike along Y? [Y/N] \n")
                # for what = 2 , where do you want velocities to come from?
                opt2_velosopt = int(input("How do you want to deal with this chunks' incident particles' velocities? [1/2/3] \n"))
                condopt2velos = True
                while (condopt2velos):
                    if (opt2_velosopt == 1 or opt2_velosopt == 2 or opt2_velosopt == 3):
                        opt2_velosopts_container.append(opt2_velosopt)
                        opt1_velosopts_container.append(0)
                        general_velosopts_container.append(opt2_velosopt)
                        condopt2velos = False
                    else:
                        print("Invalid response for velocities distribution behaviour. Try again. \n")
                        opt2_velosopt = int(input("How do you want to deal with this chunks' incident particles' velocities? [1/2/3] \n"))
                    
            else: # what = 1, it seems you don't want aperture effects.
                Rx = 0.0
                Ry = 0.0
                apsX.append(False)
                apsY.append(False) 
                # for what = 1, where do you want the velocities to come from?
                opt1_velosopt = int(input("How do you want to deal with this chunks' incident particles' velocities? [1,2,3] ; 3 doesn't work at the moment \n"))
                condopt1velos = True
                while (condopt1velos):
                    if (opt1_velosopt == 1 or opt1_velosopt == 2 or opt1_velosopt == 3):
                        opt1_velosopts_container.append(opt1_velosopt)
                        opt2_velosopts_container.append(0) # signifies that this chunk doesn't deal with option2 and any of its suboptions.
                        general_velosopts_container.append(opt1_velosopt)
                        condopt1velos = False # to allow exiting the while-loop
                    else:
                        print("Invalid response for velocities distribution behaviour. Try again. \n")
                        opt1_velosopt = int(input("How do you want to deal with this chunks' incident particles' velocities? [1/2/3] ; 3 doesn't work at the moment \n"))
        else:
            if(response == "N" or response == "n"): # user doesn't want any other chunks of particles. break
                break # go out of the while-loop and continue executing instructions appearing after the while-loop.
            else:
                print("invalid response! try again!")
                continue
    title_of_graph = input("Please specify under which name you want to save results at detector screen. It will save a .npz file, a .txt file, and plot 2 graphs, all saved with the name you give, in the current directory. \n")

# CREATE SPECIES OBJECTS
    list_of_dicts_containing_Species_Objs = []
    for j in range(counter_chunks_of_input): # for each chunk of particles
        initial_coords, initial_uzs = get_particles_init_conds(no_of_particles[j], input_MeV[j], 
                                                              whats[j], apsX[j], apsY[j], Rx, Ry, 
                                                              opt1_velosopts_container[j], opt2_velosopts_container[j]) 
        # initial_uzs is a np.array shape (no_of_particles, ). it can be populated with same float, OR with floats extracted from a Gaussian. This depends on which sub-option you chose.
        Species_Objs_dict = dict() # for this current chunk of particles
        Species_Objs_dict = create_Species_Objects(names[j], masses[names[j]], charges[names[j]], 
                                                  initial_coords, initial_uzs, 
                                                  no_of_particles[j], Species_Objs_dict)
        list_of_dicts_containing_Species_Objs.append(Species_Objs_dict)

# DO THE INTEGRATION WORK ACROSS ALL THE REGIONS
    final_coords_at_detectorscreen = [] # will be populated by dictionaries, 1 dictionary for each chunk of particles
    big_dict = {}
    for k in range(len(list_of_dicts_containing_Species_Objs)): # for each chunk of particles
        name_of_particles_from_chunk = list_of_dicts_containing_Species_Objs[k]['particle_1']._name[:-2] # why [:-2]?
        coords_at_detector_forthischunk_all = []

        for j in range(len(list_of_dicts_containing_Species_Objs[k].keys())): # for each particle out of this chunk
            clip_or_notexit = 0 # indicator for whether this current particle from this current chunk clipped on the electrode or not

            for r in range(no_of_regions): # for each E/B fields region
                parti_obj = list_of_dicts_containing_Species_Objs[k]['particle_%d'%(j+1)]
                exited_B, hit_E, results_for_this_part = RKint.RK45integrator(parti_obj.x, parti_obj.y, parti_obj.z, 
                                                                              parti_obj.ux, parti_obj.uy, parti_obj.uz, 
                                                                              yscal_maxvalues, tols[k], 
                                                                              lengths[r], y_electrode_bottom, parti_obj._qonm, 
                                                                              E[r],  B[r])   
                # print("We passed the RKint.RK45itnegrator instruction in this iteration of the loop indexed by r!")                                  
                if (exited_B == 1 and hit_E == 0 ): 
                    print("success for this region")
                    list_of_dicts_containing_Species_Objs[k]['particle_%d'%(j+1)].x = results_for_this_part[0]
                    list_of_dicts_containing_Species_Objs[k]['particle_%d'%(j+1)].y = results_for_this_part[1]
                    list_of_dicts_containing_Species_Objs[k]['particle_%d'%(j+1)].z = results_for_this_part[2]
                    list_of_dicts_containing_Species_Objs[k]['particle_%d'%(j+1)].ux = results_for_this_part[3]
                    list_of_dicts_containing_Species_Objs[k]['particle_%d'%(j+1)].uy = results_for_this_part[4]
                    list_of_dicts_containing_Species_Objs[k]['particle_%d'%(j+1)].uz = results_for_this_part[5]
                elif (exited_B == 0 or hit_E == 1): # failure for this region
                    print("either it didn't exit the B-field region or it hit the bottom electrode")
                    if (hit_E == 1):
                        print("it hit the electrode!")
                    if (exited_B == 0 and hit_E == 1):
                        print("it hit the electode and it didn't exit the B-field!")
                    clip_or_notexit = 1
                    break # want go to next j, i.e. next iteration of the outer for-loop, i.e. to the next particle of this chunk of particles. this particle we just processed is stuck.
            
            if clip_or_notexit == 0: # ONLY think about the coordinates at screen IF the particle DIDN'T CLIP
                coords_at_det = Species.Species.Species_push_from_endoffields_to_detector(results_for_this_part, z_det)
                coords_at_detector_forthischunk_all.append(coords_at_det) # coords_at_det is a numpy array of shape (2,), thus the x-y position for 1 particle from this current chunk
            if (j+1) % 100 == 0:
                print("I finished processing {} particles out of a total of {} particles from this chunk.".format(j+1, len(list_of_dicts_containing_Species_Objs[k].keys())))

        coords_at_detector_forthischunk_all = np.array(coords_at_detector_forthischunk_all) # will be shape (no_of_particles - bad_particles, 2), where no_of_particles is for this particular chunk of particles and bad_particles is again for this particular chunk (the ones which hit the bottom electrode or did not exit B-field)
        final_coords_at_detectorscreen.append( {name_of_particles_from_chunk : coords_at_detector_forthischunk_all} ) # a list of dictionaries
        # Q: Why do you need final_coords_at_detectorscreen list to be populated? A: For plotting at end
        big_dict = {**big_dict, **{name_of_particles_from_chunk : coords_at_detector_forthischunk_all}}
        print("We finished processing chunk number {} out of a total of {} chunks of particles.".format(k+1, len(list_of_dicts_containing_Species_Objs)))
    

    # saving results to a .npz file and an additional .txt file which stores miscellanouses useful for postprocessing
    # ------------------------------
    np.savez_compressed('{}.npz'.format(title_of_graph), **big_dict)

    list_of_keys = list(big_dict.keys()) # big_dict shall contain a number of keys as many chunks of particles you inputted 
    with open("{}.txt".format(title_of_graph), "w") as f:
        # need to save: 1) len(final_coords_at_detectorscreen); 2) names; 3) title_of_graph 
        f.write("The following lines signify simulation details, in order: len(final_coords_at_detectorscreen); names [spanning len(final_coords_at_detectorscreen) lines]; title_of_graph; E; B; lengthss; z_det; y_electrode_bottom; tols.\n")
        f.write("{}\n".format(len(final_coords_at_detectorscreen)))
        for item in list_of_keys:    # here names are saved
            f.write("%s\n" % item)
        f.write("{}.npz\n".format(title_of_graph))
        # f.write("E = {} V/m , B = {} T , l_E = l_B = {} m , z_det = {} m , y_electrode_bottom = {} m , Accuracy = {}\n".format(E, B, l_E, z_det, y_electrode_bottom, tols))
        f.write("{}\n".format(E))
        f.write("{}\n".format(B))
        f.write("{}\n".format(lengthss))
        f.write("{}\n".format(z_det))
        f.write("{}\n".format(y_electrode_bottom))
        f.write("{}\n".format(tols))
    
    print("We start plotting now! Please wait ... ")
    colors = iter(cm.rainbow(np.linspace(0,1, 2 * len(final_coords_at_detectorscreen)))) # if you have many chunks of particles (many species), this helps select 1 DIFFERENT color to represent each chunk. 

    # plotting in the safe way
    # -------------------------
    plt.figure()
    for j in range(len(final_coords_at_detectorscreen)): # for each chunk of particles
        c = next(colors)
        c = np.reshape(c, (1, c.shape[0]) )
        for key in final_coords_at_detectorscreen[j]: # this for-loop will only make 1 iteration as the dictionary final_coords_at_detectorscreen[j] has 1 key only
            value_from_that_key = final_coords_at_detectorscreen[j][key]   # key is a str
            plt.scatter(value_from_that_key[:, 0], value_from_that_key[:, 1] , s=0.2, label=key, c=c) # key is a str
    if (whats[0] == 1):
        pass
        # plt.title("Detector screen picture showing the captured ions. \n" + " Input energies in MeV = {} \n".format(input_MeV) + "Species = {} \n".format(names) + "Options chosen = {} ".format(whats) + "Sub-options chosen = {} \n".format(general_velosopts_container) + "Integration tolerances = {} \n".format(tols) + "E = {} V/m , B = {} T , l_E = l_B = {} m , z_det = {} m \n".format(E, B, l_E, z_det) + "Number of simulated particles = {}".format(no_of_particles))
    elif (whats[0] == 2):
        pass
        # plt.title("Detector screen picture showing the captured ions. \n" + " Input energies in MeV = {} \n".format(input_MeV) + "Species = {} \n".format(names) + "Options chosen = {}".format(whats) + "Sub-options chosen = {} \n".format(general_velosopts_container)  + "Aperture size(s): Rx = {} m, Ry = {} m \n".format(Rx, Ry) + "Integration tolerances = {} \n".format(tols) + "E = {} V/m , B = {} T , l_E = l_B = {} m , z_det = {} m \n".format(E, B, l_E, z_det) + "Number of simulated particles = {}".format(no_of_particles))
    plt.xlabel("Deflection along x axis [meters]")
    plt.ylabel("Deflection along y axis [meters]")
    plt.legend()
    plt.savefig("{}.pdf".format(title_of_graph), bbox_inches='tight')


if __name__ == '__main__':
    main()
