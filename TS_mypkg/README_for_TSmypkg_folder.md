README file for TS_mypkg folder

The lines' numbers are important for how the input sim_params.txt file is parsed.

For a line-counting starting at 1 (not at 0):
Line 1 - 3: no information for the simulation
Line 4: title
Line 5: number of regions with E / B field
Line 6: E field values
Line 7: B field values
Line 8: lengths of the geometricla regions with E / B fields
Line 9: detector's position along the z-axis
Line 10: electrode's position along the y-axis: y_electrode_bottom
Line 11: maximum possible excursion lengths of the particle along x, y, z axes
Line 12: how many chunks of particles to shoot
Line 13: names of the chunks
Line 14: number of particles from each chunk
Line 15: initial KE of the particles from each chunk (will be dealt with them later according to whether to draw the initial velocity from a Gaussian centered at Line 15 params or not)
Line 16: whats
Line 17: apsX (booleans)
Line 18: apsY (booleans)
Line 19: Rxs (floats)
Line 20: Rys (floats)
Line 21: integration tolerances (i.e. for each chunk): floats
Line 22: how to draw the velocities of each particle from the chunk (i.e. for each chunk) if what for the chunk was set to 1: {1, 2,3 }
Line 23: how to draw the velocities of each particle from the chunk (i.e. for each chunk) if what for the chunk was set to 1: int in {1, 2, 3}

