from scipy.constants import e as e_charge, m_p

""" The ''databases'' of ion species which can be simulated with this code.

This .py contains: 
------------------
1) a list of the names, 
2) a dict with keys the names and values the charges (in SI (C's)), 
3) a dict with keys the names and values the masses (in SI (Kg's))

At the moment, only these types of ion species can be simulated:
----------------------------------------------------------------
1) protons
2) C0+ ... C6+
3) Xe0+ ... Xe54+

"""

XenonA = 131.29 # the A - number of Xenon
CarbonA = 12.0 # the A - number of Carbon

## --------------------------------------------
all_possible_names = ['proton'] # list of strings which will be populated
for i in range(55): # 54 is the last
    all_possible_names.append('Xe%d+' % i)
for i in range(7): # 6 is the last
    all_possible_names.append('C%d+' % i)


## -------------------------------------------
charges = dict()
charges['proton'] = e_charge
for i in range(55): # 54 is the last
    charges['Xe%d+' % i] = i * e_charge
for i in range(7):
    charges['C%d+' % i] = i * e_charge



## ------------------------------------------
masses = dict()
masses['proton'] = 1.0 * m_p
for element in all_possible_names[1:56]:
    masses[element] = XenonA * m_p #
for element in all_possible_names[56:]:
    masses[element] = CarbonA * m_p #
