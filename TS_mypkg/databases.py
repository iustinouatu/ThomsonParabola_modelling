from scipy.constants import e as e_charge, m_p
Dalton = 1.66053906660 * 10**(-27) # kg

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
XenonZ = 54
CarbonZ = 6
ArgonZ = 18

XenonWeight = 131.29 # the atomic weight of Xenon
CarbonWeight = 12.0 # the atomic weight of Carbon
ArgonWeight = 39.48 # the atomic weight of Argon

## --------------------------------------------
all_possible_names = ['proton'] # list of strings which will be populated
for i in range(XenonZ+1): # 54 is the last i the code actually works with in this for loop
    all_possible_names.append('Xe%d+' % i)

for i in range(CarbonZ+1): # 6 is the last i the code actually works with in this for loop
    all_possible_names.append('C%d+' % i)

for i in range(ArgonZ+1): # 18 is the last i the code actually works with in this for loop
    all_possible_names.append('Ar%d+' % i)


## -------------------------------------------
charges = dict()
charges['proton'] = e_charge
for i in range(55): # 54 is the last
    charges['Xe%d+' % i] = i * e_charge

for i in range(7):
    charges['C%d+' % i] = i * e_charge

for i in range(19): # 18 is the last
    charges['Ar%d+' % i] = i * e_charge


## ------------------------------------------
masses = dict()
masses['proton'] = 1.0 * m_p

for element in all_possible_names[1 : (XenonZ+2)]: # starting from 1 because the 0-th element is 'proton'
    masses[element] = XenonWeight * Dalton 

for element in all_possible_names[(XenonZ+2) : (XenonZ+2+CarbonZ+1)]:
    masses[element] = CarbonWeight * Dalton 

for element in all_possible_names[(XenonZ+2+CarbonZ+1) : (XenonZ+2+CarbonZ+1+ArgonZ+1)]:
    masses[element] = ArgonWeight * Dalton
