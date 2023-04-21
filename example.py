#!/usr/bin/env python3

import sys
import directdm as ddm

#----------------------------------------------------#
#                                                    #
# Template python module for the DirectDM.py package #
#                                                    #
#----------------------------------------------------#


# Set the EFT scale

scale = 100 # GeV


# Give initial conditions for Wilson coefficients as a python dictionary:

dict1 = {'C61u' : 1./scale**2, 'C62u' : 1./scale**2, 'C61d' : 1./scale**2}

# The allowed keys depend on the DM type (Dirac, Majorana, ... ) 
# and the number of active quark flavors. They can be printed via

# 3-flavor, Dirac:
print('Allowed keys for Dirac DM in the three-flavor theory:\n')
print(ddm.WC_3f({}, "D").wc_name_list)
print('\n')
# 3-flavor, Majorana:
print('Allowed keys for Majorana DM in the three-flavor theory:\n')
print(ddm.WC_3f({}, "M").wc_name_list)
print('\n')
# 5-flavor, complex scalar:
print('Allowed keys for complex scalar DM in the flavor-flavor theory:\n')
print(ddm.WC_5f({}, "C").wc_name_list)
print('\n')


#-----------------------#
# Three-flavor examples #
#-----------------------#


# Initialize an instance of the 3-flavor Wilson coefficient class. 
# 
# Mandatory first argument is the dictionary for Wilson coefficients.
# Optional argument id the DM-Type "D" [default], "M", "C", "R"

wc3f = ddm.WC_3f(dict1, "D")


# The main method is to output the NR coefficients. 
#
# The mandatory arguments are the DM mass and the momentum transfer in units of GeV.
#
# Optional arguments are RGE and NLO which take the Boolean values "True" or "False". 
#
# The defaults are RGE=True, NLO=False.
#
# RGE=False switches off QCD and QED running.
# NLO=True includes the coherently enhanced NLO terms for the tensor operators.

print('The NR coefficients :\n')
print(wc3f.cNR(100, 50e-3))
print('\n')


# Finally, you can write a list of NR coefficients that can be loaded into the Mathematica package "DMFormFactor" [arxiv:1308.6288]:

wc3f.write_mma(100, 50e-3, filename='test_wc3.m')

print('\n')
print('-----------------------------------------')
print('\n')

#----------------------#
# Five-flavor examples #
#----------------------#

# The classes for four- and five flavor Wilson coefficients work basically the same as the three-flavor class. 

# E.g. 

wc5f = ddm.WC_5f(dict1, "C")

# If you like, you can do running:

print('Run in five-flavor theory from MZ to mb(mb):\n')
print(wc5f.run())
print('\n')

# And matching:

print('Match from five-flavor to four-flavor theory at scale mb(mb):\n')
print(wc5f.match())
print('\n')

print('\n')
print('-----------------------------------------')
print('\n')




#---------------------------------------------#
# Examples with new dimension-seven operators #
#---------------------------------------------#


# Low-energy coefficients for dimension-seven operators with derivatives

dict7 = {'C715u' : 1./scale**3, 'C715d' : 1./scale**3, 'C715s' : 1./scale**3, 'C716u' : 1./scale**3, 'C716d' : 1./scale**3, 'C716s' : 1./scale**3, }
wc_7 = ddm.WC_5f(dict7, DM_type="M")


print('Low-energy coefficients from dimension-seven operators, Majorana DM:\n')
print(wc_7.cNR(100, 50e-3))
print('\n')



# Rayleigh operators currently only get a matrix element from QED mixing into scalar-current operators:

dict_rayleigh = {'C711' : 1./scale**3, 'C712' : 1./scale**3, 'C713' : 1./scale**3, 'C714' : 1./scale**3}
wc_rayleigh = ddm.WC_5f(dict_rayleigh, DM_type="D")


print('Low-energy coefficients from Rayleigh operators, from QED mixing:\n')
print(wc_rayleigh.cNR(100, 50e-3, RGE=True))
print('\n')
print('If you switch of QED, the low-energy coefficients are zero:\n')
print(wc_rayleigh.cNR(100, 50e-3, RGE=False))
print('\n')



#-------------------------------------------------------------#
# Example with Wilson coefficients in the e/w unbroken theory #
#-------------------------------------------------------------#

# For the Wilson coefficients above the weak scale, several input parameters need to be specified, in this order:
#
# The dictionary of initial conditions at scale Lambda
# The NP scale Lambda in GeV
# The DM hypercharge Ychi
# The dimension of the DM SU(2) representation.

# Note that Wilson coefficients above the weak scale do *not* include the factor 1/Lambda^n in their definition!

# E.g. 

wc_ew = ddm.WC_EW({'C663' : -1./1000**2 , 'C673' : 1./1000**2 , 'C683' : 1.}, 0, 3)

# If you like, you can do running:

print('Run in unbroken e/w theory from 1000 to MZ:\n')
print(wc_ew.run(mu_Lambda=1000))
print('\n')

# Of course, you can get the low-energy coefficients cNR, as usual: 

print('Low-energy coefficients from e/w theory:\n')
print(wc_ew.cNR(100, 50e-3, mu_Lambda=1000))
print('\n')

print('\n')
print('-----------------------------------------')
print('\n')


sys.exit()

