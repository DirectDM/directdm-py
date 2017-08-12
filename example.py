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
#
# See function doc for allowed keys. Other keys are currently ignored. 

dict1 = {'C61u' : 1./scale**2, 'C62u' : 1./scale**2, 'C61d' : 1./scale**2}


# Initialize an instance of the 3-flavor Wilson coefficient class. 
# 
# Mandatory first argument is the dictionary for Wilson coefficients.
# Optional argument id the DM-Type "D" [default], "M", "C", "R"

wc3f = ddm.WC_3f(dict1, "D")


# The main method is to output the NR coefficients. 
#
# The mandatory arguments are the DM mass and the momentum transfer in units of GeV.

print(wc3f.cNR(100, 50e-3))


# Optional arguments are QCD, dict, NLO which take the Boolean values "True" or "False". 
#
# The defaults are QCD=True, dict=True, NLO=False.
#
# QCD=False switches off QCD running.
# NLO=True includes the coherently enhanced NLO terms for the tensor operators.
# dict=False returns a numpy array instead of a python dictionary:

print(wc3f.cNR(100, 50e-3, dict=False))

# The entries correspond to the following keys: 

print(wc3f.wc_name_list)


# Finally, you can write a list of NR coefficients that can be loaded into DMFormFactor [arxiv:1308.6288]

wc3f.write_mma(100, 50e-3, filename='test_wc3.m')



sys.exit()

