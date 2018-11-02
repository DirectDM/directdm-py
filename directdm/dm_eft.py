#!/usr/bin/env python3

import sys
import warnings
from directdm.num.num_input import Num_input
from directdm import wilson_coefficients as wc


#--------------------------#
# Define the default input #
#--------------------------#

default_input = Num_input().input_parameters



#---------------------------------------#
# "Wrapper" classes to be used by users #
#---------------------------------------#


class WC_3f(wc.WC_3flavor):
    def __init__(self, coeff_dict, DM_type=None, user_input_dict=None):
        """ 'wrapper' class providing input for 5 flavor Wilson coefficients """
        if user_input_dict is None:
            self.ip = default_input
        else:
            #print("Updating the default input parameters...")
            self.ip = Num_input(user_input_dict).input_parameters

        if DM_type is None:
            DM_type = "D"
        else:
            pass

        wc.WC_3flavor.__init__(self, coeff_dict, DM_type, self.ip)


class WC_4f(wc.WC_4flavor):
    def __init__(self, coeff_dict, DM_type=None, user_input_dict=None):
        """ 'wrapper' class providing input for 5 flavor Wilson coefficients """
        if user_input_dict is None:
            self.ip = default_input
        else:
            #print("Updating the default input parameters...")
            self.ip = Num_input(user_input_dict).input_parameters

        if DM_type is None:
            DM_type = "D"
        else:
            pass

        wc.WC_4flavor.__init__(self, coeff_dict, DM_type, self.ip)


class WC_5f(wc.WC_5flavor):
    def __init__(self, coeff_dict, DM_type=None, user_input_dict=None):
        """ 'wrapper' class providing input for 5 flavor Wilson coefficients """
        if user_input_dict is None:
            self.ip = default_input
        else:
            #print("Updating the default input parameters...")
            self.ip = Num_input(user_input_dict).input_parameters

        if DM_type is None:
            DM_type = "D"
        else:
            pass

        wc.WC_5flavor.__init__(self, coeff_dict, DM_type, self.ip)


class WC_EW(wc.WilCo_EW):
    def __init__(self, coeff_dict, Ychi, dchi, DM_type=None, user_input_dict=None):
        """ 'wrapper' class providing input for 5 flavor Wilson coefficients """
        if user_input_dict is None:
            self.ip = default_input
        else:
            #print("Updating the default input parameters...")
            self.ip = Num_input(user_input_dict).input_parameters

        if DM_type is None:
            DM_type = "D"
        else:
            pass

        wc.WilCo_EW.__init__(self, coeff_dict, Ychi, dchi, DM_type, self.ip)

