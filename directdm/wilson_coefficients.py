#!/usr/bin/env python3

import sys
import numpy as np
import scipy.integrate as spint
import warnings
import os.path
from directdm.run import adm
from directdm.run import rge
from directdm.num.num_input import Num_input
from directdm.match.dim4_gauge_contribution import Higgspenguin
from directdm.num.single_nucleon_form_factors import *

#----------------------------------------------#
# convert dictionaries to lists and vice versa #
#----------------------------------------------#

def dict_to_list(dictionary, order_list):
    """ Create a list from dictionary, according to ordering in order_list """
    #assert sorted(order_list) == sorted(dictionary.keys())
    wc_list = []
    for wc_name in order_list:
        wc_list.append(dictionary[wc_name])
    return wc_list

def list_to_dict(wc_list, order_list):
    """ Create a dictionary from a list wc_list, using keys in order_list """
    #assert len(order_list) == len(wc_list)
    wc_dict = {}
    for wc_ind in range(len(order_list)):
        wc_dict[order_list[wc_ind]] = wc_list[wc_ind]
    return wc_dict
    


#---------------------------------------------------#
# Classes for Wilson coefficients at various scales #
#---------------------------------------------------#


class WC_3flavor(object):
    def __init__(self, coeff_dict, DM_type, input_dict):
        """ Class for Wilson coefficients in 3 flavor QCD x QED plus DM.

        The first argument should be a dictionary for the initial conditions
        of the 2 + 24 + 4 + 36 + 4 + 48 + 12 = 130 
        dimension-five to dimension-eight three-flavor-QCD Wilson coefficients of the form
        {'C51' : value, 'C52' : value, ...}. 
        An arbitrary number of them can be given; the default values are zero. 

        The second argument is the DM type; it can take the following values: 
            "D" (Dirac fermion)
            "M" (Majorana fermion)
            "C" (Complex scalar)
            "R" (Real scalar)

        The possible names are (with an hopefully obvious notation):

        Dirac fermion:       'C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                             'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau',
                             'C79u', 'C79d', 'C79s', 'C79e', 'C79mu', 'C79tau', 
                             'C710u', 'C710d', 'C710s', 'C710e', 'C710mu', 'C710tau',
                             'C711', 'C712', 'C713', 'C714',
                             'C715u', 'C715d', 'C715s', 'C715e', 'C715mu', 'C715tau', 
                             'C716u', 'C716d', 'C716s', 'C716e', 'C716mu', 'C716tau',
                             'C717u', 'C717d', 'C717s', 'C717e', 'C717mu', 'C717tau', 
                             'C718u', 'C718d', 'C718s', 'C718e', 'C718mu', 'C718tau',
                             'C719u', 'C719d', 'C719s', 'C719e', 'C719mu', 'C719tau', 
                             'C720u', 'C720d', 'C720s', 'C720e', 'C720mu', 'C720tau', 
                             'C721u', 'C721d', 'C721s', 'C721e', 'C721mu', 'C721tau', 
                             'C722u', 'C722d', 'C722s', 'C722e', 'C722mu', 'C722tau' 
                             'C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s'
                             'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s'

        Majorana fermion:    'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                             'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau',
                             'C711', 'C712', 'C713', 'C714',
                             'C715u', 'C715d', 'C715s', 'C715e', 'C715mu', 'C715tau', 
                             'C716u', 'C716d', 'C716s', 'C716e', 'C716mu', 'C716tau',
                             'C717u', 'C717d', 'C717s', 'C717e', 'C717mu', 'C717tau', 
                             'C718u', 'C718d', 'C718s', 'C718e', 'C718mu', 'C718tau',
                             'C82u', 'C82d', 'C82s', 'C84u', 'C84d', 'C84s'

        Complex Scalar:      'C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                             'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                             'C65', 'C66', 'C67', 'C68' 
                             'C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s'

        Real Scalar:         'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                             'C65', 'C66', 'C67', 'C68'

        (the notation corresponds to the numbering in 1707.06998, 1801.04240).
        The Wilson coefficients should be specified in the MS-bar scheme at 2 GeV.


        For completeness, the default initial conditions at MZ for the corresponding 
        leptonic operator Wilson coefficients are defined as the SM values 
        (note that these operators have vanishing QCD anomalous dimension):

         'D63eu', 'D63muu', 'D63tauu', 'D63ed', 'D63mud', 'D63taud', 'D63es', 'D63mus', 'D63taus',
         'D62ue', 'D62umu', 'D62utau', 'D62de', 'D62dmu', 'D62dtau', 'D62se', 'D62smu', 'D62stau'

        The third argument is a dictionary with all input parameters.


        The class has three methods:

        run
        ---
        Run the Wilson coefficients from mu = 2 GeV to mu_low [GeV; default 2 GeV], with 3 active quark flavors


        cNR
        ---
        Calculate the cNR coefficients as defined in 1308.6288

        The class has two mandatory arguments: The DM mass in GeV and the momentum transfer in GeV

        The effects of double insertion [arxiv:1801.04240] are included also for leptons; 
        for couplings to electrons and muons, there are other contributions that are neglected.
        If the relevant initial conditions are set to non-zero values, a user warning is issued 
        upon creation of the class instance. 

        write_mma
        ---------
        Write an output file that can be loaded into mathematica, 
        to be used in the DMFormFactor package [1308.6288].

        """
        self.DM_type = DM_type

        self.sm_lepton_name_list = ['D63eu', 'D63muu', 'D63tauu', 'D63ed', 'D63mud',
                                    'D63taud', 'D63es', 'D63mus', 'D63taus',
                                    'D62ue', 'D62umu', 'D62utau', 'D62de', 'D62dmu',
                                    'D62dtau', 'D62se', 'D62smu', 'D62stau']

        if self.DM_type == "D":
            self.wc_name_list = ['C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61e', 'C61mu',
                                 'C61tau', 'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                 'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 'C64u',
                                 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau',
                                 'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau',
                                 'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau',
                                 'C79u', 'C79d', 'C79s', 'C79e', 'C79mu', 'C79tau',
                                 'C710u', 'C710d', 'C710s', 'C710e', 'C710mu', 'C710tau',
                                 'C711', 'C712', 'C713', 'C714',
                                 'C715u', 'C715d', 'C715s', 'C715e', 'C715mu', 'C715tau',
                                 'C716u', 'C716d', 'C716s', 'C716e', 'C716mu', 'C716tau',
                                 'C717u', 'C717d', 'C717s', 'C717e', 'C717mu', 'C717tau',
                                 'C718u', 'C718d', 'C718s', 'C718e', 'C718mu', 'C718tau',
                                 'C719u', 'C719d', 'C719s', 'C719e', 'C719mu', 'C719tau',
                                 'C720u', 'C720d', 'C720s', 'C720e', 'C720mu', 'C720tau', 
                                 'C721u', 'C721d', 'C721s', 'C721e', 'C721mu', 'C721tau',
                                 'C722u', 'C722d', 'C722s', 'C722e', 'C722mu', 'C722tau']

            self.wc8_name_list = ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s',
                                  'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s']

        if self.DM_type == "M":
            self.wc_name_list = ['C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau',
                                 'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau',
                                 'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau',
                                 'C711', 'C712', 'C713', 'C714',
                                 'C715u', 'C715d', 'C715s', 'C715e', 'C715mu', 'C715tau',
                                 'C716u', 'C716d', 'C716s', 'C716e', 'C716mu', 'C716tau',
                                 'C717u', 'C717d', 'C717s', 'C717e', 'C717mu', 'C717tau',
                                 'C718u', 'C718d', 'C718s', 'C718e', 'C718mu', 'C718tau']

            self.wc8_name_list = ['C82u', 'C82d', 'C82s', 'C84u', 'C84d', 'C84s']

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = np.r_[np.s_[0:8], np.s_[14:20], np.s_[54:66], np.s_[94:118]]
            # The list of indices to be deleted from the dim.8 ADM because of less operators
            del_ind_list_dim_8 = np.r_[np.s_[0:3], np.s_[6:9]]

        if self.DM_type == "C":
            self.wc_name_list = ['C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 
                                 'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                 'C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                 'C67', 'C68']

            self.wc8_name_list = ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s']

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = np.r_[np.s_[0:2], np.s_[8:14], np.s_[20:26], np.s_[27:28], np.s_[29:30],\
                                 np.s_[36:42], np.s_[48:66], np.s_[67:68], np.s_[69:70], np.s_[70:118]]
            # The list of indices to be deleted from the dim.8 ADM because of less operators
            del_ind_list_dim_8 = np.r_[np.s_[0:3], np.s_[6:9]]

        if self.DM_type == "R":
            self.wc_name_list = ['C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                 'C67', 'C68']

            self.wc8_name_list = []

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = np.r_[np.s_[0:26], np.s_[27:28], np.s_[29:30], np.s_[36:42],\
                                 np.s_[48:66], np.s_[67:68], np.s_[69:70], np.s_[70:118]]

        self.coeff_dict = {}

        # Issue a user warning if a key is not defined:

        for wc_name in coeff_dict.keys():
            if wc_name in self.wc_name_list:
                pass
            elif wc_name in self.wc8_name_list:
                pass
            else:
                warnings.warn('The key ' + wc_name + ' is not a valid key. Typo?')

        # Create the dictionary. 

        for wc_name in self.wc_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.

        for wc_name in self.wc8_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.


        # The dictionary of input parameters
        self.ip = input_dict

        # The default values for the SM lepton operators:

        # Input for lepton contribution

        sw = np.sqrt(self.ip['sw2_MSbar'])
        cw = np.sqrt(1-sw**2)

        vd = (-1/2 - 2*sw**2*(-1/3))/(2*sw*cw)
        vu = (1/2 - 2*sw**2*(2/3))/(2*sw*cw)
        ad = -(-1/2)/(2*sw*cw)
        au = -(1/2)/(2*sw*cw)
        vl = (-1/2 - 2*sw**2*(-1))/(2*sw*cw)
        al = -(-1/2)/(2*sw*cw)

        self.coeff_dict['D62ue'] = au*al * 4*sw**2*cw**2
        self.coeff_dict['D62umu'] = au*al * 4*sw**2*cw**2
        self.coeff_dict['D62utau'] = au*al * 4*sw**2*cw**2

        self.coeff_dict['D62de'] = ad*al * 4*sw**2*cw**2
        self.coeff_dict['D62dmu'] = ad*al * 4*sw**2*cw**2
        self.coeff_dict['D62dtau'] = ad*al * 4*sw**2*cw**2

        self.coeff_dict['D62se'] = ad*al * 4*sw**2*cw**2
        self.coeff_dict['D62smu'] = ad*al * 4*sw**2*cw**2
        self.coeff_dict['D62stau'] = ad*al * 4*sw**2*cw**2

        self.coeff_dict['D63eu'] = al*vu * 4*sw**2*cw**2
        self.coeff_dict['D63muu'] = al*vu * 4*sw**2*cw**2
        self.coeff_dict['D63tauu'] = al*vu * 4*sw**2*cw**2

        self.coeff_dict['D63ed'] = al*vd * 4*sw**2*cw**2
        self.coeff_dict['D63mud'] = al*vd * 4*sw**2*cw**2
        self.coeff_dict['D63taud'] = al*vd * 4*sw**2*cw**2

        self.coeff_dict['D63es'] = al*vd * 4*sw**2*cw**2
        self.coeff_dict['D63mus'] = al*vd * 4*sw**2*cw**2
        self.coeff_dict['D63taus'] = al*vd * 4*sw**2*cw**2


        for wc_name in self.sm_lepton_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                pass


        # Issue a user warning if certain electron / muon Wilson coefficients are non-zero:

        for wc_name in self.coeff_dict.keys():
            if DM_type == "D":
                for wc_name in ['C63e', 'C63mu', 'C64e', 'C64mu']:
                    if self.coeff_dict[wc_name] != 0.:
                        warnings.warn('The RG result for ' + wc_name + ' is incomplete, expect large uncertainties!')
                    else:
                        pass
            elif DM_type == "M":
                for wc_name in ['C64e', 'C64mu']:
                    if self.coeff_dict[wc_name] != 0.:
                        warnings.warn('The RG result for ' + wc_name + ' is incomplete, expect large uncertainties!')
                    else:
                        pass
            elif DM_type == "C":
                for wc_name in ['C62e', 'C62mu']:
                    if self.coeff_dict[wc_name] != 0.:
                        warnings.warn('The RG result for ' + wc_name + ' is incomplete, expect large uncertainties!')
                    else:
                        pass
            elif DM_type == "R":
                pass

        # Create the np.array of coefficients:
        self.coeff_list_dm_dim5_dim6_dim7 = np.array(dict_to_list(self.coeff_dict, self.wc_name_list))
        self.coeff_list_dm_dim8 = np.array(dict_to_list(self.coeff_dict, self.wc8_name_list))
        self.coeff_list_sm_lepton_dim6 = np.array(dict_to_list(self.coeff_dict, self.sm_lepton_name_list))



        #---------------------------#
        # The anomalous dimensions: #
        #---------------------------#
        if self.DM_type == "D":
            self.gamma_QED = adm.ADM_QED(3)
            self.gamma_QED2 = adm.ADM_QED2(3)
            self.gamma_QCD = adm.ADM_QCD(3)
            self.gamma_QCD2 = adm.ADM_QCD2(3)
            self.gamma_QCD_dim8 = adm.ADM_QCD_dim8(3)
        if self.DM_type == "M":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(3), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(3), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD_dim8 = np.delete(np.delete(adm.ADM_QCD_dim8(3), del_ind_list_dim_8, 0),\
                                            del_ind_list_dim_8, 1)
        if self.DM_type == "C":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(3), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(3), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD_dim8 = np.delete(np.delete(adm.ADM_QCD_dim8(3), del_ind_list_dim_8, 0),\
                                            del_ind_list_dim_8, 1)
        if self.DM_type == "R":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(3), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(3), del_ind_list, 1), del_ind_list, 2)


    def run(self, mu_low=None):
        """ Running of 3-flavor Wilson coefficients

        Calculate the running from 2 GeV to mu_low [GeV; default 2 GeV] in the three-flavor theory. 

        Return a dictionary of Wilson coefficients for the three-flavor Lagrangian
        at scale mu_low (this is the default).
        """
        if mu_low is None:
            mu_low=2

        #-------------#
        # The running #
        #-------------#

        alpha_at_mu = 1/self.ip['amtauinv']

        as31 = rge.AlphaS(self.ip['asMZ'], self.ip['Mz'])
        as31_high = as31.run({'mbmb': self.ip['mb_at_mb'], 'mcmc': self.ip['mc_at_mc']},\
                             {'mub': self.ip['mb_at_mb'], 'muc': self.ip['mc_at_mc']}, 2, 3, 1)
        as31_low = as31.run({'mbmb': self.ip['mb_at_mb'], 'mcmc': self.ip['mc_at_mc']},\
                            {'mub': self.ip['mb_at_mb'], 'muc': self.ip['mc_at_mc']}, mu_low, 3, 1)
        evolve1 = rge.RGE(self.gamma_QCD, 3)
        evolve2 = rge.RGE(self.gamma_QCD2, 3)
        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            evolve8 = rge.RGE([self.gamma_QCD_dim8], 3)
        else:
            pass

        C_at_mu_QCD = np.dot(evolve2.U0_as2(as31_high, as31_low),\
                             np.dot(evolve1.U0(as31_high, as31_low),\
                                    self.coeff_list_dm_dim5_dim6_dim7))
        C_at_mu_QED = np.dot(self.coeff_list_dm_dim5_dim6_dim7, self.gamma_QED)\
                      * np.log(mu_low/2) * alpha_at_mu/(4*np.pi)\
                    + np.dot(self.coeff_list_dm_dim5_dim6_dim7, self.gamma_QED2)\
                      * np.log(mu_low/2) * (alpha_at_mu/(4*np.pi))**2
        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            C_dim8_at_mu = np.dot(evolve8.U0(as31_high, as31_low), self.coeff_list_dm_dim8)
        else:
            pass

        # Revert back to dictionary

        dict_coeff_mu = list_to_dict(C_at_mu_QCD + C_at_mu_QED, self.wc_name_list)
        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            dict_dm_dim8 = list_to_dict(C_dim8_at_mu, self.wc8_name_list)
            dict_coeff_mu.update(dict_dm_dim8)

            dict_sm_lepton_dim6 = list_to_dict(self.coeff_list_sm_lepton_dim6, self.sm_lepton_name_list)
            dict_coeff_mu.update(dict_sm_lepton_dim6)
        else:
            pass

        return dict_coeff_mu



    def _my_cNR(self, DM_mass, RGE=None, NLO=None):
        """Calculate the coefficients of the NR operators, with momentum dependence factored out.
    
        DM_mass is the DM mass in GeV

        RGE is a flag to turn RGE running on (True) or off (False). (Default True)

        If NLO is set to True, the coherently enhanced NLO terms for Q_9^(7) are added. (Default False)

        Returns a dictionary of coefficients for the NR Lagrangian, 
        as in 1308.6288, plus coefficients c13 -- c23, c100 for "spurious" long-distance operators

        The possible names are

        ['cNR1p', 'cNR1n', 'cNR2p', 'cNR2n', 'cNR3p', 'cNR3n', 'cNR4p', 'cNR4n', 'cNR5p', 'cNR5n',
         'cNR6p', 'cNR6n', 'cNR7p', 'cNR7n', 'cNR8p', 'cNR8n', 'cNR9p', 'cNR9n', 'cNR10p', 'cNR10n',
         'cNR11p', 'cNR11n', 'cNR12p', 'cNR12n', 'cNR13p', 'cNR13n', 'cNR14p', 'cNR14n', 'cNR15p', 'cNR15n',
         'cNR16p', 'cNR16n', 'cNR17p', 'cNR17n', 'cNR18p', 'cNR18n', 'cNR19p', 'cNR19n', 'cNR20p', 'cNR20n',
         'cNR21p', 'cNR21n', 'cNR22p', 'cNR22n', 'cNR23p', 'cNR23n', 'cNR100p', 'cNR100n', 'cNR104p', 'cNR104n']
        """
        if RGE is None:
            RGE = True
        if NLO is None:
            NLO = False

        ### Input parameters ####

        mpi = self.ip['mpi0']
        mp = self.ip['mproton']
        mn = self.ip['mneutron']
        mN = (mp+mn)/2

        alpha = 1/self.ip['alowinv']
        GF = self.ip['GF']
        as_2GeV = rge.AlphaS(self.ip['asMZ'],\
                             self.ip['Mz']).run({'mbmb': self.ip['mb_at_mb'], 'mcmc': self.ip['mc_at_mc']},\
                                                {'mub': self.ip['mb_at_mb'], 'muc': self.ip['mc_at_mc']}, 2, 3, 1)
        gs2_2GeV = 4*np.pi*as_2GeV

        # Quark masses at 2GeV
        mu = self.ip['mu_at_2GeV']
        md = self.ip['md_at_2GeV']
        ms = self.ip['ms_at_2GeV']
        mtilde = 1/(1/mu + 1/md + 1/ms)
        
        # Lepton masses
        me = self.ip['me']
        mmu = self.ip['mmu']
        mtau = self.ip['mtau']

        # Z boson mass
        MZ = self.ip['Mz']

        ### Numerical constants
        mproton = self.ip['mproton']
        mneutron = self.ip['mneutron']

        F1up = F1('u', 'p').value_zero_mom()
        F1dp = F1('d', 'p').value_zero_mom()
        F1sp = F1('s', 'p').value_zero_mom()

        F1un = F1('u', 'n').value_zero_mom()
        F1dn = F1('d', 'n').value_zero_mom()
        F1sn = F1('s', 'n').value_zero_mom()

        F2up = F2('u', 'p', self.ip).value_zero_mom()
        F2dp = F2('d', 'p', self.ip).value_zero_mom()
        F2sp = F2('s', 'p', self.ip).value_zero_mom()

        F2un = F2('u', 'n', self.ip).value_zero_mom()
        F2dn = F2('d', 'n', self.ip).value_zero_mom()
        F2sn = F2('s', 'n', self.ip).value_zero_mom()

        FAup = FA('u', 'p', self.ip).value_zero_mom()
        FAdp = FA('d', 'p', self.ip).value_zero_mom()
        FAsp = FA('s', 'p', self.ip).value_zero_mom()

        FAun = FA('u', 'n', self.ip).value_zero_mom()
        FAdn = FA('d', 'n', self.ip).value_zero_mom()
        FAsn = FA('s', 'n', self.ip).value_zero_mom()

        FPpup_pion = FPprimed('u', 'p', self.ip).value_pion_pole()
        FPpdp_pion = FPprimed('d', 'p', self.ip).value_pion_pole()
        FPpsp_pion = FPprimed('s', 'p', self.ip).value_pion_pole()

        FPpun_pion = FPprimed('u', 'n', self.ip).value_pion_pole()
        FPpdn_pion = FPprimed('d', 'n', self.ip).value_pion_pole()
        FPpsn_pion = FPprimed('s', 'n', self.ip).value_pion_pole()

        FPpup_eta = FPprimed('u', 'p', self.ip).value_eta_pole()
        FPpdp_eta = FPprimed('d', 'p', self.ip).value_eta_pole()
        FPpsp_eta = FPprimed('s', 'p', self.ip).value_eta_pole()

        FPpun_eta = FPprimed('u', 'n', self.ip).value_eta_pole()
        FPpdn_eta = FPprimed('d', 'n', self.ip).value_eta_pole()
        FPpsn_eta = FPprimed('s', 'n', self.ip).value_eta_pole()

        FSup = FS('u', 'p', self.ip).value_zero_mom()
        FSdp = FS('d', 'p', self.ip).value_zero_mom()
        FSsp = FS('s', 'p', self.ip).value_zero_mom()

        FSun = FS('u', 'n', self.ip).value_zero_mom()
        FSdn = FS('d', 'n', self.ip).value_zero_mom()
        FSsn = FS('s', 'n', self.ip).value_zero_mom()

        FPup_pion = FP('u', 'p', self.ip).value_pion_pole()
        FPdp_pion = FP('d', 'p', self.ip).value_pion_pole()
        FPsp_pion = FP('s', 'p', self.ip).value_pion_pole()

        FPun_pion = FP('u', 'n', self.ip).value_pion_pole()
        FPdn_pion = FP('d', 'n', self.ip).value_pion_pole()
        FPsn_pion = FP('s', 'n', self.ip).value_pion_pole()

        FPup_eta = FP('u', 'p', self.ip).value_eta_pole()
        FPdp_eta = FP('d', 'p', self.ip).value_eta_pole()
        FPsp_eta = FP('s', 'p', self.ip).value_eta_pole()

        FPun_eta = FP('u', 'n', self.ip).value_eta_pole()
        FPdn_eta = FP('d', 'n', self.ip).value_eta_pole()
        FPsn_eta = FP('s', 'n', self.ip).value_eta_pole()

        FGp = FG('p', self.ip).value_zero_mom()
        FGn = FG('n', self.ip).value_zero_mom()

        FGtildep = FGtilde('p', self.ip).value_zero_mom()
        FGtilden = FGtilde('n', self.ip).value_zero_mom()

        FGtildep_pion = FGtilde('p', self.ip).value_pion_pole()
        FGtilden_pion = FGtilde('n', self.ip).value_pion_pole()

        FGtildep_eta = FGtilde('p', self.ip).value_eta_pole()
        FGtilden_eta = FGtilde('n', self.ip).value_eta_pole()

        FT0up = FT0('u', 'p', self.ip).value_zero_mom()
        FT0dp = FT0('d', 'p', self.ip).value_zero_mom()
        FT0sp = FT0('s', 'p', self.ip).value_zero_mom()

        FT0un = FT0('u', 'n', self.ip).value_zero_mom()
        FT0dn = FT0('d', 'n', self.ip).value_zero_mom()
        FT0sn = FT0('s', 'n', self.ip).value_zero_mom()

        FT1up = FT1('u', 'p', self.ip).value_zero_mom()
        FT1dp = FT1('d', 'p', self.ip).value_zero_mom()
        FT1sp = FT1('s', 'p', self.ip).value_zero_mom()

        FT1un = FT1('u', 'n', self.ip).value_zero_mom()
        FT1dn = FT1('d', 'n', self.ip).value_zero_mom()
        FT1sn = FT1('s', 'n', self.ip).value_zero_mom()


        ### The coefficients ###
        #
        # Note that all dependence on 1/q^2, 1/(m^2-q^2), q^2/(m^2-q^2) is taken care of
        # by defining spurious operators.
        #
        # Therefore, we need to split some of the coefficients
        # into the "pion part" etc. with the q-dependence factored out,
        # and introduce a few spurious "long-distance" operators.
        #
        # The coefficients cNR1 -- cNR12 correspond to the operators in 1611.00368 and 1308.6288
        #
        # Therefore, we define O13 = O6/(mpi^2+q^2); 
        #                      O14 = O6/(meta^2+q^2);
        #                      O15 = O6*q^2/(mpi^2+q^2);
        #                      O16 = O6*q^2/(meta^2+q^2);
        #                      O17 = O10/(mpi^2+q^2);
        #                      O18 = O10/(meta^2+q^2);
        #                      O19 = O10*q^2/(mpi^2+q^2);
        #                      O20 = O10*q^2/(meta^2+q^2);
        #
        # For the dipole interactions, these are the ones that have c2p1, c1N2, c2p2 as coefficients. 
        # Therefore, we define O21 = O5/q^2; 
        #                      O22 = O6/q^2.
        #                      O23 = O11/q^2.
        # 
        # For the tensors, O4 * q^2 appears as a leading contribution.
        # Therefore, we define O104 = O1 * q^2
        #
        # For the tensors, O1 * q^2 appears as a subleading contribution.
        # Therefore, we define O100 = O1 * q^2
        #
        # q^2 is here always the spatial part!!! 
        #

        if RGE:
            c3mu_dict = self.run(2)
        else:
            c3mu_dict = self.coeff_dict

        if self.DM_type == "D":
            my_cNR_dict = {
            'cNR1p' :   F1up*(c3mu_dict['C61u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C81u'])\
                      + F1dp*(c3mu_dict['C61d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C81d'])\
                      + F1up*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                              * c3mu_dict['C63e'] * c3mu_dict['D63eu'])\
                      + F1dp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                              * c3mu_dict['C63e'] * c3mu_dict['D63ed'])\
                      + F1up*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                              * c3mu_dict['C63mu'] * c3mu_dict['D63muu'])\
                      + F1dp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                              * c3mu_dict['C63mu'] * c3mu_dict['D63mud'])\
                      + F1up*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                              * c3mu_dict['C63tau'] * c3mu_dict['D63tauu'])\
                      + F1dp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                              * c3mu_dict['C63tau'] * c3mu_dict['D63taud'])\
                      + FGp*c3mu_dict['C71']\
                      + FSup*c3mu_dict['C75u'] + FSdp*c3mu_dict['C75d'] + FSsp*c3mu_dict['C75s']\
                      - alpha/(2*np.pi*DM_mass)*c3mu_dict['C51']\
                      + 2*DM_mass * (F1up*c3mu_dict['C715u'] + F1dp*c3mu_dict['C715d'] + F1sp*c3mu_dict['C715s']),
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : - 4*(  FAup*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                           + FAdp*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                           + FAsp*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
                           + FAup*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                           + FAdp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                           + FAsp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62se'])\
                           + FAup*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                           + FAdp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                           + FAsp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62smu'])\
                           + FAup*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                           + FAdp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62dtau'])\
                           + FAsp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62stau']))\
                      - 2*alpha/np.pi * self.ip['mup']/mN * c3mu_dict['C51']\
                      + 8*(FT0up*c3mu_dict['C79u'] + FT0dp*c3mu_dict['C79d'] + FT0sp*c3mu_dict['C79s']),
            'cNR5p' : - 2*mN * (F1up*c3mu_dict['C719u'] + F1dp*c3mu_dict['C719d'] + F1sp*c3mu_dict['C719s']),
            'cNR6p' : mN/DM_mass * FGtildep * c3mu_dict['C74']\
                      -2*mN*((F1up+F2up)*c3mu_dict['C719u']\
                             + (F1dp+F2dp)*c3mu_dict['C719d']\
                             + (F1sp+F2dp)*c3mu_dict['C719s']),
            'cNR7p' : - 2*(  FAup*(c3mu_dict['C63u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C83u'])\
                           + FAdp*(c3mu_dict['C63d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C83d'])\
                           + FAsp*(c3mu_dict['C63s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C83s'])\
                           + FAup*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63e'] * c3mu_dict['D62ue'])\
                           + FAdp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63e'] * c3mu_dict['D62de'])\
                           + FAsp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63e'] * c3mu_dict['D62se'])\
                           + FAup*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63mu'] * c3mu_dict['D62umu'])\
                           + FAdp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63mu'] * c3mu_dict['D62dmu'])\
                           + FAsp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63mu'] * c3mu_dict['D62smu'])\
                           + FAup*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63tau'] * c3mu_dict['D62utau'])\
                           + FAdp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63tau'] * c3mu_dict['D62dtau'])\
                           + FAsp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63tau'] * c3mu_dict['D62stau']))\
                      - 4*DM_mass * (FAup*c3mu_dict['C717u'] + FAdp*c3mu_dict['C717d'] + FAsp*c3mu_dict['C717s']),
            'cNR8p' : 2*(  F1up*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + F1dp*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + F1up*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64e'] * c3mu_dict['D63eu'])\
                         + F1dp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64e'] * c3mu_dict['D63ed'])\
                         + F1up*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64mu'] * c3mu_dict['D63muu'])\
                         + F1dp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64mu'] * c3mu_dict['D63mud'])\
                         + F1up*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64tau'] * c3mu_dict['D63tauu'])\
                         + F1dp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64tau'] * c3mu_dict['D63taud'])),
            'cNR9p' : 2*(  (F1up+F2up)*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + (F1dp+F2dp)*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + (F1sp+F2sp)*(c3mu_dict['C62s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C82s'])\
                         + (F1up+F2up)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63eu'])\
                         + (F1dp+F2dp)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63ed'])\
                         + (F1sp+F2sp)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63es'])\
                         + (F1up+F2up)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63muu'])\
                         + (F1dp+F2dp)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63mud'])\
                         + (F1sp+F2sp)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63mus'])\
                         + (F1up+F2up)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63tauu'])\
                         + (F1dp+F2dp)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63taud'])\
                         + (F1sp+F2sp)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63taus']))
                      + 2*mN*(  FAup*(c3mu_dict['C63u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C83u'])\
                              + FAdp*(c3mu_dict['C63d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C83d'])\
                              + FAsp*(c3mu_dict['C63s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C83s'])\
                              + FAup*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63e'] * c3mu_dict['D62ue'])\
                              + FAdp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63e'] * c3mu_dict['D62de'])\
                              + FAsp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63e'] * c3mu_dict['D62se'])\
                              + FAup*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63mu'] * c3mu_dict['D62umu'])\
                              + FAdp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63mu'] * c3mu_dict['D62dmu'])\
                              + FAsp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63mu'] * c3mu_dict['D62smu'])\
                              + FAup*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63tau'] * c3mu_dict['D62utau'])\
                              + FAdp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63tau'] * c3mu_dict['D62dtau'])\
                              + FAsp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63tau'] * c3mu_dict['D62stau']))/DM_mass\
                      - 4*mN * (FAup*c3mu_dict['C721u'] + FAdp*c3mu_dict['C721d'] + FAsp*c3mu_dict['C721s']),
            'cNR10p' : FGtildep * c3mu_dict['C73']\
                       -2*mN/DM_mass * (FT0up*c3mu_dict['C710u']\
                                        + FT0dp*c3mu_dict['C710d']\
                                        + FT0sp*c3mu_dict['C710s']),
            'cNR11p' : - mN/DM_mass * (FSup*c3mu_dict['C76u']\
                                       + FSdp*c3mu_dict['C76d']\
                                       + FSsp*c3mu_dict['C76s'])\
                       - mN/DM_mass * FGp * c3mu_dict['C72']\
                        + 2*((FT0up-FT1up)*c3mu_dict['C710u']\
                             + (FT0dp-FT1dp)*c3mu_dict['C710d']\
                             + (FT0sp-FT1sp)*c3mu_dict['C710s'])\
                        - 2*mN * (  F1up*(c3mu_dict['C716u']+c3mu_dict['C720u'])\
                                  + F1dp*(c3mu_dict['C716d']+c3mu_dict['C720d'])\
                                  + F1sp*(c3mu_dict['C716s']+c3mu_dict['C720s'])),
            'cNR12p' : -8*(FT0up*c3mu_dict['C710u'] + FT0dp*c3mu_dict['C710d'] + FT0sp*c3mu_dict['C710s']),
    
            'cNR13p' : mN/DM_mass * (FPup_pion*c3mu_dict['C78u'] + FPdp_pion*c3mu_dict['C78d'])\
                       + FPpup_pion*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdp_pion*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpup_pion*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                       + FPpdp_pion*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                       + FPpup_pion*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                       + FPpdp_pion*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                       + FPpup_pion*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                       + FPpdp_pion*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64tau'] * c3mu_dict['D62dtau']),
            'cNR14p' : mN/DM_mass * (FPup_eta*c3mu_dict['C78u']\
                                     + FPdp_eta*c3mu_dict['C78d']\
                                     + FPsp_eta*c3mu_dict['C78s'])\
                       + FPpup_eta*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdp_eta*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpsp_eta*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
                       + FPpup_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                       + FPpdp_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                       + FPpsp_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62se'])\
                       + FPpup_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                       + FPpdp_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                       + FPpsp_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62smu'])\
                       + FPpup_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                       + FPpdp_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62dtau'])\
                       + FPpsp_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62stau'])\
                       + 4*mN * (  FAup*(c3mu_dict['C718u']+c3mu_dict['C722u'])\
                                 + FAdp*(c3mu_dict['C718d']+c3mu_dict['C722d'])\
                                 + FAsp*(c3mu_dict['C718s']+c3mu_dict['C722s'])),
            'cNR15p' : mN/DM_mass * FGtildep_pion * c3mu_dict['C74'],
            'cNR16p' : mN/DM_mass * FGtildep_eta * c3mu_dict['C74'],
    
            'cNR17p' : FPup_pion*c3mu_dict['C77u'] + FPdp_pion*c3mu_dict['C77d'],
            'cNR18p' : FPup_eta*c3mu_dict['C77u'] + FPdp_eta*c3mu_dict['C77d'] + FPsp_eta*c3mu_dict['C77s'],
            'cNR19p' : FGtildep_pion * c3mu_dict['C73'],
            'cNR20p' : FGtildep_eta * c3mu_dict['C73'],
    
            'cNR21p' : mN* (2*alpha/np.pi*c3mu_dict['C51']),
            'cNR22p' : -mN**2* (- 2*alpha/np.pi * self.ip['mup']/mN * c3mu_dict['C51']),
            'cNR23p' : mN* (2*alpha/np.pi*c3mu_dict['C52']),

            'cNR100p' : (F1up*c3mu_dict['C719u'] + F1dp*c3mu_dict['C719d'] + F1sp*c3mu_dict['C719s'])/(2*DM_mass),
            'cNR104p' : 2*((F1up+F2up)*c3mu_dict['C719u']\
                           + (F1dp+F2dp)*c3mu_dict['C719d']\
                           + (F1sp+F2dp)*c3mu_dict['C719s'])/mN,




            'cNR1n' :   F1un*(c3mu_dict['C61u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C81u'])\
                      + F1dn*(c3mu_dict['C61d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C81d'])\
                      + FGn*c3mu_dict['C71']\
                      + FSun*c3mu_dict['C75u'] + FSdn*c3mu_dict['C75d'] + FSsn*c3mu_dict['C75s']\
                      + F1un*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                              * c3mu_dict['C63e'] * c3mu_dict['D63eu'])\
                      + F1dn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                              * c3mu_dict['C63e'] * c3mu_dict['D63ed'])\
                      + F1un*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                              * c3mu_dict['C63mu'] * c3mu_dict['D63muu'])\
                      + F1dn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                              * c3mu_dict['C63mu'] * c3mu_dict['D63mud'])\
                      + F1un*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                              * c3mu_dict['C63tau'] * c3mu_dict['D63tauu'])\
                      + F1dn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                              * c3mu_dict['C63tau'] * c3mu_dict['D63taud'])\
                      + 2*DM_mass * (F1un*c3mu_dict['C715u'] + F1dn*c3mu_dict['C715d'] + F1sn*c3mu_dict['C715s']),
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : - 4*(  FAun*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                           + FAdn*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                           + FAsn*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
                           + FAun*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                           + FAdn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                           + FAsn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62se'])\
                           + FAun*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                           + FAdn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                           + FAsn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62smu'])\
                           + FAun*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                           + FAdn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62dtau'])\
                           + FAsn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62stau']))\
                      - 2*alpha/np.pi * self.ip['mun']/mN * c3mu_dict['C51']\
                      + 8*(FT0un*c3mu_dict['C79u'] + FT0dn*c3mu_dict['C79d'] + FT0sn*c3mu_dict['C79s']),
            'cNR5n' : - 2*mN * (F1un*c3mu_dict['C719u'] + F1dn*c3mu_dict['C719d'] + F1sn*c3mu_dict['C719s']),
            'cNR6n' : mN/DM_mass * FGtilden * c3mu_dict['C74']\
                      -2*mN*((F1un+F2un)*c3mu_dict['C719u']\
                             + (F1dn+F2dn)*c3mu_dict['C719d']\
                             + (F1sn+F2dn)*c3mu_dict['C719s']),
            'cNR7n' : - 2*(  FAun*(c3mu_dict['C63u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C83u'])\
                           + FAdn*(c3mu_dict['C63d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C83d'])\
                           + FAsn*(c3mu_dict['C63s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C83s'])\
                           + FAun*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63e'] * c3mu_dict['D62ue'])\
                           + FAdn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63e'] * c3mu_dict['D62de'])\
                           + FAsn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63e'] * c3mu_dict['D62se'])\
                           + FAun*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63mu'] * c3mu_dict['D62umu'])\
                           + FAdn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63mu'] * c3mu_dict['D62dmu'])\
                           + FAsn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63mu'] * c3mu_dict['D62smu'])\
                           + FAun*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63tau'] * c3mu_dict['D62utau'])\
                           + FAdn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63tau'] * c3mu_dict['D62dtau'])\
                           + FAsn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C63tau'] * c3mu_dict['D62stau']))\
                      - 4*DM_mass * (FAun*c3mu_dict['C717u'] + FAdn*c3mu_dict['C717d']+ FAsn*c3mu_dict['C717s']),
            'cNR8n' : 2*(  F1un*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + F1dn*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + F1un*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64e'] * c3mu_dict['D63eu'])\
                         + F1dn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64e'] * c3mu_dict['D63ed'])\
                         + F1un*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64mu'] * c3mu_dict['D63muu'])\
                         + F1dn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64mu'] * c3mu_dict['D63mud'])\
                         + F1un*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64tau'] * c3mu_dict['D63tauu'])\
                         + F1dn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64tau'] * c3mu_dict['D63taud'])),
            'cNR9n' : 2*(  (F1un+F2un)*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + (F1dn+F2dn)*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + (F1sn+F2sn)*(c3mu_dict['C62s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C82s'])\
                         + (F1un+F2un)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63eu'])\
                         + (F1dn+F2dn)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63ed'])\
                         + (F1sn+F2sn)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63es'])\
                         + (F1un+F2un)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63muu'])\
                         + (F1dn+F2dn)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63mud'])\
                         + (F1sn+F2sn)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63mus'])\
                         + (F1un+F2up)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63tauu'])\
                         + (F1dn+F2dp)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63taud'])\
                         + (F1sp+F2sp)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63taus']))
                      + 2*mN*(  FAun*(c3mu_dict['C63u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C83u'])\
                              + FAdn*(c3mu_dict['C63d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C83d'])\
                              + FAsn*(c3mu_dict['C63s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C83s'])\
                              + FAun*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63e'] * c3mu_dict['D62ue'])\
                              + FAdn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63e'] * c3mu_dict['D62de'])\
                              + FAsn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63e'] * c3mu_dict['D62se'])\
                              + FAun*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63mu'] * c3mu_dict['D62umu'])\
                              + FAdn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63mu'] * c3mu_dict['D62dmu'])\
                              + FAsn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63mu'] * c3mu_dict['D62smu'])\
                              + FAun*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63tau'] * c3mu_dict['D62utau'])\
                              + FAdn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63tau'] * c3mu_dict['D62dtau'])\
                              + FAsn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                      * c3mu_dict['C63tau'] * c3mu_dict['D62stau']))/DM_mass\
                      - 4*mN * (FAun*c3mu_dict['C721u']\
                                + FAdn*c3mu_dict['C721d']\
                                + FAsn*c3mu_dict['C721s']),
            'cNR10n' : FGtilden * c3mu_dict['C73']\
                     -2*mN/DM_mass * (FT0un*c3mu_dict['C710u']\
                                      + FT0dn*c3mu_dict['C710d']\
                                      + FT0sn*c3mu_dict['C710s']),
            'cNR11n' : - mN/DM_mass * (FSun*c3mu_dict['C76u']\
                                       + FSdn*c3mu_dict['C76d']\
                                       + FSsn*c3mu_dict['C76s'])\
                       - mN/DM_mass * FGn * c3mu_dict['C72']\
                       + 2*((FT0un-FT1un)*c3mu_dict['C710u']\
                            + (FT0dn-FT1dn)*c3mu_dict['C710d']\
                            + (FT0sn-FT1sn)*c3mu_dict['C710s'])\
                       - 2*mN * (  F1un*(c3mu_dict['C716u']+c3mu_dict['C720u'])\
                                 + F1dn*(c3mu_dict['C716d']+c3mu_dict['C720d'])\
                                 + F1sn*(c3mu_dict['C716s']+c3mu_dict['C720s'])),
            'cNR12n' : -8*(FT0un*c3mu_dict['C710u'] + FT0dn*c3mu_dict['C710d'] + FT0sn*c3mu_dict['C710s']),
    
            'cNR13n' : mN/DM_mass * (FPun_pion*c3mu_dict['C78u'] + FPdn_pion*c3mu_dict['C78d'])\
                       + FPpun_pion*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdn_pion*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpun_pion*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                       + FPpdn_pion*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                       + FPpun_pion*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                       + FPpdn_pion*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                       + FPpun_pion*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                       + FPpdn_pion*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64tau'] * c3mu_dict['D62dtau']),
            'cNR14n' : mN/DM_mass * (FPun_eta*c3mu_dict['C78u']\
                                     + FPdn_eta*c3mu_dict['C78d']\
                                     + FPsn_eta*c3mu_dict['C78s'])\
                       + FPpun_eta*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdn_eta*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpsn_eta*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
                       + FPpun_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                       + FPpdn_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                       + FPpsn_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62se'])\
                       + FPpun_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                       + FPpdn_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                       + FPpsn_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62smu'])\
                       + FPpun_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                       + FPpdn_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62dtau'])\
                       + FPpsn_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62stau'])\
                       + 4*mN * (  FAun*(c3mu_dict['C718u']+c3mu_dict['C722u'])\
                                 + FAdn*(c3mu_dict['C718d']+c3mu_dict['C722d'])\
                                 + FAsn*(c3mu_dict['C718s']+c3mu_dict['C722s'])),
            'cNR15n' : mN/DM_mass * FGtilden_pion * c3mu_dict['C74'],
            'cNR16n' : mN/DM_mass * FGtilden_eta * c3mu_dict['C74'],
    
            'cNR17n' : FPun_pion*c3mu_dict['C77u'] + FPdn_pion*c3mu_dict['C77d'],
            'cNR18n' : FPun_eta*c3mu_dict['C77u'] + FPdn_eta*c3mu_dict['C77d'] + FPsn_eta*c3mu_dict['C77s'],
            'cNR19n' : FGtilden_pion * c3mu_dict['C73'],
            'cNR20n' : FGtilden_eta * c3mu_dict['C73'],
    
            'cNR21n' : 0,
            'cNR22n' : -mN**2 * (- 2*alpha/np.pi * self.ip['mun']/mN * c3mu_dict['C51']),
            'cNR23n' : 0,

            'cNR100n' : (F1un*c3mu_dict['C719u'] + F1dn*c3mu_dict['C719d'] + F1sn*c3mu_dict['C719s'])/(2*DM_mass),
            'cNR104n' : 2*((F1un+F2un)*c3mu_dict['C719u']\
                           + (F1dn+F2dn)*c3mu_dict['C719d']\
                           + (F1sn+F2dn)*c3mu_dict['C719s'])/mN
            }

            if NLO:
                my_cNR_dict['cNR5p'] = - 2*mN * (F1un*c3mu_dict['C719u']\
                                                 + F1dn*c3mu_dict['C719d']\
                                                 + F1sn*c3mu_dict['C719s'])\
                                       + 2*((FT0up-FT1up)*c3mu_dict['C79u']\
                                            + (FT0dp-FT1dp)*c3mu_dict['C79d']\
                                            + (FT0sp-FT1sp)*c3mu_dict['C79s'])
                my_cNR_dict['cNR100p'] = - ((FT0up-FT1up)*c3mu_dict['C79u']\
                                            + (FT0dp-FT1dp)*c3mu_dict['C79d']\
                                            + (FT0sp-FT1sp)*c3mu_dict['C79s'])/(2*DM_mass*mN)
                my_cNR_dict['cNR5n'] = - 2*mN * (F1un*c3mu_dict['C719u']\
                                                 + F1dn*c3mu_dict['C719d'] + F1sn*c3mu_dict['C719s'])\
                                       + 2*((FT0un-FT1un)*c3mu_dict['C79u']\
                                            + (FT0dn-FT1dn)*c3mu_dict['C79d']\
                                            + (FT0sn-FT1sn)*c3mu_dict['C79s'])
                my_cNR_dict['cNR100n'] = - ((FT0un-FT1un)*c3mu_dict['C79u']\
                                            + (FT0dn-FT1dn)*c3mu_dict['C79d']\
                                            + (FT0sn-FT1sn)*c3mu_dict['C79s'])/(2*DM_mass*mN)


        if self.DM_type == "M":
            my_cNR_dict = {
            'cNR1p' : FGp*c3mu_dict['C71']\
                      + FSup*c3mu_dict['C75u'] + FSdp*c3mu_dict['C75d'] + FSsp*c3mu_dict['C75s']\
                      + 2*DM_mass * (F1up*c3mu_dict['C715u'] + F1dp*c3mu_dict['C715d'] + F1sp*c3mu_dict['C715s']),
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : - 4*(  FAup*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                           + FAdp*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                           + FAsp*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
                           + FAup*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                           + FAdp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                           + FAsp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62se'])\
                           + FAup*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                           + FAdp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                           + FAsp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62smu'])\
                           + FAup*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                           + FAdp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62dtau'])\
                           + FAsp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62stau'])),
            'cNR5p' : 0,
            'cNR6p' : mN/DM_mass * FGtildep * c3mu_dict['C74'],
            'cNR7p' : - 4*DM_mass * (FAup*c3mu_dict['C717u'] + FAdp*c3mu_dict['C717d'] + FAsp*c3mu_dict['C717s']),
            'cNR8p' : 2*(  F1up*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + F1dp*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + F1up*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64e'] * c3mu_dict['D63eu'])\
                         + F1dp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64e'] * c3mu_dict['D63ed'])\
                         + F1up*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64mu'] * c3mu_dict['D63muu'])\
                         + F1dp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64mu'] * c3mu_dict['D63mud'])\
                         + F1up*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64tau'] * c3mu_dict['D63tauu'])\
                         + F1dp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64tau'] * c3mu_dict['D63taud'])),
            'cNR9p' : 2*(  (F1up+F2up)*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + (F1dp+F2dp)*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + (F1sp+F2sp)*(c3mu_dict['C62s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C82s'])\
                         + (F1up+F2up)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63eu'])\
                         + (F1dp+F2dp)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63ed'])\
                         + (F1sp+F2sp)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63es'])\
                         + (F1up+F2up)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63muu'])\
                         + (F1dp+F2dp)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63mud'])\
                         + (F1sp+F2sp)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63mus'])\
                         + (F1up+F2up)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63tauu'])\
                         + (F1dp+F2dp)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63taud'])\
                         + (F1sp+F2sp)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63taus'])),
            'cNR10p' : FGtildep * c3mu_dict['C73'],
            'cNR11p' : - mN/DM_mass * (FSup*c3mu_dict['C76u']\
                                       + FSdp*c3mu_dict['C76d']\
                                       + FSsp*c3mu_dict['C76s'])\
                       - mN/DM_mass * FGp * c3mu_dict['C72']\
                       - 2*mN * (  F1up*c3mu_dict['C716u']\
                                   + F1dp*c3mu_dict['C716d']\
                                   + F1sp*c3mu_dict['C716s']),
            'cNR12p' : 0,
    
            'cNR13p' : mN/DM_mass * (FPup_pion*c3mu_dict['C78u'] + FPdp_pion*c3mu_dict['C78d'])\
                       + FPpup_pion*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdp_pion*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpup_pion*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                       + FPpdp_pion*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                       + FPpup_pion*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                       + FPpdp_pion*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                       + FPpup_pion*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                       + FPpdp_pion*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64tau'] * c3mu_dict['D62dtau']),
            'cNR14p' : mN/DM_mass * (FPup_eta*c3mu_dict['C78u']\
                                     + FPdp_eta*c3mu_dict['C78d']\
                                     + FPsp_eta*c3mu_dict['C78s'])\
                       + FPpup_eta*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdp_eta*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpsp_eta*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
                       + FPpup_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                       + FPpdp_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                       + FPpsp_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62se'])\
                       + FPpup_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                       + FPpdp_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                       + FPpsp_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62smu'])\
                       + FPpup_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                       + FPpdp_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62dtau'])\
                       + FPpsp_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62stau'])\
                       + 4*mN * (FAup*c3mu_dict['C718u']\
                                 + FAdp*c3mu_dict['C718d']\
                                 + FAsp*c3mu_dict['C718s']),
            'cNR15p' : mN/DM_mass * FGtildep_pion * c3mu_dict['C74'],
            'cNR16p' : mN/DM_mass * FGtildep_eta * c3mu_dict['C74'],
    
            'cNR17p' : FPup_pion*c3mu_dict['C77u'] + FPdp_pion*c3mu_dict['C77d'],
            'cNR18p' : FPup_eta*c3mu_dict['C77u'] + FPdp_eta*c3mu_dict['C77d'] + FPsp_eta*c3mu_dict['C77s'],
            'cNR19p' : FGtildep_pion * c3mu_dict['C73'],
            'cNR20p' : FGtildep_eta * c3mu_dict['C73'],
    
            'cNR21p' : 0,
            'cNR22p' : 0,
            'cNR23p' : 0,

            'cNR100p' : 0,
            'cNR104p' : 0,




            'cNR1n' :   FGn*c3mu_dict['C71']\
                      + FSun*c3mu_dict['C75u'] + FSdn*c3mu_dict['C75d'] + FSsn*c3mu_dict['C75s']\
                      + 2*DM_mass * (F1un*c3mu_dict['C715u'] + F1dn*c3mu_dict['C715d'] + F1sn*c3mu_dict['C715s']),
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : - 4*(  FAun*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                           + FAdn*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                           + FAsn*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
                           + FAun*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                           + FAdn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                           + FAsn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64e'] * c3mu_dict['D62se'])\
                           + FAun*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                           + FAdn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                           + FAsn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64mu'] * c3mu_dict['D62smu'])\
                           + FAun*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                           + FAdn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62dtau'])\
                           + FAsn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                   * c3mu_dict['C64tau'] * c3mu_dict['D62stau'])),
            'cNR5n' : 0,
            'cNR6n' : mN/DM_mass * FGtilden * c3mu_dict['C74'],
            'cNR7n' : - 4*DM_mass * (FAun*c3mu_dict['C717u'] + FAdn*c3mu_dict['C717d'] + FAsn*c3mu_dict['C717s']),
            'cNR8n' : 2*(  F1un*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + F1dn*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + F1un*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64e'] * c3mu_dict['D63eu'])\
                         + F1dn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64e'] * c3mu_dict['D63ed'])\
                         + F1un*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64mu'] * c3mu_dict['D63muu'])\
                         + F1dn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64mu'] * c3mu_dict['D63mud'])\
                         + F1un*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64tau'] * c3mu_dict['D63tauu'])\
                         + F1dn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                 * c3mu_dict['C64tau'] * c3mu_dict['D63taud'])),
            'cNR9n' : 2*(  (F1un+F2un)*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + (F1dn+F2dn)*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + (F1sn+F2sn)*(c3mu_dict['C62s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C82s'])\
                         + (F1un+F2un)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63eu'])\
                         + (F1dn+F2dn)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63ed'])\
                         + (F1sn+F2sn)*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64e'] * c3mu_dict['D63es'])\
                         + (F1un+F2un)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63muu'])\
                         + (F1dn+F2dn)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63mud'])\
                         + (F1sn+F2sn)*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64mu'] * c3mu_dict['D63mus'])\
                         + (F1un+F2up)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63tauu'])\
                         + (F1dn+F2dp)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63taud'])\
                         + (F1sp+F2sp)*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                        * c3mu_dict['C64tau'] * c3mu_dict['D63taus'])),
            'cNR10n' : FGtilden * c3mu_dict['C73'],
            'cNR11n' : - mN/DM_mass * (FSun*c3mu_dict['C76u']\
                                       + FSdn*c3mu_dict['C76d']\
                                       + FSsn*c3mu_dict['C76s'])\
                       - mN/DM_mass * FGn * c3mu_dict['C72']\
                       - 2*mN * (  F1un*c3mu_dict['C716u']\
                                   + F1dn*c3mu_dict['C716d']\
                                   + F1sn*c3mu_dict['C716s']),
            'cNR12n' : 0,
    
            'cNR13n' : mN/DM_mass * (FPun_pion*c3mu_dict['C78u'] + FPdn_pion*c3mu_dict['C78d'])\
                       + FPpun_pion*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdn_pion*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpun_pion*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                       + FPpdn_pion*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                       + FPpun_pion*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                       + FPpdn_pion*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                       + FPpun_pion*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                       + FPpdn_pion*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                     * c3mu_dict['C64tau'] * c3mu_dict['D62dtau']),
            'cNR14n' : mN/DM_mass * (FPun_eta*c3mu_dict['C78u']\
                                     + FPdn_eta*c3mu_dict['C78d']\
                                     + FPsn_eta*c3mu_dict['C78s'])\
                       + FPpun_eta*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdn_eta*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpsn_eta*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
                       + FPpun_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62ue'])\
                       + FPpdn_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62de'])\
                       + FPpsn_eta*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64e'] * c3mu_dict['D62se'])\
                       + FPpun_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62umu'])\
                       + FPpdn_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62dmu'])\
                       + FPpsn_eta*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64mu'] * c3mu_dict['D62smu'])\
                       + FPpun_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62utau'])\
                       + FPpdn_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62dtau'])\
                       + FPpsn_eta*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                    * c3mu_dict['C64tau'] * c3mu_dict['D62stau'])\
                       + 4*mN * (FAun*c3mu_dict['C718u']\
                                 + FAdn*c3mu_dict['C718d']\
                                 + FAsn*c3mu_dict['C718s']),
            'cNR15n' : mN/DM_mass * FGtilden_pion * c3mu_dict['C74'],
            'cNR16n' : mN/DM_mass * FGtilden_eta * c3mu_dict['C74'],
    
            'cNR17n' : FPun_pion*c3mu_dict['C77u'] + FPdn_pion*c3mu_dict['C77d'],
            'cNR18n' : FPun_eta*c3mu_dict['C77u'] + FPdn_eta*c3mu_dict['C77d'] + FPsn_eta*c3mu_dict['C77s'],
            'cNR19n' : FGtilden_pion * c3mu_dict['C73'],
            'cNR20n' : FGtilden_eta * c3mu_dict['C73'],
    
            'cNR21n' : 0,
            'cNR22n' : 0,
            'cNR23n' : 0,

            'cNR100n' : 0,
            'cNR104n' : 0
            }


        if self.DM_type == "C":
            my_cNR_dict = {
            'cNR1p' :   2*DM_mass*(  F1up * (c3mu_dict['C61u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C81u'])\
                                   + F1dp * (c3mu_dict['C61d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C81d'])\
                                   + F1up*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62e'] * c3mu_dict['D63eu'])\
                                   + F1dp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62e'] * c3mu_dict['D63ed'])\
                                   + F1up*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62mu'] * c3mu_dict['D63muu'])\
                                   + F1dp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62mu'] * c3mu_dict['D63mud'])\
                                   + F1up*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62tau'] * c3mu_dict['D63tauu'])\
                                   + F1dp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62tau'] * c3mu_dict['D63taud']))\
                       + FGp*c3mu_dict['C65']\
                      + FSup*c3mu_dict['C63u'] + FSdp*c3mu_dict['C63d'] + FSsp*c3mu_dict['C63s'],
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : 0,
            'cNR5p' : 0,
            'cNR6p' : 0,
            'cNR7p' : -4*DM_mass*(  FAup * (c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                                  + FAdp * (c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                                  + FAsp * (c3mu_dict['C62s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C82s'])\
                                  + FAup*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62e'] * c3mu_dict['D62ue'])\
                                  + FAdp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62e'] * c3mu_dict['D62de'])\
                                  + FAsp*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62e'] * c3mu_dict['D62se'])\
                                  + FAup*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62mu'] * c3mu_dict['D62umu'])\
                                  + FAdp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62mu'] * c3mu_dict['D62dmu'])\
                                  + FAsp*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62mu'] * c3mu_dict['D62smu'])\
                                  + FAup*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62tau'] * c3mu_dict['D62utau'])\
                                  + FAdp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62tau'] * c3mu_dict['D62dtau'])\
                                  + FAsp*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62tau'] * c3mu_dict['D62stau'])),
            'cNR8p' : 0,
            'cNR9p' : 0,
            'cNR10p' : FGtildep * c3mu_dict['C66'],
            'cNR11p' : 0,
            'cNR12p' : 0,

            'cNR13p' : 0,
            'cNR14p' : 0,
            'cNR15p' : 0,
            'cNR16p' : 0,
    
            'cNR17p' : FPup_pion*c3mu_dict['C64u'] + FPdp_pion*c3mu_dict['C64d'],
            'cNR18p' : FPup_eta*c3mu_dict['C64u'] + FPdp_eta*c3mu_dict['C64d'] + FPsp_eta*c3mu_dict['C64s'],
            'cNR19p' : FGtildep_pion * c3mu_dict['C66'],
            'cNR20p' : FGtildep_eta * c3mu_dict['C66'],
    
            'cNR21p' : 0,
            'cNR22p' : 0,
            'cNR23p' : 0,

            'cNR100p' : 0,
            'cNR104p' : 0,




            'cNR1n' :   2*DM_mass*(  F1un * (c3mu_dict['C61u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C81u'])\
                                   + F1dn * (c3mu_dict['C61d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C81d'])\
                                   + F1un*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62e'] * c3mu_dict['D63eu'])\
                                   + F1dn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62e'] * c3mu_dict['D63ed'])\
                                   + F1un*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62mu'] * c3mu_dict['D63muu'])\
                                   + F1dn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62mu'] * c3mu_dict['D63mud'])\
                                   + F1un*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62tau'] * c3mu_dict['D63tauu'])\
                                   + F1dn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                           * c3mu_dict['C62tau'] * c3mu_dict['D63taud']))\
                      + FGn*c3mu_dict['C65']\
                      + FSun*c3mu_dict['C63u'] + FSdn*c3mu_dict['C63d'] + FSsn*c3mu_dict['C63s'],
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : 0,
            'cNR5n' : 0,
            'cNR6n' : 0,
            'cNR7n' : -4*DM_mass*(  FAun * (c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                                  + FAdn * (c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                                  + FAsn * (c3mu_dict['C62s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C82s'])\
                                  + FAun*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62e'] * c3mu_dict['D62ue'])\
                                  + FAdn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62e'] * c3mu_dict['D62de'])\
                                  + FAsn*(np.sqrt(2)*GF/np.pi**2 * me**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62e'] * c3mu_dict['D62se'])\
                                  + FAun*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62mu'] * c3mu_dict['D62umu'])\
                                  + FAdn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62mu'] * c3mu_dict['D62dmu'])\
                                  + FAsn*(np.sqrt(2)*GF/np.pi**2 * mmu**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62mu'] * c3mu_dict['D62smu'])\
                                  + FAun*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62tau'] * c3mu_dict['D62utau'])\
                                  + FAdn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62tau'] * c3mu_dict['D62dtau'])\
                                  + FAsn*(np.sqrt(2)*GF/np.pi**2 * mtau**2 * np.log(2/MZ)\
                                          * c3mu_dict['C62tau'] * c3mu_dict['D62stau'])),
            'cNR8n' : 0,
            'cNR9n' : 0,
            'cNR10n' : FGtilden * c3mu_dict['C66'],
            'cNR11n' : 0,
            'cNR12n' : 0,

            'cNR13n' : 0,
            'cNR14n' : 0,
            'cNR15n' : 0,
            'cNR16n' : 0,
    
            'cNR17n' : FPun_pion*c3mu_dict['C64u'] + FPdn_pion*c3mu_dict['C64d'],
            'cNR18n' : FPun_eta*c3mu_dict['C64u'] + FPdn_eta*c3mu_dict['C64d'] + FPsn_eta*c3mu_dict['C64s'],
            'cNR19n' : FGtilden_pion * c3mu_dict['C66'],
            'cNR20n' : FGtilden_eta * c3mu_dict['C66'],
    
            'cNR21n' : 0,
            'cNR22n' : 0,
            'cNR23n' : 0,

            'cNR100n' : 0,
            'cNR104n' : 0
            }


        if self.DM_type == "R":
            my_cNR_dict = {
            'cNR1p' :   FSup*c3mu_dict['C63u']\
                      + FSdp*c3mu_dict['C63d']\
                      + FSsp*c3mu_dict['C63s']\
                      + FGp*c3mu_dict['C65'],
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : 0,
            'cNR5p' : 0,
            'cNR6p' : 0,
            'cNR7p' : 0,
            'cNR8p' : 0,
            'cNR9p' : 0,
            'cNR10p' : FGtildep * c3mu_dict['C66'],
            'cNR11p' : 0,
            'cNR12p' : 0,

            'cNR13p' : 0,
            'cNR14p' : 0,
            'cNR15p' : 0,
            'cNR16p' : 0,
    
            'cNR17p' : FPup_pion*c3mu_dict['C64u'] + FPdp_pion*c3mu_dict['C64d'],
            'cNR18p' :   FPup_eta*c3mu_dict['C64u']\
                       + FPdp_eta*c3mu_dict['C64d']\
                       + FPsp_eta*c3mu_dict['C64s'],
            'cNR19p' : FGtildep_pion * c3mu_dict['C66'],
            'cNR20p' : FGtildep_eta * c3mu_dict['C66'],
    
            'cNR21p' : 0,
            'cNR22p' : 0,
            'cNR23p' : 0,

            'cNR100p' : 0,
            'cNR104p' : 0,




            'cNR1n' :   FSun*c3mu_dict['C63u']\
                      + FSdn*c3mu_dict['C63d']\
                      + FSsn*c3mu_dict['C63s']\
                      + FGn*c3mu_dict['C65'],
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : 0,
            'cNR5n' : 0,
            'cNR6n' : 0,
            'cNR7n' : 0,
            'cNR8n' : 0,
            'cNR9n' : 0,
            'cNR10n' : FGtilden * c3mu_dict['C66'],
            'cNR11n' : 0,
            'cNR12n' : 0,

            'cNR13n' : 0,
            'cNR14n' : 0,
            'cNR15n' : 0,
            'cNR16n' : 0,
    
            'cNR17n' : FPun_pion*c3mu_dict['C64u'] + FPdn_pion*c3mu_dict['C64d'],
            'cNR18n' :   FPun_eta*c3mu_dict['C64u']\
                       + FPdn_eta*c3mu_dict['C64d']\
                       + FPsn_eta*c3mu_dict['C64s'],
            'cNR19n' : FGtilden_pion * c3mu_dict['C66'],
            'cNR20n' : FGtilden_eta * c3mu_dict['C66'],
    
            'cNR21n' : 0,
            'cNR22n' : 0,
            'cNR23n' : 0,

            'cNR100n' : 0,
            'cNR104n' : 0
            }


        return my_cNR_dict


    def cNR(self, DM_mass, q, RGE=None, NLO=None):
        """ The operator coefficients of O_1^N -- O_12^N as in 1308.6288 

        (multiply by propagators and sum up contributions)

        DM_mass is the DM mass in GeV

        RGE is an optional argument to turn RGE running on (True) or off (False). (Default True)

        If NLO is set to True, the coherently enhanced NLO terms for Q_9^(7) are added. (Default False)

        Returns a dictionary of coefficients for the NR Lagrangian, 
        cNR1 -- cNR12, as in 1308.6288

        The possible names are

        ['cNR1p', 'cNR1n', 'cNR2p', 'cNR2n', 'cNR3p', 'cNR3n', 'cNR4p', 'cNR4n', 'cNR5p', 'cNR5n',
         'cNR6p', 'cNR6n', 'cNR7p', 'cNR7n', 'cNR8p', 'cNR8n', 'cNR9p', 'cNR9n', 'cNR10p', 'cNR10n',
         'cNR11p', 'cNR11n', 'cNR12p', 'cNR12n']
        """
        if RGE is None:
            RGE = True
        if NLO is None:
            NLO = False

        meta = self.ip['meta']
        mpi = self.ip['mpi0']

        qsq = q**2

        # The traditional coefficients, where different from above
        cNR_dict = {}
        my_cNR = self._my_cNR(DM_mass, RGE, NLO)

        # Add meson- / photon-pole contributions
        cNR_dict['cNR1p'] = my_cNR['cNR1p'] + qsq * my_cNR['cNR100p']
        cNR_dict['cNR2p'] = my_cNR['cNR2p']
        cNR_dict['cNR3p'] = my_cNR['cNR3p']
        cNR_dict['cNR4p'] = my_cNR['cNR4p'] + qsq * my_cNR['cNR104p']
        cNR_dict['cNR5p'] = my_cNR['cNR5p'] + 1/qsq * my_cNR['cNR21p']
        cNR_dict['cNR6p'] = my_cNR['cNR6p']\
                            + 1/(mpi**2 + qsq) * my_cNR['cNR13p']\
                            + 1/(meta**2 + qsq) * my_cNR['cNR14p']\
                            + qsq/(mpi**2 + qsq) * my_cNR['cNR15p']\
                            + qsq/(meta**2 + qsq) * my_cNR['cNR16p']\
                            + 1/qsq * my_cNR['cNR22p']
        cNR_dict['cNR7p'] = my_cNR['cNR7p']
        cNR_dict['cNR8p'] = my_cNR['cNR8p']
        cNR_dict['cNR9p'] = my_cNR['cNR9p']
        cNR_dict['cNR10p'] = my_cNR['cNR10p']\
                             + 1/(mpi**2 + qsq) * my_cNR['cNR17p']\
                             + 1/(meta**2 + qsq) * my_cNR['cNR18p']\
                             + qsq/(mpi**2 + qsq) * my_cNR['cNR19p']\
                             + qsq/(meta**2 + qsq) * my_cNR['cNR20p']
        cNR_dict['cNR11p'] = my_cNR['cNR11p'] + 1/qsq * my_cNR['cNR23p']
        cNR_dict['cNR12p'] = my_cNR['cNR12p']

        cNR_dict['cNR1n'] = my_cNR['cNR1n'] + qsq * my_cNR['cNR100n']
        cNR_dict['cNR2n'] = my_cNR['cNR2n']
        cNR_dict['cNR3n'] = my_cNR['cNR3n']
        cNR_dict['cNR4n'] = my_cNR['cNR4n'] + qsq * my_cNR['cNR104n']
        cNR_dict['cNR5n'] = my_cNR['cNR5n'] + 1/qsq * my_cNR['cNR21n']
        cNR_dict['cNR6n'] = my_cNR['cNR6n']\
                            + 1/(mpi**2 + qsq) * my_cNR['cNR13n']\
                            + 1/(meta**2 + qsq) * my_cNR['cNR14n']\
                            + qsq/(mpi**2 + qsq) * my_cNR['cNR15n']\
                            + qsq/(meta**2 + qsq) * my_cNR['cNR16n']\
                            + 1/qsq * my_cNR['cNR22n']
        cNR_dict['cNR7n'] = my_cNR['cNR7n']
        cNR_dict['cNR8n'] = my_cNR['cNR8n']
        cNR_dict['cNR9n'] = my_cNR['cNR9n']
        cNR_dict['cNR10n'] = my_cNR['cNR10n']\
                             + 1/(mpi**2 + qsq) * my_cNR['cNR17n']\
                             + 1/(meta**2 + qsq) * my_cNR['cNR18n']\
                             + qsq/(mpi**2 + qsq) * my_cNR['cNR19n']\
                             + qsq/(meta**2 + qsq) * my_cNR['cNR20n']
        cNR_dict['cNR11n'] = my_cNR['cNR11n'] + 1/qsq * my_cNR['cNR23n']
        cNR_dict['cNR12n'] = my_cNR['cNR12n']

        return cNR_dict


    def write_mma(self, DM_mass, qvec, RGE=None, NLO=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass DM_mass (in GeV) and the spatial momentum transfer qvec (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is './')

        <filename> is the filename (default 'cNR.m')
        """
        if RGE is None:
            RGE=True
        if NLO is None:
            NLO=False
        if path is None:
            path = './'
        assert type(path) is str
        if path.endswith('/'):
            pass
        else:
            path += '/'
        if filename is None:
            filename = 'cNR.m'

        val = self.cNR(DM_mass, qvec, RGE, NLO)
        self.cNR_list_mma = '{' + str(val['cNR1p']) + ', '\
                            + str(val['cNR2p']) + ', '\
                            + str(val['cNR3p']) + ', '\
                            + str(val['cNR4p']) + ', '\
                            + str(val['cNR5p']) + ', '\
                            + str(val['cNR6p']) + ', '\
                            + str(val['cNR7p']) + ', '\
                            + str(val['cNR8p']) + ', '\
                            + str(val['cNR9p']) + ', '\
                            + str(val['cNR10p']) + ', '\
                            + str(val['cNR11p']) + ', '\
                            + str(val['cNR12p']) + ', '\
                            + str(val['cNR1n']) + ', '\
                            + str(val['cNR2n']) + ', '\
                            + str(val['cNR3n']) + ', '\
                            + str(val['cNR4n']) + ', '\
                            + str(val['cNR5n']) + ', '\
                            + str(val['cNR6n']) + ', '\
                            + str(val['cNR7n']) + ', '\
                            + str(val['cNR8n']) + ', '\
                            + str(val['cNR9n']) + ', '\
                            + str(val['cNR10n']) + ', '\
                            + str(val['cNR11n']) + ', '\
                            + str(val['cNR12n']) + '}' + '\n'

        output_file = str(os.path.expanduser(path)) + filename

        with open(output_file,'w') as f:
            f.write(self.cNR_list_mma)



class WC_4flavor(object):
    def __init__(self, coeff_dict, DM_type, input_dict):
        """ Class for Wilson coefficients in 4 flavor QCD x QED plus DM.

        The argument should be a dictionary for the initial conditions
        of the 2 + 28 + 4 + 42 + 4 + 56 + 6 = 142 dimension-five to dimension-eight
        four-flavor-QCD Wilson coefficients (for Dirac DM) of the form
        {'C51' : value, 'C52' : value, ...}. For other DM types there are less coefficients.
        An arbitrary number of them can be given; the default values are zero. 

        The second argument is the DM type; it can take the following values: 
            "D" (Dirac fermion)
            "M" (Majorana fermion)
            "C" (Complex scalar)
            "R" (Real scalar)

        The possible name are (with an hopefully obvious notation):

        Dirac fermion:       'C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61c', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C75c', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C76c', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C77c', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C78c', 'C78e', 'C78mu', 'C78tau',
                             'C79u', 'C79d', 'C79s', 'C79c', 'C79e', 'C79mu', 'C79tau', 
                             'C710u', 'C710d', 'C710s', 'C710c', 'C710e', 'C710mu', 'C710tau',
                             'C711', 'C712', 'C713', 'C714',
                             'C715u', 'C715d', 'C715s', 'C715c', 'C715e', 'C715mu', 'C715tau', 
                             'C716u', 'C716d', 'C716s', 'C716c', 'C716e', 'C716mu', 'C716tau',
                             'C717u', 'C717d', 'C717s', 'C717c', 'C717e', 'C717mu', 'C717tau', 
                             'C718u', 'C718d', 'C718s', 'C718c', 'C718e', 'C718mu', 'C718tau',
                             'C719u', 'C719d', 'C719s', 'C719c', 'C719e', 'C719mu', 'C719tau', 
                             'C720u', 'C720d', 'C720s', 'C720c', 'C720e', 'C720mu', 'C720tau', 
                             'C721u', 'C721d', 'C721s', 'C721c', 'C721e', 'C721mu', 'C721tau', 
                             'C722u', 'C722d', 'C722s', 'C722c', 'C722e', 'C722mu', 'C722tau' 
                             'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s'


        Majorana fermion:    'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C75c', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C76c', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C77c', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C78c', 'C78e', 'C78mu', 'C78tau',
                             'C711', 'C712', 'C713', 'C714',
                             'C715u', 'C715d', 'C715s', 'C715c', 'C715e', 'C715mu', 'C715tau', 
                             'C716u', 'C716d', 'C716s', 'C716c', 'C716e', 'C716mu', 'C716tau',
                             'C717u', 'C717d', 'C717s', 'C717c', 'C717e', 'C717mu', 'C717tau', 
                             'C718u', 'C718d', 'C718s', 'C718c', 'C718e', 'C718mu', 'C718tau',

        Complex Scalar:      'C61u', 'C61d', 'C61s', 'C61c', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                             'C65', 'C66',
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                             'C67', 'C68'

        Real Scalar:         'C65', 'C66'
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                             'C67', 'C68'


        (the notation corresponds to the numbering in 1707.06998).
        The Wilson coefficients should be specified in the MS-bar scheme at mb = 4.18 GeV.


        In order to calculate consistently to dim.8 in the EFT, we need also the dim.6 SM operators. 
        The following subset of 6*8 + 4*4 = 64 operators is sufficient for our purposes:

         'D61ud', 'D62ud', 'D63ud', 'D63du', 'D64ud', 'D65ud', 'D66ud', 'D66du', 
         'D61us', 'D62us', 'D63us', 'D63su', 'D64us', 'D65us', 'D66us', 'D66su', 
         'D61uc', 'D62uc', 'D63uc', 'D63cu', 'D64uc', 'D65uc', 'D66uc', 'D66cu', 
         'D61ds', 'D62ds', 'D63ds', 'D63sd', 'D64ds', 'D65ds', 'D66ds', 'D66sd', 
         'D61dc', 'D62dc', 'D63dc', 'D63cd', 'D64dc', 'D65dc', 'D66dc', 'D66cd', 
         'D61sc', 'D62sc', 'D63sc', 'D63cs', 'D64sc', 'D65sc', 'D66sc', 'D66cs', 
         'D61u', 'D62u', 'D63u', 'D64u', 
         'D61d', 'D62d', 'D63d', 'D64d', 
         'D61s', 'D62s', 'D63s', 'D64s', 
         'D61c', 'D62c', 'D63c', 'D64c' 

        The initial conditions at scale mb have to given; e.g. using WC_5f

        The third argument is a dictionary with all input parameters.


        The class has four methods: 

        run
        ---
        Run the Wilson from mb(mb) to mu_low [GeV; default 2 GeV], with 4 active quark flavors

        match
        -----
        Match the Wilson coefficients from 4-flavor to 3-flavor QCD, at scale mu [GeV; default 2 GeV]

        cNR
        ---
        Calculate the cNR coefficients as defined in 1308.6288

        It has two mandatory arguments: The DM mass in GeV and the momentum transfer in GeV


        write_mma
        ---------
        Writes an output file that can be loaded into mathematica, 
        to be used in the DMFormFactor package [1308.6288].
        """
        self.DM_type = DM_type


        # First, we define a standard ordering for the Wilson coefficients, so that we can use arrays

        self.sm_name_list = ['D61ud', 'D62ud', 'D63ud', 'D63du', 'D64ud', 'D65ud', 'D66ud', 'D66du', 
                             'D61us', 'D62us', 'D63us', 'D63su', 'D64us', 'D65us', 'D66us', 'D66su', 
                             'D61uc', 'D62uc', 'D63uc', 'D63cu', 'D64uc', 'D65uc', 'D66uc', 'D66cu', 
                             'D61ds', 'D62ds', 'D63ds', 'D63sd', 'D64ds', 'D65ds', 'D66ds', 'D66sd', 
                             'D61dc', 'D62dc', 'D63dc', 'D63cd', 'D64dc', 'D65dc', 'D66dc', 'D66cd', 
                             'D61sc', 'D62sc', 'D63sc', 'D63cs', 'D64sc', 'D65sc', 'D66sc', 'D66cs', 
                             'D61u', 'D62u', 'D63u', 'D64u', 
                             'D61d', 'D62d', 'D63d', 'D64d', 
                             'D61s', 'D62s', 'D63s', 'D64s', 
                             'D61c', 'D62c', 'D63c', 'D64c']

        self.sm_lepton_name_list = ['D63eu', 'D63muu', 'D63tauu', 'D63ed', 'D63mud',\
                                    'D63taud', 'D63es', 'D63mus', 'D63taus',
                                    'D62ue', 'D62umu', 'D62utau', 'D62de', 'D62dmu',\
                                    'D62dtau', 'D62se', 'D62smu', 'D62stau']

        if self.DM_type == "D":
            self.wc_name_list = ['C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61c', 'C61e', 'C61mu', 'C61tau', 
                                 'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75c', 'C75e', 'C75mu', 'C75tau', 
                                 'C76u', 'C76d', 'C76s', 'C76c', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77c', 'C77e', 'C77mu', 'C77tau', 
                                 'C78u', 'C78d', 'C78s', 'C78c', 'C78e', 'C78mu', 'C78tau',
                                 'C79u', 'C79d', 'C79s', 'C79c', 'C79e', 'C79mu', 'C79tau', 
                                 'C710u', 'C710d', 'C710s', 'C710c', 'C710e', 'C710mu', 'C710tau',
                                 'C711', 'C712', 'C713', 'C714',
                                 'C715u', 'C715d', 'C715s', 'C715c', 'C715e', 'C715mu', 'C715tau', 
                                 'C716u', 'C716d', 'C716s', 'C716c', 'C716e', 'C716mu', 'C716tau',
                                 'C717u', 'C717d', 'C717s', 'C717c', 'C717e', 'C717mu', 'C717tau', 
                                 'C718u', 'C718d', 'C718s', 'C718c', 'C718e', 'C718mu', 'C718tau',
                                 'C719u', 'C719d', 'C719s', 'C719c', 'C719e', 'C719mu', 'C719tau', 
                                 'C720u', 'C720d', 'C720s', 'C720c', 'C720e', 'C720mu', 'C720tau', 
                                 'C721u', 'C721d', 'C721s', 'C721c', 'C721e', 'C721mu', 'C721tau', 
                                 'C722u', 'C722d', 'C722s', 'C722c', 'C722e', 'C722mu', 'C722tau']

            self.wc8_name_list = ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s',\
                                  'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s']

            # The 4-flavor list for matching only
            self.wc_name_list_3f = ['C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau',
                                    'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                    'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau',
                                    'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                    'C71', 'C72', 'C73', 'C74',
                                    'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau',
                                    'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                                    'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau',
                                    'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau',
                                    'C79u', 'C79d', 'C79s', 'C79e', 'C79mu', 'C79tau',
                                    'C710u', 'C710d', 'C710s', 'C710e', 'C710mu', 'C710tau',
                                    'C711', 'C712', 'C713', 'C714',
                                    'C715u', 'C715d', 'C715s', 'C715e', 'C715mu', 'C715tau', 
                                    'C716u', 'C716d', 'C716s', 'C716e', 'C716mu', 'C716tau',
                                    'C717u', 'C717d', 'C717s', 'C717e', 'C717mu', 'C717tau', 
                                    'C718u', 'C718d', 'C718s', 'C718e', 'C718mu', 'C718tau',
                                    'C719u', 'C719d', 'C719s', 'C719e', 'C719mu', 'C719tau', 
                                    'C720u', 'C720d', 'C720s', 'C720e', 'C720mu', 'C720tau', 
                                    'C721u', 'C721d', 'C721s', 'C721e', 'C721mu', 'C721tau', 
                                    'C722u', 'C722d', 'C722s', 'C722e', 'C722mu', 'C722tau']

        if self.DM_type == "M":
            self.wc_name_list = ['C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75c', 'C75e', 'C75mu', 'C75tau', 
                                 'C76u', 'C76d', 'C76s', 'C76c', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77c', 'C77e', 'C77mu', 'C77tau', 
                                 'C78u', 'C78d', 'C78s', 'C78c', 'C78e', 'C78mu', 'C78tau',
                                 'C711', 'C712', 'C713', 'C714',
                                 'C715u', 'C715d', 'C715s', 'C715c', 'C715e', 'C715mu', 'C715tau', 
                                 'C716u', 'C716d', 'C716s', 'C716c', 'C716e', 'C716mu', 'C716tau',
                                 'C717u', 'C717d', 'C717s', 'C717c', 'C717e', 'C717mu', 'C717tau', 
                                 'C718u', 'C718d', 'C718s', 'C718c', 'C718e', 'C718mu', 'C718tau']

            self.wc8_name_list = ['C82u', 'C82d', 'C82s', 'C84u', 'C84d', 'C84s']

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = np.r_[np.s_[0:9], np.s_[16:23], np.s_[62:76], np.s_[108:136]]
            # The list of indices to be deleted from the dim.8 ADM because of less operators
            del_ind_list_dim_8 = np.r_[np.s_[0:3], np.s_[6:9]]
            # The list of indices to be deleted from the ADT because of less operators (dim.6 part)
            del_ind_list_adt_quark = np.r_[np.s_[0:4]]

            # The 3-flavor list for matching only
            self.wc_name_list_3f = ['C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                    'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                    'C71', 'C72', 'C73', 'C74',
                                    'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau',
                                    'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                                    'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau',
                                    'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau',
                                    'C711', 'C712', 'C713', 'C714',
                                    'C715u', 'C715d', 'C715s', 'C715e', 'C715mu', 'C715tau', 
                                    'C716u', 'C716d', 'C716s', 'C716e', 'C716mu', 'C716tau',
                                    'C717u', 'C717d', 'C717s', 'C717e', 'C717mu', 'C717tau', 
                                    'C718u', 'C718d', 'C718s', 'C718e', 'C718mu', 'C718tau']

        if self.DM_type == "C":
            self.wc_name_list = ['C61u', 'C61d', 'C61s', 'C61c', 'C61e', 'C61mu', 'C61tau', 
                                 'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                 'C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                 'C67', 'C68']

            self.wc8_name_list = ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s']

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [0,1] + [i for i in range(9,16)] + [i for i in range(23,30)]\
                           + [31] + [33] + [i for i in range(41,48)]\
                           + [i for i in range(55,76)] + [77] + [79] + [i for i in range(80,136)]
            # The list of indices to be deleted from the dim.8 ADM because of less operators
            del_ind_list_dim_8 = np.r_[np.s_[0:3], np.s_[6:9]]
            # The list of indices to be deleted from the ADT because of less operators (dim.6 part)
            del_ind_list_adt_quark = np.r_[np.s_[0:4]]

            # The 3-flavor list for matching only
            self.wc_name_list_3f = ['C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 
                                    'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                    'C65', 'C66',
                                    'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                    'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                    'C67', 'C68']

        if self.DM_type == "R":
            self.wc_name_list = ['C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau',
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                 'C67', 'C68']

            self.wc8_name_list = []

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [i for i in range(0,30)] + [31] + [33] + [i for i in range(41,48)]\
                           + [i for i in range(55,76)]\
                           + [77] + [79] + [i for i in range(80,136)]

            # The 3-flavor list for matching only
            self.wc_name_list_3f = ['C65', 'C66',
                                    'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                    'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                    'C67', 'C68']


        self.coeff_dict = {}

        # Issue a user warning if a key is not defined:

        for wc_name in coeff_dict.keys():
            if wc_name in self.wc_name_list:
                pass
            elif wc_name in self.wc8_name_list:
                pass
            elif wc_name in self.sm_name_list:
                pass
            elif wc_name in self.sm_lepton_name_list:
                pass
            else:
                warnings.warn('The key ' + wc_name + ' is not a valid key. Typo?')

        # Create the dictionary. 

        for wc_name in self.wc_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.

        for wc_name in self.wc8_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.

        for wc_name in self.sm_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.

        for wc_name in self.sm_lepton_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.


        # Create the np.array of coefficients:
        self.coeff_list_dm_dim5_dim6_dim7 = np.array(dict_to_list(self.coeff_dict, self.wc_name_list))
        self.coeff_list_dm_dim8 = np.array(dict_to_list(self.coeff_dict, self.wc8_name_list))
        self.coeff_list_sm_dim6 = np.array(dict_to_list(self.coeff_dict, self.sm_name_list))
        self.coeff_list_sm_lepton_dim6 = np.array(dict_to_list(self.coeff_dict, self.sm_lepton_name_list))


        # The dictionary of input parameters
        self.ip = input_dict



        #---------------------------#
        # The anomalous dimensions: #
        #---------------------------#

        if self.DM_type == "D":
            self.gamma_QED = adm.ADM_QED(4)
            self.gamma_QED2 = adm.ADM_QED2(4)
            self.gamma_QCD = adm.ADM_QCD(4)
            self.gamma_QCD2 = adm.ADM_QCD2(4)
            self.gamma_QCD_dim8 = adm.ADM_QCD_dim8(4)
            self.gamma_hat = adm.ADT_QCD(4, self.ip)
        if self.DM_type == "M":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD_dim8 = np.delete(np.delete(adm.ADM_QCD_dim8(4), del_ind_list_dim_8, 0),\
                                            del_ind_list_dim_8, 1)
            self.gamma_hat = np.delete(np.delete(adm.ADT_QCD(4, self.ip), del_ind_list_dim_8, 0),\
                                       del_ind_list_adt_quark, 2)
        if self.DM_type == "C":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD_dim8 = np.delete(np.delete(adm.ADM_QCD_dim8(4), del_ind_list_dim_8, 0),\
                                            del_ind_list_dim_8, 1)
            self.gamma_hat = np.delete(np.delete(adm.ADT_QCD(4, self.ip), del_ind_list_dim_8, 0),\
                                       del_ind_list_adt_quark, 2)
        if self.DM_type == "R":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(4), del_ind_list, 1), del_ind_list, 2)

        self.ADM_SM = adm.ADM_SM_QCD(4)



        #------------------------------------------------------------------------------#
        # The effective anomalous dimension for mixing into dimension eight -- quarks: #
        #------------------------------------------------------------------------------#

        # We need to contract the ADT with a subset of the dim.-6 DM Wilson coefficients
        if self.DM_type == "D":
            DM_dim6_init = np.delete(self.coeff_list_dm_dim5_dim6_dim7,\
                                     np.r_[np.s_[0:16], np.s_[20:23], np.s_[27:136]])
        elif self.DM_type == "M":
            DM_dim6_init = np.delete(self.coeff_list_dm_dim5_dim6_dim7, np.r_[np.s_[0:7], np.s_[11:78]])
        elif self.DM_type == "C":
            DM_dim6_init = np.delete(self.coeff_list_dm_dim5_dim6_dim7, np.r_[np.s_[0:7], np.s_[11:32]])



        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            # The columns of ADM_eff correspond to SM6 operators; the rows of ADM_eff correspond to DM8 operators; 
            C6_dot_ADM_hat = np.transpose(np.tensordot(DM_dim6_init, self.gamma_hat, (0,2)))

            # The effective ADM
            #
            # Note that the mixing of the SM operators with four equal flavors
            # does not contribute if we neglect yu, yd, ys! 
            self.ADM_eff = [np.vstack((np.hstack((self.ADM_SM,\
                                                  np.vstack((C6_dot_ADM_hat,\
                                                             np.zeros((16, len(self.gamma_QCD_dim8))))))),\
                                       np.hstack((np.zeros((len(self.gamma_QCD_dim8),\
                                                            len(self.coeff_list_sm_dim6))),\
                                                  self.gamma_QCD_dim8))))]
        if self.DM_type == "R":
            pass



    def run(self, mu_low=None, double_QCD=None):
        """ Running of 4-flavor Wilson coefficients

        Calculate the running from mb(mb) to mu_low [GeV; default 2 GeV] in the four-flavor theory. 

        Return a dictionary of Wilson coefficients for the four-flavor Lagrangian
        at scale mu_low.
        """
        if mu_low is None:
            mu_low=2
        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            if double_QCD is None:
                double_QCD=True
        else:
            double_QCD=False


        #-------------#
        # The running #
        #-------------#

        mb = self.ip['mb_at_mb']
        alpha_at_mc = 1/self.ip['aMZinv']
        as_2GeV = rge.AlphaS(self.ip['asMZ'],\
                             self.ip['Mz']).run({'mbmb': self.ip['mb_at_mb'],\
                                                 'mcmc': self.ip['mc_at_mc']},\
                                                {'mub': self.ip['mb_at_mb'],\
                                                 'muc': self.ip['mc_at_mc']}, 2, 3, 1)
        gs2_2GeV = 4*np.pi*as_2GeV

        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            if double_QCD:
                adm_eff = self.ADM_eff
            else:
                projector = np.vstack((np.hstack((np.zeros((64,64)),\
                                                  np.ones((64,12)))),\
                                       np.zeros((12,76))))
                adm_eff = [np.multiply(projector, self.ADM_eff[0])]
        else:
            pass

        as41 = rge.AlphaS(self.ip['asMZ'], self.ip['Mz'])
        as41_high = as41.run({'mbmb': self.ip['mb_at_mb'], 'mcmc': self.ip['mc_at_mc']},\
                             {'mub': self.ip['mb_at_mb'], 'muc': self.ip['mc_at_mc']}, mb, 4, 1)
        as41_low = as41.run({'mbmb': self.ip['mb_at_mb'], 'mcmc': self.ip['mc_at_mc']},\
                            {'mub': self.ip['mb_at_mb'], 'muc': self.ip['mc_at_mc']}, mu_low, 4, 1)

        evolve1 = rge.RGE(self.gamma_QCD, 4)
        evolve2 = rge.RGE(self.gamma_QCD2, 4)
        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            evolve8 = rge.RGE(adm_eff, 4)
        else:
            pass

        # Mixing in the dim.6 DM-SM sector
        #
        C_at_mc_QCD = np.dot(evolve2.U0_as2(as41_high, as41_low),\
                             np.dot(evolve1.U0(as41_high, as41_low),\
                                    self.coeff_list_dm_dim5_dim6_dim7))
        C_at_mc_QED = np.dot(self.coeff_list_dm_dim5_dim6_dim7, self.gamma_QED)\
                      * np.log(mu_low/mb) * alpha_at_mc/(4*np.pi)\
                      + np.dot(self.coeff_list_dm_dim5_dim6_dim7, self.gamma_QED2)\
                      * np.log(mu_low/mb) * (alpha_at_mc/(4*np.pi))**2

        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            # Mixing in the dim.6 SM-SM and dim.8 DM-SM sector

            DIM6_DIM8_init = np.hstack((self.coeff_list_sm_dim6, self.coeff_list_dm_dim8))

            DIM6_DIM8_at_mb =   np.dot(evolve8.U0(as41_high, as41_low), DIM6_DIM8_init)

        # Revert back to dictionary

        dict_coeff_mc = list_to_dict(C_at_mc_QCD + C_at_mc_QED, self.wc_name_list)
        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            dict_dm_dim8 = list_to_dict(np.delete(DIM6_DIM8_at_mb, np.s_[0:64]), self.wc8_name_list)
            dict_sm_dim6 = list_to_dict(np.delete(DIM6_DIM8_at_mb, np.s_[64:70]), self.sm_name_list)
            dict_sm_lepton_dim6 = list_to_dict(self.coeff_list_sm_lepton_dim6, self.sm_lepton_name_list)

            dict_coeff_mc.update(dict_dm_dim8)
            dict_coeff_mc.update(dict_sm_dim6)
            dict_coeff_mc.update(dict_sm_lepton_dim6)

        return dict_coeff_mc



    def match(self, RGE=None, double_QCD=None, mu=None):
        """ Match from four-flavor to three-flavor QCD

        Calculate the matching at mu [GeV; default 2 GeV].

        Returns a dictionary of Wilson coefficients for the three-flavor Lagrangian
        at scale mu. The SM-SM Wilson coefficients are NOT returned. 

        RGE is an optional argument to turn RGE running on (True) or off (False). (Default True)
        """
        if mu is None:
            mu=2
        if RGE is None:
            RGE=True
        if double_QCD is None:
            double_QCD=True

        # The new coefficients
        cdict3f = {}
        if RGE:
            cdold = self.run(mu, double_QCD)
        else:
            cdold = self.coeff_dict

        if self.DM_type == "D" or self.DM_type == "M":
            for wcn in self.wc_name_list_3f:
                cdict3f[wcn] = cdold[wcn]
            for wcn in self.wc8_name_list:
                cdict3f[wcn] = cdold[wcn]
            cdict3f['C71'] = cdold['C71'] - cdold['C75c']
            cdict3f['C72'] = cdold['C72'] - cdold['C76c']
            cdict3f['C73'] = cdold['C73'] + cdold['C77c']
            cdict3f['C74'] = cdold['C74'] + cdold['C78c']

        if self.DM_type == "C":
            for wcn in self.wc_name_list_3f:
                cdict3f[wcn] = cdold[wcn]
            for wcn in self.wc8_name_list:
                cdict3f[wcn] = cdold[wcn]
            cdict3f['C65'] = cdold['C65'] - cdold['C63c']
            cdict3f['C66'] = cdold['C66'] + cdold['C64c']

        if self.DM_type == "R":
            for wcn in self.wc_name_list_3f:
                cdict3f[wcn] = cdold[wcn]
            cdict3f['C65'] = cdold['C65'] - cdold['C63c']
            cdict3f['C66'] = cdold['C66'] + cdold['C64c']


        # return the 3-flavor coefficients
        return cdict3f


    def _my_cNR(self, DM_mass, RGE=None, NLO=None, double_QCD=None):
        """ Calculate the NR coefficients from four-flavor theory with meson contributions split off

        (mainly for internal use)
        """
        return WC_3flavor(self.match(RGE, double_QCD), self.DM_type, self.ip)._my_cNR(DM_mass, RGE, NLO)

    def cNR(self, DM_mass, qvec, RGE=None, NLO=None, double_QCD=None):
        """ Calculate the NR coefficients from four-flavor theory """
        return WC_3flavor(self.match(RGE, double_QCD), self.DM_type, self.ip).cNR(DM_mass, qvec, RGE, NLO)

    def write_mma(self, DM_mass, qvec, RGE=None, NLO=None, double_QCD=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass DM_mass (in GeV) and the spatial momentum transfer qvec (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is './')

        <filename> is the filename (default 'cNR.m')
        """
        WC_3flavor(self.match(RGE, double_QCD), self.DM_type,\
                   self.ip).write_mma(DM_mass, qvec, RGE, NLO, path, filename)




class WC_5flavor(object):
    def __init__(self, coeff_dict, DM_type, input_dict=None):
#    def __init__(self, coeff_dict, DM_type):
        """ Class for Wilson coefficients in 5 flavor QCD x QED plus DM.

        The argument should be a dictionary for the initial conditions of the 2 + 32 + 4 + 48 + 4 + 64 + 6 = 160 
        dimension-five to dimension-eight five-flavor-QCD Wilson coefficients (for Dirac DM) of the form
        {'C51' : value, 'C52' : value, ...}. For other DM types there are less coefficients.
        An arbitrary number of them can be given; the default values are zero. 
        The possible name are (with an hopefully obvious notation):

        The second argument is the DM type; it can take the following values: 
            "D" (Dirac fermion)
            "M" (Majorana fermion)
            "C" (Complex scalar)
            "R" (Real scalar)

        Dirac fermion:       'C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61c', 'C61b', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C75c', 'C75b', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C76c', 'C76b', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C77c', 'C77b', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C78c', 'C78b', 'C78e', 'C78mu', 'C78tau',
                             'C79u', 'C79d', 'C79s', 'C79c', 'C79b', 'C79e', 'C79mu', 'C79tau', 
                             'C710u', 'C710d', 'C710s', 'C710c', 'C710b', 'C710e', 'C710mu', 'C710tau',
                             'C711', 'C712', 'C713', 'C714',
                             'C715u', 'C715d', 'C715s', 'C715c', 'C715b', 'C715e', 'C715mu', 'C715tau', 
                             'C716u', 'C716d', 'C716s', 'C716c', 'C716b', 'C716e', 'C716mu', 'C716tau',
                             'C717u', 'C717d', 'C717s', 'C717c', 'C717b', 'C717e', 'C717mu', 'C717tau', 
                             'C718u', 'C718d', 'C718s', 'C718c', 'C718b', 'C718e', 'C718mu', 'C718tau',
                             'C719u', 'C719d', 'C719s', 'C719c', 'C719b', 'C719e', 'C719mu', 'C719tau', 
                             'C720u', 'C720d', 'C720s', 'C720c', 'C720b', 'C720e', 'C720mu', 'C720tau', 
                             'C721u', 'C721d', 'C721s', 'C721c', 'C721b', 'C721e', 'C721mu', 'C721tau', 
                             'C722u', 'C722d', 'C722s', 'C722c', 'C722b', 'C722e', 'C722mu', 'C722tau' 
                             'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s'

        Majorana fermion:    'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C75c', 'C75b', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C76c', 'C76b', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C77c', 'C77b', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C78c', 'C78b', 'C78e', 'C78mu', 'C78tau',
                             'C711', 'C712', 'C713', 'C714',
                             'C715u', 'C715d', 'C715s', 'C715c', 'C715b', 'C715e', 'C715mu', 'C715tau', 
                             'C716u', 'C716d', 'C716s', 'C716c', 'C716b', 'C716e', 'C716mu', 'C716tau',
                             'C717u', 'C717d', 'C717s', 'C717c', 'C717b', 'C717e', 'C717mu', 'C717tau', 
                             'C718u', 'C718d', 'C718s', 'C718c', 'C718b', 'C718e', 'C718mu', 'C718tau'

        Complex Scalar:      'C61u', 'C61d', 'C61s', 'C61c', 'C61b', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                             'C65', 'C66', 'C67', 'C68'

        Real Scalar:         'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                             'C65', 'C66', 'C67', 'C68'

        (the notation corresponds to the numbering in 1707.06998).
        The Wilson coefficients should be specified in the MS-bar scheme at MZ = 91.1876 GeV.


        In order to calculate consistently to dim.8 in the EFT, we need also the dim.6 SM operators. 
        The following subset of 10*8 + 5*4 = 100 operator coefficients are sufficient for our purposes:

         'D61ud', 'D62ud', 'D63ud', 'D63du', 'D64ud', 'D65ud', 'D66ud', 'D66du', 
         'D61us', 'D62us', 'D63us', 'D63su', 'D64us', 'D65us', 'D66us', 'D66su', 
         'D61uc', 'D62uc', 'D63uc', 'D63cu', 'D64uc', 'D65uc', 'D66uc', 'D66cu', 
         'D61ub', 'D62ub', 'D63ub', 'D63bu', 'D64ub', 'D65ub', 'D66ub', 'D66bu', 
         'D61ds', 'D62ds', 'D63ds', 'D63sd', 'D64ds', 'D65ds', 'D66ds', 'D66sd', 
         'D61dc', 'D62dc', 'D63dc', 'D63cd', 'D64dc', 'D65dc', 'D66dc', 'D66cd', 
         'D61db', 'D62db', 'D63db', 'D63bd', 'D64db', 'D65db', 'D66db', 'D66bd', 
         'D61sc', 'D62sc', 'D63sc', 'D63cs', 'D64sc', 'D65sc', 'D66sc', 'D66cs', 
         'D61sb', 'D62sb', 'D63sb', 'D63bs', 'D64sb', 'D65sb', 'D66sb', 'D66bs', 
         'D61cb', 'D62cb', 'D63cb', 'D63bc', 'D64cb', 'D65cb', 'D66cb', 'D66bc',
         'D61u', 'D62u', 'D63u', 'D64u', 
         'D61d', 'D62d', 'D63d', 'D64d', 
         'D61s', 'D62s', 'D63s', 'D64s', 
         'D61c', 'D62c', 'D63c', 'D64c', 
         'D61b', 'D62b', 'D63b', 'D64b'

        Unless specified otherwise by the user, the tree-level SM initial conditions at MZ are provided be default. 

        For completeness, the default initial conditions at MZ for the corresponding 
        leptonic operator Wilson coefficients are also given:

         'D63eu', 'D63muu', 'D63tauu', 'D63ed', 'D63mud', 'D63taud', 'D63es', 'D63mus', 'D63taus',
         'D62ue', 'D62umu', 'D62utau', 'D62de', 'D62dmu', 'D62dtau', 'D62se', 'D62smu', 'D62stau'

        The third argument is a dictionary with all input parameters.


        The class has four methods: 

        run
        ---
        Run the Wilson from MZ = 91.1876 GeV to mu_low [GeV; default mb(mb)], with 5 active quark flavors

        match
        -----
        Match the Wilson coefficients from 5-flavor to 4-flavor QCD, at scale mu [GeV; default mu = mb(mb)]

        cNR
        ---
        Calculate the cNR coefficients as defined in 1308.6288

        It has two mandatory arguments: The DM mass in GeV and the momentum transfer in GeV


        write_mma
        ---------
        Write an output file that can be loaded into mathematica, 
        to be used in the DMFormFactor package [1308.6288].
        """

        self.DM_type = DM_type

        # First, we define a standard ordering for the Wilson coefficients, so that we can use arrays

        self.sm_lepton_name_list = ['D63eu', 'D63muu', 'D63tauu', 'D63ed', 'D63mud',\
                                    'D63taud', 'D63es', 'D63mus', 'D63taus',
                                    'D62ue', 'D62umu', 'D62utau', 'D62de', 'D62dmu',\
                                    'D62dtau', 'D62se', 'D62smu', 'D62stau']

        self.sm_name_list = ['D61ud', 'D62ud', 'D63ud', 'D63du', 'D64ud', 'D65ud', 'D66ud', 'D66du', 
                             'D61us', 'D62us', 'D63us', 'D63su', 'D64us', 'D65us', 'D66us', 'D66su', 
                             'D61uc', 'D62uc', 'D63uc', 'D63cu', 'D64uc', 'D65uc', 'D66uc', 'D66cu', 
                             'D61ub', 'D62ub', 'D63ub', 'D63bu', 'D64ub', 'D65ub', 'D66ub', 'D66bu', 
                             'D61ds', 'D62ds', 'D63ds', 'D63sd', 'D64ds', 'D65ds', 'D66ds', 'D66sd', 
                             'D61dc', 'D62dc', 'D63dc', 'D63cd', 'D64dc', 'D65dc', 'D66dc', 'D66cd', 
                             'D61db', 'D62db', 'D63db', 'D63bd', 'D64db', 'D65db', 'D66db', 'D66bd', 
                             'D61sc', 'D62sc', 'D63sc', 'D63cs', 'D64sc', 'D65sc', 'D66sc', 'D66cs', 
                             'D61sb', 'D62sb', 'D63sb', 'D63bs', 'D64sb', 'D65sb', 'D66sb', 'D66bs', 
                             'D61cb', 'D62cb', 'D63cb', 'D63bc', 'D64cb', 'D65cb', 'D66cb', 'D66bc',
                             'D61u', 'D62u', 'D63u', 'D64u', 
                             'D61d', 'D62d', 'D63d', 'D64d', 
                             'D61s', 'D62s', 'D63s', 'D64s', 
                             'D61c', 'D62c', 'D63c', 'D64c', 
                             'D61b', 'D62b', 'D63b', 'D64b']

        self.sm_name_list_4f = ['D61ud', 'D62ud', 'D63ud', 'D63du', 'D64ud', 'D65ud', 'D66ud', 'D66du', 
                                'D61us', 'D62us', 'D63us', 'D63su', 'D64us', 'D65us', 'D66us', 'D66su', 
                                'D61uc', 'D62uc', 'D63uc', 'D63cu', 'D64uc', 'D65uc', 'D66uc', 'D66cu', 
                                'D61ds', 'D62ds', 'D63ds', 'D63sd', 'D64ds', 'D65ds', 'D66ds', 'D66sd', 
                                'D61dc', 'D62dc', 'D63dc', 'D63cd', 'D64dc', 'D65dc', 'D66dc', 'D66cd', 
                                'D61sc', 'D62sc', 'D63sc', 'D63cs', 'D64sc', 'D65sc', 'D66sc', 'D66cs', 
                                'D61u', 'D62u', 'D63u', 'D64u', 
                                'D61d', 'D62d', 'D63d', 'D64d', 
                                'D61s', 'D62s', 'D63s', 'D64s', 
                                'D61c', 'D62c', 'D63c', 'D64c']

        if self.DM_type == "D":
            self.wc_name_list = ['C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61c', 'C61b', 'C61e', 'C61mu', 'C61tau', 
                                 'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75c', 'C75b', 'C75e', 'C75mu', 'C75tau', 
                                 'C76u', 'C76d', 'C76s', 'C76c', 'C76b', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77c', 'C77b', 'C77e', 'C77mu', 'C77tau', 
                                 'C78u', 'C78d', 'C78s', 'C78c', 'C78b', 'C78e', 'C78mu', 'C78tau',
                                 'C79u', 'C79d', 'C79s', 'C79c', 'C79b', 'C79e', 'C79mu', 'C79tau', 
                                 'C710u', 'C710d', 'C710s', 'C710c', 'C710b', 'C710e', 'C710mu', 'C710tau',
                                 'C711', 'C712', 'C713', 'C714',
                                 'C715u', 'C715d', 'C715s', 'C715c', 'C715b', 'C715e', 'C715mu', 'C715tau', 
                                 'C716u', 'C716d', 'C716s', 'C716c', 'C716b', 'C716e', 'C716mu', 'C716tau',
                                 'C717u', 'C717d', 'C717s', 'C717c', 'C717b', 'C717e', 'C717mu', 'C717tau', 
                                 'C718u', 'C718d', 'C718s', 'C718c', 'C718b', 'C718e', 'C718mu', 'C718tau',
                                 'C719u', 'C719d', 'C719s', 'C719c', 'C719b', 'C719e', 'C719mu', 'C719tau', 
                                 'C720u', 'C720d', 'C720s', 'C720c', 'C720b', 'C720e', 'C720mu', 'C720tau', 
                                 'C721u', 'C721d', 'C721s', 'C721c', 'C721b', 'C721e', 'C721mu', 'C721tau', 
                                 'C722u', 'C722d', 'C722s', 'C722c', 'C722b', 'C722e', 'C722mu', 'C722tau']

            self.wc8_name_list = ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s',\
                                  'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s']

            self.wc_name_list_4f = ['C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61c', 'C61e', 'C61mu', 'C61tau', 
                                    'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                    'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                                    'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                    'C71', 'C72', 'C73', 'C74',
                                    'C75u', 'C75d', 'C75s', 'C75c', 'C75e', 'C75mu', 'C75tau', 
                                    'C76u', 'C76d', 'C76s', 'C76c', 'C76e', 'C76mu', 'C76tau',
                                    'C77u', 'C77d', 'C77s', 'C77c', 'C77e', 'C77mu', 'C77tau', 
                                    'C78u', 'C78d', 'C78s', 'C78c', 'C78e', 'C78mu', 'C78tau',
                                    'C79u', 'C79d', 'C79s', 'C79c', 'C79e', 'C79mu', 'C79tau', 
                                    'C710u', 'C710d', 'C710s', 'C710c', 'C710e', 'C710mu', 'C710tau',
                                    'C711', 'C712', 'C713', 'C714',
                                    'C715u', 'C715d', 'C715s', 'C715c', 'C715e', 'C715mu', 'C715tau', 
                                    'C716u', 'C716d', 'C716s', 'C716c', 'C716e', 'C716mu', 'C716tau',
                                    'C717u', 'C717d', 'C717s', 'C717c', 'C717e', 'C717mu', 'C717tau', 
                                    'C718u', 'C718d', 'C718s', 'C718c', 'C718e', 'C718mu', 'C718tau',
                                    'C719u', 'C719d', 'C719s', 'C719c', 'C719e', 'C719mu', 'C719tau', 
                                    'C720u', 'C720d', 'C720s', 'C720c', 'C720e', 'C720mu', 'C720tau', 
                                    'C721u', 'C721d', 'C721s', 'C721c', 'C721e', 'C721mu', 'C721tau', 
                                    'C722u', 'C722d', 'C722s', 'C722c', 'C722e', 'C722mu', 'C722tau']

        if self.DM_type == "M":
            self.wc_name_list = ['C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75c', 'C75b', 'C75e', 'C75mu', 'C75tau', 
                                 'C76u', 'C76d', 'C76s', 'C76c', 'C76b', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77c', 'C77b', 'C77e', 'C77mu', 'C77tau', 
                                 'C78u', 'C78d', 'C78s', 'C78c', 'C78b', 'C78e', 'C78mu', 'C78tau',
                                 'C711', 'C712', 'C713', 'C714',
                                 'C715u', 'C715d', 'C715s', 'C715c', 'C715b', 'C715e', 'C715mu', 'C715tau', 
                                 'C716u', 'C716d', 'C716s', 'C716c', 'C716b', 'C716e', 'C716mu', 'C716tau',
                                 'C717u', 'C717d', 'C717s', 'C717c', 'C717b', 'C717e', 'C717mu', 'C717tau', 
                                 'C718u', 'C718d', 'C718s', 'C718c', 'C718b', 'C718e', 'C718mu', 'C718tau']

            self.wc8_name_list = ['C82u', 'C82d', 'C82s', 'C84u', 'C84d', 'C84s']

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [i for i in range(0,10)] + [i for i in range(18,26)]\
                           + [i for i in range(70,86)] + [i for i in range(122,154)]
            # The list of indices to be deleted from the dim.8 ADM because of less operators
            del_ind_list_dim_8 = np.r_[np.s_[0:3], np.s_[6:9]]
            # The list of indices to be deleted from the ADT because of less operators (dim.6 part)
            del_ind_list_adt_quark = np.r_[np.s_[0:5]]

            # The 4-flavor list for matching only
            self.wc_name_list_4f = ['C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                    'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                    'C71', 'C72', 'C73', 'C74',
                                    'C75u', 'C75d', 'C75s', 'C75c', 'C75e', 'C75mu', 'C75tau', 
                                    'C76u', 'C76d', 'C76s', 'C76c', 'C76e', 'C76mu', 'C76tau',
                                    'C77u', 'C77d', 'C77s', 'C77c', 'C77e', 'C77mu', 'C77tau', 
                                    'C78u', 'C78d', 'C78s', 'C78c', 'C78e', 'C78mu', 'C78tau',
                                    'C711', 'C712', 'C713', 'C714',
                                    'C715u', 'C715d', 'C715s', 'C715c', 'C715e', 'C715mu', 'C715tau', 
                                    'C716u', 'C716d', 'C716s', 'C716c', 'C716e', 'C716mu', 'C716tau',
                                    'C717u', 'C717d', 'C717s', 'C717c', 'C717e', 'C717mu', 'C717tau', 
                                    'C718u', 'C718d', 'C718s', 'C718c', 'C718e', 'C718mu', 'C718tau']

        if self.DM_type == "C":
            self.wc_name_list = ['C61u', 'C61d', 'C61s', 'C61c', 'C61b', 'C61e', 'C61mu', 'C61tau', 
                                 'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                                 'C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau',
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                                 'C67', 'C68']

            self.wc8_name_list = ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s']

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [0,1] + [i for i in range(10,18)] + [i for i in range(26,34)]\
                           + [35] + [37] + [i for i in range(46,54)]\
                           + [i for i in range(62,86)] + [87] + [89] + [i for i in range(90,154)]
            # The list of indices to be deleted from the dim.8 ADM because of less operators
            del_ind_list_dim_8 = np.r_[np.s_[0:3], np.s_[6:9]]
            # The list of indices to be deleted from the ADT because of less operators (dim.6 part)
            del_ind_list_adt_quark = np.r_[np.s_[0:5]]

            # The 4-flavor list for matching only
            self.wc_name_list_4f = ['C61u', 'C61d', 'C61s', 'C61c', 'C61e', 'C61mu', 'C61tau', 
                                    'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                    'C65', 'C66',
                                    'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                                    'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                    'C67', 'C68']

        if self.DM_type == "R":
            self.wc_name_list = ['C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                                 'C67', 'C68']

            self.wc8_name_list = []

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [i for i in range(0,34)] + [35] + [37] + [i for i in range(46,54)]\
                           + [i for i in range(62,86)]\
                           + [87] + [89] + [i for i in range(90,154)]

            # The 4-flavor list for matching only
            self.wc_name_list_4f = ['C65', 'C66',
                                    'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau',
                                    'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                    'C67', 'C68']



        self.coeff_dict = {}
        # Issue a user warning if a key is not defined:
        for wc_name in coeff_dict.keys():
            if wc_name in self.wc_name_list:
                pass
            elif wc_name in self.wc8_name_list:
                pass
            elif wc_name in self.sm_name_list:
                pass
            elif wc_name in self.sm_lepton_name_list:
                pass
            else:
                warnings.warn('The key ' + wc_name + ' is not a valid key. Typo?')


        # The dictionary of input parameters
        self.ip = input_dict
        # if input_dict is None:
        #     self.ip = Num_input().input_parameters
        # else:
        #     self.ip = Num_input(input_dict).input_parameters

        # Create the dictionary of Wilson coefficients. 
        #
        # First, the default values (0 for DM operators, SM values for SM operators):
        #
        # This is actually conceptually not so good.
        # The SM initial conditions should be moved to a matching method above the e/w scale?

        for wc_name in self.wc_name_list:
            self.coeff_dict[wc_name] = 0.
        for wc_name in self.wc8_name_list:
            self.coeff_dict[wc_name] = 0.

        sw = np.sqrt(self.ip['sw2_MSbar'])
        cw = np.sqrt(1-sw**2)
        vd = (-1/2 - 2*sw**2*(-1/3))/(2*sw*cw)
        vu = (1/2 - 2*sw**2*(2/3))/(2*sw*cw)
        ad = -(-1/2)/(2*sw*cw)
        au = -(1/2)/(2*sw*cw)

        vl = (-1/2 - 2*sw**2*(-1))/(2*sw*cw)
        al = -(-1/2)/(2*sw*cw)

        self.coeff_dict['D61ud'] = vu*vd * 4*sw**2*cw**2 + 1/6
        self.coeff_dict['D62ud'] = au*ad * 4*sw**2*cw**2 + 1/6
        self.coeff_dict['D63ud'] = au*vd * 4*sw**2*cw**2 - 1/6
        self.coeff_dict['D63du'] = ad*vu * 4*sw**2*cw**2 - 1/6
        self.coeff_dict['D64ud'] = 1
        self.coeff_dict['D65ud'] = 1
        self.coeff_dict['D66ud'] = -1
        self.coeff_dict['D66du'] = -1

        self.coeff_dict['D61us'] = vu*vd * 4*sw**2*cw**2
        self.coeff_dict['D62us'] = au*ad * 4*sw**2*cw**2
        self.coeff_dict['D63us'] = au*vd * 4*sw**2*cw**2
        self.coeff_dict['D63su'] = ad*vu * 4*sw**2*cw**2
        self.coeff_dict['D64us'] = 0
        self.coeff_dict['D65us'] = 0
        self.coeff_dict['D66us'] = 0
        self.coeff_dict['D66su'] = 0

        self.coeff_dict['D61uc'] = vu*vu * 4*sw**2*cw**2
        self.coeff_dict['D62uc'] = au*au * 4*sw**2*cw**2
        self.coeff_dict['D63uc'] = au*vu * 4*sw**2*cw**2
        self.coeff_dict['D63cu'] = au*vu * 4*sw**2*cw**2
        self.coeff_dict['D64uc'] = 0
        self.coeff_dict['D65uc'] = 0
        self.coeff_dict['D66uc'] = 0
        self.coeff_dict['D66cu'] = 0

        self.coeff_dict['D61ub'] = vu*vd * 4*sw**2*cw**2
        self.coeff_dict['D62ub'] = au*ad * 4*sw**2*cw**2
        self.coeff_dict['D63ub'] = au*vd * 4*sw**2*cw**2
        self.coeff_dict['D63bu'] = ad*vu * 4*sw**2*cw**2
        self.coeff_dict['D64ub'] = 0
        self.coeff_dict['D65ub'] = 0
        self.coeff_dict['D66ub'] = 0
        self.coeff_dict['D66bu'] = 0

        self.coeff_dict['D61ds'] = vd*vd * 4*sw**2*cw**2
        self.coeff_dict['D62ds'] = ad*ad * 4*sw**2*cw**2
        self.coeff_dict['D63ds'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['D63sd'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['D64ds'] = 0
        self.coeff_dict['D65ds'] = 0
        self.coeff_dict['D66ds'] = 0
        self.coeff_dict['D66sd'] = 0

        self.coeff_dict['D61dc'] = vd*vu * 4*sw**2*cw**2
        self.coeff_dict['D62dc'] = ad*au * 4*sw**2*cw**2
        self.coeff_dict['D63dc'] = ad*vu * 4*sw**2*cw**2
        self.coeff_dict['D63cd'] = au*vd * 4*sw**2*cw**2
        self.coeff_dict['D64dc'] = 0
        self.coeff_dict['D65dc'] = 0
        self.coeff_dict['D66dc'] = 0
        self.coeff_dict['D66cd'] = 0

        self.coeff_dict['D61db'] = vd*vd * 4*sw**2*cw**2
        self.coeff_dict['D62db'] = ad*ad * 4*sw**2*cw**2
        self.coeff_dict['D63db'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['D63bd'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['D64db'] = 0
        self.coeff_dict['D65db'] = 0
        self.coeff_dict['D66db'] = 0
        self.coeff_dict['D66bd'] = 0

        self.coeff_dict['D61sc'] = vd*vu * 4*sw**2*cw**2 + 1/6
        self.coeff_dict['D62sc'] = ad*au * 4*sw**2*cw**2 + 1/6
        self.coeff_dict['D63sc'] = ad*vu * 4*sw**2*cw**2 - 1/6
        self.coeff_dict['D63cs'] = au*vd * 4*sw**2*cw**2 - 1/6
        self.coeff_dict['D64sc'] = 1
        self.coeff_dict['D65sc'] = 1
        self.coeff_dict['D66sc'] = -1
        self.coeff_dict['D66cs'] = -1

        self.coeff_dict['D61sb'] = vd*vd * 4*sw**2*cw**2
        self.coeff_dict['D62sb'] = ad*ad * 4*sw**2*cw**2
        self.coeff_dict['D63sb'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['D63bs'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['D64sb'] = 0
        self.coeff_dict['D65sb'] = 0
        self.coeff_dict['D66sb'] = 0
        self.coeff_dict['D66bs'] = 0

        self.coeff_dict['D61cb'] = vu*vd * 4*sw**2*cw**2
        self.coeff_dict['D62cb'] = au*ad * 4*sw**2*cw**2
        self.coeff_dict['D63cb'] = au*vd * 4*sw**2*cw**2
        self.coeff_dict['D63bc'] = ad*vu * 4*sw**2*cw**2
        self.coeff_dict['D64cb'] = 0
        self.coeff_dict['D65cb'] = 0
        self.coeff_dict['D66cb'] = 0
        self.coeff_dict['D66bc'] = 0

        self.coeff_dict['D61u'] = vu**2 * 2*sw**2*cw**2
        self.coeff_dict['D62u'] = au**2 * 2*sw**2*cw**2
        self.coeff_dict['D63u'] = vu*au * 4*sw**2*cw**2
        self.coeff_dict['D64u'] = 0

        self.coeff_dict['D61d'] = vd**2 * 2*sw**2*cw**2
        self.coeff_dict['D62d'] = ad**2 * 2*sw**2*cw**2
        self.coeff_dict['D63d'] = vd*ad * 4*sw**2*cw**2
        self.coeff_dict['D64d'] = 0

        self.coeff_dict['D61s'] = vd**2 * 2*sw**2*cw**2
        self.coeff_dict['D62s'] = ad**2 * 2*sw**2*cw**2
        self.coeff_dict['D63s'] = vd*ad * 4*sw**2*cw**2
        self.coeff_dict['D64s'] = 0

        self.coeff_dict['D61c'] = vu**2 * 2*sw**2*cw**2
        self.coeff_dict['D62c'] = au**2 * 2*sw**2*cw**2
        self.coeff_dict['D63c'] = vu*au * 4*sw**2*cw**2
        self.coeff_dict['D64c'] = 0

        self.coeff_dict['D61b'] = vd**2 * 2*sw**2*cw**2
        self.coeff_dict['D62b'] = ad**2 * 2*sw**2*cw**2
        self.coeff_dict['D63b'] = vd*ad * 4*sw**2*cw**2
        self.coeff_dict['D64b'] = 0

        # Leptons

        self.coeff_dict['D62ue'] = au*al * 4*sw**2*cw**2
        self.coeff_dict['D62umu'] = au*al * 4*sw**2*cw**2
        self.coeff_dict['D62utau'] = au*al * 4*sw**2*cw**2

        self.coeff_dict['D62de'] = ad*al * 4*sw**2*cw**2
        self.coeff_dict['D62dmu'] = ad*al * 4*sw**2*cw**2
        self.coeff_dict['D62dtau'] = ad*al * 4*sw**2*cw**2

        self.coeff_dict['D62se'] = ad*al * 4*sw**2*cw**2
        self.coeff_dict['D62smu'] = ad*al * 4*sw**2*cw**2
        self.coeff_dict['D62stau'] = ad*al * 4*sw**2*cw**2

        self.coeff_dict['D63eu'] = al*vu * 4*sw**2*cw**2
        self.coeff_dict['D63muu'] = al*vu * 4*sw**2*cw**2
        self.coeff_dict['D63tauu'] = al*vu * 4*sw**2*cw**2

        self.coeff_dict['D63ed'] = al*vd * 4*sw**2*cw**2
        self.coeff_dict['D63mud'] = al*vd * 4*sw**2*cw**2
        self.coeff_dict['D63taud'] = al*vd * 4*sw**2*cw**2

        self.coeff_dict['D63es'] = al*vd * 4*sw**2*cw**2
        self.coeff_dict['D63mus'] = al*vd * 4*sw**2*cw**2
        self.coeff_dict['D63taus'] = al*vd * 4*sw**2*cw**2


        # Now update with the user-specified values, if defined

        for wc_name in self.wc_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                pass

        for wc_name in self.wc8_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                pass

        for wc_name in self.sm_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                pass

        for wc_name in self.sm_lepton_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                pass


        # Create the np.array of coefficients:
        self.coeff_list_dm_dim5_dim6_dim7 = np.array(dict_to_list(self.coeff_dict, self.wc_name_list))
        self.coeff_list_dm_dim8 = np.array(dict_to_list(self.coeff_dict, self.wc8_name_list))
        self.coeff_list_sm_dim6 = np.array(dict_to_list(self.coeff_dict, self.sm_name_list))
        self.coeff_list_sm_lepton_dim6 = np.array(dict_to_list(self.coeff_dict, self.sm_lepton_name_list))


        #---------------------------#
        # The anomalous dimensions: #
        #---------------------------#

        if self.DM_type == "D":
            self.gamma_QED = adm.ADM_QED(5)
            self.gamma_QED2 = adm.ADM_QED2(5)
            self.gamma_QCD = adm.ADM_QCD(5)
            self.gamma_QCD2 = adm.ADM_QCD2(5)
            self.gamma_QCD_dim8 = adm.ADM_QCD_dim8(5)
            self.gamma_hat = adm.ADT_QCD(5, self.ip)
        if self.DM_type == "M":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD_dim8 = np.delete(np.delete(adm.ADM_QCD_dim8(5), del_ind_list_dim_8, 0),\
                                            del_ind_list_dim_8, 1)
            self.gamma_hat = np.delete(np.delete(adm.ADT_QCD(5, self.ip), del_ind_list_dim_8, 0),\
                                       del_ind_list_adt_quark, 2)
        if self.DM_type == "C":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD_dim8 = np.delete(np.delete(adm.ADM_QCD_dim8(5), del_ind_list_dim_8, 0),\
                                            del_ind_list_dim_8, 1)
            self.gamma_hat = np.delete(np.delete(adm.ADT_QCD(5, self.ip), del_ind_list_dim_8, 0),\
                                       del_ind_list_adt_quark, 2)
        if self.DM_type == "R":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(5), del_ind_list, 1), del_ind_list, 2)

        self.ADM_SM = adm.ADM_SM_QCD(5)



        #--------------------------------------------------------------------#
        # The effective anomalous dimension for mixing into dimension eight: #
        #--------------------------------------------------------------------#

        # We need to contract the ADT with a subset of the dim.-6 Wilson coefficients
        if self.DM_type == "D":
            DM_dim6_init = np.delete(self.coeff_list_dm_dim5_dim6_dim7,\
                                     np.r_[np.s_[0:18], np.s_[23:26], np.s_[31:154]])
        elif self.DM_type == "M":
            DM_dim6_init = np.delete(self.coeff_list_dm_dim5_dim6_dim7, np.r_[np.s_[0:8], np.s_[13:88]])
        elif self.DM_type == "C":
            DM_dim6_init = np.delete(self.coeff_list_dm_dim5_dim6_dim7, np.r_[np.s_[0:8], np.s_[13:36]])


        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            # The columns of ADM_eff correspond to SM6 operators;
            # the rows of ADM_eff correspond to DM8 operators:
            C6_dot_ADM_hat = np.transpose(np.tensordot(DM_dim6_init, self.gamma_hat, (0,2)))

            # The effective ADM
            #
            # Note that the mixing of the SM operators with four equal flavors
            # does not contribute if we neglect yu, yd, ys! 

            self.ADM_eff = [np.vstack((np.hstack((self.ADM_SM,\
                                                  np.vstack((C6_dot_ADM_hat,\
                                                             np.zeros((20, len(self.gamma_QCD_dim8))))))),\
                            np.hstack((np.zeros((len(self.gamma_QCD_dim8),\
                                                 len(self.coeff_list_sm_dim6))), self.gamma_QCD_dim8))))]
        if self.DM_type == "R":
            pass



    def run(self, mu_low=None, double_QCD=None):
        """ Running of 5-flavor Wilson coefficients

        Calculate the running from MZ to mu_low [GeV; default mb(mb)] in the five-flavor theory. 

        Return a dictionary of Wilson coefficients for the five-flavor Lagrangian
        at scale mu_low.
        """

        if mu_low is None:
            mu_low=self.ip['mb_at_mb']
        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            if double_QCD is None:
                double_QCD=True
        else:
            double_QCD=False


        #-------------#
        # The running #
        #-------------#

        MZ = self.ip['Mz']
        alpha_at_mb = 1/self.ip['aMZinv']

        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            if double_QCD:
                adm_eff = self.ADM_eff
            else:
                projector = np.vstack((np.hstack((np.zeros((100,100)),\
                                                  np.ones((100,12)))), np.zeros((12,112))))
                adm_eff = [np.multiply(projector, self.ADM_eff[0])]
        else:
            double_QCD=False

        as51 = rge.AlphaS(self.ip['asMZ'], self.ip['Mz'])
        as51_high = as51.run({'mbmb': self.ip['mb_at_mb'],\
                              'mcmc': self.ip['mc_at_mc']},\
                             {'mub': self.ip['mb_at_mb'],\
                              'muc': self.ip['mc_at_mc']}, MZ, 5, 1)
        as51_low = as51.run({'mbmb': self.ip['mb_at_mb'],\
                             'mcmc': self.ip['mc_at_mc']},\
                            {'mub': self.ip['mb_at_mb'],\
                             'muc': self.ip['mc_at_mc']}, mu_low, 5, 1)

        evolve1 = rge.RGE(self.gamma_QCD, 5)
        evolve2 = rge.RGE(self.gamma_QCD2, 5)
        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            evolve8 = rge.RGE(adm_eff, 5)
        else:
            pass

        # Mixing in the dim.6 DM-SM sector
        #
        # Strictly speaking, MZ and mb should be defined at the same scale
        # (however, this is a higher-order difference)
        C_at_mb_QCD = np.dot(evolve2.U0_as2(as51_high, as51_low),\
                             np.dot(evolve1.U0(as51_high, as51_low),\
                                    self.coeff_list_dm_dim5_dim6_dim7))
        C_at_mb_QED = np.dot(self.coeff_list_dm_dim5_dim6_dim7, self.gamma_QED)\
                      * np.log(mu_low/MZ) * alpha_at_mb/(4*np.pi)\
                      + np.dot(self.coeff_list_dm_dim5_dim6_dim7, self.gamma_QED2)\
                      * np.log(mu_low/MZ) * (alpha_at_mb/(4*np.pi))**2

        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            # Mixing in the dim.6 SM-SM and dim.8 DM-SM sector

            DIM6_DIM8_init = np.hstack((self.coeff_list_sm_dim6, self.coeff_list_dm_dim8))

            DIM6_DIM8_at_mb =   np.dot(evolve8.U0(as51_high, as51_low), DIM6_DIM8_init)


        # Revert back to dictionary

        dict_coeff_mb = list_to_dict(C_at_mb_QCD + C_at_mb_QED, self.wc_name_list)
        if self.DM_type == "D" or self.DM_type == "M" or self.DM_type == "C":
            dict_dm_dim8 = list_to_dict(np.delete(DIM6_DIM8_at_mb, np.s_[0:100]), self.wc8_name_list)
            dict_sm_dim6 = list_to_dict(np.delete(DIM6_DIM8_at_mb, np.s_[100:112]), self.sm_name_list)
            dict_sm_lepton_dim6 = list_to_dict(self.coeff_list_sm_lepton_dim6, self.sm_lepton_name_list)

            dict_coeff_mb.update(dict_dm_dim8)
            dict_coeff_mb.update(dict_sm_dim6)
            dict_coeff_mb.update(dict_sm_lepton_dim6)

        return dict_coeff_mb


    def match(self, RGE=None, double_QCD=None, mu=None):
        """ Match from five-flavor to four-flavor QCD

        Calculate the matching at mu [GeV; default 4.18 GeV].

        Returns a dictionary of Wilson coefficients for the four-flavor Lagrangian
        at scale mu.

        RGE is an optional argument to turn RGE running on (True) or off (False). (Default True)
        """
        if RGE is None:
            RGE=True
        if mu is None:
            mu=self.ip['mb_at_mb']
        if double_QCD is None:
            double_QCD=True

        # The new coefficients
        cdict4f = {}
        if RGE:
            cdold = self.run(mu, double_QCD)
        else:
            cdold = self.coeff_dict

        if self.DM_type == "D" or self.DM_type == "M":
            for wcn in self.wc_name_list_4f:
                cdict4f[wcn] = cdold[wcn]
            for wcn in self.wc8_name_list:
                cdict4f[wcn] = cdold[wcn]
            for wcn in self.sm_name_list_4f:
                cdict4f[wcn] = cdold[wcn]
            for wcn in self.sm_lepton_name_list:
                cdict4f[wcn] = cdold[wcn]
            cdict4f['C71'] = cdold['C71'] - cdold['C75b']
            cdict4f['C72'] = cdold['C72'] - cdold['C76b']
            cdict4f['C73'] = cdold['C73'] + cdold['C77b']
            cdict4f['C74'] = cdold['C74'] + cdold['C78b']

        if self.DM_type == "C":
            for wcn in self.wc_name_list_4f:
                cdict4f[wcn] = cdold[wcn]
            for wcn in self.wc8_name_list:
                cdict4f[wcn] = cdold[wcn]
            for wcn in self.sm_name_list_4f:
                cdict4f[wcn] = cdold[wcn]
            for wcn in self.sm_lepton_name_list:
                cdict4f[wcn] = cdold[wcn]
            cdict4f['C65'] = cdold['C65'] - cdold['C63b']
            cdict4f['C66'] = cdold['C66'] + cdold['C64b']

        if self.DM_type == "R":
            for wcn in self.wc_name_list_4f:
                cdict4f[wcn] = cdold[wcn]
            cdict4f['C65'] = cdold['C65'] - cdold['C63b']
            cdict4f['C66'] = cdold['C66'] + cdold['C64b']

        # return the 4-flavor coefficients
        return cdict4f


    def _my_cNR(self, DM_mass, RGE=None, NLO=None, double_QCD=None):
        """ Calculate the NR coefficients from four-flavor theory with meson contributions split off

        (mainly for internal use) 
        """
        return WC_4flavor(self.match(RGE, double_QCD), self.DM_type,\
                     self.ip)._my_cNR(DM_mass, RGE, NLO, double_QCD)

    def cNR(self, DM_mass, qvec, RGE=None, NLO=None, double_QCD=None):
        """ Calculate the NR coefficients from four-flavor theory """
        return WC_4flavor(self.match(RGE, double_QCD), self.DM_type,\
                     self.ip).cNR(DM_mass, qvec, RGE, NLO, double_QCD)

    def write_mma(self, DM_mass, qvec, RGE=None, NLO=None, double_QCD=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass DM_mass (in GeV) and the spatial momentum transfer qvec (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is './')

        <filename> is the filename (default 'cNR.m')
        """
        WC_4flavor(self.match(RGE, double_QCD), self.DM_type,\
                   self.ip).write_mma(DM_mass, qvec, RGE, NLO, double_QCD, path, filename)







#-----------------------------#
# The e/w Wilson coefficients #
#-----------------------------#


class WilCo_EW(object):
    def __init__(self, coeff_dict, Ychi, dchi, DM_type, input_dict):
        """ Class for DM Wilson coefficients in the SM unbroken phase

        The first argument should be a dictionary for the initial conditions of the 8 
        dimension-five Wilson coefficients of the form
        {'C51' : value, 'C52' : value, ...}; 
        
        the 46 dimension-six Wilson coefficients of the form
        {'C611' : value, 'C621' : value, ...}; 

        and the dimension-seven Wilson coefficient (currently not yet implemented).
        An arbitrary number of them can be given; the default values are zero. 
        
        The possible keys are, for dchi != 1:

         'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58',
         'C611', 'C621', 'C631', 'C641', 'C651', 'C661', 'C671',
         'C681', 'C691', 'C6101', 'C6111', 'C6121', 'C6131', 'C6141',
         'C612', 'C622', 'C632', 'C642', 'C652', 'C662', 'C672',
         'C682', 'C692', 'C6102', 'C6112', 'C6122', 'C6132', 'C6142',
         'C613', 'C623', 'C633', 'C643', 'C653', 'C663', 'C673',
         'C683', 'C693', 'C6103', 'C6113', 'C6123', 'C6133', 'C6143',
         'C615', 'C616', 'C617', 'C618'
        
        The possible keys are, for dchi = 1:

         'C51', 'C53', 'C55', 'C57',
         'C621', 'C631', 'C641', 'C661', 'C671', 'C681', 'C6101', 'C6111', 'C6131', 'C6141',
         'C622', 'C632', 'C642', 'C662', 'C672', 'C682', 'C6102', 'C6112', 'C6132', 'C6142',
         'C623', 'C633', 'C643', 'C663', 'C673', 'C683', 'C6103', 'C6113', 'C6133', 'C6143',
         'C616', 'C618'

        The following set of 3*17 + 3*6 + 6*11 + 3*7 + 1 = 157 SM operator coefficients
        are also taken into account.
        They are generated, from the DM-SM operators, through mixing via penguin insertions. 
        Note that any mixing within the SM sector is neglected.
        For this reason, the values after running are not returned. 

        The possible keys are 

        'SM6111', 'SM6211', 'SM6311', 'SM6411', 'SM6511', 'SM6611', 'SM6711', 'SM6811', 'SM6911', 'SM61011', 
        'SM61111', 'SM61211', 'SM61311', 'SM61411', 'SM61511', 'SM61611', 'SM617711', 

        'SM6122', 'SM6222', 'SM6322', 'SM6422', 'SM6522', 'SM6622', 'SM6722', 'SM6822', 'SM6922', 'SM61022', 
        'SM61122', 'SM61222', 'SM61322', 'SM61422', 'SM61522', 'SM61622', 'SM617722', 

        'SM6133', 'SM6233', 'SM6333', 'SM6433', 'SM6533', 'SM6633', 'SM6733', 'SM6833', 'SM6933', 'SM61033', 
        'SM61133', 'SM61233', 'SM61333', 'SM61433', 'SM61533', 'SM61633', 'SM617733', 

        'SM6112', 'SM6212', 'SM6312', 'SM6321', 'SM6412', 'SM6421', 'SM6512', 'SM6612', 'SM6621', 'SM6712', 
        'SM6812', 'SM6912', 'SM6921', 'SM61012', 'SM61112', 'SM61121', 'SM61212', 'SM61221', 'SM61312', 'SM61321', 
        'SM61412', 'SM61421', 'SM61512', 'SM61521', 'SM61612', 'SM61621', 'SM617712', 'SM617721', 

        'SM6113', 'SM6213', 'SM6313', 'SM6331', 'SM6413', 'SM6431', 'SM6513', 'SM6613', 'SM6631', 'SM6713', 
        'SM6813', 'SM6913', 'SM6931', 'SM61013', 'SM61113', 'SM61131', 'SM61213', 'SM61231', 'SM61313', 'SM61331', 
        'SM61413', 'SM61431', 'SM61513', 'SM61531', 'SM61613', 'SM61631', 'SM617713', 'SM617731', 

        'SM6123', 'SM6223', 'SM6323', 'SM6332', 'SM6423', 'SM6432', 'SM6523', 'SM6623', 'SM6632', 'SM6723', 
        'SM6823', 'SM6923', 'SM6932', 'SM61023', 'SM61123', 'SM61132', 'SM61223', 'SM61232', 'SM61323', 'SM61332', 
        'SM61423', 'SM61432', 'SM61523', 'SM61532', 'SM61623', 'SM61632', 'SM617723', 'SM617732', 

        'SM6181', 'SM6191', 'SM6201', 'SM6211', 'SM6221', 'SM6231', 'SM6241', 

        'SM6182', 'SM6192', 'SM6202', 'SM6212', 'SM6222', 'SM6232', 'SM6242', 

        'SM6183', 'SM6193', 'SM6203', 'SM6213', 'SM6223', 'SM6233', 'SM6243', 

        'SM625'

        Unless specified otherwise by the user, the tree-level initial conditions are set to zero. 

        Finally, four DM operator coefficients are also taken into account.
        They are generated, from the DM-SM operators, through mixing via penguin insertions. 
        Note that any mixing within the DM sector is neglected.
        For this reason, the values after running are not returned. 

        The possible keys are 

        'DM61', 'DM62', 'DM63', 'DM64'

        Unless specified otherwise by the user, the tree-level initial conditions are set to zero. 
        Note that the numbering scheme for these coefficients is likely to change in the future. 


        dchi is the dimension of the DM SU2 representation.

        Ychi is the DM hypercharge such that Q = I^3 + Y/2

        The second-to-last argument is the DM type; it can only take the following value: 
            "D" (Dirac fermion)
        Other DM types might be implemented in the future. 

        The last argument is a dictionary with all input parameters.

        """
        self.DM_type = DM_type

        self.Ychi = Ychi
        self.dchi = dchi

        if self.DM_type == "D":
            if self.dchi == 1:
                self.wc_name_list_dim_5 = ['C51', 'C53', 'C55', 'C57']
                self.wc_name_list_dim_6 = ['C621', 'C631', 'C641', 'C661', 'C671',\
                                           'C681', 'C6101', 'C6111', 'C6131', 'C6141',\
                                           'C622', 'C632', 'C642', 'C662', 'C672',\
                                           'C682', 'C6102', 'C6112', 'C6132', 'C6142',\
                                           'C623', 'C633', 'C643', 'C663', 'C673',\
                                           'C683', 'C6103', 'C6113', 'C6133', 'C6143',\
                                           'C616', 'C618']
                self.dm_name_list_dim_6 = ['DM61', 'DM62']
            else:
                self.wc_name_list_dim_5 = ['C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58']
                self.wc_name_list_dim_6 = ['C611', 'C621', 'C631', 'C641', 'C651', 'C661', 'C671',\
                                           'C681', 'C691', 'C6101', 'C6111', 'C6121', 'C6131', 'C6141',\
                                           'C612', 'C622', 'C632', 'C642', 'C652', 'C662', 'C672',\
                                           'C682', 'C692', 'C6102', 'C6112', 'C6122', 'C6132', 'C6142',\
                                           'C613', 'C623', 'C633', 'C643', 'C653', 'C663', 'C673',\
                                           'C683', 'C693', 'C6103', 'C6113', 'C6123', 'C6133', 'C6143',\
                                           'C615', 'C616', 'C617', 'C618']
                self.dm_name_list_dim_6 = ['DM61', 'DM62', 'DM63', 'DM64']
            self.sm_name_list_dim_6 = ['SM6111', 'SM6211', 'SM6311', 'SM6411', 'SM6511',\
                                       'SM6611', 'SM6711', 'SM6811', 'SM6911', 'SM61011',\
                                       'SM61111', 'SM61211', 'SM61311', 'SM61411',\
                                       'SM61511', 'SM61611', 'SM617711',\
                                       'SM6122', 'SM6222', 'SM6322', 'SM6422', 'SM6522',\
                                       'SM6622', 'SM6722', 'SM6822', 'SM6922', 'SM61022',\
                                       'SM61122', 'SM61222', 'SM61322', 'SM61422',\
                                       'SM61522', 'SM61622', 'SM617722',\
                                       'SM6133', 'SM6233', 'SM6333', 'SM6433', 'SM6533',\
                                       'SM6633', 'SM6733', 'SM6833', 'SM6933', 'SM61033',\
                                       'SM61133', 'SM61233', 'SM61333', 'SM61433',\
                                       'SM61533', 'SM61633', 'SM617733',\
                                       'SM6112', 'SM6212', 'SM6312', 'SM6321', 'SM6412',\
                                       'SM6421', 'SM6512', 'SM6612', 'SM6621', 'SM6712',\
                                       'SM6812', 'SM6912', 'SM6921', 'SM61012', 'SM61112',\
                                       'SM61121', 'SM61212', 'SM61221', 'SM61312', 'SM61321',\
                                       'SM61412', 'SM61421', 'SM61512', 'SM61521',\
                                       'SM61612', 'SM61621', 'SM617712', 'SM617721',\
                                       'SM6113', 'SM6213', 'SM6313', 'SM6331', 'SM6413',\
                                       'SM6431', 'SM6513', 'SM6613', 'SM6631', 'SM6713',\
                                       'SM6813', 'SM6913', 'SM6931', 'SM61013', 'SM61113',\
                                       'SM61131', 'SM61213', 'SM61231', 'SM61313', 'SM61331',\
                                       'SM61413', 'SM61431', 'SM61513', 'SM61531',\
                                       'SM61613', 'SM61631', 'SM617713', 'SM617731',\
                                       'SM6123', 'SM6223', 'SM6323', 'SM6332', 'SM6423',\
                                       'SM6432', 'SM6523', 'SM6623', 'SM6632', 'SM6723',\
                                       'SM6823', 'SM6923', 'SM6932', 'SM61023', 'SM61123',\
                                       'SM61132', 'SM61223', 'SM61232', 'SM61323', 'SM61332',\
                                       'SM61423', 'SM61432', 'SM61523', 'SM61532',\
                                       'SM61623', 'SM61632', 'SM617723', 'SM617732',\
                                       'SM6181', 'SM6191', 'SM6201', 'SM6211',\
                                       'SM6221', 'SM6231', 'SM6241',\
                                       'SM6182', 'SM6192', 'SM6202', 'SM6212',\
                                       'SM6222', 'SM6232', 'SM6242',\
                                       'SM6183', 'SM6193', 'SM6203', 'SM6213',\
                                       'SM6223', 'SM6233', 'SM6243', 'SM625']

        else: raise Exception("Only Dirac fermion DM is implemented at the moment.")


        # Issue a user warning if a key is not defined or belongs to a redundant operator:
        for wc_name in coeff_dict.keys():
            if wc_name in self.wc_name_list_dim_5:
                pass
            elif wc_name in self.wc_name_list_dim_6:
                pass
            elif wc_name in self.sm_name_list_dim_6:
                pass
            elif wc_name in self.dm_name_list_dim_6:
                pass
            else:
                if self.dchi == 1:
                    warnings.warn('The key ' + wc_name + ' is not a valid key. Typo; or belongs to an operator that is redundant for dchi = 1?')
                else:
                    warnings.warn('The key ' + wc_name + ' is not a valid key. Typo?')


        self.coeff_dict = {}
        # Create the dictionary:
        for wc_name in (self.wc_name_list_dim_5 + self.wc_name_list_dim_6\
                        + self.sm_name_list_dim_6 + self.dm_name_list_dim_6):
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.

        # Create the np.array of coefficients:
        self.coeff_list_dim_5    = np.array(dict_to_list(self.coeff_dict, self.wc_name_list_dim_5))
        self.coeff_list_dim_6    = np.array(dict_to_list(self.coeff_dict, self.wc_name_list_dim_6))
        self.coeff_list_sm_dim_6 = np.array(dict_to_list(self.coeff_dict, self.sm_name_list_dim_6))
        self.coeff_list_dm_dim_6 = np.array(dict_to_list(self.coeff_dict, self.dm_name_list_dim_6))


        # The dictionary of input parameters
        self.ip = input_dict


    #---------#
    # Running #
    #---------#

    def run(self, mu_Lambda, muz=None):
        """Calculate the e/w running from scale mu_Lambda (to be given in GeV) to scale muz [muz = MZ by default].
        
        """
        if muz is None:
            muz = self.ip['Mz']+0.01


        # Define the dictionary of initial condictions for gauge / Yukawa couplings
        # at the scale mu = MZ (MSbar):
        #
        # (In the future, implement also the "running and matching" of mtop to mu = MZ)

        # The quark masses at MZ:
        
        def mb(mu, mub, muc, nf, loop):
            return rge.M_Quark_MSbar('b', self.ip['mb_at_mb'], self.ip['mb_at_mb'], self.ip['asMZ'],\
                                     self.ip['Mz']).run(mu, {'mbmb': self.ip['mb_at_mb'],\
                                                             'mcmc': self.ip['mc_at_mc']},\
                                                        {'mub': mub, 'muc': muc}, nf, loop)

        def mc(mu, mub, muc, nf, loop):
            return rge.M_Quark_MSbar('c', self.ip['mc_at_mc'], self.ip['mc_at_mc'], self.ip['asMZ'],\
                                     self.ip['Mz']).run(mu, {'mbmb': self.ip['mb_at_mb'],\
                                                             'mcmc': self.ip['mc_at_mc']},\
                                                        {'mub': mub, 'muc': muc}, nf, loop)

        self.mb_at_MZ = mb(self.ip['Mz'], self.ip['mb_at_mb'], self.ip['mc_at_mc'], 5, 1)
        self.mc_at_MZ = mc(self.ip['Mz'], self.ip['mb_at_mb'], self.ip['mc_at_mc'], 5, 1)

        self.g2_at_MZ   = np.sqrt(4*np.pi/self.ip['aMZinv']/self.ip['sw2_MSbar'])
        self.g1_at_MZ   = np.sqrt(self.g2_at_MZ**2/(1/self.ip['sw2_MSbar'] - 1))
        self.g3_at_MZ   = np.sqrt(4*np.pi*self.ip['asMZ'])
        self.yc_at_MZ   = np.sqrt(np.sqrt(2)*self.ip['GF'])*np.sqrt(2) * self.mc_at_MZ
        self.yb_at_MZ   = np.sqrt(np.sqrt(2)*self.ip['GF'])*np.sqrt(2) * self.mb_at_MZ
        self.ytau_at_MZ = np.sqrt(np.sqrt(2)*self.ip['GF'])*np.sqrt(2) * self.ip['mtau']
        self.yt_at_MZ   = np.sqrt(np.sqrt(2)*self.ip['GF'])*np.sqrt(2) * self.ip['mt_at_MZ']
        self.lam_at_MZ  = 2*np.sqrt(2) * self.ip['GF'] * self.ip['Mh']**2

        self.coupl_init_dict = {'g1': self.g1_at_MZ,\
                                'g2': self.g2_at_MZ,\
                                'gs': self.g3_at_MZ,\
                                'ytau': self.ytau_at_MZ,\
                                'yc': self.yc_at_MZ,\
                                'yb': self.yb_at_MZ,\
                                'yt': self.yt_at_MZ,\
                                'lam': self.lam_at_MZ}





            

        # The full vector of dim.-6 Wilson coefficients
        C6_at_Lambda = np.concatenate((self.coeff_list_dim_6, self.coeff_list_sm_dim_6,\
                                       self.coeff_list_dm_dim_6))

        # The vector of rescaled dim.-5 Wilson coefficients
        #
        # The e/w dipole operators are defined with a prefactor g_{1,2}/(8*pi^2).
        # The ADM are calculated in a basis with prefactors 1/g_{1,2}. 
        # Therefore, we need to rescale the Wilson coefficients by g_{1,2}(Lambda)^2/(8*pi^2) at mu=Lambda, 
        # and then again by (8*pi^2)/g_{1,2}(MZ)^2 at mu=MZ. 

        alpha1_at_Lambda = rge.CmuEW([], [], self.coupl_init_dict, self.ip['Mz'],\
                                     mu_Lambda, muz, self.Ychi,\
                                     self.dchi)._alphai(rge.CmuEW([], [],\
                                                                  self.coupl_init_dict,\
                                                                  self.ip['Mz'],\
                                                                  mu_Lambda, muz,\
                                                                  self.Ychi, self.dchi).ginit,\
                                     self.ip['Mz'], mu_Lambda, self.Ychi, self.dchi)[0]
        alpha2_at_Lambda = rge.CmuEW([], [], self.coupl_init_dict, self.ip['Mz'],\
                                     mu_Lambda, muz, self.Ychi,\
                                     self.dchi)._alphai(rge.CmuEW([], [],\
                                                                  self.coupl_init_dict,\
                                                                  self.ip['Mz'],\
                                                                  mu_Lambda, muz,\
                                                                  self.Ychi, self.dchi).ginit,\
                                     self.ip['Mz'], mu_Lambda, self.Ychi, self.dchi)[1]

        if self.dchi == 1:
            C5_at_Lambda_rescaled = self.coeff_list_dim_5 * np.array([alpha1_at_Lambda/(2*np.pi), 1,\
                                                                      alpha1_at_Lambda/(2*np.pi), 1])
        else:
            C5_at_Lambda_rescaled = self.coeff_list_dim_5 * np.array([alpha1_at_Lambda/(2*np.pi),\
                                                                      alpha2_at_Lambda/(2*np.pi), 1, 1,\
                                                                      alpha1_at_Lambda/(2*np.pi),\
                                                                      alpha2_at_Lambda/(2*np.pi), 1, 1])



        # The actual running 

        C5_at_muz = rge.CmuEW(C5_at_Lambda_rescaled, adm.ADM5(self.Ychi, self.dchi),\
                              self.coupl_init_dict, self.ip['Mz'],\
                              mu_Lambda, muz, self.Ychi, self.dchi).run()
        C6_at_muz = rge.CmuEW(C6_at_Lambda,          adm.ADM6(self.Ychi, self.dchi),\
                              self.coupl_init_dict, self.ip['Mz'],\
                              mu_Lambda, muz, self.Ychi, self.dchi).run()



        # Rescaling back to original normalization of dim.-5 Wilson coefficients

        alpha1_at_muz = rge.CmuEW([], [], self.coupl_init_dict, self.ip['Mz'],\
                                  mu_Lambda, muz, self.Ychi,\
                                  self.dchi)._alphai(rge.CmuEW([], [],\
                                                               self.coupl_init_dict,\
                                                               self.ip['Mz'],\
                                                               mu_Lambda, muz,\
                                                               self.Ychi, self.dchi).ginit,\
                                     self.ip['Mz'], muz, self.Ychi, self.dchi)[0]
        alpha2_at_muz = rge.CmuEW([], [], self.coupl_init_dict, self.ip['Mz'],\
                                  mu_Lambda, muz, self.Ychi,\
                                  self.dchi)._alphai(rge.CmuEW([], [],\
                                                               self.coupl_init_dict,\
                                                               self.ip['Mz'],\
                                                               mu_Lambda, muz,\
                                                               self.Ychi, self.dchi).ginit,\
                                     self.ip['Mz'], muz, self.Ychi, self.dchi)[1]


        if self.dchi == 1:
            C5_at_muz_rescaled = C5_at_muz * np.array([(2*np.pi)/alpha1_at_muz, 1,\
                                                       (2*np.pi)/alpha1_at_muz, 1])
        else:
            C5_at_muz_rescaled = C5_at_muz * np.array([(2*np.pi)/alpha1_at_muz,\
                                                       (2*np.pi)/alpha2_at_muz, 1, 1,\
                                                       (2*np.pi)/alpha1_at_muz,\
                                                       (2*np.pi)/alpha2_at_muz, 1, 1])




        # Convert arrays to dictionaries

        C5_at_muz_dict = list_to_dict(C5_at_muz_rescaled, self.wc_name_list_dim_5)
        C6_at_muz_dict = list_to_dict(C6_at_muz,          self.wc_name_list_dim_6)

        C_at_muz_dict = {}
        for wc_name in self.wc_name_list_dim_5:
            C_at_muz_dict[wc_name] = C5_at_muz_dict[wc_name]
        for wc_name in self.wc_name_list_dim_6:
            C_at_muz_dict[wc_name] = C6_at_muz_dict[wc_name]

        return C_at_muz_dict





    #----------#
    # Matching #
    #----------#

    def match(self, DM_mass, mu_Lambda, DM_mass_threshold=None, RUN_EW=None, DIM4=None):
        """Calculate the matching from the relativistic theory to the five-flavor theory at scale MZ

        DM_mass is the DM mass, as it appears in the UV Lagrangian. It is not the physical DM mass after EWSB. 

        mu_Lambda (to be given in GeV) is the starting scale of the RG evolution

        DM_mass_threshold is the DM mass below which DM is treated as "light" [default is 40 GeV]

        RUN_EW can have three values: 

         - RUN_EW = True  does the full leading-logarithmic resummation (this is the default)
         - RUN_EW = False -- no electroweak running

        DIM4 multiplies the dimension-four matching contributions.
        To be considered as an "checking tool", will be removed in the future

        Return a dictionary of Wilson coefficients for the five-flavor Lagrangian, 
        with the following keys (only Dirac DM is implemented so far):

        Dirac fermion:       'C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61c', 'C61b', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C65c', 'C65b', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C66c', 'C66b', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C67c', 'C67b', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C68c', 'C68b', 'C78e', 'C78mu', 'C78tau',
                             'C79u', 'C79d', 'C79s', 'C69c', 'C69b', 'C79e', 'C79mu', 'C79tau', 
                             'C710u', 'C710d', 'C710s', 'C610c', 'C610b', 'C710e', 'C710mu', 'C710tau'
        """
        if RUN_EW is None:
            RUN_EW = True
        self.RUN_EW = RUN_EW

        if DM_mass_threshold is None:
            DM_mass_threshold = 40 # GeV
        self.DM_mass_threshold = DM_mass_threshold

        if DIM4 is None:
            DIM4 = 1
        else:
            DIM4 = 0


        # Issue a user warning that matching results are not complete for Ychi != 0

        if self.Ychi != 0:
            warnings.warn('Matching contributions from gauge interactions are not complete for Ychi != 0')

 
        # Some input parameters:

        vev = 1/np.sqrt(np.sqrt(2)*self.ip['GF'])
        alpha = 1/self.ip['aMZinv']
        MW = self.ip['Mw']
        MZ = self.ip['Mz']
        Mh = self.ip['Mh']
        sw = np.sqrt(self.ip['sw2_MSbar'])
        cw = np.sqrt(1-sw**2)


        # The Wilson coefficients in the "UV" EFT at scale MZ
        if RUN_EW:
            wcew_dict = self.run(mu_Lambda, muz=self.ip['Mz'])
        else:
            wcew_dict = self.coeff_dict


        # Calculate the physical DM mass in terms of DM_mass and the Wilson coefficients, 
        # and the corresponding shift in the dimension-five Wilson coefficients.

        if DM_mass > DM_mass_threshold:
            if self.dchi == 1:
                self.DM_mass_phys = DM_mass - vev**2/2 * wcew_dict['C53']
                wc5_dict_shifted = {}

                wc5_dict_shifted['C51'] = wcew_dict['C51']\
                                          + vev**2/2/DM_mass * wcew_dict['C57'] * wcew_dict['C55']
                wc5_dict_shifted['C53'] = wcew_dict['C53']\
                                          + vev**2/2/DM_mass * wcew_dict['C57'] * wcew_dict['C57']
                wc5_dict_shifted['C55'] = wcew_dict['C55']\
                                          - vev**2/2/DM_mass * wcew_dict['C57'] * wcew_dict['C51']
                wc5_dict_shifted['C57'] = wcew_dict['C57']\
                                          - vev**2/2/DM_mass * wcew_dict['C57'] * wcew_dict['C53']

            else:
                self.DM_mass_phys = DM_mass - vev**2/2 * (wcew_dict['C53'] + self.Ychi/4 * wcew_dict['C54'])
                wc5_dict_shifted = {}

                wc5_dict_shifted['C51'] = wcew_dict['C51']\
                                          + vev**2/2/DM_mass * (wcew_dict['C57'] + self.Ychi/4\
                                                                * wcew_dict['C58']) * wcew_dict['C55']
                wc5_dict_shifted['C52'] = wcew_dict['C52']\
                                          + vev**2/2/DM_mass * (wcew_dict['C57'] + self.Ychi/4\
                                                                * wcew_dict['C58']) * wcew_dict['C56']
                wc5_dict_shifted['C53'] = wcew_dict['C53']\
                                          + vev**2/2/DM_mass * (wcew_dict['C57'] + self.Ychi/4\
                                                                * wcew_dict['C58']) * wcew_dict['C57']
                wc5_dict_shifted['C54'] = wcew_dict['C54']\
                                          + vev**2/2/DM_mass * (wcew_dict['C57'] + self.Ychi/4\
                                                                * wcew_dict['C58']) * wcew_dict['C58']
                wc5_dict_shifted['C55'] = wcew_dict['C55']\
                                          - vev**2/2/DM_mass * (wcew_dict['C57'] + self.Ychi/4\
                                                                * wcew_dict['C58']) * wcew_dict['C51']
                wc5_dict_shifted['C56'] = wcew_dict['C56']\
                                          - vev**2/2/DM_mass * (wcew_dict['C57'] + self.Ychi/4\
                                                                * wcew_dict['C58']) * wcew_dict['C52']
                wc5_dict_shifted['C57'] = wcew_dict['C57']\
                                          - vev**2/2/DM_mass * (wcew_dict['C57'] + self.Ychi/4\
                                                                * wcew_dict['C58']) * wcew_dict['C53']
                wc5_dict_shifted['C58'] = wcew_dict['C58']\
                                          - vev**2/2/DM_mass * (wcew_dict['C57'] + self.Ychi/4\
                                                                * wcew_dict['C58']) * wcew_dict['C54']

        else:
            if self.dchi == 1:
                cosphi = np.sqrt((wcew_dict['C53'] - 2*DM_mass/vev**2)**2/\
                                 ((wcew_dict['C53'] - 2*DM_mass/vev**2)**2 + wcew_dict['C57']**2))
                sinphi = np.sqrt((wcew_dict['C57'])**2/\
                                 ((wcew_dict['C53'] - 2*DM_mass/vev**2)**2 + wcew_dict['C57']**2))
                pre_DM_mass_phys = DM_mass*cosphi + vev**2/2\
                                   * (wcew_dict['C57'] * sinphi - wcew_dict['C53'] * cosphi)
                if pre_DM_mass_phys > 0:
                    self.DM_mass_phys = pre_DM_mass_phys

                    wc5_dict_shifted = {}

                    wc5_dict_shifted['C51'] = cosphi * wcew_dict['C51'] + sinphi * wcew_dict['C55'] 
                    wc5_dict_shifted['C53'] = cosphi * wcew_dict['C53'] + sinphi * wcew_dict['C57'] 
                    wc5_dict_shifted['C55'] = cosphi * wcew_dict['C55'] - sinphi * wcew_dict['C51'] 
                    wc5_dict_shifted['C57'] = cosphi * wcew_dict['C57'] - sinphi * wcew_dict['C53'] 
                else:
                    self.DM_mass_phys = - pre_DM_mass_phys

                    wc5_dict_shifted = {}

                    wc5_dict_shifted['C51'] = cosphi * wcew_dict['C51'] - sinphi * wcew_dict['C55'] 
                    wc5_dict_shifted['C53'] = cosphi * wcew_dict['C53'] - sinphi * wcew_dict['C57'] 
                    wc5_dict_shifted['C55'] = cosphi * wcew_dict['C55'] + sinphi * wcew_dict['C51'] 
                    wc5_dict_shifted['C57'] = cosphi * wcew_dict['C57'] + sinphi * wcew_dict['C53'] 
            else:
                cosphi = np.sqrt((wcew_dict['C53'] + self.Ychi/4 * wcew_dict['C54'] - 2*DM_mass/vev**2)**2/\
                                ((wcew_dict['C53'] + self.Ychi/4 * wcew_dict['C54'] - 2*DM_mass/vev**2)**2\
                                +(wcew_dict['C57'] + self.Ychi/4 * wcew_dict['C58'])**2))
                sinphi = np.sqrt((wcew_dict['C57'] + self.Ychi/4 * wcew_dict['C58'])**2/\
                                ((wcew_dict['C53'] + self.Ychi/4 * wcew_dict['C54'] - 2*DM_mass/vev**2)**2\
                                +(wcew_dict['C57'] + self.Ychi/4 * wcew_dict['C58'])**2))
                pre_DM_mass_phys = DM_mass*cosphi + vev**2/2 * ((wcew_dict['C57'] + self.Ychi/4\
                                                                 * wcew_dict['C58'])*sinphi\
                                                              - (wcew_dict['C53'] + self.Ychi/4\
                                                                 * wcew_dict['C54'])*cosphi)
                if pre_DM_mass_phys > 0:
                    self.DM_mass_phys = pre_DM_mass_phys

                    wc5_dict_shifted = {}

                    wc5_dict_shifted['C51'] = cosphi * wcew_dict['C51'] + sinphi * wcew_dict['C55'] 
                    wc5_dict_shifted['C52'] = cosphi * wcew_dict['C52'] + sinphi * wcew_dict['C56'] 
                    wc5_dict_shifted['C53'] = cosphi * wcew_dict['C53'] + sinphi * wcew_dict['C57'] 
                    wc5_dict_shifted['C54'] = cosphi * wcew_dict['C54'] + sinphi * wcew_dict['C58'] 
                    wc5_dict_shifted['C55'] = cosphi * wcew_dict['C55'] - sinphi * wcew_dict['C51'] 
                    wc5_dict_shifted['C56'] = cosphi * wcew_dict['C56'] - sinphi * wcew_dict['C52'] 
                    wc5_dict_shifted['C57'] = cosphi * wcew_dict['C57'] - sinphi * wcew_dict['C53'] 
                    wc5_dict_shifted['C58'] = cosphi * wcew_dict['C58'] - sinphi * wcew_dict['C54'] 
                else:
                    self.DM_mass_phys = - pre_DM_mass_phys

                    wc5_dict_shifted = {}

                    wc5_dict_shifted['C51'] = cosphi * wcew_dict['C51'] - sinphi * wcew_dict['C55'] 
                    wc5_dict_shifted['C52'] = cosphi * wcew_dict['C52'] - sinphi * wcew_dict['C56'] 
                    wc5_dict_shifted['C53'] = cosphi * wcew_dict['C53'] - sinphi * wcew_dict['C57'] 
                    wc5_dict_shifted['C54'] = cosphi * wcew_dict['C54'] - sinphi * wcew_dict['C58'] 
                    wc5_dict_shifted['C55'] = cosphi * wcew_dict['C55'] + sinphi * wcew_dict['C51'] 
                    wc5_dict_shifted['C56'] = cosphi * wcew_dict['C56'] + sinphi * wcew_dict['C52'] 
                    wc5_dict_shifted['C57'] = cosphi * wcew_dict['C57'] + sinphi * wcew_dict['C53'] 
                    wc5_dict_shifted['C58'] = cosphi * wcew_dict['C58'] + sinphi * wcew_dict['C54'] 

        # The redefinitions of the dim.-5 Wilson coefficients resulting from the mass shift:

        coeff_dict_shifted = wcew_dict
        coeff_dict_shifted.update(wc5_dict_shifted)

        # The Higgs penguin function. 
        # The result is valid for all input values and gives (in principle) a real output.
        # Note that currently there is no distinction between e/w and light DM,
        # as the two-loop function for light DM is unknown.
        def higgs_penguin_fermion(dchi):
            return Higgspenguin(dchi, self.ip).f_q_hisano(self.DM_mass_phys)
        def W_box_fermion(dchi):
            return Higgspenguin(dchi, self.ip).d_q_hisano(self.DM_mass_phys)
        def higgs_penguin_gluon(dchi):
            return Higgspenguin(dchi, self.ip).hisano_fa(self.DM_mass_phys)\
                   + Higgspenguin(dchi, self.ip).hisano_fbc(self.DM_mass_phys)


        #-----------------------#
        # The new coefficients: #
        #-----------------------#
        

        coeff_dict_5f = {}

        if self.dchi == 1:
            coeff_dict_5f['C51'] = coeff_dict_shifted['C51']
            coeff_dict_5f['C52'] = coeff_dict_shifted['C55']

            coeff_dict_5f['C61u'] = coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C631']/2\
                  + (3-8*sw**2)/6 * coeff_dict_shifted['C616']\
                  + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-8*sw**2) * DIM4
            coeff_dict_5f['C61d'] = coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C641']/2\
                  - (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                  - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4
            coeff_dict_5f['C61s'] = coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C642']/2\
                  - (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                  - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4
            coeff_dict_5f['C61c'] = coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C632']/2\
                  + (3-8*sw**2)/6 * coeff_dict_shifted['C616']\
                  + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-8*sw**2) * DIM4
            coeff_dict_5f['C61b'] = coeff_dict_shifted['C623']/2 + coeff_dict_shifted['C643']/2\
                  - (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                  - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4
            coeff_dict_5f['C61e'] = coeff_dict_shifted['C6101']/2 + coeff_dict_shifted['C6111']/2\
                  - (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                  - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4
            coeff_dict_5f['C61mu'] = coeff_dict_shifted['C6102']/2 + coeff_dict_shifted['C6112']/2\
                  - (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                  - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4
            coeff_dict_5f['C61tau'] = coeff_dict_shifted['C6103']/2 + coeff_dict_shifted['C6113']/2\
                  - (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                  - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4

            coeff_dict_5f['C62u'] = coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C671']/2\
                   + (3-8*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62d'] = coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C681']/2\
                   - (3-4*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62s'] = coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C682']/2\
                   - (3-4*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62c'] = coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C672']/2\
                   + (3-8*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62b'] = coeff_dict_shifted['C663']/2 + coeff_dict_shifted['C683']/2\
                   - (3-4*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62e'] = coeff_dict_shifted['C6131']/2 + coeff_dict_shifted['C6141']/2\
                   - (1-4*sw**2)/2 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62mu'] = coeff_dict_shifted['C6132']/2 + coeff_dict_shifted['C6142']/2\
                   - (1-4*sw**2)/2 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62tau'] = coeff_dict_shifted['C6133']/2 + coeff_dict_shifted['C6143']/2\
                   - (1-4*sw**2)/2 * coeff_dict_shifted['C618']

            coeff_dict_5f['C63u'] = - coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C631']/2\
                   - 1/2 * coeff_dict_shifted['C616']\
                   - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63d'] = - coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C641']/2\
                   + 1/2 * coeff_dict_shifted['C616']\
                   + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63s'] = - coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C642']/2\
                   + 1/2 * coeff_dict_shifted['C616']\
                   + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63c'] = - coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C632']/2\
                   - 1/2 * coeff_dict_shifted['C616']\
                   - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63b'] = - coeff_dict_shifted['C623']/2 + coeff_dict_shifted['C643']/2\
                   + 1/2 * coeff_dict_shifted['C616']\
                   + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63e'] = - coeff_dict_shifted['C6101']/2 + coeff_dict_shifted['C6111']/2\
                   + 1/2 * coeff_dict_shifted['C616']\
                   + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63mu'] = - coeff_dict_shifted['C6102']/2 + coeff_dict_shifted['C6112']/2\
                   + 1/2 * coeff_dict_shifted['C616']\
                   + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63tau'] = - coeff_dict_shifted['C6103']/2 + coeff_dict_shifted['C6113']/2\
                   + 1/2 * coeff_dict_shifted['C616']\
                   + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4

            coeff_dict_5f['C64u'] = - coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C671']/2\
                    - 1/2 * coeff_dict_shifted['C618']\
                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64d'] = - coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C681']/2\
                    + 1/2 * coeff_dict_shifted['C618']\
                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64s'] = - coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C682']/2\
                    + 1/2 * coeff_dict_shifted['C618']\
                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64c'] = - coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C672']/2\
                    - 1/2 * coeff_dict_shifted['C618']\
                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64b'] = - coeff_dict_shifted['C663']/2 + coeff_dict_shifted['C683']/2\
                    + 1/2 * coeff_dict_shifted['C618']\
                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64e'] = - coeff_dict_shifted['C6131']/2 + coeff_dict_shifted['C6141']/2\
                    + 1/2 * coeff_dict_shifted['C618']\
                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64mu'] = - coeff_dict_shifted['C6132']/2 + coeff_dict_shifted['C6142']/2\
                    + 1/2 * coeff_dict_shifted['C618']\
                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64tau'] = - coeff_dict_shifted['C6133']/2 + coeff_dict_shifted['C6143']/2\
                    + 1/2 * coeff_dict_shifted['C618']\
                    + W_box_fermion(self.dchi) * DIM4

            coeff_dict_5f['C71'] = (1/Mh**2 * (coeff_dict_shifted['C53']))\
                                   + higgs_penguin_gluon(self.dchi) * DIM4
            coeff_dict_5f['C72'] = (1/Mh**2 * (coeff_dict_shifted['C57']))

            coeff_dict_5f['C75u'] = - 1/Mh**2 * (coeff_dict_shifted['C53'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75d'] = - 1/Mh**2 * (coeff_dict_shifted['C53'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75s'] = - 1/Mh**2 * (coeff_dict_shifted['C53'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75c'] = - 1/Mh**2 * (coeff_dict_shifted['C53'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75b'] = - 1/Mh**2 * (coeff_dict_shifted['C53'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75e'] = - 1/Mh**2 * (coeff_dict_shifted['C53'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75mu'] = - 1/Mh**2 * (coeff_dict_shifted['C53'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75tau'] = - 1/Mh**2 * (coeff_dict_shifted['C53'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4

            coeff_dict_5f['C76u'] = - 1/Mh**2 * coeff_dict_shifted['C57']
            coeff_dict_5f['C76d'] = - 1/Mh**2 * coeff_dict_shifted['C57']
            coeff_dict_5f['C76s'] = - 1/Mh**2 * coeff_dict_shifted['C57']
            coeff_dict_5f['C76c'] = - 1/Mh**2 * coeff_dict_shifted['C57']
            coeff_dict_5f['C76b'] = - 1/Mh**2 * coeff_dict_shifted['C57']
            coeff_dict_5f['C76e'] = - 1/Mh**2 * coeff_dict_shifted['C57']
            coeff_dict_5f['C76mu'] = - 1/Mh**2 * coeff_dict_shifted['C57']
            coeff_dict_5f['C76tau'] = - 1/Mh**2 * coeff_dict_shifted['C57']

        else:
            coeff_dict_5f['C51'] = coeff_dict_shifted['C51'] + self.Ychi/2 * coeff_dict_shifted['C52']
            coeff_dict_5f['C52'] = coeff_dict_shifted['C55'] + self.Ychi/2 * coeff_dict_shifted['C56']

            coeff_dict_5f['C61u'] = - self.Ychi/8 * coeff_dict_shifted['C611']\
                                    + coeff_dict_shifted['C621']/2\
                                    + coeff_dict_shifted['C631']/2\
                                    + self.Ychi * (3-8*sw**2)/24 * coeff_dict_shifted['C615']\
                                    + (3-8*sw**2)/6 * coeff_dict_shifted['C616']\
                                    + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-8*sw**2) * DIM4
            coeff_dict_5f['C61d'] = self.Ychi/8*coeff_dict_shifted['C611']\
                                    + coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C641']/2\
                                    - self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C615']\
                                    - (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                                    - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4
            coeff_dict_5f['C61s'] = self.Ychi/8*coeff_dict_shifted['C612']\
                                    + coeff_dict_shifted['C622']/2\
                                    + coeff_dict_shifted['C642']/2\
                                    - self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C615']\
                                    - (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                                    - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4
            coeff_dict_5f['C61c'] = - self.Ychi/8*coeff_dict_shifted['C612']\
                                    + coeff_dict_shifted['C622']/2\
                                    + coeff_dict_shifted['C632']/2\
                                    + self.Ychi * (3-8*sw**2)/24 * coeff_dict_shifted['C615']\
                                    + (3-8*sw**2)/6 * coeff_dict_shifted['C616']\
                                    + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-8*sw**2) * DIM4
            coeff_dict_5f['C61b'] = self.Ychi/8*coeff_dict_shifted['C613']\
                                    + coeff_dict_shifted['C623']/2\
                                    + coeff_dict_shifted['C643']/2\
                                    - self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C615']\
                                    - (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                                    - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4
            coeff_dict_5f['C61e'] = self.Ychi/8*coeff_dict_shifted['C691']\
                                    + coeff_dict_shifted['C6101']/2\
                                    + coeff_dict_shifted['C6111']/2\
                                    - self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C615']\
                                    - (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                                    - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4
            coeff_dict_5f['C61mu'] = self.Ychi/8*coeff_dict_shifted['C692']\
                                     + coeff_dict_shifted['C6102']/2\
                                     + coeff_dict_shifted['C6112']/2\
                                     - self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C615']\
                                     - (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                                     - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4
            coeff_dict_5f['C61tau'] = self.Ychi/8*coeff_dict_shifted['C693']\
                                      + coeff_dict_shifted['C6103']/2\
                                      + coeff_dict_shifted['C6113']/2\
                                      - self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C615']\
                                      - (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                                      - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4

            coeff_dict_5f['C62u'] = - self.Ychi/8*coeff_dict_shifted['C651']\
                                    + coeff_dict_shifted['C661']/2\
                                    + coeff_dict_shifted['C671']/2\
                                    + self.Ychi * (3-8*sw**2)/24 * coeff_dict_shifted['C617']\
                                    + (3-8*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62d'] = self.Ychi/8*coeff_dict_shifted['C651']\
                                    + coeff_dict_shifted['C661']/2\
                                    + coeff_dict_shifted['C681']/2\
                                    - self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C617']\
                                    - (3-4*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62s'] = self.Ychi/8*coeff_dict_shifted['C652']\
                                    + coeff_dict_shifted['C662']/2\
                                    + coeff_dict_shifted['C682']/2\
                                    - self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C617']\
                                    - (3-4*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62c'] = - self.Ychi/8*coeff_dict_shifted['C652']\
                                    + coeff_dict_shifted['C662']/2\
                                    + coeff_dict_shifted['C672']/2\
                                    + self.Ychi * (3-8*sw**2)/24 * coeff_dict_shifted['C617']\
                                    + (3-8*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62b'] = self.Ychi/8*coeff_dict_shifted['C653']\
                                    + coeff_dict_shifted['C663']/2\
                                    + coeff_dict_shifted['C683']/2\
                                    - self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C617']\
                                    - (3-4*sw**2)/6 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62e'] = self.Ychi/8*coeff_dict_shifted['C6121']\
                                    + coeff_dict_shifted['C6131']/2\
                                    + coeff_dict_shifted['C6141']/2\
                                    - self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C617']\
                                    - (1-4*sw**2)/2 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62mu'] = self.Ychi/8*coeff_dict_shifted['C6122']\
                                     + coeff_dict_shifted['C6132']/2\
                                     + coeff_dict_shifted['C6142']/2\
                                     - self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C617']\
                                     - (1-4*sw**2)/2 * coeff_dict_shifted['C618']
            coeff_dict_5f['C62tau'] = self.Ychi/8*coeff_dict_shifted['C6123']\
                                      + coeff_dict_shifted['C6133']/2\
                                      + coeff_dict_shifted['C6143']/2\
                                      - self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C617']\
                                      - (1-4*sw**2)/2 * coeff_dict_shifted['C618']

            coeff_dict_5f['C63u'] = self.Ychi/8*coeff_dict_shifted['C611']\
                                    - coeff_dict_shifted['C621']/2\
                                    + coeff_dict_shifted['C631']/2\
                                    - self.Ychi/8 * coeff_dict_shifted['C615']\
                                    - 1/2 * coeff_dict_shifted['C616']\
                                    - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63d'] = - self.Ychi/8*coeff_dict_shifted['C611']\
                                    - coeff_dict_shifted['C621']/2\
                                    + coeff_dict_shifted['C641']/2\
                                    + self.Ychi/8 * coeff_dict_shifted['C615']\
                                    + 1/2 * coeff_dict_shifted['C616']\
                                    + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63s'] = - self.Ychi/8*coeff_dict_shifted['C612']\
                                    - coeff_dict_shifted['C622']/2\
                                    + coeff_dict_shifted['C642']/2\
                                    + self.Ychi/8 * coeff_dict_shifted['C615']\
                                    + 1/2 * coeff_dict_shifted['C616']\
                                    + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63c'] = self.Ychi/8*coeff_dict_shifted['C612']\
                                    - coeff_dict_shifted['C622']/2\
                                    + coeff_dict_shifted['C632']/2\
                                    - self.Ychi/8 * coeff_dict_shifted['C615']\
                                    - 1/2 * coeff_dict_shifted['C616']\
                                    - 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63b'] = - self.Ychi/8*coeff_dict_shifted['C613']\
                                    - coeff_dict_shifted['C623']/2\
                                    + coeff_dict_shifted['C643']/2\
                                    + self.Ychi/8 * coeff_dict_shifted['C615']\
                                    + 1/2 * coeff_dict_shifted['C616']\
                                    + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63e'] = - self.Ychi/8*coeff_dict_shifted['C691']\
                                    - coeff_dict_shifted['C6101']/2\
                                    + coeff_dict_shifted['C6111']/2\
                                    + self.Ychi/8 * coeff_dict_shifted['C615']\
                                    + 1/2 * coeff_dict_shifted['C616']\
                                    + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63mu'] = - self.Ychi/8*coeff_dict_shifted['C692']\
                                     - coeff_dict_shifted['C6102']/2\
                                     + coeff_dict_shifted['C6112']/2\
                                     + self.Ychi/8 * coeff_dict_shifted['C615']\
                                     + 1/2 * coeff_dict_shifted['C616']\
                                     + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4
            coeff_dict_5f['C63tau'] = - self.Ychi/8*coeff_dict_shifted['C693']\
                                      - coeff_dict_shifted['C6103']/2\
                                      + coeff_dict_shifted['C6113']/2\
                                      + self.Ychi/8 * coeff_dict_shifted['C615']\
                                      + 1/2 * coeff_dict_shifted['C616']\
                                      + 1/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4

            coeff_dict_5f['C64u'] = self.Ychi/8*coeff_dict_shifted['C651']\
                                    - coeff_dict_shifted['C661']/2\
                                    + coeff_dict_shifted['C671']/2\
                                    - self.Ychi/8 * coeff_dict_shifted['C617']\
                                    - 1/2 * coeff_dict_shifted['C618']\
                                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64d'] = - self.Ychi/8*coeff_dict_shifted['C651']\
                                    - coeff_dict_shifted['C661']/2\
                                    + coeff_dict_shifted['C681']/2\
                                    + self.Ychi/8 * coeff_dict_shifted['C617']\
                                    + 1/2 * coeff_dict_shifted['C618']\
                                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64s'] = - self.Ychi/8*coeff_dict_shifted['C652']\
                                    - coeff_dict_shifted['C662']/2\
                                    + coeff_dict_shifted['C682']/2\
                                    + self.Ychi/8 * coeff_dict_shifted['C617']\
                                    + 1/2 * coeff_dict_shifted['C618']\
                                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64c'] = self.Ychi/8*coeff_dict_shifted['C652']\
                                    - coeff_dict_shifted['C662']/2\
                                    + coeff_dict_shifted['C672']/2\
                                    - self.Ychi/8 * coeff_dict_shifted['C617']\
                                    - 1/2 * coeff_dict_shifted['C618']\
                                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64b'] = - self.Ychi/8*coeff_dict_shifted['C653']\
                                    - coeff_dict_shifted['C663']/2\
                                    + coeff_dict_shifted['C683']/2\
                                    + self.Ychi/8 * coeff_dict_shifted['C617']\
                                    + 1/2 * coeff_dict_shifted['C618']\
                                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64e'] = - self.Ychi/8*coeff_dict_shifted['C6121']\
                                    - coeff_dict_shifted['C6131']/2\
                                    + coeff_dict_shifted['C6141']/2\
                                    + self.Ychi/8 * coeff_dict_shifted['C617']\
                                    + 1/2 * coeff_dict_shifted['C618']\
                                    + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64mu'] = - self.Ychi/8*coeff_dict_shifted['C6122']\
                                     - coeff_dict_shifted['C6132']/2\
                                     + coeff_dict_shifted['C6142']/2\
                                     + self.Ychi/8 * coeff_dict_shifted['C617']\
                                     + 1/2 * coeff_dict_shifted['C618']\
                                     + W_box_fermion(self.dchi) * DIM4
            coeff_dict_5f['C64tau'] = - self.Ychi/8*coeff_dict_shifted['C6123']\
                                      - coeff_dict_shifted['C6133']/2\
                                      + coeff_dict_shifted['C6143']/2\
                                      + self.Ychi/8 * coeff_dict_shifted['C617']\
                                      + 1/2 * coeff_dict_shifted['C618']\
                                      + W_box_fermion(self.dchi) * DIM4

            coeff_dict_5f['C71'] = 1/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4\
                                              * coeff_dict_shifted['C54'])\
                                   + higgs_penguin_gluon(self.dchi) * DIM4
            coeff_dict_5f['C72'] = 1/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4\
                                              * coeff_dict_shifted['C58'])

            coeff_dict_5f['C75u'] = - 1/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C54'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75d'] = - 1/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C54'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75s'] = - 1/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C54'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75c'] = - 1/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C54'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75b'] = - 1/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C54'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75e'] = - 1/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C54'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75mu'] = - 1/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4\
                                                  * coeff_dict_shifted['C54'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4
            coeff_dict_5f['C75tau'] = - 1/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4\
                                                   * coeff_dict_shifted['C54'])\
                                    + higgs_penguin_fermion(self.dchi) * DIM4

            coeff_dict_5f['C76u'] = - 1/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C58'])
            coeff_dict_5f['C76d'] = - 1/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C58'])
            coeff_dict_5f['C76s'] = - 1/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C58'])
            coeff_dict_5f['C76c'] = - 1/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C58'])
            coeff_dict_5f['C76b'] = - 1/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C58'])
            coeff_dict_5f['C76e'] = - 1/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4\
                                                 * coeff_dict_shifted['C58'])
            coeff_dict_5f['C76mu'] = - 1/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4\
                                                  * coeff_dict_shifted['C58'])
            coeff_dict_5f['C76tau'] = - 1/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4\
                                                   * coeff_dict_shifted['C58'])

        coeff_dict_5f['C73'] = 0
        coeff_dict_5f['C74'] = 0

        coeff_dict_5f['C77u'] = 0
        coeff_dict_5f['C77d'] = 0
        coeff_dict_5f['C77s'] = 0
        coeff_dict_5f['C77c'] = 0
        coeff_dict_5f['C77b'] = 0
        coeff_dict_5f['C77e'] = 0
        coeff_dict_5f['C77mu'] = 0
        coeff_dict_5f['C77tau'] = 0

        coeff_dict_5f['C78u'] = 0
        coeff_dict_5f['C78d'] = 0
        coeff_dict_5f['C78s'] = 0
        coeff_dict_5f['C78c'] = 0
        coeff_dict_5f['C78b'] = 0
        coeff_dict_5f['C78e'] = 0
        coeff_dict_5f['C78mu'] = 0
        coeff_dict_5f['C78tau'] = 0

        coeff_dict_5f['C79u'] = 0
        coeff_dict_5f['C79d'] = 0
        coeff_dict_5f['C79s'] = 0
        coeff_dict_5f['C79c'] = 0
        coeff_dict_5f['C79b'] = 0
        coeff_dict_5f['C79e'] = 0
        coeff_dict_5f['C79mu'] = 0
        coeff_dict_5f['C79tau'] = 0

        coeff_dict_5f['C710u'] = 0
        coeff_dict_5f['C710d'] = 0
        coeff_dict_5f['C710s'] = 0
        coeff_dict_5f['C710c'] = 0
        coeff_dict_5f['C710b'] = 0
        coeff_dict_5f['C710e'] = 0
        coeff_dict_5f['C710mu'] = 0
        coeff_dict_5f['C710tau'] = 0

        coeff_dict_5f['C711'] = 0
        coeff_dict_5f['C712'] = 0
        coeff_dict_5f['C713'] = 0
        coeff_dict_5f['C714'] = 0

        coeff_dict_5f['C715u'] = 0
        coeff_dict_5f['C715d'] = 0
        coeff_dict_5f['C715s'] = 0
        coeff_dict_5f['C715c'] = 0
        coeff_dict_5f['C715b'] = 0
        coeff_dict_5f['C715e'] = 0
        coeff_dict_5f['C715mu'] = 0
        coeff_dict_5f['C715tau'] = 0

        coeff_dict_5f['C716u'] = 0
        coeff_dict_5f['C716d'] = 0
        coeff_dict_5f['C716s'] = 0
        coeff_dict_5f['C716c'] = 0
        coeff_dict_5f['C716b'] = 0
        coeff_dict_5f['C716e'] = 0
        coeff_dict_5f['C716mu'] = 0
        coeff_dict_5f['C716tau'] = 0

        coeff_dict_5f['C717u'] = 0
        coeff_dict_5f['C717d'] = 0
        coeff_dict_5f['C717s'] = 0
        coeff_dict_5f['C717c'] = 0
        coeff_dict_5f['C717b'] = 0
        coeff_dict_5f['C717e'] = 0
        coeff_dict_5f['C717mu'] = 0
        coeff_dict_5f['C717tau'] = 0

        coeff_dict_5f['C718u'] = 0
        coeff_dict_5f['C718d'] = 0
        coeff_dict_5f['C718s'] = 0
        coeff_dict_5f['C718c'] = 0
        coeff_dict_5f['C718b'] = 0
        coeff_dict_5f['C718e'] = 0
        coeff_dict_5f['C718mu'] = 0
        coeff_dict_5f['C718tau'] = 0

        coeff_dict_5f['C719u'] = 0
        coeff_dict_5f['C719d'] = 0
        coeff_dict_5f['C719s'] = 0
        coeff_dict_5f['C719c'] = 0
        coeff_dict_5f['C719b'] = 0
        coeff_dict_5f['C719e'] = 0
        coeff_dict_5f['C719mu'] = 0
        coeff_dict_5f['C719tau'] = 0

        coeff_dict_5f['C720u'] = 0
        coeff_dict_5f['C720d'] = 0
        coeff_dict_5f['C720s'] = 0
        coeff_dict_5f['C720c'] = 0
        coeff_dict_5f['C720b'] = 0
        coeff_dict_5f['C720e'] = 0
        coeff_dict_5f['C720mu'] = 0
        coeff_dict_5f['C720tau'] = 0

        coeff_dict_5f['C721u'] = 0
        coeff_dict_5f['C721d'] = 0
        coeff_dict_5f['C721s'] = 0
        coeff_dict_5f['C721c'] = 0
        coeff_dict_5f['C721b'] = 0
        coeff_dict_5f['C721e'] = 0
        coeff_dict_5f['C721mu'] = 0
        coeff_dict_5f['C721tau'] = 0

        coeff_dict_5f['C722u'] = 0
        coeff_dict_5f['C722d'] = 0
        coeff_dict_5f['C722s'] = 0
        coeff_dict_5f['C722c'] = 0
        coeff_dict_5f['C722b'] = 0
        coeff_dict_5f['C722e'] = 0
        coeff_dict_5f['C722mu'] = 0
        coeff_dict_5f['C722tau'] = 0

        return coeff_dict_5f


    def _my_cNR(self, DM_mass, mu_Lambda, RGE=None, NLO=None, DM_mass_threshold=None, RUN_EW=None, DIM4=None):
        """ Calculate the NR coefficients from four-flavor theory with meson contributions split off

        (mainly for internal use)
        """
        return WC_5flavor(self.match(DM_mass, mu_Lambda, DM_mass_threshold, RUN_EW, DIM4),\
                          self.DM_type, self.ip)._my_cNR(self.DM_mass_phys, RGE, NLO)

    def cNR(self, DM_mass, qvec, mu_Lambda, RGE=None, NLO=None, DM_mass_threshold=None, RUN_EW=None, DIM4=None):
        """ Calculate the NR coefficients from four-flavor theory """
        return WC_5flavor(self.match(DM_mass, mu_Lambda, DM_mass_threshold, RUN_EW, DIM4),\
                          self.DM_type, self.ip).cNR(DM_mass, qvec, RGE, NLO)

    def write_mma(self, DM_mass, qvec, mu_Lambda, RGE=None, NLO=None,\
                  DM_mass_threshold=None, RUN_EW=None, DIM4=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass DM_mass (in GeV) and the momentum transfer qvec (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is '.')

        <filename> is the filename (default 'cNR.m')
        """
        WC_5flavor(self.match(DM_mass, mu_Lambda, DM_mass_threshold, RUN_EW, DIM4),\
                   self.DM_type, self.ip).write_mma(DM_mass, qvec, RGE, NLO, path, filename)




