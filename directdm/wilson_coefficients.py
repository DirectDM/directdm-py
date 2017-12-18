#!/usr/bin/env python3

import sys
import numpy as np
import scipy.integrate as spint
import warnings
import os.path
from directdm.run import adm
from directdm.run import rge
from directdm.num.num_input import Num_input
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


class WC_3f(object):
    def __init__(self, coeff_dict, DM_type=None):
        """ Class for Wilson coefficients in 3 flavor QCD x QED plus DM.

        The first argument should be a dictionary for the initial conditions of the 2 + 24 + 4 + 36 + 4 + 48 + 6 = 124 
        dimension-five to dimension-seven three-flavor-QCD Wilson coefficients of the form
        {'C51' : value, 'C52' : value, ...}. 
        An arbitrary number of them can be given; the default values are zero. 

        The second argument is the DM type; it can take the following values: 
            "D" (Dirac fermion; this is the default)
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

        Complex Scalar:      'C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                             'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                             'C65', 'C66', 'C67', 'C68' 

        Real Scalar:         'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                             'C65', 'C66', 'C67', 'C68'

        (the notation corresponds to the numbering in 1707.06998).
        The Wilson coefficients should be specified in the MS-bar scheme at 2 GeV.

        The class has three methods:

        run
        ---
        Run the Wilson coefficients from mu = 2 GeV to mu_low [GeV; default 2 GeV], with 3 active quark flavors


        cNR
        ---
        Calculate the cNR coefficients as defined in 1308.6288

        The class has two mandatory arguments: The DM mass in GeV and the momentum transfer in GeV


        write_mma
        ---------
        Write an output file that can be loaded into mathematica, 
        to be used in the DMFormFactor package [1308.6288].

        """
        if DM_type is None:
            DM_type = "D"
        self.DM_type = DM_type

        if self.DM_type == "D":
            self.wc_name_list = ['C51', 'C52', 'C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                 'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau', 'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau', 'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau',
                                 'C79u', 'C79d', 'C79s', 'C79e', 'C79mu', 'C79tau', 'C710u', 'C710d', 'C710s', 'C710e', 'C710mu', 'C710tau',
                                 'C711', 'C712', 'C713', 'C714',
                                 'C715u', 'C715d', 'C715s', 'C715e', 'C715mu', 'C715tau', 'C716u', 'C716d', 'C716s', 'C716e', 'C716mu', 'C716tau',
                                 'C717u', 'C717d', 'C717s', 'C717e', 'C717mu', 'C717tau', 'C718u', 'C718d', 'C718s', 'C718e', 'C718mu', 'C718tau',
                                 'C719u', 'C719d', 'C719s', 'C719e', 'C719mu', 'C719tau', 'C720u', 'C720d', 'C720s', 'C720e', 'C720mu', 'C720tau', 
                                 'C721u', 'C721d', 'C721s', 'C721e', 'C721mu', 'C721tau', 'C722u', 'C722d', 'C722s', 'C722e', 'C722mu', 'C722tau']

            self.wc8_name_list = ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s', 'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s']

        if self.DM_type == "M":
            self.wc_name_list = ['C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau', 'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau', 'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau',
                                 'C711', 'C712', 'C713', 'C714',
                                 'C715u', 'C715d', 'C715s', 'C715e', 'C715mu', 'C715tau', 'C716u', 'C716d', 'C716s', 'C716e', 'C716mu', 'C716tau',
                                 'C717u', 'C717d', 'C717s', 'C717e', 'C717mu', 'C717tau', 'C718u', 'C718d', 'C718s', 'C718e', 'C718mu', 'C718tau']

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [i for i in range(0,8)] + [i for i in range(14,20)] + [i for i in range(54,66)] + [i for i in range(94,118)]

        if self.DM_type == "C":
            self.wc_name_list = ['C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 
                                 'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                 'C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                 'C67', 'C68']

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [0,1] + [i for i in range(8,14)] + [i for i in range(20,26)] + [27] + [29] + [i for i in range(36,42)]\
                           + [i for i in range(48,66)] + [67] + [69] + [i for i in range(70,118)]

        if self.DM_type == "R":
            self.wc_name_list = ['C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                 'C67', 'C68']

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [i for i in range(0,26)] + [27] + [29] + [i for i in range(36,42)] + [i for i in range(48,66)]\
                           + [67] + [69] + [i for i in range(70,118)]

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

        # Create the np.array of coefficients:
        self.coeff_list_dm_dim6_dim7 = np.array(dict_to_list(self.coeff_dict, self.wc_name_list))
        self.coeff_list_dm_dim8 = np.array(dict_to_list(self.coeff_dict, self.wc8_name_list))



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
        if self.DM_type == "C":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(3), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(3), del_ind_list, 1), del_ind_list, 2)
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

        ip = Num_input()
        alpha_at_mu = 1/ip.amtauinv

        as31 = rge.AlphaS(3,1)
        evolve1 = rge.RGE(self.gamma_QCD, 3)
        evolve2 = rge.RGE(self.gamma_QCD2, 3)
        evolve8 = rge.RGE([self.gamma_QCD_dim8], 3)

        C_at_mu_QCD = np.dot(evolve2.U0_as2(as31.run(2),as31.run(mu_low)), np.dot(evolve1.U0(as31.run(2),as31.run(mu_low)), self.coeff_list_dm_dim6_dim7))
        C_at_mu_QED = np.dot(self.coeff_list_dm_dim6_dim7, self.gamma_QED) * np.log(mu_low/2) * alpha_at_mu/(4*np.pi)\
                      + np.dot(self.coeff_list_dm_dim6_dim7, self.gamma_QED2) * np.log(mu_low/2) * (alpha_at_mu/(4*np.pi))**2
        C_dim8_at_mu = np.dot(evolve8.U0(as31.run(2),as31.run(mu_low)), self.coeff_list_dm_dim8)

        # Revert back to dictionary

        dict_coeff_mu = list_to_dict(C_at_mu_QCD + C_at_mu_QED, self.wc_name_list)
        dict_dm_dim8 = list_to_dict(C_dim8_at_mu, self.wc8_name_list)

        dict_coeff_mu.update(dict_dm_dim8)

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
        ip = Num_input()

        mpi = ip.mpi0
        mp = ip.mproton
        mn = ip.mneutron
        mN = (mp+mn)/2

        alpha = 1/ip.alowinv
        GF = ip.GF
        as_2GeV = rge.AlphaS(3,1).run(2)
        gs2_2GeV = 4*np.pi*rge.AlphaS(3,1).run(2)

        # Quark masses at 2GeV
        mu = ip.mu_at_2GeV
        md = ip.md_at_2GeV
        ms = ip.ms_at_2GeV
        mtilde = ip.mtilde

        ### Numerical constants
        ip = Num_input()

        mproton = ip.mproton
        mneutron = ip.mneutron

        F1up = F1('u', 'p').value_zero_mom()
        F1dp = F1('d', 'p').value_zero_mom()
        F1sp = F1('s', 'p').value_zero_mom()

        F1un = F1('u', 'n').value_zero_mom()
        F1dn = F1('d', 'n').value_zero_mom()
        F1sn = F1('s', 'n').value_zero_mom()

        F2up = F2('u', 'p').value_zero_mom()
        F2dp = F2('d', 'p').value_zero_mom()
        F2sp = F2('s', 'p').value_zero_mom()

        F2un = F2('u', 'n').value_zero_mom()
        F2dn = F2('d', 'n').value_zero_mom()
        F2sn = F2('s', 'n').value_zero_mom()

        FAup = FA('u', 'p').value_zero_mom()
        FAdp = FA('d', 'p').value_zero_mom()
        FAsp = FA('s', 'p').value_zero_mom()

        FAun = FA('u', 'n').value_zero_mom()
        FAdn = FA('d', 'n').value_zero_mom()
        FAsn = FA('s', 'n').value_zero_mom()

        FPpup_pion = FPprimed('u', 'p').value_pion_pole()
        FPpdp_pion = FPprimed('d', 'p').value_pion_pole()
        FPpsp_pion = FPprimed('s', 'p').value_pion_pole()

        FPpun_pion = FPprimed('u', 'n').value_pion_pole()
        FPpdn_pion = FPprimed('d', 'n').value_pion_pole()
        FPpsn_pion = FPprimed('s', 'n').value_pion_pole()

        FPpup_eta = FPprimed('u', 'p').value_eta_pole()
        FPpdp_eta = FPprimed('d', 'p').value_eta_pole()
        FPpsp_eta = FPprimed('s', 'p').value_eta_pole()

        FPpun_eta = FPprimed('u', 'n').value_eta_pole()
        FPpdn_eta = FPprimed('d', 'n').value_eta_pole()
        FPpsn_eta = FPprimed('s', 'n').value_eta_pole()

        FSup = FS('u', 'p').value_zero_mom()
        FSdp = FS('d', 'p').value_zero_mom()
        FSsp = FS('s', 'p').value_zero_mom()

        FSun = FS('u', 'n').value_zero_mom()
        FSdn = FS('d', 'n').value_zero_mom()
        FSsn = FS('s', 'n').value_zero_mom()

        FPup_pion = FP('u', 'p').value_pion_pole()
        FPdp_pion = FP('d', 'p').value_pion_pole()
        FPsp_pion = FP('s', 'p').value_pion_pole()

        FPun_pion = FP('u', 'n').value_pion_pole()
        FPdn_pion = FP('d', 'n').value_pion_pole()
        FPsn_pion = FP('s', 'n').value_pion_pole()

        FPup_eta = FP('u', 'p').value_eta_pole()
        FPdp_eta = FP('d', 'p').value_eta_pole()
        FPsp_eta = FP('s', 'p').value_eta_pole()

        FPun_eta = FP('u', 'n').value_eta_pole()
        FPdn_eta = FP('d', 'n').value_eta_pole()
        FPsn_eta = FP('s', 'n').value_eta_pole()

        FGp = FG('p').value_zero_mom()
        FGn = FG('n').value_zero_mom()

        FGtildep = FGtilde('p').value_zero_mom()
        FGtilden = FGtilde('n').value_zero_mom()

        FGtildep_pion = FGtilde('p').value_pion_pole()
        FGtilden_pion = FGtilde('n').value_pion_pole()

        FGtildep_eta = FGtilde('p').value_eta_pole()
        FGtilden_eta = FGtilde('n').value_eta_pole()

        FT0up = FT0('u', 'p').value_zero_mom()
        FT0dp = FT0('d', 'p').value_zero_mom()
        FT0sp = FT0('s', 'p').value_zero_mom()

        FT0un = FT0('u', 'n').value_zero_mom()
        FT0dn = FT0('d', 'n').value_zero_mom()
        FT0sn = FT0('s', 'n').value_zero_mom()

        FT1up = FT1('u', 'p').value_zero_mom()
        FT1dp = FT1('d', 'p').value_zero_mom()
        FT1sp = FT1('s', 'p').value_zero_mom()

        FT1un = FT1('u', 'n').value_zero_mom()
        FT1dn = FT1('d', 'n').value_zero_mom()
        FT1sn = FT1('s', 'n').value_zero_mom()



        ### The coefficients ###
        #
        # Note that all dependence on 1/q^2, 1/(m^2-q^2), q^2/(m^2-q^2) is taken care of by defining spurious operators.
        #
        # Therefore, we need to split some of the coefficients
        # into the "pion part" etc. with the q-dependence factored out, and introduce a few spurious "long-distance" operators.
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
                      + F1dp*(c3mu_dict['C61d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C81d']) + FGp*c3mu_dict['C71']\
                      + FSup*c3mu_dict['C75u'] + FSdp*c3mu_dict['C75d'] + FSsp*c3mu_dict['C75s']\
                      - alpha/(2*np.pi*DM_mass)*c3mu_dict['C51']\
                      + 2*DM_mass * (F1up*c3mu_dict['C715u'] + F1dp*c3mu_dict['C715d'] + F1sp*c3mu_dict['C715s']),
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : - 4*(  FAup*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                           + FAdp*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                           + FAsp*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s']))\
                      - 2*alpha/np.pi * ip.mup/mN * c3mu_dict['C51']\
                      + 8*(FT0up*c3mu_dict['C79u'] + FT0dp*c3mu_dict['C79d'] + FT0sp*c3mu_dict['C79s']),
            'cNR5p' : - 2*mN * (F1up*c3mu_dict['C719u'] + F1dp*c3mu_dict['C719d'] + F1sp*c3mu_dict['C719s']),
            'cNR6p' : mN/DM_mass * FGtildep * c3mu_dict['C74']\
                      -2*mN*((F1up+F2up)*c3mu_dict['C719u'] + (F1dp+F2dp)*c3mu_dict['C719d'] + (F1sp+F2dp)*c3mu_dict['C719s']),
            'cNR7p' : - 2*(  FAup*(c3mu_dict['C63u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C83u'])\
                           + FAdp*(c3mu_dict['C63d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C83d'])\
                           + FAsp*(c3mu_dict['C63s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C83s']))\
                      - 4*DM_mass * (FAup*c3mu_dict['C717u'] + FAdp*c3mu_dict['C717d'] + FAsp*c3mu_dict['C717s']),
            'cNR8p' : 2*(  F1up*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + F1dp*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])),
            'cNR9p' : 2*(  (F1up+F2up)*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + (F1dp+F2dp)*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + (F1sp+F2sp)*(c3mu_dict['C62s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C82s']))\
                      + 2*mN*(  FAup*(c3mu_dict['C63u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C83u'])\
                              + FAdp*(c3mu_dict['C63d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C83d'])\
                              + FAsp*(c3mu_dict['C63s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C83s']))/DM_mass\
                      - 4*mN * (FAup*c3mu_dict['C721u'] + FAdp*c3mu_dict['C721d'] + FAsp*c3mu_dict['C721s']),
            'cNR10p' : FGtildep * c3mu_dict['C73']\
                       -2*mN/DM_mass * (FT0up*c3mu_dict['C710u'] + FT0dp*c3mu_dict['C710d'] + FT0sp*c3mu_dict['C710s']),
            'cNR11p' : - mN/DM_mass * (FSup*c3mu_dict['C76u'] + FSdp*c3mu_dict['C76d'] + FSsp*c3mu_dict['C76s'])\
                       - mN/DM_mass * FGp * c3mu_dict['C72']\
                        + 2*((FT0up-FT1up)*c3mu_dict['C710u'] + (FT0dp-FT1dp)*c3mu_dict['C710d'] + (FT0sp-FT1sp)*c3mu_dict['C710s'])\
                        - 2*mN * (  F1up*(c3mu_dict['C716u']+c3mu_dict['C720u'])\
                                  + F1dp*(c3mu_dict['C716d']+c3mu_dict['C720d'])\
                                  + F1sp*(c3mu_dict['C716s']+c3mu_dict['C720s'])),
            'cNR12p' : -8*(FT0up*c3mu_dict['C710u'] + FT0dp*c3mu_dict['C710d'] + FT0sp*c3mu_dict['C710s']),
    
            'cNR13p' : mN/DM_mass * (FPup_pion*c3mu_dict['C78u'] + FPdp_pion*c3mu_dict['C78d'])\
                       + FPpup_pion*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdp_pion*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d']),
            'cNR14p' : mN/DM_mass * (FPup_eta*c3mu_dict['C78u'] + FPdp_eta*c3mu_dict['C78d'] + FPsp_eta*c3mu_dict['C78s'])\
                       + FPpup_eta*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdp_eta*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpsp_eta*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
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
            'cNR22p' : -mN**2* (- 2*alpha/np.pi * ip.mup/mN * c3mu_dict['C51']),
            'cNR23p' : mN* (2*alpha/np.pi*c3mu_dict['C52']),

            'cNR100p' : (F1up*c3mu_dict['C719u'] + F1dp*c3mu_dict['C719d'] + F1sp*c3mu_dict['C719s'])/(2*DM_mass),
            'cNR104p' : 2*((F1up+F2up)*c3mu_dict['C719u'] + (F1dp+F2dp)*c3mu_dict['C719d'] + (F1sp+F2dp)*c3mu_dict['C719s'])/mN,




            'cNR1n' :   F1un*(c3mu_dict['C61u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C81u'])\
                      + F1dn*(c3mu_dict['C61d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C81d']) + FGn*c3mu_dict['C71']\
                      + FSun*c3mu_dict['C75u'] + FSdn*c3mu_dict['C75d'] + FSsn*c3mu_dict['C75s']\
                      + 2*DM_mass * (F1un*c3mu_dict['C715u'] + F1dn*c3mu_dict['C715d'] + F1sn*c3mu_dict['C715s']),
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : - 4*(  FAun*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                           + FAdn*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                           + FAsn*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s']))\
                      - 2*alpha/np.pi * ip.mun/mN * c3mu_dict['C51']\
                      + 8*(FT0un*c3mu_dict['C79u'] + FT0dn*c3mu_dict['C79d'] + FT0sn*c3mu_dict['C79s']),
            'cNR5n' : - 2*mN * (F1un*c3mu_dict['C719u'] + F1dn*c3mu_dict['C719d'] + F1sn*c3mu_dict['C719s']),
            'cNR6n' : mN/DM_mass * FGtilden * c3mu_dict['C74']\
                      -2*mN*((F1un+F2un)*c3mu_dict['C719u'] + (F1dn+F2dn)*c3mu_dict['C719d'] + (F1sn+F2dn)*c3mu_dict['C719s']),
            'cNR7n' : - 2*(  FAun*(c3mu_dict['C63u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C83u'])\
                           + FAdn*(c3mu_dict['C63d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C83d'])\
                           + FAsn*(c3mu_dict['C63s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C83s']))\
                      - 4*DM_mass * (FAun*c3mu_dict['C717u'] + FAdn*c3mu_dict['C717d']+ FAsn*c3mu_dict['C717s']),
            'cNR8n' : 2*(  F1un*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + F1dn*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])),
            'cNR9n' : 2*(  (F1un+F2un)*(c3mu_dict['C62u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C82u'])\
                         + (F1dn+F2dn)*(c3mu_dict['C62d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C82d'])\
                         + (F1sn+F2sn)*(c3mu_dict['C62s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C82s']))\
                      + 2*mN*(  FAun*(c3mu_dict['C63u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C83u'])\
                              + FAdn*(c3mu_dict['C63d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C83d'])\
                              + FAsn*(c3mu_dict['C63s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C83s']))/DM_mass\
                      - 4*mN * (FAun*c3mu_dict['C721u'] + FAdn*c3mu_dict['C721d'] + FAsn*c3mu_dict['C721s']),
            'cNR10n' : FGtilden * c3mu_dict['C73']\
                     -2*mN/DM_mass * (FT0un*c3mu_dict['C710u'] + FT0dn*c3mu_dict['C710d'] + FT0sn*c3mu_dict['C710s']),
            'cNR11n' : - mN/DM_mass * (FSun*c3mu_dict['C76u'] + FSdn*c3mu_dict['C76d'] + FSsn*c3mu_dict['C76s'])\
                       - mN/DM_mass * FGn * c3mu_dict['C72']\
                       + 2*((FT0un-FT1un)*c3mu_dict['C710u'] + (FT0dn-FT1dn)*c3mu_dict['C710d'] + (FT0sn-FT1sn)*c3mu_dict['C710s'])\
                       - 2*mN * (  F1un*(c3mu_dict['C716u']+c3mu_dict['C720u'])\
                                 + F1dn*(c3mu_dict['C716d']+c3mu_dict['C720d'])\
                                 + F1sn*(c3mu_dict['C716s']+c3mu_dict['C720s'])),
            'cNR12n' : -8*(FT0un*c3mu_dict['C710u'] + FT0dn*c3mu_dict['C710d'] + FT0sn*c3mu_dict['C710s']),
    
            'cNR13n' : mN/DM_mass * (FPun_pion*c3mu_dict['C78u'] + FPdn_pion*c3mu_dict['C78d'])\
                       + FPpun_pion*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdn_pion*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d']),
            'cNR14n' : mN/DM_mass * (FPun_eta*c3mu_dict['C78u'] + FPdn_eta*c3mu_dict['C78d'] + FPsn_eta*c3mu_dict['C78s'])\
                       + FPpun_eta*(c3mu_dict['C64u'] - np.sqrt(2)*GF*mu**2 / gs2_2GeV * c3mu_dict['C84u'])\
                       + FPpdn_eta*(c3mu_dict['C64d'] - np.sqrt(2)*GF*md**2 / gs2_2GeV * c3mu_dict['C84d'])\
                       + FPpsn_eta*(c3mu_dict['C64s'] - np.sqrt(2)*GF*ms**2 / gs2_2GeV * c3mu_dict['C84s'])\
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
            'cNR22n' : -mN**2 * (- 2*alpha/np.pi * ip.mun/mN * c3mu_dict['C51']),
            'cNR23n' : 0,

            'cNR100n' : (F1un*c3mu_dict['C719u'] + F1dn*c3mu_dict['C719d'] + F1sn*c3mu_dict['C719s'])/(2*DM_mass),
            'cNR104n' : 2*((F1un+F2un)*c3mu_dict['C719u'] + (F1dn+F2dn)*c3mu_dict['C719d'] + (F1sn+F2dn)*c3mu_dict['C719s'])/mN
            }

            if NLO:
                my_cNR_dict['cNR5p'] = - 2*mN * (F1un*c3mu_dict['C719u'] + F1dn*c3mu_dict['C719d'] + F1sn*c3mu_dict['C719s'])\
                                       + 2*((FT0up-FT1up)*c3mu_dict['C79u'] + (FT0dp-FT1dp)*c3mu_dict['C79d'] + (FT0sp-FT1sp)*c3mu_dict['C79s'])
                my_cNR_dict['cNR100p'] = - ((FT0up-FT1up)*c3mu_dict['C79u'] + (FT0dp-FT1dp)*c3mu_dict['C79d'] + (FT0sp-FT1sp)*c3mu_dict['C79s'])/(2*DM_mass*mN)
                my_cNR_dict['cNR5n'] = - 2*mN * (F1un*c3mu_dict['C719u'] + F1dn*c3mu_dict['C719d'] + F1sn*c3mu_dict['C719s'])\
                                       + 2*((FT0un-FT1un)*c3mu_dict['C79u'] + (FT0dn-FT1dn)*c3mu_dict['C79d'] + (FT0sn-FT1sn)*c3mu_dict['C79s'])
                my_cNR_dict['cNR100n'] = - ((FT0un-FT1un)*c3mu_dict['C79u'] + (FT0dn-FT1dn)*c3mu_dict['C79d'] + (FT0sn-FT1sn)*c3mu_dict['C79s'])/(2*DM_mass*mN)


        if self.DM_type == "M":
            my_cNR_dict = {
            'cNR1p' : FGp*c3mu_dict['C71']\
                      + FSup*c3mu_dict['C75u'] + FSdp*c3mu_dict['C75d'] + FSsp*c3mu_dict['C75s']\
                      + 2*DM_mass * (F1up*c3mu_dict['C715u'] + F1dp*c3mu_dict['C715d'] + F1sp*c3mu_dict['C715s']),
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : - 4*(FAup*c3mu_dict['C64u'] + FAdp*c3mu_dict['C64d'] + FAsp*c3mu_dict['C64s']),
            'cNR5p' : 0,
            'cNR6p' : mN/DM_mass * FGtildep * c3mu_dict['C74'],
            'cNR7p' : - 4*DM_mass * (FAup*c3mu_dict['C717u'] + FAdp*c3mu_dict['C717d'] + FAsp*c3mu_dict['C717s']),
            'cNR8p' : 2*(F1up*c3mu_dict['C62u'] + F1dp*c3mu_dict['C62d']),
            'cNR9p' : 2*((F1up+F2up)*c3mu_dict['C62u'] + (F1dp+F2dp)*c3mu_dict['C62d'] + (F1sp+F2sp)*c3mu_dict['C62s']),
            'cNR10p' : FGtildep * c3mu_dict['C73'],
            'cNR11p' : - mN/DM_mass * (FSup*c3mu_dict['C76u'] + FSdp*c3mu_dict['C76d'] + FSsp*c3mu_dict['C76s'])\
                       - mN/DM_mass * FGp * c3mu_dict['C72']\
                       - 2*mN * (  F1up*c3mu_dict['C716u'] + F1dp*c3mu_dict['C716d'] + F1sp*c3mu_dict['C716s']),
            'cNR12p' : 0,
    
            'cNR13p' : mN/DM_mass * (FPup_pion*c3mu_dict['C78u'] + FPdp_pion*c3mu_dict['C78d'])\
                       + FPpup_pion*c3mu_dict['C64u'] + FPpdp_pion*c3mu_dict['C64d'],
            'cNR14p' : mN/DM_mass * (FPup_eta*c3mu_dict['C78u'] + FPdp_eta*c3mu_dict['C78d'] + FPsp_eta*c3mu_dict['C78s'])\
                       + FPpup_eta*c3mu_dict['C64u'] + FPpdp_eta*c3mu_dict['C64d'] + FPpsp_eta*c3mu_dict['C64s']\
                       + 4*mN * (FAup*c3mu_dict['C718u'] + FAdp*c3mu_dict['C718d'] + FAsp*c3mu_dict['C718s']),
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
            'cNR4n' : - 4*(FAun*c3mu_dict['C64u'] + FAdn*c3mu_dict['C64d'] + FAsn*c3mu_dict['C64s']),
            'cNR5n' : 0,
            'cNR6n' : mN/DM_mass * FGtilden * c3mu_dict['C74'],
            'cNR7n' : - 4*DM_mass * (FAun*c3mu_dict['C717u'] + FAdn*c3mu_dict['C717d'] + FAsn*c3mu_dict['C717s']),
            'cNR8n' : 2*(F1un*c3mu_dict['C62u'] + F1dn*c3mu_dict['C62d']),
            'cNR9n' : 2*((F1un+F2un)*c3mu_dict['C62u'] + (F1dn+F2dn)*c3mu_dict['C62d'] + (F1sn+F2sn)*c3mu_dict['C62s']),
            'cNR10n' : FGtilden * c3mu_dict['C73'],
            'cNR11n' : - mN/DM_mass * (FSun*c3mu_dict['C76u'] + FSdn*c3mu_dict['C76d'] + FSsn*c3mu_dict['C76s'])\
                       - mN/DM_mass * FGn * c3mu_dict['C72']\
                       - 2*mN * (  F1un*c3mu_dict['C716u'] + F1dn*c3mu_dict['C716d'] + F1sn*c3mu_dict['C716s']),
            'cNR12n' : 0,
    
            'cNR13n' : mN/DM_mass * (FPun_pion*c3mu_dict['C78u'] + FPdn_pion*c3mu_dict['C78d'])\
                       + FPpun_pion*c3mu_dict['C64u'] + FPpdn_pion*c3mu_dict['C64d'],
            'cNR14n' : mN/DM_mass * (FPun_eta*c3mu_dict['C78u'] + FPdn_eta*c3mu_dict['C78d'] + FPsn_eta*c3mu_dict['C78s'])\
                       + FPpun_eta*c3mu_dict['C64u'] + FPpdn_eta*c3mu_dict['C64d'] + FPpsn_eta*c3mu_dict['C64s']\
                       + 4*mN * (FAun*c3mu_dict['C718u'] + FAdn*c3mu_dict['C718d'] + FAsn*c3mu_dict['C718s']),
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
            'cNR1p' :   2*DM_mass*(F1up*c3mu_dict['C61u'] + F1un*c3mu_dict['C61d']) + FGp*c3mu_dict['C65']\
                      + FSup*c3mu_dict['C63u'] + FSdp*c3mu_dict['C63d'] + FSsp*c3mu_dict['C63s'],
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : 0,
            'cNR5p' : 0,
            'cNR6p' : 0,
            'cNR7p' : -4*DM_mass*(FAup*c3mu_dict['C62u'] + FAdp*c3mu_dict['C62d'] + FAsp*c3mu_dict['C62s']),
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




            'cNR1n' :   2*DM_mass*(F1un*c3mu_dict['C61u'] + F1dn*c3mu_dict['C61d']) + FGn*c3mu_dict['C65']\
                      + FSun*c3mu_dict['C63u'] + FSdn*c3mu_dict['C63d'] + FSsn*c3mu_dict['C63s'],
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : 0,
            'cNR5n' : 0,
            'cNR6n' : 0,
            'cNR7n' : -4*DM_mass*(FAun*c3mu_dict['C62u'] + FAdn*c3mu_dict['C62d'] + FAsn*c3mu_dict['C62s']),
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
            'cNR1p' : FSup*c3mu_dict['C63u'] + FSdp*c3mu_dict['C63d'] + FSsp*c3mu_dict['C63s'] + FGp*c3mu_dict['C65'],
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
            'cNR18p' : FPup_eta*c3mu_dict['C64u'] + FPdp_eta*c3mu_dict['C64d'] + FPsp_eta*c3mu_dict['C64s'],
            'cNR19p' : FGtildep_pion * c3mu_dict['C66'],
            'cNR20p' : FGtildep_eta * c3mu_dict['C66'],
    
            'cNR21p' : 0,
            'cNR22p' : 0,
            'cNR23p' : 0,

            'cNR100p' : 0,
            'cNR104p' : 0,




            'cNR1n' : FSun*c3mu_dict['C63u'] + FSdn*c3mu_dict['C63d'] + FSsn*c3mu_dict['C63s'] + FGn*c3mu_dict['C65'],
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
            'cNR18n' : FPun_eta*c3mu_dict['C64u'] + FPdn_eta*c3mu_dict['C64d'] + FPsn_eta*c3mu_dict['C64s'],
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
        """ The operator coefficients of O_1^N -- O_12^N as in 1308.6288 -- multiply by propagators and sum up contributions 

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

        ip = Num_input()
        meta = ip.meta
        mpi = ip.mpi0

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


    def write_mma(self, DM_mass, q, RGE=None, NLO=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass DM_mass (in GeV) and the momentum transfer q (in GeV) 

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

        val = self.cNR(DM_mass, q, RGE, NLO)
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



class WC_4f(object):
    def __init__(self, coeff_dict, DM_type=None):
        """ Class for Wilson coefficients in 4 flavor QCD x QED plus DM.

        The argument should be a dictionary for the initial conditions of the 2 + 28 + 4 + 42 + 4 + 56 + 6 = 142 
        dimension-five to dimension-eight four-flavor-QCD Wilson coefficients (for Dirac DM) of the form
        {'C51' : value, 'C52' : value, ...}. For other DM types there are less coefficients.
        An arbitrary number of them can be given; the default values are zero. 

        The second argument is the DM type; it can take the following values: 
            "D" (Dirac fermion; this is the default)
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

         'P61ud', 'P62ud', 'P63ud', 'P63du', 'P64ud', 'P65ud', 'P66ud', 'P66du', 
         'P61us', 'P62us', 'P63us', 'P63su', 'P64us', 'P65us', 'P66us', 'P66su', 
         'P61uc', 'P62uc', 'P63uc', 'P63cu', 'P64uc', 'P65uc', 'P66uc', 'P66cu', 
         'P61ds', 'P62ds', 'P63ds', 'P63sd', 'P64ds', 'P65ds', 'P66ds', 'P66sd', 
         'P61dc', 'P62dc', 'P63dc', 'P63cd', 'P64dc', 'P65dc', 'P66dc', 'P66cd', 
         'P61sc', 'P62sc', 'P63sc', 'P63cs', 'P64sc', 'P65sc', 'P66sc', 'P66cs', 
         'P61u', 'P62u', 'P63u', 'P64u', 
         'P61d', 'P62d', 'P63d', 'P64d', 
         'P61s', 'P62s', 'P63s', 'P64s', 
         'P61c', 'P62c', 'P63c', 'P64c' 



        The class has three methods: 

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
        if DM_type is None:
            DM_type = "D"
        self.DM_type = DM_type


        # First, we define a standard ordering for the Wilson coefficients, so that we can use arrays

        self.sm_name_list = ['P61ud', 'P62ud', 'P63ud', 'P63du', 'P64ud', 'P65ud', 'P66ud', 'P66du', 
                             'P61us', 'P62us', 'P63us', 'P63su', 'P64us', 'P65us', 'P66us', 'P66su', 
                             'P61uc', 'P62uc', 'P63uc', 'P63cu', 'P64uc', 'P65uc', 'P66uc', 'P66cu', 
                             'P61ds', 'P62ds', 'P63ds', 'P63sd', 'P64ds', 'P65ds', 'P66ds', 'P66sd', 
                             'P61dc', 'P62dc', 'P63dc', 'P63cd', 'P64dc', 'P65dc', 'P66dc', 'P66cd', 
                             'P61sc', 'P62sc', 'P63sc', 'P63cs', 'P64sc', 'P65sc', 'P66sc', 'P66cs', 
                             'P61u', 'P62u', 'P63u', 'P64u', 
                             'P61d', 'P62d', 'P63d', 'P64d', 
                             'P61s', 'P62s', 'P63s', 'P64s', 
                             'P61c', 'P62c', 'P63c', 'P64c']

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

            self.wc8_name_list = ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s', 'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s']

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

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [i for i in range(0,9)] + [i for i in range(16,23)] + [i for i in range(62,76)] + [i for i in range(108,136)]

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
            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [0,1] + [i for i in range(9,16)] + [i for i in range(23,30)] + [31] + [33] + [i for i in range(41,48)]\
                           + [i for i in range(55,76)] + [77] + [79] + [i for i in range(80,136)]

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
            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [i for i in range(0,30)] + [31] + [33] + [i for i in range(41,48)] + [i for i in range(55,76)]\
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


        # Create the np.array of coefficients:
        self.coeff_list_dm_dim6_dim7 = np.array(dict_to_list(self.coeff_dict, self.wc_name_list))
        self.coeff_list_dm_dim8 = np.array(dict_to_list(self.coeff_dict, self.wc8_name_list))
        self.coeff_list_sm_dim6 = np.array(dict_to_list(self.coeff_dict, self.sm_name_list))




        #---------------------------#
        # The anomalous dimensions: #
        #---------------------------#

        if self.DM_type == "D":
            self.gamma_QED = adm.ADM_QED(4)
            self.gamma_QED2 = adm.ADM_QED2(4)
            self.gamma_QCD = adm.ADM_QCD(4)
            self.gamma_QCD2 = adm.ADM_QCD2(4)
            self.gamma_QCD_dim8 = adm.ADM_QCD_dim8(4)
            self.gamma_hat = adm.ADT_QCD(4)
        if self.DM_type == "M":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(4), del_ind_list, 1), del_ind_list, 2)
        if self.DM_type == "C":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(4), del_ind_list, 1), del_ind_list, 2)
        if self.DM_type == "R":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(4), del_ind_list, 1), del_ind_list, 2)

        self.ADM_SM = adm.ADM_SM_QCD(4)



        #--------------------------------------------------------------------#
        # The effective anomalous dimension for mixing into dimension eight: #
        #--------------------------------------------------------------------#

        # We need to contract the ADT with a subset of the dim.-6 Wilson coefficients
        if self.DM_type == "D":
            DM_dim6_init = np.delete(self.coeff_list_dm_dim6_dim7, np.r_[np.s_[0:16], np.s_[20:23], np.s_[27:136]])

        # The columns of ADM_eff correspond to SM6 operators; the rows of ADM_eff correspond to DM8 operators; 
        C6_dot_ADM_hat = np.transpose(np.tensordot(DM_dim6_init, self.gamma_hat, (0,2)))

        # The effective ADM
        #
        # Note that the mixing of the SM operators with four equal flavors does not contribute if we neglect yu, yd, ys! 

        self.ADM_eff = [np.vstack((np.hstack((self.ADM_SM, np.vstack((C6_dot_ADM_hat, np.zeros((16,12)))))),\
                              np.hstack((np.zeros((12,64)), self.gamma_QCD_dim8))))]




    def run(self, mu_low=None, double_QCD=None):
        """ Running of 4-flavor Wilson coefficients

        Calculate the running from mb(mb) to mu_low [GeV; default 2 GeV] in the four-flavor theory. 

        Return a dictionary of Wilson coefficients for the four-flavor Lagrangian
        at scale mu_low.
        """
        if mu_low is None:
            mu_low=2
        if double_QCD is None:
            double_QCD=True


        #-------------#
        # The running #
        #-------------#

        ip = Num_input()

        mb = ip.mb_at_mb
        alpha_at_mc = 1/ip.aMZinv

        if double_QCD:
            adm_eff = self.ADM_eff
        else:
            projector = np.vstack((np.hstack((np.zeros((64,64)), np.ones((64,12)))), np.zeros((12,76))))
            adm_eff = [np.multiply(projector, self.ADM_eff[0])]

        as41 = rge.AlphaS(4,1)
        evolve1 = rge.RGE(self.gamma_QCD, 4)
        evolve2 = rge.RGE(self.gamma_QCD2, 4)
        evolve8 = rge.RGE(adm_eff, 4)

        # Mixing in the dim.6 DM-SM sector
        #
        # Strictly speaking, mb should be defined at scale mu_low (however, this is a higher-order difference)
        C_at_mc_QCD = np.dot(evolve2.U0_as2(as41.run(mb),as41.run(mu_low)), np.dot(evolve1.U0(as41.run(mb),as41.run(mu_low)), self.coeff_list_dm_dim6_dim7))
        C_at_mc_QED = np.dot(self.coeff_list_dm_dim6_dim7, self.gamma_QED) * np.log(mu_low/mb) * alpha_at_mc/(4*np.pi)\
                      + np.dot(self.coeff_list_dm_dim6_dim7, self.gamma_QED2) * np.log(mu_low/mb) * (alpha_at_mc/(4*np.pi))**2

        # Mixing in the dim.6 SM-SM and dim.8 DM-SM sector

        DIM6_DIM8_init = np.hstack((self.coeff_list_sm_dim6, self.coeff_list_dm_dim8))

        DIM6_DIM8_at_mb = np.dot(evolve8.U0(as41.run(mb),as41.run(mu_low)), DIM6_DIM8_init)

        # Revert back to dictionary

        dict_coeff_mc = list_to_dict(C_at_mc_QCD + C_at_mc_QED, self.wc_name_list)
        dict_dm_dim8 = list_to_dict(np.delete(DIM6_DIM8_at_mb, np.s_[0:64]), self.wc8_name_list)
        dict_sm_dim6 = list_to_dict(np.delete(DIM6_DIM8_at_mb, np.s_[64:70]), self.sm_name_list)

        dict_coeff_mc.update(dict_dm_dim8)
        dict_coeff_mc.update(dict_sm_dim6)

        return dict_coeff_mc



    def match(self, mu=None, RGE=None, double_QCD=None):
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
        """ Calculate the NR coefficients from four-flavor theory with meson contributions split off (mainly for internal use) """
        return WC_3f(self.match(RGE, double_QCD), self.DM_type)._my_cNR(DM_mass, RGE, NLO)

    def cNR(self, DM_mass, qvec, RGE=None, NLO=None, double_QCD=None):
        """ Calculate the NR coefficients from four-flavor theory """
        return WC_3f(self.match(RGE, double_QCD), self.DM_type).cNR(DM_mass, qvec, RGE, NLO)

    def write_mma(self, DM_mass, q, RGE=None, NLO=None, double_QCD=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass DM_mass (in GeV) and the momentum transfer q (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is './')

        <filename> is the filename (default 'cNR.m')
        """
        WC_3f(self.match(RGE, double_QCD), self.DM_type).write_mma(DM_mass, q, RGE, NLO, path, filename)




class WC_5f(object):
    def __init__(self, coeff_dict, DM_type=None):
        """ Class for Wilson coefficients in 5 flavor QCD x QED plus DM.

        The argument should be a dictionary for the initial conditions of the 2 + 32 + 4 + 48 + 4 + 64 + 6 = 160 
        dimension-five to dimension-eight five-flavor-QCD Wilson coefficients (for Dirac DM) of the form
        {'C51' : value, 'C52' : value, ...}. For other DM types there are less coefficients.
        An arbitrary number of them can be given; the default values are zero. 
        The possible name are (with an hopefully obvious notation):

        The second argument is the DM type; it can take the following values: 
            "D" (Dirac fermion; this is the default)
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
        The following subset of 10*8 + 5*4 = 100 operators is sufficient for our purposes:

         'P61ud', 'P62ud', 'P63ud', 'P63du', 'P64ud', 'P65ud', 'P66ud', 'P66du', 
         'P61us', 'P62us', 'P63us', 'P63su', 'P64us', 'P65us', 'P66us', 'P66su', 
         'P61uc', 'P62uc', 'P63uc', 'P63cu', 'P64uc', 'P65uc', 'P66uc', 'P66cu', 
         'P61ub', 'P62ub', 'P63ub', 'P63bu', 'P64ub', 'P65ub', 'P66ub', 'P66bu', 
         'P61ds', 'P62ds', 'P63ds', 'P63sd', 'P64ds', 'P65ds', 'P66ds', 'P66sd', 
         'P61dc', 'P62dc', 'P63dc', 'P63cd', 'P64dc', 'P65dc', 'P66dc', 'P66cd', 
         'P61db', 'P62db', 'P63db', 'P63bd', 'P64db', 'P65db', 'P66db', 'P66bd', 
         'P61sc', 'P62sc', 'P63sc', 'P63cs', 'P64sc', 'P65sc', 'P66sc', 'P66cs', 
         'P61sb', 'P62sb', 'P63sb', 'P63bs', 'P64sb', 'P65sb', 'P66sb', 'P66bs', 
         'P61cb', 'P62cb', 'P63cb', 'P63bc', 'P64cb', 'P65cb', 'P66cb', 'P66bc',
         'P61u', 'P62u', 'P63u', 'P64u', 
         'P61d', 'P62d', 'P63d', 'P64d', 
         'P61s', 'P62s', 'P63s', 'P64s', 
         'P61c', 'P62c', 'P63c', 'P64c', 
         'P61b', 'P62b', 'P63b', 'P64b'



        The class has three methods: 

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

        if DM_type is None:
            DM_type = "D"
        self.DM_type = DM_type

        # First, we define a standard ordering for the Wilson coefficients, so that we can use arrays

        self.sm_name_list = ['P61ud', 'P62ud', 'P63ud', 'P63du', 'P64ud', 'P65ud', 'P66ud', 'P66du', 
                             'P61us', 'P62us', 'P63us', 'P63su', 'P64us', 'P65us', 'P66us', 'P66su', 
                             'P61uc', 'P62uc', 'P63uc', 'P63cu', 'P64uc', 'P65uc', 'P66uc', 'P66cu', 
                             'P61ub', 'P62ub', 'P63ub', 'P63bu', 'P64ub', 'P65ub', 'P66ub', 'P66bu', 
                             'P61ds', 'P62ds', 'P63ds', 'P63sd', 'P64ds', 'P65ds', 'P66ds', 'P66sd', 
                             'P61dc', 'P62dc', 'P63dc', 'P63cd', 'P64dc', 'P65dc', 'P66dc', 'P66cd', 
                             'P61db', 'P62db', 'P63db', 'P63bd', 'P64db', 'P65db', 'P66db', 'P66bd', 
                             'P61sc', 'P62sc', 'P63sc', 'P63cs', 'P64sc', 'P65sc', 'P66sc', 'P66cs', 
                             'P61sb', 'P62sb', 'P63sb', 'P63bs', 'P64sb', 'P65sb', 'P66sb', 'P66bs', 
                             'P61cb', 'P62cb', 'P63cb', 'P63bc', 'P64cb', 'P65cb', 'P66cb', 'P66bc',
                             'P61u', 'P62u', 'P63u', 'P64u', 
                             'P61d', 'P62d', 'P63d', 'P64d', 
                             'P61s', 'P62s', 'P63s', 'P64s', 
                             'P61c', 'P62c', 'P63c', 'P64c', 
                             'P61b', 'P62b', 'P63b', 'P64b']

        self.sm_name_list_4f = ['P61ud', 'P62ud', 'P63ud', 'P63du', 'P64ud', 'P65ud', 'P66ud', 'P66du', 
                                'P61us', 'P62us', 'P63us', 'P63su', 'P64us', 'P65us', 'P66us', 'P66su', 
                                'P61uc', 'P62uc', 'P63uc', 'P63cu', 'P64uc', 'P65uc', 'P66uc', 'P66cu', 
                                'P61ds', 'P62ds', 'P63ds', 'P63sd', 'P64ds', 'P65ds', 'P66ds', 'P66sd', 
                                'P61dc', 'P62dc', 'P63dc', 'P63cd', 'P64dc', 'P65dc', 'P66dc', 'P66cd', 
                                'P61sc', 'P62sc', 'P63sc', 'P63cs', 'P64sc', 'P65sc', 'P66sc', 'P66cs', 
                                'P61u', 'P62u', 'P63u', 'P64u', 
                                'P61d', 'P62d', 'P63d', 'P64d', 
                                'P61s', 'P62s', 'P63s', 'P64s', 
                                'P61c', 'P62c', 'P63c', 'P64c']

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

            self.wc8_name_list = ['C81u', 'C81d', 'C81s', 'C82u', 'C82d', 'C82s', 'C83u', 'C83d', 'C83s', 'C84u', 'C84d', 'C84s']

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

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [i for i in range(0,10)] + [i for i in range(18,26)] + [i for i in range(70,86)] + [i for i in range(122,154)]

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

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [0,1] + [i for i in range(10,18)] + [i for i in range(26,34)] + [35] + [37] + [i for i in range(46,54)]\
                           + [i for i in range(62,86)] + [87] + [89] + [i for i in range(90,154)]

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

            # The list of indices to be deleted from the QCD/QED ADM because of less operators
            del_ind_list = [i for i in range(0,34)] + [35] + [37] + [i for i in range(46,54)] + [i for i in range(62,86)]\
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
            else:
                warnings.warn('The key ' + wc_name + ' is not a valid key. Typo?')

        # Create the dictionary. 
        #
        # First, the default values (0 for DM operators, SM values for SM operators):
        #
        # This is actually conceptually not so good. The SM initial conditions should be moved to a matching method above the e/w scale.

        for wc_name in self.wc_name_list:
            self.coeff_dict[wc_name] = 0.
        for wc_name in self.wc8_name_list:
            self.coeff_dict[wc_name] = 0.

        ip = Num_input()

        sw = np.sqrt(ip.sw2_MSbar)
        cw = np.sqrt(1-sw**2)
        vd = (-1/2 - 2*sw**2*(-1/3))/(2*sw*cw)
        vu = (1/2 - 2*sw**2*(2/3))/(2*sw*cw)
        ad = -(-1/2)/(2*sw*cw)
        au = -(1/2)/(2*sw*cw)

        self.coeff_dict['P61ud'] = vu*vd * 4*sw**2*cw**2 + 1/6
        self.coeff_dict['P62ud'] = au*ad * 4*sw**2*cw**2 + 1/6
        self.coeff_dict['P63ud'] = au*vd * 4*sw**2*cw**2 - 1/6
        self.coeff_dict['P63du'] = ad*vu * 4*sw**2*cw**2 - 1/6
        self.coeff_dict['P64ud'] = 1
        self.coeff_dict['P65ud'] = 1
        self.coeff_dict['P66ud'] = -1
        self.coeff_dict['P66du'] = -1

        self.coeff_dict['P61us'] = vu*vd * 4*sw**2*cw**2
        self.coeff_dict['P62us'] = au*ad * 4*sw**2*cw**2
        self.coeff_dict['P63us'] = au*vd * 4*sw**2*cw**2
        self.coeff_dict['P63su'] = ad*vu * 4*sw**2*cw**2
        self.coeff_dict['P64us'] = 0
        self.coeff_dict['P65us'] = 0
        self.coeff_dict['P66us'] = 0
        self.coeff_dict['P66su'] = 0

        self.coeff_dict['P61uc'] = vu*vu * 4*sw**2*cw**2
        self.coeff_dict['P62uc'] = au*au * 4*sw**2*cw**2
        self.coeff_dict['P63uc'] = au*vu * 4*sw**2*cw**2
        self.coeff_dict['P63cu'] = au*vu * 4*sw**2*cw**2
        self.coeff_dict['P64uc'] = 0
        self.coeff_dict['P65uc'] = 0
        self.coeff_dict['P66uc'] = 0
        self.coeff_dict['P66cu'] = 0

        self.coeff_dict['P61ub'] = vu*vd * 4*sw**2*cw**2
        self.coeff_dict['P62ub'] = au*ad * 4*sw**2*cw**2
        self.coeff_dict['P63ub'] = au*vd * 4*sw**2*cw**2
        self.coeff_dict['P63bu'] = ad*vu * 4*sw**2*cw**2
        self.coeff_dict['P64ub'] = 0
        self.coeff_dict['P65ub'] = 0
        self.coeff_dict['P66ub'] = 0
        self.coeff_dict['P66bu'] = 0

        self.coeff_dict['P61ds'] = vd*vd * 4*sw**2*cw**2
        self.coeff_dict['P62ds'] = ad*ad * 4*sw**2*cw**2
        self.coeff_dict['P63ds'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['P63sd'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['P64ds'] = 0
        self.coeff_dict['P65ds'] = 0
        self.coeff_dict['P66ds'] = 0
        self.coeff_dict['P66sd'] = 0

        self.coeff_dict['P61dc'] = vd*vu * 4*sw**2*cw**2
        self.coeff_dict['P62dc'] = ad*au * 4*sw**2*cw**2
        self.coeff_dict['P63dc'] = ad*vu * 4*sw**2*cw**2
        self.coeff_dict['P63cd'] = au*vd * 4*sw**2*cw**2
        self.coeff_dict['P64dc'] = 0
        self.coeff_dict['P65dc'] = 0
        self.coeff_dict['P66dc'] = 0
        self.coeff_dict['P66cd'] = 0

        self.coeff_dict['P61db'] = vd*vd * 4*sw**2*cw**2
        self.coeff_dict['P62db'] = ad*ad * 4*sw**2*cw**2
        self.coeff_dict['P63db'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['P63bd'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['P64db'] = 0
        self.coeff_dict['P65db'] = 0
        self.coeff_dict['P66db'] = 0
        self.coeff_dict['P66bd'] = 0

        self.coeff_dict['P61sc'] = vd*vu * 4*sw**2*cw**2 + 1/6
        self.coeff_dict['P62sc'] = ad*au * 4*sw**2*cw**2 + 1/6
        self.coeff_dict['P63sc'] = ad*vu * 4*sw**2*cw**2 - 1/6
        self.coeff_dict['P63cs'] = au*vd * 4*sw**2*cw**2 - 1/6
        self.coeff_dict['P64sc'] = 1
        self.coeff_dict['P65sc'] = 1
        self.coeff_dict['P66sc'] = -1
        self.coeff_dict['P66cs'] = -1

        self.coeff_dict['P61sb'] = vd*vd * 4*sw**2*cw**2
        self.coeff_dict['P62sb'] = ad*ad * 4*sw**2*cw**2
        self.coeff_dict['P63sb'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['P63bs'] = ad*vd * 4*sw**2*cw**2
        self.coeff_dict['P64sb'] = 0
        self.coeff_dict['P65sb'] = 0
        self.coeff_dict['P66sb'] = 0
        self.coeff_dict['P66bs'] = 0

        self.coeff_dict['P61cb'] = vu*vd * 4*sw**2*cw**2
        self.coeff_dict['P62cb'] = au*ad * 4*sw**2*cw**2
        self.coeff_dict['P63cb'] = au*vd * 4*sw**2*cw**2
        self.coeff_dict['P63bc'] = ad*vu * 4*sw**2*cw**2
        self.coeff_dict['P64cb'] = 0
        self.coeff_dict['P65cb'] = 0
        self.coeff_dict['P66cb'] = 0
        self.coeff_dict['P66bc'] = 0

        self.coeff_dict['P61u'] = vu**2 * 2*sw**2*cw**2
        self.coeff_dict['P62u'] = au**2 * 2*sw**2*cw**2
        self.coeff_dict['P63u'] = vu*au * 4*sw**2*cw**2
        self.coeff_dict['P64u'] = 0

        self.coeff_dict['P61d'] = vd**2 * 2*sw**2*cw**2
        self.coeff_dict['P62d'] = ad**2 * 2*sw**2*cw**2
        self.coeff_dict['P63d'] = vd*ad * 4*sw**2*cw**2
        self.coeff_dict['P64d'] = 0

        self.coeff_dict['P61s'] = vd**2 * 2*sw**2*cw**2
        self.coeff_dict['P62s'] = ad**2 * 2*sw**2*cw**2
        self.coeff_dict['P63s'] = vd*ad * 4*sw**2*cw**2
        self.coeff_dict['P64s'] = 0

        self.coeff_dict['P61c'] = vu**2 * 2*sw**2*cw**2
        self.coeff_dict['P62c'] = au**2 * 2*sw**2*cw**2
        self.coeff_dict['P63c'] = vu*au * 4*sw**2*cw**2
        self.coeff_dict['P64c'] = 0

        self.coeff_dict['P61b'] = vd**2 * 2*sw**2*cw**2
        self.coeff_dict['P62b'] = ad**2 * 2*sw**2*cw**2
        self.coeff_dict['P63b'] = vd*ad * 4*sw**2*cw**2
        self.coeff_dict['P64b'] = 0

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


        # Create the np.array of coefficients:
        self.coeff_list_dm_dim6_dim7 = np.array(dict_to_list(self.coeff_dict, self.wc_name_list))
        self.coeff_list_dm_dim8 = np.array(dict_to_list(self.coeff_dict, self.wc8_name_list))
        self.coeff_list_sm_dim6 = np.array(dict_to_list(self.coeff_dict, self.sm_name_list))


        #---------------------------#
        # The anomalous dimensions: #
        #---------------------------#

        if self.DM_type == "D":
            self.gamma_QED = adm.ADM_QED(5)
            self.gamma_QED2 = adm.ADM_QED2(5)
            self.gamma_QCD = adm.ADM_QCD(5)
            self.gamma_QCD2 = adm.ADM_QCD2(5)
            self.gamma_QCD_dim8 = adm.ADM_QCD_dim8(5)
            self.gamma_hat = adm.ADT_QCD(5)
        if self.DM_type == "M":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(5), del_ind_list, 1), del_ind_list, 2)
        if self.DM_type == "C":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QED2 = np.delete(np.delete(adm.ADM_QED2(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(5), del_ind_list, 1), del_ind_list, 2)
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
            DM_dim6_init = np.delete(self.coeff_list_dm_dim6_dim7, np.r_[np.s_[0:18], np.s_[23:26], np.s_[31:154]])

        # The columns of ADM_eff correspond to SM6 operators; the rows of ADM_eff correspond to DM8 operators; 
        C6_dot_ADM_hat = np.transpose(np.tensordot(DM_dim6_init, self.gamma_hat, (0,2)))

        # The effective ADM
        #
        # Note that the mixing of the SM operators with four equal flavors does not contribute if we neglect yu, yd, ys! 

        self.ADM_eff = [np.vstack((np.hstack((self.ADM_SM, np.vstack((C6_dot_ADM_hat, np.zeros((20,12)))))),\
                              np.hstack((np.zeros((12,100)), self.gamma_QCD_dim8))))]




    def run(self, mu_low=None, double_QCD=None):
        """ Running of 5-flavor Wilson coefficients

        Calculate the running from MZ to mu_low [GeV; default mb(mb)] in the five-flavor theory. 

        Return a dictionary of Wilson coefficients for the five-flavor Lagrangian
        at scale mu_low.
        """

        ip = Num_input()
        if mu_low is None:
            mu_low=ip.mb_at_mb
        if double_QCD is None:
            double_QCD=True

        #-------------#
        # The running #
        #-------------#

        MZ = ip.Mz
        alpha_at_mb = 1/ip.aMZinv

        if double_QCD:
            adm_eff = self.ADM_eff
        else:
            projector = np.vstack((np.hstack((np.zeros((100,100)), np.ones((100,12)))), np.zeros((12,112))))
            adm_eff = [np.multiply(projector, self.ADM_eff[0])]

        as51 = rge.AlphaS(5,1)
        evolve1 = rge.RGE(self.gamma_QCD, 5)
        evolve2 = rge.RGE(self.gamma_QCD2, 5)
        evolve8 = rge.RGE(adm_eff, 5)

        # Mixing in the dim.6 DM-SM sector
        #
        # Strictly speaking, MZ and mb should be defined at the same scale (however, this is a higher-order difference)
        C_at_mb_QCD = np.dot(evolve2.U0_as2(as51.run(MZ),as51.run(mu_low)), np.dot(evolve1.U0(as51.run(MZ),as51.run(mu_low)), self.coeff_list_dm_dim6_dim7))
        C_at_mb_QED = np.dot(self.coeff_list_dm_dim6_dim7, self.gamma_QED) * np.log(mu_low/MZ) * alpha_at_mb/(4*np.pi)\
                      + np.dot(self.coeff_list_dm_dim6_dim7, self.gamma_QED2) * np.log(mu_low/MZ) * (alpha_at_mb/(4*np.pi))**2

        # Mixing in the dim.6 SM-SM and dim.8 DM-SM sector

        DIM6_DIM8_init = np.hstack((self.coeff_list_sm_dim6, self.coeff_list_dm_dim8))

        DIM6_DIM8_at_mb = np.dot(evolve8.U0(as51.run(MZ),as51.run(mu_low)), DIM6_DIM8_init)

        # Revert back to dictionary

        dict_coeff_mb = list_to_dict(C_at_mb_QCD + C_at_mb_QED, self.wc_name_list)
        dict_dm_dim8 = list_to_dict(np.delete(DIM6_DIM8_at_mb, np.s_[0:100]), self.wc8_name_list)
        dict_sm_dim6 = list_to_dict(np.delete(DIM6_DIM8_at_mb, np.s_[100:106]), self.sm_name_list)

        dict_coeff_mb.update(dict_dm_dim8)
        dict_coeff_mb.update(dict_sm_dim6)

        return dict_coeff_mb


    def match(self, mu=None, RGE=None, double_QCD=None):
        """ Match from five-flavor to four-flavor QCD

        Calculate the matching at mu [GeV; default 4.18 GeV].

        Returns a dictionary of Wilson coefficients for the four-flavor Lagrangian
        at scale mu.

        RGE is an optional argument to turn RGE running on (True) or off (False). (Default True)
        """
        ip = Num_input()
        if RGE is None:
            RGE=True
        if mu is None:
            mu=ip.mb_at_mb
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
            cdict4f['C71'] = cdold['C71'] - cdold['C75b']
            cdict4f['C72'] = cdold['C72'] - cdold['C76b']
            cdict4f['C73'] = cdold['C73'] + cdold['C77b']
            cdict4f['C74'] = cdold['C74'] + cdold['C78b']

        if self.DM_type == "C":
            for wcn in self.wc_name_list_4f:
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
        """ Calculate the NR coefficients from four-flavor theory with meson contributions split off (mainly for internal use) """
        return WC_4f(self.match(RGE, double_QCD), self.DM_type)._my_cNR(DM_mass, RGE, NLO, double_QCD)

    def cNR(self, DM_mass, qvec, RGE=None, NLO=None, double_QCD=None):
        """ Calculate the NR coefficients from four-flavor theory """
        return WC_4f(self.match(RGE, double_QCD), self.DM_type).cNR(DM_mass, qvec, RGE, NLO, double_QCD)

    def write_mma(self, DM_mass, q, RGE=None, NLO=None, double_QCD=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass DM_mass (in GeV) and the momentum transfer q (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is './')

        <filename> is the filename (default 'cNR.m')
        """
        WC_4f(self.match(RGE, double_QCD), self.DM_type).write_mma(DM_mass, q, RGE, NLO, double_QCD, path, filename)



