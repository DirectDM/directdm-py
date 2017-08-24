#!/usr/bin/env python3

import sys
import numpy as np
import scipy.integrate as spint
import warnings
from directdm.run import adm
from directdm.run import rge
from directdm.num.num_input import Num_input
from directdm.match.higgs_penguin import Higgspenguin


#----------------------------------------------#
# convert dictionaries to lists and vice versa #
#----------------------------------------------#

def dict_to_list(dictionary, order_list):
    """ Create a list from dictionary, according to ordering in oerder_list """
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

        The first argument should be a dictionary for the initial conditions of the 2 + 24 + 4 + 36 = 66 
        dimension-five to dimension-seven three-flavor-QCD Wilson coefficients of the form
        {'C51' : value, 'C52' : value, ...}. 
        An arbitrary number of them can be given; the default values are zero. 

        The second argument is the DM type; it can take the following values: 
            "D" (Dirac fermion; this is the default)
            "M" (Majorana fermion)
            "C" (Complex scalar)
            "R" (Real scalar)

        The possible name are (with an hopefully obvious notation):

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
                             'C710u', 'C710d', 'C710s', 'C710e', 'C710mu', 'C710tau'

        Majorana fermion:    'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                             'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau',

        Complex Scalar:      'C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                             'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                             'C65', 'C66'

        Real Scalar:         'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                             'C65', 'C66'

        (the notation corresponds to the numbering in 1707.06998).
        The Wilson coefficients should be specified in the MS-bar scheme at 2 GeV.

        The class has three methods:

        run
        ---
        Runs the Wilson coefficients from mu = 2 GeV to mu_low [GeV; default 2 GeV], with 3 active quark flavors


        cNR
        ---
        Calculates the cNR coefficients as defined in 1308.6288

        The class has two mandatory arguments: The DM mass in GeV and the momentum transfer in GeV


        write_mma
        ---------
        Writes an output file that can be loaded into mathematica, 
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
                                 'C79u', 'C79d', 'C79s', 'C79e', 'C79mu', 'C79tau', 'C710u', 'C710d', 'C710s', 'C710e', 'C710mu', 'C710tau']
        if self.DM_type == "M":
            self.wc_name_list = ['C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau', 'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau', 'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau']
            del_ind_list = [i for i in range(0,8)] + [i for i in range(14,20)] + [i for i in range(54,66)]

        if self.DM_type == "C":
            self.wc_name_list = ['C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 
                                 'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                 'C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau']
            del_ind_list = [0,1] + [i for i in range(8,14)] + [i for i in range(20,26)] + [27] + [29] + [i for i in range(36,42)] + [i for i in range(48,66)]

        if self.DM_type == "R":
            self.wc_name_list = ['C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau']
            del_ind_list = [i for i in range(0,26)] + [27] + [29] + [i for i in range(36,42)] + [i for i in range(48,66)]

        self.my_cNR_name_list = ['cNR1p', 'cNR1n', 'cNR2p', 'cNR2n', 'cNR3p', 'cNR3n', 'cNR4p', 'cNR4n', 'cNR5p', 'cNR5n',
                                 'cNR6p', 'cNR6n', 'cNR7p', 'cNR7n', 'cNR8p', 'cNR8n', 'cNR9p', 'cNR9n', 'cNR10p', 'cNR10n',
                                 'cNR11p', 'cNR11n', 'cNR12p', 'cNR12n', 'cNR13p', 'cNR13n', 'cNR14p', 'cNR14n', 'cNR15p', 'cNR15n',
                                 'cNR16p', 'cNR16n', 'cNR17p', 'cNR17n', 'cNR18p', 'cNR18n', 'cNR19p', 'cNR19n', 'cNR20p', 'cNR20n',
                                 'cNR21p', 'cNR21n', 'cNR22p', 'cNR22n', 'cNR23p', 'cNR23n', 'cNR100p', 'cNR100n']

        self.cNR_name_list = ['cNR1p', 'cNR1n', 'cNR2p', 'cNR2n', 'cNR3p', 'cNR3n', 'cNR4p', 'cNR4n', 'cNR5p', 'cNR5n',
                              'cNR6p', 'cNR6n', 'cNR7p', 'cNR7n', 'cNR8p', 'cNR8n', 'cNR9p', 'cNR9n', 'cNR10p', 'cNR10n',
                              'cNR11p', 'cNR11n', 'cNR12p', 'cNR12n']

        self.coeff_dict = {}
        # Issue a user warning if a key is not defined:
        for wc_name in coeff_dict.keys():
            if wc_name in self.wc_name_list:
                pass
            else:
                warnings.warn('The key ' + wc_name + ' is not a default key value. Typo?')
        # Create the dictionary:
        for wc_name in self.wc_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.

        # Create the np.array of coefficients:
        self.coeff_list = np.array(dict_to_list(self.coeff_dict, self.wc_name_list))


        #---------------------------#
        # The anomalous dimensions: #
        #---------------------------#
        if self.DM_type == "D":
            self.gamma_QED = adm.ADM_QED(3)
            self.gamma_QCD = adm.ADM_QCD(3)
            self.gamma_QCD2 = adm.ADM_QCD2(3)
        if self.DM_type == "M":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(3), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(3), del_ind_list, 1), del_ind_list, 2)
        if self.DM_type == "C":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(3), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(3), del_ind_list, 1), del_ind_list, 2)
        if self.DM_type == "R":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(3), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(3), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(3), del_ind_list, 1), del_ind_list, 2)


    def run(self, mu_low=None, dict=None):
        """ Running of 3-flavor Wilson coefficients

        Calculate the running from 2 GeV to mu_low [GeV; default 2 GeV] in the three-flavor theory. 

        For dict = True, returns a dictionary of Wilson coefficients for the three-flavor Lagrangian
        at scale mu_low (this is the default).

        For dict = False, returns a numpy array of Wilson coefficients for the three-flavor Lagrangian
        at scale mu_low.
        """
        if mu_low is None:
            mu_low=2
        if dict is None:
            dict = True

        #-------------#
        # The running #
        #-------------#

        ip = Num_input()
        alpha_at_mu = 1/ip.amtauinv

        as31 = rge.AlphaS(3,1)
        evolve1 = rge.RGE(self.gamma_QCD, 3)
        evolve2 = rge.RGE(self.gamma_QCD2, 3)

        C_at_mu_QCD = np.dot(evolve2.U0_as2(as31.run(2),as31.run(mu_low)), np.dot(evolve1.U0(as31.run(2),as31.run(mu_low)), self.coeff_list))
        C_at_mu_QED = np.dot(self.coeff_list, self.gamma_QED) * np.log(mu_low/2) * alpha_at_mu/(4*np.pi)

        if dict:
            return list_to_dict(C_at_mu_QCD + C_at_mu_QED, self.wc_name_list)
        else:
            return C_at_mu_QCD + C_at_mu_QED


    def _my_cNR(self, mchi, RGE=None, dict=None, NLO=None):
        """Calculate the coefficients of the NR operators, with momentum dependence factored out.
    
        mchi is the DM mass in GeV

        RGE is a flag to turn RGE running on (True) or off (False). (Default True)

        If NLO is set to True, the coherently enhanced NLO terms for Q_9^(7) are added. (Default False)

        For dict = True (default), returns a dictionary of coefficients for the NR Lagrangian, 
        as in 1308.6288, plus coefficients c13 -- c23, c100 for "spurious" long-distance operators

        The possible names are

        ['cNR1p', 'cNR1n', 'cNR2p', 'cNR2n', 'cNR3p', 'cNR3n', 'cNR4p', 'cNR4n', 'cNR5p', 'cNR5n',
         'cNR6p', 'cNR6n', 'cNR7p', 'cNR7n', 'cNR8p', 'cNR8n', 'cNR9p', 'cNR9n', 'cNR10p', 'cNR10n',
         'cNR11p', 'cNR11n', 'cNR12p', 'cNR12n', 'cNR13p', 'cNR13n', 'cNR14p', 'cNR14n', 'cNR15p', 'cNR15n',
         'cNR16p', 'cNR16n', 'cNR17p', 'cNR17n', 'cNR18p', 'cNR18n', 'cNR19p', 'cNR19n', 'cNR20p', 'cNR20n',
         'cNR21p', 'cNR21n', 'cNR22p', 'cNR22n', 'cNR23p', 'cNR23n', 'cNR100p', 'cNR100n']

        For dict = False, returns a numpy array of values according to the list above.
        """
        if RGE is None:
            RGE = True
        if dict is None:
            dict = True
        if NLO is None:
            NLO = False

        ### Input parameters ####
        ip = Num_input()

        mpi = ip.mpi0
        mp = ip.mproton
        mn = ip.mneutron
        mN = (mp+mn)/2

        alpha = 1/ip.alowinv

        # Quark masses at 2GeV
        mu = ip.mu_at_2GeV
        md = ip.md_at_2GeV
        ms = ip.ms_at_2GeV
        mtilde = ip.mtilde

        ### Numerical constants
        ip = Num_input()

        mproton = ip.mproton
        mneutron = ip.mneutron
    
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
        # For the tensors, O1 appears as a subleading contribution.
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
            'cNR1p' : 2*c3mu_dict['C61u'] + c3mu_dict['C61d'] - 2*ip.mG/27*c3mu_dict['C71']\
                      + ip.sigmaup*c3mu_dict['C75u'] + ip.sigmadp*c3mu_dict['C75d'] + ip.sigmas*c3mu_dict['C75s']\
                      - alpha/(2*np.pi*mchi)*c3mu_dict['C51'],
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : - 4*(ip.Deltaup*c3mu_dict['C64u'] + ip.Deltadp*c3mu_dict['C64d'] + ip.Deltas*c3mu_dict['C64s'])\
                      - 2*alpha/np.pi * ip.mup/mN * c3mu_dict['C51']\
                      + 8*(ip.FT0up*c3mu_dict['C79u'] + ip.FT0dp*c3mu_dict['C79d'] + ms*ip.gTs*c3mu_dict['C79s']),
            'cNR5p' : 0,
            'cNR6p' : -mN**2 * mtilde * (ip.Deltaup/mu + ip.Deltadp/md + ip.Deltas/ms)/mchi * c3mu_dict['C74'],
            'cNR7p' : -2*(ip.Deltaup*c3mu_dict['C63u'] + ip.Deltadp*c3mu_dict['C63d'] + ip.Deltas*c3mu_dict['C63s']),
            'cNR8p' : 4*c3mu_dict['C62u'] + 2*c3mu_dict['C62d'],
            'cNR9p' : mN*((4*ip.muup*c3mu_dict['C62u'] + 2*ip.mudp*c3mu_dict['C62d'] - 6*ip.mus*c3mu_dict['C62s'])/mN\
                      + 2*(ip.Deltaup*c3mu_dict['C63u'] + ip.Deltadp*c3mu_dict['C63d'] + ip.Deltas*c3mu_dict['C63s'])/mchi),
            'cNR10p' : -mN * mtilde * (ip.Deltaup/mu + ip.Deltadp/md + ip.Deltas/ms) * c3mu_dict['C73']\
                       -2*mN/mchi * (ip.FT0up*c3mu_dict['C710u'] + ip.FT0dp*c3mu_dict['C710d'] + ms*ip.gTs*c3mu_dict['C710s']),
            'cNR11p' : mN * (-(ip.sigmaup*c3mu_dict['C76u'] + ip.sigmadp*c3mu_dict['C76d'] + ip.sigmas*c3mu_dict['C76s'])/mchi\
                            + 2*ip.mG/27*c3mu_dict['C72']/mchi)\
                        + 2*(ip.FT0up*c3mu_dict['C710u'] + ip.FT0dp*c3mu_dict['C710d'] + ms*ip.gTs*c3mu_dict['C710s'])\
                        + 2*(mu*ip.BT10up*c3mu_dict['C710u'] + md*ip.BT10dp*c3mu_dict['C710d'] + ms*ip.BT10s*c3mu_dict['C710s']),
            'cNR12p' : -8*(ip.FT0up*c3mu_dict['C710u'] + ip.FT0dp*c3mu_dict['C710d'] + ms*ip.gTs*c3mu_dict['C710s']),
    
            'cNR13p' : mN**2 * (ip.gA * (ip.B0mu*c3mu_dict['C78u'] - ip.B0md*c3mu_dict['C78d'])/mchi + 2*ip.gA * (c3mu_dict['C64u'] - c3mu_dict['C64d'])),
            'cNR14p' : mN**2 * ((ip.Deltaup + ip.Deltadp - 2*ip.Deltas)/3\
                               * (ip.B0mu*c3mu_dict['C78u'] + ip.B0md*c3mu_dict['C78d'] - 2*ip.B0ms*c3mu_dict['C78s'])/mchi\
                               + 2/3 * (ip.Deltaup + ip.Deltadp - 2*ip.Deltas) * (c3mu_dict['C64u'] + c3mu_dict['C64d'] - 2*c3mu_dict['C64s'])),
            'cNR15p' : -mN**2 * (-mtilde/2 * ip.gA * (1/mu - 1/md) * c3mu_dict['C74']/mchi),
            'cNR16p' : -mN**2 * (-mtilde/6 * (ip.Deltaup + ip.Deltadp - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C74']/mchi),
    
            'cNR17p' : mN * (ip.gA * (ip.B0mu*c3mu_dict['C77u'] - ip.B0md*c3mu_dict['C77d'])),
            'cNR18p' : mN * ((ip.Deltaup + ip.Deltadp - 2*ip.Deltas)/3\
                           * (ip.B0mu*c3mu_dict['C77u'] + ip.B0md*c3mu_dict['C77d'] - 2*ip.B0ms*c3mu_dict['C77s'])),
            'cNR19p' : mN * (mtilde/2 * ip.gA * (1/mu - 1/md) * c3mu_dict['C73']),
            'cNR20p' : mN * (mtilde/6 * (ip.Deltaup + ip.Deltadp - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C73']),
    
            'cNR21p' : mN* (2*alpha/np.pi*c3mu_dict['C51']),
            'cNR22p' : -mN**2* (- 2*alpha/np.pi * ip.mup/mN * c3mu_dict['C51']),
            'cNR23p' : mN* (2*alpha/np.pi*c3mu_dict['C52']),

            'cNR100p' : 0,




            'cNR1n' : 2*c3mu_dict['C61d'] + c3mu_dict['C61u'] - 2*ip.mG/27*c3mu_dict['C71']\
                      + ip.sigmadn*c3mu_dict['C75d'] + ip.sigmaun*c3mu_dict['C75u'] + ip.sigmas*c3mu_dict['C75s'],
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : - 4*(ip.Deltadn*c3mu_dict['C64d'] + ip.Deltaun*c3mu_dict['C64u'] + ip.Deltas*c3mu_dict['C64s'])\
                      - 2*alpha/np.pi * ip.mun/mN * c3mu_dict['C51']\
                      + 8*(ip.FT0dn*c3mu_dict['C79d'] + ip.FT0un*c3mu_dict['C79u'] + ms*ip.gTs*c3mu_dict['C79s']),
            'cNR5n' : 0,
            'cNR6n' : -mN**2 * (mtilde * (ip.Deltadn/md + ip.Deltaun/mu + ip.Deltas/ms)/mchi * c3mu_dict['C74']),
            'cNR7n' : -2*(ip.Deltadn*c3mu_dict['C63d'] + ip.Deltaun*c3mu_dict['C63u'] + ip.Deltas*c3mu_dict['C63s']),
            'cNR8n' : 2*(2*c3mu_dict['C62d'] + c3mu_dict['C62u']),
            'cNR9n' : mN * ((4*ip.mudn*c3mu_dict['C62d'] + 2*ip.muun*c3mu_dict['C62u'] - 6*ip.mus*c3mu_dict['C62s'])/mN\
                           + 2*(ip.Deltadn*c3mu_dict['C63d'] + ip.Deltaun*c3mu_dict['C63u'] + ip.Deltas*c3mu_dict['C63s'])/mchi),
            'cNR10n' : mN * (- mtilde * (ip.Deltadn/md + ip.Deltaun/mu + ip.Deltas/ms) * c3mu_dict['C73'])\
                     -2*mN/mchi * (ip.FT0dn*c3mu_dict['C710d'] + ip.FT0un*c3mu_dict['C710u'] + ms*ip.gTs*c3mu_dict['C710s']),
            'cNR11n' : mN * (-(ip.sigmadn*c3mu_dict['C76d'] + ip.sigmaun*c3mu_dict['C76u'] + ip.sigmas*c3mu_dict['C76s'])/mchi\
                           + 2*ip.mG/27*c3mu_dict['C72']/mchi)\
                       + 2*(ip.FT0dn*c3mu_dict['C710d'] + ip.FT0un*c3mu_dict['C710u'] + ms*ip.gTs*c3mu_dict['C710s'])\
                       + 2*(mu*ip.BT10dn*c3mu_dict['C710d'] + md*ip.BT10un*c3mu_dict['C710u'] + ms*ip.BT10s*c3mu_dict['C710s']),
            'cNR12n' : -8*(ip.FT0dn*c3mu_dict['C710d'] + ip.FT0un*c3mu_dict['C710u'] + ms*ip.gTs*c3mu_dict['C710s']),
    
            'cNR13n' : mN**2 * (ip.gA * (ip.B0md*c3mu_dict['C78d'] - ip.B0mu*c3mu_dict['C78u'])/mchi + 2*ip.gA * (c3mu_dict['C64d'] - c3mu_dict['C64u'])),
            'cNR14n' : mN**2 * ((ip.Deltadn + ip.Deltaun - 2*ip.Deltas)/3\
                                * (ip.B0md*c3mu_dict['C78d'] + ip.B0mu*c3mu_dict['C78u'] - 2*ip.B0ms*c3mu_dict['C78s'])/mchi\
                               + 2/3 * (ip.Deltadn + ip.Deltaun - 2*ip.Deltas) * (c3mu_dict['C64d'] + c3mu_dict['C64u'] - 2*c3mu_dict['C64s'])),
            'cNR15n' : -mN**2 * (-mtilde/2 * ip.gA * (1/md - 1/mu) * c3mu_dict['C74']/mchi),
            'cNR16n' : -mN**2 * (-mtilde/6 * (ip.Deltadn + ip.Deltaun - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C74']/mchi),
    
            'cNR17n' : mN * (ip.gA * (ip.B0md*c3mu_dict['C77d'] - ip.B0mu*c3mu_dict['C77u'])),
            'cNR18n' : mN * ((ip.Deltadn + ip.Deltaun - 2*ip.Deltas)/3\
                            * (ip.B0md*c3mu_dict['C77d'] + ip.B0mu*c3mu_dict['C77u'] - 2*ip.B0ms*c3mu_dict['C77s'])),
            'cNR19n' : mN * (mtilde/2 * ip.gA * (1/md - 1/mu) * c3mu_dict['C73']),
            'cNR20n' : mN * (mtilde/6 * (ip.Deltadn + ip.Deltaun - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C73']),
    
            'cNR21n' : 0,
            'cNR22n' : -mN**2 * (- 2*alpha/np.pi * ip.mun/mN * c3mu_dict['C51']),
            'cNR23n' : 0,

            'cNR100n' : 0
            }

            if NLO:
                my_cNR_dict['cNR5p'] = 2*(ip.FT0up*c3mu_dict['C79u'] + ip.FT0dp*c3mu_dict['C79d'] + ms*ip.gTs*c3mu_dict['C79s'])\
                                       + 2*(mu*ip.BT10up*c3mu_dict['C79u'] + md*ip.BT10dp*c3mu_dict['C79d'] + ms*ip.BT10s*c3mu_dict['C79s']),
                my_cNR_dict['cNR100p'] = - (ip.FT0up*c3mu_dict['C79u'] + ip.FT0dp*c3mu_dict['C79d'] + ms*ip.gTs*c3mu_dict['C79s'])/(2*mchi*mN)\
                                         - (mu*ip.BT10up*c3mu_dict['C79u'] + md*ip.BT10dp*c3mu_dict['C79d'] + ms*ip.BT10s*c3mu_dict['C79s'])/(2*mchi*mN),
                my_cNR_dict['cNR5n'] =  2*(ip.FT0dn*c3mu_dict['C79d'] + ip.FT0un*c3mu_dict['C79u'] + ms*ip.gTs*c3mu_dict['C79s'])\
                                        + 2*(mu*ip.BT10dn*c3mu_dict['C79d'] + md*ip.BT10un*c3mu_dict['C79u'] + ms*ip.BT10s*c3mu_dict['C79s']),
                my_cNR_dict['cNR100n'] = - (ip.FT0dn*c3mu_dict['C79d'] + ip.FT0un*c3mu_dict['C79u'] + ms*ip.gTs*c3mu_dict['C79s'])/(2*mchi*mN)\
                                         - (mu*ip.BT10dn*c3mu_dict['C79d'] + md*ip.BT10un*c3mu_dict['C79u'] + ms*ip.BT10s*c3mu_dict['C79s'])/(2*mchi*mN)

        if self.DM_type == "M":
            my_cNR_dict = {
            'cNR1p' : - 2*ip.mG/27*c3mu_dict['C71']\
                      + ip.sigmaup*c3mu_dict['C75u'] + ip.sigmadp*c3mu_dict['C75d'] + ip.sigmas*c3mu_dict['C75s'],
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : - 4*(ip.Deltaup*c3mu_dict['C64u'] + ip.Deltadp*c3mu_dict['C64d'] + ip.Deltas*c3mu_dict['C64s']),
            'cNR5p' : 0,
            'cNR6p' : -mN**2 * mtilde * (ip.Deltaup/mu + ip.Deltadp/md + ip.Deltas/ms)/mchi * c3mu_dict['C74'],
            'cNR7p' : 0,
            'cNR8p' : 4*c3mu_dict['C62u'] + 2*c3mu_dict['C62d'],
            'cNR9p' : mN*(4*ip.muup*c3mu_dict['C62u'] + 2*ip.mudp*c3mu_dict['C62d'] - 6*ip.mus*c3mu_dict['C62s'])/mN,
            'cNR10p' : -mN * mtilde * (ip.Deltaup/mu + ip.Deltadp/md + ip.Deltas/ms) * c3mu_dict['C73'],
            'cNR11p' : mN * (-(ip.sigmaup*c3mu_dict['C76u'] + ip.sigmadp*c3mu_dict['C76d'] + ip.sigmas*c3mu_dict['C76s'])/mchi\
                            + 2*ip.mG/27*c3mu_dict['C72']/mchi),
            'cNR12p' : 0,
    
            'cNR13p' : mN**2 * (ip.gA * (ip.B0mu*c3mu_dict['C78u'] - ip.B0md*c3mu_dict['C78d'])/mchi + 2*ip.gA * (c3mu_dict['C64u'] - c3mu_dict['C64d'])),
            'cNR14p' : mN**2 * ((ip.Deltaup + ip.Deltadp - 2*ip.Deltas)/3\
                               * (ip.B0mu*c3mu_dict['C78u'] + ip.B0md*c3mu_dict['C78d'] - 2*ip.B0ms*c3mu_dict['C78s'])/mchi\
                               + 2/3 * (ip.Deltaup + ip.Deltadp - 2*ip.Deltas) * (c3mu_dict['C64u'] + c3mu_dict['C64d'] - 2*c3mu_dict['C64s'])),
            'cNR15p' : -mN**2 * (-mtilde/2 * ip.gA * (1/mu - 1/md) * c3mu_dict['C74']/mchi),
            'cNR16p' : -mN**2 * (-mtilde/6 * (ip.Deltaup + ip.Deltadp - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C74']/mchi),
    
            'cNR17p' : mN * (ip.gA * (ip.B0mu*c3mu_dict['C77u'] - ip.B0md*c3mu_dict['C77d'])),
            'cNR18p' : mN * ((ip.Deltaup + ip.Deltadp - 2*ip.Deltas)/3\
                           * (ip.B0mu*c3mu_dict['C77u'] + ip.B0md*c3mu_dict['C77d'] - 2*ip.B0ms*c3mu_dict['C77s'])),
            'cNR19p' : mN * (mtilde/2 * ip.gA * (1/mu - 1/md) * c3mu_dict['C73']),
            'cNR20p' : mN * (mtilde/6 * (ip.Deltaup + ip.Deltadp - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C73']),
    
            'cNR21p' : 0,
            'cNR22p' : 0,
            'cNR23p' : 0,

            'cNR100p' : 0,




            'cNR1n' : - 2*ip.mG/27*c3mu_dict['C71']\
                      + ip.sigmadn*c3mu_dict['C75d'] + ip.sigmaun*c3mu_dict['C75u'] + ip.sigmas*c3mu_dict['C75s'],
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : - 4*(ip.Deltadn*c3mu_dict['C64d'] + ip.Deltaun*c3mu_dict['C64u'] + ip.Deltas*c3mu_dict['C64s']),
            'cNR5n' : 0,
            'cNR6n' : -mN**2 * (mtilde * (ip.Deltadn/md + ip.Deltaun/mu + ip.Deltas/ms)/mchi * c3mu_dict['C74']),
            'cNR7n' : 0,
            'cNR8n' : 2*(2*c3mu_dict['C62d'] + c3mu_dict['C62u']),
            'cNR9n' : mN * (4*ip.mudn*c3mu_dict['C62d'] + 2*ip.muun*c3mu_dict['C62u'] - 6*ip.mus*c3mu_dict['C62s'])/mN,
            'cNR10n' : -mN * mtilde * (ip.Deltadn/md + ip.Deltaun/mu + ip.Deltas/ms) * c3mu_dict['C73'],
            'cNR11n' : mN * (-(ip.sigmadn*c3mu_dict['C76d'] + ip.sigmaun*c3mu_dict['C76u'] + ip.sigmas*c3mu_dict['C76s'])/mchi\
                           + 2*ip.mG/27*c3mu_dict['C72']/mchi),
            'cNR12n' : 0,
    
            'cNR13n' : mN**2 * (ip.gA * (ip.B0md*c3mu_dict['C78d'] - ip.B0mu*c3mu_dict['C78u'])/mchi + 2*ip.gA * (c3mu_dict['C64d'] - c3mu_dict['C64u'])),
            'cNR14n' : mN**2 * ((ip.Deltadn + ip.Deltaun - 2*ip.Deltas)/3\
                                * (ip.B0md*c3mu_dict['C78d'] + ip.B0mu*c3mu_dict['C78u'] - 2*ip.B0ms*c3mu_dict['C78s'])/mchi\
                               + 2/3 * (ip.Deltadn + ip.Deltaun - 2*ip.Deltas) * (c3mu_dict['C64d'] + c3mu_dict['C64u'] - 2*c3mu_dict['C64s'])),
            'cNR15n' : -mN**2 * (-mtilde/2 * ip.gA * (1/md - 1/mu) * c3mu_dict['C74']/mchi),
            'cNR16n' : -mN**2 * (-mtilde/6 * (ip.Deltadn + ip.Deltaun - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C74']/mchi),
    
            'cNR17n' : mN * (ip.gA * (ip.B0md*c3mu_dict['C77d'] - ip.B0mu*c3mu_dict['C77u'])),
            'cNR18n' : mN * ((ip.Deltadn + ip.Deltaun - 2*ip.Deltas)/3\
                            * (ip.B0md*c3mu_dict['C77d'] + ip.B0mu*c3mu_dict['C77u'] - 2*ip.B0ms*c3mu_dict['C77s'])),
            'cNR19n' : mN * (mtilde/2 * ip.gA * (1/md - 1/mu) * c3mu_dict['C73']),
            'cNR20n' : mN * (mtilde/6 * (ip.Deltadn + ip.Deltaun - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C73']),
    
            'cNR21n' : 0,
            'cNR22n' : 0,
            'cNR23n' : 0,

            'cNR100n' : 0
            }


        if self.DM_type == "C":
            my_cNR_dict = {
            'cNR1p' : 2*mchi*(2*c3mu_dict['C61u'] + c3mu_dict['C61d']) - 2*ip.mG/27*c3mu_dict['C65']\
                      + ip.sigmaup*c3mu_dict['C63u'] + ip.sigmadp*c3mu_dict['C63d'] + ip.sigmas*c3mu_dict['C63s'],
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : 0,
            'cNR5p' : 0,
            'cNR6p' : 0,
            'cNR7p' : -4*mchi*(ip.Deltaup*c3mu_dict['C62u'] + ip.Deltadp*c3mu_dict['C62d'] + ip.Deltas*c3mu_dict['C62s']),
            'cNR8p' : 0,
            'cNR9p' : 0,
            'cNR10p' : -mN * mtilde * (ip.Deltaup/mu + ip.Deltadp/md + ip.Deltas/ms) * c3mu_dict['C66'],
            'cNR11p' : 0,
            'cNR12p' : 0,

            'cNR13p' : 0,
            'cNR14p' : 0,
            'cNR15p' : 0,
            'cNR16p' : 0,
    
            'cNR17p' : mN * (ip.gA * (ip.B0mu*c3mu_dict['C64u'] - ip.B0md*c3mu_dict['C64d'])),
            'cNR18p' : mN * ((ip.Deltaup + ip.Deltadp - 2*ip.Deltas)/3\
                           * (ip.B0mu*c3mu_dict['C64u'] + ip.B0md*c3mu_dict['C64d'] - 2*ip.B0ms*c3mu_dict['C64s'])),
            'cNR19p' : mN * (mtilde/2 * ip.gA * (1/mu - 1/md) * c3mu_dict['C66']),
            'cNR20p' : mN * (mtilde/6 * (ip.Deltaup + ip.Deltadp - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C66']),
    
            'cNR21p' : 0,
            'cNR22p' : 0,
            'cNR23p' : 0,

            'cNR100p' : 0,




            'cNR1n' : 2*mchi*(2*c3mu_dict['C61d'] + c3mu_dict['C61u']) - 2*ip.mG/27*c3mu_dict['C65']\
                      + ip.sigmadn*c3mu_dict['C63d'] + ip.sigmaun*c3mu_dict['C63u'] + ip.sigmas*c3mu_dict['C63s'],
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : 0,
            'cNR5n' : 0,
            'cNR6n' : 0,
            'cNR7n' : -4*mchi*(ip.Deltadn*c3mu_dict['C62d'] + ip.Deltaun*c3mu_dict['C62u'] + ip.Deltas*c3mu_dict['C62s']),
            'cNR8n' : 0,
            'cNR9n' : 0,
            'cNR10n' : -mN * mtilde * (ip.Deltadn/md + ip.Deltaun/mu + ip.Deltas/ms) * c3mu_dict['C66'],
            'cNR11n' : 0,
            'cNR12n' : 0,

            'cNR13n' : 0,
            'cNR14n' : 0,
            'cNR15n' : 0,
            'cNR16n' : 0,
    
            'cNR17n' : mN * (ip.gA * (ip.B0md*c3mu_dict['C64d'] - ip.B0mu*c3mu_dict['C64u'])),
            'cNR18n' : mN * ((ip.Deltadn + ip.Deltaun - 2*ip.Deltas)/3\
                            * (ip.B0md*c3mu_dict['C64d'] + ip.B0mu*c3mu_dict['C64u'] - 2*ip.B0ms*c3mu_dict['C64s'])),
            'cNR19n' : mN * (mtilde/2 * ip.gA * (1/md - 1/mu) * c3mu_dict['C66']),
            'cNR20n' : mN * (mtilde/6 * (ip.Deltadn + ip.Deltaun - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C66']),
    
            'cNR21n' : 0,
            'cNR22n' : 0,
            'cNR23n' : 0,

            'cNR100n' : 0
            }


        if self.DM_type == "R":
            my_cNR_dict = {
            'cNR1p' :  + ip.sigmaup*c3mu_dict['C63u'] + ip.sigmadp*c3mu_dict['C63d'] + ip.sigmas*c3mu_dict['C63s'] - 2*ip.mG/27*c3mu_dict['C65'],
            'cNR2p' : 0,
            'cNR3p' : 0,
            'cNR4p' : 0,
            'cNR5p' : 0,
            'cNR6p' : 0,
            'cNR7p' : 0,
            'cNR8p' : 0,
            'cNR9p' : 0,
            'cNR10p' : -mN * mtilde * (ip.Deltaup/mu + ip.Deltadp/md + ip.Deltas/ms) * c3mu_dict['C66'],
            'cNR11p' : 0,
            'cNR12p' : 0,

            'cNR13p' : 0,
            'cNR14p' : 0,
            'cNR15p' : 0,
            'cNR16p' : 0,
    
            'cNR17p' : mN * (ip.gA * (ip.B0mu*c3mu_dict['C64u'] - ip.B0md*c3mu_dict['C64d'])),
            'cNR18p' : mN * ((ip.Deltaup + ip.Deltadp - 2*ip.Deltas)/3\
                           * (ip.B0mu*c3mu_dict['C64u'] + ip.B0md*c3mu_dict['C64d'] - 2*ip.B0ms*c3mu_dict['C64s'])),
            'cNR19p' : mN * (mtilde/2 * ip.gA * (1/mu - 1/md) * c3mu_dict['C66']),
            'cNR20p' : mN * (mtilde/6 * (ip.Deltaup + ip.Deltadp - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C66']),
    
            'cNR21p' : 0,
            'cNR22p' : 0,
            'cNR23p' : 0,

            'cNR100p' : 0,




            'cNR1n' : ip.sigmadn*c3mu_dict['C63d'] + ip.sigmaun*c3mu_dict['C63u'] + ip.sigmas*c3mu_dict['C63s'] - 2*ip.mG/27*c3mu_dict['C65'],
            'cNR2n' : 0,
            'cNR3n' : 0,
            'cNR4n' : 0,
            'cNR5n' : 0,
            'cNR6n' : 0,
            'cNR7n' : 0,
            'cNR8n' : 0,
            'cNR9n' : 0,
            'cNR10n' : -mN * mtilde * (ip.Deltadn/md + ip.Deltaun/mu + ip.Deltas/ms) * c3mu_dict['C66'],
            'cNR11n' : 0,
            'cNR12n' : 0,

            'cNR13n' : 0,
            'cNR14n' : 0,
            'cNR15n' : 0,
            'cNR16n' : 0,
    
            'cNR17n' : mN * (ip.gA * (ip.B0md*c3mu_dict['C64d'] - ip.B0mu*c3mu_dict['C64u'])),
            'cNR18n' : mN * ((ip.Deltadn + ip.Deltaun - 2*ip.Deltas)/3\
                            * (ip.B0md*c3mu_dict['C64d'] + ip.B0mu*c3mu_dict['C64u'] - 2*ip.B0ms*c3mu_dict['C64s'])),
            'cNR19n' : mN * (mtilde/2 * ip.gA * (1/md - 1/mu) * c3mu_dict['C66']),
            'cNR20n' : mN * (mtilde/6 * (ip.Deltadn + ip.Deltaun - 2*ip.Deltas) * (1/mu + 1/md - 2/ms) * c3mu_dict['C66']),
    
            'cNR21n' : 0,
            'cNR22n' : 0,
            'cNR23n' : 0,

            'cNR100n' : 0
            }


        if dict:
            return my_cNR_dict
        else:
            return dict_to_list(my_cNR_dict, self.my_cNR_name_list)


    def cNR(self, mchi, qvector, RGE=None, dict=None, NLO=None):
        """ The operator coefficients of O_1^N -- O_12^N as in 1308.6288 -- multiply by propagators and sum up contributions 

        mchi is the DM mass in GeV

        RGE is a flag to turn RGE running on (True) or off (False). (Default True)

        If NLO is set to True, the coherently enhanced NLO terms for Q_9^(7) are added. (Default False)

        For dict = True (default), returns a dictionary of coefficients for the NR Lagrangian, 
        cNR1 -- cNR12, as in 1308.6288

        The possible names are

        ['cNR1p', 'cNR1n', 'cNR2p', 'cNR2n', 'cNR3p', 'cNR3n', 'cNR4p', 'cNR4n', 'cNR5p', 'cNR5n',
         'cNR6p', 'cNR6n', 'cNR7p', 'cNR7n', 'cNR8p', 'cNR8n', 'cNR9p', 'cNR9n', 'cNR10p', 'cNR10n',
         'cNR11p', 'cNR11n', 'cNR12p', 'cNR12n']

        For dict = False, returns a numpy array of values according to the list above.
        """
        if RGE is None:
            RGE = True
        if dict is None:
            dict = True
        if NLO is None:
            NLO = False

        ip = Num_input()
        meta = ip.meta
        mpi = ip.mpi0

        qsq = qvector**2

        # The traditional coefficients, where different from above
        cNR_dict = {}
        my_cNR = self._my_cNR(mchi, RGE, True, NLO)

        # Add meson- / photon-pole contributions
        cNR_dict['cNR1p'] = my_cNR['cNR1p'] + qsq * my_cNR['cNR100p']
        cNR_dict['cNR2p'] = my_cNR['cNR2p']
        cNR_dict['cNR3p'] = my_cNR['cNR3p']
        cNR_dict['cNR4p'] = my_cNR['cNR4p']
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
        cNR_dict['cNR4n'] = my_cNR['cNR4n']
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

        if dict:
            return cNR_dict
        else:
            return dict_to_list(cNR_dict, self.cNR_name_list)


    def write_mma(self, mchi, qvector, RGE=None, NLO=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass mchi (in GeV) and the momentum transfer qvector (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is '.')

        <filename> is the filename (default 'cNR.m')
        """
        if RGE is None:
            RGE=True
        if NLO is None:
            NLO=False
        if path is None:
            path = './'
        if filename is None:
            filename = 'cNR.m'

        val = self.cNR(mchi, qvector, RGE, True, NLO)
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

        output_file = path + filename

        with open(output_file,'w') as f:
            f.write(self.cNR_list_mma)

class WC_4f(object):
    def __init__(self, coeff_dict, DM_type=None):
        """ Class for Wilson coefficients in 4 flavor QCD x QED plus DM.

        The argument should be a dictionary for the initial conditions of the 2 + 28 + 4 + 42 = 76 
        dimension-five to dimension-seven four-flavor-QCD Wilson coefficients of the form
        {'C51' : value, 'C52' : value, ...}. 
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
                             'C710u', 'C710d', 'C710s', 'C710c', 'C710e', 'C710mu', 'C710tau'

        Majorana fermion:    'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C75c', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C76c', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C77c', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C78c', 'C78e', 'C78mu', 'C78tau',

        Complex Scalar:      'C61u', 'C61d', 'C61s', 'C61c', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                             'C65', 'C66',
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau'

        Real Scalar:         'C65', 'C66'
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',

        (the notation corresponds to the numbering in 1707.06998).
        The Wilson coefficients should be specified in the MS-bar scheme at mb = 4.18 GeV.

        The class has three methods: 

        run
        ---
        Runs the Wilson from mb = 4.18 GeV to muc [GeV; default 2 GeV], with 4 active quark flavors

        match
        -----
        Matches the Wilson coefficients from 4-flavor to 3-flavor QCD, at scale muc [default 2 GeV]

        cNR
        ---
        Calculates the cNR coefficients as defined in 1308.6288

        It has two mandatory arguments: The DM mass in GeV and the momentum transfer in GeV


        write_mma
        ---------
        Writes an output file that can be loaded into mathematica, 
        to be used in the DMFormFactor package [1308.6288].
        """
        if DM_type is None:
            DM_type = "D"
        self.DM_type = DM_type


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
                                 'C710u', 'C710d', 'C710s', 'C710c', 'C710e', 'C710mu', 'C710tau']
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
                                 'C710u', 'C710d', 'C710s', 'C710e', 'C710mu', 'C710tau']

        if self.DM_type == "M":
            self.wc_name_list = ['C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75c', 'C75e', 'C75mu', 'C75tau', 
                                 'C76u', 'C76d', 'C76s', 'C76c', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77c', 'C77e', 'C77mu', 'C77tau', 
                                 'C78u', 'C78d', 'C78s', 'C78c', 'C78e', 'C78mu', 'C78tau']
            self.wc_name_list_3f = ['C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                    'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau',
                                    'C71', 'C72', 'C73', 'C74',
                                    'C75u', 'C75d', 'C75s', 'C75e', 'C75mu', 'C75tau',
                                    'C76u', 'C76d', 'C76s', 'C76e', 'C76mu', 'C76tau',
                                    'C77u', 'C77d', 'C77s', 'C77e', 'C77mu', 'C77tau',
                                    'C78u', 'C78d', 'C78s', 'C78e', 'C78mu', 'C78tau']
            del_ind_list = [i for i in range(0,9)] + [i for i in range(16,23)] + [i for i in range(62,76)]

        if self.DM_type == "C":
            self.wc_name_list = ['C61u', 'C61d', 'C61s', 'C61c', 'C61e', 'C61mu', 'C61tau', 
                                 'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                 'C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau']
            self.wc_name_list_3f = ['C61u', 'C61d', 'C61s', 'C61e', 'C61mu', 'C61tau', 
                                    'C62u', 'C62d', 'C62s', 'C62e', 'C62mu', 'C62tau',
                                    'C65', 'C66',
                                    'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                    'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau']
            del_ind_list = [0,1] + [i for i in range(9,16)] + [i for i in range(23,30)] + [31] + [33] + [i for i in range(41,48)] + [i for i in range(55,76)]

        if self.DM_type == "R":
            self.wc_name_list = ['C65', 'C66',
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau']
            self.wc_name_list_3f = ['C65', 'C66',
                                    'C63u', 'C63d', 'C63s', 'C63e', 'C63mu', 'C63tau', 
                                    'C64u', 'C64d', 'C64s', 'C64e', 'C64mu', 'C64tau']
            del_ind_list = [i for i in range(0,30)] + [31] + [33] + [i for i in range(41,48)] + [i for i in range(55,76)]


        self.coeff_dict = {}
        # Issue a user warning if a key is not defined:
        for wc_name in coeff_dict.keys():
            if wc_name in self.wc_name_list:
                pass
            else:
                warnings.warn('The key ' + wc_name + ' is not a default key value. Typo?')
        # Create the dictionary:
        for wc_name in self.wc_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.

        # Create the np.array of coefficients:
        self.coeff_list = np.array(dict_to_list(self.coeff_dict, self.wc_name_list))


        #---------------------------#
        # The anomalous dimensions: #
        #---------------------------#
        if self.DM_type == "D":
            self.gamma_QED = adm.ADM_QED(4)
            self.gamma_QCD = adm.ADM_QCD(4)
            self.gamma_QCD2 = adm.ADM_QCD2(4)
        if self.DM_type == "M":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(4), del_ind_list, 1), del_ind_list, 2)
        if self.DM_type == "C":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(4), del_ind_list, 1), del_ind_list, 2)
        if self.DM_type == "R":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(4), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(4), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(4), del_ind_list, 1), del_ind_list, 2)


    def run(self, muc=None, dict=None):
        """ Running of 4-flavor Wilson coefficients

        Calculate the running from mb to muc [GeV; default 2 GeV] in the four-flavor theory. 

        For dict = True, returns a dictionary of Wilson coefficients for the four-flavor Lagrangian
        at scale muc (this is the default).

        For dict = False, returns a numpy array of Wilson coefficients for the four-flavor Lagrangian
        at scale muc.
        """
        if muc is None:
            muc=2
        if dict is None:
            dict = True

        #-------------#
        # The running #
        #-------------#

        ip = Num_input()

        mb = ip.mb_at_mb
        alpha_at_mc = 1/ip.aMZinv

        as41 = rge.AlphaS(4,1)
        evolve1 = rge.RGE(self.gamma_QCD, 4)
        evolve2 = rge.RGE(self.gamma_QCD2, 4)

        # Strictly speaking, mb should be defined at scale muc (however, this is a higher-order difference)
        C_at_mc_QCD = np.dot(evolve2.U0_as2(as41.run(mb),as41.run(muc)), np.dot(evolve1.U0(as41.run(mb),as41.run(muc)), self.coeff_list))
        C_at_mc_QED = np.dot(self.coeff_list, self.gamma_QED) * np.log(muc/mb) * alpha_at_mc/(4*np.pi)

        if dict:
            return list_to_dict(C_at_mc_QCD + C_at_mc_QED, self.wc_name_list)
        else:
            return C_at_mc_QCD + C_at_mc_QED


    def match(self, mu=None, dict=None):
        """ Match from four-flavor to three-flavor QCD

        Calculate the matching at mu [GeV; default 2 GeV].

        For dict = True, returns a dictionary of Wilson coefficients for the three-flavor Lagrangian
        at scale mu (this is the default).

        For dict = False, returns a numpy array of Wilson coefficients for the three-flavor Lagrangian
        at scale mu.
        """
        if mu is None:
            mu=2
        if dict is None:
            dict=True

        # The new coefficients
        cdict3f = {}
        cdold = self.run(mu)

        if self.DM_type == "D" or self.DM_type == "M":
            for wcn in self.wc_name_list_3f:
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
        if dict:
            return cdict3f
        else:
            return dict_to_list(cdict3f, self.wc_name_list_3f)

    def _my_cNR(self, mchi, RGE=None, dict=None, NLO=None):
        """ Calculate the NR coefficients from four-flavor theory with meson contributions split off (mainly for internal use) """
        return WC_3f(self.match(), self.DM_type)._my_cNR(mchi, RGE, dict, NLO)

    def cNR(self, mchi, qvec, RGE=None, dict=None, NLO=None):
        """ Calculate the NR coefficients from four-flavor theory """
        return WC_3f(self.match(), self.DM_type).cNR(mchi, qvec, RGE, dict, NLO)

    def write_mma(self, mchi, qvector, RGE=None, NLO=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass mchi (in GeV) and the momentum transfer qvector (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is '.')

        <filename> is the filename (default 'cNR.m')
        """
        WC_3f(self.match(), self.DM_type).write_mma(mchi, qvector, RGE, NLO, path, filename)




class WC_5f(object):
    def __init__(self, coeff_dict, DM_type=None):
        """ Class for Wilson coefficients in 5 flavor QCD x QED plus DM.

        The argument should be a dictionary for the initial conditions of the 2 + 32 + 4 + 48 = 86 
        dimension-five to dimension-seven four-flavor-QCD Wilson coefficients of the form
        {'C51' : value, 'C52' : value, ...}. 
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
                             'C710u', 'C710d', 'C710s', 'C710c', 'C710b', 'C710e', 'C710mu', 'C710tau'

        Majorana fermion:    'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                             'C71', 'C72', 'C73', 'C74',
                             'C75u', 'C75d', 'C75s', 'C75c', 'C75b', 'C75e', 'C75mu', 'C75tau', 
                             'C76u', 'C76d', 'C76s', 'C76c', 'C76b', 'C76e', 'C76mu', 'C76tau',
                             'C77u', 'C77d', 'C77s', 'C77c', 'C77b', 'C77e', 'C77mu', 'C77tau', 
                             'C78u', 'C78d', 'C78s', 'C78c', 'C78b', 'C78e', 'C78mu', 'C78tau',

        Complex Scalar:      'C61u', 'C61d', 'C61s', 'C61c', 'C61b', 'C61e', 'C61mu', 'C61tau', 
                             'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                             'C65', 'C66',
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau'

        Real Scalar:         'C65', 'C66'
                             'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
                             'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',

        (the notation corresponds to the numbering in 1707.06998).
        The Wilson coefficients should be specified in the MS-bar scheme at MZ = 91.1876 GeV.

        The class has three methods: 

        run
        ---
        Runs the Wilson from MZ = 91.1876 GeV to mub [GeV; default mb = 4.18 GeV], with 5 active quark flavors

        match
        -----
        Matches the Wilson coefficients from 5-flavor to 4-flavor QCD, at scale mub

        cNR
        ---
        Calculates the cNR coefficients as defined in 1308.6288

        It has two mandatory arguments: The DM mass in GeV and the momentum transfer in GeV


        write_mma
        ---------
        Writes an output file that can be loaded into mathematica, 
        to be used in the DMFormFactor package [1308.6288].
        """
        if DM_type is None:
            DM_type = "D"
        self.DM_type = DM_type

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
                                 'C710u', 'C710d', 'C710s', 'C710c', 'C710b', 'C710e', 'C710mu', 'C710tau']
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
                                    'C710u', 'C710d', 'C710s', 'C710c', 'C710e', 'C710mu', 'C710tau']

        if self.DM_type == "M":
            self.wc_name_list = ['C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                                 'C71', 'C72', 'C73', 'C74',
                                 'C75u', 'C75d', 'C75s', 'C75c', 'C75b', 'C75e', 'C75mu', 'C75tau', 
                                 'C76u', 'C76d', 'C76s', 'C76c', 'C76b', 'C76e', 'C76mu', 'C76tau',
                                 'C77u', 'C77d', 'C77s', 'C77c', 'C77b', 'C77e', 'C77mu', 'C77tau', 
                                 'C78u', 'C78d', 'C78s', 'C78c', 'C78b', 'C78e', 'C78mu', 'C78tau']
            self.wc_name_list_4f = ['C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                    'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                    'C71', 'C72', 'C73', 'C74',
                                    'C75u', 'C75d', 'C75s', 'C75c', 'C75e', 'C75mu', 'C75tau', 
                                    'C76u', 'C76d', 'C76s', 'C76c', 'C76e', 'C76mu', 'C76tau',
                                    'C77u', 'C77d', 'C77s', 'C77c', 'C77e', 'C77mu', 'C77tau', 
                                    'C78u', 'C78d', 'C78s', 'C78c', 'C78e', 'C78mu', 'C78tau']
            del_ind_list = [i for i in range(0,10)] + [i for i in range(18,26)] + [i for i in range(70,86)]

        if self.DM_type == "C":
            self.wc_name_list = ['C61u', 'C61d', 'C61s', 'C61c', 'C61b', 'C61e', 'C61mu', 'C61tau', 
                                 'C62u', 'C62d', 'C62s', 'C62c', 'C62b', 'C62e', 'C62mu', 'C62tau',
                                 'C65', 'C66',
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau']
            self.wc_name_list_4f = ['C61u', 'C61d', 'C61s', 'C61c', 'C61e', 'C61mu', 'C61tau', 
                                    'C62u', 'C62d', 'C62s', 'C62c', 'C62e', 'C62mu', 'C62tau',
                                    'C65', 'C66',
                                    'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau', 
                                    'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau']
            del_ind_list = [0,1] + [i for i in range(10,18)] + [i for i in range(26,34)] + [35] + [37] + [i for i in range(46,54)] + [i for i in range(62,86)]

        if self.DM_type == "R":
            self.wc_name_list = ['C65', 'C66',
                                 'C63u', 'C63d', 'C63s', 'C63c', 'C63b', 'C63e', 'C63mu', 'C63tau', 
                                 'C64u', 'C64d', 'C64s', 'C64c', 'C64b', 'C64e', 'C64mu', 'C64tau']
            self.wc_name_list_4f = ['C65', 'C66',
                                    'C64u', 'C64d', 'C64s', 'C64c', 'C64e', 'C64mu', 'C64tau',
                                    'C63u', 'C63d', 'C63s', 'C63c', 'C63e', 'C63mu', 'C63tau']
            del_ind_list = [i for i in range(0,34)] + [35] + [37] + [i for i in range(46,54)] + [i for i in range(62,86)]



        self.coeff_dict = {}
        # Issue a user warning if a key is not defined:
        for wc_name in coeff_dict.keys():
            if wc_name in self.wc_name_list:
                pass
            else:
                warnings.warn('The key ' + wc_name + ' is not a default key value. Typo?')
        # Create the dictionary:
        for wc_name in self.wc_name_list:
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.

        # Create the np.array of coefficients:
        self.coeff_list = np.array(dict_to_list(self.coeff_dict, self.wc_name_list))


        #---------------------------#
        # The anomalous dimensions: #
        #---------------------------#
        if self.DM_type == "D":
            self.gamma_QED = adm.ADM_QED(5)
            self.gamma_QCD = adm.ADM_QCD(5)
            self.gamma_QCD2 = adm.ADM_QCD2(5)
        if self.DM_type == "M":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(5), del_ind_list, 1), del_ind_list, 2)
        if self.DM_type == "C":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(5), del_ind_list, 1), del_ind_list, 2)
        if self.DM_type == "R":
            self.gamma_QED = np.delete(np.delete(adm.ADM_QED(5), del_ind_list, 0), del_ind_list, 1)
            self.gamma_QCD = np.delete(np.delete(adm.ADM_QCD(5), del_ind_list, 1), del_ind_list, 2)
            self.gamma_QCD2 = np.delete(np.delete(adm.ADM_QCD2(5), del_ind_list, 1), del_ind_list, 2)


    def run(self, mub=None, dict=None):
        """ Running of 5-flavor Wilson coefficients

        Calculate the running from MZ to mub [GeV; default 4.18 GeV] in the five-flavor theory. 

        For dict = True, returns a dictionary of Wilson coefficients for the five-flavor Lagrangian
        at scale mub (this is the default).

        For dict = False, returns a numpy array of Wilson coefficients for the five-flavor Lagrangian
        at scale mub.
        """
        ip = Num_input()
        if mub is None:
            mub=ip.mb_at_mb
        if dict is None:
            dict = True


        #-------------#
        # The running #
        #-------------#

        MZ = ip.Mz
        mb = ip.mb_at_mb
        alpha_at_mb = 1/ip.aMZinv

        as51 = rge.AlphaS(5,1)
        evolve1 = rge.RGE(self.gamma_QCD, 5)
        evolve2 = rge.RGE(self.gamma_QCD2, 5)

        # Strictly speaking, MZ and mb should be defined at the same scale (however, this is a higher-order difference)
        C_at_mb_QCD = np.dot(evolve2.U0_as2(as51.run(MZ),as51.run(mub)), np.dot(evolve1.U0(as51.run(MZ),as51.run(mub)), self.coeff_list))
        C_at_mb_QED = np.dot(self.coeff_list, self.gamma_QED) * np.log(mub/MZ) * alpha_at_mb/(4*np.pi)

        if dict:
            return list_to_dict(C_at_mb_QCD + C_at_mb_QED, self.wc_name_list)
        else:
            return C_at_mb_QCD + C_at_mb_QED


    def match(self, mu=None, dict=None):
        """ Match from five-flavor to four-flavor QCD

        Calculate the matching at mu [GeV; default 4.18 GeV].

        For dict = True, returns a dictionary of Wilson coefficients for the four-flavor Lagrangian
        at scale mu (this is the default).

        For dict = False, returns a numpy array of Wilson coefficients for the four-flavor Lagrangian
        at scale mu.
        """
        ip = Num_input()
        if mu is None:
            mu=ip.mb_at_mb
        if dict is None:
            dict=True

        # The new coefficients
        cdict4f = {}
        cdold = self.run(mu)

        if self.DM_type == "D" or self.DM_type == "M":
            for wcn in self.wc_name_list_4f:
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
        if dict:
            return cdict4f
        else:
            return dict_to_list(cdict4f, self.wc_name_list_4f)

    def _my_cNR(self, mchi, RGE=None, dict=None, NLO=None):
        """ Calculate the NR coefficients from four-flavor theory with meson contributions split off (mainly for internal use) """
        return WC_4f(self.match(), self.DM_type)._my_cNR(mchi, RGE, dict, NLO)

    def cNR(self, mchi, qvec, RGE=None, dict=None, NLO=None):
        """ Calculate the NR coefficients from four-flavor theory """
        return WC_4f(self.match(), self.DM_type).cNR(mchi, qvec, RGE, dict, NLO)

    def write_mma(self, mchi, qvector, RGE=None, NLO=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass mchi (in GeV) and the momentum transfer qvector (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is '.')

        <filename> is the filename (default 'cNR.m')
        """
        WC_4f(self.match(), self.DM_type).write_mma(mchi, qvector, RGE, NLO, path, filename)





#-------------------------------#
# The e/w Wilson coefficicients #
#-------------------------------#


class WC_EW(object):
    def __init__(self, coeff_dict, Lambda, Ychi, Jchi, DM_type=None, DM_mass_scale=None):
        """ Class for DM Wilson coefficients in the SM unbroken phase

        The first argument should be a dictionary for the initial conditions of the 8 
        dimension-five Wilson coefficients of the form
        {'C51' : value, 'C52' : value, ...}; 
        
        the 46 dimension-six Wilson coefficients of the form
        {'C611' : value, 'C621' : value, ...}; 

        and the dimension-seven Wilson coefficient (currently not yet implemented).
        An arbitrary number of them can be given; the default values are zero. 
        
        The possible keys are, for Jchi != 0:

         'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58',
         'C611', 'C621', 'C631', 'C641', 'C651', 'C661', 'C671', 'C681', 'C691', 'C6101', 'C6111', 'C6121', 'C6131', 'C6141',
         'C612', 'C622', 'C632', 'C642', 'C652', 'C662', 'C672', 'C682', 'C692', 'C6102', 'C6112', 'C6122', 'C6132', 'C6142',
         'C613', 'C623', 'C633', 'C643', 'C653', 'C663', 'C673', 'C683', 'C693', 'C6103', 'C6113', 'C6123', 'C6133', 'C6143',
         'C615', 'C616', 'C617', 'C618'
        
        The possible keys are, for Jchi = 0:

         'C51', 'C53', 'C55', 'C57',
         'C621', 'C631', 'C641', 'C661', 'C671', 'C681', 'C6101', 'C6111', 'C6131', 'C6141',
         'C622', 'C632', 'C642', 'C662', 'C672', 'C682', 'C6102', 'C6112', 'C6132', 'C6142',
         'C623', 'C633', 'C643', 'C663', 'C673', 'C683', 'C6103', 'C6113', 'C6133', 'C6143',
         'C616', 'C618'
        
        Lambda is the NP scale in GeV
        Jchi is the DM weak isospin
        Ychi is the DM hypercharge such that Q = I^3 + Y/2

        The second-to-last argument is the DM type; it is optional and can take the following values: 
            "D" (Dirac fermion; this is the default)

        The last argument is currently ignored.
        """
        if DM_type is None:
            DM_type = "D"
        self.DM_type = DM_type

        self.Lambda = Lambda
        self.Ychi = Ychi
        self.Jchi = Jchi

        if self.DM_type == "D":
            if self.Jchi == 0:
                self.wc_name_list_dim_5 = ['C51', 'C53', 'C55', 'C57']
                self.wc_name_list_dim_6 = ['C621', 'C631', 'C641', 'C661', 'C671', 'C681', 'C6101', 'C6111', 'C6131', 'C6141',\
                                           'C622', 'C632', 'C642', 'C662', 'C672', 'C682', 'C6102', 'C6112', 'C6132', 'C6142',\
                                           'C623', 'C633', 'C643', 'C663', 'C673', 'C683', 'C6103', 'C6113', 'C6133', 'C6143',\
                                           'C616', 'C618']
            else:
                self.wc_name_list_dim_5 = ['C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58']
                self.wc_name_list_dim_6 = ['C611', 'C621', 'C631', 'C641', 'C651', 'C661', 'C671', 'C681', 'C691', 'C6101', 'C6111', 'C6121', 'C6131', 'C6141',\
                                           'C612', 'C622', 'C632', 'C642', 'C652', 'C662', 'C672', 'C682', 'C692', 'C6102', 'C6112', 'C6122', 'C6132', 'C6142',\
                                           'C613', 'C623', 'C633', 'C643', 'C653', 'C663', 'C673', 'C683', 'C693', 'C6103', 'C6113', 'C6123', 'C6133', 'C6143',\
                                           'C615', 'C616', 'C617', 'C618']


        # Issue a user warning if a key is not defined or belongs to a redundant operator:
        for wc_name in coeff_dict.keys():
            if wc_name in self.wc_name_list_dim_5:
                pass
            elif wc_name in self.wc_name_list_dim_6:
                pass
            else:
                if self.Jchi == 0:
                    warnings.warn('The key ' + wc_name + ' is not a default key value. Typo; or belongs to an operator that is redundant for Jchi = 0?')
                else:
                    warnings.warn('The key ' + wc_name + ' is not a default key value. Typo?')

        self.coeff_dict = {}
        # Create the dictionary:
        for wc_name in (self.wc_name_list_dim_5 + self.wc_name_list_dim_6):
            if wc_name in coeff_dict.keys():
                self.coeff_dict[wc_name] = coeff_dict[wc_name]
            else:
                self.coeff_dict[wc_name] = 0.

        # Create the np.array of coefficients:
        self.coeff_list_dim_5 = np.array(dict_to_list(self.coeff_dict, self.wc_name_list_dim_5))
        self.coeff_list_dim_6 = np.array(dict_to_list(self.coeff_dict, self.wc_name_list_dim_6))

    #---------#
    # Running #
    #---------#

    def run(self, muz=None, resum=None, dict=None):
        """Calculate the e/w running from scale Lambda to scale muz [muz = MZ by default].

        resum = True yields full resummation (default)
        resum = False gives only the linear log 
        """
        ip = Num_input()

        if resum is None:
            resum=True
        if muz is None:
            muz = ip.Mz
        if dict is None:
            dict = True

        # Some abbeviations
        nsu2 = 2*self.Jchi + 1
        jj1 = self.Jchi*(self.Jchi+1)

        # Number of colors
        nc = 3

        # Input parameters
        ip = Num_input()

        alpha = 1/ip.aMZinv
        el = np.sqrt(4*np.pi*alpha)
        MW = ip.Mw
        MZ = ip.Mz
        Mh = ip.Mh
        cw = MW/MZ
        sw = np.sqrt(1-cw**2)
        g1 = el/cw
        g2 = el/sw
        yt = np.sqrt(2)*ip.mt_pole/246.


        # Add entries for unphysical operators
        C6_at_Lambda = np.concatenate((self.coeff_list_dim_6, np.array([0 for i in range(130)])))

        if resum:
            C5_at_muz = rge.CmuEW(self.coeff_list_dim_5, adm.ADM5(self.Ychi, self.Jchi), self.Lambda, muz, self.Ychi, self.Jchi, 1, 1, 1, 1)
            C6_at_muz = rge.CmuEW(C6_at_Lambda, adm.ADM6(self.Ychi, self.Jchi), self.Lambda, muz, self.Ychi, self.Jchi, 1, 1, 1, 1)

            if dict:
                return [C5_at_muz.run()[0][1], C6_at_muz.run()[0][1]]
            else:
                raise Exception("Currently, only a dictionary can be returned.")
        else:
            ADM5 = g1**2*adm.ADM5(self.Ychi, self.Jchi)[0] + g2**2*adm.ADM5(self.Ychi, self.Jchi)[1] + yt**2*adm.ADM5(self.Ychi, self.Jchi)[3]
            C5_at_muz = self.coeff_list_dim_5 + np.log(muz**2/self.Lambda**2)/(16*np.pi**2) * np.dot(self.coeff_list_dim_5, ADM5) 
            ADM6 = g1**2*adm.ADM6(self.Ychi, self.Jchi)[0] + g2**2*adm.ADM6(self.Ychi, self.Jchi)[1] + yt**2*adm.ADM6(self.Ychi, self.Jchi)[3]
            C6_at_muz = C6_at_Lambda + np.log(muz**2/self.Lambda**2)/(16*np.pi**2) * np.dot(C6_at_Lambda, ADM6)

            if dict:
                dict56 = list_to_dict(C5_at_muz, self.wc_name_list_dim_5)
                dict6 = list_to_dict(C6_at_muz, self.wc_name_list_dim_6)
                dict56.update(dict6)
                return dict56
            else:
                raise Exception("Currently, only a dictionary can be returned.")

    #----------#
    # Matching #
    #----------#

    def match(self, mchi, mchi_threshold=None, RUN_EW=None, dict=None, DIM4=None):
        """Calculate the matching from the relativistic theory to the five-flavor theory at scale MZ

        mchi is the DM mass, as it appears in the UV Lagrangian. It is not the physical DM mass after EWSB. 

        mchi_threshold is the DM mass below which DM is treated as "light" [default is 40 GeV]

        RUN_EW can have three values: 

         - RUN_EW = 'FULL'  does the full leading-logarithm resummation (this is the default)
         - RUN_EW = 'LL'    keeps only the linear e/w logarithm
         - RUN_EW = 'OFF'   no electroweak running

        DIM4 multiplies the dimension-four matching contributions. To be considered as an "analysis tool", might be removed

        Returns a dictionary of Wilson coefficients for the five-flavor Lagrangian, 
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
            RUN_EW = 'FULL'
        self.RUN_EW = RUN_EW

        if mchi_threshold is None:
            mchi_threshold = 40 # GeV
        self.mchi_threshold = mchi_threshold

        if dict is None:
            dict = True

        if DIM4 is None:
            DIM4 = 1
        else:
            DIM4 = 0

        # Some input parameters:

        ip = Num_input()

        vev = 1/np.sqrt(np.sqrt(2)*ip.GF)
        alpha = 1/ip.aMZinv
        MW = ip.Mw
        MZ = ip.Mz
        Mh = ip.Mh
        cw = MW/MZ
        sw = np.sqrt(1-cw**2)

        # Calculate the physical DM mass in terms of mchi and the Wilson coefficients, 
        # and the corresponding shift in the dimension-five Wilson coefficients.

        if mchi > mchi_threshold:
            if self.Jchi == 0:
                self.mchi_phys = mchi - vev**2/2/self.Lambda * self.coeff_dict['C53']
                wc5_dict_shifted = {}

                wc5_dict_shifted['C51'] = self.coeff_dict['C51'] + vev**2/2/self.Lambda/mchi * self.coeff_dict['C57'] * self.coeff_dict['C55']
                wc5_dict_shifted['C53'] = self.coeff_dict['C53'] + vev**2/2/self.Lambda/mchi * self.coeff_dict['C57'] * self.coeff_dict['C57']
                wc5_dict_shifted['C55'] = self.coeff_dict['C55'] - vev**2/2/self.Lambda/mchi * self.coeff_dict['C57'] * self.coeff_dict['C51']
                wc5_dict_shifted['C57'] = self.coeff_dict['C57'] - vev**2/2/self.Lambda/mchi * self.coeff_dict['C57'] * self.coeff_dict['C53']

            else:
                self.mchi_phys = mchi - vev**2/2/self.Lambda * (self.coeff_dict['C53'] + self.Ychi/4 * self.coeff_dict['C54'])
                wc5_dict_shifted = {}

                wc5_dict_shifted['C51'] = self.coeff_dict['C51']\
                                          + vev**2/2/self.Lambda/mchi * (self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58']) * self.coeff_dict['C55']
                wc5_dict_shifted['C52'] = self.coeff_dict['C52']\
                                          + vev**2/2/self.Lambda/mchi * (self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58']) * self.coeff_dict['C56']
                wc5_dict_shifted['C53'] = self.coeff_dict['C53']\
                                          + vev**2/2/self.Lambda/mchi * (self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58']) * self.coeff_dict['C57']
                wc5_dict_shifted['C54'] = self.coeff_dict['C54']\
                                          + vev**2/2/self.Lambda/mchi * (self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58']) * self.coeff_dict['C58']
                wc5_dict_shifted['C55'] = self.coeff_dict['C55']\
                                          - vev**2/2/self.Lambda/mchi * (self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58']) * self.coeff_dict['C51']
                wc5_dict_shifted['C56'] = self.coeff_dict['C56']\
                                          - vev**2/2/self.Lambda/mchi * (self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58']) * self.coeff_dict['C52']
                wc5_dict_shifted['C57'] = self.coeff_dict['C57']\
                                          - vev**2/2/self.Lambda/mchi * (self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58']) * self.coeff_dict['C53']
                wc5_dict_shifted['C58'] = self.coeff_dict['C58']\
                                          - vev**2/2/self.Lambda/mchi * (self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58']) * self.coeff_dict['C54']

        else:
            if self.Jchi == 0:
                cosphi = np.sqrt((self.coeff_dict['C53'] - 2*mchi*self.Lambda/vev**2)**2/\
                                 ((self.coeff_dict['C53'] - 2*mchi*self.Lambda/vev**2)**2 + self.coeff_dict['C57']**2))
                sinphi = np.sqrt((self.coeff_dict['C57'])**2/\
                                 ((self.coeff_dict['C53'] - 2*mchi*self.Lambda/vev**2)**2 + self.coeff_dict['C57']**2))
                pre_mchi_phys = mchi*cosphi + vev**2/2/self.Lambda * (self.coeff_dict['C57'] * sinphi - self.coeff_dict['C53'] * cosphi)
                if pre_mchi_phys > 0:
                    self.mchi_phys = pre_mchi_phys

                    wc5_dict_shifted = {}

                    wc5_dict_shifted['C51'] = cosphi * self.coeff_dict['C51'] + sinphi * self.coeff_dict['C55'] 
                    wc5_dict_shifted['C53'] = cosphi * self.coeff_dict['C53'] + sinphi * self.coeff_dict['C57'] 
                    wc5_dict_shifted['C55'] = cosphi * self.coeff_dict['C55'] - sinphi * self.coeff_dict['C51'] 
                    wc5_dict_shifted['C57'] = cosphi * self.coeff_dict['C57'] - sinphi * self.coeff_dict['C53'] 
                else:
                    self.mchi_phys = - pre_mchi_phys

                    wc5_dict_shifted = {}

                    wc5_dict_shifted['C51'] = cosphi * self.coeff_dict['C51'] - sinphi * self.coeff_dict['C55'] 
                    wc5_dict_shifted['C53'] = cosphi * self.coeff_dict['C53'] - sinphi * self.coeff_dict['C57'] 
                    wc5_dict_shifted['C55'] = cosphi * self.coeff_dict['C55'] + sinphi * self.coeff_dict['C51'] 
                    wc5_dict_shifted['C57'] = cosphi * self.coeff_dict['C57'] + sinphi * self.coeff_dict['C53'] 
            else:
                cosphi = np.sqrt((self.coeff_dict['C53'] + self.Ychi/4 * self.coeff_dict['C54'] - 2*mchi*self.Lambda/vev**2)**2/\
                                ((self.coeff_dict['C53'] + self.Ychi/4 * self.coeff_dict['C54'] - 2*mchi*self.Lambda/vev**2)**2\
                                +(self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58'])**2))
                sinphi = np.sqrt((self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58'])**2/\
                                ((self.coeff_dict['C53'] + self.Ychi/4 * self.coeff_dict['C54'] - 2*mchi*self.Lambda/vev**2)**2\
                                +(self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58'])**2))
                pre_mchi_phys = mchi*cosphi + vev**2/2/self.Lambda * ((self.coeff_dict['C57'] + self.Ychi/4 * self.coeff_dict['C58'])*sinphi\
                                                              - (self.coeff_dict['C53'] + self.Ychi/4 * self.coeff_dict['C54'])*cosphi)
                if pre_mchi_phys > 0:
                    self.mchi_phys = pre_mchi_phys

                    wc5_dict_shifted = {}

                    wc5_dict_shifted['C51'] = cosphi * self.coeff_dict['C51'] + sinphi * self.coeff_dict['C55'] 
                    wc5_dict_shifted['C52'] = cosphi * self.coeff_dict['C52'] + sinphi * self.coeff_dict['C56'] 
                    wc5_dict_shifted['C53'] = cosphi * self.coeff_dict['C53'] + sinphi * self.coeff_dict['C57'] 
                    wc5_dict_shifted['C54'] = cosphi * self.coeff_dict['C54'] + sinphi * self.coeff_dict['C58'] 
                    wc5_dict_shifted['C55'] = cosphi * self.coeff_dict['C55'] - sinphi * self.coeff_dict['C51'] 
                    wc5_dict_shifted['C56'] = cosphi * self.coeff_dict['C56'] - sinphi * self.coeff_dict['C52'] 
                    wc5_dict_shifted['C57'] = cosphi * self.coeff_dict['C57'] - sinphi * self.coeff_dict['C53'] 
                    wc5_dict_shifted['C58'] = cosphi * self.coeff_dict['C58'] - sinphi * self.coeff_dict['C54'] 
                else:
                    self.mchi_phys = - pre_mchi_phys

                    wc5_dict_shifted = {}

                    wc5_dict_shifted['C51'] = cosphi * self.coeff_dict['C51'] - sinphi * self.coeff_dict['C55'] 
                    wc5_dict_shifted['C52'] = cosphi * self.coeff_dict['C52'] - sinphi * self.coeff_dict['C56'] 
                    wc5_dict_shifted['C53'] = cosphi * self.coeff_dict['C53'] - sinphi * self.coeff_dict['C57'] 
                    wc5_dict_shifted['C54'] = cosphi * self.coeff_dict['C54'] - sinphi * self.coeff_dict['C58'] 
                    wc5_dict_shifted['C55'] = cosphi * self.coeff_dict['C55'] + sinphi * self.coeff_dict['C51'] 
                    wc5_dict_shifted['C56'] = cosphi * self.coeff_dict['C56'] + sinphi * self.coeff_dict['C52'] 
                    wc5_dict_shifted['C57'] = cosphi * self.coeff_dict['C57'] + sinphi * self.coeff_dict['C53'] 
                    wc5_dict_shifted['C58'] = cosphi * self.coeff_dict['C58'] + sinphi * self.coeff_dict['C54'] 

        # The redefinitions of the dim.-5 Wilson coefficients resulting from the mass shift:

        coeff_dict_shifted = self.coeff_dict
        coeff_dict_shifted.update(wc5_dict_shifted)

        # The Higgs penguin function. 
        # The result is valid for all input values and gives (in principle) a real output.
        # Note that currently there is no distinction between e/w and light DM, as the two-loop function for light DM is unknown.
        def higgs_penguin_fermion(Ychi,Jchi):
            return Higgspenguin(Ychi, Jchi).oneloop_ew(self.mchi_phys)
        def higgs_penguin_gluon(Ychi,Jchi):
            return Higgspenguin(Ychi, Jchi).twoloop_ew_fa(self.mchi_phys) + 0*Higgspenguin(Ychi, Jchi).hisano_fbc(self.mchi_phys)


        #-----------------------#
        # The new coefficients: #
        #-----------------------#
        
        # Note that in the RG-DM paper we introduced the hat notation. We will not do that here, but instead put in the appropriate powers of Lambda explicitly. 

        coeff_dict_5f = {}

        if self.Jchi == 0:
            coeff_dict_5f['C51'] = 1/(4*np.pi*alpha)*(cw**2 * coeff_dict_shifted['C51'])/self.Lambda
            coeff_dict_5f['C52'] = 1/(4*np.pi*alpha)*(cw**2 * coeff_dict_shifted['C55'])/self.Lambda

            coeff_dict_5f['C61u'] = (coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C631']/2\
                  - (3-8*sw**2)/6 * coeff_dict_shifted['C616']\
                  + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-8*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61d'] = (coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C641']/2\
                  + (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61s'] = (coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C642']/2\
                  + (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61c'] = (coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C632']/2\
                  - (3-8*sw**2)/6 * coeff_dict_shifted['C616']\
                  + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-8*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61b'] = (coeff_dict_shifted['C623']/2 + coeff_dict_shifted['C643']/2\
                  + (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61e'] = (coeff_dict_shifted['C6101']/2 + coeff_dict_shifted['C6111']/2\
                  + (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61mu'] = (coeff_dict_shifted['C6102']/2 + coeff_dict_shifted['C6112']/2\
                  + (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61tau'] = (coeff_dict_shifted['C6103']/2 + coeff_dict_shifted['C6113']/2\
                  + (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4)/self.Lambda**2

            coeff_dict_5f['C62u'] = (coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C671']/2\
                   - (3-8*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62d'] = (coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C681']/2\
                   + (3-4*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62s'] = (coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C682']/2\
                   + (3-4*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62c'] = (coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C672']/2\
                   - (3-8*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62b'] = (coeff_dict_shifted['C663']/2 + coeff_dict_shifted['C683']/2\
                   + (3-4*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62e'] = (coeff_dict_shifted['C6131']/2 + coeff_dict_shifted['C6141']/2\
                   + (1-4*sw**2)/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62mu'] = (coeff_dict_shifted['C6132']/2 + coeff_dict_shifted['C6142']/2\
                   + (1-4*sw**2)/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62tau'] = (coeff_dict_shifted['C6133']/2 + coeff_dict_shifted['C6143']/2\
                   + (1-4*sw**2)/2 * coeff_dict_shifted['C618'])/self.Lambda**2

            coeff_dict_5f['C63u'] = (- coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C631']/2\
                   + 1/2 * coeff_dict_shifted['C616']\
                   - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63d'] = (- coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C641']/2\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63s'] = (- coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C642']/2\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63c'] = (- coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C632']/2\
                   + 1/2 * coeff_dict_shifted['C616']\
                   - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63b'] = (- coeff_dict_shifted['C623']/2 + coeff_dict_shifted['C643']/2\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63e'] = (- coeff_dict_shifted['C6101']/2 + coeff_dict_shifted['C6111']/2\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63mu'] = (- coeff_dict_shifted['C6102']/2 + coeff_dict_shifted['C6112']/2\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63tau'] = (- coeff_dict_shifted['C6103']/2 + coeff_dict_shifted['C6113']/2\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2

            coeff_dict_5f['C64u'] = (- coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C671']/2\
                    + 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64d'] = (- coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C681']/2\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64s'] = (- coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C682']/2\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64c'] = (- coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C672']/2\
                    + 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64b'] = (- coeff_dict_shifted['C663']/2 + coeff_dict_shifted['C683']/2\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64e'] = (- coeff_dict_shifted['C6131']/2 + coeff_dict_shifted['C6141']/2\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64mu'] = (- coeff_dict_shifted['C6132']/2 + coeff_dict_shifted['C6142']/2\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64tau'] = (- coeff_dict_shifted['C6133']/2 + coeff_dict_shifted['C6143']/2\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2

            coeff_dict_5f['C71'] = (self.Lambda**2/Mh**2 * (coeff_dict_shifted['C53']))/self.Lambda**3\
                                   + higgs_penguin_gluon(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C72'] = (self.Lambda**2/Mh**2 * (coeff_dict_shifted['C57']))/self.Lambda**3

            coeff_dict_5f['C75u'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75d'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75s'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75c'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75b'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75e'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75mu'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C61tau'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4

            coeff_dict_5f['C76u'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'])/self.Lambda**2
            coeff_dict_5f['C76d'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'])/self.Lambda**2
            coeff_dict_5f['C76s'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'])/self.Lambda**2
            coeff_dict_5f['C76c'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'])/self.Lambda**2
            coeff_dict_5f['C76b'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'])/self.Lambda**2
            coeff_dict_5f['C76e'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'])/self.Lambda**2
            coeff_dict_5f['C76mu'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'])/self.Lambda**2
            coeff_dict_5f['C76tau'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'])/self.Lambda**2
        else:
            coeff_dict_5f['C51'] = 1/(4*np.pi*alpha)*(cw**2 * coeff_dict_shifted['C51'] + sw**2 * self.Ychi/2 * coeff_dict_shifted['C52'])/self.Lambda
            coeff_dict_5f['C52'] = 1/(4*np.pi*alpha)*(cw**2 * coeff_dict_shifted['C55'] + sw**2 * self.Ychi/2 * coeff_dict_shifted['C56'])/self.Lambda

            coeff_dict_5f['C61u'] = (- self.Ychi/8 * coeff_dict_shifted['C611'] + coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C631']/2\
                  - self.Ychi * (3-8*sw**2)/24 * coeff_dict_shifted['C615']\
                  - (3-8*sw**2)/6 * coeff_dict_shifted['C616']\
                  + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-8*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61d'] = (self.Ychi/8*coeff_dict_shifted['C611'] + coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C641']/2\
                  + self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C615']\
                  + (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61s'] = (self.Ychi/8*coeff_dict_shifted['C612'] + coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C642']/2\
                  + self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C615']\
                  + (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61c'] = (- self.Ychi/8*coeff_dict_shifted['C612'] + coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C632']/2\
                  - self.Ychi * (3-8*sw**2)/24 * coeff_dict_shifted['C615']\
                  - (3-8*sw**2)/6 * coeff_dict_shifted['C616']\
                  + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-8*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61b'] = (self.Ychi/8*coeff_dict_shifted['C613'] + coeff_dict_shifted['C623']/2 + coeff_dict_shifted['C643']/2\
                  + self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C615']\
                  + (3-4*sw**2)/6 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(6*sw**2*cw**2) * (3-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61e'] = (self.Ychi/8*coeff_dict_shifted['C691'] + coeff_dict_shifted['C6101']/2 + coeff_dict_shifted['C6111']/2\
                  + self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C615']\
                  + (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61mu'] = (self.Ychi/8*coeff_dict_shifted['C692'] + coeff_dict_shifted['C6102']/2 + coeff_dict_shifted['C6112']/2\
                  + self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C615']\
                  + (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C61tau'] = (self.Ychi/8*coeff_dict_shifted['C693'] + coeff_dict_shifted['C6103']/2 + coeff_dict_shifted['C6113']/2\
                  + self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C615']\
                  + (1-4*sw**2)/2 * coeff_dict_shifted['C616']\
                  - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * (1-4*sw**2) * DIM4)/self.Lambda**2

            coeff_dict_5f['C62u'] = (- self.Ychi/8*coeff_dict_shifted['C651'] + coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C671']/2\
                   - self.Ychi * (3-8*sw**2)/24 * coeff_dict_shifted['C617']\
                   - (3-8*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62d'] = (self.Ychi/8*coeff_dict_shifted['C651'] + coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C681']/2\
                   + self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C617']\
                   + (3-4*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62s'] = (self.Ychi/8*coeff_dict_shifted['C652'] + coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C682']/2\
                   + self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C617']\
                   + (3-4*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62c'] = (- self.Ychi/8*coeff_dict_shifted['C652'] + coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C672']/2\
                   - self.Ychi * (3-8*sw**2)/24 * coeff_dict_shifted['C617']\
                   - (3-8*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62b'] = (self.Ychi/8*coeff_dict_shifted['C653'] + coeff_dict_shifted['C663']/2 + coeff_dict_shifted['C683']/2\
                   + self.Ychi * (3-4*sw**2)/24 * coeff_dict_shifted['C617']\
                   + (3-4*sw**2)/6 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62e'] = (self.Ychi/8*coeff_dict_shifted['C6121'] + coeff_dict_shifted['C6131']/2 + coeff_dict_shifted['C6141']/2\
                   + self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C617']\
                   + (1-4*sw**2)/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62mu'] = (self.Ychi/8*coeff_dict_shifted['C6122'] + coeff_dict_shifted['C6132']/2 + coeff_dict_shifted['C6142']/2\
                   + self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C617']\
                   + (1-4*sw**2)/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C62tau'] = (self.Ychi/8*coeff_dict_shifted['C6123'] + coeff_dict_shifted['C6133']/2 + coeff_dict_shifted['C6143']/2\
                   + self.Ychi * (1-4*sw**2)/8 * coeff_dict_shifted['C617']\
                   + (1-4*sw**2)/2 * coeff_dict_shifted['C618'])/self.Lambda**2

            coeff_dict_5f['C63u'] = (self.Ychi/8*coeff_dict_shifted['C611'] - coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C631']/2\
                   + self.Ychi/8 * coeff_dict_shifted['C615']\
                   + 1/2 * coeff_dict_shifted['C616']\
                   - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63d'] = (- self.Ychi/8*coeff_dict_shifted['C611'] - coeff_dict_shifted['C621']/2 + coeff_dict_shifted['C641']/2\
                   - self.Ychi/8 * coeff_dict_shifted['C615']\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63s'] = (- self.Ychi/8*coeff_dict_shifted['C612'] - coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C642']/2\
                   - self.Ychi/8 * coeff_dict_shifted['C615']\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63c'] = (self.Ychi/8*coeff_dict_shifted['C612'] - coeff_dict_shifted['C622']/2 + coeff_dict_shifted['C632']/2\
                   + self.Ychi/8 * coeff_dict_shifted['C615']\
                   + 1/2 * coeff_dict_shifted['C616']\
                   - self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63b'] = (- self.Ychi/8*coeff_dict_shifted['C613'] - coeff_dict_shifted['C623']/2 + coeff_dict_shifted['C643']/2\
                   - self.Ychi/8 * coeff_dict_shifted['C615']\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63e'] = (- self.Ychi/8*coeff_dict_shifted['C691'] - coeff_dict_shifted['C6101']/2 + coeff_dict_shifted['C6111']/2\
                   - self.Ychi/8 * coeff_dict_shifted['C615']\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63mu'] = (- self.Ychi/8*coeff_dict_shifted['C692'] - coeff_dict_shifted['C6102']/2 + coeff_dict_shifted['C6112']/2\
                   - self.Ychi/8 * coeff_dict_shifted['C615']\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2
            coeff_dict_5f['C63tau'] = (- self.Ychi/8*coeff_dict_shifted['C693'] - coeff_dict_shifted['C6103']/2 + coeff_dict_shifted['C6113']/2\
                   - self.Ychi/8 * coeff_dict_shifted['C615']\
                   - 1/2 * coeff_dict_shifted['C616']\
                   + self.Lambda**2/MZ**2 * (np.pi*alpha*self.Ychi)/(2*sw**2*cw**2) * DIM4)/self.Lambda**2

            coeff_dict_5f['C64u'] = (self.Ychi/8*coeff_dict_shifted['C651'] - coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C671']/2\
                    + self.Ychi/8 * coeff_dict_shifted['C617']\
                    + 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64d'] = (- self.Ychi/8*coeff_dict_shifted['C651'] - coeff_dict_shifted['C661']/2 + coeff_dict_shifted['C681']/2\
                    - self.Ychi/8 * coeff_dict_shifted['C617']\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64s'] = (- self.Ychi/8*coeff_dict_shifted['C652'] - coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C682']/2\
                    - self.Ychi/8 * coeff_dict_shifted['C617']\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64c'] = (self.Ychi/8*coeff_dict_shifted['C652'] - coeff_dict_shifted['C662']/2 + coeff_dict_shifted['C672']/2\
                    + self.Ychi/8 * coeff_dict_shifted['C617']\
                    + 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64b'] = (- self.Ychi/8*coeff_dict_shifted['C653'] - coeff_dict_shifted['C663']/2 + coeff_dict_shifted['C683']/2\
                    - self.Ychi/8 * coeff_dict_shifted['C617']\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64e'] = (- self.Ychi/8*coeff_dict_shifted['C6121'] - coeff_dict_shifted['C6131']/2 + coeff_dict_shifted['C6141']/2\
                    - self.Ychi/8 * coeff_dict_shifted['C617']\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64mu'] = (- self.Ychi/8*coeff_dict_shifted['C6122'] - coeff_dict_shifted['C6132']/2 + coeff_dict_shifted['C6142']/2\
                    - self.Ychi/8 * coeff_dict_shifted['C617']\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2
            coeff_dict_5f['C64tau'] = (- self.Ychi/8*coeff_dict_shifted['C6123'] - coeff_dict_shifted['C6133']/2 + coeff_dict_shifted['C6143']/2\
                    - self.Ychi/8 * coeff_dict_shifted['C617']\
                    - 1/2 * coeff_dict_shifted['C618'])/self.Lambda**2

            coeff_dict_5f['C71'] = (self.Lambda**2/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4 * coeff_dict_shifted['C54']))/self.Lambda**3\
                                   + higgs_penguin_gluon(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C72'] = (self.Lambda**2/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4 * coeff_dict_shifted['C58']))/self.Lambda**3

            coeff_dict_5f['C75u'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4 * coeff_dict_shifted['C54'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75d'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4 * coeff_dict_shifted['C54'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75s'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4 * coeff_dict_shifted['C54'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75c'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4 * coeff_dict_shifted['C54'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75b'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4 * coeff_dict_shifted['C54'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75e'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4 * coeff_dict_shifted['C54'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C75mu'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4 * coeff_dict_shifted['C54'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4
            coeff_dict_5f['C61tau'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C53'] + self.Ychi/4 * coeff_dict_shifted['C54'])/self.Lambda**2\
                                    + higgs_penguin_fermion(self.Ychi,self.Jchi) * DIM4

            coeff_dict_5f['C76u'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4 * coeff_dict_shifted['C58'])/self.Lambda**2
            coeff_dict_5f['C76d'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4 * coeff_dict_shifted['C58'])/self.Lambda**2
            coeff_dict_5f['C76s'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4 * coeff_dict_shifted['C58'])/self.Lambda**2
            coeff_dict_5f['C76c'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4 * coeff_dict_shifted['C58'])/self.Lambda**2
            coeff_dict_5f['C76b'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4 * coeff_dict_shifted['C58'])/self.Lambda**2
            coeff_dict_5f['C76e'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4 * coeff_dict_shifted['C58'])/self.Lambda**2
            coeff_dict_5f['C76mu'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4 * coeff_dict_shifted['C58'])/self.Lambda**2
            coeff_dict_5f['C76tau'] = self.Lambda/Mh**2 * (coeff_dict_shifted['C57'] + self.Ychi/4 * coeff_dict_shifted['C58'])/self.Lambda**2

        coeff_dict_5f['C73'] = 0
        coeff_dict_5f['C74'] = 0

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

        return coeff_dict_5f


    def _my_cNR(self, mchi, RGE=None, dict=None, NLO=None, mchi_threshold=None, RUN_EW=None, DIM4=None):
        """ Calculate the NR coefficients from four-flavor theory with meson contributions split off (mainly for internal use) """
        return WC_5f(self.match(mchi, mchi_threshold, RUN_EW, True, DIM4), self.DM_type)._my_cNR(self.mchi_phys, RGE, dict, NLO)

    def cNR(self, mchi, qvec, RGE=None, dict=None, NLO=None, mchi_threshold=None, RUN_EW=None, DIM4=None):
        """ Calculate the NR coefficients from four-flavor theory """
        return WC_5f(self.match(mchi, mchi_threshold, RUN_EW, True, DIM4), self.DM_type).cNR(mchi, qvec, RGE, dict, NLO)

    def write_mma(self, mchi, qvector, RGE=None, NLO=None, mchi_threshold=None, RUN_EW=None, DIM4=None, path=None, filename=None):
        """ Write a text file with the NR coefficients that can be read into DMFormFactor 

        The order is {cNR1p, cNR2p, ... , cNR1n, cNR1n, ... }

        Mandatory arguments are the DM mass mchi (in GeV) and the momentum transfer qvector (in GeV) 

        <path> should be a string with the path (including the trailing "/") where the file should be saved
        (default is '.')

        <filename> is the filename (default 'cNR.m')
        """
        WC_5f(self.match(mchi, mchi_threshold, RUN_EW, True, DIM4), self.DM_type).write_mma(mchi, qvector, RGE, NLO, path, filename)




